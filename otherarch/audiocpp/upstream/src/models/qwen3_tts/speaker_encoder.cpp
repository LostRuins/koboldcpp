#include "engine/models/qwen3_tts/speaker_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::qwen3_tts {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

constexpr int64_t kSampleRate = 24000;
constexpr int64_t kFeatureDim = 128;
constexpr int64_t kNfft = 1024;
constexpr int64_t kHopLength = 256;
constexpr int64_t kWinLength = 1024;
constexpr float kStatsEps = 1.0e-12F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ConvWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t dilation = 1;
    core::TensorValue weight;
    core::TensorValue bias;
};

struct SERes2NetWeights {
    ConvWeights tdnn1;
    std::vector<ConvWeights> res2net;
    ConvWeights tdnn2;
    ConvWeights se_conv1;
    ConvWeights se_conv2;
    int64_t dilation = 1;
};

ConvWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t dilation = 1) {
    const std::string full = "speaker_encoder." + prefix;
    ConvWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.dilation = dilation;
    conv.weight = store.load_tensor(source, full + ".weight", storage_type, {out_channels, in_channels, kernel});
    conv.bias = store.load_tensor(source, full + ".bias", assets::TensorStorageType::F32, {out_channels});
    return conv;
}

}  // namespace

audio::AudioTensor compute_qwen3_speaker_mel(const runtime::AudioBuffer & audio, int threads) {
    engine::audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = kSampleRate;
    config.n_fft = kNfft;
    config.hop_length = kHopLength;
    config.win_length = kWinLength;
    config.n_mels = kFeatureDim;
    config.mel_fmax = 12000.0f;
    config.waveform_padding = engine::audio::MelWaveformPadding::ReflectOrRepeatSingleton;
    config.filterbank_projection = engine::audio::MelFilterbankProjection::DenseLongDouble;
    config.resample_mode = engine::audio::MelResampleMode::Linear;
    const auto frontend = engine::audio::get_cached_mel_spectrogram_frontend(config);
    auto features = frontend->extract_audio(
        audio.samples, audio.sample_rate, audio.channels,
        static_cast<size_t>(std::max(1, threads)));
    return {std::move(features.values), {1, features.mel_bins, features.frames}};
}

namespace {

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const ConvWeights & conv,
    core::ConstantTensorCache & constants) {
    if (x.shape.dims[1] != conv.in_channels) {
        throw std::runtime_error("Qwen3 speaker conv channel mismatch");
    }
    const int64_t padding = conv.dilation * (conv.kernel - 1) / 2;
    if (padding > 0) {
        x = modules::ReflectPad1dModule({padding, padding}).build(ctx, x);
    }
    return modules::Conv1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        1,
        0,
        static_cast<int>(conv.dilation),
        true,
    }).build(ctx, x, binding::conv1d_data(constants, conv.weight, conv.bias));
}

core::TensorValue tdnn(
    core::ModuleBuildContext & ctx,
    core::TensorValue x,
    const ConvWeights & conv,
    core::ConstantTensorCache & constants) {
    x = conv1d(ctx, x, conv, constants);
    return modules::ReluModule{}.build(ctx, x);
}

core::TensorValue se_res2net(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SERes2NetWeights & block,
    core::ConstantTensorCache & constants) {
    auto y = tdnn(ctx, input, block.tdnn1, constants);
    core::TensorValue merged;
    core::TensorValue previous;
    constexpr int64_t scale = 8;
    constexpr int64_t width = 64;
    for (int64_t i = 0; i < scale; ++i) {
        auto chunk = modules::SliceModule({1, i * width, width}).build(ctx, y);
        core::TensorValue out;
        if (i == 0) {
            out = chunk;
        } else if (i == 1) {
            out = tdnn(ctx, chunk, block.res2net[0], constants);
        } else {
            out = tdnn(ctx, modules::ResidualAddModule{}.build(ctx, chunk, previous), block.res2net[static_cast<size_t>(i - 1)], constants);
        }
        previous = out;
        merged = merged.valid() ? modules::ConcatModule({1}).build(ctx, merged, out) : out;
    }
    y = tdnn(ctx, merged, block.tdnn2, constants);
    y = modules::SqueezeExcite1dModule({512, 128, true}).build(
        ctx,
        y,
        {binding::conv1d_data(constants, block.se_conv1.weight, block.se_conv1.bias),
         binding::conv1d_data(constants, block.se_conv2.weight, block.se_conv2.bias)});
    return modules::ResidualAddModule{}.build(ctx, y, input);
}

core::TensorValue attentive_statistics_pool(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const ConvWeights & tdnn_conv,
    const ConvWeights & attention_conv,
    core::ConstantTensorCache & constants) {
    auto mean = modules::ReduceMeanModule({2}).build(ctx, x);
    auto mean_rep = modules::RepeatModule({x.shape}).build(ctx, mean);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean_rep.tensor), x.shape, GGML_TYPE_F32);
    auto centered_sq = modules::MulModule{}.build(ctx, centered, centered);
    auto variance = modules::ReduceMeanModule({2}).build(ctx, centered_sq);
    auto eps = constants.make_f32(core::TensorShape::from_dims({1, 1, 1}), std::vector<float>{kStatsEps});
    auto eps_rep = modules::RepeatModule({variance.shape}).build(ctx, eps);
    auto std = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, variance, eps_rep));
    auto std_rep = modules::RepeatModule({x.shape}).build(ctx, std);

    auto attention = modules::ConcatModule({1}).build(
        ctx,
        modules::ConcatModule({1}).build(ctx, x, mean_rep),
        std_rep);
    attention = tdnn(ctx, attention, tdnn_conv, constants);
    attention = modules::TanhModule{}.build(ctx, attention);
    attention = conv1d(ctx, attention, attention_conv, constants);
    auto weights = modules::SoftmaxModule{}.build(ctx, attention);

    auto weighted = modules::MulModule{}.build(ctx, x, weights);
    auto mean_att = modules::ReduceSumModule({2}).build(ctx, weighted);
    auto mean_att_rep = modules::RepeatModule({x.shape}).build(ctx, mean_att);
    auto diff = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean_att_rep.tensor), x.shape, GGML_TYPE_F32);
    auto diff_sq = modules::MulModule{}.build(ctx, diff, diff);
    auto weighted_var = modules::MulModule{}.build(ctx, diff_sq, weights);
    auto var_att = modules::ReduceSumModule({2}).build(ctx, weighted_var);
    eps_rep = modules::RepeatModule({var_att.shape}).build(ctx, eps);
    auto std_att = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, var_att, eps_rep));
    return modules::ConcatModule({1}).build(ctx, mean_att, std_att);
}

}  // namespace

struct Qwen3TTSEcapaTdnnEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    ConvWeights block0;
    std::vector<SERes2NetWeights> blocks;
    ConvWeights mfa;
    ConvWeights asp_tdnn;
    ConvWeights asp_conv;
    ConvWeights fc;
    int64_t embedding_dim = 0;
};

namespace {

std::shared_ptr<const Qwen3TTSEcapaTdnnEncoderWeights> load_weights(
    const Qwen3TTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType conv_weight_storage_type) {
    const auto & source = *assets.model_weights;
    auto weights = std::make_shared<Qwen3TTSEcapaTdnnEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "qwen3_tts.speaker_encoder.weights",
        32ull * 1024ull * 1024ull);
    weights->block0 = load_conv(*weights->store, source, "blocks.0.conv", conv_weight_storage_type, 512, 128, 5, 1);
    for (int64_t block = 1; block <= 3; ++block) {
        SERes2NetWeights layer;
        const int64_t dilation = block + 1;
        const std::string prefix = "blocks." + std::to_string(block);
        layer.tdnn1 = load_conv(*weights->store, source, prefix + ".tdnn1.conv", conv_weight_storage_type, 512, 512, 1, 1);
        for (int64_t i = 0; i < 7; ++i) {
            layer.res2net.push_back(load_conv(
                *weights->store,
                source,
                prefix + ".res2net_block.blocks." + std::to_string(i) + ".conv",
                conv_weight_storage_type,
                64,
                64,
                3,
                dilation));
        }
        layer.tdnn2 = load_conv(*weights->store, source, prefix + ".tdnn2.conv", conv_weight_storage_type, 512, 512, 1, 1);
        layer.se_conv1 = load_conv(*weights->store, source, prefix + ".se_block.conv1", conv_weight_storage_type, 128, 512, 1, 1);
        layer.se_conv2 = load_conv(*weights->store, source, prefix + ".se_block.conv2", conv_weight_storage_type, 512, 128, 1, 1);
        layer.dilation = dilation;
        weights->blocks.push_back(std::move(layer));
    }
    weights->mfa = load_conv(*weights->store, source, "mfa.conv", conv_weight_storage_type, 1536, 1536, 1, 1);
    weights->asp_tdnn = load_conv(*weights->store, source, "asp.tdnn.conv", conv_weight_storage_type, 128, 4608, 1, 1);
    weights->asp_conv = load_conv(*weights->store, source, "asp.conv", conv_weight_storage_type, 1536, 128, 1, 1);
    weights->embedding_dim = assets.config.speaker_encoder.embedding_dim;
    weights->fc = load_conv(*weights->store, source, "fc", conv_weight_storage_type, weights->embedding_dim, 3072, 1, 1);
    weights->store->upload();
    return weights;
}

}  // namespace

class Qwen3TTSEcapaTdnnEncoderGraph {
public:
    Qwen3TTSEcapaTdnnEncoderGraph(
        std::shared_ptr<const Qwen3TTSEcapaTdnnEncoderWeights> weights,
        int64_t frames,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          frames_(frames),
          constants_(execution_context.backend(), std::max(1, execution_context.config().threads), "qwen3_tts.speaker_encoder.constants"),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Qwen3 speaker encoder graph requires weights");
        }
        if (frames_ <= 0) {
            throw std::runtime_error("Qwen3 speaker encoder graph requires positive frame count");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 speaker encoder backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 speaker encoder ggml context");
        }

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "qwen3_tts.speaker_encoder",
            execution_context.backend_type(),
        };
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kFeatureDim, frames_}));
        input_ = input.tensor;
        constants_.begin_graph();
        auto x = tdnn(build_ctx, input, weights_->block0, constants_);
        std::vector<core::TensorValue> hidden;
        for (const auto & block : weights_->blocks) {
            x = se_res2net(build_ctx, x, block, constants_);
            hidden.push_back(x);
        }
        x = modules::ConcatModule({1}).build(
            build_ctx,
            modules::ConcatModule({1}).build(build_ctx, hidden[0], hidden[1]),
            hidden[2]);
        x = tdnn(build_ctx, x, weights_->mfa, constants_);
        x = attentive_statistics_pool(build_ctx, x, weights_->asp_tdnn, weights_->asp_conv, constants_);
        output_ = conv1d(build_ctx, x, weights_->fc, constants_).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output_);
        constants_.finish_graph();
        constants_.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3 speaker encoder graph");
        }
    }

    ~Qwen3TTSEcapaTdnnEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const Qwen3TTSEcapaTdnnEncoderWeights & weights, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && frames_ >= frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const audio::AudioTensor & features) {
        if (features.shape.size() != 3 || features.shape[0] != 1 || features.shape[1] != kFeatureDim) {
            throw std::runtime_error("Qwen3 speaker feature tensor has invalid shape");
        }
        const int64_t valid_frames = features.shape[2];
        if (valid_frames > frames_) {
            throw std::runtime_error("Qwen3 speaker feature tensor exceeds graph capacity");
        }
        std::vector<float> padded(static_cast<size_t>(kFeatureDim * frames_), 0.0F);
        for (int64_t c = 0; c < kFeatureDim; ++c) {
            for (int64_t t = 0; t < valid_frames; ++t) {
                padded[static_cast<size_t>(c * frames_ + t)] = features.values[static_cast<size_t>(c * valid_frames + t)];
            }
        }
        ggml_backend_tensor_set(input_, padded.data(), 0, padded.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 speaker encoder graph compute failed");
        }
        std::vector<float> embedding(static_cast<size_t>(weights_->embedding_dim), 0.0F);
        ggml_backend_tensor_get(output_, embedding.data(), 0, embedding.size() * sizeof(float));
        return embedding;
    }

private:
    std::shared_ptr<const Qwen3TTSEcapaTdnnEncoderWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::ConstantTensorCache constants_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

Qwen3TTSEcapaTdnnEncoderRuntime::Qwen3TTSEcapaTdnnEncoderRuntime(
    std::shared_ptr<const Qwen3TTSAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    assets::TensorStorageType conv_weight_storage_type)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Qwen3 speaker encoder requires assets");
    }
    if (!assets_->config.has_speaker_encoder) {
        throw std::runtime_error("Qwen3 speaker encoder requires a base model with speaker_encoder_config");
    }
    weights_ = load_weights(*assets_, execution_context_->backend(), execution_context_->backend_type(), conv_weight_storage_type);
}

Qwen3TTSEcapaTdnnEncoderRuntime::~Qwen3TTSEcapaTdnnEncoderRuntime() = default;

Qwen3SpeakerFeatures Qwen3TTSEcapaTdnnEncoderRuntime::extract_features(const runtime::AudioBuffer & audio) const {
    const int threads = execution_context_ != nullptr ? std::max(1, execution_context_->config().threads) : 1;
    auto mel = compute_qwen3_speaker_mel(audio, threads);
    if (mel.shape.size() != 3 || mel.shape[1] != kFeatureDim) {
        throw std::runtime_error("Qwen3 speaker frontend produced invalid feature shape");
    }
    Qwen3SpeakerFeatures features;
    features.values = std::move(mel.values);
    features.mel_bins = mel.shape[1];
    features.frames = mel.shape[2];
    return features;
}

Qwen3SpeakerEmbedding Qwen3TTSEcapaTdnnEncoderRuntime::encode(const runtime::AudioBuffer & audio) const {
    return encode_features(extract_features(audio));
}

Qwen3SpeakerEmbedding Qwen3TTSEcapaTdnnEncoderRuntime::encode_features(const Qwen3SpeakerFeatures & extracted) const {
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Qwen3 speaker encoder execution context is missing");
    }
    const int threads = std::max(1, execution_context_->config().threads);
    audio::AudioTensor features;
    features.values = extracted.values;
    features.shape = {1, extracted.mel_bins, extracted.frames};
    const int64_t frames = extracted.frames;
    if (graph_ == nullptr || !graph_->matches(*weights_, frames, execution_context_->backend(), threads)) {
        const auto build_start = Clock::now();
        graph_.reset();
        graph_ = std::make_unique<Qwen3TTSEcapaTdnnEncoderGraph>(
            weights_,
            frames,
            *execution_context_,
            graph_arena_bytes_);
        debug::timing_log_scalar("qwen3_tts.speaker_encoder.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
    } else {
        debug::timing_log_scalar("qwen3_tts.speaker_encoder.graph.build_ms", 0.0);
    }
    Qwen3SpeakerEmbedding embedding;
    embedding.values = graph_->run(features);
    embedding.dims = weights_->embedding_dim;
    return embedding;
}

}  // namespace engine::models::qwen3_tts

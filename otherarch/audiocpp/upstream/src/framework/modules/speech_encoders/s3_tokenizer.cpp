#include "engine/framework/modules/speech_encoders/s3_tokenizer.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

inline int torch_round_to_int(float value) {
    return static_cast<int>(std::nearbyint(value));
}

bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

struct S3TokenizerLogMelOutputs {
    std::vector<float> log_mel;
    int64_t n_mels = 0;
    int64_t frames = 0;
};

std::vector<float> resample_s3_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ActualOutput;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = "S3 tokenizer log-mel";
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(input, input_sample_rate, output_sample_rate, options);
}

S3TokenizerLogMelOutputs compute_s3tokenizer_log_mel(const runtime::AudioBuffer & audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("S3 tokenizer log-mel expects mono audio");
    }
    std::vector<float> mono = audio.sample_rate == 16000
        ? audio.samples
        : resample_s3_mono(audio.samples, audio.sample_rate, 16000);
    constexpr int64_t n_fft = 400;
    constexpr int64_t hop = 160;
    constexpr int64_t n_mels = 128;
    const int64_t pad = n_fft / 2;
    const int64_t padded_samples = static_cast<int64_t>(mono.size()) + 2 * pad;
    const int64_t total_frames = 1 + (padded_samples - n_fft) / hop;
    const int64_t freq_bins = (n_fft / 2) + 1;
    const int64_t frames = total_frames - 1;
    const auto filterbank = engine::audio::MelFilterbank().build(
        engine::audio::MelFilterbankConfig{16000, n_fft, n_mels, 0.0f, 8000.0f, true});
    const engine::audio::STFTConfig window_config{
        n_fft,
        hop,
        n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(window_config);
    const engine::audio::STFTConfig stft_config{
        n_fft,
        hop,
        n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        mono,
        window,
        1,
        static_cast<int64_t>(mono.size()),
        stft_config);

    S3TokenizerLogMelOutputs out;
    out.n_mels = n_mels;
    out.frames = frames;
    out.log_mel.assign(static_cast<size_t>(n_mels * frames), 0.0f);

#ifdef _OPENMP
#pragma omp parallel for if (frames > 8)
#endif
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < n_mels; ++mel_bin) {
            float sum = 0.0f;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * total_frames + frame_index)];
                sum += filterbank.values[static_cast<size_t>(mel_bin * freq_bins + freq)] * (mag * mag);
            }
            out.log_mel[static_cast<size_t>(mel_bin * frames + frame_index)] =
                std::log10(std::max(sum, 1.0e-10f));
        }
    }
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : out.log_mel) {
        max_value = std::max(max_value, value);
    }
    for (float & value : out.log_mel) {
        value = std::max(value, max_value - 8.0f);
        value = (value + 4.0f) / 4.0f;
    }
    return out;
}

engine::core::TensorValue view_bthd_from_btc_chunk(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t time,
    int64_t heads,
    int64_t head_dim) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        head_dim,
        heads,
        time,
        input.shape.dims[0],
        static_cast<size_t>(head_dim) * sizeof(float),
        input.tensor->nb[1],
        input.tensor->nb[2],
        static_cast<size_t>(start) * sizeof(float));
    return engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], time, heads, head_dim}),
        input.type);
}

class S3TokenizerBackendRunner {
public:
    S3TokenizerBackendRunner(
        const S3TokenizerWeights & weights,
        int64_t input_frames)
        : time_(input_frames),
          config_(weights.config),
          execution_context_(*weights.execution_context) {
        const auto & config = config_;
        constexpr int64_t batch = 1;
        const int64_t input_mels = config.input_mels;
        const int64_t channels = config.channels;
        const int64_t heads = config.heads;
        const int64_t head_dim = config.head_dim;
        const int64_t half_head_dim = head_dim / 2;

        ggml_init_params params = {};
        params.mem_size = 512ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for S3 tokenizer");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_tokenizer";
        ctx.backend_type = execution_context_.backend_type();

        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, input_mels, input_frames}));

        auto x = Conv1dModule({
            input_mels,
            channels,
            config.conv_kernel,
            static_cast<int>(config.conv_stride),
            static_cast<int>(config.conv_padding),
            1,
            true,
        }).build(ctx, input_, weights.conv1);
        x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
        time_ = x.shape.dims[2];

        x = Conv1dModule({
            channels,
            channels,
            config.conv_kernel,
            static_cast<int>(config.conv_stride),
            static_cast<int>(config.conv_padding),
            1,
            true,
        }).build(ctx, x, weights.conv2);
        x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
        time_ = x.shape.dims[2];

        cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, time_, heads, half_head_dim}));
        sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, time_, heads, half_head_dim}));

        auto seq = TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        for (const auto & block : weights.blocks) {
            auto norm = LayerNormModule({channels, 1.0e-5f, true, true}).build(ctx, seq, block.attn_ln);
            auto qkv = LinearModule({
                channels,
                channels * 3,
                true,
                GGML_PREC_DEFAULT,
            }).build(ctx, norm, block.attn_qkv_packed);
            qkv = engine::core::ensure_backend_addressable_layout(ctx, qkv);

            auto q = view_bthd_from_btc_chunk(ctx, qkv, 0, time_, heads, head_dim);
            auto k = view_bthd_from_btc_chunk(ctx, qkv, channels, time_, heads, head_dim);
            auto v_flat = SliceModule({2, 2 * channels, channels}).build(ctx, qkv);
            auto v_bct = TransposeModule({{0, 2, 1, 3}, v_flat.shape.rank}).build(ctx, v_flat);
            auto fsmn = DepthwiseConv1dModule({
                channels,
                config.fsmn_kernel,
                1,
                static_cast<int>(config.fsmn_padding),
                1,
                false,
            }).build(ctx, v_bct, block.fsmn);
            fsmn = AddModule{}.build(ctx, fsmn, v_bct);
            fsmn = TransposeModule({{0, 2, 1, 3}, fsmn.shape.rank}).build(ctx, fsmn);

            auto v = view_bthd_from_btc_chunk(ctx, qkv, 2 * channels, time_, heads, head_dim);

            q = SplitRoPEModule({head_dim}).build(ctx, q, cos_, sin_);
            k = SplitRoPEModule({head_dim}).build(ctx, k, cos_, sin_);

            q = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
            k = TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
            v = TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);

            auto context = ScaledDotProductAttentionModule({
                head_dim,
                ScaledDotProductAttentionLowering::Explicit,
                GGML_PREC_DEFAULT,
                AttentionCausality::NonCausal,
            }).build(ctx, q, k, v);
            context = engine::core::ensure_backend_addressable_layout(ctx, context);
            context = engine::core::reshape_tensor(
                ctx,
                context,
                engine::core::TensorShape::from_dims({batch, time_, channels}));
            auto attn_out = LinearModule({
                channels,
                channels,
                true,
                GGML_PREC_DEFAULT,
            }).build(ctx, context, block.attn_out);

            seq = AddModule{}.build(ctx, AddModule{}.build(ctx, seq, attn_out), fsmn);

            auto mlp_in = LayerNormModule({channels, 1.0e-5f, true, true}).build(ctx, seq, block.mlp_ln);
            auto ff = LinearModule({
                channels,
                channels * 4,
                true,
                GGML_PREC_DEFAULT,
            }).build(ctx, mlp_in, block.mlp_fc1);
            ff = GeluModule({GeluApproximation::ExactErf}).build(ctx, ff);
            ff = LinearModule({
                channels * 4,
                channels,
                true,
                GGML_PREC_DEFAULT,
            }).build(ctx, ff, block.mlp_fc2);
            seq = AddModule{}.build(ctx, seq, ff);
        }

        quant_out_ = LinearModule({
            channels,
            config.quantizer_dim,
            true,
            GGML_PREC_DEFAULT,
        }).build(ctx, seq, weights.quantizer_project_down);
        ggml_set_name(quant_out_.tensor, "framework_s3tokenizer_quant_out");

        graph_ = ggml_new_graph_custom(ggml_, 65536, false);
        ggml_build_forward_expand(graph_, quant_out_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            ggml_free(ggml_);
            ggml_ = nullptr;
            throw std::runtime_error("failed to allocate backend tensors for S3 tokenizer");
        }

        std::vector<float> cos_values(static_cast<size_t>(time_ * heads * half_head_dim), 0.0f);
        std::vector<float> sin_values(static_cast<size_t>(time_ * heads * half_head_dim), 0.0f);
        for (int64_t t = 0; t < time_; ++t) {
            for (int64_t head = 0; head < heads; ++head) {
                for (int64_t i = 0; i < half_head_dim; ++i) {
                    const float inv_freq =
                        1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                    const float angle = static_cast<float>(t) * inv_freq;
                    const size_t offset = static_cast<size_t>((t * heads + head) * half_head_dim + i);
                    cos_values[offset] = std::cos(angle);
                    sin_values[offset] = std::sin(angle);
                }
            }
        }
        engine::core::write_tensor_f32(cos_, cos_values);
        engine::core::write_tensor_f32(sin_, sin_values);
    }

    ~S3TokenizerBackendRunner() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
    }

    S3TokenizerOutputs run(const std::vector<float> & mel_bct) {
        std::lock_guard<std::mutex> lock(run_mutex_);
        engine::core::write_tensor_f32(input_, mel_bct);
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for S3 tokenizer");
        }
        auto down = engine::core::read_tensor_f32(quant_out_.tensor);

        S3TokenizerOutputs outputs;
        outputs.token_count = time_;
        outputs.tokens.assign(static_cast<size_t>(time_), 0);
        const auto & config = config_;
        std::vector<int> powers(static_cast<size_t>(config.quantizer_dim), 1);
        for (int64_t index = 1; index < config.quantizer_dim; ++index) {
            powers[static_cast<size_t>(index)] =
                powers[static_cast<size_t>(index - 1)] * static_cast<int>(config.quantizer_base);
        }
        for (int64_t t = 0; t < time_; ++t) {
            int token = 0;
            for (int64_t d = 0; d < config.quantizer_dim; ++d) {
                float xval = std::tanh(down[static_cast<size_t>(t * config.quantizer_dim + d)]);
                xval *= config.quantizer_scale;
                const int level = torch_round_to_int(xval) + static_cast<int>(config.quantizer_offset);
                token += level * powers[static_cast<size_t>(d)];
            }
            outputs.tokens[static_cast<size_t>(t)] = token;
        }
        return outputs;
    }

private:
    int64_t time_ = 0;
    S3TokenizerConfig config_;
    const engine::core::ExecutionContext & execution_context_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    engine::core::TensorValue cos_;
    engine::core::TensorValue sin_;
    engine::core::TensorValue quant_out_;
    std::mutex run_mutex_;
};

}  // namespace

class S3TokenizerInferenceSessionCache {
public:
    S3TokenizerInferenceSessionCache();
    ~S3TokenizerInferenceSessionCache();
    S3TokenizerInferenceSessionCache(S3TokenizerInferenceSessionCache &&) noexcept;
    S3TokenizerInferenceSessionCache & operator=(S3TokenizerInferenceSessionCache &&) noexcept;

    S3TokenizerInferenceSessionCache(const S3TokenizerInferenceSessionCache &) = delete;
    S3TokenizerInferenceSessionCache & operator=(const S3TokenizerInferenceSessionCache &) = delete;

    S3TokenizerOutputs run_backend(
        const S3TokenizerWeights & weights,
        const std::vector<float> & mel_bct,
        int64_t frames,
        const engine::core::BackendConfig & backend);

private:
    struct State;
    std::unique_ptr<State> state_;
};

struct S3TokenizerInferenceSessionCache::State {
    std::mutex mutex;
    std::shared_ptr<S3TokenizerBackendRunner> runner;
    const S3TokenizerWeights * weights = nullptr;
    int64_t frames = 0;
    engine::core::BackendConfig backend;
};

S3TokenizerInferenceSessionCache::S3TokenizerInferenceSessionCache()
    : state_(std::make_unique<State>()) {}

S3TokenizerInferenceSessionCache::~S3TokenizerInferenceSessionCache() = default;
S3TokenizerInferenceSessionCache::S3TokenizerInferenceSessionCache(S3TokenizerInferenceSessionCache &&) noexcept = default;
S3TokenizerInferenceSessionCache &
S3TokenizerInferenceSessionCache::operator=(S3TokenizerInferenceSessionCache &&) noexcept = default;

S3TokenizerOutputs S3TokenizerInferenceSessionCache::run_backend(
    const S3TokenizerWeights & weights,
    const std::vector<float> & mel_bct,
    int64_t frames,
    const engine::core::BackendConfig & backend) {
    std::shared_ptr<S3TokenizerBackendRunner> runner;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->runner ||
            state_->weights != &weights ||
            state_->frames != frames ||
            !same_backend(state_->backend, backend)) {
            if (!same_backend(weights.execution_context->config(), backend)) {
                throw std::runtime_error("S3 tokenizer backend does not match uploaded weight backend");
            }
            state_->runner = std::make_shared<S3TokenizerBackendRunner>(weights, frames);
            state_->weights = &weights;
            state_->frames = frames;
            state_->backend = backend;
        }
        runner = state_->runner;
    }
    return runner->run(mel_bct);
}

static std::shared_ptr<S3TokenizerInferenceSessionCache> make_s3_tokenizer_inference_session_cache() {
    return std::make_shared<S3TokenizerInferenceSessionCache>();
}

static int64_t discover_s3_block_count(
    const engine::assets::TensorSource & source,
    const std::string & encoder_prefix) {
    int64_t count = 0;
    while (source.has_tensor(encoder_prefix + ".blocks." + std::to_string(count) + ".attn_ln.weight")) {
        ++count;
    }
    if (count <= 0) {
        throw std::runtime_error("S3 tokenizer has no encoder blocks");
    }
    return count;
}

static std::shared_ptr<const S3TokenizerWeights> load_s3tokenizer_weights(
    std::shared_ptr<const engine::assets::TensorSource> source,
    const engine::core::ExecutionContext & execution_context,
    S3TokenizerConfig config) {
    auto weights = std::make_shared<S3TokenizerWeights>();
    weights->config = std::move(config);
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "framework.s3_tokenizer.weights",
        weights->config.weight_context_bytes);
    const auto & tensor_source = *source;
    const std::string encoder_prefix = weights->config.tensor_prefix + ".encoder";
    weights->conv1 = binding::conv1d_from_source(
        *weights->store,
        tensor_source,
        encoder_prefix + ".conv1",
        weights->config.weight_storage_type,
        weights->config.channels,
        weights->config.input_mels,
        weights->config.conv_kernel,
        true);
    weights->conv2 = binding::conv1d_from_source(
        *weights->store,
        tensor_source,
        encoder_prefix + ".conv2",
        weights->config.weight_storage_type,
        weights->config.channels,
        weights->config.channels,
        weights->config.conv_kernel,
        true);
    const int64_t block_count = discover_s3_block_count(tensor_source, encoder_prefix);
    weights->blocks.resize(static_cast<size_t>(block_count));
    for (int64_t layer = 0; layer < block_count; ++layer) {
        const std::string prefix = encoder_prefix + ".blocks." + std::to_string(layer);
        auto & block = weights->blocks[static_cast<size_t>(layer)];
        block.attn_ln = binding::norm_from_source(*weights->store, tensor_source, prefix + ".attn_ln", weights->config.channels);

        const int64_t qkv_out_features = weights->config.channels * 3;
        const int64_t qkv_in_features = weights->config.channels;
        std::vector<float> qkv_weight(static_cast<size_t>(qkv_out_features * qkv_in_features));
        auto query_weight = tensor_source.require_f32(prefix + ".attn.query.weight", {weights->config.channels, weights->config.channels});
        auto key_weight = tensor_source.require_f32(prefix + ".attn.key.weight", {weights->config.channels, weights->config.channels});
        auto value_weight = tensor_source.require_f32(prefix + ".attn.value.weight", {weights->config.channels, weights->config.channels});
        std::copy(
            query_weight.begin(),
            query_weight.end(),
            qkv_weight.begin());
        std::copy(
            key_weight.begin(),
            key_weight.end(),
            qkv_weight.begin() + static_cast<ptrdiff_t>(weights->config.channels * weights->config.channels));
        std::copy(
            value_weight.begin(),
            value_weight.end(),
            qkv_weight.begin() + static_cast<ptrdiff_t>(2 * weights->config.channels * weights->config.channels));
        std::vector<float> qkv_bias(static_cast<size_t>(qkv_out_features), 0.0f);
        auto query_bias = tensor_source.require_f32(prefix + ".attn.query.bias", {weights->config.channels});
        auto value_bias = tensor_source.require_f32(prefix + ".attn.value.bias", {weights->config.channels});
        std::copy(
            query_bias.begin(),
            query_bias.end(),
            qkv_bias.begin());
        std::copy(
            value_bias.begin(),
            value_bias.end(),
            qkv_bias.begin() + static_cast<ptrdiff_t>(2 * weights->config.channels));
        block.attn_qkv_packed.weight = weights->store->make_from_f32(
            engine::core::TensorShape::from_dims({qkv_out_features, qkv_in_features}),
            weights->config.weight_storage_type,
            std::move(qkv_weight));
        block.attn_qkv_packed.bias = weights->store->make_f32(
            engine::core::TensorShape::from_dims({qkv_out_features}),
            std::move(qkv_bias));
        block.attn_out = binding::linear_from_source(
            *weights->store,
            tensor_source,
            prefix + ".attn.out",
            weights->config.weight_storage_type,
            weights->config.channels,
            weights->config.channels,
            true);
        block.fsmn = binding::depthwise_conv1d_from_source(
            *weights->store,
            tensor_source,
            prefix + ".attn.fsmn_block",
            weights->config.weight_storage_type,
            weights->config.channels,
            weights->config.fsmn_kernel,
            false);
        block.mlp_ln = binding::norm_from_source(*weights->store, tensor_source, prefix + ".mlp_ln", weights->config.channels);
        block.mlp_fc1 = binding::linear_from_source(
            *weights->store,
            tensor_source,
            prefix + ".mlp.0",
            weights->config.weight_storage_type,
            weights->config.channels * 4,
            weights->config.channels,
            true);
        block.mlp_fc2 = binding::linear_from_source(
            *weights->store,
            tensor_source,
            prefix + ".mlp.2",
            weights->config.weight_storage_type,
            weights->config.channels,
            weights->config.channels * 4,
            true);
    }
    weights->quantizer_project_down =
        binding::linear_from_source(
            *weights->store,
            tensor_source,
            weights->config.tensor_prefix + ".quantizer._codebook.project_down",
            weights->config.weight_storage_type,
            weights->config.quantizer_dim,
            weights->config.channels,
            true);
    weights->store->upload();
    return weights;
}

S3TokenizerOutputs compute_s3tokenizer_v2_codes(
    const S3TokenizerWeights & weights,
    const runtime::AudioBuffer & audio,
    std::optional<int64_t> max_len,
    S3TokenizerInferenceSessionCache * cache,
    engine::core::BackendConfig backend) {
    auto mel = compute_s3tokenizer_log_mel(audio);
    if (max_len && *max_len > 0) {
        const int64_t max_frames = *max_len * 4;
        if (mel.frames > max_frames) {
            std::vector<float> trimmed(static_cast<size_t>(mel.n_mels * max_frames), 0.0f);
            for (int64_t m = 0; m < mel.n_mels; ++m) {
                const float * src = mel.log_mel.data() + static_cast<ptrdiff_t>(m * mel.frames);
                float * dst = trimmed.data() + static_cast<ptrdiff_t>(m * max_frames);
                std::copy(src, src + max_frames, dst);
            }
            mel.log_mel = std::move(trimmed);
            mel.frames = max_frames;
        }
    }
    if (cache == nullptr) {
        S3TokenizerInferenceSessionCache local_cache;
        return local_cache.run_backend(weights, mel.log_mel, mel.frames, backend);
    }
    return cache->run_backend(weights, mel.log_mel, mel.frames, backend);
}

struct S3TokenizerComponent::State {
    std::shared_ptr<S3TokenizerInferenceSessionCache> cache;
};

S3TokenizerComponent S3TokenizerComponent::load_from_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    const engine::core::ExecutionContext & execution_context,
    S3TokenizerConfig config) {
    if (source == nullptr) {
        throw std::runtime_error("S3 tokenizer requires tensor source");
    }
    auto weights = load_s3tokenizer_weights(std::move(source), execution_context, std::move(config));
    S3TokenizerComponent component(std::move(weights), execution_context);
    component.state_ = std::make_shared<State>(State{
        make_s3_tokenizer_inference_session_cache(),
    });
    return component;
}

S3TokenizerComponent::S3TokenizerComponent(
    std::shared_ptr<const S3TokenizerWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context) {}

const engine::core::BackendConfig & S3TokenizerComponent::backend() const noexcept {
    return execution_context_->config();
}

const std::shared_ptr<const S3TokenizerWeights> & S3TokenizerComponent::weights() const noexcept {
    return weights_;
}

S3TokenizerOutputs S3TokenizerComponent::tokenize(
    const runtime::AudioBuffer & audio,
    std::optional<int64_t> max_len) const {
    return compute_s3tokenizer_v2_codes(
        *weights_,
        audio,
        max_len,
        state_->cache.get(),
        execution_context_->config());
}

void S3TokenizerComponent::release_runtime_cache() const {
    if (state_ != nullptr) {
        state_->cache = make_s3_tokenizer_inference_session_cache();
    }
}

S3TokenizerComponent::~S3TokenizerComponent() = default;
S3TokenizerComponent::S3TokenizerComponent(S3TokenizerComponent &&) noexcept = default;
S3TokenizerComponent & S3TokenizerComponent::operator=(S3TokenizerComponent &&) noexcept = default;

}  // namespace engine::modules

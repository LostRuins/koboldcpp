#include "engine/community_models/soprano_tts/vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::soprano_tts {

struct SopranoConvNeXtBlockWeights {
    engine::modules::DepthwiseConv1dWeights dwconv;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pwconv1;
    engine::modules::LinearWeights pwconv2;
    engine::core::TensorValue gamma;
};

struct SopranoVocosWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<SopranoConvNeXtBlockWeights> convnext;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights head_out;
    std::vector<float> istft_window;
};

namespace {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

engine::core::TensorValue scale_last_dim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & scale) {
    const auto view = engine::core::reshape_tensor(
        ctx, scale, engine::core::TensorShape::from_dims({1, 1, scale.shape.dims[0]}));
    const auto repeated = engine::modules::RepeatModule({input.shape}).build(ctx, view);
    return engine::modules::MulModule{}.build(ctx, input, repeated);
}

}  // namespace
std::shared_ptr<const SopranoVocosWeights> load_decoder_weights(
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::assets::TensorSource & source,
    const SopranoTTSConfig & config,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<SopranoVocosWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend, backend_type, "soprano_tts.decoder.weights", weight_context_bytes);
    weights->embed = engine::modules::binding::conv1d_from_source(
        *weights->store, source, "decoder.embed", conv_storage_type,
        config.decoder_dim, config.decoder_input_channels, 1, true);
    weights->norm = engine::modules::binding::norm_from_source(
        *weights->store, source, "decoder.norm", config.decoder_dim);
    weights->convnext.reserve(static_cast<size_t>(config.decoder_num_layers));
    for (int64_t layer = 0; layer < config.decoder_num_layers; ++layer) {
        const std::string prefix = "decoder.convnext." + std::to_string(layer);
        SopranoConvNeXtBlockWeights block;
        block.dwconv = engine::modules::binding::depthwise_conv1d_from_source(
            *weights->store, source, prefix + ".dwconv", conv_storage_type,
            config.decoder_dim, static_cast<int>(config.dw_kernel), true);
        block.norm = engine::modules::binding::norm_from_source(
            *weights->store, source, prefix + ".norm", config.decoder_dim);
        block.pwconv1 = engine::modules::binding::linear_from_source(
            *weights->store, source, prefix + ".pwconv1", matmul_storage_type,
            config.decoder_intermediate_dim, config.decoder_dim, true);
        block.pwconv2 = engine::modules::binding::linear_from_source(
            *weights->store, source, prefix + ".pwconv2", matmul_storage_type,
            config.decoder_dim, config.decoder_intermediate_dim, true);
        block.gamma = weights->store->load_f32_tensor(
            source, prefix + ".gamma", {config.decoder_dim});
        weights->convnext.push_back(std::move(block));
    }
    weights->final_norm = engine::modules::binding::norm_from_source(
        *weights->store, source, "decoder.final_layer_norm", config.decoder_dim);
    weights->head_out = engine::modules::binding::linear_from_source(
        *weights->store, source, "decoder.head.out", matmul_storage_type,
        config.n_fft + 2, config.decoder_dim, true);
    weights->istft_window = source.require_f32("decoder.head.istft.window", {config.n_fft});
    weights->store->upload();
    return weights;
}

engine::core::TensorValue build_convnext_block(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const SopranoConvNeXtBlockWeights & weights,
    const SopranoTTSConfig & config) {
    auto hidden = engine::modules::DepthwiseConv1dModule({
        static_cast<int>(config.decoder_dim), static_cast<int>(config.dw_kernel),
        1, static_cast<int>(config.dw_kernel / 2), 1,
        weights.dwconv.bias.has_value(),
    }).build(ctx, input_bct, weights.dwconv);
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.decoder_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::LinearModule({
        config.decoder_dim, config.decoder_intermediate_dim, true, GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv1);
    hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = engine::modules::LinearModule({
        config.decoder_intermediate_dim, config.decoder_dim, true, GGML_PREC_F32,
    }).build(ctx, hidden, weights.pwconv2);
    hidden = scale_last_dim(ctx, hidden, weights.gamma);
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    return engine::modules::AddModule{}.build(ctx, input_bct, hidden);
}

engine::core::TensorValue build_decoder_head(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & feat_bct,
    const SopranoVocosWeights & weights,
    const SopranoTTSConfig & config,
    int64_t output_frames) {
    // SopranoDecoder: interpolate upscale (4*(T-1)+1 frames), embed, ConvNeXt,
    // then a single ISTFT head projection (Linear(dim -> n_fft+2)).
    engine::core::TensorValue hidden;
    // Use align_corners=True to match F.interpolate(mode='linear', align_corners=True).
    // Interpolate1dModule::Linear does NOT set ALIGN_CORNERS, so call ggml directly.
    {
        const auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, feat_bct);
        auto output_shape = feat_bct.shape;
        output_shape.dims[output_shape.rank - 1] = output_frames;
        ggml_tensor * interp = ggml_interpolate(
            ctx.ggml,
            contiguous.tensor,
            output_frames,
            contiguous.tensor->ne[1],
            contiguous.tensor->ne[2],
            contiguous.tensor->ne[3],
            static_cast<enum ggml_scale_mode>(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS));
        hidden = engine::core::wrap_tensor(interp, output_shape, GGML_TYPE_F32);
    }
    hidden = engine::modules::Conv1dModule({
        config.decoder_input_channels, config.decoder_dim, 1, 1, 0, 1,
        weights.embed.bias.has_value(),
    }).build(ctx, hidden, weights.embed);
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.decoder_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.norm);
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    for (const auto & block : weights.convnext) {
        hidden = build_convnext_block(ctx, hidden, block, config);
    }
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = engine::modules::LayerNormModule({config.decoder_dim, 1.0e-6F, true, true})
                 .build(ctx, hidden, weights.final_norm);
    // Keep channel-last (…, T2, 768) for the head.out linear.
    hidden = engine::modules::LinearModule({
        config.decoder_dim, config.n_fft + 2, true, GGML_PREC_F32,
    }).build(ctx, hidden, weights.head_out);
    return hidden;
}
namespace {

// Reconstruct audio from head output (log-magnitude|phase halves) with a single
// non-iterative ISTFT pass, mirroring the SopranoDecoder head.
// Matches torch.istft(spec, n_fft, hop, win, window, center=True):
// - Produces (frames-1)*hop_length + n_fft raw samples
// - Trims n_fft//2 from each side → (frames-1)*hop_length output samples
std::vector<float> istft_center_from_head(
    const std::vector<float> & head,
    int64_t frames,
    const SopranoTTSConfig & config,
    const std::vector<float> & window,
    size_t threads) {
    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t out_dim = config.n_fft + 2;
    if (static_cast<int64_t>(head.size()) != frames * out_dim) {
        throw std::runtime_error("Soprano decoder head output shape mismatch");
    }
    if (static_cast<int64_t>(window.size()) != config.n_fft) {
        throw std::runtime_error("Soprano decoder ISTFT window shape mismatch");
    }
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(frames * freq_bins));
    const int omp_threads = static_cast<int>(std::max<size_t>(1, threads));
#ifdef _OPENMP
#pragma omp parallel for num_threads(omp_threads) if (frames >= 8)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float * row = head.data() + static_cast<size_t>(frame * out_dim);
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            float mag = std::min(std::exp(row[freq]), 100.0F);
            // Zero out first and last freq bins (matching reference bugfix)
            if (freq == 0 || freq == freq_bins - 1) {
                mag = 0.0F;
            }
            const float phase = row[freq_bins + freq];
            spectrum[static_cast<size_t>(frame * freq_bins + freq)] = {
                mag * std::cos(phase), mag * std::sin(phase)};
        }
    }
    std::vector<float> framed(static_cast<size_t>(frames * config.n_fft), 0.0F);
    engine::audio::real_fft_inverse(
        {static_cast<size_t>(frames), static_cast<size_t>(config.n_fft)},
        {
            static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
            static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
        },
        {
            static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
            static_cast<std::ptrdiff_t>(sizeof(float)),
        },
        1, spectrum.data(), framed.data(),
        1.0F / static_cast<float>(config.n_fft), threads);

    // No output trimming: match torch.istft with center=True which produces
    // (frames-1)*hop_length + n_fft samples.
    const int64_t output_size = (frames - 1) * config.hop_length + config.n_fft;
    std::vector<float> folded(static_cast<size_t>(output_size), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(output_size), 0.0F);
    // OLA parallelized over contiguous output blocks; each block gathers the
    // overlapping window contributions frame-by-frame in ascending order, so
    // the per-sample accumulation order matches the serial version exactly.
    {
        const int64_t block = 4096;
        const int64_t nblocks = (output_size + block - 1) / block;
#ifdef _OPENMP
#pragma omp parallel for num_threads(omp_threads) if (nblocks > 1)
#endif
        for (int64_t b = 0; b < nblocks; ++b) {
            const int64_t b0 = b * block;
            const int64_t b1 = std::min(output_size, b0 + block);
            int64_t f0 = (b0 - config.n_fft) / config.hop_length + 1;
            if (f0 < 0) {
                f0 = 0;
            }
            int64_t f1 = (b1 - 1) / config.hop_length;
            if (f1 >= frames) {
                f1 = frames - 1;
            }
            for (int64_t frame = f0; frame <= f1; ++frame) {
                const int64_t start = frame * config.hop_length;
                int64_t i0 = b0 - start;
                if (i0 < 0) {
                    i0 = 0;
                }
                int64_t i1 = b1 - start;
                if (i1 > config.n_fft) {
                    i1 = config.n_fft;
                }
                const float * src = framed.data() + static_cast<size_t>(frame * config.n_fft);
                for (int64_t i = i0; i < i1; ++i) {
                    const float w = window[static_cast<size_t>(i)];
                    folded[static_cast<size_t>(start + i)] += src[i] * w;
                    envelope[static_cast<size_t>(start + i)] += w * w;
                }
            }
        }
    }
    if (output_size <= 0) {
        throw std::runtime_error("Soprano decoder ISTFT produced non-positive output size");
    }
    // torch.istft with center=True: trim n_fft//2 from each side.
    // Final output: (frames-1)*hop_length samples.
    const int64_t pad = config.n_fft / 2;
    const int64_t samples = output_size - 2 * pad;
    if (samples <= 0) {
        throw std::runtime_error("Soprano decoder ISTFT produced non-positive samples after trim");
    }
    std::vector<float> audio(static_cast<size_t>(samples), 0.0F);
    for (int64_t i = 0; i < samples; ++i) {
        const int64_t src = i + pad;
        const float denom = envelope[static_cast<size_t>(src)];
        if (denom > 1.0e-11F) {
            audio[static_cast<size_t>(i)] = folded[static_cast<size_t>(src)] / denom;
        }
    }
    return audio;
}

}  // namespace

struct SopranoVocosGraph {
    SopranoVocosGraph(
        ggml_backend_t backend,
        engine::core::BackendType backend_type,
        size_t graph_context_bytes,
        const SopranoTTSConfig & config,
        std::shared_ptr<const SopranoVocosWeights> weights,
        int64_t frames_in)
        : backend(backend),
          weights(std::move(weights)),
          frames(frames_in),
          input_channels(config.decoder_input_channels),
          head_dim(config.n_fft + 2),
          output_frames(config.upscale * (frames_in - 1) + 1),
          config(&config) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Soprano decoder graph requires backend and weights");
        }
        if (frames_in <= 0) {
            throw std::runtime_error("Soprano decoder graph requires positive frame count");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize soprano decoder graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx.get(), "soprano_tts.decoder", backend_type};
        input = engine::core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config.decoder_input_channels, frames})).tensor;
        auto feat = engine::core::wrap_tensor(
            input,
            engine::core::TensorShape::from_dims({1, config.decoder_input_channels, frames}),
            GGML_TYPE_F32);
        auto head = build_decoder_head(build_ctx, feat, *this->weights, config, output_frames);
        head = engine::core::ensure_backend_addressable_layout(build_ctx, head);
        output = head.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, output);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
            !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate soprano decoder graph");
        }
    }

    ~SopranoVocosGraph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }

    bool matches(const SopranoVocosWeights & other, int64_t other_frames) const noexcept {
        return weights.get() == &other && frames == other_frames;
    }

    std::vector<float> run(
        const std::vector<float> & features,
        int64_t frame_count,
        const std::vector<float> & window,
        size_t threads) {
        std::vector<float> bct(static_cast<size_t>(input_channels * frame_count), 0.0F);
        for (int64_t frame = 0; frame < frame_count; ++frame) {
            for (int64_t c = 0; c < input_channels; ++c) {
                bct[static_cast<size_t>(c * frame_count + frame)] =
                    features[static_cast<size_t>(frame * input_channels + c)];
            }
        }
        ggml_backend_tensor_set(input, bct.data(), 0, bct.size() * sizeof(float));
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Soprano decoder graph compute failed");
        }
        std::vector<float> head(static_cast<size_t>(output_frames * head_dim), 0.0F);
        ggml_backend_tensor_get(output, head.data(), 0, head.size() * sizeof(float));

        return istft_center_from_head(head, output_frames, *this->config, window, threads);
    }

    
    ggml_backend_t backend = nullptr;
    std::shared_ptr<const SopranoVocosWeights> weights;
    int64_t frames = 0;
    int64_t input_channels = 0;
    int64_t head_dim = 0;
    int64_t output_frames = 0;
    const SopranoTTSConfig * config = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
};

SopranoVocosRuntime::SopranoVocosRuntime(
    const SopranoTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : config_(assets.config),
      execution_context_(execution_context),
      graph_context_bytes_(graph_context_bytes),
      weights_(load_decoder_weights(
          execution_context.backend(),
          execution_context.backend_type(),
          *assets.decoder_weights,
          assets.config,
          weight_context_bytes,
          matmul_storage_type,
          conv_storage_type)) {}

SopranoVocosRuntime::~SopranoVocosRuntime() = default;

runtime::AudioBuffer SopranoVocosRuntime::decode(
    const std::vector<float> & features,
    int64_t frames) const {
    if (frames <= 0 || static_cast<int64_t>(features.size()) != frames * config_.decoder_input_channels) {
        throw std::runtime_error("Soprano decoder requires consistent feature frames");
    }
    if (graph_ == nullptr || !graph_->matches(*weights_, frames)) {
        graph_ = std::make_unique<SopranoVocosGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            graph_context_bytes_,
            config_,
            weights_,
            frames);
    }
    auto audio = graph_->run(
        features, frames, weights_->istft_window,
        static_cast<size_t>(execution_context_.config().threads));
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(config_.sample_rate);
    out.channels = 1;
    out.samples = std::move(audio);
    return out;
}

// --------------------------------------------------------------------------- // assembly helpers live below; see session.cpp and
// the CMake target for the LM generator + full pipeline.
// --------------------------------------------------------------------------- //
}  // namespace engine::community_models::soprano_tts

#include "engine/framework/codecs/oobleck_audio_vae_runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::codecs {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct OobleckResidualUnitWeights {
    modules::SnakeBeta1dWeights snake_0;
    modules::Conv1dWeights conv_1;
    modules::SnakeBeta1dWeights snake_2;
    modules::Conv1dWeights conv_3;
};

struct OobleckDecoderBlockWeights {
    modules::SnakeBeta1dWeights snake_0;
    modules::ConvTranspose1dWeights upsample;
    OobleckResidualUnitWeights residual_2;
    OobleckResidualUnitWeights residual_3;
    OobleckResidualUnitWeights residual_4;
};

struct OobleckEncoderBlockWeights {
    OobleckResidualUnitWeights residual_0;
    OobleckResidualUnitWeights residual_1;
    OobleckResidualUnitWeights residual_2;
    modules::SnakeBeta1dWeights snake_3;
    modules::Conv1dWeights downsample;
};

struct OobleckAudioVaeWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights encoder_in_conv;
    std::vector<OobleckEncoderBlockWeights> encoder_blocks;
    modules::SnakeBeta1dWeights encoder_final_snake;
    modules::Conv1dWeights encoder_out_conv;
    modules::Conv1dWeights decoder_in_conv;
    std::vector<OobleckDecoderBlockWeights> decoder_blocks;
    modules::SnakeBeta1dWeights decoder_final_snake;
    modules::Conv1dWeights decoder_out_conv;
};

std::vector<int64_t> channel_plan(const OobleckAudioVaeConfig & config) {
    if (config.c_mults.empty()) {
        throw std::runtime_error("Oobleck audio VAE c_mults must not be empty");
    }
    std::vector<int64_t> channels;
    channels.reserve(config.c_mults.size() + 1);
    channels.push_back(1);
    channels.insert(channels.end(), config.c_mults.begin(), config.c_mults.end());
    for (int64_t & value : channels) {
        value *= config.channels;
    }
    return channels;
}

void validate_config(const OobleckAudioVaeConfig & config) {
    if (config.sample_rate <= 0 || config.audio_channels <= 0 || config.channels <= 0 ||
        config.encoder_latent_dim <= 0 || config.decoder_latent_dim <= 0) {
        throw std::runtime_error("Oobleck audio VAE config dimensions must be positive");
    }
    if (config.c_mults.empty() || config.c_mults.size() != config.strides.size()) {
        throw std::runtime_error("Oobleck audio VAE c_mults and strides must have matching positive sizes");
    }
    if (!config.use_snake) {
        throw std::runtime_error("Oobleck audio VAE framework runtime currently supports SnakeBeta checkpoints");
    }
    for (const int64_t stride : config.strides) {
        if (stride <= 0) {
            throw std::runtime_error("Oobleck audio VAE strides must be positive");
        }
    }
}

std::string join_name(const std::string & prefix, const std::string & name) {
    return prefix.empty() ? name : prefix + "." + name;
}

std::vector<float> effective_weight_norm_conv(
    const assets::TensorSource & source,
    const std::string & prefix,
    const std::vector<int64_t> & shape) {
    if (shape.size() != 3) {
        throw std::runtime_error("Oobleck audio VAE weight-norm conv shape must be rank 3");
    }
    auto v = source.require_f32(prefix + ".weight_v", shape);
    auto g = source.require_f32(prefix + ".weight_g", {shape[0], 1, 1});
    const int64_t outer = shape[0];
    const int64_t inner = shape[1] * shape[2];
    for (int64_t o = 0; o < outer; ++o) {
        double norm_sq = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            const float value = v[static_cast<size_t>(o * inner + i)];
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(o)] / std::sqrt(static_cast<float>(std::max(norm_sq, 1.0e-24)));
        for (int64_t i = 0; i < inner; ++i) {
            v[static_cast<size_t>(o * inner + i)] *= scale;
        }
    }
    return v;
}

modules::Conv1dWeights load_wn_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    bool use_bias) {
    modules::Conv1dWeights weights;
    if (source.has_tensor(prefix + ".weight")) {
        weights.weight = store.load_tensor(
            source,
            prefix + ".weight",
            storage_type,
            {out_channels, in_channels, kernel});
    } else {
        weights.weight = store.make_from_f32(
            core::TensorShape::from_dims({out_channels, in_channels, kernel}),
            storage_type,
            effective_weight_norm_conv(source, prefix, {out_channels, in_channels, kernel}));
    }
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::ConvTranspose1dWeights load_wn_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    bool use_bias) {
    modules::ConvTranspose1dWeights weights;
    if (source.has_tensor(prefix + ".weight")) {
        weights.weight = store.load_tensor(
            source,
            prefix + ".weight",
            storage_type,
            {in_channels, out_channels, kernel});
    } else {
        weights.weight = store.make_from_f32(
            core::TensorShape::from_dims({in_channels, out_channels, kernel}),
            storage_type,
            effective_weight_norm_conv(source, prefix, {in_channels, out_channels, kernel}));
    }
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::SnakeBeta1dWeights load_snake(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return {
        store.load_f32_tensor(source, prefix + ".alpha", {channels}),
        store.load_f32_tensor(source, prefix + ".beta", {channels}),
    };
}

OobleckResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels) {
    return {
        load_snake(store, source, prefix + ".layers.0", channels),
        load_wn_conv1d(store, source, prefix + ".layers.1", storage_type, channels, channels, 7, true),
        load_snake(store, source, prefix + ".layers.2", channels),
        load_wn_conv1d(store, source, prefix + ".layers.3", storage_type, channels, channels, 1, true),
    };
}

OobleckEncoderBlockWeights load_encoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride) {
    return {
        load_residual_unit(store, source, prefix + ".layers.0", storage_type, in_channels),
        load_residual_unit(store, source, prefix + ".layers.1", storage_type, in_channels),
        load_residual_unit(store, source, prefix + ".layers.2", storage_type, in_channels),
        load_snake(store, source, prefix + ".layers.3", in_channels),
        load_wn_conv1d(store, source, prefix + ".layers.4", storage_type, out_channels, in_channels, 2 * stride, true),
    };
}

OobleckDecoderBlockWeights load_decoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride) {
    return {
        load_snake(store, source, prefix + ".layers.0", in_channels),
        load_wn_conv_transpose1d(store, source, prefix + ".layers.1", storage_type, in_channels, out_channels, 2 * stride, true),
        load_residual_unit(store, source, prefix + ".layers.2", storage_type, out_channels),
        load_residual_unit(store, source, prefix + ".layers.3", storage_type, out_channels),
        load_residual_unit(store, source, prefix + ".layers.4", storage_type, out_channels),
    };
}

OobleckAudioVaeWeights load_weights(
    const assets::TensorSource & source,
    const OobleckAudioVaeConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const OobleckAudioVaeRuntimeOptions & options) {
    const auto total_start = Clock::now();
    const auto channels = channel_plan(config);
    OobleckAudioVaeWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.oobleck_audio_vae.weights",
        options.weight_context_bytes);
    const auto bind_start = Clock::now();
    weights.encoder_in_conv = load_wn_conv1d(
        *weights.store,
        source,
        join_name(config.encoder_prefix, "layers.0"),
        options.weight_storage_type,
        channels.front(),
        config.audio_channels,
        7,
        true);
    weights.encoder_blocks.reserve(config.strides.size());
    for (size_t i = 0; i < config.strides.size(); ++i) {
        weights.encoder_blocks.push_back(load_encoder_block(
            *weights.store,
            source,
            join_name(config.encoder_prefix, "layers." + std::to_string(i + 1)),
            options.weight_storage_type,
            channels[i],
            channels[i + 1],
            config.strides[i]));
    }
    weights.encoder_final_snake = load_snake(
        *weights.store,
        source,
        join_name(config.encoder_prefix, "layers." + std::to_string(config.strides.size() + 1)),
        channels.back());
    weights.encoder_out_conv = load_wn_conv1d(
        *weights.store,
        source,
        join_name(config.encoder_prefix, "layers." + std::to_string(config.strides.size() + 2)),
        options.weight_storage_type,
        config.encoder_latent_dim,
        channels.back(),
        3,
        true);
    weights.decoder_in_conv = load_wn_conv1d(
        *weights.store,
        source,
        join_name(config.decoder_prefix, "layers.0"),
        options.weight_storage_type,
        channels.back(),
        config.decoder_latent_dim,
        7,
        true);
    weights.decoder_blocks.reserve(config.strides.size());
    for (int64_t block = static_cast<int64_t>(config.strides.size()); block > 0; --block) {
        const int64_t layer_index = static_cast<int64_t>(config.strides.size()) - block + 1;
        weights.decoder_blocks.push_back(load_decoder_block(
            *weights.store,
            source,
            join_name(config.decoder_prefix, "layers." + std::to_string(layer_index)),
            options.weight_storage_type,
            channels[static_cast<size_t>(block)],
            channels[static_cast<size_t>(block - 1)],
            config.strides[static_cast<size_t>(block - 1)]));
    }
    weights.decoder_final_snake = load_snake(
        *weights.store,
        source,
        join_name(config.decoder_prefix, "layers." + std::to_string(config.strides.size() + 1)),
        channels.front());
    weights.decoder_out_conv = load_wn_conv1d(
        *weights.store,
        source,
        join_name(config.decoder_prefix, "layers." + std::to_string(config.strides.size() + 2)),
        options.weight_storage_type,
        config.audio_channels,
        channels.front(),
        7,
        false);
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.weights_bind_ms", engine::debug::elapsed_ms(bind_start));
    const auto upload_start = Clock::now();
    weights.store->upload();
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.weights_upload_ms", engine::debug::elapsed_ms(upload_start));
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.weights_total_ms", engine::debug::elapsed_ms(total_start));
    return weights;
}

core::TensorValue snake_beta(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::SnakeBeta1dWeights & weights,
    int64_t channels,
    bool logscale) {
    return modules::SnakeBeta1dModule({channels, logscale}).build(ctx, input, weights);
}

core::TensorValue residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const OobleckResidualUnitWeights & weights,
    int64_t channels,
    int64_t dilation,
    bool logscale) {
    auto hidden = snake_beta(ctx, input, weights.snake_0, channels, logscale);
    hidden = modules::Conv1dModule({channels, channels, 7, 1, static_cast<int>(dilation * 3), static_cast<int>(dilation), true})
                 .build(ctx, hidden, weights.conv_1);
    hidden = snake_beta(ctx, hidden, weights.snake_2, channels, logscale);
    hidden = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true}).build(ctx, hidden, weights.conv_3);
    return modules::AddModule{}.build(ctx, input, hidden);
}

core::TensorValue encoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const OobleckEncoderBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride,
    bool logscale) {
    auto hidden = residual_unit(ctx, input, weights.residual_0, in_channels, 1, logscale);
    hidden = residual_unit(ctx, hidden, weights.residual_1, in_channels, 3, logscale);
    hidden = residual_unit(ctx, hidden, weights.residual_2, in_channels, 9, logscale);
    hidden = snake_beta(ctx, hidden, weights.snake_3, in_channels, logscale);
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        2 * stride,
        static_cast<int>(stride),
        static_cast<int>(std::ceil(static_cast<float>(stride) / 2.0F)),
        1,
        true})
        .build(ctx, hidden, weights.downsample);
}

core::TensorValue decoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const OobleckDecoderBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride,
    bool logscale) {
    auto hidden = snake_beta(ctx, input, weights.snake_0, in_channels, logscale);
    hidden = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        2 * stride,
        static_cast<int>(stride),
        0,
        1,
        true})
        .build(ctx, hidden, weights.upsample);
    hidden = modules::SliceModule({
        2,
        static_cast<int64_t>(std::ceil(static_cast<float>(stride) / 2.0F)),
        input.shape.dims[2] * stride + stride - 2 * static_cast<int64_t>(std::ceil(static_cast<float>(stride) / 2.0F))})
        .build(ctx, hidden);
    hidden = residual_unit(ctx, hidden, weights.residual_2, out_channels, 1, logscale);
    hidden = residual_unit(ctx, hidden, weights.residual_3, out_channels, 3, logscale);
    hidden = residual_unit(ctx, hidden, weights.residual_4, out_channels, 9, logscale);
    return hidden;
}

}  // namespace

struct OobleckAudioVaeRuntime::Impl {
    class EncodeGraph;
    class DecodeGraph;

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        OobleckAudioVaeConfig config,
        OobleckAudioVaeRuntimeOptions options)
        : source(std::move(source)),
          execution(&execution),
          config(std::move(config)),
          options(options) {
        if (!this->source) {
            throw std::runtime_error("Oobleck audio VAE runtime requires tensor source");
        }
        validate_config(this->config);
    }

    const OobleckAudioVaeWeights & require_weights() {
        if (!weights) {
            const auto start = Clock::now();
            weights = std::make_unique<OobleckAudioVaeWeights>(load_weights(
                *source,
                config,
                execution->backend(),
                execution->backend_type(),
                options));
            source->release_storage();
            engine::debug::timing_log_scalar("framework.oobleck_audio_vae.require_weights_ms", engine::debug::elapsed_ms(start));
        }
        return *weights;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext * execution = nullptr;
    OobleckAudioVaeConfig config;
    OobleckAudioVaeRuntimeOptions options;
    std::unique_ptr<OobleckAudioVaeWeights> weights;
    std::unique_ptr<EncodeGraph> encode_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
};

class OobleckAudioVaeRuntime::Impl::EncodeGraph {
public:
    EncodeGraph(core::ExecutionContext & execution, const OobleckAudioVaeConfig & config, const OobleckAudioVaeRuntimeOptions & options, const OobleckAudioVaeWeights & weights, int64_t frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          frames_(frames) {
        if (backend_ == nullptr || frames_ <= 0) {
            throw std::runtime_error("Oobleck audio VAE encode graph initialization failed");
        }
        build();
    }

    ~EncodeGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t frames) const noexcept {
        return frames == frames_;
    }

    std::vector<float> run(const std::vector<float> & planar_audio) const {
        if (static_cast<int64_t>(planar_audio.size()) != config_.audio_channels * frames_) {
            throw std::runtime_error("Oobleck audio VAE encode input shape mismatch");
        }
        const auto total_start = Clock::now();
        const auto upload_start = Clock::now();
        core::write_tensor_f32(input_, planar_audio);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.input_upload_ms", engine::debug::elapsed_ms(upload_start));
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "framework.oobleck_audio_vae.encode");
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.graph_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Oobleck audio VAE encode graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.output_read_ms", engine::debug::elapsed_ms(read_start));
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    void build() {
        const auto start = Clock::now();
        ggml_init_params params{options_.graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Oobleck audio VAE encode ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "framework.oobleck_audio_vae.encode.inputs", backend_type_};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config_.audio_channels, frames_}));
        ggml_set_input(input_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "framework.oobleck_audio_vae.encode", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Oobleck audio VAE encode backend buffer allocation failed");
        }
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.graph_frames", frames_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode.graph_build_ms", engine::debug::elapsed_ms(start));
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        const auto channels = channel_plan(config_);
        auto hidden = modules::Conv1dModule({config_.audio_channels, channels.front(), 7, 1, 3, 1, true})
                          .build(ctx, input_, weights_.encoder_in_conv);
        for (size_t i = 0; i < weights_.encoder_blocks.size(); ++i) {
            hidden = encoder_block(ctx, hidden, weights_.encoder_blocks[i], channels[i], channels[i + 1], config_.strides[i], config_.snake_logscale);
        }
        hidden = snake_beta(ctx, hidden, weights_.encoder_final_snake, channels.back(), config_.snake_logscale);
        return modules::Conv1dModule({channels.back(), config_.encoder_latent_dim, 3, 1, 1, 1, true})
            .build(ctx, hidden, weights_.encoder_out_conv);
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    OobleckAudioVaeConfig config_;
    OobleckAudioVaeRuntimeOptions options_;
    const OobleckAudioVaeWeights & weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class OobleckAudioVaeRuntime::Impl::DecodeGraph {
public:
    DecodeGraph(core::ExecutionContext & execution, const OobleckAudioVaeConfig & config, const OobleckAudioVaeRuntimeOptions & options, const OobleckAudioVaeWeights & weights, int64_t batch, int64_t latent_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          batch_(batch),
          latent_frames_(latent_frames) {
        if (backend_ == nullptr || batch_ <= 0 || latent_frames_ <= 0) {
            throw std::runtime_error("Oobleck audio VAE decode graph initialization failed");
        }
        build();
    }

    ~DecodeGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t latent_frames) const noexcept {
        return batch == batch_ && latent_frames == latent_frames_;
    }

    std::vector<float> run(const std::vector<float> & latents) const {
        if (static_cast<int64_t>(latents.size()) != batch_ * config_.decoder_latent_dim * latent_frames_) {
            throw std::runtime_error("Oobleck audio VAE latent shape mismatch");
        }
        const auto total_start = Clock::now();
        const auto upload_start = Clock::now();
        core::write_tensor_f32(input_, latents);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.input_upload_ms", engine::debug::elapsed_ms(upload_start));
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "framework.oobleck_audio_vae.decode");
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.graph_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Oobleck audio VAE decode graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.output_read_ms", engine::debug::elapsed_ms(read_start));
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    void build() {
        const auto start = Clock::now();
        ggml_init_params params{options_.graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Oobleck audio VAE decode ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "framework.oobleck_audio_vae.decode.inputs", backend_type_};
        input_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config_.decoder_latent_dim, latent_frames_}));
        ggml_set_input(input_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "framework.oobleck_audio_vae.decode", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        output_frames_ = output.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Oobleck audio VAE decode backend buffer allocation failed");
        }
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.graph_batch", batch_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.graph_latent_frames", latent_frames_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.graph_output_frames", output_frames_);
        engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.graph_build_ms", engine::debug::elapsed_ms(start));
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        const auto channels = channel_plan(config_);
        auto hidden = modules::Conv1dModule({config_.decoder_latent_dim, channels.back(), 7, 1, 3, 1, true})
                          .build(ctx, input_, weights_.decoder_in_conv);
        for (size_t i = 0; i < weights_.decoder_blocks.size(); ++i) {
            const int64_t block = static_cast<int64_t>(config_.strides.size() - i);
            hidden = decoder_block(
                ctx,
                hidden,
                weights_.decoder_blocks[i],
                channels[static_cast<size_t>(block)],
                channels[static_cast<size_t>(block - 1)],
                config_.strides[static_cast<size_t>(block - 1)],
                config_.snake_logscale);
        }
        hidden = snake_beta(ctx, hidden, weights_.decoder_final_snake, channels.front(), config_.snake_logscale);
        auto output = modules::Conv1dModule({channels.front(), config_.audio_channels, 7, 1, 3, 1, false})
                          .build(ctx, hidden, weights_.decoder_out_conv);
        if (config_.final_tanh) {
            output = modules::TanhModule{}.build(ctx, output);
        }
        return output;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    OobleckAudioVaeConfig config_;
    OobleckAudioVaeRuntimeOptions options_;
    const OobleckAudioVaeWeights & weights_;
    int64_t batch_ = 0;
    int64_t latent_frames_ = 0;
    int64_t output_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

OobleckAudioVaeRuntime::OobleckAudioVaeRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    OobleckAudioVaeConfig config,
    OobleckAudioVaeRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options)) {}

OobleckAudioVaeRuntime::~OobleckAudioVaeRuntime() = default;
OobleckAudioVaeRuntime::OobleckAudioVaeRuntime(OobleckAudioVaeRuntime &&) noexcept = default;
OobleckAudioVaeRuntime & OobleckAudioVaeRuntime::operator=(OobleckAudioVaeRuntime &&) noexcept = default;

void OobleckAudioVaeRuntime::prepare_encode(int64_t frames) {
    const auto & weights = impl_->require_weights();
    if (!impl_->encode_graph || !impl_->encode_graph->matches(frames)) {
        impl_->encode_graph = std::make_unique<Impl::EncodeGraph>(*impl_->execution, impl_->config, impl_->options, weights, frames);
    }
}

void OobleckAudioVaeRuntime::prepare_decode(int64_t batch, int64_t latent_frames) {
    const auto & weights = impl_->require_weights();
    if (!impl_->decode_graph || !impl_->decode_graph->matches(batch, latent_frames)) {
        impl_->decode_graph = std::make_unique<Impl::DecodeGraph>(*impl_->execution, impl_->config, impl_->options, weights, batch, latent_frames);
    }
}

std::vector<float> OobleckAudioVaeRuntime::encode_planar(const std::vector<float> & planar_audio, int64_t frames) {
    const auto start = Clock::now();
    prepare_encode(frames);
    auto out = impl_->encode_graph->run(planar_audio);
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.encode_planar_ms", engine::debug::elapsed_ms(start));
    return out;
}

std::vector<float> OobleckAudioVaeRuntime::decode_planar(const std::vector<float> & latents, int64_t batch, int64_t latent_frames) {
    const auto start = Clock::now();
    prepare_decode(batch, latent_frames);
    auto out = impl_->decode_graph->run(latents);
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode_planar_ms", engine::debug::elapsed_ms(start));
    return out;
}

std::vector<runtime::AudioBuffer> OobleckAudioVaeRuntime::decode(const std::vector<float> & latents, int64_t batch, int64_t latent_frames) {
    const auto total_start = Clock::now();
    auto planar = decode_planar(latents, batch, latent_frames);
    const int64_t frames = static_cast<int64_t>(planar.size()) / (batch * impl_->config.audio_channels);
    std::vector<runtime::AudioBuffer> out;
    out.reserve(static_cast<size_t>(batch));
    const auto interleave_start = Clock::now();
    for (int64_t b = 0; b < batch; ++b) {
        runtime::AudioBuffer audio;
        audio.sample_rate = static_cast<int>(impl_->config.sample_rate);
        audio.channels = static_cast<int>(impl_->config.audio_channels);
        audio.samples.assign(static_cast<size_t>(frames * impl_->config.audio_channels), 0.0F);
        for (int64_t c = 0; c < impl_->config.audio_channels; ++c) {
            for (int64_t t = 0; t < frames; ++t) {
                audio.samples[static_cast<size_t>(t * impl_->config.audio_channels + c)] =
                    planar[static_cast<size_t>((b * impl_->config.audio_channels + c) * frames + t)];
            }
        }
        out.push_back(std::move(audio));
    }
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.interleave_ms", engine::debug::elapsed_ms(interleave_start));
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.audio_frames", frames);
    engine::debug::timing_log_scalar("framework.oobleck_audio_vae.decode.full_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

void OobleckAudioVaeRuntime::release_runtime_graphs() {
    impl_->encode_graph.reset();
    impl_->decode_graph.reset();
}

}  // namespace engine::codecs

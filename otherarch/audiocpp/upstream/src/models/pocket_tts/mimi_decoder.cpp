#include "engine/models/pocket_tts/mimi_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "graph_common.h"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

struct MimiFullSequenceTrace {
    double latents_pack_ms = 0.0;
    double quantizer_ms = 0.0;
    double encoder_upsample_ms = 0.0;
    double transformer_ms = 0.0;
    double input_projection_ms = 0.0;
    double stage0_ms = 0.0;
    double stage1_ms = 0.0;
    double stage2_ms = 0.0;
    double output_projection_ms = 0.0;
    double tail_graph_ms = 0.0;
};

core::TensorValue build_decoder_resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PocketTTSBackendResidualBlockWeights & weights,
    int64_t channels,
    int64_t hidden_channels);

struct StreamingConv1dState {
    int64_t channels = 0;
    int64_t history_frames = 0;
    bool first = true;
    std::vector<float> previous;
    std::vector<float> scratch;
};

struct StreamingConvTranspose1dState {
    int64_t channels = 0;
    int64_t partial_frames = 0;
    std::vector<float> partial;
};

struct TransformerState {
    int64_t current_end = 0;
};

struct DecoderState {
    StreamingConvTranspose1dState encoder_rate_upsample;
    TransformerState transformer;
    StreamingConv1dState input_projection;
    std::vector<StreamingConvTranspose1dState> stage_upsamples;
    std::vector<std::vector<StreamingConv1dState>> stage_residual_convs;
    StreamingConv1dState output_projection;
};

class DepthwiseConvTranspose1dRuntime;
class Conv1dRuntime;
class ConvTranspose1dRuntime;
class MimiTransformerRuntime;

void release_partial_graph_runtime(
    ggml_gallocr_t & gallocr,
    ggml_backend_buffer_t & params_buffer,
    ggml_context *& context) {
    if (gallocr != nullptr) {
        ggml_gallocr_free(gallocr);
        gallocr = nullptr;
    }
    if (params_buffer != nullptr) {
        ggml_backend_buffer_free(params_buffer);
        params_buffer = nullptr;
    }
    if (context != nullptr) {
        ggml_free(context);
        context = nullptr;
    }
}

struct TransformerRunResult {
    std::vector<float> output_bct;
};

std::vector<float> tail_bct(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t keep_frames) {
    if (keep_frames <= 0) {
        return {};
    }
    std::vector<float> output(static_cast<size_t>(channels * keep_frames));
    for (int64_t channel = 0; channel < channels; ++channel) {
        const auto * src = values.data() + channel * frames + (frames - keep_frames);
        auto * dst = output.data() + channel * keep_frames;
        std::copy(src, src + keep_frames, dst);
    }
    return output;
}

std::vector<float> head_bct(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t keep_frames) {
    if (keep_frames <= 0) {
        return {};
    }
    std::vector<float> output(static_cast<size_t>(channels * keep_frames));
    for (int64_t channel = 0; channel < channels; ++channel) {
        const auto * src = values.data() + channel * frames;
        auto * dst = output.data() + channel * keep_frames;
        std::copy(src, src + keep_frames, dst);
    }
    return output;
}

std::vector<float> slice_bct(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t start_frame,
    int64_t slice_frames) {
    if (slice_frames <= 0) {
        return {};
    }
    std::vector<float> output(static_cast<size_t>(channels * slice_frames));
    for (int64_t channel = 0; channel < channels; ++channel) {
        const auto * src = values.data() + channel * frames + start_frame;
        auto * dst = output.data() + channel * slice_frames;
        std::copy(src, src + slice_frames, dst);
    }
    return output;
}

std::vector<float> add_bct(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    std::vector<float> output(lhs.size(), 0.0F);
    for (size_t i = 0; i < lhs.size(); ++i) {
        output[i] = lhs[i] + rhs[i];
    }
    return output;
}

std::vector<float> elu(const std::vector<float> & values) {
    std::vector<float> output(values.size(), 0.0F);
    for (size_t i = 0; i < values.size(); ++i) {
        output[i] = values[i] >= 0.0F ? values[i] : std::exp(values[i]) - 1.0F;
    }
    return output;
}

int64_t compute_full_decoder_tail_trim_frames() {
    const int64_t stage0_trim = 12 - 6;
    const int64_t stage1_trim = 10 - 5;
    const int64_t stage2_trim = 8 - 4;
    return stage0_trim * 5 * 4 + stage1_trim * 4 + stage2_trim;
}

std::vector<float> build_transformer_attention_mask(int64_t frames, int64_t cache_steps, int64_t context, int64_t keep_steps) {
    if (frames <= 0 || cache_steps < 0 || keep_steps < 0 || keep_steps > cache_steps) {
        throw std::runtime_error("Mimi transformer attention mask received invalid dimensions");
    }
    std::vector<float> values(static_cast<size_t>(frames * (cache_steps + frames)), -INFINITY);
    const int64_t prefix_start = cache_steps - keep_steps;
    for (int64_t q = 0; q < frames; ++q) {
        const int64_t local_begin = std::max<int64_t>(0, cache_steps + q + 1 - context);
        const int64_t begin = std::max<int64_t>(prefix_start, local_begin);
        for (int64_t k = begin; k < cache_steps + q + 1; ++k) {
            values[static_cast<size_t>(q * (cache_steps + frames) + k)] = 0.0F;
        }
    }
    return values;
}

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const MimiDecoderConfig & config,
    int64_t layer);

class DepthwiseConvTranspose1dRuntime {
public:
    DepthwiseConvTranspose1dRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        const core::TensorValue & weight,
        int64_t frames,
        int64_t channels,
        int64_t kernel_size,
        int stride)
        : backend_(backend), threads_(threads), frames_(frames), channels_(channels), kernel_size_(kernel_size), stride_(stride) {
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize depthwise conv transpose runtime context");
        }

        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "depthwise_convtranspose_runtime", backend_type};
        input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, channels_, frames_}));
        if (core::uses_host_graph_plan(backend_type)) {
            output_ = modules::DepthwiseConvTranspose1dModule({
                channels_,
                kernel_size_,
                // One frame needs no inserted zeros; overlap still uses stride_.
                frames_ == 1 ? 1 : stride_,
                false,
            }).build(ctx, input_, {weight, std::nullopt});
        } else {
            output_ = modules::ConvTranspose1dModule({
                channels_, channels_, kernel_size_, stride_, 0, 1, false,
            }).build(ctx, input_, {weight, std::nullopt});
        }

        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi depthwise conv transpose context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr ||
            !ggml_gallocr_reserve(galloc_, graph_) ||
            !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi depthwise conv transpose graph allocation failed");
        }
        core::write_tensor_f32(input_, std::vector<float>(static_cast<size_t>(channels_ * frames_), 0.0F));
    }

    ~DepthwiseConvTranspose1dRuntime() {
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    std::vector<float> run(const std::vector<float> & input) const {
        if (static_cast<int64_t>(input.size()) != channels_ * frames_) {
            throw std::runtime_error("DepthwiseConvTranspose1dRuntime input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        engine::core::compute_backend_graph(backend_, graph_);
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t frames_ = 0;
    int64_t channels_ = 0;
    int64_t kernel_size_ = 0;
    int stride_ = 1;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    core::TensorValue output_;
};

class Conv1dRuntime {
public:
    Conv1dRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        const core::TensorValue & weight,
        const std::optional<core::TensorValue> & bias,
        int64_t in_channels,
        int64_t frames,
        int64_t out_channels,
        int64_t kernel_size,
        int stride,
        int dilation)
        : backend_(backend),
          threads_(threads),
          in_channels_(in_channels),
          frames_(frames),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          stride_(stride),
          dilation_(dilation) {
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize conv1d runtime context");
        }

        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "mimi_conv1d_runtime", backend_type};
        input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, in_channels_, frames_}));
        output_ = modules::Conv1dModule({
            in_channels_,
            out_channels_,
            kernel_size_,
            stride_,
            0,
            dilation_,
            bias.has_value(),
        }).build(ctx, input_, modules::Conv1dWeights{weight, bias});

        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi conv1d context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr ||
            !ggml_gallocr_reserve(galloc_, graph_) ||
            !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi conv1d graph allocation failed");
        }
        core::write_tensor_f32(input_, std::vector<float>(static_cast<size_t>(in_channels_ * frames_), 0.0F));
    }

    ~Conv1dRuntime() {
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    std::vector<float> run(const std::vector<float> & input) const {
        if (static_cast<int64_t>(input.size()) != in_channels_ * frames_) {
            throw std::runtime_error("Conv1dRuntime input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        engine::core::compute_backend_graph(backend_, graph_);
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t in_channels_ = 0;
    int64_t frames_ = 0;
    int64_t out_channels_ = 0;
    int64_t kernel_size_ = 0;
    int stride_ = 1;
    int dilation_ = 1;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    core::TensorValue output_;
};

class ConvTranspose1dRuntime {
public:
    ConvTranspose1dRuntime(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        const core::TensorValue & weight,
        const std::optional<core::TensorValue> & bias,
        int64_t in_channels,
        int64_t frames,
        int64_t out_channels,
        int64_t kernel_size,
        int stride)
        : backend_(backend),
          threads_(threads),
          in_channels_(in_channels),
          frames_(frames),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          stride_(stride) {
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize convtranspose1d runtime context");
        }

        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "mimi_convtranspose1d_runtime", backend_type};
        input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, in_channels_, frames_}));
        output_ = modules::ConvTranspose1dModule({
            in_channels_,
            out_channels_,
            kernel_size_,
            stride_,
            0,
            1,
            bias.has_value(),
        }).build(ctx, input_, modules::ConvTranspose1dWeights{weight, bias});

        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi conv transpose context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr ||
            !ggml_gallocr_reserve(galloc_, graph_) ||
            !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi conv transpose graph allocation failed");
        }
        core::write_tensor_f32(input_, std::vector<float>(static_cast<size_t>(in_channels_ * frames_), 0.0F));
    }

    ~ConvTranspose1dRuntime() {
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    std::vector<float> run(const std::vector<float> & input) const {
        if (static_cast<int64_t>(input.size()) != in_channels_ * frames_) {
            throw std::runtime_error("ConvTranspose1dRuntime input size mismatch");
        }
        core::write_tensor_f32(input_, input);
        engine::core::compute_backend_graph(backend_, graph_);
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t in_channels_ = 0;
    int64_t frames_ = 0;
    int64_t out_channels_ = 0;
    int64_t kernel_size_ = 0;
    int stride_ = 1;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_;
    core::TensorValue output_;
};

class MimiFullDecoderRuntime {
public:
    MimiFullDecoderRuntime(
        ggml_backend_t backend,
        int threads,
        size_t graph_context_bytes,
        const MimiDecoderConfig & config,
        const models::pocket_tts::PocketTTSBackendWeights & weights,
        int64_t steps)
        : backend_(backend),
          threads_(threads),
          config_(config),
          steps_(steps),
          transformed_frames_(steps * config.encoder_upsample_stride),
          cache_steps_(std::max<int64_t>(0, 250 - transformed_frames_)),
          head_dim_(config.hidden_size / config.num_heads) {
        if (steps_ <= 0 || transformed_frames_ <= 0) {
            throw std::runtime_error("Mimi full decoder runtime requires positive dimensions");
        }
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize Mimi full decoder runtime context");
        }

        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "mimi_full_decoder_runtime", weights.backend_type};
        const auto & decoder = weights.mimi_decoder;
        latents_bct_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.latent_size, steps_}));

        auto x = modules::Conv1dModule({
            config_.latent_size,
            config_.hidden_size,
            1,
            1,
            0,
            1,
            false,
        }).build(
            ctx,
            latents_bct_,
            modules::Conv1dWeights{decoder.quantizer_output_proj_weight, std::nullopt});

        x = modules::ConvTranspose1dModule({
            config_.hidden_size,
            config_.hidden_size,
            config_.encoder_upsample_stride * 2,
            static_cast<int>(config_.encoder_upsample_stride),
            0,
            1,
            false,
        }).build(
            ctx,
            x,
            modules::ConvTranspose1dWeights{decoder.encoder_upsample_weight, std::nullopt});
        x = modules::SliceModule({2, 0, transformed_frames_}).build(ctx, x);

        positions_ = core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({transformed_frames_}));
        attention_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, transformed_frames_, cache_steps_ + transformed_frames_}));

        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        for (int64_t layer = 0; layer < config_.transformer_layers; ++layer) {
            std::optional<core::TensorValue> prefix_key = std::nullopt;
            std::optional<core::TensorValue> prefix_value = std::nullopt;
            if (cache_steps_ > 0) {
                prefix_key = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_}));
                prefix_value = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_}));
                zero_prefix_keys_.push_back(*prefix_key);
                zero_prefix_values_.push_back(*prefix_value);
            }

            auto outputs = modules::StreamingTransformerEncoderBlockModule({
                config_.hidden_size,
                config_.num_heads,
                config_.intermediate_size,
                1.0e-5F,
                false,
            }).build(
                ctx,
                x,
                positions_,
                make_transformer_layer_weights(ctx, weights, config_, layer),
                prefix_key,
                prefix_value,
                attention_mask_);
            x = outputs.output;
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);

        x = modules::StreamingConv1dModule({
            config_.hidden_size,
            config_.hidden_size,
            7,
            1,
            1,
            true,
            modules::StreamingPadMode::Constant,
        }).build(
            ctx,
            x,
            decoder.input_projection);

        x = modules::EluModule().build(ctx, x);
        x = modules::ConvTranspose1dModule({
            config_.hidden_size,
            256,
            12,
            6,
            0,
            1,
            true,
        }).build(
            ctx,
            x,
            decoder.stage0_upsample);
        x = build_decoder_resblock(
            ctx,
            x,
            decoder.stage0_block,
            256,
            128);

        x = modules::EluModule().build(ctx, x);
        x = modules::ConvTranspose1dModule({
            256,
            128,
            10,
            5,
            0,
            1,
            true,
        }).build(
            ctx,
            x,
            decoder.stage1_upsample);
        x = build_decoder_resblock(
            ctx,
            x,
            decoder.stage1_block,
            128,
            64);

        x = modules::EluModule().build(ctx, x);
        x = modules::ConvTranspose1dModule({
            128,
            64,
            8,
            4,
            0,
            1,
            true,
        }).build(
            ctx,
            x,
            decoder.stage2_upsample);
        x = build_decoder_resblock(
            ctx,
            x,
            decoder.stage2_block,
            64,
            32);

        x = modules::EluModule().build(ctx, x);
        auto output_full = modules::StreamingConv1dModule({
            64,
            1,
            3,
            1,
            1,
            true,
            modules::StreamingPadMode::Constant,
        }).build(
            ctx,
            x,
            decoder.output_projection);
        const int64_t trim_frames = compute_full_decoder_tail_trim_frames();
        if (output_full.shape.dims[2] <= trim_frames) {
            throw std::runtime_error("Mimi full decoder runtime trim exceeds output frames");
        }
        output_ = modules::SliceModule({2, 0, output_full.shape.dims[2] - trim_frames}).build(ctx, output_full);

        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi full decoder context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr ||
            !ggml_gallocr_reserve(galloc_, graph_) ||
            !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi full decoder graph allocation failed");
        }
        core::write_tensor_f32(latents_bct_, std::vector<float>(static_cast<size_t>(config_.latent_size * steps_), 0.0F));
        std::vector<int32_t> positions(static_cast<size_t>(transformed_frames_), 0);
        for (int64_t i = 0; i < transformed_frames_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(positions_, positions);
        core::write_tensor_f32(
            attention_mask_,
            build_transformer_attention_mask(transformed_frames_, cache_steps_, 250, 0));
        if (cache_steps_ > 0) {
            const std::vector<float> zero_prefix(static_cast<size_t>(cache_steps_ * config_.num_heads * head_dim_), 0.0F);
            for (size_t layer = 0; layer < zero_prefix_keys_.size(); ++layer) {
                core::write_tensor_f32(zero_prefix_keys_[layer], zero_prefix);
                core::write_tensor_f32(zero_prefix_values_[layer], zero_prefix);
            }
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            const auto plan_started = std::chrono::steady_clock::now();
            plan_ = engine::core::create_backend_graph_plan_if_host(backend_, graph_);
            plan_create_ms_ = engine::debug::elapsed_ms(plan_started);
            if (plan_ == nullptr) {
                throw std::runtime_error("Failed to create Mimi full decoder graph plan");
            }
        }
    }

    ~MimiFullDecoderRuntime() {
        if (plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, plan_);
        }
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    std::vector<float> run(const std::vector<float> & latents_bct) const {
        if (static_cast<int64_t>(latents_bct.size()) != config_.latent_size * steps_) {
            throw std::runtime_error("Mimi full decoder runtime input size mismatch");
        }
        core::write_tensor_f32(latents_bct_, latents_bct);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.single.graph.plan_create_ms", plan_create_ms_);
        engine::core::compute_backend_graph(backend_, graph_, plan_);
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    MimiDecoderConfig config_;
    int64_t steps_ = 0;
    int64_t transformed_frames_ = 0;
    int64_t cache_steps_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
    double plan_create_ms_ = 0.0;
    core::TensorValue latents_bct_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    std::vector<core::TensorValue> zero_prefix_keys_;
    std::vector<core::TensorValue> zero_prefix_values_;
    core::TensorValue output_;
};

class MimiTransformerRuntime {
public:
    MimiTransformerRuntime(
        ggml_backend_t backend,
        int threads,
        size_t graph_context_bytes,
        const models::pocket_tts::PocketTTSBackendWeights & weights,
        const MimiDecoderConfig & config,
        int64_t frames,
        int64_t context)
        : backend_(backend),
          threads_(threads),
          config_(config),
          frames_(frames),
          context_(context),
          cache_steps_(std::max<int64_t>(0, context - frames)),
          head_dim_(config.hidden_size / config.num_heads) {
        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize Mimi transformer runtime context");
        }

        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "mimi_transformer_runtime", weights.backend_type};
        input_bct_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.hidden_size, frames_}));
        positions_buffer_.resize(static_cast<size_t>(frames_));
        positions_ = core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({frames_}));
        attention_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, frames_, cache_steps_ + frames_}));
        attention_masks_.reserve(static_cast<size_t>(cache_steps_ + 1));
        attention_mask_values_.reserve(static_cast<size_t>(cache_steps_ + 1));
        for (int64_t keep_steps = 0; keep_steps <= cache_steps_; ++keep_steps) {
            attention_masks_.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, frames_, cache_steps_ + frames_})));
            attention_mask_values_.push_back(build_transformer_attention_mask(frames_, cache_steps_, context_, keep_steps));
        }

        auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_bct_);
        prefix_keys_.reserve(static_cast<size_t>(config_.transformer_layers));
        prefix_values_.reserve(static_cast<size_t>(config_.transformer_layers));
        work_prefix_keys_.reserve(static_cast<size_t>(config_.transformer_layers));
        work_prefix_values_.reserve(static_cast<size_t>(config_.transformer_layers));
        keys_.reserve(static_cast<size_t>(config_.transformer_layers));
        values_.reserve(static_cast<size_t>(config_.transformer_layers));
        for (int64_t layer = 0; layer < config_.transformer_layers; ++layer) {
            std::optional<core::TensorValue> prefix_key = std::nullopt;
            std::optional<core::TensorValue> prefix_value = std::nullopt;
            if (cache_steps_ > 0) {
                prefix_key = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_}));
                prefix_value = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_}));
                prefix_keys_.push_back(*prefix_key);
                prefix_values_.push_back(*prefix_value);
                work_prefix_keys_.push_back(core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));
                work_prefix_values_.push_back(core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));
                zero_prefix_keys_.push_back(core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));
                zero_prefix_values_.push_back(core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, cache_steps_, config_.num_heads, head_dim_})));
            }

            auto outputs = modules::StreamingTransformerEncoderBlockModule({
                config_.hidden_size,
                config_.num_heads,
                config_.intermediate_size,
                1.0e-5F,
                false,
            }).build(
                ctx,
                x,
                positions_,
                make_transformer_layer_weights(ctx, weights, config_, layer),
                prefix_key,
                prefix_value,
                attention_mask_);
            x = outputs.output;
            keys_.push_back(outputs.key);
            values_.push_back(outputs.value);
        }

        output_bct_ = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        build_transfer_views();

        if (core::is_host_backend(backend_)) {
            params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
            if (params_buffer_ == nullptr) {
                release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
                throw std::runtime_error("Mimi transformer context tensor allocation failed");
            }
        }
        core::set_backend_threads(backend_, threads_);
        graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
        for (const auto & mask : attention_masks_) {
            ggml_build_forward_expand(graph_, mask.tensor);
        }
        for (const auto & tensor : work_prefix_keys_) {
            ggml_build_forward_expand(graph_, tensor.tensor);
        }
        for (const auto & tensor : work_prefix_values_) {
            ggml_build_forward_expand(graph_, tensor.tensor);
        }
        for (const auto & tensor : zero_prefix_keys_) {
            ggml_build_forward_expand(graph_, tensor.tensor);
        }
        for (const auto & tensor : zero_prefix_values_) {
            ggml_build_forward_expand(graph_, tensor.tensor);
        }
        ggml_build_forward_expand(graph_, output_bct_.tensor);
        galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (galloc_ == nullptr ||
            !ggml_gallocr_reserve(galloc_, graph_) ||
            !ggml_gallocr_alloc_graph(galloc_, graph_)) {
            release_partial_graph_runtime(galloc_, params_buffer_, ggml_ctx_);
            throw std::runtime_error("Mimi transformer graph allocation failed");
        }
        if (!core::is_host_backend(backend_)) {
            for (size_t keep = 0; keep < carry_key_sources_.size(); ++keep) {
                for (size_t layer = 0; layer < carry_key_sources_[keep].size(); ++layer) {
                    if (carry_key_sources_[keep][layer] != nullptr) {
                        ggml_backend_view_init(carry_key_sources_[keep][layer]);
                        ggml_backend_view_init(carry_value_sources_[keep][layer]);
                        ggml_backend_view_init(carry_key_destinations_[keep][layer]);
                        ggml_backend_view_init(carry_value_destinations_[keep][layer]);
                    }
                    ggml_backend_view_init(append_key_sources_[keep][layer]);
                    ggml_backend_view_init(append_value_sources_[keep][layer]);
                    ggml_backend_view_init(append_key_destinations_[keep][layer]);
                    ggml_backend_view_init(append_value_destinations_[keep][layer]);
                }
            }
        }
        core::write_tensor_f32(input_bct_, std::vector<float>(static_cast<size_t>(config_.hidden_size * frames_), 0.0F));
        core::write_tensor_i32(positions_, std::vector<int32_t>(static_cast<size_t>(frames_), 0));
        core::write_tensor_f32(attention_mask_, std::vector<float>(static_cast<size_t>(frames_ * (cache_steps_ + frames_)), -INFINITY));
        for (size_t i = 0; i < attention_masks_.size(); ++i) {
            core::write_tensor_f32(attention_masks_[i], attention_mask_values_[i]);
        }
        if (cache_steps_ > 0) {
            const std::vector<float> zero_prefix(static_cast<size_t>(cache_steps_ * config_.num_heads * head_dim_), 0.0F);
            for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                core::write_tensor_f32(prefix_keys_[layer], zero_prefix);
                core::write_tensor_f32(prefix_values_[layer], zero_prefix);
                core::write_tensor_f32(work_prefix_keys_[layer], zero_prefix);
                core::write_tensor_f32(work_prefix_values_[layer], zero_prefix);
                core::write_tensor_f32(zero_prefix_keys_[layer], zero_prefix);
                core::write_tensor_f32(zero_prefix_values_[layer], zero_prefix);
            }
        }
    }

    ~MimiTransformerRuntime() {
        if (galloc_ != nullptr) {
            ggml_gallocr_free(galloc_);
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    void reset_sequence(int64_t current_end) {
        current_end_ = current_end;
        current_mask_keep_steps_ = -1;
        runtime_keep_steps_ = 0;
        sequence_initialized_ = true;

        if (cache_steps_ <= 0) {
            return;
        }

        for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
            ggml_backend_tensor_copy(zero_prefix_keys_[layer].tensor, prefix_keys_[layer].tensor);
            ggml_backend_tensor_copy(zero_prefix_values_[layer].tensor, prefix_values_[layer].tensor);
        }
    }

    TransformerRunResult run(const std::vector<float> & input_bct) {
        if (static_cast<int64_t>(input_bct.size()) != config_.hidden_size * frames_) {
            throw std::runtime_error("MimiTransformerRuntime input size mismatch");
        }
        if (!sequence_initialized_) {
            throw std::runtime_error("MimiTransformerRuntime must be reset before run");
        }

        core::write_tensor_f32(input_bct_, input_bct);
        for (int64_t i = 0; i < frames_; ++i) {
            positions_buffer_[static_cast<size_t>(i)] = static_cast<int32_t>(current_end_ + i);
        }
        core::write_tensor_i32(positions_, positions_buffer_);

        const int64_t keep_steps = runtime_keep_steps_;
        if (keep_steps != current_mask_keep_steps_) {
            ggml_backend_tensor_copy(attention_masks_[static_cast<size_t>(keep_steps)].tensor, attention_mask_.tensor);
            current_mask_keep_steps_ = keep_steps;
        }

        engine::core::compute_backend_graph(backend_, graph_);

        TransformerRunResult result;
        result.output_bct = core::read_tensor_f32(output_bct_.tensor);
        if (cache_steps_ > 0) {
            const int64_t next_keep_steps = std::min<int64_t>(cache_steps_, keep_steps + frames_);
            for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                if (carry_key_sources_[static_cast<size_t>(next_keep_steps)][layer] != nullptr) {
                    ggml_backend_tensor_copy(
                        carry_key_sources_[static_cast<size_t>(next_keep_steps)][layer],
                        carry_key_destinations_[static_cast<size_t>(next_keep_steps)][layer]);
                    ggml_backend_tensor_copy(
                        carry_value_sources_[static_cast<size_t>(next_keep_steps)][layer],
                        carry_value_destinations_[static_cast<size_t>(next_keep_steps)][layer]);
                }
                ggml_backend_tensor_copy(
                    append_key_sources_[static_cast<size_t>(next_keep_steps)][layer],
                    append_key_destinations_[static_cast<size_t>(next_keep_steps)][layer]);
                ggml_backend_tensor_copy(
                    append_value_sources_[static_cast<size_t>(next_keep_steps)][layer],
                    append_value_destinations_[static_cast<size_t>(next_keep_steps)][layer]);
                ggml_backend_tensor_copy(work_prefix_keys_[layer].tensor, prefix_keys_[layer].tensor);
                ggml_backend_tensor_copy(work_prefix_values_[layer].tensor, prefix_values_[layer].tensor);
            }
            runtime_keep_steps_ = next_keep_steps;
        }
        current_end_ += frames_;
        return result;
    }

private:
    void build_transfer_views() {
        if (cache_steps_ <= 0) {
            return;
        }
        const int64_t step_elems = config_.num_heads * head_dim_;
        carry_key_sources_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        carry_value_sources_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        carry_key_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        carry_value_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        append_key_sources_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        append_value_sources_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        append_key_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});
        append_value_destinations_.assign(static_cast<size_t>(cache_steps_ + 1), {});

        for (int64_t next_keep_steps = 1; next_keep_steps <= cache_steps_; ++next_keep_steps) {
            const int64_t append_steps = std::min<int64_t>(frames_, next_keep_steps);
            const int64_t carry_steps = next_keep_steps - append_steps;
            auto & carry_key_src = carry_key_sources_[static_cast<size_t>(next_keep_steps)];
            auto & carry_value_src = carry_value_sources_[static_cast<size_t>(next_keep_steps)];
            auto & carry_key_dst = carry_key_destinations_[static_cast<size_t>(next_keep_steps)];
            auto & carry_value_dst = carry_value_destinations_[static_cast<size_t>(next_keep_steps)];
            auto & append_key_src = append_key_sources_[static_cast<size_t>(next_keep_steps)];
            auto & append_value_src = append_value_sources_[static_cast<size_t>(next_keep_steps)];
            auto & append_key_dst = append_key_destinations_[static_cast<size_t>(next_keep_steps)];
            auto & append_value_dst = append_value_destinations_[static_cast<size_t>(next_keep_steps)];
            carry_key_src.reserve(static_cast<size_t>(config_.transformer_layers));
            carry_value_src.reserve(static_cast<size_t>(config_.transformer_layers));
            carry_key_dst.reserve(static_cast<size_t>(config_.transformer_layers));
            carry_value_dst.reserve(static_cast<size_t>(config_.transformer_layers));
            append_key_src.reserve(static_cast<size_t>(config_.transformer_layers));
            append_value_src.reserve(static_cast<size_t>(config_.transformer_layers));
            append_key_dst.reserve(static_cast<size_t>(config_.transformer_layers));
            append_value_dst.reserve(static_cast<size_t>(config_.transformer_layers));

            const size_t append_elems = static_cast<size_t>(append_steps * step_elems);
            const size_t append_src_offset = static_cast<size_t>((frames_ - append_steps) * step_elems) * sizeof(float);
            const size_t append_dst_offset = static_cast<size_t>((cache_steps_ - append_steps) * step_elems) * sizeof(float);
            const size_t carry_elems = static_cast<size_t>(carry_steps * step_elems);
            const size_t carry_src_offset = static_cast<size_t>((cache_steps_ - carry_steps) * step_elems) * sizeof(float);
            const size_t carry_dst_offset = static_cast<size_t>((cache_steps_ - next_keep_steps) * step_elems) * sizeof(float);

            for (size_t layer = 0; layer < prefix_keys_.size(); ++layer) {
                if (carry_steps > 0) {
                    carry_key_src.push_back(ggml_view_1d(ggml_ctx_, prefix_keys_[layer].tensor, carry_elems, carry_src_offset));
                    carry_value_src.push_back(ggml_view_1d(ggml_ctx_, prefix_values_[layer].tensor, carry_elems, carry_src_offset));
                    carry_key_dst.push_back(ggml_view_1d(ggml_ctx_, work_prefix_keys_[layer].tensor, carry_elems, carry_dst_offset));
                    carry_value_dst.push_back(ggml_view_1d(ggml_ctx_, work_prefix_values_[layer].tensor, carry_elems, carry_dst_offset));
                } else {
                    carry_key_src.push_back(nullptr);
                    carry_value_src.push_back(nullptr);
                    carry_key_dst.push_back(nullptr);
                    carry_value_dst.push_back(nullptr);
                }
                append_key_src.push_back(ggml_view_1d(ggml_ctx_, keys_[layer].tensor, append_elems, append_src_offset));
                append_value_src.push_back(ggml_view_1d(ggml_ctx_, values_[layer].tensor, append_elems, append_src_offset));
                append_key_dst.push_back(ggml_view_1d(ggml_ctx_, work_prefix_keys_[layer].tensor, append_elems, append_dst_offset));
                append_value_dst.push_back(ggml_view_1d(ggml_ctx_, work_prefix_values_[layer].tensor, append_elems, append_dst_offset));
            }
        }
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    MimiDecoderConfig config_;
    int64_t frames_ = 0;
    int64_t context_ = 0;
    int64_t cache_steps_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    core::TensorValue input_bct_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    std::vector<core::TensorValue> attention_masks_;
    std::vector<std::vector<float>> attention_mask_values_;
    std::vector<core::TensorValue> prefix_keys_;
    std::vector<core::TensorValue> prefix_values_;
    std::vector<core::TensorValue> work_prefix_keys_;
    std::vector<core::TensorValue> work_prefix_values_;
    std::vector<core::TensorValue> zero_prefix_keys_;
    std::vector<core::TensorValue> zero_prefix_values_;
    std::vector<core::TensorValue> keys_;
    std::vector<core::TensorValue> values_;
    std::vector<int32_t> positions_buffer_;
    std::vector<std::vector<ggml_tensor *>> carry_key_sources_;
    std::vector<std::vector<ggml_tensor *>> carry_value_sources_;
    std::vector<std::vector<ggml_tensor *>> carry_key_destinations_;
    std::vector<std::vector<ggml_tensor *>> carry_value_destinations_;
    std::vector<std::vector<ggml_tensor *>> append_key_sources_;
    std::vector<std::vector<ggml_tensor *>> append_value_sources_;
    std::vector<std::vector<ggml_tensor *>> append_key_destinations_;
    std::vector<std::vector<ggml_tensor *>> append_value_destinations_;
    core::TensorValue output_bct_;
    int64_t current_end_ = 0;
    int64_t runtime_keep_steps_ = 0;
    int64_t current_mask_keep_steps_ = -1;
    bool sequence_initialized_ = false;
};

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const MimiDecoderConfig & config,
    int64_t layer) {
    return graph_common::make_transformer_block_weights(
        ctx,
        weights.mimi_decoder.transformer_layers.at(static_cast<size_t>(layer)),
        config.hidden_size);
}

core::TensorValue build_decoder_resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PocketTTSBackendResidualBlockWeights & weights,
    int64_t channels,
    int64_t hidden_channels) {
    auto x = modules::EluModule().build(ctx, input);
    x = modules::StreamingConv1dModule({
        channels,
        hidden_channels,
        3,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(
        ctx,
        x,
        modules::StreamingConv1dWeights{
            weights.conv1.weight,
            weights.conv1.bias,
        });
    x = modules::EluModule().build(ctx, x);
    x = modules::StreamingConv1dModule({
        hidden_channels,
        channels,
        1,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(
        ctx,
        x,
        modules::StreamingConv1dWeights{
            weights.conv2.weight,
            weights.conv2.bias,
        });
    return modules::ResidualAddModule().build(ctx, x, input);
}

StreamingConv1dState make_streaming_conv1d_state(int64_t channels, int64_t kernel_size, int stride, int dilation) {
    const int64_t history_frames = (kernel_size - 1) * dilation + 1 - stride;
    StreamingConv1dState state;
    state.channels = channels;
    state.history_frames = std::max<int64_t>(0, history_frames);
    state.first = true;
    state.previous.assign(static_cast<size_t>(state.channels * state.history_frames), 0.0F);
    return state;
}

StreamingConvTranspose1dState make_streaming_convtranspose_state(int64_t channels, int64_t kernel_size, int stride) {
    StreamingConvTranspose1dState state;
    state.channels = channels;
    state.partial_frames = std::max<int64_t>(0, kernel_size - stride);
    state.partial.assign(static_cast<size_t>(state.channels * state.partial_frames), 0.0F);
    return state;
}

DecoderState make_decoder_state(const MimiDecoderConfig & config) {
    DecoderState state;
    state.encoder_rate_upsample = make_streaming_convtranspose_state(
        config.hidden_size,
        config.encoder_upsample_stride * 2,
        static_cast<int>(config.encoder_upsample_stride));
    state.transformer.current_end = 0;
    state.input_projection = make_streaming_conv1d_state(config.hidden_size, 7, 1, 1);
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(256, 12, 6));
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(128, 10, 5));
    state.stage_upsamples.push_back(make_streaming_convtranspose_state(64, 8, 4));
    state.stage_residual_convs = {
        {
            make_streaming_conv1d_state(256, 3, 1, 1),
            make_streaming_conv1d_state(128, 1, 1, 1),
        },
        {
            make_streaming_conv1d_state(128, 3, 1, 1),
            make_streaming_conv1d_state(64, 1, 1, 1),
        },
        {
            make_streaming_conv1d_state(64, 3, 1, 1),
            make_streaming_conv1d_state(32, 1, 1, 1),
        },
    };
    state.output_projection = make_streaming_conv1d_state(64, 3, 1, 1);
    return state;
}

std::vector<float> run_streaming_conv1d_step(
    const Conv1dRuntime & runtime,
    const std::vector<float> & input,
    int64_t in_channels,
    int64_t frames,
    int64_t out_channels,
    int64_t kernel_size,
    int stride,
    int dilation,
    modules::StreamingPadMode pad_mode,
    StreamingConv1dState & state) {
    GGML_UNUSED(out_channels);
    GGML_UNUSED(kernel_size);
    GGML_UNUSED(stride);
    GGML_UNUSED(dilation);
    if (state.history_frames <= 0) {
        state.first = false;
        return runtime.run(input);
    }
    const int64_t total_frames = state.history_frames + frames;
    auto & full_input = state.scratch;
    full_input.resize(static_cast<size_t>(in_channels * total_frames));
    for (int64_t channel = 0; channel < in_channels; ++channel) {
        auto * dst = full_input.data() + channel * total_frames;
        if (pad_mode == modules::StreamingPadMode::Replicate && state.first) {
            const float first_value = input[static_cast<size_t>(channel * frames)];
            std::fill_n(dst, static_cast<size_t>(state.history_frames), first_value);
        } else {
            const auto * src = state.previous.data() + channel * state.history_frames;
            std::copy(src, src + state.history_frames, dst);
        }
        const auto * src = input.data() + channel * frames;
        std::copy(src, src + frames, dst + state.history_frames);
    }
    auto output = runtime.run(full_input);
    state.previous.resize(static_cast<size_t>(in_channels * state.history_frames));
    for (int64_t channel = 0; channel < in_channels; ++channel) {
        const auto * src = full_input.data() + channel * total_frames + (total_frames - state.history_frames);
        auto * dst = state.previous.data() + channel * state.history_frames;
        std::copy(src, src + state.history_frames, dst);
    }
    state.first = false;
    return output;
}

std::vector<float> run_streaming_convtranspose1d_step(
    const ConvTranspose1dRuntime & runtime,
    const std::vector<float> & input,
    int64_t in_channels,
    int64_t frames,
    int64_t out_channels,
    int64_t kernel_size,
    int stride,
    const std::vector<float> & bias_values,
    StreamingConvTranspose1dState & state) {
    GGML_UNUSED(in_channels);
    auto output = runtime.run(input);
    const int64_t raw_frames = frames == 0 ? 0 : (frames - 1) * stride + kernel_size;
    if (state.partial_frames > 0) {
        for (int64_t channel = 0; channel < out_channels; ++channel) {
            for (int64_t t = 0; t < state.partial_frames; ++t) {
                output[static_cast<size_t>(channel * raw_frames + t)] +=
                    state.partial[static_cast<size_t>(channel * state.partial_frames + t)];
            }
        }
        state.partial = tail_bct(output, out_channels, raw_frames, state.partial_frames);
        if (!bias_values.empty()) {
            for (int64_t channel = 0; channel < out_channels; ++channel) {
                const float bias_value = bias_values[static_cast<size_t>(channel)];
                for (int64_t t = 0; t < state.partial_frames; ++t) {
                    state.partial[static_cast<size_t>(channel * state.partial_frames + t)] -= bias_value;
                }
            }
        }
        output = head_bct(output, out_channels, raw_frames, raw_frames - state.partial_frames);
    }
    return output;
}

std::vector<float> run_depthwise_convtranspose1d_step(
    const DepthwiseConvTranspose1dRuntime & runtime,
    const std::vector<float> & input,
    int64_t channels,
    int64_t frames,
    int64_t kernel_size,
    int stride,
    StreamingConvTranspose1dState & state) {
    auto output = runtime.run(input);
    const int64_t raw_frames = frames == 0 ? 0 : (frames - 1) * stride + kernel_size;
    if (state.partial_frames > 0) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t t = 0; t < state.partial_frames; ++t) {
                output[static_cast<size_t>(channel * raw_frames + t)] +=
                    state.partial[static_cast<size_t>(channel * state.partial_frames + t)];
            }
        }
        state.partial = tail_bct(output, channels, raw_frames, state.partial_frames);
        output = head_bct(output, channels, raw_frames, raw_frames - state.partial_frames);
    }
    return output;
}

}  // namespace

struct MimiDecoder::RuntimeCache {
    const models::pocket_tts::PocketTTSAssets * manifest = nullptr;
    ggml_backend_t backend = nullptr;
    int threads = 0;
    size_t conv_graph_context_bytes = 0;
    size_t transformer_graph_context_bytes = 0;
    size_t tail_graph_context_bytes = 0;

    std::unique_ptr<DepthwiseConvTranspose1dRuntime> encoder_rate_upsample_runtime;
    int64_t encoder_rate_upsample_steps = -1;
    std::unique_ptr<Conv1dRuntime> quantizer_runtime;
    int64_t quantizer_steps = -1;
    std::unique_ptr<MimiTransformerRuntime> transformer_runtime;
    int64_t transformer_frames = -1;
    std::unique_ptr<Conv1dRuntime> input_projection_runtime;
    int64_t input_projection_frames = -1;
    std::unique_ptr<ConvTranspose1dRuntime> stage0_upsample_runtime;
    int64_t stage0_upsample_frames = -1;
    std::unique_ptr<Conv1dRuntime> stage0_conv1_runtime;
    std::unique_ptr<Conv1dRuntime> stage0_conv2_runtime;
    std::unique_ptr<ConvTranspose1dRuntime> stage1_upsample_runtime;
    int64_t stage1_upsample_frames = -1;
    std::unique_ptr<Conv1dRuntime> stage1_conv1_runtime;
    std::unique_ptr<Conv1dRuntime> stage1_conv2_runtime;
    std::unique_ptr<ConvTranspose1dRuntime> stage2_upsample_runtime;
    int64_t stage2_upsample_frames = -1;
    std::unique_ptr<Conv1dRuntime> stage2_conv1_runtime;
    std::unique_ptr<Conv1dRuntime> stage2_conv2_runtime;
    std::unique_ptr<Conv1dRuntime> output_projection_runtime;
    int64_t output_projection_frames = -1;
    std::array<int64_t, 3> resblock_conv1_frames = {-1, -1, -1};
    std::array<int64_t, 3> resblock_conv2_frames = {-1, -1, -1};

    std::unique_ptr<Conv1dRuntime> full_quantizer_runtime;
    int64_t full_quantizer_steps = -1;
    std::unique_ptr<DepthwiseConvTranspose1dRuntime> full_encoder_rate_upsample_runtime;
    int64_t full_encoder_steps = -1;
    std::unique_ptr<MimiTransformerRuntime> full_transformer_runtime;
    int64_t full_transformer_frames = -1;
    std::unique_ptr<MimiFullDecoderRuntime> full_decoder_runtime;
    int64_t full_decoder_steps = -1;
    int64_t full_input_runtime_frames = -1;
    int64_t full_stage0_runtime_frames = -1;
    int64_t full_stage1_runtime_frames = -1;
    int64_t full_stage2_runtime_frames = -1;
    int64_t full_output_runtime_frames = -1;
};

struct MimiDecoder::StreamingState {
    DecoderState decoder_state;
    bool transformer_sequence_initialized = false;

    explicit StreamingState(const MimiDecoderConfig & config)
        : decoder_state(make_decoder_state(config)) {}
};

MimiDecoder::MimiDecoder(MimiDecoderConfig config) : config_(std::move(config)) {}

MimiDecoder::~MimiDecoder() {
    clear_runtime_cache();
}

const MimiDecoderConfig & MimiDecoder::config() const noexcept {
    return config_;
}

std::vector<float> MimiDecoder::decode(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & latents,
    int64_t steps,
    size_t conv_graph_context_bytes,
    size_t transformer_graph_context_bytes,
    size_t tail_graph_context_bytes,
    int64_t full_chunk_frames,
    int64_t stage2_chunk_frames,
    bool use_full_sequence_path) const {
    const auto decode_started = std::chrono::steady_clock::now();
    if (steps <= 0) {
        throw std::runtime_error("PocketTTS Mimi decoder requires positive step count");
    }
    if (latents.size() != static_cast<size_t>(steps) * static_cast<size_t>(config_.latent_size)) {
        throw std::runtime_error("PocketTTS Mimi decoder latents must match steps * latent_size");
    }
    auto & runtime_cache = runtime_cache_;
    if (!runtime_cache || runtime_cache->manifest != &manifest || runtime_cache->backend != backend || runtime_cache->threads != threads
        || runtime_cache->conv_graph_context_bytes != conv_graph_context_bytes
        || runtime_cache->transformer_graph_context_bytes != transformer_graph_context_bytes
        || runtime_cache->tail_graph_context_bytes != tail_graph_context_bytes) {
        runtime_cache = std::make_unique<RuntimeCache>();
        runtime_cache->manifest = &manifest;
        runtime_cache->backend = backend;
        runtime_cache->threads = threads;
        runtime_cache->conv_graph_context_bytes = conv_graph_context_bytes;
        runtime_cache->transformer_graph_context_bytes = transformer_graph_context_bytes;
        runtime_cache->tail_graph_context_bytes = tail_graph_context_bytes;
    }
    auto & cache = *runtime_cache;
    const auto & decoder_weights = weights.mimi_decoder;
    const auto & quantizer_weight = decoder_weights.quantizer_output_proj_weight;
    const auto & encoder_upsample_weight = decoder_weights.encoder_upsample_weight;
    const auto & input_projection = decoder_weights.input_projection;
    const auto & stage0_upsample = decoder_weights.stage0_upsample;
    const auto & stage1_upsample = decoder_weights.stage1_upsample;
    const auto & stage2_upsample = decoder_weights.stage2_upsample;
    const auto & output_projection = decoder_weights.output_projection;
    auto state = make_decoder_state(config_);
    auto & quantizer_runtime = cache.quantizer_runtime;
    auto & transformer_runtime = cache.transformer_runtime;
    auto & input_projection_runtime = cache.input_projection_runtime;
    auto & stage0_upsample_runtime = cache.stage0_upsample_runtime;
    auto & stage0_conv1_runtime = cache.stage0_conv1_runtime;
    auto & stage0_conv2_runtime = cache.stage0_conv2_runtime;
    auto & stage1_upsample_runtime = cache.stage1_upsample_runtime;
    auto & stage1_conv1_runtime = cache.stage1_conv1_runtime;
    auto & stage1_conv2_runtime = cache.stage1_conv2_runtime;
    auto & stage2_upsample_runtime = cache.stage2_upsample_runtime;
    auto & stage2_conv1_runtime = cache.stage2_conv1_runtime;
    auto & stage2_conv2_runtime = cache.stage2_conv2_runtime;
    auto & output_projection_runtime = cache.output_projection_runtime;
    auto & resblock_conv1_frames = cache.resblock_conv1_frames;
    auto & resblock_conv2_frames = cache.resblock_conv2_frames;
    std::vector<float> audio;
    audio.reserve(static_cast<size_t>(steps) * 1920);

    auto run_resblock = [&](DecoderState & state_ref,
                            const std::vector<float> & input_bct,
                            int64_t channels,
                            int64_t hidden_channels,
                            int stage_index,
                            std::unique_ptr<Conv1dRuntime> & conv1_runtime,
                            std::unique_ptr<Conv1dRuntime> & conv2_runtime,
                            const PocketTTSBackendResidualBlockWeights & block_weights) {
        const int64_t frames_bct = static_cast<int64_t>(input_bct.size()) / channels;
        const int64_t conv1_needed_frames =
            frames_bct + state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][0].history_frames;
        if (!conv1_runtime || resblock_conv1_frames[static_cast<size_t>(stage_index)] != conv1_needed_frames) {
            conv1_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                block_weights.conv1.weight,
                block_weights.conv1.bias,
                channels,
                conv1_needed_frames,
                hidden_channels,
                3,
                1,
                1);
            resblock_conv1_frames[static_cast<size_t>(stage_index)] = conv1_needed_frames;
        }
        const int64_t conv2_needed_frames =
            frames_bct + state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][1].history_frames;
        if (!conv2_runtime || resblock_conv2_frames[static_cast<size_t>(stage_index)] != conv2_needed_frames) {
            conv2_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                block_weights.conv2.weight,
                block_weights.conv2.bias,
                hidden_channels,
                conv2_needed_frames,
                channels,
                1,
                1,
                1);
            resblock_conv2_frames[static_cast<size_t>(stage_index)] = conv2_needed_frames;
        }
        auto x = elu(input_bct);
        x = run_streaming_conv1d_step(
            *conv1_runtime,
            x,
            channels,
            frames_bct,
            hidden_channels,
            3,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][0]);
        x = elu(x);
        x = run_streaming_conv1d_step(
            *conv2_runtime,
            x,
            hidden_channels,
            frames_bct,
            channels,
            1,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][1]);
        return add_bct(input_bct, x);
    };

    if (use_full_sequence_path) {
        MimiFullSequenceTrace trace;
        std::vector<float> latents_bct(static_cast<size_t>(config_.latent_size * steps), 0.0F);
        trace.latents_pack_ms = engine::debug::measure_ms([&]() {
            for (int64_t step = 0; step < steps; ++step) {
                for (int64_t channel = 0; channel < config_.latent_size; ++channel) {
                    latents_bct[static_cast<size_t>(channel * steps + step)] =
                        latents[static_cast<size_t>(step * config_.latent_size + channel)];
                }
            }
        });

        if (engine::core::uses_host_graph_plan(backend)) {
            if (!cache.full_quantizer_runtime || cache.full_quantizer_steps != steps) {
                cache.full_quantizer_runtime = std::make_unique<Conv1dRuntime>(
                    backend,
                    weights.backend_type,
                    threads,
                    conv_graph_context_bytes,
                    quantizer_weight,
                    std::nullopt,
                    config_.latent_size,
                    steps,
                    config_.hidden_size,
                    1,
                    1,
                    1);
                cache.full_quantizer_steps = steps;
            }
            if (!cache.full_encoder_rate_upsample_runtime || cache.full_encoder_steps != steps) {
                cache.full_encoder_rate_upsample_runtime = std::make_unique<DepthwiseConvTranspose1dRuntime>(
                    backend,
                    weights.backend_type,
                    threads,
                    conv_graph_context_bytes,
                    encoder_upsample_weight,
                    steps,
                    config_.hidden_size,
                    config_.encoder_upsample_stride * 2,
                    static_cast<int>(config_.encoder_upsample_stride));
                cache.full_encoder_steps = steps;
            }
            const int64_t transformed_frames = steps * config_.encoder_upsample_stride;
            if (!cache.full_transformer_runtime || cache.full_transformer_frames != transformed_frames) {
                cache.full_transformer_runtime =
                    std::make_unique<MimiTransformerRuntime>(
                        backend,
                        threads,
                        transformer_graph_context_bytes,
                        weights,
                        config_,
                        transformed_frames,
                        250);
                cache.full_transformer_frames = transformed_frames;
            }

            std::vector<float> transformed_bct(
                static_cast<size_t>(config_.hidden_size * transformed_frames),
                0.0F);
            std::vector<float> x;
            trace.quantizer_ms = engine::debug::measure_ms([&]() {
                x = cache.full_quantizer_runtime->run(latents_bct);
            });
            trace.encoder_upsample_ms = engine::debug::measure_ms([&]() {
                x = run_depthwise_convtranspose1d_step(
                    *cache.full_encoder_rate_upsample_runtime,
                    x,
                    config_.hidden_size,
                    steps,
                    config_.encoder_upsample_stride * 2,
                    static_cast<int>(config_.encoder_upsample_stride),
                    state.encoder_rate_upsample);
            });
            cache.full_transformer_runtime->reset_sequence(state.transformer.current_end);
            trace.transformer_ms = engine::debug::measure_ms([&]() {
                transformed_bct = cache.full_transformer_runtime->run(x).output_bct;
            });

            engine::debug::trace_log_scalar("pocket_tts.mimi.full.tail_mode", "chunked_runtimes");
            auto decoder_state = make_decoder_state(config_);
            const int64_t encoder_frames = transformed_frames;
            const int64_t kDecoderChunkFrames = full_chunk_frames;
            const int64_t kStage2ChunkFrames = stage2_chunk_frames;
            auto & input_runtime_frames = cache.full_input_runtime_frames;
            auto & stage0_runtime_frames = cache.full_stage0_runtime_frames;
            auto & stage1_runtime_frames = cache.full_stage1_runtime_frames;
            auto & stage2_runtime_frames = cache.full_stage2_runtime_frames;
            auto & output_runtime_frames = cache.full_output_runtime_frames;

            for (int64_t start = 0; start < encoder_frames; start += kDecoderChunkFrames) {
                const int64_t chunk_frames = std::min<int64_t>(kDecoderChunkFrames, encoder_frames - start);
                auto chunk = slice_bct(transformed_bct, config_.hidden_size, encoder_frames, start, chunk_frames);

                const int64_t needed_input_frames = chunk_frames + decoder_state.input_projection.history_frames;
                if (!input_projection_runtime || input_runtime_frames != needed_input_frames) {
                    input_projection_runtime = std::make_unique<Conv1dRuntime>(
                        backend,
                        weights.backend_type,
                        threads,
                        conv_graph_context_bytes,
                        input_projection.weight,
                        input_projection.bias,
                        config_.hidden_size,
                        needed_input_frames,
                        config_.hidden_size,
                        7,
                        1,
                        1);
                    input_runtime_frames = needed_input_frames;
                }
                trace.input_projection_ms += engine::debug::measure_ms([&]() {
                    chunk = run_streaming_conv1d_step(
                        *input_projection_runtime,
                        chunk,
                        config_.hidden_size,
                        chunk_frames,
                        config_.hidden_size,
                        7,
                        1,
                        1,
                        modules::StreamingPadMode::Constant,
                        decoder_state.input_projection);
                });

                chunk = elu(chunk);
                if (!stage0_upsample_runtime || stage0_runtime_frames != chunk_frames) {
                    stage0_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                        backend,
                        weights.backend_type,
                        threads,
                        conv_graph_context_bytes,
                        stage0_upsample.weight,
                        stage0_upsample.bias,
                        config_.hidden_size,
                        chunk_frames,
                        256,
                        12,
                        6);
                    stage0_runtime_frames = chunk_frames;
                }
                trace.stage0_ms += engine::debug::measure_ms([&]() {
                    chunk = run_streaming_convtranspose1d_step(
                        *stage0_upsample_runtime,
                        chunk,
                        config_.hidden_size,
                        chunk_frames,
                        256,
                        12,
                        6,
                        decoder_weights.stage0_upsample_bias_values,
                        decoder_state.stage_upsamples[0]);
                    chunk = run_resblock(
                        decoder_state,
                        chunk,
                        256,
                        128,
                        0,
                        stage0_conv1_runtime,
                        stage0_conv2_runtime,
                        decoder_weights.stage0_block);
                });

                const int64_t stage1_frames = static_cast<int64_t>(chunk.size()) / 256;
                chunk = elu(chunk);
                if (!stage1_upsample_runtime || stage1_runtime_frames != stage1_frames) {
                    stage1_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                        backend,
                        weights.backend_type,
                        threads,
                        conv_graph_context_bytes,
                        stage1_upsample.weight,
                        stage1_upsample.bias,
                        256,
                        stage1_frames,
                        128,
                        10,
                        5);
                    stage1_runtime_frames = stage1_frames;
                }
                trace.stage1_ms += engine::debug::measure_ms([&]() {
                    chunk = run_streaming_convtranspose1d_step(
                        *stage1_upsample_runtime,
                        chunk,
                        256,
                        stage1_frames,
                        128,
                        10,
                        5,
                        decoder_weights.stage1_upsample_bias_values,
                        decoder_state.stage_upsamples[1]);
                    chunk = run_resblock(
                        decoder_state,
                        chunk,
                        128,
                        64,
                        1,
                        stage1_conv1_runtime,
                        stage1_conv2_runtime,
                        decoder_weights.stage1_block);
                });

                const int64_t stage2_source_frames = static_cast<int64_t>(chunk.size()) / 128;
                for (int64_t stage2_start = 0; stage2_start < stage2_source_frames; stage2_start += kStage2ChunkFrames) {
                    const int64_t stage2_frames = std::min<int64_t>(kStage2ChunkFrames, stage2_source_frames - stage2_start);
                    auto stage2_input = slice_bct(chunk, 128, stage2_source_frames, stage2_start, stage2_frames);

                    stage2_input = elu(stage2_input);
                    if (!stage2_upsample_runtime || stage2_runtime_frames != stage2_frames) {
                        stage2_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                            backend,
                            weights.backend_type,
                            threads,
                            conv_graph_context_bytes,
                            stage2_upsample.weight,
                            stage2_upsample.bias,
                            128,
                            stage2_frames,
                            64,
                            8,
                            4);
                        stage2_runtime_frames = stage2_frames;
                    }
                    trace.stage2_ms += engine::debug::measure_ms([&]() {
                        stage2_input = run_streaming_convtranspose1d_step(
                            *stage2_upsample_runtime,
                            stage2_input,
                            128,
                            stage2_frames,
                            64,
                            8,
                            4,
                            decoder_weights.stage2_upsample_bias_values,
                            decoder_state.stage_upsamples[2]);
                        stage2_input = run_resblock(
                            decoder_state,
                            stage2_input,
                            64,
                            32,
                            2,
                            stage2_conv1_runtime,
                            stage2_conv2_runtime,
                            decoder_weights.stage2_block);
                    });

                    const int64_t output_frames = static_cast<int64_t>(stage2_input.size()) / 64;
                    stage2_input = elu(stage2_input);
                    const int64_t needed_output_frames = output_frames + decoder_state.output_projection.history_frames;
                    if (!output_projection_runtime || output_runtime_frames != needed_output_frames) {
                        output_projection_runtime = std::make_unique<Conv1dRuntime>(
                            backend,
                            weights.backend_type,
                            threads,
                            conv_graph_context_bytes,
                            output_projection.weight,
                            output_projection.bias,
                            64,
                            needed_output_frames,
                            1,
                            3,
                            1,
                            1);
                        output_runtime_frames = needed_output_frames;
                    }
                    std::vector<float> chunk_audio;
                    trace.output_projection_ms += engine::debug::measure_ms([&]() {
                        chunk_audio = run_streaming_conv1d_step(
                            *output_projection_runtime,
                            stage2_input,
                            64,
                            output_frames,
                            1,
                            3,
                            1,
                            1,
                            modules::StreamingPadMode::Constant,
                            decoder_state.output_projection);
                    });
                    audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
                }
            }

            const double decode_ms = engine::debug::elapsed_ms(decode_started);
            engine::debug::timing_log_scalar("pocket_tts.mimi_decoder_ms", decode_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.latents_pack_ms", trace.latents_pack_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.quantizer_ms", trace.quantizer_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.encoder_upsample_ms", trace.encoder_upsample_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.transformer_ms", trace.transformer_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.input_projection_ms", trace.input_projection_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage0_ms", trace.stage0_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage1_ms", trace.stage1_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage2_ms", trace.stage2_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.output_projection_ms", trace.output_projection_ms);
            engine::debug::timing_log_scalar("pocket_tts.mimi.full.tail.graph.total_ms", trace.tail_graph_ms);
            return audio;
        }

        engine::debug::trace_log_scalar("pocket_tts.mimi.full.tail_mode", "single_graph");
        const bool full_decoder_cache_hit = cache.full_decoder_runtime && cache.full_decoder_steps == steps;
        double full_decoder_build_ms = 0.0;
        if (!full_decoder_cache_hit) {
            const size_t full_decoder_graph_context_bytes =
                conv_graph_context_bytes + transformer_graph_context_bytes + tail_graph_context_bytes;
            full_decoder_build_ms = engine::debug::measure_ms([&]() {
                cache.full_decoder_runtime =
                    std::make_unique<MimiFullDecoderRuntime>(
                        backend,
                        threads,
                        full_decoder_graph_context_bytes,
                        config_,
                        weights,
                        steps);
            });
            cache.full_decoder_steps = steps;
        }
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.single.graph.rebuild_ms", full_decoder_build_ms);
        trace.tail_graph_ms = engine::debug::measure_ms([&]() {
            audio = cache.full_decoder_runtime->run(latents_bct);
        });
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.single.graph.total_ms", trace.tail_graph_ms);

        const double decode_ms = engine::debug::elapsed_ms(decode_started);
        engine::debug::timing_log_scalar("pocket_tts.mimi_decoder_ms", decode_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.latents_pack_ms", trace.latents_pack_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.quantizer_ms", trace.quantizer_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.encoder_upsample_ms", trace.encoder_upsample_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.transformer_ms", trace.transformer_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.input_projection_ms", trace.input_projection_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage0_ms", trace.stage0_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage1_ms", trace.stage1_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.stage2_ms", trace.stage2_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.output_projection_ms", trace.output_projection_ms);
        engine::debug::timing_log_scalar("pocket_tts.mimi.full.tail.graph.total_ms", trace.tail_graph_ms);
        return audio;
    }

    std::vector<float> latent_step(static_cast<size_t>(config_.latent_size), 0.0F);

    bool transformer_sequence_initialized = false;
    for (int64_t step = 0; step < steps; ++step) {
        const size_t offset = static_cast<size_t>(step * config_.latent_size);
        std::copy_n(latents.begin() + static_cast<ptrdiff_t>(offset), static_cast<size_t>(config_.latent_size), latent_step.begin());

        if (!quantizer_runtime || cache.quantizer_steps != 1) {
            quantizer_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                quantizer_weight,
                std::nullopt,
                config_.latent_size,
                1,
                config_.hidden_size,
                1,
                1,
                1);
            cache.quantizer_steps = 1;
        }
        if (!cache.encoder_rate_upsample_runtime || cache.encoder_rate_upsample_steps != 1) {
            cache.encoder_rate_upsample_runtime = std::make_unique<DepthwiseConvTranspose1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                encoder_upsample_weight,
                1,
                config_.hidden_size,
                config_.encoder_upsample_stride * 2,
                static_cast<int>(config_.encoder_upsample_stride));
            cache.encoder_rate_upsample_steps = 1;
        }
        auto & encoder_rate_upsample_runtime = *cache.encoder_rate_upsample_runtime;

        auto x = quantizer_runtime->run(latent_step);
        x = run_depthwise_convtranspose1d_step(
            encoder_rate_upsample_runtime,
            x,
            config_.hidden_size,
            1,
            config_.encoder_upsample_stride * 2,
            static_cast<int>(config_.encoder_upsample_stride),
            state.encoder_rate_upsample);
        const int64_t encoder_frames = static_cast<int64_t>(x.size()) / config_.hidden_size;
        if (!transformer_runtime || cache.transformer_frames != encoder_frames) {
            transformer_runtime = std::make_unique<MimiTransformerRuntime>(
                backend,
                threads,
                transformer_graph_context_bytes,
                weights,
                config_,
                encoder_frames,
                250);
            cache.transformer_frames = encoder_frames;
        }
        if (!transformer_sequence_initialized) {
            transformer_runtime->reset_sequence(state.transformer.current_end);
            transformer_sequence_initialized = true;
        }
        x = transformer_runtime->run(x).output_bct;
        const int64_t needed_input_frames = encoder_frames + state.input_projection.history_frames;
        if (!input_projection_runtime || cache.input_projection_frames != needed_input_frames) {
            input_projection_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                input_projection.weight,
                input_projection.bias,
                config_.hidden_size,
                needed_input_frames,
                config_.hidden_size,
                7,
                1,
                1);
            cache.input_projection_frames = needed_input_frames;
        }
        x = run_streaming_conv1d_step(
            *input_projection_runtime,
            x,
            config_.hidden_size,
            encoder_frames,
            config_.hidden_size,
            7,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state.input_projection);

        x = elu(x);
        if (!stage0_upsample_runtime || cache.stage0_upsample_frames != encoder_frames) {
            stage0_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                stage0_upsample.weight,
                stage0_upsample.bias,
                config_.hidden_size,
                encoder_frames,
                256,
                12,
                6);
            cache.stage0_upsample_frames = encoder_frames;
        }
        x = run_streaming_convtranspose1d_step(
            *stage0_upsample_runtime,
            x,
            config_.hidden_size,
            encoder_frames,
            256,
            12,
            6,
            decoder_weights.stage0_upsample_bias_values,
            state.stage_upsamples[0]);
        x = run_resblock(
            state,
            x,
            256,
            128,
            0,
            stage0_conv1_runtime,
            stage0_conv2_runtime,
            decoder_weights.stage0_block);

        const int64_t stage1_frames = static_cast<int64_t>(x.size()) / 256;
        x = elu(x);
        if (!stage1_upsample_runtime || cache.stage1_upsample_frames != stage1_frames) {
            stage1_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                stage1_upsample.weight,
                stage1_upsample.bias,
                256,
                stage1_frames,
                128,
                10,
                5);
            cache.stage1_upsample_frames = stage1_frames;
        }
        x = run_streaming_convtranspose1d_step(
            *stage1_upsample_runtime,
            x,
            256,
            stage1_frames,
            128,
            10,
            5,
            decoder_weights.stage1_upsample_bias_values,
            state.stage_upsamples[1]);
        x = run_resblock(
            state,
            x,
            128,
            64,
            1,
            stage1_conv1_runtime,
            stage1_conv2_runtime,
            decoder_weights.stage1_block);

        const int64_t stage2_frames = static_cast<int64_t>(x.size()) / 128;
        x = elu(x);
        if (!stage2_upsample_runtime || cache.stage2_upsample_frames != stage2_frames) {
            stage2_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                stage2_upsample.weight,
                stage2_upsample.bias,
                128,
                stage2_frames,
                64,
                8,
                4);
            cache.stage2_upsample_frames = stage2_frames;
        }
        x = run_streaming_convtranspose1d_step(
            *stage2_upsample_runtime,
            x,
            128,
            stage2_frames,
            64,
            8,
            4,
            decoder_weights.stage2_upsample_bias_values,
            state.stage_upsamples[2]);
        x = run_resblock(
            state,
            x,
            64,
            32,
            2,
            stage2_conv1_runtime,
            stage2_conv2_runtime,
            decoder_weights.stage2_block);

        const int64_t output_frames = static_cast<int64_t>(x.size()) / 64;
        x = elu(x);
        const int64_t needed_output_frames = output_frames + state.output_projection.history_frames;
        if (!output_projection_runtime || cache.output_projection_frames != needed_output_frames) {
            output_projection_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                output_projection.weight,
                output_projection.bias,
                64,
                needed_output_frames,
                1,
                3,
                1,
                1);
            cache.output_projection_frames = needed_output_frames;
        }
        x = run_streaming_conv1d_step(
            *output_projection_runtime,
            x,
            64,
            output_frames,
            1,
            3,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state.output_projection);
        audio.insert(audio.end(), x.begin(), x.end());
    }

    const double decode_ms = engine::debug::elapsed_ms(decode_started);
    engine::debug::timing_log_scalar("pocket_tts.mimi_decoder_ms", decode_ms);
    return audio;
}

void MimiDecoder::reset_streaming_state() const {
    streaming_state_ = std::make_unique<StreamingState>(config_);
}

std::vector<float> MimiDecoder::decode_streaming_step(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & latent,
    size_t conv_graph_context_bytes,
    size_t transformer_graph_context_bytes,
    size_t tail_graph_context_bytes) const {
    const auto decode_started = std::chrono::steady_clock::now();
    if (latent.size() != static_cast<size_t>(config_.latent_size)) {
        throw std::runtime_error("PocketTTS Mimi streaming decoder expects one latent step");
    }
    auto & runtime_cache = runtime_cache_;
    if (!runtime_cache || runtime_cache->manifest != &manifest || runtime_cache->backend != backend || runtime_cache->threads != threads
        || runtime_cache->conv_graph_context_bytes != conv_graph_context_bytes
        || runtime_cache->transformer_graph_context_bytes != transformer_graph_context_bytes
        || runtime_cache->tail_graph_context_bytes != tail_graph_context_bytes) {
        runtime_cache = std::make_unique<RuntimeCache>();
        runtime_cache->manifest = &manifest;
        runtime_cache->backend = backend;
        runtime_cache->threads = threads;
        runtime_cache->conv_graph_context_bytes = conv_graph_context_bytes;
        runtime_cache->transformer_graph_context_bytes = transformer_graph_context_bytes;
        runtime_cache->tail_graph_context_bytes = tail_graph_context_bytes;
    }
    if (!streaming_state_) {
        reset_streaming_state();
    }
    auto & cache = *runtime_cache;
    auto & state = streaming_state_->decoder_state;
    const auto & decoder_weights = weights.mimi_decoder;
    const auto & quantizer_weight = decoder_weights.quantizer_output_proj_weight;
    const auto & encoder_upsample_weight = decoder_weights.encoder_upsample_weight;
    const auto & input_projection = decoder_weights.input_projection;
    const auto & stage0_upsample = decoder_weights.stage0_upsample;
    const auto & stage1_upsample = decoder_weights.stage1_upsample;
    const auto & stage2_upsample = decoder_weights.stage2_upsample;
    const auto & output_projection = decoder_weights.output_projection;
    auto & quantizer_runtime = cache.quantizer_runtime;
    auto & transformer_runtime = cache.transformer_runtime;
    auto & input_projection_runtime = cache.input_projection_runtime;
    auto & stage0_upsample_runtime = cache.stage0_upsample_runtime;
    auto & stage0_conv1_runtime = cache.stage0_conv1_runtime;
    auto & stage0_conv2_runtime = cache.stage0_conv2_runtime;
    auto & stage1_upsample_runtime = cache.stage1_upsample_runtime;
    auto & stage1_conv1_runtime = cache.stage1_conv1_runtime;
    auto & stage1_conv2_runtime = cache.stage1_conv2_runtime;
    auto & stage2_upsample_runtime = cache.stage2_upsample_runtime;
    auto & stage2_conv1_runtime = cache.stage2_conv1_runtime;
    auto & stage2_conv2_runtime = cache.stage2_conv2_runtime;
    auto & output_projection_runtime = cache.output_projection_runtime;
    auto & resblock_conv1_frames = cache.resblock_conv1_frames;
    auto & resblock_conv2_frames = cache.resblock_conv2_frames;

    auto run_resblock = [&](DecoderState & state_ref,
                            const std::vector<float> & input_bct,
                            int64_t channels,
                            int64_t hidden_channels,
                            int stage_index,
                            std::unique_ptr<Conv1dRuntime> & conv1_runtime,
                            std::unique_ptr<Conv1dRuntime> & conv2_runtime,
                            const PocketTTSBackendResidualBlockWeights & block_weights) {
        const int64_t frames_bct = static_cast<int64_t>(input_bct.size()) / channels;
        const int64_t conv1_needed_frames =
            frames_bct + state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][0].history_frames;
        if (!conv1_runtime || resblock_conv1_frames[static_cast<size_t>(stage_index)] != conv1_needed_frames) {
            conv1_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                block_weights.conv1.weight,
                block_weights.conv1.bias,
                channels,
                conv1_needed_frames,
                hidden_channels,
                3,
                1,
                1);
            resblock_conv1_frames[static_cast<size_t>(stage_index)] = conv1_needed_frames;
        }
        const int64_t conv2_needed_frames =
            frames_bct + state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][1].history_frames;
        if (!conv2_runtime || resblock_conv2_frames[static_cast<size_t>(stage_index)] != conv2_needed_frames) {
            conv2_runtime = std::make_unique<Conv1dRuntime>(
                backend,
                weights.backend_type,
                threads,
                conv_graph_context_bytes,
                block_weights.conv2.weight,
                block_weights.conv2.bias,
                hidden_channels,
                conv2_needed_frames,
                channels,
                1,
                1,
                1);
            resblock_conv2_frames[static_cast<size_t>(stage_index)] = conv2_needed_frames;
        }
        auto x = elu(input_bct);
        x = run_streaming_conv1d_step(
            *conv1_runtime,
            x,
            channels,
            frames_bct,
            hidden_channels,
            3,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][0]);
        x = elu(x);
        x = run_streaming_conv1d_step(
            *conv2_runtime,
            x,
            hidden_channels,
            frames_bct,
            channels,
            1,
            1,
            1,
            modules::StreamingPadMode::Constant,
            state_ref.stage_residual_convs[static_cast<size_t>(stage_index)][1]);
        return add_bct(input_bct, x);
    };

    if (!quantizer_runtime || cache.quantizer_steps != 1) {
        quantizer_runtime = std::make_unique<Conv1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            quantizer_weight,
            std::nullopt,
            config_.latent_size,
            1,
            config_.hidden_size,
            1,
            1,
            1);
        cache.quantizer_steps = 1;
    }
    if (!cache.encoder_rate_upsample_runtime || cache.encoder_rate_upsample_steps != 1) {
        cache.encoder_rate_upsample_runtime = std::make_unique<DepthwiseConvTranspose1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            encoder_upsample_weight,
            1,
            config_.hidden_size,
            config_.encoder_upsample_stride * 2,
            static_cast<int>(config_.encoder_upsample_stride));
        cache.encoder_rate_upsample_steps = 1;
    }
    auto & encoder_rate_upsample_runtime = *cache.encoder_rate_upsample_runtime;

    auto x = quantizer_runtime->run(latent);
    x = run_depthwise_convtranspose1d_step(
        encoder_rate_upsample_runtime,
        x,
        config_.hidden_size,
        1,
        config_.encoder_upsample_stride * 2,
        static_cast<int>(config_.encoder_upsample_stride),
        state.encoder_rate_upsample);
    const int64_t encoder_frames = static_cast<int64_t>(x.size()) / config_.hidden_size;
    if (!transformer_runtime || cache.transformer_frames != encoder_frames) {
        transformer_runtime = std::make_unique<MimiTransformerRuntime>(
            backend,
            threads,
            transformer_graph_context_bytes,
            weights,
            config_,
            encoder_frames,
            250);
        cache.transformer_frames = encoder_frames;
    }
    if (!streaming_state_->transformer_sequence_initialized) {
        transformer_runtime->reset_sequence(state.transformer.current_end);
        streaming_state_->transformer_sequence_initialized = true;
    }
    x = transformer_runtime->run(x).output_bct;
    const int64_t needed_input_frames = encoder_frames + state.input_projection.history_frames;
    if (!input_projection_runtime || cache.input_projection_frames != needed_input_frames) {
        input_projection_runtime = std::make_unique<Conv1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            input_projection.weight,
            input_projection.bias,
            config_.hidden_size,
            needed_input_frames,
            config_.hidden_size,
            7,
            1,
            1);
        cache.input_projection_frames = needed_input_frames;
    }
    x = run_streaming_conv1d_step(
        *input_projection_runtime,
        x,
        config_.hidden_size,
        encoder_frames,
        config_.hidden_size,
        7,
        1,
        1,
        modules::StreamingPadMode::Constant,
        state.input_projection);

    x = elu(x);
    if (!stage0_upsample_runtime || cache.stage0_upsample_frames != encoder_frames) {
        stage0_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            stage0_upsample.weight,
            stage0_upsample.bias,
            config_.hidden_size,
            encoder_frames,
            256,
            12,
            6);
        cache.stage0_upsample_frames = encoder_frames;
    }
    x = run_streaming_convtranspose1d_step(
        *stage0_upsample_runtime,
        x,
        config_.hidden_size,
        encoder_frames,
        256,
        12,
        6,
        decoder_weights.stage0_upsample_bias_values,
        state.stage_upsamples[0]);
    x = run_resblock(
        state,
        x,
        256,
        128,
        0,
        stage0_conv1_runtime,
        stage0_conv2_runtime,
        decoder_weights.stage0_block);

    const int64_t stage1_frames = static_cast<int64_t>(x.size()) / 256;
    x = elu(x);
    if (!stage1_upsample_runtime || cache.stage1_upsample_frames != stage1_frames) {
        stage1_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            stage1_upsample.weight,
            stage1_upsample.bias,
            256,
            stage1_frames,
            128,
            10,
            5);
        cache.stage1_upsample_frames = stage1_frames;
    }
    x = run_streaming_convtranspose1d_step(
        *stage1_upsample_runtime,
        x,
        256,
        stage1_frames,
        128,
        10,
        5,
        decoder_weights.stage1_upsample_bias_values,
        state.stage_upsamples[1]);
    x = run_resblock(
        state,
        x,
        128,
        64,
        1,
        stage1_conv1_runtime,
        stage1_conv2_runtime,
        decoder_weights.stage1_block);

    const int64_t stage2_frames = static_cast<int64_t>(x.size()) / 128;
    x = elu(x);
    if (!stage2_upsample_runtime || cache.stage2_upsample_frames != stage2_frames) {
        stage2_upsample_runtime = std::make_unique<ConvTranspose1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            stage2_upsample.weight,
            stage2_upsample.bias,
            128,
            stage2_frames,
            64,
            8,
            4);
        cache.stage2_upsample_frames = stage2_frames;
    }
    x = run_streaming_convtranspose1d_step(
        *stage2_upsample_runtime,
        x,
        128,
        stage2_frames,
        64,
        8,
        4,
        decoder_weights.stage2_upsample_bias_values,
        state.stage_upsamples[2]);
    x = run_resblock(
        state,
        x,
        64,
        32,
        2,
        stage2_conv1_runtime,
        stage2_conv2_runtime,
        decoder_weights.stage2_block);

    const int64_t output_frames = static_cast<int64_t>(x.size()) / 64;
    x = elu(x);
    const int64_t needed_output_frames = output_frames + state.output_projection.history_frames;
    if (!output_projection_runtime || cache.output_projection_frames != needed_output_frames) {
        output_projection_runtime = std::make_unique<Conv1dRuntime>(
            backend,
            weights.backend_type,
            threads,
            conv_graph_context_bytes,
            output_projection.weight,
            output_projection.bias,
            64,
            needed_output_frames,
            1,
            3,
            1,
            1);
        cache.output_projection_frames = needed_output_frames;
    }
    x = run_streaming_conv1d_step(
        *output_projection_runtime,
        x,
        64,
        output_frames,
        1,
        3,
        1,
        1,
        modules::StreamingPadMode::Constant,
        state.output_projection);

    engine::debug::timing_log_scalar(
        "pocket_tts.mimi.streaming_decoder_step_ms",
        engine::debug::elapsed_ms(decode_started));
    return x;
}

void MimiDecoder::clear_runtime_cache() const noexcept {
    runtime_cache_.reset();
    streaming_state_.reset();
}

}  // namespace engine::models::pocket_tts

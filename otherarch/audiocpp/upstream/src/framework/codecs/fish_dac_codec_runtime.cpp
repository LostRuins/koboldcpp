#include "engine/framework/codecs/fish_dac_codec_runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/attention/streaming_frame_attention_cache.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformer_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::codecs {
namespace {

namespace binding = engine::modules::binding;

// kcpp: Audio.cpp's GGML carries a private ggml_round_bf16 operation. Preserve
// its numerical behavior with the portable cast round-trip used elsewhere in
// the selective integration instead of extending KoboldCpp's GGML ABI.
ggml_tensor * ggml_round_bf16(ggml_context * ctx, ggml_tensor * input) {
    auto * round_input = input->type == GGML_TYPE_F32
        ? input
        : ggml_cast(ctx, input, GGML_TYPE_F32);
    return ggml_cast(ctx, ggml_cast(ctx, round_input, GGML_TYPE_BF16), GGML_TYPE_F32);
}

constexpr int64_t kCodecDim = 1024;
constexpr int64_t kCodecTransformerHeads = 16;
constexpr int64_t kCodecHeadDim = 64;
constexpr int64_t kCodecIntermediate = 3072;
constexpr int64_t kCodecTransformerLayers = 8;
constexpr float kCodecNormEps = 1.0e-5F;
constexpr float kConvNextNormEps = 1.0e-6F;
constexpr float kCodecRopeTheta = 10000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct CodecStreamingState {
    modules::StreamingFrameAttentionCache attention;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> histories;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask = nullptr;
    int64_t frames = 0;

    core::TensorValue with_history(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t count) {
        if (count == 0) return input;
        auto history = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], count}));
        ggml_set_input(history.tensor);
        ggml_set_output(history.tensor);
        auto full = modules::ConcatModule({2}).build(ctx, history, input);
        auto next = modules::SliceModule({2, full.shape.dims[2] - count, count}).build(ctx, full);
        next = core::ensure_backend_addressable_layout(ctx, next);
        ggml_set_output(next.tensor);
        histories.emplace_back(history.tensor, next.tensor);
        return full;
    }

    void prepare(ggml_backend_t backend) {
        if (frames == 0) {
            attention.zero_inputs(backend);
            for (const auto & history : histories) {
                std::vector<float> zeros(static_cast<size_t>(ggml_nelements(history.first)), 0.0F);
                ggml_backend_tensor_set(history.first, zeros.data(), 0, zeros.size() * sizeof(float));
            }
        }
        const int32_t position = static_cast<int32_t>(frames);
        ggml_backend_tensor_set(positions, &position, 0, sizeof(position));
        std::vector<ggml_fp16_t> values(128, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        std::fill(values.begin() + std::max<int64_t>(0, 127 - frames), values.end(), ggml_fp32_to_fp16(0.0F));
        ggml_backend_tensor_set(mask, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    }

    void expand(ggml_cgraph * graph) {
        attention.set_outputs();
        attention.build_forward_expand(graph);
        for (const auto & history : histories) ggml_build_forward_expand(graph, history.second);
    }

    void commit(ggml_backend_t backend) {
        attention.commit_outputs(backend);
        for (const auto & history : histories) {
            ggml_backend_tensor_copy_async(backend, backend, history.second, history.first);
        }
        ggml_backend_synchronize(backend);
        ++frames;
    }
};

std::vector<float> prepare_codec_mono(
    const runtime::AudioBuffer & audio,
    int target_sample_rate_hz) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != target_sample_rate_hz) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            target_sample_rate_hz);
    }
    return mono;
}

struct CodecTransformerLayerWeights {
    modules::NormWeights attention_norm;
    modules::AttentionWeights attention;
    modules::LayerScaleWeights attention_scale;
    modules::NormWeights ffn_norm;
    modules::DecoderMLPWeights feed_forward;
    modules::LayerScaleWeights ffn_scale;
};

struct CodecTransformerWeights {
    std::vector<CodecTransformerLayerWeights> layers;
    modules::NormWeights norm;
};

struct ResidualUnitWeights {
    modules::Snake1dWeights snake1;
    modules::Conv1dWeights conv1;
    modules::Snake1dWeights snake2;
    modules::Conv1dWeights conv2;
};

struct EncoderBlockWeights {
    ResidualUnitWeights residual1;
    ResidualUnitWeights residual3;
    ResidualUnitWeights residual9;
    modules::Snake1dWeights snake;
    modules::Conv1dWeights conv;
    std::optional<CodecTransformerWeights> transformer;
};

struct DecoderBlockWeights {
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dWeights conv;
    ResidualUnitWeights residual1;
    ResidualUnitWeights residual3;
    ResidualUnitWeights residual9;
};

struct ConvNeXtBlockWeights {
    modules::DepthwiseConv1dWeights dwconv;
    modules::NormWeights norm;
    modules::LinearWeights pwconv1;
    modules::LinearWeights pwconv2;
    modules::LayerScaleWeights gamma;
};

struct QuantizerUnitWeights {
    modules::Conv1dWeights in_proj;
    modules::Conv1dWeights out_proj;
    core::TensorValue codebook;
    core::TensorValue normalized_codebook;
};

struct FishCodecWeights {
    bool encoder_bf16_compute = false;
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights encoder_first;
    std::vector<EncoderBlockWeights> encoder_blocks;
    modules::Snake1dWeights encoder_final_snake;
    modules::Conv1dWeights encoder_final;

    std::vector<std::pair<modules::Conv1dWeights, ConvNeXtBlockWeights>> downsample;
    CodecTransformerWeights pre_module;
    QuantizerUnitWeights semantic_quantizer;
    std::vector<QuantizerUnitWeights> residual_quantizers;
    CodecTransformerWeights post_module;
    std::vector<std::pair<modules::ConvTranspose1dWeights, ConvNeXtBlockWeights>> upsample;

    modules::Conv1dWeights decoder_first;
    std::vector<DecoderBlockWeights> decoder_blocks;
    modules::Snake1dWeights decoder_final_snake;
    modules::Conv1dWeights decoder_final;
};

void validate_config(const FishDacCodecConfig & config) {
    if (config.sample_rate <= 0 || config.frame_length <= 0) {
        throw std::runtime_error("Fish DAC codec requires positive sample rate and frame length");
    }
    if (config.latent_dim != kCodecDim ||
        config.codebook_dim != 8 ||
        config.semantic_codebook_size != 4096 ||
        config.residual_codebook_size != 1024 ||
        config.total_codebooks != config.quantizer_codebooks + 1) {
        throw std::runtime_error("Fish DAC codec config does not match the supported Fish DAC architecture");
    }
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

std::vector<float> normalized_rows(const std::vector<float> & values, int64_t rows, int64_t cols, bool bf16 = false) {
    if (static_cast<int64_t>(values.size()) != rows * cols) {
        throw std::runtime_error("Fish Audio normalized_rows shape mismatch");
    }
    std::vector<float> out(values.size(), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            const float value = values[static_cast<size_t>(row * cols + col)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        float inv;
        if (bf16) {
            const float norm = ggml_bf16_to_fp32(ggml_fp32_to_bf16(static_cast<float>(std::sqrt(sum))));
            inv = 1.0F / std::max(norm, 1.0e-12F);
        } else {
            inv = sum > 0.0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0F;
        }
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            out[index] = values[index] * inv;
        }
    }
    return out;
}

core::TensorValue slice_frames(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t start, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish DAC codec slice_frames requires positive frames");
    }
    return modules::SliceModule({2, start, frames}).build(ctx, input);
}

core::TensorValue zero_prefix_like(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish Audio zero_prefix_like requires positive frames");
    }
    auto first = modules::SliceModule({2, 0, 1}).build(ctx, input);
    if (ctx.backend_type == core::BackendType::Cpu) {
        first = core::ensure_backend_addressable_layout(ctx, first);
    }
    first = core::wrap_tensor(ggml_scale(ctx.ggml, first.tensor, 0.0F), first.shape, GGML_TYPE_F32);
    return modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames})})
        .build(ctx, first);
}

core::TensorValue zero_suffix_like(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish Audio zero_suffix_like requires positive frames");
    }
    auto last = modules::SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
    if (ctx.backend_type == core::BackendType::Cpu) {
        last = core::ensure_backend_addressable_layout(ctx, last);
    }
    last = core::wrap_tensor(ggml_scale(ctx.ggml, last.tensor, 0.0F), last.shape, GGML_TYPE_F32);
    return modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames})})
        .build(ctx, last);
}

int64_t extra_padding_for_conv1d(int64_t frames, int64_t effective_kernel, int64_t stride, int64_t left_pad) {
    const double n_frames = (static_cast<double>(frames - effective_kernel + left_pad) / static_cast<double>(stride)) + 1.0;
    const int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride + (effective_kernel - left_pad);
    return ideal_length - frames;
}

core::TensorValue causal_pad(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t left_pad, int64_t right_pad) {
    if (left_pad < 0) {
        throw std::runtime_error("Fish Audio causal conv requires non-negative left padding");
    }
    if (right_pad < 0) {
        throw std::runtime_error("Fish Audio causal conv requires non-negative right padding");
    }
    if (left_pad == 0 && right_pad == 0) {
        return input;
    }
    auto out = input;
    if (left_pad > 0) {
        out = modules::ConcatModule({2}).build(ctx, zero_prefix_like(ctx, input, left_pad), out);
    }
    if (right_pad > 0) {
        out = modules::ConcatModule({2}).build(ctx, out, zero_suffix_like(ctx, input, right_pad));
    }
    return out;
}

core::TensorValue causal_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int dilation,
    bool use_bias,
    CodecStreamingState * stream = nullptr,
    bool bf16_compute = false) {
    const int64_t effective_kernel = (kernel - 1) * dilation + 1;
    const int64_t left_pad = effective_kernel - stride;
    const int64_t right_pad = extra_padding_for_conv1d(input.shape.dims[2], effective_kernel, stride, left_pad);
    auto padded = stream == nullptr ? causal_pad(ctx, input, left_pad, right_pad)
                                    : stream->with_history(ctx, input, left_pad);
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        stride,
        0,
        dilation,
        use_bias,
        bf16_compute,
    }).build(ctx, padded, weights);
}

core::TensorValue causal_depthwise_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::DepthwiseConv1dWeights & weights,
    int64_t channels,
    int64_t kernel,
    int stride,
    int dilation,
    bool use_bias,
    CodecStreamingState * stream = nullptr) {
    const int64_t effective_kernel = (kernel - 1) * dilation + 1;
    const int64_t left_pad = effective_kernel - stride;
    const int64_t right_pad = extra_padding_for_conv1d(input.shape.dims[2], effective_kernel, stride, left_pad);
    auto padded = stream == nullptr ? causal_pad(ctx, input, left_pad, right_pad)
                                    : stream->with_history(ctx, input, left_pad);
    return modules::DepthwiseConv1dModule({
        channels,
        kernel,
        stride,
        0,
        dilation,
        use_bias,
    }).build(ctx, padded, weights);
}

core::TensorValue causal_conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    bool use_bias,
    CodecStreamingState * stream = nullptr) {
    const int64_t history = stream == nullptr ? 0 : (kernel - 1) / stride;
    auto full_input = stream == nullptr ? input : stream->with_history(ctx, input, history);
    auto out = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel,
        stride,
        0,
        1,
        use_bias,
    }).build(ctx, full_input, weights);
    const int64_t pad = kernel - stride;
    const int64_t padding_right = static_cast<int64_t>(std::ceil(static_cast<double>(pad)));
    const int64_t padding_left = pad - padding_right;
    const int64_t start = padding_left + history * stride;
    return slice_frames(ctx, out, start, out.shape.dims[2] - start - padding_right);
}

core::TensorValue l2_normalize_last(core::ModuleBuildContext & ctx, const core::TensorValue & input, bool bf16 = false) {
    const bool materialize_input = ctx.backend_type == core::BackendType::Metal || ctx.backend_type == core::BackendType::Cpu;
    const auto normalized_input = materialize_input
        ? core::ensure_backend_addressable_layout(ctx, input)
        : input;
    auto squared = modules::MulModule{}.build(ctx, normalized_input, normalized_input);
    auto sum = modules::ReduceSumModule({static_cast<int>(input.shape.rank - 1)}).build(ctx, squared);
    if (bf16) {
        auto * norm = ggml_round_bf16(ctx.ggml, ggml_sqrt(ctx.ggml, sum.tensor));
        norm = ggml_clamp(ctx.ggml, norm, 1.0e-12F, std::numeric_limits<float>::max());
        return core::wrap_tensor(ggml_round_bf16(ctx.ggml,
            ggml_div(ctx.ggml, normalized_input.tensor, norm)), input.shape, GGML_TYPE_F32);
    }
    auto shifted = core::wrap_tensor(ggml_scale_bias(ctx.ggml, sum.tensor, 1.0F, 1.0e-12F), sum.shape, GGML_TYPE_F32);
    auto denom = modules::SqrtModule{}.build(ctx, shifted);
    auto repeated = modules::RepeatModule({normalized_input.shape}).build(ctx, denom);
    return core::wrap_tensor(ggml_div(ctx.ggml, normalized_input.tensor, repeated.tensor), normalized_input.shape, GGML_TYPE_F32);
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::DecoderMLPWeights & weights,
    bool bf16 = false) {
    auto gate = modules::LinearModule({kCodecDim, kCodecIntermediate, false, GGML_PREC_F32})
                    .build(ctx, input, weights.gate_proj);
    if (bf16) gate = core::wrap_tensor(ggml_round_bf16(ctx.ggml, gate.tensor), gate.shape, GGML_TYPE_F32);
    gate = modules::SiluModule{}.build(ctx, gate);
    if (bf16) gate = core::wrap_tensor(ggml_round_bf16(ctx.ggml, gate.tensor), gate.shape, GGML_TYPE_F32);
    auto up = modules::LinearModule({kCodecDim, kCodecIntermediate, false, GGML_PREC_F32})
                  .build(ctx, input, weights.up_proj);
    if (bf16) up = core::wrap_tensor(ggml_round_bf16(ctx.ggml, up.tensor), up.shape, GGML_TYPE_F32);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    if (bf16) hidden = core::wrap_tensor(ggml_round_bf16(ctx.ggml, hidden.tensor), hidden.shape, GGML_TYPE_F32);
    auto out = modules::LinearModule({kCodecIntermediate, kCodecDim, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.down_proj);
    if (bf16) out = core::wrap_tensor(ggml_round_bf16(ctx.ggml, out.tensor), out.shape, GGML_TYPE_F32);
    return out;
}

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kCodecTransformerHeads, kCodecHeadDim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const core::TensorValue & attention_mask,
    CodecStreamingState * stream = nullptr) {
    auto q = modules::TransposeModule({{0, 2, 1, 3}, q_heads.shape.rank}).build(ctx, q_heads);
    auto k = modules::TransposeModule({{0, 2, 1, 3}, k_heads.shape.rank}).build(ctx, k_heads);
    auto v = modules::TransposeModule({{0, 2, 1, 3}, v_heads.shape.rank}).build(ctx, v_heads);
    q = core::wrap_tensor(ggml_cont(ctx.ggml, q.tensor), q.shape, q.type);
    k = core::wrap_tensor(ggml_cont(ctx.ggml, k.tensor), k.shape, k.type);
    v = core::wrap_tensor(ggml_cont(ctx.ggml, v.tensor), v.shape, v.type);
    if (stream != nullptr) {
        const auto shape = core::TensorShape::from_dims({1, kCodecTransformerHeads, 127, kCodecHeadDim});
        auto key = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        ggml_set_input(key.tensor);
        ggml_set_input(value.tensor);
        // Cache destinations must not alias graph temporaries during copy-back.
        ggml_set_output(key.tensor);
        ggml_set_output(value.tensor);
        const auto cached = modules::StreamingFrameAttentionCacheModule({127}).build(ctx, key, value, k, v);
        stream->attention.add_layer(key.tensor, value.tensor, cached.next_key_cache.tensor, cached.next_value_cache.tensor);
        k = cached.key_context;
        v = cached.value_context;
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(kCodecHeadDim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], kCodecHeadDim}),
        GGML_TYPE_F32);
}

core::TensorValue build_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const CodecTransformerLayerWeights & weights,
    CodecStreamingState * stream = nullptr,
    const core::TensorValue * rotary_cos = nullptr,
    const core::TensorValue * rotary_sin = nullptr) {
    const bool bf16 = rotary_cos != nullptr;
    auto x = input;
    if (bf16) {
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::wrap_tensor(ggml_round_bf16(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    }
    auto normed = modules::RMSNormModule({kCodecDim, kCodecNormEps, !bf16, false}).build(ctx, x, weights.attention_norm);
    if (bf16) {
        auto * base = ggml_round_bf16(ctx.ggml, normed.tensor);
        auto * gain = ggml_round_bf16(ctx.ggml, weights.attention_norm.weight->tensor);
        normed = core::wrap_tensor(ggml_round_bf16(ctx.ggml, ggml_mul(ctx.ggml, base, gain)), normed.shape, GGML_TYPE_F32);
    }
    auto q = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.q_weight, std::nullopt});
    auto k = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.k_weight, std::nullopt});
    auto v = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.v_weight, std::nullopt});
    const auto rotate = [&](core::TensorValue value) {
        if (!bf16) return modules::RoPEModule({kCodecHeadDim, GGML_ROPE_TYPE_NORMAL, kCodecRopeTheta})
            .build(ctx, reshape_heads(ctx, value), positions);
        value = core::wrap_tensor(ggml_round_bf16(ctx.ggml, value.tensor), value.shape, GGML_TYPE_F32);
        auto heads = reshape_heads(ctx, value);
        auto pairs = core::reshape_tensor(ctx, heads, core::TensorShape::from_dims(
            {input.shape.dims[0] * input.shape.dims[1], kCodecTransformerHeads, kCodecHeadDim / 2, 2}));
        auto even = modules::SliceModule({3, 0, 1}).build(ctx, pairs);
        auto odd = modules::SliceModule({3, 1, 1}).build(ctx, pairs);
        auto rotated = modules::RopeInterleavedPairsModule{}.build(ctx, even, odd, *rotary_cos, *rotary_sin);
        rotated = core::wrap_tensor(ggml_round_bf16(ctx.ggml, rotated.tensor), rotated.shape, GGML_TYPE_F32);
        return core::reshape_tensor(ctx, rotated, heads.shape);
    };
    q = rotate(q);
    k = rotate(k);
    if (bf16) v = core::wrap_tensor(ggml_round_bf16(ctx.ggml, v.tensor), v.shape, GGML_TYPE_F32);
    v = reshape_heads(ctx, v);
    auto context = attention_from_heads(ctx, q, k, v, attention_mask, stream);
    context = core::ensure_backend_addressable_layout(ctx, context);
    if (bf16) context = core::wrap_tensor(ggml_round_bf16(ctx.ggml, context.tensor), context.shape, GGML_TYPE_F32);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kCodecDim}));
    auto attn = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                    .build(ctx, context, {weights.attention.out_weight, std::nullopt});
    if (bf16) attn = core::wrap_tensor(ggml_round_bf16(ctx.ggml, attn.tensor), attn.shape, GGML_TYPE_F32);
    auto attention_scale = weights.attention_scale;
    if (bf16) attention_scale.scale = core::wrap_tensor(ggml_round_bf16(ctx.ggml, attention_scale.scale.tensor), attention_scale.scale.shape, GGML_TYPE_F32);
    attn = modules::LayerScaleModule{}.build(ctx, attn, attention_scale);
    if (bf16) attn = core::wrap_tensor(ggml_round_bf16(ctx.ggml, attn.tensor), attn.shape, GGML_TYPE_F32);
    auto hidden = modules::AddModule{}.build(ctx, x, attn);
    if (bf16) hidden = core::wrap_tensor(ggml_round_bf16(ctx.ggml, hidden.tensor), hidden.shape, GGML_TYPE_F32);
    auto ffn_in = modules::RMSNormModule({kCodecDim, kCodecNormEps, !bf16, false}).build(ctx, hidden, weights.ffn_norm);
    if (bf16) {
        auto * base = ggml_round_bf16(ctx.ggml, ffn_in.tensor);
        auto * gain = ggml_round_bf16(ctx.ggml, weights.ffn_norm.weight->tensor);
        ffn_in = core::wrap_tensor(ggml_round_bf16(ctx.ggml, ggml_mul(ctx.ggml, base, gain)), ffn_in.shape, GGML_TYPE_F32);
    }
    auto ff = build_mlp(ctx, ffn_in, weights.feed_forward, bf16);
    auto ffn_scale = weights.ffn_scale;
    if (bf16) ffn_scale.scale = core::wrap_tensor(ggml_round_bf16(ctx.ggml, ffn_scale.scale.tensor), ffn_scale.scale.shape, GGML_TYPE_F32);
    ff = modules::LayerScaleModule{}.build(ctx, ff, ffn_scale);
    if (bf16) ff = core::wrap_tensor(ggml_round_bf16(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
    auto out = modules::AddModule{}.build(ctx, hidden, ff);
    if (bf16) out = core::wrap_tensor(ggml_round_bf16(ctx.ggml, out.tensor), out.shape, GGML_TYPE_F32);
    return out;
}

core::TensorValue make_positions(
    core::ModuleBuildContext &,
    core::ConstantTensorCache & constants,
    int64_t frames) {
    std::vector<int32_t> values(static_cast<size_t>(frames));
    for (int64_t i = 0; i < frames; ++i) {
        values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return constants.make_tensor(core::TensorShape::from_dims({frames}), GGML_TYPE_I32, values.data(), values.size() * sizeof(int32_t));
}

core::TensorValue make_causal_mask(
    core::ModuleBuildContext &,
    core::ConstantTensorCache & constants,
    int64_t frames,
    int64_t window_size) {
    std::vector<ggml_fp16_t> values(static_cast<size_t>(frames * frames), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    for (int64_t row = 0; row < frames; ++row) {
        const int64_t begin = window_size > 0 ? std::max<int64_t>(0, row - window_size + 1) : 0;
        for (int64_t col = begin; col <= row; ++col) {
            values[static_cast<size_t>(row * frames + col)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    return constants.make_tensor(core::TensorShape::from_dims({frames, frames}), GGML_TYPE_F16, values.data(), values.size() * sizeof(ggml_fp16_t));
}

core::TensorValue build_window_transformer(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input_bct,
    const CodecTransformerWeights & weights,
    int64_t window_size,
    CodecStreamingState * stream = nullptr,
    bool bf16 = false) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_bct);
    auto positions = stream == nullptr ? make_positions(ctx, constants, x.shape.dims[1])
        : core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    auto mask = stream == nullptr ? make_causal_mask(ctx, constants, x.shape.dims[1], window_size)
        : core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, window_size}));
    if (stream != nullptr) {
        stream->positions = positions.tensor;
        stream->mask = mask.tensor;
        ggml_set_input(positions.tensor);
        ggml_set_input(mask.tensor);
    }
    core::TensorValue rotary_cos, rotary_sin;
    if (bf16) {
        const int64_t frames = x.shape.dims[1];
        std::vector<float> cos(static_cast<size_t>(frames * kCodecHeadDim / 2)), sin(cos.size());
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t d = 0; d < kCodecHeadDim / 2; ++d) {
                const float frequency = 1.0F / std::pow(kCodecRopeTheta, static_cast<float>(2 * d) / kCodecHeadDim);
                const float angle = static_cast<float>(t) * frequency;
                const auto index = static_cast<size_t>(t * (kCodecHeadDim / 2) + d);
                cos[index] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::cos(angle)));
                sin[index] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(angle)));
            }
        }
        const auto shape = core::TensorShape::from_dims({frames, 1, kCodecHeadDim / 2, 1});
        rotary_cos = constants.make_tensor(shape, GGML_TYPE_F32, cos.data(), cos.size() * sizeof(float));
        rotary_sin = constants.make_tensor(shape, GGML_TYPE_F32, sin.data(), sin.size() * sizeof(float));
        const auto expanded = core::TensorShape::from_dims({x.shape.dims[0] * frames, kCodecTransformerHeads, kCodecHeadDim / 2, 1});
        rotary_cos = modules::RepeatModule({expanded}).build(ctx, rotary_cos);
        rotary_sin = modules::RepeatModule({expanded}).build(ctx, rotary_sin);
    }
    for (const auto & layer : weights.layers) {
        x = build_transformer_layer(ctx, x, positions, mask, layer, stream,
            bf16 ? &rotary_cos : nullptr, bf16 ? &rotary_sin : nullptr);
    }
    x = modules::RMSNormModule({kCodecDim, kCodecNormEps, !bf16, false}).build(ctx, x, weights.norm);
    if (bf16) {
        auto * base = ggml_round_bf16(ctx.ggml, x.tensor);
        auto * gain = ggml_round_bf16(ctx.ggml, weights.norm.weight->tensor);
        x = core::wrap_tensor(ggml_round_bf16(ctx.ggml, ggml_mul(ctx.ggml, base, gain)), x.shape, GGML_TYPE_F32);
    }
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

core::TensorValue build_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels,
    int dilation,
    CodecStreamingState * stream = nullptr,
    bool bf16_compute = false) {
    auto y = modules::Snake1dModule({channels}).build(ctx, input, weights.snake1);
    if (bf16_compute) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = causal_conv1d(ctx, y, weights.conv1, channels, channels, 7, 1, dilation, true, stream, bf16_compute);
    y = modules::Snake1dModule({channels}).build(ctx, y, weights.snake2);
    if (bf16_compute) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = causal_conv1d(ctx, y, weights.conv2, channels, channels, 1, 1, 1, true, nullptr, bf16_compute);
    core::TensorValue x = input;
    if (x.shape.dims[2] != y.shape.dims[2]) {
        x = slice_frames(ctx, x, 0, y.shape.dims[2]);
    }
    auto out = modules::AddModule{}.build(ctx, x, y);
    if (bf16_compute) out = core::wrap_tensor(ggml_round_bf16(ctx.ggml, out.tensor), out.shape, GGML_TYPE_F32);
    return out;
}

core::TensorValue build_convnext(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvNeXtBlockWeights & weights,
    int64_t channels,
    CodecStreamingState * stream = nullptr,
    bool bf16 = false) {
    auto depthwise = weights.dwconv;
    if (bf16) depthwise.weight = core::wrap_tensor(ggml_round_bf16(ctx.ggml,
        ggml_cast(ctx.ggml, depthwise.weight.tensor, GGML_TYPE_F32)), depthwise.weight.shape, GGML_TYPE_F32);
    auto y = causal_depthwise_conv1d(ctx, input, depthwise, channels, 7, 1, 1, !bf16, stream);
    if (bf16) {
        auto bias = core::wrap_tensor(ggml_round_bf16(ctx.ggml, weights.dwconv.bias->tensor), weights.dwconv.bias->shape, GGML_TYPE_F32);
        bias = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({1, channels, 1}));
        bias = modules::RepeatModule({y.shape}).build(ctx, bias);
        y = modules::AddModule{}.build(ctx, y, bias);
        y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    }
    y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
    auto norm = weights.norm;
    auto expansion = weights.pwconv1;
    auto projection = weights.pwconv2;
    auto scale = weights.gamma;
    if (bf16) {
        norm.weight = core::wrap_tensor(ggml_round_bf16(ctx.ggml, norm.weight->tensor), norm.weight->shape, GGML_TYPE_F32);
        norm.bias = core::wrap_tensor(ggml_round_bf16(ctx.ggml, norm.bias->tensor), norm.bias->shape, GGML_TYPE_F32);
        expansion.bias = core::wrap_tensor(ggml_round_bf16(ctx.ggml, expansion.bias->tensor), expansion.bias->shape, GGML_TYPE_F32);
        projection.bias = core::wrap_tensor(ggml_round_bf16(ctx.ggml, projection.bias->tensor), projection.bias->shape, GGML_TYPE_F32);
        scale.scale = core::wrap_tensor(ggml_round_bf16(ctx.ggml, scale.scale.tensor), scale.scale.shape, GGML_TYPE_F32);
    }
    y = modules::LayerNormModule({channels, kConvNextNormEps, true, true}).build(ctx, y, norm);
    if (bf16) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = modules::LinearModule({channels, channels * 4, true, GGML_PREC_F32}).build(ctx, y, expansion);
    if (bf16) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
    if (bf16) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = modules::LinearModule({channels * 4, channels, true, GGML_PREC_F32}).build(ctx, y, projection);
    if (bf16) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = modules::LayerScaleModule{}.build(ctx, y, scale);
    if (bf16) y = core::wrap_tensor(ggml_round_bf16(ctx.ggml, y.tensor), y.shape, GGML_TYPE_F32);
    y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
    core::TensorValue x = input;
    if (x.shape.dims[2] != y.shape.dims[2]) {
        x = slice_frames(ctx, x, 0, y.shape.dims[2]);
    }
    auto out = modules::AddModule{}.build(ctx, x, y);
    if (bf16) out = core::wrap_tensor(ggml_round_bf16(ctx.ggml, out.tensor), out.shape, GGML_TYPE_F32);
    return out;
}

core::TensorValue build_encoder_conv_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderBlockWeights & block,
    int64_t channels,
    int stride,
    bool bf16_input) {
    auto x = build_residual_unit(ctx, input, block.residual1, channels, 1, nullptr, bf16_input);
    x = build_residual_unit(ctx, x, block.residual3, channels, 3, nullptr, bf16_input);
    x = build_residual_unit(ctx, x, block.residual9, channels, 9, nullptr, bf16_input);
    x = modules::Snake1dModule({channels}).build(ctx, x, block.snake);
    if (bf16_input) x = core::wrap_tensor(ggml_round_bf16(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    return causal_conv1d(ctx, x, block.conv, channels, channels * 2, 2 * stride, stride, 1, true, nullptr, bf16_input);
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const FishCodecWeights & weights) {
    const bool bf16_input = weights.encoder_bf16_compute &&
        (weights.encoder_first.weight.type == GGML_TYPE_BF16 || weights.encoder_first.weight.type == GGML_TYPE_F16);
    auto encoder_input = bf16_input
        ? core::wrap_tensor(ggml_round_bf16(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32)
        : input;
    auto x = causal_conv1d(ctx, encoder_input, weights.encoder_first, 1, 64, 7, 1, 1, true, nullptr, bf16_input);
    int64_t channels = 64;
    const int strides[] = {2, 4, 8, 8};
    for (size_t index = 0; index < weights.encoder_blocks.size(); ++index) {
        const auto & block = weights.encoder_blocks[index];
        x = build_encoder_conv_block(ctx, x, block, channels, strides[index], bf16_input);
        channels *= 2;
        if (block.transformer.has_value()) {
            x = build_window_transformer(ctx, constants, x, *block.transformer, 512, nullptr, bf16_input);
        }
    }
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.encoder_final_snake);
    if (bf16_input) x = core::wrap_tensor(ggml_round_bf16(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    return causal_conv1d(ctx, x, weights.encoder_final, channels, kCodecDim, 3, 1, 1, true, nullptr, bf16_input);
}

// Bound convolution workspace with overlapping causal windows. The transformer
// and quantizer still process the complete feature sequence.
constexpr int64_t kEncoderConvHop = 512;            // product of encoder strides {2, 4, 8, 8}
constexpr int64_t kEncodeContextSamples = 16384;    // >= conv receptive field (~6.4k samples)

bool encoder_is_chunkable(const FishCodecWeights & weights) {
    for (size_t index = 0; index + 1 < weights.encoder_blocks.size(); ++index) {
        if (weights.encoder_blocks[index].transformer.has_value()) {
            return false;
        }
    }
    return !weights.encoder_blocks.empty();
}

core::TensorValue build_encoder_convs(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FishCodecWeights & weights,
    bool bf16_input) {
    auto encoder_input = bf16_input
        ? core::wrap_tensor(ggml_round_bf16(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32)
        : input;
    auto x = causal_conv1d(ctx, encoder_input, weights.encoder_first, 1, 64, 7, 1, 1, true, nullptr, bf16_input);
    int64_t channels = 64;
    const int strides[] = {2, 4, 8, 8};
    for (size_t index = 0; index < weights.encoder_blocks.size(); ++index) {
        const auto & block = weights.encoder_blocks[index];
        x = build_encoder_conv_block(ctx, x, block, channels, strides[index], bf16_input);
        channels *= 2;
    }
    return x;
}

core::TensorValue build_encoder_tail(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & features,
    const FishCodecWeights & weights,
    bool bf16_input) {
    auto x = features;
    const auto & last = weights.encoder_blocks.back();
    if (last.transformer.has_value()) {
        x = build_window_transformer(ctx, constants, x, *last.transformer, 512, nullptr, bf16_input);
    }
    const int64_t channels = features.shape.dims[1];
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.encoder_final_snake);
    if (bf16_input) x = core::wrap_tensor(ggml_round_bf16(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    return causal_conv1d(ctx, x, weights.encoder_final, channels, kCodecDim, 3, 1, 1, true, nullptr, bf16_input);
}

core::TensorValue build_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FishCodecWeights & weights,
    CodecStreamingState * stream = nullptr) {
    auto x = causal_conv1d(ctx, input, weights.decoder_first, kCodecDim, 1536, 7, 1, 1, true, stream);
    int64_t channels = 1536;
    const int strides[] = {8, 8, 4, 2};
    for (size_t index = 0; index < weights.decoder_blocks.size(); ++index) {
        const auto & block = weights.decoder_blocks[index];
        x = modules::Snake1dModule({channels}).build(ctx, x, block.snake);
        x = causal_conv_transpose1d(ctx, x, block.conv, channels, channels / 2, 2 * strides[index], strides[index], true, stream);
        channels /= 2;
        x = build_residual_unit(ctx, x, block.residual1, channels, 1, stream);
        x = build_residual_unit(ctx, x, block.residual3, channels, 3, stream);
        x = build_residual_unit(ctx, x, block.residual9, channels, 9, stream);
    }
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.decoder_final_snake);
    x = causal_conv1d(ctx, x, weights.decoder_final, channels, 1, 7, 1, 1, true, stream);
    return modules::TanhModule{}.build(ctx, x);
}

core::TensorValue build_quantizer_out(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & ids_bt,
    const QuantizerUnitWeights & weights,
    int64_t codebook_size,
    bool streaming = false,
    const core::TensorValue * projected_bf16 = nullptr) {
    auto emb_btd = modules::CodebookLookupModule({codebook_size, 8}).build(ctx, ids_bt, weights.codebook);
    auto emb_bdt = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, emb_btd);
    if (projected_bf16 != nullptr) {
        auto * embedding = ggml_cast(ctx.ggml,
            core::ensure_backend_addressable_layout(ctx, emb_bdt).tensor, GGML_TYPE_F32);
        embedding = ggml_round_bf16(ctx.ggml, embedding);
        auto * delta = ggml_round_bf16(ctx.ggml, ggml_sub(ctx.ggml, embedding, projected_bf16->tensor));
        emb_bdt = core::wrap_tensor(ggml_round_bf16(ctx.ggml,
            ggml_add(ctx.ggml, projected_bf16->tensor, delta)), emb_bdt.shape, GGML_TYPE_F32);
    }
    if (streaming) {
        // Singleton time axes can be contiguous while retaining a non-unit stride.
        emb_bdt = core::wrap_tensor(ggml_cont(ctx.ggml, emb_bdt.tensor), emb_bdt.shape, emb_bdt.type);
    }
    return modules::Conv1dModule({8, kCodecDim, 1, 1, 0, 1, true, projected_bf16 != nullptr})
        .build(ctx, emb_bdt, weights.out_proj);
}

core::TensorValue build_zq_from_codes(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & code_inputs,
    const FishDacCodecConfig & config,
    const FishCodecWeights & weights,
    bool streaming = false) {
    auto latent = build_quantizer_out(ctx, code_inputs[0], weights.semantic_quantizer, config.semantic_codebook_size, streaming);
    for (size_t index = 0; index < weights.residual_quantizers.size(); ++index) {
        auto residual = build_quantizer_out(ctx, code_inputs[index + 1], weights.residual_quantizers[index], config.residual_codebook_size, streaming);
        latent = modules::AddModule{}.build(ctx, latent, residual);
    }
    if (static_cast<int64_t>(code_inputs.size()) != config.total_codebooks) {
        throw std::runtime_error("Fish DAC codec codebook input count mismatch");
    }
    return latent;
}

core::TensorValue build_decode_from_zq(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & z_q,
    const FishCodecWeights & weights,
    CodecStreamingState * stream = nullptr) {
    auto latent = build_window_transformer(ctx, constants, z_q, weights.post_module, 128, stream);
    for (const auto & stage : weights.upsample) {
        latent = causal_conv_transpose1d(ctx, latent, stage.first, kCodecDim, kCodecDim, 2, 2, true, stream);
        latent = build_convnext(ctx, latent, stage.second, kCodecDim, stream);
    }
    return latent;
}

core::TensorValue build_decode_quantizer(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const std::vector<core::TensorValue> & code_inputs,
    const FishDacCodecConfig & config,
    const FishCodecWeights & weights,
    CodecStreamingState * stream = nullptr) {
    return build_decode_from_zq(
        ctx,
        constants,
        build_zq_from_codes(ctx, code_inputs, config, weights, stream != nullptr),
        weights, stream);
}

core::TensorValue build_encode_quantizer(
    core::ModuleBuildContext & ctx,
    core::ConstantTensorCache & constants,
    const core::TensorValue & encoder_latent,
    const FishCodecWeights & weights,
    std::vector<ggml_tensor *> & code_outputs,
    core::TensorValue * z_q_out) {
    auto x = encoder_latent;
    const bool bf16 = weights.encoder_bf16_compute &&
        (weights.pre_module.layers.front().attention.q_weight.type == GGML_TYPE_BF16 ||
         weights.pre_module.layers.front().attention.q_weight.type == GGML_TYPE_F16);
    for (const auto & stage : weights.downsample) {
        x = causal_conv1d(ctx, x, stage.first, kCodecDim, kCodecDim, 2, 2, 1, true, nullptr, bf16);
        x = build_convnext(ctx, x, stage.second, kCodecDim, nullptr, bf16);
    }
    x = build_window_transformer(ctx, constants, x, weights.pre_module, 128, nullptr, bf16);
    if (ctx.backend_type == core::BackendType::Cpu) {
        x = core::ensure_backend_addressable_layout(ctx, x);
    }

    auto residual = x;
    auto quantize_one = [&](const QuantizerUnitWeights & quantizer, int64_t codebook_size) {
        const bool bf16 = weights.encoder_bf16_compute &&
            (quantizer.in_proj.weight.type == GGML_TYPE_BF16 || quantizer.in_proj.weight.type == GGML_TYPE_F16);
        if (bf16) {
            residual = core::ensure_backend_addressable_layout(ctx, residual);
            residual = core::wrap_tensor(ggml_round_bf16(ctx.ggml, residual.tensor), residual.shape, GGML_TYPE_F32);
        }
        auto projected = modules::Conv1dModule({kCodecDim, 8, 1, 1, 0, 1, true, bf16}).build(ctx, residual, quantizer.in_proj);
        auto projected_btd = l2_normalize_last(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projected), bf16);
        auto logits = modules::LinearModule({8, codebook_size, false, GGML_PREC_F32})
                          .build(ctx, projected_btd, {quantizer.normalized_codebook, std::nullopt});
        auto flat_logits = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({logits.shape.dims[1], codebook_size}));
        auto * scores = flat_logits.tensor;
        if (bf16) {
            auto * encodings = core::ensure_backend_addressable_layout(ctx, projected_btd).tensor;
            auto * codebook = ggml_cast(ctx.ggml, quantizer.normalized_codebook.tensor, GGML_TYPE_F32);
            auto * encoding_norms = ggml_round_bf16(ctx.ggml,
                ggml_sum_rows(ctx.ggml, ggml_round_bf16(ctx.ggml, ggml_sqr(ctx.ggml, encodings))));
            auto * codebook_norms = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml,
                ggml_round_bf16(ctx.ggml, ggml_sum_rows(ctx.ggml,
                    ggml_round_bf16(ctx.ggml, ggml_sqr(ctx.ggml, codebook))))));
            auto * twice_dot = ggml_scale(ctx.ggml, ggml_round_bf16(ctx.ggml, scores), 2.0F);
            auto * distance = ggml_round_bf16(ctx.ggml,
                ggml_sub(ctx.ggml, ggml_repeat(ctx.ggml, encoding_norms, scores), twice_dot));
            scores = ggml_neg(ctx.ggml, ggml_round_bf16(ctx.ggml, ggml_add(ctx.ggml, distance, codebook_norms)));
            auto * maximum = ggml_pool_2d(ctx.ggml, scores, GGML_OP_POOL_MAX,
                codebook_size, 1, codebook_size, 1, 0, 0);
            auto * tied = ggml_scale_bias(ctx.ggml,
                ggml_step(ctx.ggml, ggml_sub(ctx.ggml, ggml_repeat(ctx.ggml, maximum, scores), scores)), -1.0F, 1.0F);
            std::vector<float> priorities(static_cast<size_t>(codebook_size));
            for (int64_t i = 0; i < codebook_size; ++i) priorities[static_cast<size_t>(i)] = static_cast<float>(codebook_size - i);
            auto priority = constants.make_tensor(core::TensorShape::from_dims({codebook_size}), GGML_TYPE_F32,
                priorities.data(), priorities.size() * sizeof(float));
            scores = ggml_mul(ctx.ggml, tied, priority.tensor);
        }
        auto * ids_raw = ggml_argmax(ctx.ggml, scores);
        ggml_set_output(ids_raw);
        code_outputs.push_back(ids_raw);
        auto ids = core::reshape_tensor(
            ctx,
            core::wrap_tensor(ids_raw, core::TensorShape::from_dims({logits.shape.dims[1]}), GGML_TYPE_I32),
            core::TensorShape::from_dims({1, logits.shape.dims[1]}));
        auto quantized = build_quantizer_out(ctx, ids, quantizer, codebook_size, false, bf16 ? &projected : nullptr);
        residual = core::wrap_tensor(ggml_sub(ctx.ggml, residual.tensor, quantized.tensor), residual.shape, GGML_TYPE_F32);
        if (bf16) residual = core::wrap_tensor(ggml_round_bf16(ctx.ggml, residual.tensor), residual.shape, GGML_TYPE_F32);
    };
    quantize_one(weights.semantic_quantizer, 4096);
    for (const auto & quantizer : weights.residual_quantizers) {
        quantize_one(quantizer, 1024);
    }
    if (z_q_out != nullptr) {
        *z_q_out = core::wrap_tensor(
            ggml_sub(ctx.ggml, x.tensor, residual.tensor), x.shape, GGML_TYPE_F32);
    }
    return x;
}

modules::Snake1dWeights load_snake(core::BackendWeightStore & store, const assets::TensorSource & source, const std::string & name, int64_t channels, bool bf16 = false) {
    if (bf16) {
        auto alpha = source.require_f32(name + ".alpha", {1, channels, 1});
        core::round_f32_to_bf16_in_place(alpha);
        std::vector<float> inverse(alpha.size());
        for (size_t i = 0; i < alpha.size(); ++i) {
            // TorchScript fuses Snake in F32 and rounds only its output.
            const float denominator = alpha[i] + ggml_bf16_to_fp32(ggml_fp32_to_bf16(1.0e-9F));
            inverse[i] = 1.0F / denominator;
        }
        const auto shape = core::TensorShape::from_dims({1, channels, 1});
        return {store.make_f32(shape, std::move(alpha)),
            store.make_f32(shape, std::move(inverse))};
    }
    return {store.make_f32(
        core::TensorShape::from_dims({channels}),
        source.require_f32(name + ".alpha", {1, channels, 1}))};
}

CodecTransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t layers) {
    CodecTransformerWeights out;
    out.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string layer_prefix = prefix + ".layers." + std::to_string(layer);
        CodecTransformerLayerWeights weights;
        weights.attention_norm = binding::norm_weight_from_source(store, source, layer_prefix + ".attention_norm", kCodecDim);
        weights.attention.q_weight = store.load_tensor(source, layer_prefix + ".attention.q_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.k_weight = store.load_tensor(source, layer_prefix + ".attention.k_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.v_weight = store.load_tensor(source, layer_prefix + ".attention.v_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.out_weight = store.load_tensor(source, layer_prefix + ".attention.wo.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention_scale = binding::layer_scale_from_named_source(store, source, layer_prefix + ".attention_layer_scale.gamma");
        weights.ffn_norm = binding::norm_weight_from_source(store, source, layer_prefix + ".ffn_norm", kCodecDim);
        weights.feed_forward.gate_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w1.weight", storage_type, {kCodecIntermediate, kCodecDim});
        weights.feed_forward.down_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w2.weight", storage_type, {kCodecDim, kCodecIntermediate});
        weights.feed_forward.up_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w3.weight", storage_type, {kCodecIntermediate, kCodecDim});
        weights.ffn_scale = binding::layer_scale_from_named_source(store, source, layer_prefix + ".ffn_layer_scale.gamma");
        out.layers.push_back(std::move(weights));
    }
    out.norm = binding::norm_weight_from_source(store, source, prefix + ".norm", kCodecDim);
    return out;
}

ResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    bool bf16 = false) {
    ResidualUnitWeights out;
    out.snake1 = load_snake(store, source, prefix + ".block.0", channels, bf16);
    out.conv1 = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".block.1.conv.weight",
        prefix + ".block.1.conv.bias",
        storage_type);
    out.snake2 = load_snake(store, source, prefix + ".block.2", channels, bf16);
    out.conv2 = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".block.3.conv.weight",
        prefix + ".block.3.conv.bias",
        storage_type);
    return out;
}

ConvNeXtBlockWeights load_convnext(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels) {
    ConvNeXtBlockWeights out;
    out.dwconv = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".dwconv.conv",
        storage_type,
        channels,
        7,
        true);
    out.norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.pwconv1 = binding::linear_from_source(store, source, prefix + ".pwconv1", storage_type, channels * 4, channels, true);
    out.pwconv2 = binding::linear_from_source(store, source, prefix + ".pwconv2", storage_type, channels, channels * 4, true);
    out.gamma = binding::layer_scale_from_named_source(store, source, prefix + ".gamma");
    return out;
}

QuantizerUnitWeights load_quantizer_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t codebook_size,
    bool encoder_bf16_compute) {
    QuantizerUnitWeights out;
    out.in_proj = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".in_proj.weight",
        prefix + ".in_proj.bias",
        storage_type);
    out.out_proj = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".out_proj.weight",
        prefix + ".out_proj.bias",
        storage_type);
    const auto codebook = source.require_f32(prefix + ".codebook.weight", {codebook_size, 8});
    // Reference encoding uses BF16 arithmetic independently of F16 GGUF storage.
    const bool bf16 = encoder_bf16_compute &&
        (out.in_proj.weight.type == GGML_TYPE_BF16 || out.in_proj.weight.type == GGML_TYPE_F16);
    auto encoder_codebook = codebook;
    if (bf16) core::round_f32_to_bf16_in_place(encoder_codebook);
    out.codebook = store.make_from_f32(core::TensorShape::from_dims({codebook_size, 8}), storage_type, codebook);
    out.normalized_codebook = store.make_from_f32(
        core::TensorShape::from_dims({codebook_size, 8}),
        bf16 ? assets::TensorStorageType::BF16 : storage_type,
        normalized_rows(encoder_codebook, codebook_size, 8, bf16));
    return out;
}

std::shared_ptr<FishCodecWeights> load_weights(
    const assets::TensorSource & source,
    const FishDacCodecConfig & config,
    const FishDacCodecWeightBinding & binding_config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    bool encoder_bf16_compute) {
    auto weights = std::make_shared<FishCodecWeights>();
    weights->encoder_bf16_compute = encoder_bf16_compute && backend_type == core::BackendType::Cuda;
    weights->store = std::make_shared<core::BackendWeightStore>(backend, backend_type, "Fish DAC codec", weight_context_bytes);
    auto & store = *weights->store;
    const auto & encoder = binding_config.encoder_prefix;
    const auto & quantizer = binding_config.quantizer_prefix;
    const auto & decoder = binding_config.decoder_prefix;

    weights->encoder_first = binding::conv1d_from_named_source(
        store,
        source,
        encoder + ".block.0.conv.weight",
        encoder + ".block.0.conv.bias",
        conv_storage_type);
    const auto encoder_matmul_storage = weights->encoder_bf16_compute &&
        (matmul_storage_type == assets::TensorStorageType::Native || matmul_storage_type == assets::TensorStorageType::F16) &&
        (weights->encoder_first.weight.type == GGML_TYPE_BF16 || weights->encoder_first.weight.type == GGML_TYPE_F16)
        ? assets::TensorStorageType::BF16 : matmul_storage_type;
    const bool bf16_encoder = weights->encoder_bf16_compute &&
        (weights->encoder_first.weight.type == GGML_TYPE_BF16 || weights->encoder_first.weight.type == GGML_TYPE_F16);
    const auto encoder_conv_storage = bf16_encoder && conv_storage_type == assets::TensorStorageType::Native
        ? assets::TensorStorageType::BF16 : conv_storage_type;
    int64_t encoder_channels = 64;
    for (int64_t block_index = 0; block_index < 4; ++block_index) {
        const std::string prefix = encoder + ".block." + std::to_string(block_index + 1) + ".block";
        EncoderBlockWeights block;
        block.residual1 = load_residual_unit(store, source, prefix + ".0", encoder_conv_storage, encoder_channels, bf16_encoder);
        block.residual3 = load_residual_unit(store, source, prefix + ".1", encoder_conv_storage, encoder_channels, bf16_encoder);
        block.residual9 = load_residual_unit(store, source, prefix + ".2", encoder_conv_storage, encoder_channels, bf16_encoder);
        block.snake = load_snake(store, source, prefix + ".3", encoder_channels, bf16_encoder);
        block.conv = binding::conv1d_from_named_source(
            store,
            source,
            prefix + ".4.conv.weight",
            prefix + ".4.conv.bias",
            encoder_conv_storage);
        encoder_channels *= 2;
        if (block_index == 3) {
            block.transformer = load_transformer(store, source, prefix + ".5", encoder_matmul_storage, 4);
        }
        weights->encoder_blocks.push_back(std::move(block));
    }
    weights->encoder_final_snake = load_snake(store, source, encoder + ".block.5", kCodecDim, bf16_encoder);
    weights->encoder_final = binding::conv1d_from_named_source(
        store,
        source,
        encoder + ".block.6.conv.weight",
        encoder + ".block.6.conv.bias",
        encoder_conv_storage);

    for (int64_t i = 0; i < 2; ++i) {
        const std::string prefix = quantizer + ".downsample." + std::to_string(i);
        weights->downsample.push_back({
            binding::conv1d_from_named_source(
                store,
                source,
                prefix + ".0.conv.weight",
                prefix + ".0.conv.bias",
                conv_storage_type),
            load_convnext(store, source, prefix + ".1", encoder_matmul_storage, kCodecDim),
        });
    }
    weights->pre_module = load_transformer(store, source, quantizer + ".pre_module", encoder_matmul_storage, kCodecTransformerLayers);
    weights->semantic_quantizer = load_quantizer_unit(
        store,
        source,
        quantizer + ".semantic_quantizer.quantizers.0",
        matmul_storage_type,
        config.semantic_codebook_size, weights->encoder_bf16_compute);
    for (int64_t i = 0; i < config.quantizer_codebooks; ++i) {
        weights->residual_quantizers.push_back(
            load_quantizer_unit(
                store,
                source,
                quantizer + ".quantizer.quantizers." + std::to_string(i),
                matmul_storage_type,
                config.residual_codebook_size, weights->encoder_bf16_compute));
    }
    weights->post_module = load_transformer(store, source, quantizer + ".post_module", matmul_storage_type, kCodecTransformerLayers);
    for (int64_t i = 0; i < 2; ++i) {
        const std::string prefix = quantizer + ".upsample." + std::to_string(i);
        weights->upsample.push_back({
            binding::conv_transpose1d_from_named_source(
                store,
                source,
                prefix + ".0.conv.weight",
                prefix + ".0.conv.bias",
                conv_storage_type),
            load_convnext(store, source, prefix + ".1", matmul_storage_type, kCodecDim),
        });
    }

    weights->decoder_first = binding::conv1d_from_named_source(
        store,
        source,
        decoder + ".model.0.conv.weight",
        decoder + ".model.0.conv.bias",
        conv_storage_type);
    int64_t decoder_channels = 1536;
    for (int64_t block_index = 0; block_index < 4; ++block_index) {
        const std::string prefix = decoder + ".model." + std::to_string(block_index + 1) + ".block";
        DecoderBlockWeights block;
        block.snake = load_snake(store, source, prefix + ".0", decoder_channels);
        block.conv = binding::conv_transpose1d_from_named_source(
            store,
            source,
            prefix + ".1.conv.weight",
            prefix + ".1.conv.bias",
            conv_storage_type);
        decoder_channels /= 2;
        block.residual1 = load_residual_unit(store, source, prefix + ".2", conv_storage_type, decoder_channels);
        block.residual3 = load_residual_unit(store, source, prefix + ".3", conv_storage_type, decoder_channels);
        block.residual9 = load_residual_unit(store, source, prefix + ".4", conv_storage_type, decoder_channels);
        weights->decoder_blocks.push_back(std::move(block));
    }
    weights->decoder_final_snake = load_snake(store, source, decoder + ".model.5", 96);
    weights->decoder_final = binding::conv1d_from_named_source(
        store,
        source,
        decoder + ".model.6.conv.weight",
        decoder + ".model.6.conv.bias",
        conv_storage_type);

    store.upload();
    return weights;
}

struct DecodeGraph {
    DecodeGraph(
        FishDacCodecConfig config,
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t frames,
        bool streaming = false)
        : config_(std::move(config)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          frame_capacity_(frames),
          constants_(backend_, threads_, "Fish DAC codec decode constants") {
        if (streaming) stream_ = std::make_unique<CodecStreamingState>();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish DAC codec decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.decode", backend_type_};
        constants_.begin_graph();
        for (int64_t codebook = 0; codebook < config_.total_codebooks; ++codebook) {
            auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frame_capacity_}));
            ggml_set_input(ids.tensor);
            code_inputs_.push_back(ids);
        }
        auto latent = build_decode_quantizer(ctx, constants_, code_inputs_, config_, *weights_, stream_.get());
        auto waveform = build_decoder(ctx, latent, *weights_, stream_.get());
        output_ = waveform.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        if (stream_) stream_->expand(graph_);
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish DAC codec decode graph");
        }
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t frames, ggml_backend_t backend, int threads) const {
        return frame_capacity_ >= frames && backend_ == backend && threads_ == std::max(1, threads);
    }

    runtime::AudioBuffer run_stream(const FishDacCodes & codes) {
        stream_->prepare(backend_);
        auto audio = run(codes);
        stream_->commit(backend_);
        return audio;
    }

    void reset_stream() {
        stream_->frames = 0;
    }

    runtime::AudioBuffer run(const FishDacCodes & codes) {
        const int64_t codebooks = config_.total_codebooks;
        if (codes.codebooks != codebooks || codes.frames <= 0 ||
            static_cast<int64_t>(codes.codes.size()) != codebooks * codes.frames) {
            std::ostringstream oss;
            oss << "Fish DAC codec decode code shape mismatch: expected_codebooks=" << codebooks
                << " actual_codebooks=" << codes.codebooks
                << " frames=" << codes.frames
                << " values=" << codes.codes.size()
                << " expected_values=" << (codebooks * codes.frames);
            throw std::runtime_error(oss.str());
        }
        if (codes.frames > frame_capacity_) {
            throw std::runtime_error("Fish DAC codec decode request exceeds graph capacity");
        }
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            std::vector<int32_t> padded(static_cast<size_t>(frame_capacity_), 0);
            for (int64_t frame = 0; frame < codes.frames; ++frame) {
                int32_t value = codes.codes[static_cast<size_t>(codebook * codes.frames + frame)];
                if (codebook == 0) {
                    value = std::clamp<int32_t>(value, 0, static_cast<int32_t>(config_.semantic_codebook_size - 1));
                } else {
                    value = std::clamp<int32_t>(value, 0, static_cast<int32_t>(config_.residual_codebook_size - 1));
                }
                padded[static_cast<size_t>(frame)] = value;
            }
            core::write_tensor_i32(code_inputs_[static_cast<size_t>(codebook)], padded);
        }
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish DAC codec decode graph compute failed");
        }
        auto values = core::read_tensor_f32(output_);
        const int64_t expected_samples = codes.frames * config_.frame_length;
        if (static_cast<int64_t>(values.size()) > expected_samples) {
            values.resize(static_cast<size_t>(expected_samples));
        }
        return runtime::AudioBuffer{config_.sample_rate, 1, std::move(values)};
    }

private:
    FishDacCodecConfig config_;
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::vector<core::TensorValue> code_inputs_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    core::ConstantTensorCache constants_;
    std::unique_ptr<CodecStreamingState> stream_;
};

struct LatentDecodeGraph {
    LatentDecodeGraph(
        FishDacCodecConfig config,
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t frames)
        : config_(std::move(config)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          frame_capacity_(frames),
          constants_(backend_, threads_, "Fish DAC codec latent decode constants") {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish DAC codec latent decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_dac.codec.latent_decode", backend_type_};
        constants_.begin_graph();
        latent_input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.latent_dim, frame_capacity_}));
        ggml_set_input(latent_input_.tensor);
        auto latent = build_decode_from_zq(ctx, constants_, latent_input_, *weights_);
        auto waveform = build_decoder(ctx, latent, *weights_);
        output_ = waveform.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish DAC codec latent decode graph");
        }
    }

    ~LatentDecodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t frames, ggml_backend_t backend, int threads) const {
        return frame_capacity_ >= frames && backend_ == backend && threads_ == std::max(1, threads);
    }

    runtime::AudioBuffer run(const FishDacLatents & latents) {
        if (latents.frames <= 0 ||
            latents.channels != config_.latent_dim ||
            static_cast<int64_t>(latents.values.size()) != latents.frames * config_.latent_dim) {
            throw std::runtime_error("Fish DAC codec latent decode received a mis-shaped latent buffer");
        }
        if (latents.frames > frame_capacity_) {
            throw std::runtime_error("Fish DAC codec latent decode request exceeds graph capacity");
        }
        std::vector<float> padded(static_cast<size_t>(config_.latent_dim * frame_capacity_), 0.0F);
        for (int64_t frame = 0; frame < latents.frames; ++frame) {
            for (int64_t channel = 0; channel < config_.latent_dim; ++channel) {
                padded[static_cast<size_t>(channel * frame_capacity_ + frame)] =
                    latents.values[static_cast<size_t>(frame * config_.latent_dim + channel)];
            }
        }
        core::write_tensor_f32(latent_input_, padded);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish DAC codec latent decode graph compute failed");
        }
        auto values = core::read_tensor_f32(output_);
        const int64_t expected_samples = latents.frames * config_.frame_length;
        if (static_cast<int64_t>(values.size()) > expected_samples) {
            values.resize(static_cast<size_t>(expected_samples));
        }
        return runtime::AudioBuffer{config_.sample_rate, 1, std::move(values)};
    }

private:
    FishDacCodecConfig config_;
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue latent_input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    core::ConstantTensorCache constants_;
};

void set_encoder_matmul_precision(
    ggml_cgraph * graph,
    core::BackendType backend_type,
    bool encoder_f32_accumulation) {
    if (!encoder_f32_accumulation || backend_type != core::BackendType::Vulkan) {
        return;
    }
    // Preserve F16 storage while avoiding F16 accumulation during reference encoding.
    for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
        auto * tensor = ggml_graph_node(graph, node);
        if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0]->type == GGML_TYPE_F16 &&
            tensor->src[1]->type == GGML_TYPE_F16) {
            ggml_mul_mat_set_prec(tensor, GGML_PREC_F32);
        }
    }
}

struct EncodeGraph {
    EncodeGraph(
        FishDacCodecConfig config,
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t samples,
        int64_t frames,
        bool want_latents,
        bool encoder_f32_accumulation)
        : config_(std::move(config)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          sample_capacity_(samples),
          frame_capacity_(frames),
          wants_latents_(want_latents),
          constants_(backend_, threads_, "Fish DAC codec encode constants") {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish DAC codec encode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.encode", backend_type_};
        constants_.begin_graph();
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, sample_capacity_}));
        ggml_set_input(input_.tensor);
        auto encoded = build_encoder(ctx, constants_, input_, *weights_);
        core::TensorValue z_q;
        build_encode_quantizer(
            ctx,
            constants_,
            encoded,
            *weights_,
            code_outputs_,
            wants_latents_ ? &z_q : nullptr);
        if (wants_latents_) {
            z_q_output_ = core::ensure_backend_addressable_layout(ctx, z_q).tensor;
            ggml_set_output(z_q_output_);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        if (z_q_output_ != nullptr) {
            ggml_build_forward_expand(graph_, z_q_output_);
        }
        for (ggml_tensor * code_output : code_outputs_) {
            ggml_build_forward_expand(graph_, code_output);
        }
        set_encoder_matmul_precision(graph_, backend_type_, encoder_f32_accumulation);
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish DAC codec encode graph");
        }
    }

    ~EncodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t samples, int64_t frames, ggml_backend_t backend, int threads, bool want_latents) const {
        return sample_capacity_ >= samples &&
            frame_capacity_ >= frames &&
            backend_ == backend &&
            threads_ == std::max(1, threads) &&
            wants_latents_ == want_latents;
    }

    FishDacCodes run(const runtime::AudioBuffer & audio) {
        auto mono = prepare_codec_mono(audio, config_.sample_rate);
        const int64_t original_samples = static_cast<int64_t>(mono.size());
        const int64_t padded_samples = ceil_div(original_samples, config_.frame_length) * config_.frame_length;
        const int64_t frames = ceil_div(original_samples, config_.frame_length);
        if (padded_samples > sample_capacity_ || frames > frame_capacity_) {
            throw std::runtime_error("Fish DAC codec encode request exceeds graph capacity");
        }
        mono.resize(static_cast<size_t>(sample_capacity_), 0.0F);
        core::write_tensor_f32(input_, mono);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish DAC codec encode graph compute failed");
        }
        FishDacCodes out;
        out.codebooks = static_cast<int64_t>(code_outputs_.size());
        out.frames = frames;
        out.codes.resize(static_cast<size_t>(out.codebooks * out.frames));
        for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
            auto values = core::read_tensor_i32(code_outputs_[static_cast<size_t>(codebook)]);
            for (int64_t frame = 0; frame < out.frames; ++frame) {
                out.codes[static_cast<size_t>(codebook * out.frames + frame)] = values[static_cast<size_t>(frame)];
            }
        }
        return out;
    }

    FishDacLatents read_latents(int64_t frames) const {
        if (z_q_output_ == nullptr) {
            throw std::runtime_error("Fish DAC codec encode graph was not built with latent output");
        }
        auto values = core::read_tensor_f32(z_q_output_);
        const size_t wanted = static_cast<size_t>(frames * config_.latent_dim);
        if (values.size() < static_cast<size_t>(config_.latent_dim * frame_capacity_)) {
            throw std::runtime_error("Fish DAC codec latent output is smaller than graph capacity");
        }
        FishDacLatents out;
        out.frames = frames;
        out.channels = config_.latent_dim;
        out.values.resize(wanted);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < config_.latent_dim; ++channel) {
                out.values[static_cast<size_t>(frame * config_.latent_dim + channel)] =
                    values[static_cast<size_t>(channel * frame_capacity_ + frame)];
            }
        }
        return out;
    }

private:
    FishDacCodecConfig config_;
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    ggml_tensor * z_q_output_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t sample_capacity_ = 0;
    int64_t frame_capacity_ = 0;
    bool wants_latents_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    std::vector<ggml_tensor *> code_outputs_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    core::ConstantTensorCache constants_;
};

// One cached waveform window -> convolution features.
struct EncodeConvGraph {
    EncodeConvGraph(
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t window_samples,
        bool bf16_input,
        bool encoder_f32_accumulation)
        : weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          window_samples_(window_samples),
          bf16_input_(bf16_input) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish DAC codec encode-conv graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.encode_convs", backend_type_};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, window_samples_}));
        ggml_set_input(input_.tensor);
        auto features = build_encoder_convs(ctx, input_, *weights_, bf16_input_);
        channels_ = features.shape.dims[1];
        frames_ = features.shape.dims[2];
        output_ = core::ensure_backend_addressable_layout(ctx, features).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        set_encoder_matmul_precision(graph_, backend_type_, encoder_f32_accumulation);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish DAC codec encode-conv graph");
        }
    }

    ~EncodeConvGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t window_samples, ggml_backend_t backend, int threads) const {
        return window_samples_ == window_samples && backend_ == backend && threads_ == std::max(1, threads);
    }

    int64_t window_samples() const { return window_samples_; }
    int64_t channels() const { return channels_; }
    int64_t frames() const { return frames_; }

    // Returns channel-major features: value(c, t) at index c * frames() + t.
    std::vector<float> run(const std::vector<float> & window) {
        core::write_tensor_f32(input_, window);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish DAC codec encode-conv graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

private:
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t window_samples_ = 0;
    bool bf16_input_ = false;
    int64_t channels_ = 0;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

// Complete convolution features -> reference codes.
struct EncodeTailGraph {
    EncodeTailGraph(
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t channels,
        int64_t feature_frames,
        bool bf16_input,
        bool encoder_f32_accumulation)
        : weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          feature_frames_(feature_frames),
          bf16_input_(bf16_input),
          constants_(backend_, threads_, "Fish DAC codec encode-tail constants") {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish DAC codec encode-tail graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.encode_tail", backend_type_};
        constants_.begin_graph();
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, channels, feature_frames_}));
        ggml_set_input(input_.tensor);
        auto encoded = build_encoder_tail(ctx, constants_, input_, *weights_, bf16_input_);
        build_encode_quantizer(ctx, constants_, encoded, *weights_, code_outputs_, nullptr);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        for (ggml_tensor * code_output : code_outputs_) {
            ggml_build_forward_expand(graph_, code_output);
        }
        set_encoder_matmul_precision(graph_, backend_type_, encoder_f32_accumulation);
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish DAC codec encode-tail graph");
        }
    }

    ~EncodeTailGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t feature_frames, ggml_backend_t backend, int threads) const {
        return feature_frames_ == feature_frames && backend_ == backend && threads_ == std::max(1, threads);
    }

    FishDacCodes run(const std::vector<float> & features, int64_t frames) {
        core::write_tensor_f32(input_, features);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish DAC codec encode-tail graph compute failed");
        }
        FishDacCodes out;
        out.codebooks = static_cast<int64_t>(code_outputs_.size());
        out.frames = frames;
        out.codes.resize(static_cast<size_t>(out.codebooks * out.frames));
        for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
            auto values = core::read_tensor_i32(code_outputs_[static_cast<size_t>(codebook)]);
            if (static_cast<int64_t>(values.size()) < frames) {
                throw std::runtime_error("Fish DAC codec encode-tail produced too few frames");
            }
            for (int64_t frame = 0; frame < frames; ++frame) {
                out.codes[static_cast<size_t>(codebook * frames + frame)] = values[static_cast<size_t>(frame)];
            }
        }
        return out;
    }

private:
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t feature_frames_ = 0;
    bool bf16_input_ = false;
    core::ConstantTensorCache constants_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    std::vector<ggml_tensor *> code_outputs_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

}  // namespace

struct FishDacCodecComponent::Impl {
    FishDacCodecConfig config;
    std::shared_ptr<const FishCodecWeights> weights;
};

class FishDacCodecRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FishDacCodecComponent> component,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        bool encoder_f32_accumulation,
        int64_t encoder_chunk_samples)
        : component_(std::move(component)),
          execution_(std::move(backend)),
          threads_(std::max(1, threads)),
          graph_arena_bytes_(graph_arena_bytes),
          encoder_f32_accumulation_(encoder_f32_accumulation),
          encoder_chunk_samples_(encoder_chunk_samples) {
        if (component_ == nullptr || component_->impl_ == nullptr || component_->impl_->weights == nullptr) {
            throw std::runtime_error("Fish DAC codec runtime requires a loaded codec component");
        }
        if (encoder_chunk_samples_ < 0 || encoder_chunk_samples_ % kEncoderConvHop != 0) {
            throw std::invalid_argument("Fish DAC codec encoder_chunk_samples must be 0 or a positive multiple of 512 samples");
        }
        if (encoder_chunk_samples_ > 0 &&
            (component_->impl_->config.frame_length % kEncoderConvHop != 0 ||
             !encoder_is_chunkable(*component_->impl_->weights))) {
            throw std::invalid_argument("Fish DAC codec chunked encoding requires hop-aligned frames and a convolution stack with no intermediate transformers");
        }
    }

    FishDacCodes encode_codes(const runtime::AudioBuffer & audio) {
        const auto & config = component_->impl_->config;
        auto mono = prepare_codec_mono(audio, config.sample_rate);
        const int64_t samples = ceil_div(static_cast<int64_t>(mono.size()), config.frame_length) * config.frame_length;
        const int64_t frames = ceil_div(static_cast<int64_t>(mono.size()), config.frame_length);
        if (encoder_chunk_samples_ > 0 && samples > encoder_chunk_samples_ &&
            samples - encoder_chunk_samples_ > kEncodeContextSamples) {
            return encode_codes_chunked(std::move(mono), samples, frames, encoder_chunk_samples_);
        }
        if (encode_graph_ == nullptr || !encode_graph_->matches(samples, frames, execution_.backend(), threads_, false)) {
            encode_graph_.reset();
            encode_graph_ = std::make_unique<EncodeGraph>(
                config,
                component_->impl_->weights,
                execution_,
                graph_arena_bytes_,
                samples,
                frames,
                false,
                encoder_f32_accumulation_);
        }
        return encode_graph_->run(audio);
    }

    // Keep the first window's boundary padding; subsequent windows discard left context.
    FishDacCodes encode_codes_chunked(std::vector<float> mono, int64_t samples, int64_t frames, int64_t chunk) {
        const auto & weights = component_->impl_->weights;
        const int64_t window = kEncodeContextSamples + chunk;
        mono.resize(static_cast<size_t>(samples), 0.0F);
        const bool bf16 = weights->encoder_bf16_compute &&
            (weights->encoder_first.weight.type == GGML_TYPE_BF16 || weights->encoder_first.weight.type == GGML_TYPE_F16);
        if (encode_conv_graph_ == nullptr || !encode_conv_graph_->matches(window, execution_.backend(), threads_)) {
            encode_conv_graph_.reset();
            encode_conv_graph_ = std::make_unique<EncodeConvGraph>(
                weights, execution_, graph_arena_bytes_, window, bf16, encoder_f32_accumulation_);
        }
        const int64_t channels = encode_conv_graph_->channels();
        const int64_t window_frames = encode_conv_graph_->frames();
        if (window_frames * kEncoderConvHop != window) {
            throw std::runtime_error("Fish DAC codec encode-conv window is not hop-aligned");
        }
        const int64_t total_frames = samples / kEncoderConvHop;
        std::vector<float> features(static_cast<size_t>(channels * total_frames), 0.0F);
        std::vector<float> buffer(static_cast<size_t>(window));
        for (int64_t k = 0;; ++k) {
            const int64_t start = k * chunk;
            const int64_t keep_from = k == 0 ? 0 : kEncodeContextSamples / kEncoderConvHop;
            const int64_t global0 = (start / kEncoderConvHop) + keep_from;
            std::fill(buffer.begin(), buffer.end(), 0.0F);
            const int64_t n = std::min(window, samples - start);
            std::copy_n(mono.begin() + start, n, buffer.begin());
            const auto values = encode_conv_graph_->run(buffer);
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t t = keep_from; t < window_frames; ++t) {
                    const int64_t g = global0 + (t - keep_from);
                    if (g >= total_frames) {
                        break;
                    }
                    features[static_cast<size_t>(c * total_frames + g)] =
                        values[static_cast<size_t>(c * window_frames + t)];
                }
            }
            if (start + window >= samples) {
                break;
            }
        }
        if (encode_tail_graph_ == nullptr || !encode_tail_graph_->matches(total_frames, execution_.backend(), threads_)) {
            encode_tail_graph_.reset();
            encode_tail_graph_ = std::make_unique<EncodeTailGraph>(
                weights, execution_, graph_arena_bytes_, channels, total_frames, bf16, encoder_f32_accumulation_);
        }
        return encode_tail_graph_->run(features, frames);
    }

    FishDacLatents encode_latents(const runtime::AudioBuffer & audio) {
        const auto & config = component_->impl_->config;
        auto mono = prepare_codec_mono(audio, config.sample_rate);
        const int64_t samples = ceil_div(static_cast<int64_t>(mono.size()), config.frame_length) * config.frame_length;
        const int64_t frames = ceil_div(static_cast<int64_t>(mono.size()), config.frame_length);
        if (encode_graph_ == nullptr || !encode_graph_->matches(samples, frames, execution_.backend(), threads_, true)) {
            encode_graph_.reset();
            encode_graph_ = std::make_unique<EncodeGraph>(
                config,
                component_->impl_->weights,
                execution_,
                graph_arena_bytes_,
                samples,
                frames,
                true,
                encoder_f32_accumulation_);
        }
        (void)encode_graph_->run(audio);
        return encode_graph_->read_latents(frames);
    }

    runtime::AudioBuffer decode_codes(const FishDacCodes & codes) {
        if (decode_graph_ == nullptr || !decode_graph_->matches(codes.frames, execution_.backend(), threads_)) {
            decode_graph_.reset();
            decode_graph_ = std::make_unique<DecodeGraph>(
                component_->impl_->config,
                component_->impl_->weights,
                execution_,
                graph_arena_bytes_,
                codes.frames);
        }
        return decode_graph_->run(codes);
    }

    runtime::AudioBuffer decode_stream(const FishDacCodes & codes) {
        if (codes.frames != 1) {
            throw std::runtime_error("Fish DAC streaming decode requires one codec frame");
        }
        if (!stream_decode_graph_) {
            stream_decode_graph_ = std::make_unique<DecodeGraph>(
                component_->impl_->config, component_->impl_->weights,
                execution_, graph_arena_bytes_, 1, true);
        }
        return stream_decode_graph_->run_stream(codes);
    }

    void reset_decode_stream() {
        if (stream_decode_graph_) stream_decode_graph_->reset_stream();
    }

    runtime::AudioBuffer decode_latents(const FishDacLatents & latents) {
        if (latent_decode_graph_ == nullptr || !latent_decode_graph_->matches(latents.frames, execution_.backend(), threads_)) {
            latent_decode_graph_.reset();
            latent_decode_graph_ = std::make_unique<LatentDecodeGraph>(
                component_->impl_->config,
                component_->impl_->weights,
                execution_,
                graph_arena_bytes_,
                latents.frames);
        }
        return latent_decode_graph_->run(latents);
    }

    runtime::AudioBuffer decode_latents(const std::vector<float> & values, int64_t frames) {
        FishDacLatents latents;
        latents.values = values;
        latents.frames = frames;
        latents.channels = component_->impl_->config.latent_dim;
        return decode_latents(latents);
    }

    void release_decode_graphs() {
        decode_graph_.reset();
        latent_decode_graph_.reset();
        stream_decode_graph_.reset();
    }

    void release_encode_graph() {
        encode_graph_.reset();
        encode_conv_graph_.reset();
        encode_tail_graph_.reset();
    }

    void release_runtime_graphs() {
        encode_graph_.reset();
        encode_conv_graph_.reset();
        encode_tail_graph_.reset();
        decode_graph_.reset();
        latent_decode_graph_.reset();
        stream_decode_graph_.reset();
    }

private:
    std::shared_ptr<const FishDacCodecComponent> component_;
    core::ExecutionContext execution_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    bool encoder_f32_accumulation_ = false;
    int64_t encoder_chunk_samples_ = 0;
    std::unique_ptr<EncodeGraph> encode_graph_;
    std::unique_ptr<EncodeConvGraph> encode_conv_graph_;
    std::unique_ptr<EncodeTailGraph> encode_tail_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
    std::unique_ptr<DecodeGraph> stream_decode_graph_;
    std::unique_ptr<LatentDecodeGraph> latent_decode_graph_;
};

std::shared_ptr<const FishDacCodecComponent> FishDacCodecComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    FishDacCodecConfig config,
    FishDacCodecWeightBinding binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    FishDacCodecRuntimeOptions options) {
    if (source == nullptr) {
        throw std::runtime_error("Fish DAC codec component requires a tensor source");
    }
    validate_config(config);
    auto impl = std::make_shared<Impl>();
    impl->config = std::move(config);
    impl->weights = load_weights(
        *source,
        impl->config,
        binding,
        backend,
        backend_type,
        options.weight_context_bytes,
        options.matmul_weight_storage_type,
        options.conv_weight_storage_type,
        options.encoder_bf16_compute);
    return std::shared_ptr<const FishDacCodecComponent>(new FishDacCodecComponent(std::move(impl)));
}

FishDacCodecComponent::FishDacCodecComponent(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

FishDacCodecComponent::~FishDacCodecComponent() = default;

const FishDacCodecConfig & FishDacCodecComponent::config() const noexcept {
    return impl_->config;
}

FishDacCodecRuntime::FishDacCodecRuntime(
    std::shared_ptr<const FishDacCodecComponent> component,
    core::BackendConfig backend,
    int threads,
    size_t graph_arena_bytes,
    bool encoder_f32_accumulation,
    int64_t encoder_chunk_samples)
    : impl_(std::make_unique<Impl>(
          std::move(component),
          std::move(backend),
          threads,
          graph_arena_bytes,
          encoder_f32_accumulation,
          encoder_chunk_samples)) {}

FishDacCodecRuntime::~FishDacCodecRuntime() = default;

FishDacCodes FishDacCodecRuntime::encode_codes(const runtime::AudioBuffer & audio) {
    return impl_->encode_codes(audio);
}

FishDacLatents FishDacCodecRuntime::encode_latents(const runtime::AudioBuffer & audio) {
    return impl_->encode_latents(audio);
}

runtime::AudioBuffer FishDacCodecRuntime::decode_codes(const FishDacCodes & codes) {
    return impl_->decode_codes(codes);
}

runtime::AudioBuffer FishDacCodecRuntime::decode_stream(const FishDacCodes & codes) {
    return impl_->decode_stream(codes);
}

void FishDacCodecRuntime::reset_decode_stream() {
    impl_->reset_decode_stream();
}

runtime::AudioBuffer FishDacCodecRuntime::decode_latents(const FishDacLatents & latents) {
    return impl_->decode_latents(latents);
}

runtime::AudioBuffer FishDacCodecRuntime::decode_latents(const std::vector<float> & values, int64_t frames) {
    return impl_->decode_latents(values, frames);
}

void FishDacCodecRuntime::release_decode_graphs() {
    impl_->release_decode_graphs();
}

void FishDacCodecRuntime::release_encode_graph() {
    impl_->release_encode_graph();
}

void FishDacCodecRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::codecs

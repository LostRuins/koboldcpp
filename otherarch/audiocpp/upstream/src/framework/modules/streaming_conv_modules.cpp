#include "engine/framework/modules/streaming_conv_modules.h"

#include "tensor_layout_utils.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {

namespace {

core::TensorValue ensure_f32(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue regular_conv_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight,
    const char * module_name) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32 || contiguous.type == GGML_TYPE_F16) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_BF16) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F16), contiguous.shape, GGML_TYPE_F16);
    }
    if (ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string(module_name) + " does not support weight type with the current ggml conv path: " +
        ggml_type_name(contiguous.type));
}

int64_t depthwise_conv1d_output_frames(const DepthwiseConv1dConfig & config, int64_t input_frames) {
    return (input_frames + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) / config.stride + 1;
}

core::TensorValue add_bias_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, channels, 1}));
    auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue zeros_like_prefix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t prefix_frames) {
    auto prefix = RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})})
                      .build(ctx, SliceModule({2, 0, 1}).build(ctx, input));
    auto prefix_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, prefix);
    return core::wrap_tensor(ggml_scale(ctx.ggml, prefix_contiguous.tensor, 0.0f), prefix.shape, GGML_TYPE_F32);
}

core::TensorValue repeat_first_frame(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t prefix_frames) {
    auto first = SliceModule({2, 0, 1}).build(ctx, input);
    return RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})}).build(ctx, first);
}

core::TensorValue zeros_like_suffix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t suffix_frames) {
    auto suffix = RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], suffix_frames})})
                      .build(ctx, SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input));
    auto suffix_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, suffix);
    return core::wrap_tensor(ggml_scale(ctx.ggml, suffix_contiguous.tensor, 0.0f), suffix.shape, GGML_TYPE_F32);
}

core::TensorValue repeat_last_frame(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t suffix_frames) {
    auto last = SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
    return RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], suffix_frames})}).build(ctx, last);
}

std::pair<int64_t, int64_t> streaming_conv1d_padding(const StreamingConv1dConfig & config, int64_t effective_kernel) {
    switch (config.padding_mode) {
        case StreamingConv1dPaddingMode::StreamingSame:
            return {effective_kernel - config.stride, 0};
        case StreamingConv1dPaddingMode::StrictCausal:
            return {effective_kernel - 1, 0};
        case StreamingConv1dPaddingMode::Explicit:
            return {config.explicit_left, config.explicit_right};
    }
    throw std::runtime_error("StreamingConv1dModule unknown padding mode");
}

}

DepthwiseConv1dModule::DepthwiseConv1dModule(DepthwiseConv1dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("DepthwiseConv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("DepthwiseConv1d stride and dilation must be positive");
    }
}

core::TensorValue DepthwiseConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DepthwiseConv1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.channels, 1, config_.kernel_size}),
        "weight");
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "DepthwiseConv1dModule");
    auto input_4d = core::reshape_tensor(
        ctx,
        input_contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, 1, input.shape.dims[2]}));
    auto weight_4d = core::reshape_tensor(
        ctx,
        weight_contiguous,
        core::TensorShape::from_dims({config_.channels, 1, 1, config_.kernel_size}));
    // This 2D depthwise lowering greatly improves performance but may affect parity.
    auto output_4d = DepthwiseConv2dModule({
        config_.channels,
        1,
        config_.kernel_size,
        1,
        config_.stride,
        0,
        config_.padding,
        1,
        config_.dilation,
        config_.use_bias,
    }).build(ctx, input_4d, {weight_4d, weights.bias});
    return core::reshape_tensor(
        ctx,
        output_4d,
        core::TensorShape::from_dims({
            input.shape.dims[0],
            config_.channels,
            depthwise_conv1d_output_frames(config_, input.shape.dims[2]),
        }));
}

PointwiseConv1dModule::PointwiseConv1dModule(PointwiseConv1dConfig config) : config_(config) {}

core::TensorValue PointwiseConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PointwiseConv1dWeights & weights) const {
    if (config_.quant) {
        core::validate_rank_between(input, 3, 3, "input");
        core::validate_shape(
            input,
            core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
            "input");
        core::TensorValue weight = weights.weight;
        if (weight.shape.rank == 3) {
            core::validate_shape(
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels, 1}),
                "weight");
            weight = core::reshape_tensor(
                ctx,
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels}));
        } else {
            core::validate_shape(
                weight,
                core::TensorShape::from_dims({config_.out_channels, config_.in_channels}),
                "weight");
        }
        auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
        x = LinearModule({config_.in_channels, config_.out_channels, config_.use_bias}).build(ctx, x, {weight, weights.bias});
        return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    }
    return Conv1dModule({config_.in_channels, config_.out_channels, 1, 1, 0, 1, config_.use_bias}).build(ctx, input, weights);
}

StreamingConv1dModule::StreamingConv1dModule(StreamingConv1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("StreamingConv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("StreamingConv1d stride and dilation must be positive");
    }
    if (config_.explicit_left < 0 || config_.explicit_right < 0) {
        throw std::runtime_error("StreamingConv1d explicit padding must be non-negative");
    }
}

core::TensorValue StreamingConv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const StreamingConv1dWeights & weights) const {
    const int64_t effective_kernel = (config_.kernel_size - 1) * config_.dilation + 1;
    const auto [left_pad, right_pad] = streaming_conv1d_padding(config_, effective_kernel);
    if (left_pad < 0 || right_pad < 0) {
        throw std::runtime_error("StreamingConv1dModule computed negative padding");
    }
    if (input.shape.dims[2] <= 0) {
        throw std::runtime_error("StreamingConv1dModule input must have frames");
    }

    auto padded = input;
    if (left_pad > 0) {
        core::TensorValue prefix = config_.pad_mode == StreamingPadMode::Replicate
            ? repeat_first_frame(ctx, input, left_pad)
            : zeros_like_prefix(ctx, input, left_pad);
        padded = ConcatModule({2}).build(ctx, prefix, input);
    }
    if (right_pad > 0) {
        core::TensorValue suffix = config_.pad_mode == StreamingPadMode::Replicate
            ? repeat_last_frame(ctx, input, right_pad)
            : zeros_like_suffix(ctx, input, right_pad);
        padded = ConcatModule({2}).build(ctx, padded, suffix);
    }
    return Conv1dModule({
        config_.in_channels,
        config_.out_channels,
        config_.kernel_size,
        config_.stride,
        0,
        config_.dilation,
        config_.use_bias,
    }).build(ctx, padded, weights);
}

DepthwiseConvTranspose1dModule::DepthwiseConvTranspose1dModule(DepthwiseConvTranspose1dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("DepthwiseConvTranspose1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0) {
        throw std::runtime_error("DepthwiseConvTranspose1d stride must be positive");
    }
}

core::TensorValue DepthwiseConvTranspose1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DepthwiseConvTranspose1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, 3, "input");
    const bool channel_time_layout = input.shape.rank == 2;
    const int64_t frames = channel_time_layout ? input.shape.dims[1] : input.shape.dims[2];
    if (channel_time_layout) {
        core::validate_shape(input, core::TensorShape::from_dims({config_.channels, frames}), "input");
    } else {
        core::validate_shape(input, core::TensorShape::from_dims({1, config_.channels, frames}), "input");
    }
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.channels, 1, 1, config_.kernel_size}),
        "weight");

    const auto x = core::ensure_backend_addressable_layout(ctx, input);
    ggml_tensor * x3 = ggml_reshape_3d(ctx.ggml, x.tensor, 1, frames, config_.channels);
    ggml_tensor * zero3 = ggml_scale(ctx.ggml, x3, 0.0F);
    ggml_tensor * interleaved3 = x3;
    for (int index = 1; index < config_.stride; ++index) {
        interleaved3 = ggml_concat(ctx.ggml, interleaved3, zero3, 0);
    }
    ggml_tensor * interleaved = ggml_reshape_2d(ctx.ggml, interleaved3, frames * config_.stride, config_.channels);
    interleaved = ggml_view_2d(
        ctx.ggml,
        interleaved,
        frames * config_.stride - (config_.stride - 1),
        config_.channels,
        interleaved->nb[1],
        0);
    ggml_tensor * input4 = ggml_reshape_4d(
        ctx.ggml,
        core::has_backend_addressable_layout(interleaved) ? interleaved : ggml_cont(ctx.ggml, interleaved),
        interleaved->ne[0],
        1,
        config_.channels,
        1);
    ggml_tensor * y4 = ggml_conv_2d_dw_direct(
        ctx.ggml,
        weights.weight.tensor,
        input4,
        1,
        1,
        config_.kernel_size - 1,
        0,
        1,
        1);
    y4 = core::has_backend_addressable_layout(y4) ? y4 : ggml_cont(ctx.ggml, y4);
    if (channel_time_layout) {
        auto output = core::wrap_tensor(
            ggml_reshape_2d(ctx.ggml, y4, y4->ne[0], config_.channels),
            core::TensorShape::from_dims({config_.channels, y4->ne[0]}),
            GGML_TYPE_F32);
        if (!weights.bias.has_value()) {
            return output;
        }
        auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
        auto bias_view = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({config_.channels, 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
        return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    auto output = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, y4, y4->ne[0], config_.channels, 1),
        core::TensorShape::from_dims({1, config_.channels, y4->ne[0]}),
        GGML_TYPE_F32);
    return add_bias_bct(ctx, output, config_.channels, weights.bias);
}

}  // namespace engine::modules

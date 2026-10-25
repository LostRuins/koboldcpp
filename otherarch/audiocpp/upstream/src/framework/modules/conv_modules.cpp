#include "engine/framework/modules/conv_modules.h"
#include "tensor_layout_utils.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <stdexcept>
#include <string>

namespace engine::modules {

namespace {

const core::ModulePortSpec kConvInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, false},
    {"bias", core::PortKind::Parameter, true},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModulePortSpec kCausalResBlockInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"conv1.weight", core::PortKind::Parameter, false},
    {"conv1.bias", core::PortKind::Parameter, false},
    {"conv2.weight", core::PortKind::Parameter, false},
    {"conv2.bias", core::PortKind::Parameter, false},
    {"shortcut.weight", core::PortKind::Parameter, true},
    {"shortcut.bias", core::PortKind::Parameter, true},
};

const core::ModuleSchema kConv1dSchema = {
    "Conv1d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 1D convolution to channel-first inputs [batch, channels, frames].",
};

const core::ModuleSchema kConv2dSchema = {
    "Conv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 2D convolution to channel-first inputs [batch, channels, height, width].",
};

const core::ModuleSchema kConv3dSchema = {
    "Conv3d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 3D convolution to ggml-flattened channel-first inputs [batch * channels, depth, height, width].",
};

const core::ModuleSchema kCausalConv2dSchema = {
    "CausalConv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies explicit asymmetric 2D padding followed by Conv2d.",
};

const core::ModuleSchema kSameWidthCausalConv2dSchema = {
    "SameWidthCausalConv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies causal-height and same-width 2D padding followed by Conv2d.",
};

const core::ModuleSchema kPixelNormCausalConv2dResBlockSchema = {
    "PixelNormCausalConv2dResBlock",
    "nn.conv",
    kCausalResBlockInputs,
    7,
    kSingleOutput,
    1,
    "Applies a PixelNorm/Silu causal Conv2d residual block.",
};

const core::ModuleSchema kCausalConv2dUpsampleSchema = {
    "CausalConv2dUpsample",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies nearest 2D upsampling followed by causal Conv2d and temporal crop.",
};

const core::ModuleSchema kDepthwiseConv2dSchema = {
    "DepthwiseConv2d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a depthwise 2D convolution to channel-first inputs [batch, channels, height, width].",
};

const core::ModuleSchema kConvTranspose1dSchema = {
    "ConvTranspose1d",
    "nn.conv",
    kConvInputs,
    3,
    kSingleOutput,
    1,
    "Applies a 1D transposed convolution to channel-first inputs [batch, channels, frames].",
};

core::TensorValue ensure_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

ggml_tensor * round_to_bf16_f32(ggml_context * ctx, ggml_tensor * value) {
    // kcpp: portable replacement for Audio.cpp's unofficial GGML BF16-round op.
    return ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32);
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

core::TensorValue conv_transpose1d_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_F16 && core::uses_host_graph_plan(ctx.backend_type)) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_BF16 && core::uses_host_graph_plan(ctx.backend_type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F16), contiguous.shape, GGML_TYPE_F16);
    }
    if (contiguous.type == GGML_TYPE_F16 || contiguous.type == GGML_TYPE_BF16) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    if (ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string("ConvTranspose1dModule does not support weight type with the current ggml conv-transpose path: ") +
        ggml_type_name(contiguous.type));
}

core::TensorValue depthwise_conv2d_weight(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & weight) {
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weight);
    if (contiguous.type == GGML_TYPE_F32) {
        return contiguous;
    }
    if (contiguous.type == GGML_TYPE_F16 || contiguous.type == GGML_TYPE_BF16 || ggml_is_quantized(contiguous.type)) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, contiguous.tensor, GGML_TYPE_F32), contiguous.shape, GGML_TYPE_F32);
    }
    throw std::runtime_error(
        std::string("DepthwiseConv2dModule does not support weight type with the current ggml depthwise conv path: ") +
        ggml_type_name(contiguous.type));
}

int64_t conv1d_output_frames(const Conv1dConfig & config, int64_t input_frames) {
    return (input_frames + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) / config.stride + 1;
}
bool is_conv1d_pertap_fast_path_eligible(
    const core::ModuleBuildContext & ctx,
    const Conv1dConfig & config,
    const core::TensorValue & input) noexcept {
    return ctx.backend_type == core::BackendType::Metal &&
           config.padding == 0 &&
           config.stride == 1 &&
           input.shape.dims[0] == 1 &&
           input.type == GGML_TYPE_F32 &&
           input.tensor->ne[0] == input.shape.dims[2] &&
           input.tensor->ne[1] == config.in_channels &&
           ggml_is_contiguous(input.tensor);
}

// Per-tap GEMM accumulation on a channel-fast [in_channels, frames] F32 input: one
// contiguous GEMM per kernel tap over shifted column views, accumulated into
// [out_channels, output_frames]. No layout conversion here -- callers at region edges
// transpose; chained callers keep everything channel-fast.
ggml_tensor * conv1d_pertap_gemm_channel_fast(
    core::ModuleBuildContext & ctx,
    ggml_tensor * input_cf,
    ggml_tensor * weight_f32,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t dilation,
    int64_t output_frames) {
    // weight logical [OC, IC, K] -> ggml ne [K, IC, OC]; regroup rows so each tap slice
    // [IC, OC] is a contiguous view: row index = channel + in_channels * tap.
    auto * weight_taps = ggml_reshape_2d(
        ctx.ggml,
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight_f32, 1, 0, 2, 3)),
        in_channels * kernel_size,
        out_channels);
    // accumulate-in-place GEMM only where the tensor-core mm kernel applies
    // (mirrors the Metal supports gate); otherwise keep the mul_mat + add chain
    const bool use_acc = in_channels >= 64 && output_frames > 8;
    ggml_tensor * acc = nullptr;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
        // columns[c, j] = input[tap * dilation + j, c]: contiguous column view of input_cf.
        auto * columns = ggml_view_2d(
            ctx.ggml,
            input_cf,
            in_channels,
            output_frames,
            input_cf->nb[1],
            static_cast<size_t>(tap * dilation) * in_channels * sizeof(float));
        auto * tap_weights = ggml_view_2d(
            ctx.ggml,
            weight_taps,
            in_channels,
            out_channels,
            weight_taps->nb[1],
            static_cast<size_t>(tap) * in_channels * sizeof(float));
        if (acc == nullptr) {
            acc = ggml_mul_mat(ctx.ggml, tap_weights, columns);
        } else {
            // kcpp: ggml_mul_mat_acc is an Audio.cpp-only optimization. The
            // regular add graph has identical values and stays GGML-compatible.
            acc = ggml_add(ctx.ggml, acc, ggml_mul_mat(ctx.ggml, tap_weights, columns));
        }
    }
    return acc;
}

// Metal fast path for stride-1 conv1d on the time-fast [frames, channels] layout used by
// the audio codecs. ggml_conv_1d materializes an im2col matrix whose kernel taps are
// strided gathers in this layout (~200 ms per conv at [569k, 96] on M4); instead, transpose
// the input to channel-fast once, run one contiguous GEMM per kernel tap, and transpose
// the accumulator back (~3-6x faster).
core::TensorValue build_conv1d_pertap_fast_path(
    core::ModuleBuildContext & ctx,
    const Conv1dConfig & config,
    const core::TensorValue & input,
    const core::TensorValue & weight_f32,
    const core::TensorShape & output_shape) {
    // channel-fast copy of the input: [IC, frames]; kernel taps become contiguous columns.
    auto * input_cf = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, input.tensor));
    auto * acc = conv1d_pertap_gemm_channel_fast(
        ctx,
        input_cf,
        weight_f32.tensor,
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        config.dilation,
        output_shape.dims[2]);
    // mul_mat yields [OC, frames]; restore the canonical [frames, OC] orientation.
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, acc)),
        output_shape,
        GGML_TYPE_F32);
}

int64_t conv2d_output_dim(int64_t input, int kernel, int stride, int padding, int dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv3d_output_dim(int64_t input, int kernel, int stride, int padding, int dilation) {
    return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv_transpose1d_output_frames(const ConvTranspose1dConfig & config, int64_t input_frames) {
    return (input_frames - 1) * config.stride - 2 * config.padding + config.dilation * (config.kernel_size - 1) + 1;
}

core::TensorValue add_bias_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t out_channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }

    core::validate_shape(*bias, core::TensorShape::from_dims({out_channels}), "bias");
    const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    const auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, out_channels, 1}));
    const auto bias_expanded = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue add_4d_channel_bias_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & output,
    int64_t channels,
    const std::optional<core::TensorValue> & bias) {
    if (!bias.has_value()) {
        return output;
    }
    core::validate_shape(*bias, core::TensorShape::from_dims({channels}), "bias");
    const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
    const auto bias_view = core::reshape_tensor(ctx, *bias, core::TensorShape::from_dims({1, channels, 1, 1}));
    const auto bias_expanded =
        core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
}

core::TensorValue build_conv_transpose1d_cuda_col2im_path(
    core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config,
    const core::TensorValue & input,
    const ConvTranspose1dWeights & weights,
    const core::TensorShape & output_shape) {
    if (!is_conv_transpose1d_col2im_fast_path_eligible(ctx, config)) {
        throw std::runtime_error("ConvTranspose1dModule CUDA col2im path was requested for an ineligible config");
    }
    const auto input_contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto weight_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weights.weight);
    if (weight_contiguous.type != GGML_TYPE_F32) {
        weight_contiguous = core::wrap_tensor(ggml_cast(ctx.ggml, weight_contiguous.tensor, GGML_TYPE_F32), weight_contiguous.shape, GGML_TYPE_F32);
    }
    auto * weight_perm = ggml_reshape_2d(
        ctx.ggml,
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight_contiguous.tensor, 1, 2, 0, 3)),
        config.in_channels,
        config.kernel_size * config.out_channels);
    ggml_tensor * bias_matrix = nullptr;
    if (config.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("ConvTranspose1dModule col2im fast path requires bias when use_bias is true");
        }
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config.out_channels}), "bias");
        bias_matrix = ggml_reshape_2d(ctx.ggml, weights.bias->tensor, 1, config.out_channels);
    }
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        auto * batch_input = ggml_view_2d(
            ctx.ggml,
            input_contiguous.tensor,
            input_contiguous.tensor->ne[0],
            input_contiguous.tensor->ne[1],
            input_contiguous.tensor->nb[1],
            static_cast<size_t>(batch_index) * input_contiguous.tensor->nb[2]);
        auto * transposed_input = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, batch_input));
        auto * columns = ggml_mul_mat(ctx.ggml, weight_perm, transposed_input);
        auto * batch_output = ggml_col2im_1d(
            ctx.ggml,
            columns,
            config.stride,
            static_cast<int>(config.out_channels),
            config.padding);
        if (bias_matrix != nullptr) {
            batch_output = ggml_add(ctx.ggml, batch_output, bias_matrix);
        }
        auto batch_value = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1),
            core::TensorShape::from_dims({1, config.out_channels, batch_output->ne[0]}),
            GGML_TYPE_F32);
        if (batch_value.shape.dims[2] != output_shape.dims[2]) {
            throw std::runtime_error("ConvTranspose1dModule col2im fast path produced unexpected frame count");
        }
        output = output.valid() ? ConcatModule({0}).build(ctx, output, batch_value) : batch_value;
    }
    return output;
}

core::TensorValue view_batch_matrix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t batch_index,
    int64_t channels,
    int64_t frames) {
    auto * view = ggml_view_2d(
        ctx.ggml,
        input.tensor,
        frames,
        channels,
        input.tensor->nb[1],
        static_cast<size_t>(batch_index) * input.tensor->nb[2]);
    return core::wrap_tensor(view, core::TensorShape::from_dims({channels, frames}), input.type);
}

}  // namespace

bool is_conv_transpose1d_col2im_fast_path_eligible(
    const core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config) noexcept {
    return (core::uses_ggml_cuda_or_hip_backend(ctx.backend_type) ||
            ctx.backend_type == core::BackendType::Metal ||
            ctx.backend_type == core::BackendType::Vulkan) &&
           config.dilation == 1;
}

ggml_tensor * conv1d_pertap_channel_fast(
    core::ModuleBuildContext & ctx,
    const Conv1dWeights & weights,
    ggml_tensor * input_cf,
    const Conv1dConfig & config) {
    if (ctx.ggml == nullptr || input_cf == nullptr) {
        throw std::runtime_error("conv1d_pertap_channel_fast requires a ggml context and an input tensor");
    }
    if (config.padding != 0 || config.stride != 1) {
        throw std::runtime_error("conv1d_pertap_channel_fast requires padding=0 and stride=1");
    }
    if (input_cf->type != GGML_TYPE_F32 || input_cf->ne[0] != config.in_channels ||
        !ggml_is_contiguous(input_cf)) {
        throw std::runtime_error("conv1d_pertap_channel_fast requires contiguous F32 [in_channels, frames] input");
    }
    auto weight = regular_conv_weight(ctx, weights.weight, "conv1d_pertap_channel_fast");
    if (weight.type != GGML_TYPE_F32) {
        weight = core::wrap_tensor(
            ggml_cast(ctx.ggml, weight.tensor, GGML_TYPE_F32), weight.shape, GGML_TYPE_F32);
    }
    const int64_t output_frames = input_cf->ne[1] - config.dilation * (config.kernel_size - 1);
    ggml_tensor * acc = conv1d_pertap_gemm_channel_fast(
        ctx,
        input_cf,
        weight.tensor,
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        config.dilation,
        output_frames);
    if (config.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("conv1d_pertap_channel_fast requires bias when use_bias is true");
        }
        const auto bias = ensure_f32(ctx, *weights.bias);
        core::validate_shape(bias, core::TensorShape::from_dims({config.out_channels}), "bias");
        acc = ggml_add(ctx.ggml, acc, ggml_reshape_2d(ctx.ggml, bias.tensor, config.out_channels, 1));
    }
    return acc;
}

ggml_tensor * conv_transpose1d_col2im_channel_fast(
    core::ModuleBuildContext & ctx,
    const ConvTranspose1dWeights & weights,
    ggml_tensor * input_cf,
    const ConvTranspose1dConfig & config) {
    if (!is_conv_transpose1d_col2im_fast_path_eligible(ctx, config)) {
        throw std::runtime_error("conv_transpose1d_col2im_channel_fast called with an ineligible config");
    }
    if (input_cf == nullptr || input_cf->type != GGML_TYPE_F32 ||
        input_cf->ne[0] != config.in_channels || !ggml_is_contiguous(input_cf)) {
        throw std::runtime_error(
            "conv_transpose1d_col2im_channel_fast requires contiguous F32 [in_channels, frames] input");
    }
    auto weight_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weights.weight);
    if (weight_contiguous.type != GGML_TYPE_F32) {
        weight_contiguous = core::wrap_tensor(
            ggml_cast(ctx.ggml, weight_contiguous.tensor, GGML_TYPE_F32),
            weight_contiguous.shape,
            GGML_TYPE_F32);
    }
    auto * weight_perm = ggml_reshape_2d(
        ctx.ggml,
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, weight_contiguous.tensor, 1, 2, 0, 3)),
        config.in_channels,
        config.kernel_size * config.out_channels);
    auto * columns = ggml_mul_mat(ctx.ggml, weight_perm, input_cf);
    auto * output = ggml_col2im_1d(
        ctx.ggml,
        columns,
        config.stride,
        static_cast<int>(config.out_channels),
        config.padding);
    if (config.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("conv_transpose1d_col2im_channel_fast requires bias when use_bias is true");
        }
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config.out_channels}), "bias");
        output = ggml_add(
            ctx.ggml,
            output,
            ggml_reshape_2d(ctx.ggml, weights.bias->tensor, 1, config.out_channels));
    }
    return output;
}

Conv1dModule::Conv1dModule(Conv1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("Conv1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("Conv1d stride and dilation must be positive");
    }
}

const Conv1dConfig & Conv1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Conv1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Conv1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.out_channels, config_.in_channels, config_.kernel_size}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims(
        {input.shape.dims[0], config_.out_channels, conv1d_output_frames(config_, input.shape.dims[2])});
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    if (config_.bf16_compute) {
        const auto weight_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, weights.weight);
        auto * weight = weight_contiguous.type == GGML_TYPE_BF16 ? weight_contiguous.tensor
            : ggml_cast(ctx.ggml, ensure_f32(ctx, weight_contiguous).tensor, GGML_TYPE_BF16);
        const auto build_batch = [&](const core::TensorValue & batch_input) {
            ggml_tensor * columns;
            if (ctx.backend_type == core::BackendType::Cuda) {
                columns = ggml_im2col(ctx.ggml, weight, batch_input.tensor,
                    config_.stride, 0, config_.padding, 0, config_.dilation, 0, false, GGML_TYPE_BF16);
            } else {
                auto * rounded_input = round_to_bf16_f32(ctx.ggml, batch_input.tensor);
                columns = ggml_im2col(ctx.ggml, weight, rounded_input,
                    config_.stride, 0, config_.padding, 0, config_.dilation, 0, false, GGML_TYPE_F32);
                columns = ggml_cast(ctx.ggml, columns, GGML_TYPE_BF16);
            }
            auto * projected = ggml_mul_mat(ctx.ggml,
                ggml_reshape_2d(ctx.ggml, columns, columns->ne[0], output_shape.dims[2]),
                ggml_reshape_2d(ctx.ggml, weight, columns->ne[0], config_.out_channels));
            projected = round_to_bf16_f32(ctx.ggml, projected);
            if (config_.use_bias) {
                if (!weights.bias) throw std::runtime_error("Conv1d bias is required");
                core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
                auto * bias = round_to_bf16_f32(ctx.ggml, ensure_f32(ctx, *weights.bias).tensor);
                projected = ggml_add(ctx.ggml, projected, ggml_reshape_2d(ctx.ggml, bias, 1, config_.out_channels));
                projected = round_to_bf16_f32(ctx.ggml, projected);
            }
            return core::wrap_tensor(projected,
                core::TensorShape::from_dims({1, config_.out_channels, output_shape.dims[2]}), GGML_TYPE_F32);
        };
        core::TensorValue result;
        for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
            const auto batch_input = view_batch_matrix(ctx, input_contiguous, batch_index,
                config_.in_channels, input.shape.dims[2]);
            const auto batch_output = build_batch(batch_input);
            result = result.tensor ? ConcatModule({0}).build(ctx, result, batch_output) : batch_output;
        }
        return result;
    }
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv1dModule");
    core::TensorValue output;
    if (is_conv1d_pertap_fast_path_eligible(ctx, config_, input) &&
        weight_contiguous.type == GGML_TYPE_F32) {
        output = build_conv1d_pertap_fast_path(ctx, config_, input_contiguous, weight_contiguous, output_shape);
    } else if (input.shape.dims[0] == 1) {
        output = core::wrap_tensor(
            ggml_conv_1d(
                ctx.ggml,
                weight_contiguous.tensor,
                input_contiguous.tensor,
                config_.stride,
                config_.padding,
                config_.dilation),
            output_shape,
            GGML_TYPE_F32);
    } else {
        for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
            const auto matrix_input = view_batch_matrix(
                ctx,
                input_contiguous,
                batch_index,
                config_.in_channels,
                input.shape.dims[2]);
            auto batch_output = core::wrap_tensor(
                ggml_conv_1d(
                    ctx.ggml,
                    weight_contiguous.tensor,
                    matrix_input.tensor,
                    config_.stride,
                    config_.padding,
                    config_.dilation),
                core::TensorShape::from_dims({1, config_.out_channels, output_shape.dims[2]}),
                GGML_TYPE_F32);
            output = output.valid() ? ConcatModule({0}).build(ctx, output, batch_output) : batch_output;
        }
    }
    if (config_.use_bias) {
        output = add_bias_if_needed(ctx, output, config_.out_channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & Conv1dModule::static_schema() noexcept {
    return kConv1dSchema;
}

Conv2dModule::Conv2dModule(Conv2dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_height <= 0 || config_.kernel_width <= 0) {
        throw std::runtime_error("Conv2dConfig dimensions must be positive");
    }
    if (config_.stride_height <= 0 || config_.stride_width <= 0 || config_.dilation_height <= 0 || config_.dilation_width <= 0) {
        throw std::runtime_error("Conv2d stride and dilation must be positive");
    }
    if (config_.cuda_f16_output && !config_.cuda_cudnn) {
        throw std::runtime_error("FP16 Conv2D output requires cuDNN lowering");
    }
}

Conv2dModule::Conv2dModule(Conv2dConfig config, bool cuda_cudnn, bool cuda_f16_output) : Conv2dModule(config) {
    config_.cuda_cudnn = cuda_cudnn;
    config_.cuda_f16_output = cuda_f16_output;
}

Conv2dModule::Conv2dModule(Conv2dConfig config, bool cuda_cudnn, bool cuda_f16_output,
                         bool metal_per_tap) : Conv2dModule(config, cuda_cudnn, cuda_f16_output) {
    config_.metal_per_tap = metal_per_tap;
}

const Conv2dConfig & Conv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Conv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Conv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2], input.shape.dims[3]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.out_channels, config_.in_channels, config_.kernel_height, config_.kernel_width}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims({
        input.shape.dims[0],
        config_.out_channels,
        conv2d_output_dim(
            input.shape.dims[2],
            static_cast<int>(config_.kernel_height),
            config_.stride_height,
            config_.padding_height,
            config_.dilation_height),
        conv2d_output_dim(
            input.shape.dims[3],
            static_cast<int>(config_.kernel_width),
            config_.stride_width,
            config_.padding_width,
            config_.dilation_width),
    });
    const auto height = output_shape.dims[2];
    const auto width = output_shape.dims[3];
    if (config_.metal_per_tap && ctx.backend_type == core::BackendType::Metal &&
        input.type == GGML_TYPE_F32 && weights.weight.type == GGML_TYPE_F32 &&
        config_.in_channels >= 64 && config_.out_channels > 8 && width > 8 && !config_.cuda_cudnn) {
        // Metal's FP32 accumulate kernel needs tap depth >= 64; small outputs favor im2col.
        // Channel-fast tap views avoid materializing im2col. Each tap accumulates
        // into the same FP32 output.
        auto * x = core::ensure_backend_addressable_layout(ctx, input).tensor;
        x = ggml_pad_ext(ctx.ggml, x, config_.padding_width, config_.padding_width, config_.padding_height, config_.padding_height, 0, 0, 0, 0);
        x = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, x, 1, 2, 0, 3));
        auto * w = weights.weight.tensor;
        w = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, w, 1, 2, 0, 3));
        ggml_tensor * result = nullptr;
        for (int64_t ky = 0; ky < config_.kernel_height; ++ky) {
            for (int64_t kx = 0; kx < config_.kernel_width; ++kx) {
                auto * columns = ggml_view_4d(ctx.ggml, x, config_.in_channels, width, height, input.shape.dims[0],
                    x->nb[1] * config_.stride_width, x->nb[2] * config_.stride_height, x->nb[3],
                    ky * config_.dilation_height * x->nb[2] + kx * config_.dilation_width * x->nb[1]);
                auto * kernel = ggml_view_2d(ctx.ggml, w, config_.in_channels, config_.out_channels, w->nb[3],
                    ky * w->nb[2] + kx * w->nb[1]);
                // kcpp: use the portable graph in place of Audio.cpp's fused
                // ggml_mul_mat_acc extension.
                result = result
                    ? ggml_add(ctx.ggml, result, ggml_mul_mat(ctx.ggml, kernel, columns))
                    : ggml_mul_mat(ctx.ggml, kernel, columns);
            }
        }
        result = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, result, 2, 0, 1, 3));
        if (config_.use_bias) {
            if (!weights.bias) throw std::runtime_error("bias is required when Conv2dConfig.use_bias is true");
            core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
            result = ggml_add(ctx.ggml, result,
                ggml_reshape_4d(ctx.ggml, weights.bias->tensor, 1, 1, config_.out_channels, 1));
        }
        return core::wrap_tensor(result, output_shape);
    }
    if (config_.cuda_cudnn) {
        const auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
        const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv2dModule");
        if (ctx.backend_type != core::BackendType::Cuda || weight_contiguous.type != GGML_TYPE_F16) {
            throw std::runtime_error("cuDNN Conv2D requires CUDA and F16 weights");
        }
        auto * f16_input = input_contiguous.type == GGML_TYPE_F16 ? input_contiguous.tensor :
            ggml_cast(ctx.ggml, input_contiguous.tensor, GGML_TYPE_F16);
        auto * f16_output = ggml_conv_2d_direct(ctx.ggml, weight_contiguous.tensor, f16_input,
            config_.stride_width, config_.stride_height, config_.padding_width, config_.padding_height,
            config_.dilation_width, config_.dilation_height);
        // kcpp: Audio.cpp annotates this node with private cuDNN op params and
        // optionally fuses bias. Use the standard direct-conv graph and apply
        // bias explicitly in F32 instead.
        auto output = core::wrap_tensor(
            ggml_cast(ctx.ggml, f16_output, GGML_TYPE_F32), output_shape, GGML_TYPE_F32);
        if (config_.use_bias) {
            if (!weights.bias.has_value()) {
                throw std::runtime_error("bias is required when Conv2dConfig.use_bias is true");
            }
            core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
            const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
            const auto bias_view = core::reshape_tensor(
                ctx, *weights.bias, core::TensorShape::from_dims({1, config_.out_channels, 1, 1}));
            const auto bias_expanded =
                core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
            output = core::wrap_tensor(
                ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
        }
        return output;
    }

    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv2dModule");
    auto output = core::wrap_tensor(
        ggml_conv_2d(
            ctx.ggml,
            weight_contiguous.tensor,
            input_contiguous.tensor,
            config_.stride_width,
            config_.stride_height,
            config_.padding_width,
            config_.padding_height,
            config_.dilation_width,
            config_.dilation_height),
        output_shape,
        GGML_TYPE_F32);
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when Conv2dConfig.use_bias is true");
        }
        const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
        const auto bias_view = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, config_.out_channels, 1, 1}));
        const auto bias_expanded =
            core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

const core::ModuleSchema & Conv2dModule::static_schema() noexcept {
    return kConv2dSchema;
}

Conv3dModule::Conv3dModule(Conv3dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_depth <= 0 || config_.kernel_height <= 0 ||
        config_.kernel_width <= 0) {
        throw std::runtime_error("Conv3dConfig dimensions must be positive");
    }
    if (config_.stride_depth <= 0 || config_.stride_height <= 0 || config_.stride_width <= 0 ||
        config_.dilation_depth <= 0 || config_.dilation_height <= 0 || config_.dilation_width <= 0) {
        throw std::runtime_error("Conv3d stride and dilation must be positive");
    }
}

const Conv3dConfig & Conv3dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Conv3dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Conv3dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv3dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    if (input.shape.dims[0] % config_.in_channels != 0) {
        throw std::runtime_error("Conv3dModule input first dimension must be batch * in_channels");
    }
    const int64_t batch = input.shape.dims[0] / config_.in_channels;
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims(
            {config_.out_channels * config_.in_channels, config_.kernel_depth, config_.kernel_height, config_.kernel_width}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims({
        batch * config_.out_channels,
        conv3d_output_dim(
            input.shape.dims[1],
            static_cast<int>(config_.kernel_depth),
            config_.stride_depth,
            config_.padding_depth,
            config_.dilation_depth),
        conv3d_output_dim(
            input.shape.dims[2],
            static_cast<int>(config_.kernel_height),
            config_.stride_height,
            config_.padding_height,
            config_.dilation_height),
        conv3d_output_dim(
            input.shape.dims[3],
            static_cast<int>(config_.kernel_width),
            config_.stride_width,
            config_.padding_width,
            config_.dilation_width),
    });
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = regular_conv_weight(ctx, weights.weight, "Conv3dModule");
    auto output = core::wrap_tensor(
        ggml_conv_3d(
            ctx.ggml,
            weight_contiguous.tensor,
            input_contiguous.tensor,
            config_.in_channels,
            config_.stride_width,
            config_.stride_height,
            config_.stride_depth,
            config_.padding_width,
            config_.padding_height,
            config_.padding_depth,
            config_.dilation_width,
            config_.dilation_height,
            config_.dilation_depth),
        output_shape,
        GGML_TYPE_F32);
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when Conv3dConfig.use_bias is true");
        }
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config_.out_channels}), "bias");
        const auto output_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, output);
        const auto bias_view =
            core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({config_.out_channels, 1, 1, 1}));
        const auto bias_expanded =
            core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, output_contiguous.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output_contiguous.tensor, bias_expanded.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

const core::ModuleSchema & Conv3dModule::static_schema() noexcept {
    return kConv3dSchema;
}

CausalConv2dModule::CausalConv2dModule(CausalConv2dConfig config) : config_(config) {
    if (config_.pad_left < 0 || config_.pad_right < 0 || config_.pad_top < 0 || config_.pad_bottom < 0) {
        throw std::runtime_error("CausalConv2d padding must be non-negative");
    }
    if (config_.conv.padding_height != 0 || config_.conv.padding_width != 0) {
        throw std::runtime_error("CausalConv2d uses explicit Pad2d padding; Conv2dConfig padding must be zero");
    }
}

const CausalConv2dConfig & CausalConv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & CausalConv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue CausalConv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    auto padded = Pad2dModule({
        config_.pad_left,
        config_.pad_right,
        config_.pad_top,
        config_.pad_bottom,
    }).build(ctx, input);
    return Conv2dModule(config_.conv).build(ctx, padded, weights);
}

const core::ModuleSchema & CausalConv2dModule::static_schema() noexcept {
    return kCausalConv2dSchema;
}

SameWidthCausalConv2dModule::SameWidthCausalConv2dModule(SameWidthCausalConv2dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("SameWidthCausalConv2dConfig dimensions must be positive");
    }
    if (config_.kernel_size % 2 == 0) {
        throw std::runtime_error("SameWidthCausalConv2dConfig.kernel_size must be odd");
    }
}

const SameWidthCausalConv2dConfig & SameWidthCausalConv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SameWidthCausalConv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SameWidthCausalConv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    const int64_t pad_h = config_.kernel_size - 1;
    const int64_t pad_w = config_.kernel_size - 1;
    return CausalConv2dModule({
        {
            config_.in_channels,
            config_.out_channels,
            config_.kernel_size,
            config_.kernel_size,
            1,
            1,
            0,
            0,
            1,
            1,
            config_.use_bias,
        },
        pad_w / 2,
        pad_w - pad_w / 2,
        pad_h,
        0,
    }).build(ctx, input, weights);
}

const core::ModuleSchema & SameWidthCausalConv2dModule::static_schema() noexcept {
    return kSameWidthCausalConv2dSchema;
}

PixelNormCausalConv2dResBlockModule::PixelNormCausalConv2dResBlockModule(
    PixelNormCausalConv2dResBlockConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("PixelNormCausalConv2dResBlockConfig dimensions must be positive");
    }
    if (config_.kernel_size % 2 == 0) {
        throw std::runtime_error("PixelNormCausalConv2dResBlockConfig.kernel_size must be odd");
    }
    if (config_.pixel_norm_eps <= 0.0F) {
        throw std::runtime_error("PixelNormCausalConv2dResBlockConfig.pixel_norm_eps must be positive");
    }
}

const PixelNormCausalConv2dResBlockConfig & PixelNormCausalConv2dResBlockModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & PixelNormCausalConv2dResBlockModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue PixelNormCausalConv2dResBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PixelNormCausalConv2dResBlockWeights & weights) const {
    auto h = PixelNormModule({config_.pixel_norm_axis, config_.pixel_norm_eps}).build(ctx, input);
    h = SiluModule{}.build(ctx, h);
    h = SameWidthCausalConv2dModule({config_.in_channels, config_.out_channels, config_.kernel_size, true})
            .build(ctx, h, weights.conv1);
    h = PixelNormModule({config_.pixel_norm_axis, config_.pixel_norm_eps}).build(ctx, h);
    h = SiluModule{}.build(ctx, h);
    h = SameWidthCausalConv2dModule({config_.out_channels, config_.out_channels, config_.kernel_size, true})
            .build(ctx, h, weights.conv2);
    auto residual = core::ensure_backend_addressable_layout(ctx, input);
    if (weights.shortcut.has_value()) {
        residual = SameWidthCausalConv2dModule({config_.in_channels, config_.out_channels, 1, true})
                       .build(ctx, residual, *weights.shortcut);
    }
    return AddModule{}.build(ctx, residual, h);
}

const core::ModuleSchema & PixelNormCausalConv2dResBlockModule::static_schema() noexcept {
    return kPixelNormCausalConv2dResBlockSchema;
}

CausalConv2dUpsampleModule::CausalConv2dUpsampleModule(CausalConv2dUpsampleConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_size <= 0 || config_.scale_height <= 0 || config_.scale_width <= 0) {
        throw std::runtime_error("CausalConv2dUpsampleConfig dimensions must be positive");
    }
    if (config_.kernel_size % 2 == 0) {
        throw std::runtime_error("CausalConv2dUpsampleConfig.kernel_size must be odd");
    }
    if (config_.crop_top < 0 || config_.crop_bottom < 0) {
        throw std::runtime_error("CausalConv2dUpsampleConfig crop must be non-negative");
    }
}

const CausalConv2dUpsampleConfig & CausalConv2dUpsampleModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & CausalConv2dUpsampleModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue CausalConv2dUpsampleModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    auto x = NearestUpsample2dModule({
        input.shape.dims[2] * config_.scale_height,
        input.shape.dims[3] * config_.scale_width,
    }).build(ctx, input);
    x = SameWidthCausalConv2dModule({config_.channels, config_.channels, config_.kernel_size, true}).build(ctx, x, weights);
    const int64_t height = x.shape.dims[2] - config_.crop_top - config_.crop_bottom;
    if (height <= 0) {
        throw std::runtime_error("CausalConv2dUpsample crop removes all rows");
    }
    return core::ensure_backend_addressable_layout(
        ctx,
        SliceModule({2, config_.crop_top, height}).build(ctx, x));
}

const core::ModuleSchema & CausalConv2dUpsampleModule::static_schema() noexcept {
    return kCausalConv2dUpsampleSchema;
}

DepthwiseConv2dModule::DepthwiseConv2dModule(DepthwiseConv2dConfig config) : config_(config) {
    if (config_.channels <= 0 || config_.kernel_height <= 0 || config_.kernel_width <= 0) {
        throw std::runtime_error("DepthwiseConv2dConfig dimensions must be positive");
    }
    if (config_.stride_height <= 0 || config_.stride_width <= 0 || config_.dilation_height <= 0 || config_.dilation_width <= 0) {
        throw std::runtime_error("DepthwiseConv2d stride and dilation must be positive");
    }
}

const DepthwiseConv2dConfig & DepthwiseConv2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & DepthwiseConv2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue DepthwiseConv2dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, input.shape.dims[2], input.shape.dims[3]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.channels, 1, config_.kernel_height, config_.kernel_width}),
        "weight");

    const auto output_shape = core::TensorShape::from_dims({
        input.shape.dims[0],
        config_.channels,
        conv2d_output_dim(
            input.shape.dims[2],
            static_cast<int>(config_.kernel_height),
            config_.stride_height,
            config_.padding_height,
            config_.dilation_height),
        conv2d_output_dim(
            input.shape.dims[3],
            static_cast<int>(config_.kernel_width),
            config_.stride_width,
            config_.padding_width,
            config_.dilation_width),
    });
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = depthwise_conv2d_weight(ctx, weights.weight);
    auto output = core::wrap_tensor(
        ggml_conv_2d_dw_direct(
            ctx.ggml,
            weight_contiguous.tensor,
            input_contiguous.tensor,
            config_.stride_width,
            config_.stride_height,
            config_.padding_width,
            config_.padding_height,
            config_.dilation_width,
            config_.dilation_height),
        output_shape,
        GGML_TYPE_F32);
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when DepthwiseConv2dConfig.use_bias is true");
        }
        output = add_4d_channel_bias_if_needed(ctx, output, config_.channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & DepthwiseConv2dModule::static_schema() noexcept {
    return kDepthwiseConv2dSchema;
}

ConvTranspose1dModule::ConvTranspose1dModule(ConvTranspose1dConfig config) : config_(config) {
    if (config_.in_channels <= 0 || config_.out_channels <= 0 || config_.kernel_size <= 0) {
        throw std::runtime_error("ConvTranspose1dConfig dimensions must be positive");
    }
    if (config_.stride <= 0 || config_.dilation <= 0) {
        throw std::runtime_error("ConvTranspose1d stride and dilation must be positive");
    }
}

const ConvTranspose1dConfig & ConvTranspose1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ConvTranspose1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ConvTranspose1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvTranspose1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.in_channels, input.shape.dims[2]}),
        "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.in_channels, config_.out_channels, config_.kernel_size}),
        "weight");
    const auto output_shape = core::TensorShape::from_dims(
        {input.shape.dims[0], config_.out_channels, conv_transpose1d_output_frames(config_, input.shape.dims[2])});
    if (is_conv_transpose1d_col2im_fast_path_eligible(ctx, config_)) {
        return build_conv_transpose1d_cuda_col2im_path(ctx, config_, input, weights, output_shape);
    }
    const auto input_contiguous = ensure_f32(ctx, tensor_layout::ensure_contiguous_layout_if_needed(ctx, input));
    const auto weight_contiguous = conv_transpose1d_weight(ctx, weights.weight);
    core::TensorValue output;
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        const auto matrix_input = view_batch_matrix(
            ctx,
            input_contiguous,
            batch_index,
            config_.in_channels,
            input.shape.dims[2]);
        auto batch_output = core::wrap_tensor(
            ggml_conv_transpose_1d(
                ctx.ggml,
                weight_contiguous.tensor,
                matrix_input.tensor,
                config_.stride,
                config_.padding,
                config_.dilation),
            core::TensorShape::from_dims({1, config_.out_channels, output_shape.dims[2]}),
            GGML_TYPE_F32);
        if (!output.valid()) {
            output = batch_output;
        } else {
            output = ConcatModule({0}).build(ctx, output, batch_output);
        }
    }
    if (config_.use_bias) {
        output = add_bias_if_needed(ctx, output, config_.out_channels, weights.bias);
    }
    return output;
}

const core::ModuleSchema & ConvTranspose1dModule::static_schema() noexcept {
    return kConvTranspose1dSchema;
}

}  // namespace engine::modules

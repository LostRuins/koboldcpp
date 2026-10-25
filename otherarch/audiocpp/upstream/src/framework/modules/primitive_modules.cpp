#include "engine/framework/modules/primitive_modules.h"
#include "tensor_layout_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kBinaryInputs[] = {
    {"lhs", core::PortKind::Activation, false},
    {"rhs", core::PortKind::Activation, false},
};

const core::ModulePortSpec kRopeInterleavedPairsInputs[] = {
    {"even", core::PortKind::Activation, false},
    {"odd", core::PortKind::Activation, false},
    {"cos", core::PortKind::Activation, false},
    {"sin", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleInput[] = {
    {"input", core::PortKind::Activation, false},
};

const core::ModulePortSpec kResidualInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"residual", core::PortKind::Activation, false},
};

const core::ModulePortSpec kBypassInputs[] = {
    {"original", core::PortKind::Activation, false},
    {"transformed", core::PortKind::Activation, false},
    {"channel_scale", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kAddSchema = {
    "Add",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Elementwise addition for tensors with the same logical shape.",
};

const core::ModuleSchema kMatMulSchema = {
    "MatMul",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Performs batched matrix multiplication with logical shapes [..., M, K] x [..., K, N] -> [..., M, N].",
};

const core::ModuleSchema kMulSchema = {
    "Mul",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Elementwise multiplication for tensors with the same logical shape.",
};

const core::ModulePortSpec kTimeMask4dInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"mask", core::PortKind::Activation, false},
};

const core::ModuleSchema kRopeInterleavedPairsSchema = {
    "RopeInterleavedPairs",
    "tensor.position",
    kRopeInterleavedPairsInputs,
    4,
    kSingleOutput,
    1,
    "Applies rotary position embedding to split even/odd pairs and returns interleaved pairs.",
};

const core::ModuleSchema kTimeMask4dSchema = {
    "TimeMask4d",
    "tensor.mask",
    kTimeMask4dInputs,
    2,
    kSingleOutput,
    1,
    "Applies a [batch, time] keep mask to a [batch, channels, time, features] tensor.",
};

const core::ModuleSchema kResidualAddSchema = {
    "ResidualAdd",
    "nn.primitive",
    kResidualInputs,
    2,
    kSingleOutput,
    1,
    "Adds a residual tensor to the main input.",
};

const core::ModuleSchema kScaledBypassSchema = {
    "ScaledBypass",
    "nn.primitive",
    kBypassInputs,
    3,
    kSingleOutput,
    1,
    "Applies scaled residual bypass: original + (transformed - original) * channel_scale.",
};

const core::ModuleSchema kReduceMeanSchema = {
    "ReduceMean",
    "tensor.primitive",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Reduces one logical axis by averaging values.",
};

const core::ModuleSchema kReduceSumSchema = {
    "ReduceSum",
    "tensor.primitive",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Reduces one logical axis by summing values.",
};

void validate_same_shape(const core::TensorValue & lhs, const core::TensorValue & rhs, const char * op_name) {
    core::validate_shape(rhs, lhs.shape, op_name);
}

core::TensorShape make_last_dim_broadcast_shape(const core::TensorShape & target) {
    core::TensorShape shape = {};
    shape.rank = target.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 1] = target.last_dim();
    return shape;
}

core::TensorShape transpose_last_two(const core::TensorShape & shape) {
    core::TensorShape result = shape;
    const size_t last = result.rank - 1;
    const size_t second_last = result.rank - 2;
    std::swap(result.dims[second_last], result.dims[last]);
    return result;
}

core::TensorValue transpose_last_two(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    const size_t last = value.shape.rank - 1;
    const size_t second_last = value.shape.rank - 2;
    axes[second_last] = static_cast<int>(last);
    axes[last] = static_cast<int>(second_last);

    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_logical_axis = 0; out_logical_axis < value.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        const int out_ggml_axis = static_cast<int>(value.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = core::logical_axis_to_ggml_axis(value.shape.rank, in_logical_axis);
    }

    return core::wrap_tensor(
        ggml_permute(ctx.ggml, value.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        transpose_last_two(value.shape),
        value.type);
}

int normalize_reduce_axis(int axis, size_t rank) {
    if (rank == 0) {
        throw std::runtime_error("Reduce input rank must be positive");
    }
    const int normalized = axis < 0 ? static_cast<int>(rank) + axis : axis;
    if (normalized < 0 || normalized >= static_cast<int>(rank)) {
        throw std::runtime_error("Reduce axis out of range");
    }
    return normalized;
}

core::TensorShape reduced_shape(const core::TensorShape & shape, int axis) {
    core::TensorShape out = shape;
    out.dims[static_cast<size_t>(axis)] = 1;
    return out;
}

template <typename Fn>
core::TensorValue reduce_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ReduceConfig & config,
    Fn fn) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const int axis = normalize_reduce_axis(config.axis, input.shape.rank);
    if (axis == static_cast<int>(input.shape.rank - 1)) {
        const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
        return core::wrap_tensor(fn(ctx.ggml, contiguous.tensor), reduced_shape(input.shape, axis), GGML_TYPE_F32);
    }

    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    std::swap(axes[static_cast<size_t>(axis)], axes[input.shape.rank - 1]);
    const auto transposed = TransposeModule({axes, input.shape.rank}).build(ctx, input);
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, transposed);
    auto reduced = core::wrap_tensor(
        fn(ctx.ggml, contiguous.tensor),
        reduced_shape(transposed.shape, static_cast<int>(transposed.shape.rank - 1)),
        GGML_TYPE_F32);
    return TransposeModule({axes, reduced.shape.rank}).build(ctx, reduced);
}

}  // namespace

const core::ModuleSchema & MatMulModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue MatMulModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(lhs, 2, core::kMaxTensorRank, "lhs");
    core::validate_rank_between(rhs, lhs.shape.rank, lhs.shape.rank, "rhs");

    const size_t rank = lhs.shape.rank;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (lhs.shape.dims[i] != rhs.shape.dims[i]) {
            throw std::runtime_error("MatMul batch dimensions must match");
        }
    }
    if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2]) {
        throw std::runtime_error("MatMul inner dimensions must match");
    }

    auto rhs_transposed = transpose_last_two(ctx, rhs);
    rhs_transposed = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs_transposed);
    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    ggml_tensor * output = ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor);
    return core::wrap_tensor(output, output_shape, GGML_TYPE_F32);
}

const core::ModuleSchema & MatMulModule::static_schema() noexcept {
    return kMatMulSchema;
}

const core::ModuleSchema & AddModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue AddModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (allow_broadcast_) {
        if (!ggml_can_repeat(rhs.tensor, lhs.tensor)) {
            throw std::runtime_error("Add rhs shape cannot broadcast to lhs");
        }
    } else {
        validate_same_shape(lhs, rhs, "Add rhs");
    }
    if (preserve_input_type_ && lhs.type != rhs.type) {
        throw std::runtime_error("Add input types must match when preserving the result type");
    }
    const auto lhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, lhs);
    const auto rhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs);
    return core::wrap_tensor(ggml_add(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor), lhs.shape,
        preserve_input_type_ ? lhs.type : GGML_TYPE_F32);
}

const core::ModuleSchema & AddModule::static_schema() noexcept {
    return kAddSchema;
}

const core::ModuleSchema & MulModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue MulModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(lhs, rhs, "Mul rhs");
    const auto lhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, lhs);
    const auto rhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs);
    return core::wrap_tensor(ggml_mul(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor), lhs.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & MulModule::static_schema() noexcept {
    return kMulSchema;
}

const core::ModuleSchema & RopeInterleavedPairsModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue RopeInterleavedPairsModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & even,
    const core::TensorValue & odd,
    const core::TensorValue & cos,
    const core::TensorValue & sin) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(even, odd, "RopeInterleavedPairs");
    validate_same_shape(even, cos, "RopeInterleavedPairs");
    validate_same_shape(even, sin, "RopeInterleavedPairs");
    if (even.type != odd.type || cos.type != GGML_TYPE_F32 || sin.type != GGML_TYPE_F32) {
        throw std::runtime_error("RopeInterleavedPairs expects matching even/odd type and F32 cos/sin");
    }

    core::TensorShape output_shape = even.shape;
    output_shape.dims[output_shape.rank - 1] = 2;
    // kcpp: expand Audio.cpp's fused rope-interleave op into portable GGML.
    // Inputs are singleton rows; concatenating the two rotated components on
    // ggml dimension 0 recreates the interleaved pair layout.
    auto * even_f32 = even.type == GGML_TYPE_F32
        ? even.tensor : ggml_cast(ctx.ggml, even.tensor, GGML_TYPE_F32);
    auto * odd_f32 = odd.type == GGML_TYPE_F32
        ? odd.tensor : ggml_cast(ctx.ggml, odd.tensor, GGML_TYPE_F32);
    auto * rotated_even = ggml_sub(ctx.ggml,
        ggml_mul(ctx.ggml, even_f32, cos.tensor),
        ggml_mul(ctx.ggml, odd_f32, sin.tensor));
    auto * rotated_odd = ggml_add(ctx.ggml,
        ggml_mul(ctx.ggml, odd_f32, cos.tensor),
        ggml_mul(ctx.ggml, even_f32, sin.tensor));
    auto * result = ggml_concat(ctx.ggml, rotated_even, rotated_odd, 0);
    if (even.type != GGML_TYPE_F32) {
        result = ggml_cast(ctx.ggml, result, even.type);
    }
    return core::wrap_tensor(
        result,
        output_shape,
        even.type);
}

const core::ModuleSchema & RopeInterleavedPairsModule::static_schema() noexcept {
    return kRopeInterleavedPairsSchema;
}

const core::ModuleSchema & TimeMask4dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TimeMask4dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        mask,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[2]}),
        "time_mask");
    const auto mask_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    const auto mask_4d = core::reshape_tensor(
        ctx,
        mask_f32,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2], 1}));
    const auto broadcast = core::wrap_tensor(ggml_repeat(ctx.ggml, mask_4d.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, broadcast.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & TimeMask4dModule::static_schema() noexcept {
    return kTimeMask4dSchema;
}

const core::ModuleSchema & ResidualAddModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ResidualAddModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & residual) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(input, residual, "ResidualAdd residual");
    const auto input_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, input);
    const auto residual_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, residual);
    return core::wrap_tensor(ggml_add(ctx.ggml, input_contiguous.tensor, residual_contiguous.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & ResidualAddModule::static_schema() noexcept {
    return kResidualAddSchema;
}

const core::ModuleSchema & ScaledBypassModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ScaledBypassModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & original,
    const core::TensorValue & transformed,
    const core::TensorValue & channel_scale) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(original, 1, core::kMaxTensorRank, "original");
    validate_same_shape(original, transformed, "ScaledBypass transformed");
    core::validate_shape(channel_scale, core::TensorShape::from_dims({original.shape.last_dim()}), "channel_scale");

    const auto original_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, original);
    const auto transformed_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, transformed);
    const auto scale_view = core::reshape_tensor(ctx, channel_scale, make_last_dim_broadcast_shape(original.shape));
    const auto repeated_scale = core::wrap_tensor(ggml_repeat(ctx.ggml, scale_view.tensor, original_contiguous.tensor), original.shape, GGML_TYPE_F32);
    const auto delta = core::wrap_tensor(ggml_sub(ctx.ggml, transformed_contiguous.tensor, original_contiguous.tensor), original.shape, GGML_TYPE_F32);
    const auto scaled_delta = core::wrap_tensor(ggml_mul(ctx.ggml, delta.tensor, repeated_scale.tensor), original.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, original_contiguous.tensor, scaled_delta.tensor), original.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & ScaledBypassModule::static_schema() noexcept {
    return kScaledBypassSchema;
}

ReduceMeanModule::ReduceMeanModule(ReduceConfig config) : config_(config) {
}

const ReduceConfig & ReduceMeanModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReduceMeanModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReduceMeanModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return reduce_axis(ctx, input, config_, ggml_mean);
}

const core::ModuleSchema & ReduceMeanModule::static_schema() noexcept {
    return kReduceMeanSchema;
}

ReduceSumModule::ReduceSumModule(ReduceConfig config) : config_(config) {
}

const ReduceConfig & ReduceSumModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReduceSumModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReduceSumModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return reduce_axis(ctx, input, config_, ggml_sum_rows);
}

const core::ModuleSchema & ReduceSumModule::static_schema() noexcept {
    return kReduceSumSchema;
}

ScaleBiasModule::ScaleBiasModule(ScaleBiasConfig config) : config_(config) {}

const core::ModuleSchema & ScaleBiasModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ScaleBiasModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    auto * output = ggml_scale_bias(ctx.ggml, input.tensor, config_.scale, config_.bias);
    return core::wrap_tensor(output, input.shape, output->type);
}

const core::ModuleSchema & ScaleBiasModule::static_schema() noexcept {
    static const core::ModuleSchema schema{"ScaleBias", "tensor.primitive", kSingleInput, 1, kSingleOutput, 1,
        "Applies a scalar scale and bias to every element."};
    return schema;
}

WeightedStatisticsModule::WeightedStatisticsModule(WeightedStatisticsConfig config) : config_(config) {
    if (!std::isfinite(config_.variance_min) || config_.variance_min < 0 ||
        std::isnan(config_.variance_max) || config_.variance_max < config_.variance_min) {
        throw std::runtime_error("WeightedStatistics variance bounds must be nonnegative and ordered");
    }
}

const core::ModuleSchema & WeightedStatisticsModule::schema() const noexcept {
    return static_schema();
}

WeightedStatisticsOutputs WeightedStatisticsModule::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & normalized_weights) const {
    core::validate_shape(normalized_weights, input.shape, "normalized_weights");
    const auto mean = ReduceSumModule({config_.axis}).build(ctx, MulModule().build(ctx, normalized_weights, input));
    auto second = MulModule().build(ctx, input, input);
    second = ReduceSumModule({config_.axis}).build(ctx, MulModule().build(ctx, normalized_weights, second));
    auto * variance = ggml_sub(ctx.ggml, second.tensor, ggml_sqr(ctx.ggml, mean.tensor));
    variance = ggml_clamp(ctx.ggml, variance, config_.variance_min, config_.variance_max);
    return {mean, core::wrap_tensor(ggml_sqrt(ctx.ggml, variance), mean.shape, GGML_TYPE_F32)};
}

const core::ModuleSchema & WeightedStatisticsModule::static_schema() noexcept {
    static const core::ModulePortSpec inputs[] = {
        {"input", core::PortKind::Activation, false},
        {"normalized_weights", core::PortKind::Activation, false},
    };
    static const core::ModulePortSpec outputs[] = {
        {"mean", core::PortKind::Activation, false},
        {"stddev", core::PortKind::Activation, false},
    };
    static const core::ModuleSchema schema{"WeightedStatistics", "tensor.pooling", inputs, 2, outputs, 2,
        "Reduces normalized weights into a mean and clamped standard deviation, retaining the reduced axis."};
    return schema;
}

}  // namespace engine::modules

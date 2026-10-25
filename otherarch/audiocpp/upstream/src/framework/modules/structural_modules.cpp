#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kSingleInput[] = {
    {"input", core::PortKind::Activation, false},
};

const core::ModulePortSpec kBinaryInput[] = {
    {"lhs", core::PortKind::Activation, false},
    {"rhs", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kReshapeSchema = {
    "Reshape",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Reinterprets a tensor with a different logical shape but the same number of elements.",
};

const core::ModuleSchema kTransposeSchema = {
    "Transpose",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Permutes logical tensor axes without changing values.",
};

const core::ModuleSchema kSliceSchema = {
    "Slice",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Extracts a contiguous range along a logical axis.",
};

const core::ModuleSchema kConcatSchema = {
    "Concat",
    "tensor.layout",
    kBinaryInput,
    2,
    kSingleOutput,
    1,
    "Concatenates two tensors along a logical axis.",
};

const core::ModuleSchema kRepeatSchema = {
    "Repeat",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Repeats a tensor into a larger target shape.",
};

const core::ModulePortSpec kInputMaskInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"mask", core::PortKind::Activation, false},
};

const core::ModuleSchema kPaddingSchema = {
    "Padding",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Pads the frame axis with zeros up to a fixed target length.",
};

const core::ModuleSchema kReflectPad1dSchema = {
    "ReflectPad1d",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Pads the last logical axis using reflected boundary values.",
};

const core::ModuleSchema kPad2dSchema = {
    "Pad2d",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Applies asymmetric zero padding to the last two logical axes of a 4-D tensor.",
};

const core::ModuleSchema kInterpolate1dSchema = {
    "Interpolate1d",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Interpolates the last logical axis of channel-first tensors to a fixed frame count.",
};

const core::ModuleSchema kNearestUpsample2dSchema = {
    "NearestUpsample2d",
    "tensor.layout",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Upsamples the last two logical axes of a 4-D tensor with nearest-neighbor interpolation.",
};

const core::ModuleSchema kMaskingSchema = {
    "Masking",
    "tensor.layout",
    kInputMaskInputs,
    2,
    kSingleOutput,
    1,
    "Applies a frame mask to zero out inactive time steps.",
};

core::TensorShape permute_shape(const core::TensorShape & input_shape, const TransposeConfig & config) {
    if (config.rank != input_shape.rank) {
        throw std::runtime_error("Transpose config rank does not match input rank");
    }

    core::TensorShape output = {};
    output.rank = input_shape.rank;
    std::array<bool, core::kMaxTensorRank> seen = {false, false, false, false};
    for (size_t i = 0; i < input_shape.rank; ++i) {
        const int axis = config.axes[i];
        if (axis < 0 || axis >= static_cast<int>(input_shape.rank)) {
            throw std::runtime_error("Transpose axis out of range");
        }
        if (seen[axis]) {
            throw std::runtime_error("Transpose axes must form a permutation");
        }
        seen[axis] = true;
        output.dims[i] = input_shape.dims[axis];
    }
    return output;
}

size_t checked_slice_offset_bytes(const core::TensorValue & input, int ggml_axis, int64_t start) {
    if (start < 0 || start >= input.tensor->ne[ggml_axis]) {
        throw std::runtime_error("Slice start is out of bounds");
    }
    return static_cast<size_t>(start) * input.tensor->nb[ggml_axis];
}

ggml_tensor * make_view_with_shape(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorShape & output_shape,
    size_t offset_bytes) {
    const auto dims = core::to_ggml_dims(output_shape);
    switch (output_shape.rank) {
        case 1:
            return ggml_view_1d(ctx.ggml, input.tensor, dims[0], offset_bytes);
        case 2:
            return ggml_view_2d(ctx.ggml, input.tensor, dims[0], dims[1], input.tensor->nb[1], offset_bytes);
        case 3:
            return ggml_view_3d(
                ctx.ggml,
                input.tensor,
                dims[0],
                dims[1],
                dims[2],
                input.tensor->nb[1],
                input.tensor->nb[2],
                offset_bytes);
        case 4:
            return ggml_view_4d(
                ctx.ggml,
                input.tensor,
                dims[0],
                dims[1],
                dims[2],
                dims[3],
                input.tensor->nb[1],
                input.tensor->nb[2],
                input.tensor->nb[3],
                offset_bytes);
        default:
            throw std::runtime_error("Unsupported view rank");
    }
}

void validate_concat_shapes(const core::TensorShape & lhs, const core::TensorShape & rhs, int axis) {
    if (lhs.rank != rhs.rank) {
        throw std::runtime_error("Concat inputs must have the same rank");
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        if (lhs.dims[i] != rhs.dims[i]) {
            throw std::runtime_error("Concat non-axis dimensions must match");
        }
    }
}

}  // namespace

ReshapeModule::ReshapeModule(ReshapeConfig config) : config_(config) {
}

const ReshapeConfig & ReshapeModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReshapeModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReshapeModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return core::reshape_tensor(ctx, input, config_.output_shape);
}

const core::ModuleSchema & ReshapeModule::static_schema() noexcept {
    return kReshapeSchema;
}

TransposeModule::TransposeModule(TransposeConfig config) : config_(config) {
}

const TransposeConfig & TransposeModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & TransposeModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TransposeModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");

    const auto output_shape = permute_shape(input.shape, config_);
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = config_.axes[out_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        const int in_ggml_axis = core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
        ggml_axes[static_cast<size_t>(in_ggml_axis)] = out_ggml_axis;
    }

    return core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

const core::ModuleSchema & TransposeModule::static_schema() noexcept {
    return kTransposeSchema;
}

SliceModule::SliceModule(SliceConfig config) : config_(config) {
}

const SliceConfig & SliceModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SliceModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SliceModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    if (config_.axis < 0 || config_.axis >= static_cast<int>(input.shape.rank)) {
        throw std::runtime_error("Slice axis is out of bounds");
    }
    if (config_.length <= 0) {
        throw std::runtime_error("Slice length must be positive");
    }
    if (config_.start < 0 || config_.start + config_.length > input.shape.dims[config_.axis]) {
        throw std::runtime_error("Slice range is out of bounds");
    }

    core::TensorShape output_shape = input.shape;
    output_shape.dims[config_.axis] = config_.length;
    const int ggml_axis = core::logical_axis_to_ggml_axis(input.shape.rank, config_.axis);
    const size_t offset_bytes = checked_slice_offset_bytes(input, ggml_axis, config_.start);
    return core::wrap_tensor(make_view_with_shape(ctx, input, output_shape, offset_bytes), output_shape, input.type);
}

const core::ModuleSchema & SliceModule::static_schema() noexcept {
    return kSliceSchema;
}

ConcatModule::ConcatModule(ConcatConfig config) : config_(config) {
}

const ConcatConfig & ConcatModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ConcatModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ConcatModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (config_.axis < 0 || config_.axis >= static_cast<int>(lhs.shape.rank)) {
        throw std::runtime_error("Concat axis is out of bounds");
    }
    validate_concat_shapes(lhs.shape, rhs.shape, config_.axis);

    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[config_.axis] += rhs.shape.dims[config_.axis];
    const int ggml_axis = core::logical_axis_to_ggml_axis(lhs.shape.rank, config_.axis);
    auto * raw = ggml_concat(ctx.ggml, lhs.tensor, rhs.tensor, ggml_axis);
    // kcpp: cuda_contiguous_4d_lowering is an Audio.cpp-private performance
    // hint. The normal concat graph is portable and semantically identical.
    return core::wrap_tensor(raw, output_shape, lhs.type);
}

const core::ModuleSchema & ConcatModule::static_schema() noexcept {
    return kConcatSchema;
}

RepeatModule::RepeatModule(RepeatConfig config) : config_(config) {
}

const RepeatConfig & RepeatModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & RepeatModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue RepeatModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (config_.output_shape.rank != input.shape.rank) {
        throw std::runtime_error("Repeat output rank must match input rank");
    }
    for (size_t i = 0; i < input.shape.rank; ++i) {
        if (config_.output_shape.dims[i] % input.shape.dims[i] != 0) {
            throw std::runtime_error("Repeat output dimensions must be multiples of input dimensions");
        }
    }

    const auto dims = core::to_ggml_dims(config_.output_shape);
    return core::wrap_tensor(
        ggml_repeat_4d(ctx.ggml, input.tensor, dims[0], dims[1], dims[2], dims[3]),
        config_.output_shape,
        input.type);
}

const core::ModuleSchema & RepeatModule::static_schema() noexcept {
    return kRepeatSchema;
}

PaddingModule::PaddingModule(PaddingConfig config) : config_(config) {
    if (config_.target_frames <= 0) {
        throw std::runtime_error("PaddingConfig.target_frames must be positive");
    }
}

const PaddingConfig & PaddingModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & PaddingModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue PaddingModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    const int64_t frames = input.shape.dims[1];
    if (config_.target_frames < frames) {
        throw std::runtime_error("Padding target_frames must be >= input frames");
    }

    auto * padded = ggml_pad(ctx.ggml, input.tensor, 0, static_cast<int>(config_.target_frames - frames), 0, 0);
    return core::wrap_tensor(
        padded,
        core::TensorShape::from_dims({input.shape.dims[0], config_.target_frames, input.shape.dims[2]}),
        GGML_TYPE_F32);
}

const core::ModuleSchema & PaddingModule::static_schema() noexcept {
    return kPaddingSchema;
}

ReflectPad1dModule::ReflectPad1dModule(ReflectPad1dConfig config) : config_(config) {
    if (config_.left < 0 || config_.right < 0) {
        throw std::runtime_error("ReflectPad1d padding must be non-negative");
    }
}

const ReflectPad1dConfig & ReflectPad1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReflectPad1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReflectPad1dModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const int64_t frames = input.shape.last_dim();
    if (config_.left >= frames || config_.right >= frames) {
        throw std::runtime_error("ReflectPad1d padding must be smaller than the input frame count");
    }
    auto output_shape = input.shape;
    output_shape.dims[output_shape.rank - 1] += config_.left + config_.right;
    return core::wrap_tensor(
        ggml_pad_reflect_1d(ctx.ggml, input.tensor, static_cast<int>(config_.left), static_cast<int>(config_.right)),
        output_shape,
        GGML_TYPE_F32);
}

const core::ModuleSchema & ReflectPad1dModule::static_schema() noexcept {
    return kReflectPad1dSchema;
}

Pad2dModule::Pad2dModule(Pad2dConfig config) : config_(config) {
    if (config_.left < 0 || config_.right < 0 || config_.top < 0 || config_.bottom < 0) {
        throw std::runtime_error("Pad2d padding must be non-negative");
    }
}

const Pad2dConfig & Pad2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Pad2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Pad2dModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto output_shape = input.shape;
    output_shape.dims[2] += config_.top + config_.bottom;
    output_shape.dims[3] += config_.left + config_.right;
    auto * raw = ggml_pad_ext(
        ctx.ggml,
        contiguous.tensor,
        static_cast<int>(config_.left),
        static_cast<int>(config_.right),
        static_cast<int>(config_.top),
        static_cast<int>(config_.bottom),
        0,
        0,
        0,
        0);
    return core::wrap_tensor(raw, output_shape, input.type);
}

const core::ModuleSchema & Pad2dModule::static_schema() noexcept {
    return kPad2dSchema;
}

Interpolate1dModule::Interpolate1dModule(Interpolate1dConfig config) : config_(config) {
    if (config_.output_frames <= 0) {
        throw std::runtime_error("Interpolate1dConfig.output_frames must be positive");
    }
}

const Interpolate1dConfig & Interpolate1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & Interpolate1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue Interpolate1dModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, 3, "input");

    ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST;
    switch (config_.mode) {
        case Interpolate1dMode::Nearest:
            mode = GGML_SCALE_MODE_NEAREST;
            break;
        case Interpolate1dMode::Linear:
            mode = GGML_SCALE_MODE_BILINEAR;
            break;
        default:
            throw std::runtime_error("Unsupported Interpolate1d mode");
    }

    auto output_shape = input.shape;
    output_shape.dims[output_shape.rank - 1] = config_.output_frames;
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    ggml_tensor * raw = ggml_interpolate(
        ctx.ggml,
        contiguous.tensor,
        config_.output_frames,
        contiguous.tensor->ne[1],
        contiguous.tensor->ne[2],
        contiguous.tensor->ne[3],
        mode);
    return core::wrap_tensor(raw, output_shape, GGML_TYPE_F32);
}

const core::ModuleSchema & Interpolate1dModule::static_schema() noexcept {
    return kInterpolate1dSchema;
}

NearestUpsample2dModule::NearestUpsample2dModule(NearestUpsample2dConfig config) : config_(config) {
    if (config_.output_height <= 0 || config_.output_width <= 0) {
        throw std::runtime_error("NearestUpsample2dConfig output dimensions must be positive");
    }
}

const NearestUpsample2dConfig & NearestUpsample2dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & NearestUpsample2dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue NearestUpsample2dModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    auto output_shape = input.shape;
    output_shape.dims[2] = config_.output_height;
    output_shape.dims[3] = config_.output_width;
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto * raw = ggml_interpolate(
        ctx.ggml,
        contiguous.tensor,
        config_.output_width,
        config_.output_height,
        contiguous.tensor->ne[2],
        contiguous.tensor->ne[3],
        GGML_SCALE_MODE_NEAREST);
    return core::wrap_tensor(raw, output_shape, GGML_TYPE_F32);
}

const core::ModuleSchema & NearestUpsample2dModule::static_schema() noexcept {
    return kNearestUpsample2dSchema;
}

const core::ModuleSchema & MaskingModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue MaskingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_shape(mask, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1]}), "mask");
    if (mask.type != GGML_TYPE_I32) {
        throw std::runtime_error("Masking mask must be GGML_TYPE_I32");
    }

    auto mask_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    auto mask_3d = core::reshape_tensor(ctx, mask_f32, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 1}));
    auto mask_broadcast = core::wrap_tensor(ggml_repeat(ctx.ggml, mask_3d.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, mask_broadcast.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & MaskingModule::static_schema() noexcept {
    return kMaskingSchema;
}

}  // namespace engine::modules

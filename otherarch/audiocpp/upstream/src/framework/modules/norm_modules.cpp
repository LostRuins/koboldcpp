#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "tensor_layout_utils.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kNormInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, true},
    {"bias", core::PortKind::Parameter, true},
};

const core::ModulePortSpec kNormOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kLayerNormSchema = {
    "LayerNorm",
    "nn.normalization",
    kNormInputs,
    3,
    kNormOutputs,
    1,
    "Applies layer normalization over the last logical dimension.",
};

const core::ModuleSchema kRmsNormSchema = {
    "RMSNorm",
    "nn.normalization",
    kNormInputs,
    3,
    kNormOutputs,
    1,
    "Applies RMS normalization over the last logical dimension.",
};

const core::ModuleSchema kGemmaRmsNormSchema = {
    "GemmaRMSNorm",
    "nn.normalization",
    kNormInputs,
    3,
    kNormOutputs,
    1,
    "Applies Gemma RMS normalization over the last logical dimension using x * (1 + weight).",
};

const core::ModuleSchema kGroupNormSchema = {
    "GroupNorm",
    "nn.normalization",
    kNormInputs,
    3,
    kNormOutputs,
    1,
    "Applies group normalization over channel-first rank-3 or rank-4 tensors.",
};

const core::ModuleSchema kPixelNormSchema = {
    "PixelNorm",
    "nn.normalization",
    kNormInputs,
    1,
    kNormOutputs,
    1,
    "Normalizes an input by RMS energy along one logical axis.",
};

const core::ModulePortSpec kBiasNormInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"bias", core::PortKind::Parameter, false},
};

const core::ModuleSchema kBiasNormSchema = {
    "BiasNorm",
    "nn.normalization",
    kBiasNormInputs,
    2,
    kNormOutputs,
    1,
    "Applies Zipformer BiasNorm using bias-centered variance and the original input numerator.",
};

const core::ModulePortSpec kAdaptiveInstanceNorm1dInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"gamma", core::PortKind::Parameter, false},
    {"beta", core::PortKind::Parameter, false},
};

const core::ModuleSchema kAdaptiveInstanceNorm1dSchema = {
    "AdaptiveInstanceNorm1d",
    "nn.normalization",
    kAdaptiveInstanceNorm1dInputs,
    3,
    kNormOutputs,
    1,
    "Applies instance normalization over the last logical dimension and then per-channel affine modulation.",
};

const core::ModulePortSpec kBatchNorm1dEvalInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"scale", core::PortKind::Parameter, false},
    {"bias", core::PortKind::Parameter, true},
};

const core::ModuleSchema kBatchNorm1dEvalSchema = {
    "BatchNorm1dEval",
    "nn.normalization",
    kBatchNorm1dEvalInputs,
    3,
    kNormOutputs,
    1,
    "Applies precomputed 1D batch-normalization eval scale and bias to channel-first tensors.",
};

const core::ModuleSchema kBatchNorm2dEvalSchema = {
    "BatchNorm2dEval",
    "nn.normalization",
    kBatchNorm1dEvalInputs,
    3,
    kNormOutputs,
    1,
    "Applies precomputed 2D batch-normalization eval scale and bias to [batch, channels, height, width] tensors.",
};

core::TensorValue ensure_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

bool same_shape(const core::TensorShape & lhs, const core::TensorShape & rhs) {
    if (lhs.rank != rhs.rank) {
        return false;
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            return false;
        }
    }
    return true;
}

core::TensorValue apply_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & normalized,
    const NormConfig & config,
    const NormWeights & weights) {
    core::TensorValue result = normalized;
    if (config.use_weight) {
        if (!weights.weight.has_value()) {
            throw std::runtime_error("weight is required when NormConfig.use_weight is true");
        }
        core::validate_shape(*weights.weight, core::TensorShape::from_dims({config.hidden_size}), "weight");
        const auto weight = config.cuda_f16_output
            ? ggml_cast(ctx.ggml, weights.weight->tensor, GGML_TYPE_F16)
            : ensure_f32(ctx, *weights.weight).tensor;
        result = core::wrap_tensor(ggml_mul(ctx.ggml, result.tensor, weight), result.shape,
            config.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32);
    }
    if (config.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("bias is required when NormConfig.use_bias is true");
        }
        core::validate_shape(*weights.bias, core::TensorShape::from_dims({config.hidden_size}), "bias");
        const auto bias = config.cuda_f16_output
            ? ggml_cast(ctx.ggml, weights.bias->tensor, GGML_TYPE_F16)
            : ensure_f32(ctx, *weights.bias).tensor;
        result = core::wrap_tensor(ggml_add(ctx.ggml, result.tensor, bias), result.shape,
            config.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32);
    }
    return result;
}

void validate_norm_config(const NormConfig & config) {
    if (config.hidden_size <= 0) {
        throw std::runtime_error("NormConfig.hidden_size must be positive");
    }
    if (!(config.eps > 0.0f)) {
        throw std::runtime_error("NormConfig.eps must be positive");
    }
}

template <typename NormFn>
core::TensorValue build_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormConfig & config,
    const NormWeights & weights,
    NormFn fn) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config.hidden_size, "input");
    if (config.cuda_f16_output && (ctx.backend_type != core::BackendType::Cuda || input.type != GGML_TYPE_F16)) {
        throw std::runtime_error("FP16 normalization requires CUDA and FP16 input");
    }
    const auto norm_input = config.preserve_input_layout
        ? input
        : tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
    auto * normalized_tensor = fn(ctx.ggml, norm_input.tensor, config.eps);
    if (config.cuda_f16_output) {
        if (normalized_tensor->op != GGML_OP_NORM) {
            throw std::runtime_error("FP16 normalization is supported only for LayerNorm");
        }
        // kcpp: represent Audio.cpp's private CUDA F16-output hint with a
        // standard cast after normalization.
        normalized_tensor = ggml_cast(ctx.ggml, normalized_tensor, GGML_TYPE_F16);
    }
    core::TensorValue normalized = core::wrap_tensor(normalized_tensor, input.shape,
        config.cuda_f16_output ? GGML_TYPE_F16 : GGML_TYPE_F32);
    if (config.normalized_round_type.has_value()) {
        // kcpp: the generic cast round-trip also covers BF16 without the
        // Audio.cpp-only ggml_round_bf16 operation.
        normalized.tensor = ggml_cast(ctx.ggml,
            ggml_cast(ctx.ggml, normalized.tensor, *config.normalized_round_type), GGML_TYPE_F32);
        normalized.type = GGML_TYPE_F32;
    }
    return apply_affine(ctx, normalized, config, weights);
}

core::TensorShape make_channel_broadcast_shape(const core::TensorShape & input, int64_t hidden_size) {
    core::TensorShape shape = {};
    shape.rank = input.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 2] = hidden_size;
    return shape;
}

core::TensorShape make_last_dim_broadcast_shape(const core::TensorShape & input, int64_t hidden_size) {
    core::TensorShape shape = {};
    shape.rank = input.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 1] = hidden_size;
    return shape;
}

core::TensorValue repeat_channels(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like,
    int64_t channels,
    const char * name) {
    core::validate_shape(value, core::TensorShape::from_dims({channels}), name);
    const auto reshaped = core::reshape_tensor(ctx, ensure_f32(ctx, value), make_channel_broadcast_shape(like.shape, channels));
    return core::wrap_tensor(ggml_repeat(ctx.ggml, reshaped.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue repeat_channels_2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like,
    int64_t channels,
    const char * name) {
    core::validate_shape(value, core::TensorShape::from_dims({channels}), name);
    auto reshaped = core::reshape_tensor(
        ctx,
        ensure_f32(ctx, value),
        core::TensorShape::from_dims({1, channels, 1, 1}));
    return core::wrap_tensor(ggml_repeat(ctx.ggml, reshaped.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

}  // namespace

LayerNormModule::LayerNormModule(NormConfig config) : config_(config) {
    validate_norm_config(config_);
}

const core::ModuleSchema & LayerNormModule::schema() const noexcept {
    return static_schema();
}

const NormConfig & LayerNormModule::config() const noexcept {
    return config_;
}

core::TensorValue LayerNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights) const {
    return build_norm(ctx, input, config_, weights, ggml_norm);
}

const core::ModuleSchema & LayerNormModule::static_schema() noexcept {
    return kLayerNormSchema;
}

RMSNormModule::RMSNormModule(NormConfig config) : config_(config) {
    validate_norm_config(config_);
}

const core::ModuleSchema & RMSNormModule::schema() const noexcept {
    return static_schema();
}

const NormConfig & RMSNormModule::config() const noexcept {
    return config_;
}

core::TensorValue RMSNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights) const {
    return build_norm(ctx, input, config_, weights, ggml_rms_norm);
}

const core::ModuleSchema & RMSNormModule::static_schema() noexcept {
    return kRmsNormSchema;
}

GemmaRMSNormModule::GemmaRMSNormModule(NormConfig config) : config_(config) {
    validate_norm_config(config_);
    if (!config_.use_weight || config_.use_bias) {
        throw std::runtime_error("GemmaRMSNormConfig requires use_weight=true and use_bias=false");
    }
}

const core::ModuleSchema & GemmaRMSNormModule::schema() const noexcept {
    return static_schema();
}

const NormConfig & GemmaRMSNormModule::config() const noexcept {
    return config_;
}

core::TensorValue GemmaRMSNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (!weights.weight.has_value()) {
        throw std::runtime_error("GemmaRMSNorm weight is required");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    core::validate_shape(*weights.weight, core::TensorShape::from_dims({config_.hidden_size}), "weight");

    const auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
    auto normalized = core::wrap_tensor(
        ggml_rms_norm(ctx.ggml, input_contiguous.tensor, config_.eps),
        input.shape,
        GGML_TYPE_F32);
    const auto one_plus_weight = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, ensure_f32(ctx, *weights.weight).tensor, 1.0F, 1.0F),
        weights.weight->shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, one_plus_weight.tensor),
        input.shape,
        GGML_TYPE_F32);
}

const core::ModuleSchema & GemmaRMSNormModule::static_schema() noexcept {
    return kGemmaRmsNormSchema;
}

GroupNormModule::GroupNormModule(GroupNormConfig config) : config_(config) {
    if (config_.channels <= 0) {
        throw std::runtime_error("GroupNormConfig.channels must be positive");
    }
    if (config_.groups <= 0) {
        throw std::runtime_error("GroupNormConfig.groups must be positive");
    }
    if (config_.channels % config_.groups != 0) {
        throw std::runtime_error("GroupNormConfig.channels must be divisible by groups");
    }
    if (!(config_.eps > 0.0F)) {
        throw std::runtime_error("GroupNormConfig.eps must be positive");
    }
}

const core::ModuleSchema & GroupNormModule::schema() const noexcept {
    return static_schema();
}

const GroupNormConfig & GroupNormModule::config() const noexcept {
    return config_;
}

core::TensorValue GroupNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 4, "input");
    if (input.shape.dims[1] != config_.channels) {
        throw std::runtime_error("GroupNorm input channel count mismatch");
    }

    if (config_.cuda_f16_output && (ctx.backend_type != core::BackendType::Cuda ||
            input.shape.rank != 4 || input.type != GGML_TYPE_F16)) {
        throw std::runtime_error("FP16 GroupNorm requires a rank-4 CUDA FP16 input");
    }
    // kcpp: the generic graph below replaces Audio.cpp's private fused CUDA
    // affine GroupNorm path. Cast its completed result when F16 was requested.
    auto x = core::ensure_backend_addressable_layout(ctx, ensure_f32(ctx, input));
    if (config_.metal_row_norm && ctx.backend_type == core::BackendType::Metal) {
        // Each batch/group is one independent row with the same mean, variance and epsilon.
        const auto rows = x.shape.dims[0] * config_.groups;
        auto * rows_tensor = ggml_reshape_2d(ctx.ggml, x.tensor, x.shape.num_elements() / rows, rows);
        auto * normalized = ggml_norm(ctx.ggml, rows_tensor, config_.eps);
        x = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, normalized,
            x.tensor->ne[0], x.tensor->ne[1], x.tensor->ne[2], x.tensor->ne[3]), x.shape, GGML_TYPE_F32);
    } else if (x.shape.rank == 3) {
        auto x_4d = ggml_reshape_4d(
            ctx.ggml,
            x.tensor,
            x.shape.dims[2],
            1,
            config_.channels,
            x.shape.dims[0]);
        auto normalized_4d = ggml_group_norm(ctx.ggml, x_4d, config_.groups, config_.eps);
        x = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, normalized_4d, x.shape.dims[2], config_.channels, x.shape.dims[0]),
            x.shape,
            GGML_TYPE_F32);
    } else {
        x = core::wrap_tensor(
            ggml_group_norm(ctx.ggml, x.tensor, config_.groups, config_.eps),
            x.shape,
            GGML_TYPE_F32);
    }
    if (config_.use_weight) {
        if (!weights.weight.has_value()) {
            throw std::runtime_error("GroupNorm weight is required");
        }
        auto weight_rep = x.shape.rank == 3
            ? repeat_channels(ctx, *weights.weight, x, config_.channels, "weight")
            : repeat_channels_2d(ctx, *weights.weight, x, config_.channels, "weight");
        x = core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, weight_rep.tensor), x.shape, GGML_TYPE_F32);
    }
    if (config_.use_bias) {
        if (!weights.bias.has_value()) {
            throw std::runtime_error("GroupNorm bias is required");
        }
        auto bias_rep = x.shape.rank == 3
            ? repeat_channels(ctx, *weights.bias, x, config_.channels, "bias")
            : repeat_channels_2d(ctx, *weights.bias, x, config_.channels, "bias");
        x = core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, bias_rep.tensor), x.shape, GGML_TYPE_F32);
    }
    if (config_.cuda_f16_output) {
        return core::wrap_tensor(
            ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
    }
    return x;
}

const core::ModuleSchema & GroupNormModule::static_schema() noexcept {
    return kGroupNormSchema;
}

PixelNormModule::PixelNormModule(PixelNormConfig config) : config_(config) {
    if (!(config_.eps > 0.0F)) {
        throw std::runtime_error("PixelNormConfig.eps must be positive");
    }
}

const core::ModuleSchema & PixelNormModule::schema() const noexcept {
    return static_schema();
}

const PixelNormConfig & PixelNormModule::config() const noexcept {
    return config_;
}

core::TensorValue PixelNormModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const auto x = core::ensure_backend_addressable_layout(ctx, input);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    auto mean = ReduceMeanModule({config_.axis}).build(ctx, squared);
    auto denom = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, core::ensure_backend_addressable_layout(ctx, mean).tensor, 1.0F, config_.eps)),
        mean.shape,
        GGML_TYPE_F32);
    auto denom_full = core::wrap_tensor(ggml_repeat(ctx.ggml, denom.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, denom_full.tensor), x.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & PixelNormModule::static_schema() noexcept {
    return kPixelNormSchema;
}

BiasNormModule::BiasNormModule(BiasNormConfig config) : config_(config) {
    if (config_.hidden_size <= 0) {
        throw std::runtime_error("BiasNormConfig.hidden_size must be positive");
    }
}

const core::ModuleSchema & BiasNormModule::schema() const noexcept {
    return static_schema();
}

const BiasNormConfig & BiasNormModule::config() const noexcept {
    return config_;
}

core::TensorValue BiasNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const BiasNormWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    core::validate_shape(weights.bias, core::TensorShape::from_dims({config_.hidden_size}), "bias");

    const auto bias_view = core::reshape_tensor(ctx, ensure_f32(ctx, weights.bias), make_last_dim_broadcast_shape(input.shape, config_.hidden_size));
    const auto bias_repeated = RepeatModule({input.shape}).build(ctx, bias_view);
    const auto input_contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const auto bias_contiguous = core::ensure_backend_addressable_layout(ctx, bias_repeated);
    const auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_contiguous.tensor, bias_contiguous.tensor), input.shape, GGML_TYPE_F32);
    const auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, centered.tensor), input.shape, GGML_TYPE_F32);
    const auto mean = ReduceMeanModule({static_cast<int>(input.shape.rank - 1)}).build(ctx, squared);
    const auto denominator = core::wrap_tensor(ggml_sqrt(ctx.ggml, mean.tensor), mean.shape, GGML_TYPE_F32);
    const auto denominator_repeated = RepeatModule({input.shape}).build(ctx, denominator);
    const auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, input.tensor, denominator_repeated.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_scale(ctx.ggml, normalized.tensor, std::exp(weights.log_scale)), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & BiasNormModule::static_schema() noexcept {
    return kBiasNormSchema;
}

AdaptiveInstanceNorm1dModule::AdaptiveInstanceNorm1dModule(AdaptiveInstanceNorm1dConfig config) : config_(config) {
    if (config_.hidden_size <= 0) {
        throw std::runtime_error("AdaptiveInstanceNorm1dConfig.hidden_size must be positive");
    }
    if (!(config_.eps > 0.0f)) {
        throw std::runtime_error("AdaptiveInstanceNorm1dConfig.eps must be positive");
    }
}

const core::ModuleSchema & AdaptiveInstanceNorm1dModule::schema() const noexcept {
    return static_schema();
}

const AdaptiveInstanceNorm1dConfig & AdaptiveInstanceNorm1dModule::config() const noexcept {
    return config_;
}

core::TensorValue AdaptiveInstanceNorm1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AdaptiveInstanceNorm1dWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    if (input.shape.dims[input.shape.rank - 2] != config_.hidden_size) {
        throw std::runtime_error("AdaptiveInstanceNorm1d input hidden dimension mismatch");
    }

    const auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
    const auto normalized = core::wrap_tensor(ggml_norm(ctx.ggml, input_contiguous.tensor, config_.eps), input.shape, GGML_TYPE_F32);
    core::TensorValue gamma_broadcast = {};
    core::TensorValue beta_broadcast = {};
    if (same_shape(weights.gamma.shape, input.shape) && same_shape(weights.beta.shape, input.shape)) {
        gamma_broadcast = ensure_f32(ctx, weights.gamma);
        beta_broadcast = ensure_f32(ctx, weights.beta);
    } else {
        core::validate_shape(weights.gamma, core::TensorShape::from_dims({config_.hidden_size}), "gamma");
        core::validate_shape(weights.beta, core::TensorShape::from_dims({config_.hidden_size}), "beta");
        const auto broadcast_shape = make_channel_broadcast_shape(input.shape, config_.hidden_size);
        gamma_broadcast = core::reshape_tensor(ctx, ensure_f32(ctx, weights.gamma), broadcast_shape);
        beta_broadcast = core::reshape_tensor(ctx, ensure_f32(ctx, weights.beta), broadcast_shape);
    }
    const auto scaled = core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, gamma_broadcast.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, beta_broadcast.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & AdaptiveInstanceNorm1dModule::static_schema() noexcept {
    return kAdaptiveInstanceNorm1dSchema;
}

BatchNorm1dEvalModule::BatchNorm1dEvalModule(BatchNorm1dEvalConfig config) : config_(config) {
    if (config_.channels <= 0) {
        throw std::runtime_error("BatchNorm1dEvalConfig.channels must be positive");
    }
}

const core::ModuleSchema & BatchNorm1dEvalModule::schema() const noexcept {
    return static_schema();
}

const BatchNorm1dEvalConfig & BatchNorm1dEvalModule::config() const noexcept {
    return config_;
}

core::TensorValue BatchNorm1dEvalModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const BatchNorm1dEvalWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "input");
    if (input.shape.dims[1] != config_.channels) {
        throw std::runtime_error("BatchNorm1dEval input channel count mismatch");
    }

    const auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, ensure_f32(ctx, input));
    const auto scale = repeat_channels(ctx, weights.scale, input_contiguous, config_.channels, "scale");
    const auto bias = repeat_channels(ctx, weights.bias, input_contiguous, config_.channels, "bias");
    const auto scaled = core::wrap_tensor(
        ggml_mul(ctx.ggml, input_contiguous.tensor, scale.tensor),
        input.shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bias.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & BatchNorm1dEvalModule::static_schema() noexcept {
    return kBatchNorm1dEvalSchema;
}

BatchNorm2dEvalModule::BatchNorm2dEvalModule(BatchNorm2dEvalConfig config) : config_(config) {
    if (config_.channels <= 0) {
        throw std::runtime_error("BatchNorm2dEvalConfig.channels must be positive");
    }
}

const core::ModuleSchema & BatchNorm2dEvalModule::schema() const noexcept {
    return static_schema();
}

const BatchNorm2dEvalConfig & BatchNorm2dEvalModule::config() const noexcept {
    return config_;
}

core::TensorValue BatchNorm2dEvalModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const BatchNorm2dEvalWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    if (input.shape.dims[1] != config_.channels) {
        throw std::runtime_error("BatchNorm2dEval input channel count mismatch");
    }

    const auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, ensure_f32(ctx, input));
    const auto scale = repeat_channels_2d(ctx, weights.scale, input_contiguous, config_.channels, "scale");
    const auto bias = repeat_channels_2d(ctx, weights.bias, input_contiguous, config_.channels, "bias");
    const auto scaled = core::wrap_tensor(
        ggml_mul(ctx.ggml, input_contiguous.tensor, scale.tensor),
        input.shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bias.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & BatchNorm2dEvalModule::static_schema() noexcept {
    return kBatchNorm2dEvalSchema;
}

}  // namespace engine::modules

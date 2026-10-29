#include "engine/framework/modules/conditioning_modules.h"

#include "tensor_layout_utils.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <stdexcept>

namespace engine::modules {
namespace {

void require_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    return core::wrap_tensor(ggml_repeat(ctx.ggml, value.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue expand_conditioning_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    if (value.shape.rank == like.shape.rank) {
        return value;
    }
    if (value.shape.rank + 1 != like.shape.rank) {
        throw std::runtime_error("Conditioning tensor rank is not broadcast-compatible with target");
    }
    core::TensorShape reshaped = {};
    reshaped.rank = like.shape.rank;
    reshaped.dims[0] = value.shape.dims[0];
    reshaped.dims[1] = 1;
    for (size_t i = 1; i < value.shape.rank; ++i) {
        reshaped.dims[i + 1] = value.shape.dims[i];
    }
    auto contiguous_value = tensor_layout::ensure_contiguous_layout_if_needed(ctx, value);
    auto expanded = core::reshape_tensor(ctx, contiguous_value, reshaped);
    return repeat_like(ctx, expanded, like);
}

core::TensorValue slice_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t offset,
    int64_t length) {
    return SliceModule({static_cast<int>(input.shape.rank - 1), offset, length}).build(ctx, input);
}

core::TensorValue add_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(ggml_add(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue mul(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(ggml_mul(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue scale_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float scale) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, scale), input.shape, GGML_TYPE_F32);
}

core::TensorValue apply_adaln(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & shift,
    const core::TensorValue & scale,
    float eps,
    const std::optional<NormWeights> & norm_weights) {
    core::TensorValue normalized;
    if (norm_weights.has_value()) {
        normalized = LayerNormModule({input.shape.dims[input.shape.rank - 1], eps, true, true}).build(ctx, input, *norm_weights);
    } else {
        auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
        normalized = core::wrap_tensor(ggml_norm(ctx.ggml, input_contiguous.tensor, eps), input.shape, GGML_TYPE_F32);
    }
    normalized = tensor_layout::ensure_contiguous_layout_if_needed(ctx, normalized);
    auto scaled = mul(ctx, normalized, scale);
    auto modulated = add_tensors(ctx, normalized, scaled);
    return add_tensors(ctx, modulated, shift);
}

core::TensorValue concat_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return ConcatModule({static_cast<int>(lhs.shape.rank - 1)}).build(ctx, lhs, rhs);
}

core::TensorValue variance_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & alpha,
    float eps) {
    core::validate_shape(alpha, core::TensorShape::from_dims({input.shape.dims[input.shape.rank - 1]}), "alpha");

    core::TensorShape reduced_shape = input.shape;
    reduced_shape.dims[reduced_shape.rank - 1] = 1;

    auto mean = core::wrap_tensor(ggml_mean(ctx.ggml, input.tensor), reduced_shape, GGML_TYPE_F32);
    auto mean_rep = repeat_like(ctx, mean, input);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, mean_rep.tensor), input.shape, GGML_TYPE_F32);
    auto squared = core::wrap_tensor(ggml_mul(ctx.ggml, centered.tensor, centered.tensor), input.shape, GGML_TYPE_F32);
    auto variance = core::wrap_tensor(ggml_mean(ctx.ggml, squared.tensor), reduced_shape, GGML_TYPE_F32);

    const int64_t hidden = input.shape.dims[input.shape.rank - 1];
    if (hidden > 1) {
        const float correction = static_cast<float>(hidden) / static_cast<float>(hidden - 1);
        variance = core::wrap_tensor(ggml_scale(ctx.ggml, variance.tensor, correction), variance.shape, GGML_TYPE_F32);
    }
    auto alpha_first = SliceModule({0, 0, 1}).build(ctx, alpha);
    auto ones = core::wrap_tensor(ggml_div(ctx.ggml, alpha_first.tensor, alpha_first.tensor), alpha_first.shape, GGML_TYPE_F32);
    auto eps_tensor = scale_tensor(ctx, repeat_like(ctx, ones, variance), eps);
    variance = add_tensors(ctx, variance, eps_tensor);
    auto sqrt_var = core::wrap_tensor(ggml_sqrt(ctx.ggml, variance.tensor), variance.shape, GGML_TYPE_F32);
    auto alpha_rep = repeat_like(ctx, alpha, input);
    auto sqrt_rep = repeat_like(ctx, sqrt_var, input);
    auto scale = core::wrap_tensor(ggml_div(ctx.ggml, alpha_rep.tensor, sqrt_rep.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, scale.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModulePortSpec kSingleInput[] = {
    {"input", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModulePortSpec kInputConditioningInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"conditioning", core::PortKind::Activation, false},
};

const core::ModulePortSpec kInputSpeakerInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"speaker", core::PortKind::Activation, false},
};

const core::ModuleSchema kSpeakerConditioningSchema = {
    "SpeakerConditioning",
    "nn.conditioning",
    kInputSpeakerInputs,
    2,
    kSingleOutput,
    1,
    "Projects a speaker embedding to hidden size and adds it across all frames.",
};

const core::ModuleSchema kFiLMSchema = {
    "FiLM",
    "nn.conditioning",
    kInputConditioningInputs,
    2,
    kSingleOutput,
    1,
    "Applies feature-wise linear modulation from a conditioning vector.",
};

const core::ModuleSchema kAdaptiveLayerNormSchema = {
    "AdaptiveLayerNorm",
    "nn.conditioning",
    kInputConditioningInputs,
    2,
    kSingleOutput,
    1,
    "Applies table-backed adaptive affine or RMS-normalized affine modulation.",
};

const core::ModuleSchema kPriorEncoderBlockSchema = {
    "PriorEncoderBlock",
    "nn.conditioning",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Projects hidden states to a latent size, applies GELU, then layer norm.",
};

const core::ModulePortSpec kSqueezeExcite1dInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"fc1_weight", core::PortKind::Parameter, false},
    {"fc1_bias", core::PortKind::Parameter, true},
    {"fc2_weight", core::PortKind::Parameter, false},
    {"fc2_bias", core::PortKind::Parameter, true},
};

const core::ModuleSchema kSqueezeExcite1dSchema = {
    "SqueezeExcite1d",
    "nn.conditioning",
    kSqueezeExcite1dInputs,
    5,
    kSingleOutput,
    1,
    "Applies squeeze-excite channel recalibration to channel-first 1D tensors.",
};

}  // namespace

core::TensorValue LayerScaleModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LayerScaleWeights & weights) const {
    auto scale = repeat_like(ctx, weights.scale, input);
    return mul(ctx, input, scale);
}

TimestepEmbeddingModule::TimestepEmbeddingModule(TimestepEmbeddingConfig config) : config_(config) {
    require_positive(config_.frequency_embedding_size, "TimestepEmbeddingConfig.frequency_embedding_size");
    require_positive(config_.hidden_size, "TimestepEmbeddingConfig.hidden_size");
    if (config_.frequency_embedding_size % 2 != 0) {
        throw std::runtime_error("TimestepEmbeddingConfig.frequency_embedding_size must be even");
    }
}

core::TensorValue TimestepEmbeddingModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timestep,
    const TimestepEmbeddingWeights & weights) const {
    core::validate_rank_between(timestep, 1, core::kMaxTensorRank, "timestep");
    core::validate_last_dim(timestep, 1, "timestep");
    core::validate_shape(weights.freqs, core::TensorShape::from_dims({config_.frequency_embedding_size / 2}), "freqs");

    core::TensorShape expanded_shape = timestep.shape;
    expanded_shape.dims[expanded_shape.rank - 1] = config_.frequency_embedding_size / 2;

    core::TensorShape freqs_shape = {};
    freqs_shape.rank = timestep.shape.rank;
    for (size_t i = 0; i + 1 < freqs_shape.rank; ++i) {
        freqs_shape.dims[i] = 1;
    }
    freqs_shape.dims[freqs_shape.rank - 1] = config_.frequency_embedding_size / 2;

    auto timestep_expanded = RepeatModule({expanded_shape}).build(ctx, timestep);
    auto freqs = core::reshape_tensor(ctx, weights.freqs, freqs_shape);
    freqs = RepeatModule({expanded_shape}).build(ctx, freqs);
    auto args = mul(ctx, timestep_expanded, freqs);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto embedding = concat_last_dim(ctx, cos_part, sin_part);
    auto hidden = LinearModule({config_.frequency_embedding_size, config_.hidden_size, true}).build(ctx, embedding, weights.fc1);
    hidden = SiluModule().build(ctx, hidden);
    hidden = LinearModule({config_.hidden_size, config_.hidden_size, true}).build(ctx, hidden, weights.fc2);
    if (!weights.rms_weight.has_value()) {
        throw std::runtime_error("TimestepEmbeddingWeights.rms_weight is required");
    }
    return variance_rms_norm(ctx, hidden, *weights.rms_weight, config_.rms_eps);
}

AdaLNResidualMLPModule::AdaLNResidualMLPModule(AdaLNResidualMLPConfig config) : config_(config) {
    require_positive(config_.hidden_size, "AdaLNResidualMLPConfig.hidden_size");
    require_positive(config_.intermediate_size, "AdaLNResidualMLPConfig.intermediate_size");
    require_positive(config_.conditioning_size, "AdaLNResidualMLPConfig.conditioning_size");
}

core::TensorValue AdaLNResidualMLPModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conditioning,
    const AdaLNResidualMLPWeights & weights) const {
    auto modulation = LinearModule({config_.conditioning_size, config_.hidden_size * 3, config_.use_bias})
                          .build(ctx, SiluModule().build(ctx, conditioning), weights.modulation.projection);
    auto shift = tensor_layout::ensure_contiguous_layout_if_needed(
        ctx,
        expand_conditioning_like(ctx, slice_last_dim(ctx, modulation, 0, config_.hidden_size), input));
    auto scale = tensor_layout::ensure_contiguous_layout_if_needed(
        ctx,
        expand_conditioning_like(ctx, slice_last_dim(ctx, modulation, config_.hidden_size, config_.hidden_size), input));
    auto gate = tensor_layout::ensure_contiguous_layout_if_needed(
        ctx,
        expand_conditioning_like(ctx, slice_last_dim(ctx, modulation, config_.hidden_size * 2, config_.hidden_size), input));

    auto hidden = apply_adaln(ctx, input, shift, scale, config_.eps, weights.norm);
    hidden = tensor_layout::ensure_contiguous_layout_if_needed(ctx, hidden);
    hidden = LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, hidden, weights.fc1);
    hidden = SiluModule().build(ctx, hidden);
    hidden = tensor_layout::ensure_contiguous_layout_if_needed(ctx, hidden);
    hidden = LinearModule({config_.intermediate_size, config_.hidden_size, config_.use_bias}).build(ctx, hidden, weights.fc2);
    return add_tensors(ctx, input, mul(ctx, gate, hidden));
}

FinalAdaLNProjectionModule::FinalAdaLNProjectionModule(AdaLNResidualMLPConfig config) : config_(config) {
    require_positive(config_.hidden_size, "FinalAdaLNProjectionModule.hidden_size");
    require_positive(config_.conditioning_size, "FinalAdaLNProjectionModule.conditioning_size");
}

core::TensorValue FinalAdaLNProjectionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conditioning,
    const FinalAdaLNProjectionWeights & weights) const {
    auto modulation = LinearModule({config_.conditioning_size, config_.hidden_size * 2, config_.use_bias})
                          .build(ctx, SiluModule().build(ctx, conditioning), weights.modulation.projection);
    auto shift = tensor_layout::ensure_contiguous_layout_if_needed(
        ctx,
        expand_conditioning_like(ctx, slice_last_dim(ctx, modulation, 0, config_.hidden_size), input));
    auto scale = tensor_layout::ensure_contiguous_layout_if_needed(
        ctx,
        expand_conditioning_like(ctx, slice_last_dim(ctx, modulation, config_.hidden_size, config_.hidden_size), input));
    auto input_contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
    auto normalized = core::wrap_tensor(ggml_norm(ctx.ggml, input_contiguous.tensor, config_.eps), input.shape, GGML_TYPE_F32);
    normalized = tensor_layout::ensure_contiguous_layout_if_needed(ctx, normalized);
    auto scaled = mul(ctx, normalized, scale);
    auto modulated = add_tensors(ctx, normalized, scaled);
    auto hidden = add_tensors(ctx, modulated, shift);
    hidden = tensor_layout::ensure_contiguous_layout_if_needed(ctx, hidden);
    return LinearModule({config_.hidden_size, config_.intermediate_size, config_.use_bias}).build(ctx, hidden, weights.projection);
}

ConditionedFlowMLPModule::ConditionedFlowMLPModule(ConditionedFlowMLPConfig config) : config_(config) {
    require_positive(config_.input_size, "ConditionedFlowMLPConfig.input_size");
    require_positive(config_.hidden_size, "ConditionedFlowMLPConfig.hidden_size");
    require_positive(config_.condition_size, "ConditionedFlowMLPConfig.condition_size");
    require_positive(config_.output_size, "ConditionedFlowMLPConfig.output_size");
    require_positive(config_.layers, "ConditionedFlowMLPConfig.layers");
}

core::TensorValue ConditionedFlowMLPModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & condition,
    const core::TensorValue & input,
    const ConditionedFlowMLPWeights & weights) const {
    if (static_cast<int64_t>(weights.residual_layers.size()) != config_.layers) {
        throw std::runtime_error("ConditionedFlowMLPWeights layer count does not match ConditionedFlowMLPConfig.layers");
    }

    auto x = LinearModule({config_.input_size, config_.hidden_size, config_.use_bias}).build(ctx, input, weights.input_projection);
    auto c = LinearModule({config_.condition_size, config_.hidden_size, config_.use_bias}).build(ctx, condition, weights.condition_projection);

    for (const auto & layer_weights : weights.residual_layers) {
        x = AdaLNResidualMLPModule({
            config_.hidden_size,
            config_.hidden_size,
            config_.hidden_size,
            config_.eps,
            config_.use_bias,
        }).build(ctx, x, c, layer_weights);
    }

    return FinalAdaLNProjectionModule({
        config_.hidden_size,
        config_.output_size,
        config_.hidden_size,
        config_.eps,
        config_.use_bias,
    }).build(ctx, x, c, weights.output_projection);
}

TimedConditionedFlowMLPModule::TimedConditionedFlowMLPModule(ConditionedFlowMLPConfig config) : config_(config) {
    require_positive(config_.input_size, "TimedConditionedFlowMLPConfig.input_size");
    require_positive(config_.hidden_size, "TimedConditionedFlowMLPConfig.hidden_size");
    require_positive(config_.condition_size, "TimedConditionedFlowMLPConfig.condition_size");
    require_positive(config_.output_size, "TimedConditionedFlowMLPConfig.output_size");
    require_positive(config_.layers, "TimedConditionedFlowMLPConfig.layers");
}

core::TensorValue TimedConditionedFlowMLPModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & condition,
    const core::TensorValue & start_time,
    const core::TensorValue & end_time,
    const core::TensorValue & input,
    const TimedConditionedFlowMLPWeights & weights) const {
    if (static_cast<int64_t>(weights.residual_layers.size()) != config_.layers) {
        throw std::runtime_error("TimedConditionedFlowMLPWeights layer count does not match config.layers");
    }

    auto x = LinearModule({config_.input_size, config_.hidden_size, config_.use_bias}).build(ctx, input, weights.input_projection);
    auto c = LinearModule({config_.condition_size, config_.hidden_size, config_.use_bias}).build(ctx, condition, weights.condition_projection);
    auto t0 = TimestepEmbeddingModule({256, config_.hidden_size, 1.0e-5F}).build(ctx, start_time, weights.start_time_embedding);
    auto t1 = TimestepEmbeddingModule({256, config_.hidden_size, 1.0e-5F}).build(ctx, end_time, weights.end_time_embedding);
    auto y = add_tensors(ctx, c, scale_tensor(ctx, add_tensors(ctx, t0, t1), 0.5F));

    for (const auto & layer_weights : weights.residual_layers) {
        x = AdaLNResidualMLPModule({
            config_.hidden_size,
            config_.hidden_size,
            config_.hidden_size,
            config_.eps,
            config_.use_bias,
        }).build(ctx, x, y, layer_weights);
    }

    return FinalAdaLNProjectionModule({
        config_.hidden_size,
        config_.output_size,
        config_.hidden_size,
        config_.eps,
        config_.use_bias,
    }).build(ctx, x, y, weights.output_projection);
}

SpeakerConditioningModule::SpeakerConditioningModule(SpeakerConditioningConfig config) : config_(config) {
    require_positive(config_.hidden_size, "SpeakerConditioningConfig.hidden_size");
    require_positive(config_.speaker_dim, "SpeakerConditioningConfig.speaker_dim");
}

const SpeakerConditioningConfig & SpeakerConditioningModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SpeakerConditioningModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SpeakerConditioningModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & speaker,
    const SpeakerConditioningWeights & weights) const {
    core::validate_shape(speaker, core::TensorShape::from_dims({input.shape.dims[0], config_.speaker_dim}), "speaker");
    core::validate_last_dim(input, config_.hidden_size, "input");
    const auto projected = LinearModule({config_.speaker_dim, config_.hidden_size, config_.use_bias}).build(
        ctx,
        speaker,
        LinearWeights{weights.proj_weight, weights.proj_bias});
    return add_tensors(ctx, input, expand_conditioning_like(ctx, projected, input));
}

const core::ModuleSchema & SpeakerConditioningModule::static_schema() noexcept {
    return kSpeakerConditioningSchema;
}

FiLMModule::FiLMModule(FiLMConfig config) : config_(config) {
    require_positive(config_.hidden_size, "FiLMConfig.hidden_size");
    require_positive(config_.conditioning_dim, "FiLMConfig.conditioning_dim");
}

const FiLMConfig & FiLMModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & FiLMModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue FiLMModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conditioning,
    const FiLMWeights & weights) const {
    core::validate_shape(
        conditioning,
        core::TensorShape::from_dims({input.shape.dims[0], config_.conditioning_dim}),
        "conditioning");
    core::validate_last_dim(input, config_.hidden_size, "input");

    const LinearModule gamma({config_.conditioning_dim, config_.hidden_size, config_.use_bias});
    const LinearModule beta({config_.conditioning_dim, config_.hidden_size, config_.use_bias});
    auto gamma_vec = gamma.build(ctx, conditioning, LinearWeights{weights.gamma_weight, weights.gamma_bias});
    auto beta_vec = beta.build(ctx, conditioning, LinearWeights{weights.beta_weight, weights.beta_bias});
    auto scaled = mul(ctx, input, expand_conditioning_like(ctx, gamma_vec, input));
    return add_tensors(ctx, scaled, expand_conditioning_like(ctx, beta_vec, input));
}

const core::ModuleSchema & FiLMModule::static_schema() noexcept {
    return kFiLMSchema;
}

AdaptiveLayerNormModule::AdaptiveLayerNormModule(AdaptiveLayerNormConfig config) : config_(config) {
    require_positive(config_.hidden_size, "AdaptiveLayerNormConfig.hidden_size");
    if (config_.scale_index < 0) {
        throw std::runtime_error("AdaptiveLayerNormConfig.scale_index must be non-negative");
    }
    if (config_.mode != AdaptiveLayerNormMode::Scale && config_.shift_index < 0) {
        throw std::runtime_error("AdaptiveLayerNormConfig.shift_index must be non-negative unless mode is Scale");
    }
}

const AdaptiveLayerNormConfig & AdaptiveLayerNormModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & AdaptiveLayerNormModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue AdaptiveLayerNormModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & modulation,
    const AdaptiveLayerNormWeights & weights) const {
    core::validate_last_dim(input, config_.hidden_size, "input");
    if (modulation.shape.rank != input.shape.rank) {
        throw std::runtime_error("AdaptiveLayerNorm modulation rank must match input rank");
    }
    for (size_t dim = 0; dim + 1 < input.shape.rank; ++dim) {
        if (modulation.shape.dims[dim] != input.shape.dims[dim] && modulation.shape.dims[dim] != 1) {
            throw std::runtime_error("AdaptiveLayerNorm modulation shape must match or broadcast to input except the last dimension");
        }
    }
    if (weights.table.shape.rank != 2 || weights.table.shape.dims[1] != config_.hidden_size) {
        throw std::runtime_error("AdaptiveLayerNorm table shape mismatch");
    }
    const int64_t max_index = std::max(config_.shift_index, config_.scale_index);
    if (weights.table.shape.dims[0] <= max_index ||
        modulation.shape.dims[modulation.shape.rank - 1] < (max_index + 1) * config_.hidden_size) {
        throw std::runtime_error("AdaptiveLayerNorm modulation/table index out of range");
    }

    auto scale = SliceModule({static_cast<int>(modulation.shape.rank - 1), config_.scale_index * config_.hidden_size, config_.hidden_size})
        .build(ctx, modulation);
    auto scale_table = SliceModule({0, config_.scale_index, 1}).build(ctx, weights.table);
    core::TensorShape table_shape = {};
    table_shape.rank = modulation.shape.rank;
    for (size_t dim = 0; dim + 1 < table_shape.rank; ++dim) {
        table_shape.dims[dim] = 1;
    }
    table_shape.dims[table_shape.rank - 1] = config_.hidden_size;
    scale_table = core::reshape_tensor(ctx, scale_table, table_shape);
    scale = add_tensors(ctx, scale, scale_table);

    if (config_.mode == AdaptiveLayerNormMode::Scale) {
        return mul(ctx, input, scale);
    }

    auto shift = SliceModule({static_cast<int>(modulation.shape.rank - 1), config_.shift_index * config_.hidden_size, config_.hidden_size})
        .build(ctx, modulation);
    auto shift_table = SliceModule({0, config_.shift_index, 1}).build(ctx, weights.table);
    shift_table = core::reshape_tensor(ctx, shift_table, table_shape);
    shift = add_tensors(ctx, shift, shift_table);

    const auto normalized = config_.mode == AdaptiveLayerNormMode::RmsAffine
        ? RMSNormModule({config_.hidden_size, config_.eps, false, false}).build(ctx, input, {})
        : input;
    auto one_plus_scale = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, scale.tensor, 1.0F, 1.0F),
        scale.shape,
        GGML_TYPE_F32);
    auto scaled = mul(ctx, normalized, one_plus_scale);
    return add_tensors(ctx, scaled, shift);
}

const core::ModuleSchema & AdaptiveLayerNormModule::static_schema() noexcept {
    return kAdaptiveLayerNormSchema;
}

PriorEncoderBlockModule::PriorEncoderBlockModule(PriorEncoderBlockConfig config) : config_(config) {
    require_positive(config_.hidden_size, "PriorEncoderBlockConfig.hidden_size");
    require_positive(config_.latent_size, "PriorEncoderBlockConfig.latent_size");
}

const PriorEncoderBlockConfig & PriorEncoderBlockModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & PriorEncoderBlockModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue PriorEncoderBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const PriorEncoderBlockWeights & weights) const {
    core::validate_last_dim(input, config_.hidden_size, "input");
    const LinearModule proj({config_.hidden_size, config_.latent_size, config_.use_bias});
    const GeluModule gelu({GeluApproximation::ExactErf});
    const LayerNormModule norm({config_.latent_size, config_.eps, true, true});
    auto output = proj.build(ctx, input, LinearWeights{weights.proj_weight, weights.proj_bias});
    output = gelu.build(ctx, output);
    return norm.build(ctx, output, weights.norm);
}

const core::ModuleSchema & PriorEncoderBlockModule::static_schema() noexcept {
    return kPriorEncoderBlockSchema;
}

SqueezeExcite1dModule::SqueezeExcite1dModule(SqueezeExcite1dConfig config) : config_(config) {
    require_positive(config_.channels, "SqueezeExcite1dConfig.channels");
    require_positive(config_.hidden_channels, "SqueezeExcite1dConfig.hidden_channels");
}

const SqueezeExcite1dConfig & SqueezeExcite1dModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SqueezeExcite1dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SqueezeExcite1dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SqueezeExcite1dWeights & weights) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_shape(
        input,
        core::TensorShape::from_dims({input.shape.dims[0], config_.channels, input.shape.dims[2]}),
        "input");
    auto pooled = ReduceMeanModule({static_cast<int>(input.shape.rank - 1)}).build(ctx, input);
    auto gate = Conv1dModule({config_.channels, config_.hidden_channels, 1, 1, 0, 1, config_.use_bias}).build(ctx, pooled, weights.fc1);
    gate = ReluModule().build(ctx, gate);
    gate = Conv1dModule({config_.hidden_channels, config_.channels, 1, 1, 0, 1, config_.use_bias}).build(ctx, gate, weights.fc2);
    gate = SigmoidModule().build(ctx, gate);
    gate = RepeatModule({input.shape}).build(ctx, gate);
    return MulModule().build(ctx, input, gate);
}

const core::ModuleSchema & SqueezeExcite1dModule::static_schema() noexcept {
    return kSqueezeExcite1dSchema;
}

}  // namespace engine::modules

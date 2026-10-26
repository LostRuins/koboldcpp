#include "engine/framework/modules/attention/projected_grouped_self_attention.h"

#include "attention_internal.h"
#include "engine/framework/modules/positional_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

const core::ModulePortSpec kInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"positions", core::PortKind::Activation, false},
};

const core::ModulePortSpec kOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kProjectedGroupedSelfAttentionSchema = {
    "ProjectedGroupedSelfAttention",
    "nn.attention",
    kInputs,
    2,
    kOutputs,
    1,
    "Applies Q/K/V projections, optional Q/K head norm, RoPE, grouped-query attention, and output projection.",
};

void validate_config(const ProjectedGroupedSelfAttentionConfig & config) {
    if (config.hidden_size <= 0 || config.attention_heads <= 0 || config.kv_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("ProjectedGroupedSelfAttentionConfig dimensions must be positive");
    }
    if (config.attention_heads % config.kv_heads != 0) {
        throw std::runtime_error("ProjectedGroupedSelfAttentionConfig attention_heads must be divisible by kv_heads");
    }
    if (config.qk_norm == ProjectedGroupedSelfAttentionQKNorm::GemmaRMSNorm && !(config.qk_norm_eps > 0.0F)) {
        throw std::runtime_error("ProjectedGroupedSelfAttentionConfig qk_norm_eps must be positive");
    }
    if (config.use_rope && (!(config.rope_theta > 0.0F) || !(config.local_rope_theta > 0.0F))) {
        throw std::runtime_error("ProjectedGroupedSelfAttentionConfig RoPE theta values must be positive");
    }
    if (config.apply_rope_to_projected_prefix &&
        config.qk_norm != ProjectedGroupedSelfAttentionQKNorm::None) {
        throw std::runtime_error("ProjectedGroupedSelfAttention projected-prefix RoPE is incompatible with Q/K norm");
    }
    if (config.query_pre_attn_scalar > 0.0F && config.query_pre_attn_scalar != static_cast<float>(config.head_dim)) {
        throw std::runtime_error("ProjectedGroupedSelfAttention requires query_pre_attn_scalar == head_dim");
    }
}

core::TensorValue reshape_projected_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue apply_qk_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights,
    const ProjectedGroupedSelfAttentionConfig & config) {
    switch (config.qk_norm) {
        case ProjectedGroupedSelfAttentionQKNorm::None:
            return input;
        case ProjectedGroupedSelfAttentionQKNorm::GemmaRMSNorm:
            return GemmaRMSNormModule({config.head_dim, config.qk_norm_eps, true, false}).build(ctx, input, weights);
    }
    throw std::runtime_error("Unsupported projected grouped self-attention QK norm");
}

bool full_attention_layer(const ProjectedGroupedSelfAttentionConfig & config, int64_t layer_index) {
    return config.sliding_window_pattern > 0 && ((layer_index + 1) % config.sliding_window_pattern) == 0;
}

core::TensorValue apply_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const ProjectedGroupedSelfAttentionConfig & config,
    int64_t layer_index) {
    if (!config.use_rope) {
        return input;
    }
    const bool full = full_attention_layer(config, layer_index);
    const float theta = full ? config.rope_theta : config.local_rope_theta;
    const float freq_scale = full ? config.rope_freq_scale : 1.0F;
    return RoPEModule({config.head_dim, config.rope_type, theta, freq_scale}).build(ctx, input, positions);
}

core::TensorValue apply_projected_prefix_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const ProjectedGroupedSelfAttentionConfig & config,
    int64_t layer_index) {
    auto projected = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], 1, input.shape.dims[2]}));
    projected = apply_rope(ctx, projected, positions, config, layer_index);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, projected),
        input.shape);
}

}  // namespace

ProjectedGroupedSelfAttentionModule::ProjectedGroupedSelfAttentionModule(ProjectedGroupedSelfAttentionConfig config)
    : config_(config) {
    validate_config(config_);
}

const ProjectedGroupedSelfAttentionConfig & ProjectedGroupedSelfAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ProjectedGroupedSelfAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ProjectedGroupedSelfAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const ProjectedGroupedSelfAttentionWeights & weights,
    int64_t layer_index,
    const std::optional<core::TensorValue> & attention_mask) const {
    core::validate_rank_between(input, 3, 3, "ProjectedGroupedSelfAttention input");
    core::validate_last_dim(input, config_.hidden_size, "ProjectedGroupedSelfAttention input");
    core::validate_shape(
        positions,
        core::TensorShape::from_dims({input.shape.dims[1]}),
        "ProjectedGroupedSelfAttention positions");

    auto q = attention::internal::build_linear_projection_impl(
        ctx,
        input,
        weights.q_proj,
        config_.hidden_size,
        config_.attention_heads * config_.head_dim,
        config_.use_bias,
        config_.projection_precision,
        config_.use_weight_type_projection_precision,
        config_.use_fast_cuda_projection);
    auto k = attention::internal::build_linear_projection_impl(
        ctx,
        input,
        weights.k_proj,
        config_.hidden_size,
        config_.kv_heads * config_.head_dim,
        config_.use_bias,
        config_.projection_precision,
        config_.use_weight_type_projection_precision,
        config_.use_fast_cuda_projection);
    auto v = attention::internal::build_linear_projection_impl(
        ctx,
        input,
        weights.v_proj,
        config_.hidden_size,
        config_.kv_heads * config_.head_dim,
        config_.use_bias,
        config_.projection_precision,
        config_.use_weight_type_projection_precision,
        config_.use_fast_cuda_projection);

    if (config_.apply_rope_to_projected_prefix) {
        q = apply_projected_prefix_rope(ctx, q, positions, config_, layer_index);
        k = apply_projected_prefix_rope(ctx, k, positions, config_, layer_index);
    }
    q = reshape_projected_heads(ctx, q, config_.attention_heads, config_.head_dim);
    k = reshape_projected_heads(ctx, k, config_.kv_heads, config_.head_dim);
    v = reshape_projected_heads(ctx, v, config_.kv_heads, config_.head_dim);
    q = apply_qk_norm(ctx, q, weights.q_norm, config_);
    k = apply_qk_norm(ctx, k, weights.k_norm, config_);
    if (!config_.apply_rope_to_projected_prefix) {
        q = apply_rope(ctx, q, positions, config_, layer_index);
        k = apply_rope(ctx, k, positions, config_, layer_index);
    }

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = GroupedQueryAttentionModule({
        config_.head_dim,
        config_.lowering,
        config_.attention_precision,
        config_.causality,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.attention_heads * config_.head_dim}));
    return attention::internal::build_linear_projection_impl(
        ctx,
        context,
        weights.o_proj,
        config_.attention_heads * config_.head_dim,
        config_.hidden_size,
        config_.use_bias,
        config_.projection_precision,
        config_.use_weight_type_projection_precision,
        config_.use_fast_cuda_projection);
}

const core::ModuleSchema & ProjectedGroupedSelfAttentionModule::static_schema() noexcept {
    return kProjectedGroupedSelfAttentionSchema;
}

}  // namespace engine::modules

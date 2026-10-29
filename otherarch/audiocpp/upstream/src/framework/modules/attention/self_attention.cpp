#include "attention_internal.h"

#include "engine/framework/modules/attention/grouped_query_attention.h"

namespace engine::modules {

using namespace attention::internal;

namespace {

void require_packed_self_attention_config(const AttentionConfig & config) {
    if (!config.use_packed_qkv) {
        throw std::runtime_error("SelfAttentionModule packed QKV path requires use_packed_qkv");
    }
}

LinearWeights require_packed_qkv_weights(const AttentionWeights & weights, bool use_bias) {
    if (!weights.qkv_weight.has_value()) {
        throw std::runtime_error("SelfAttentionModule packed QKV path requires qkv_weight");
    }
    if (use_bias && !weights.qkv_bias.has_value()) {
        throw std::runtime_error("SelfAttentionModule packed QKV path requires qkv_bias when bias is enabled");
    }
    return {*weights.qkv_weight, weights.qkv_bias};
}

struct PackedQkvHeads {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

PackedQkvHeads build_packed_qkv_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AttentionConfig & config,
    const AttentionWeights & weights) {
    const int64_t head_dim = config.hidden_size / config.num_heads;
    auto qkv = LinearModule({
        config.hidden_size,
        3 * config.hidden_size,
        config.use_bias,
        config.projection_precision,
    }).build(ctx, input, require_packed_qkv_weights(weights, config.use_bias));

    auto q = SliceModule({2, 0, config.hidden_size}).build(ctx, qkv);
    q = reshape_heads(ctx, ensure_contiguous_layout(ctx, q), config.num_heads, head_dim);
    auto k = SliceModule({2, config.hidden_size, config.hidden_size}).build(ctx, qkv);
    k = reshape_heads(ctx, ensure_contiguous_layout(ctx, k), config.num_heads, head_dim);
    auto v = SliceModule({2, 2 * config.hidden_size, config.hidden_size}).build(ctx, qkv);
    v = reshape_heads(ctx, ensure_contiguous_layout(ctx, v), config.num_heads, head_dim);
    return {q, k, v};
}

core::TensorValue project_attention_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & context,
    const core::TensorShape & input_shape,
    const AttentionConfig & config,
    const AttentionWeights & weights) {
    auto output = ensure_contiguous_layout(ctx, context);
    output = core::reshape_tensor(
        ctx,
        output,
        core::TensorShape::from_dims({input_shape.dims[0], input_shape.dims[1], config.hidden_size}));
    return LinearModule({
        config.hidden_size,
        config.hidden_size,
        config.use_bias,
        config.projection_precision,
    }).build(ctx, output, make_linear_weights(weights.out_weight, weights.out_bias));
}

}  // namespace

SelfAttentionModule::SelfAttentionModule(AttentionConfig config) : config_(config) {
    validate_attention_config(config_);
}

const AttentionConfig & SelfAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & SelfAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue SelfAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AttentionWeights & weights) const {
    return build(ctx, input, weights, std::nullopt);
}

core::TensorValue SelfAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AttentionWeights & weights,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (!config_.use_packed_qkv) {
        if (attention_mask.has_value()) {
            throw std::runtime_error("SelfAttentionModule attention_mask requires packed QKV path");
        }
        return build_attention_impl(ctx, input, input, config_, require_attention_weights(weights, config_.use_bias));
    }
    require_packed_self_attention_config(config_);
    validate_sequence_input(input, config_.hidden_size, "input");
    const int64_t head_dim = config_.hidden_size / config_.num_heads;
    const auto qkv = build_packed_qkv_heads(ctx, input, config_, weights);
    auto q_heads = permute_tensor(ctx, qkv.q, {0, 2, 1, 3});
    q_heads = ensure_contiguous_layout(ctx, q_heads);
    auto k_heads = permute_tensor(ctx, qkv.k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, qkv.v, {0, 2, 1, 3});
    core::TensorValue context;
    if (attention_mask.has_value()) {
        context = GroupedQueryAttentionModule({
            head_dim,
            GroupedQueryAttentionLowering::FlashGroupedViewKV,
            config_.attention_precision,
            config_.causal ? AttentionCausality::Causal : AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    } else if (config_.use_flash_attention && !config_.causal) {
        context = GroupedQueryAttentionModule({
            head_dim,
            GroupedQueryAttentionLowering::FlashGroupedViewKV,
            config_.attention_precision,
            AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads);
    } else {
        auto kt = permute_tensor(ctx, k_heads, {0, 1, 3, 2});
        auto scores = MatMulModule().build(ctx, q_heads, kt);
        if (config_.causal) {
            scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        }
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))), scores.shape, GGML_TYPE_F32);
        auto probs = SoftmaxModule().build(ctx, scores);
        context = MatMulModule().build(ctx, probs, v_heads);
        context = permute_tensor(ctx, context, {0, 2, 1, 3});
    }
    return project_attention_output(ctx, context, input.shape, config_, weights);
}

StreamingAttentionOutputs SelfAttentionModule::build_cached_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AttentionWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    FastKVSetRowsMode set_rows_mode) const {
    require_packed_self_attention_config(config_);
    if (!config_.causal) {
        throw std::runtime_error("SelfAttentionModule cached packed tail requires causal attention");
    }
    validate_sequence_input(input, config_.hidden_size, "input");
    const int64_t head_dim = config_.hidden_size / config_.num_heads;
    auto qkv = build_packed_qkv_heads(ctx, input, config_, weights);
    qkv.k = ensure_contiguous_layout(ctx, qkv.k);
    qkv.v = ensure_contiguous_layout(ctx, qkv.v);
    const FastKVSetRowsModule set_rows({set_rows_mode});
    auto updated_key = set_rows.build(ctx, cache_key, qkv.k, cache_slot);
    auto updated_value = set_rows.build(ctx, cache_value, qkv.v, cache_slot);
    auto q_heads = permute_tensor(ctx, qkv.q, {0, 2, 1, 3});
    q_heads = ensure_contiguous_layout(ctx, q_heads);
    auto k_heads = permute_tensor(ctx, updated_key, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, updated_value, {0, 2, 1, 3});
    auto context = GroupedQueryAttentionModule({
        head_dim,
        GroupedQueryAttentionLowering::FlashGroupedViewKV,
        config_.attention_precision,
        AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    return {
        project_attention_output(ctx, context, input.shape, config_, weights),
        updated_key,
        updated_value,
    };
}

const core::ModuleSchema & SelfAttentionModule::static_schema() noexcept {
    return kSelfAttentionSchema;
}

StreamingSelfAttentionModule::StreamingSelfAttentionModule(AttentionConfig config) : config_(config) {
    validate_attention_config(config_);
}

const AttentionConfig & StreamingSelfAttentionModule::config() const noexcept {
    return config_;
}

StreamingAttentionOutputs StreamingSelfAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const AttentionWeights & weights,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    if (prefix_key.has_value() != prefix_value.has_value()) {
        throw std::runtime_error("Streaming self-attention requires both prefix_key and prefix_value or neither");
    }
    if (prefix_key.has_value()) {
        core::validate_rank_between(*prefix_key, 4, 4, "prefix_key");
        core::validate_rank_between(*prefix_value, 4, 4, "prefix_value");
        if (prefix_key->shape.dims[0] != input.shape.dims[0] || prefix_value->shape.dims[0] != input.shape.dims[0]) {
            throw std::runtime_error("Streaming self-attention prefix batch size must match input");
        }
        if (config_.prefix_cache_layout == AttentionPrefixCacheLayout::HeadsSequence) {
            if (prefix_key->shape.dims[1] != config_.num_heads || prefix_value->shape.dims[1] != config_.num_heads) {
                throw std::runtime_error("Streaming self-attention BHTD prefix head count mismatch");
            }
        } else if (prefix_key->shape.dims[2] != config_.num_heads || prefix_value->shape.dims[2] != config_.num_heads) {
            throw std::runtime_error("Streaming self-attention BTHD prefix head count mismatch");
        }
    }

    const int64_t head_dim = config_.hidden_size / config_.num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const LinearModule q_proj({config_.hidden_size, config_.hidden_size, config_.use_bias, config_.projection_precision});
    const LinearModule k_proj({config_.hidden_size, config_.hidden_size, config_.use_bias, config_.projection_precision});
    const LinearModule v_proj({config_.hidden_size, config_.hidden_size, config_.use_bias, config_.projection_precision});
    const LinearModule out_proj({config_.hidden_size, config_.hidden_size, config_.use_bias, config_.projection_precision});
    const MatMulModule matmul;

    auto q = q_proj.build(ctx, input, make_linear_weights(weights.q_weight, weights.q_bias));
    auto k = k_proj.build(ctx, input, make_linear_weights(weights.k_weight, weights.k_bias));
    auto v = v_proj.build(ctx, input, make_linear_weights(weights.v_weight, weights.v_bias));

    q = reshape_heads(ctx, q, config_.num_heads, head_dim);
    k = reshape_heads(ctx, k, config_.num_heads, head_dim);
    v = reshape_heads(ctx, v, config_.num_heads, head_dim);

    q = build_positions_rope(ctx, q, positions, head_dim);
    k = build_positions_rope(ctx, k, positions, head_dim);

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});

    core::TensorValue context;
    if (prefix_key.has_value()) {
        core::TensorValue all_k_heads;
        core::TensorValue all_v_heads;
        int64_t prefix_steps = 0;
        if (config_.prefix_cache_layout == AttentionPrefixCacheLayout::HeadsSequence) {
            prefix_steps = prefix_key->shape.dims[2];
            all_k_heads = ConcatModule({2}).build(ctx, *prefix_key, k_heads);
            all_v_heads = ConcatModule({2}).build(ctx, *prefix_value, v_heads);
        } else {
            prefix_steps = prefix_key->shape.dims[1];
            auto all_k = concat_sequence_axis(ctx, prefix_key, k);
            auto all_v = concat_sequence_axis(ctx, prefix_value, v);
            all_k_heads = permute_tensor(ctx, all_k, {0, 2, 1, 3});
            all_v_heads = permute_tensor(ctx, all_v, {0, 2, 1, 3});
        }
        auto k_transposed = permute_tensor(ctx, all_k_heads, {0, 1, 3, 2});
        auto scores = matmul.build(ctx, q_heads, k_transposed);
        if (config_.attention_precision != GGML_PREC_DEFAULT) {
            ggml_mul_mat_set_prec(scores.tensor, config_.attention_precision);
        }
        core::TensorValue attn;
        if (attention_mask.has_value()) {
            attn = core::wrap_tensor(
                ggml_soft_max_ext(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor, attention_mask->tensor, scale, 0.0f),
                scores.shape,
                GGML_TYPE_F32);
        } else {
            scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
            scores = core::wrap_tensor(
                ggml_diag_mask_inf(ctx.ggml, scores.tensor, static_cast<int>(prefix_steps)),
                scores.shape,
                GGML_TYPE_F32);
            attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
        }
        context = matmul.build(ctx, attn, all_v_heads);
        if (config_.attention_precision != GGML_PREC_DEFAULT) {
            ggml_mul_mat_set_prec(context.tensor, config_.attention_precision);
        }
    } else {
        auto k_transposed = permute_tensor(ctx, k_heads, {0, 1, 3, 2});
        auto scores = matmul.build(ctx, q_heads, k_transposed);
        if (config_.attention_precision != GGML_PREC_DEFAULT) {
            ggml_mul_mat_set_prec(scores.tensor, config_.attention_precision);
        }
        core::TensorValue attn;
        if (attention_mask.has_value()) {
            attn = core::wrap_tensor(
                ggml_soft_max_ext(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor, attention_mask->tensor, scale, 0.0f),
                scores.shape,
                GGML_TYPE_F32);
        } else {
            scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
            scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
            attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
        }
        context = matmul.build(ctx, attn, v_heads);
        if (config_.attention_precision != GGML_PREC_DEFAULT) {
            ggml_mul_mat_set_prec(context.tensor, config_.attention_precision);
        }
    }

    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ensure_contiguous_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.hidden_size}));

    auto output = out_proj.build(ctx, context, make_linear_weights(weights.out_weight, weights.out_bias));
    if (config_.prefix_cache_layout == AttentionPrefixCacheLayout::HeadsSequence) {
        return {
            output,
            ensure_contiguous_layout(ctx, k_heads),
            ensure_contiguous_layout(ctx, v_heads),
        };
    }
    return {output, k, v};
}

}  // namespace engine::modules

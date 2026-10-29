#include "attention_internal.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"

namespace engine::modules {

using namespace attention::internal;

namespace {

int64_t cross_key_value_size(const AttentionConfig & config) {
    return config.key_value_size > 0 ? config.key_value_size : config.hidden_size;
}

int64_t cross_attention_size(const AttentionConfig & config) {
    return config.attention_size > 0 ? config.attention_size : config.hidden_size;
}

int64_t cross_head_dim(const AttentionConfig & config) {
    const int64_t attention_size = cross_attention_size(config);
    return config.head_dim > 0 ? config.head_dim : attention_size / config.num_heads;
}

LinearWeights require_packed_kv_weights(const AttentionWeights & weights, bool use_bias) {
    if (!weights.qkv_weight.has_value()) {
        throw std::runtime_error("CrossAttentionModule packed KV path requires qkv_weight");
    }
    if (use_bias && !weights.qkv_bias.has_value()) {
        throw std::runtime_error("CrossAttentionModule packed KV path requires qkv_bias when bias is enabled");
    }
    return {*weights.qkv_weight, weights.qkv_bias};
}

void validate_cross_query(const core::TensorValue & query, const AttentionConfig & config) {
    validate_sequence_input(query, config.hidden_size, "query");
    const int64_t attention_size = cross_attention_size(config);
    const int64_t head_dim = cross_head_dim(config);
    if (attention_size <= 0 || head_dim <= 0 || config.num_heads <= 0 ||
        attention_size != config.num_heads * head_dim) {
        throw std::runtime_error("CrossAttentionModule attention_size must equal num_heads * head_dim");
    }
}

void validate_cross_memory(const core::TensorValue & memory, const AttentionConfig & config) {
    validate_sequence_input(memory, cross_key_value_size(config), "memory");
}

void validate_cross_cache(
    const CrossAttentionKeyValue & key_value,
    const core::TensorValue & query,
    int64_t memory_frames,
    const AttentionConfig & config) {
    const int64_t head_dim = cross_head_dim(config);
    if (key_value.key.shape.rank != 4 || key_value.value.shape.rank != 4 ||
        key_value.key.shape.dims[0] != query.shape.dims[0] ||
        key_value.value.shape.dims[0] != query.shape.dims[0] ||
        key_value.key.shape.dims[1] != config.num_heads ||
        key_value.value.shape.dims[1] != config.num_heads ||
        key_value.key.shape.dims[2] != memory_frames ||
        key_value.value.shape.dims[2] != memory_frames ||
        key_value.key.shape.dims[3] != head_dim ||
        key_value.value.shape.dims[3] != head_dim) {
        throw std::runtime_error("CrossAttentionModule cached KV shape is invalid");
    }
}

core::TensorValue build_cross_query(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const AttentionConfig & config,
    const AttentionWeights & weights) {
    const int64_t attention_size = cross_attention_size(config);
    const int64_t head_dim = cross_head_dim(config);
    auto projected = LinearModule({
        config.hidden_size,
        attention_size,
        config.use_bias,
        config.projection_precision,
    }).build(ctx, query, make_linear_weights(weights.q_weight, weights.q_bias));
    projected = core::reshape_tensor(
        ctx,
        ensure_contiguous_layout(ctx, projected),
        core::TensorShape::from_dims({projected.shape.dims[0], projected.shape.dims[1], config.num_heads, head_dim}));
    return permute_tensor(ctx, projected, {0, 2, 1, 3});
}

core::TensorValue build_cross_probabilities(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query_heads,
    const core::TensorValue & key_heads,
    const core::TensorValue & memory_mask,
    int64_t head_dim,
    const core::TensorValue * attention_prior,
    core::TensorValue * last_attention) {
    auto kt = permute_tensor(ctx, key_heads, {0, 1, 3, 2});
    auto scores = MatMulModule().build(ctx, query_heads, kt);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    auto key_mask = core::reshape_tensor(
        ctx,
        ensure_contiguous_layout(ctx, core::wrap_tensor(ggml_cast(ctx.ggml, memory_mask.tensor, GGML_TYPE_F32), memory_mask.shape, GGML_TYPE_F32)),
        core::TensorShape::from_dims({memory_mask.shape.dims[0], 1, 1, memory_mask.shape.dims[1]}));
    auto score_mask = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, key_mask.tensor, 1.0e30F, -1.0e30F),
        key_mask.shape,
        GGML_TYPE_F32);
    scores = AddModule().build(ctx, scores, RepeatModule({scores.shape}).build(ctx, score_mask));
    auto probs = SoftmaxModule().build(ctx, scores);
    probs = MulModule().build(ctx, probs, RepeatModule({probs.shape}).build(ctx, key_mask));
    if (attention_prior != nullptr) {
        auto prior_repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, attention_prior->tensor, probs.tensor), probs.shape, GGML_TYPE_F32);
        probs = MulModule().build(ctx, probs, prior_repeated);
        auto normalizer = core::wrap_tensor(
            ggml_repeat(ctx.ggml, ggml_sum_rows(ctx.ggml, probs.tensor), probs.tensor),
            probs.shape,
            GGML_TYPE_F32);
        probs = core::wrap_tensor(ggml_div(ctx.ggml, probs.tensor, normalizer.tensor), probs.shape, GGML_TYPE_F32);
    }
    if (last_attention != nullptr) {
        *last_attention = SliceModule({2, query_heads.shape.dims[2] - 1, 1}).build(ctx, probs);
        *last_attention = ensure_contiguous_layout(ctx, *last_attention);
    }
    return probs;
}

core::TensorValue build_cross_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & probabilities,
    const core::TensorValue & value_heads,
    const AttentionConfig & config,
    const AttentionWeights & weights) {
    const int64_t attention_size = cross_attention_size(config);
    auto context = MatMulModule().build(ctx, probabilities, value_heads);
    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ensure_contiguous_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], attention_size}));
    return LinearModule({
        attention_size,
        config.hidden_size,
        config.use_bias,
        config.projection_precision,
    }).build(ctx, context, make_linear_weights(weights.out_weight, weights.out_bias));
}

}  // namespace

CrossAttentionModule::CrossAttentionModule(AttentionConfig config) : config_(config) {
    validate_attention_config(config_);
}

const AttentionConfig & CrossAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & CrossAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue CrossAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & memory,
    const AttentionWeights & weights) const {
    if (config_.use_packed_kv) {
        throw std::runtime_error("CrossAttentionModule packed KV path requires memory_mask");
    }
    return build_attention_impl(ctx, query, memory, config_, require_attention_weights(weights, config_.use_bias));
}

core::TensorValue CrossAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & memory,
    const AttentionWeights & weights,
    const core::TensorValue & memory_mask,
    const core::TensorValue * attention_prior,
    core::TensorValue * last_attention) const {
    if (!config_.use_packed_kv) {
        throw std::runtime_error("CrossAttentionModule masked path currently requires packed KV");
    }
    validate_cross_query(query, config_);
    validate_cross_memory(memory, config_);
    const auto key_value = build_key_value(ctx, memory, weights);
    return build_cached(ctx, query, key_value, weights, memory_mask, attention_prior, last_attention);
}

core::TensorValue CrossAttentionModule::build_cached(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const CrossAttentionKeyValue & key_value,
    const AttentionWeights & weights,
    const core::TensorValue & memory_mask,
    const core::TensorValue * attention_prior,
    core::TensorValue * last_attention) const {
    if (!config_.use_packed_kv) {
        throw std::runtime_error("CrossAttentionModule cached path requires packed KV");
    }
    validate_cross_query(query, config_);
    validate_cross_cache(key_value, query, memory_mask.shape.dims[1], config_);
    auto query_heads = build_cross_query(ctx, query, config_, weights);
    auto probs = build_cross_probabilities(
        ctx,
        query_heads,
        key_value.key,
        memory_mask,
        cross_head_dim(config_),
        attention_prior,
        last_attention);
    return build_cross_output(ctx, query, probs, key_value.value, config_, weights);
}

core::TensorValue CrossAttentionModule::build_cached_flash(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const CrossAttentionKeyValue & key_value,
    const AttentionWeights & weights,
    const core::TensorValue & attention_mask) const {
    if (!config_.use_packed_kv) {
        throw std::runtime_error("CrossAttentionModule cached flash path requires packed KV");
    }
    validate_cross_query(query, config_);
    core::validate_rank_between(attention_mask, 4, 4, "cross_attention.flash_mask");
    const auto & shape = attention_mask.shape;
    if (attention_mask.type != GGML_TYPE_F16 || !ggml_is_contiguous(attention_mask.tensor) ||
        (shape.dims[0] != 1 && shape.dims[0] != query.shape.dims[0]) ||
        (shape.dims[1] != 1 && shape.dims[1] != config_.num_heads) ||
        shape.dims[2] != query.shape.dims[1]) {
        throw std::runtime_error("CrossAttentionModule flash mask must be contiguous F16 [B|1,H|1,Q,K]");
    }
    validate_cross_cache(key_value, query, shape.dims[3], config_);
    const auto query_heads = build_cross_query(ctx, query, config_, weights);
    const auto precision = config_.attention_precision == GGML_PREC_DEFAULT
        ? GGML_PREC_F32 : config_.attention_precision;
    auto context = ScaledDotProductAttentionModule({cross_head_dim(config_),
        ScaledDotProductAttentionLowering::Flash, precision})
        .build(ctx, query_heads, key_value.key, key_value.value, attention_mask);
    context = core::reshape_tensor(ctx, context,
        core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], cross_attention_size(config_)}));
    return LinearModule({cross_attention_size(config_), config_.hidden_size,
        config_.use_bias, config_.projection_precision})
        .build(ctx, context, make_linear_weights(weights.out_weight, weights.out_bias));
}

CrossAttentionKeyValue CrossAttentionModule::build_key_value(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & memory,
    const AttentionWeights & weights) const {
    if (!config_.use_packed_kv) {
        throw std::runtime_error("CrossAttentionModule build_key_value requires packed KV");
    }
    validate_cross_memory(memory, config_);
    const int64_t attention_size = cross_attention_size(config_);
    const int64_t head_dim = cross_head_dim(config_);
    auto kv = LinearModule({
        cross_key_value_size(config_),
        2 * attention_size,
        config_.use_bias,
        config_.projection_precision,
    }).build(ctx, memory, require_packed_kv_weights(weights, config_.use_bias));
    auto key = SliceModule({2, 0, attention_size}).build(ctx, kv);
    key = core::reshape_tensor(
        ctx,
        ensure_contiguous_layout(ctx, key),
        core::TensorShape::from_dims({key.shape.dims[0], key.shape.dims[1], config_.num_heads, head_dim}));
    auto value = SliceModule({2, attention_size, attention_size}).build(ctx, kv);
    value = core::reshape_tensor(
        ctx,
        ensure_contiguous_layout(ctx, value),
        core::TensorShape::from_dims({value.shape.dims[0], value.shape.dims[1], config_.num_heads, head_dim}));
    return {
        ensure_contiguous_layout(ctx, permute_tensor(ctx, key, {0, 2, 1, 3})),
        ensure_contiguous_layout(ctx, permute_tensor(ctx, value, {0, 2, 1, 3})),
    };
}

const core::ModuleSchema & CrossAttentionModule::static_schema() noexcept {
    return kCrossAttentionSchema;
}

}  // namespace engine::modules

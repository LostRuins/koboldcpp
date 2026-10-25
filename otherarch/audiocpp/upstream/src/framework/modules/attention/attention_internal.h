#pragma once

#include "engine/framework/modules/attention_modules.h"

#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "../module_internal.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::modules::attention::internal {

inline const core::ModulePortSpec kInputOutputInputs[] = {
    {"input", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kQueryMemoryInputs[] = {
    {"query", core::PortKind::Activation, false},
    {"memory", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kPositionalEncodingSchema = {
    "PositionalEncoding",
    "nn.position",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Adds a precomputed positional encoding table to the sequence axis.",
};

inline const core::ModuleSchema kFeedForwardSchema = {
    "FeedForward",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies a two-layer MLP block with GELU activation.",
};

inline const core::ModuleSchema kFeedForwardGeluSchema = {
    "FeedForwardGelu",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies a two-layer MLP block with tanh-approximated GELU activation.",
};

inline const core::ModuleSchema kGatedFeedForwardSchema = {
    "GatedFeedForward",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies a gated feed-forward block using gate, up, and down projections.",
};

inline const core::ModuleSchema kConvFeedForwardSchema = {
    "ConvFeedForward",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies a convolutional feed-forward block to [batch, frames, hidden] inputs.",
};

inline const core::ModuleSchema kGatedConvFeedForwardSchema = {
    "GatedConvFeedForward",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies a gated convolutional feed-forward block to [batch, frames, hidden] inputs.",
};

inline const core::ModuleSchema kSelfAttentionSchema = {
    "SelfAttention",
    "nn.attention",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies multi-head self-attention over [batch, frames, hidden] inputs.",
};

inline const core::ModuleSchema kStreamingSelfAttentionSchema = {
    "StreamingSelfAttention",
    "nn.attention",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Applies causal self-attention with optional prefix KV cache and explicit RoPE positions.",
};

inline const core::ModuleSchema kCrossAttentionSchema = {
    "CrossAttention",
    "nn.attention",
    kQueryMemoryInputs,
    2,
    kSingleOutput,
    1,
    "Applies multi-head cross-attention from query to memory sequences.",
};

inline const core::ModuleSchema kTransformerEncoderBlockSchema = {
    "TransformerEncoderBlock",
    "nn.block",
    kInputOutputInputs,
    1,
    kSingleOutput,
    1,
    "Pre-norm transformer encoder block with self-attention and feed-forward sublayers.",
};

inline const core::ModuleSchema kTransformerDecoderBlockSchema = {
    "TransformerDecoderBlock",
    "nn.block",
    kQueryMemoryInputs,
    2,
    kSingleOutput,
    1,
    "Pre-norm transformer decoder block with self-attention, cross-attention, and feed-forward sublayers.",
};

inline void validate_hidden_positive(int64_t hidden_size, const char * name) {
    if (hidden_size <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

inline void validate_attention_config(const AttentionConfig & config) {
    validate_hidden_positive(config.hidden_size, "AttentionConfig.hidden_size");
    if (config.num_heads <= 0) {
        throw std::runtime_error("AttentionConfig.num_heads must be positive");
    }
    if (config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("AttentionConfig.hidden_size must be divisible by num_heads");
    }
}

inline void validate_relative_attention_config(const RelativeAttentionConfig & config) {
    validate_hidden_positive(config.hidden_size, "RelativeAttentionConfig.hidden_size");
    if (config.num_heads <= 0) {
        throw std::runtime_error("RelativeAttentionConfig.num_heads must be positive");
    }
    if (config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("RelativeAttentionConfig.hidden_size must be divisible by num_heads");
    }
    if (config.cache_drop_size < 0) {
        throw std::runtime_error("RelativeAttentionConfig.cache_drop_size must be non-negative");
    }
}

using engine::modules::internal::concat_all;
using engine::modules::internal::concat_range;
using engine::modules::internal::validate_sequence_input;

inline core::TensorValue ensure_contiguous_layout(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

inline core::TensorValue permute_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    std::array<int, core::kMaxTensorRank> axes) {
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;

    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        if (in_logical_axis < 0 || in_logical_axis >= static_cast<int>(input.shape.rank)) {
            throw std::runtime_error("Permute axis out of range");
        }
        output_shape.dims[out_logical_axis] = input.shape.dims[in_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
    }

    return core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

inline core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t num_heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        input,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
}

inline core::TensorValue add_attention_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & bias,
    int64_t num_heads,
    int64_t head_dim) {
    core::validate_shape(bias, core::TensorShape::from_dims({num_heads, head_dim}), "bias");
    auto bias_view = core::reshape_tensor(ctx, bias, core::TensorShape::from_dims({1, num_heads, 1, head_dim}));
    auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, repeated.tensor), input.shape, GGML_TYPE_F32);
}

inline core::TensorValue zero_like_last_column(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto first = SliceModule({3, 0, 1}).build(ctx, input);
    auto first_contiguous = ensure_contiguous_layout(ctx, first);
    return core::wrap_tensor(ggml_scale(ctx.ggml, first_contiguous.tensor, 0.0f), first.shape, GGML_TYPE_F32);
}

inline core::TensorValue relative_shift(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    core::validate_rank_between(input, 4, 4, "relative_shift.input");
    const int64_t batch = input.shape.dims[0];
    const int64_t heads = input.shape.dims[1];
    const int64_t query = input.shape.dims[2];
    const int64_t pos = input.shape.dims[3];

    auto padded = ConcatModule({3}).build(ctx, zero_like_last_column(ctx, input), input);
    auto reshaped = core::reshape_tensor(
        ctx,
        ensure_contiguous_layout(ctx, padded),
        core::TensorShape::from_dims({batch, heads, pos + 1, query}));
    auto sliced = SliceModule({2, 1, pos}).build(ctx, reshaped);
    return core::reshape_tensor(ctx, ensure_contiguous_layout(ctx, sliced), input.shape);
}

inline RelativeAttentionWeights require_relative_attention_weights(const RelativeAttentionWeights & weights, bool use_bias) {
    if (use_bias &&
        (!weights.attention.q_bias.has_value() || !weights.attention.k_bias.has_value() ||
         !weights.attention.v_bias.has_value() || !weights.attention.out_bias.has_value())) {
        throw std::runtime_error("Relative attention biases are required when use_bias is true");
    }
    return weights;
}

core::TensorValue build_global_relative_attention_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & key_value,
    const std::optional<core::TensorValue> & pos_emb,
    const RelativeAttentionConfig & config,
    const RelativeAttentionWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & query_keep_mask,
    const std::optional<core::TensorValue> & projected_pos_emb);

inline core::TensorValue add_range(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & values,
    size_t begin,
    size_t end) {
    if (begin + 1 == end) {
        return values[begin];
    }
    const AddModule add;
    const size_t mid = begin + (end - begin) / 2;
    return add.build(ctx, add_range(ctx, values, begin, mid), add_range(ctx, values, mid, end));
}

inline core::TensorValue add_all(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & values) {
    if (values.empty()) {
        throw std::runtime_error("add_all requires at least one tensor");
    }
    return add_range(ctx, values, 0, values.size());
}

core::TensorValue build_windowed_relative_attention_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & key_value,
    const core::TensorValue & pos_emb,
    const RelativeAttentionConfig & config,
    const RelativeAttentionWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & query_keep_mask);

inline core::TensorValue build_positions_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    int64_t head_dim) {
    if (positions.type != GGML_TYPE_I32) {
        throw std::runtime_error("Streaming attention RoPE positions must be GGML_TYPE_I32");
    }
    core::validate_shape(positions, core::TensorShape::from_dims({input.shape.dims[1]}), "positions");
    return core::wrap_tensor(
        ggml_rope_ext(
            ctx.ggml,
            input.tensor,
            positions.tensor,
            nullptr,
            static_cast<int>(head_dim),
            GGML_ROPE_TYPE_NORMAL,
            0,
            10000.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f),
        input.shape,
        GGML_TYPE_F32);
}

inline core::TensorValue concat_sequence_axis(
    core::ModuleBuildContext & ctx,
    const std::optional<core::TensorValue> & prefix,
    const core::TensorValue & current) {
    if (!prefix.has_value()) {
        return current;
    }
    return ConcatModule({1}).build(ctx, *prefix, current);
}

inline core::TensorValue update_hidden_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::optional<core::TensorValue> & prefix_cache,
    int64_t cache_drop_size) {
    if (!prefix_cache.has_value()) {
        return input;
    }
    if (cache_drop_size < 0 || cache_drop_size > input.shape.dims[1]) {
        throw std::runtime_error("Streaming relative attention cache_drop_size must be in [0, input_frames]");
    }

    const int64_t keep_from_query = input.shape.dims[1] - cache_drop_size;
    if (keep_from_query == 0) {
        return core::wrap_tensor(ggml_cont(ctx.ggml, prefix_cache->tensor), prefix_cache->shape, prefix_cache->type);
    }
    if (keep_from_query == input.shape.dims[1]) {
        return core::wrap_tensor(ggml_cont(ctx.ggml, input.tensor), input.shape, input.type);
    }
    if (keep_from_query >= prefix_cache->shape.dims[1]) {
        return SliceModule({1, 0, keep_from_query}).build(ctx, input);
    }

    auto cache_tail = SliceModule({1, keep_from_query, prefix_cache->shape.dims[1] - keep_from_query}).build(ctx, *prefix_cache);
    auto query_head = SliceModule({1, 0, keep_from_query}).build(ctx, input);
    return ConcatModule({1}).build(ctx, cache_tail, query_head);
}

inline core::TensorValue build_attention_context_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const AttentionConfig & config) {
    const int64_t head_dim = config.hidden_size / config.num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const MatMulModule matmul;

    auto k_transposed = permute_tensor(ctx, k_heads, {0, 1, 3, 2});
    auto scores = matmul.build(ctx, q_heads, k_transposed);
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
    auto attn = core::wrap_tensor(
        ggml_soft_max(ctx.ggml, ensure_contiguous_layout(ctx, scores).tensor),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

inline LinearWeights make_linear_weights(const core::TensorValue & weight, const std::optional<core::TensorValue> & bias) {
    return LinearWeights{weight, bias};
}

inline bool dense_projection_weight(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

inline core::TensorValue build_linear_projection_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LinearWeights & weights,
    int64_t in_features,
    int64_t out_features,
    bool use_bias,
    ggml_prec precision,
    bool use_weight_type_precision,
    bool use_fast_cuda_projection) {
    const ggml_prec resolved_precision = use_weight_type_precision
        ? (ggml_is_quantized(weights.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32)
        : precision;
    if (use_fast_cuda_projection && !use_bias && ctx.backend_type == core::BackendType::Cuda &&
        dense_projection_weight(weights.weight.type) && out_features >= 128 && out_features % 4 == 0) {
        return FastPackedProjection4Module({in_features, out_features, resolved_precision}).build(ctx, input, weights);
    }
    return LinearModule({in_features, out_features, use_bias, resolved_precision}).build(ctx, input, weights);
}

inline FeedForwardWeights require_feed_forward_weights(const FeedForwardWeights & weights, bool use_bias) {
    if (use_bias && (!weights.fc1_bias.has_value() || !weights.fc2_bias.has_value())) {
        throw std::runtime_error("FeedForward biases are required when use_bias is true");
    }
    return weights;
}

inline core::TensorValue build_feed_forward_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FeedForwardConfig & config,
    const FeedForwardWeights & weights) {
    const LinearModule fc1({config.hidden_size, config.intermediate_size, config.use_bias, config.projection_precision});
    const GeluModule gelu({config.gelu_approximation});
    const LinearModule fc2({config.intermediate_size, config.hidden_size, config.use_bias, config.projection_precision});

    auto hidden = fc1.build(ctx, input, make_linear_weights(weights.fc1_weight, weights.fc1_bias));
    switch (config.activation) {
        case FeedForwardActivation::Gelu: hidden = gelu.build(ctx, hidden); break;
        case FeedForwardActivation::Relu: hidden = ReluModule().build(ctx, hidden); break;
        default: throw std::runtime_error("Unsupported feed-forward activation");
    }
    return fc2.build(ctx, hidden, make_linear_weights(weights.fc2_weight, weights.fc2_bias));
}

inline GatedFeedForwardWeights require_gated_feed_forward_weights(
    const GatedFeedForwardWeights & weights,
    bool use_bias) {
    if (use_bias &&
        (!weights.gate_proj.bias.has_value() || !weights.up_proj.bias.has_value() || !weights.down_proj.bias.has_value())) {
        throw std::runtime_error("GatedFeedForward biases are required when use_bias is true");
    }
    return weights;
}

inline core::TensorValue build_gated_feed_forward_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const GatedFeedForwardConfig & config,
    const GatedFeedForwardWeights & weights) {
    auto gate = build_linear_projection_impl(
        ctx,
        input,
        weights.gate_proj,
        config.hidden_size,
        config.intermediate_size,
        config.use_bias,
        config.projection_precision,
        config.use_weight_type_projection_precision,
        config.use_fast_cuda_projection);
    switch (config.activation) {
        case GatedFeedForwardActivation::Gelu:
            gate = GeluModule({config.gelu_approximation}).build(ctx, gate);
            break;
        case GatedFeedForwardActivation::Silu:
            gate = SiluModule{}.build(ctx, gate);
            break;
    }
    auto up = build_linear_projection_impl(
        ctx,
        input,
        weights.up_proj,
        config.hidden_size,
        config.intermediate_size,
        config.use_bias,
        config.projection_precision,
        config.use_weight_type_projection_precision,
        config.use_fast_cuda_projection);
    auto hidden = MulModule{}.build(ctx, gate, up);
    return build_linear_projection_impl(
        ctx,
        hidden,
        weights.down_proj,
        config.intermediate_size,
        config.hidden_size,
        config.use_bias,
        config.projection_precision,
        config.use_weight_type_projection_precision,
        config.use_fast_cuda_projection);
}

inline AttentionWeights require_attention_weights(const AttentionWeights & weights, bool use_bias) {
    if (use_bias &&
        (!weights.q_bias.has_value() || !weights.k_bias.has_value() || !weights.v_bias.has_value() ||
         !weights.out_bias.has_value())) {
        throw std::runtime_error("Attention biases are required when use_bias is true");
    }
    return weights;
}

inline core::TensorValue build_attention_impl(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & key_value,
    const AttentionConfig & config,
    const AttentionWeights & weights) {
    validate_sequence_input(query, config.hidden_size, "query");
    validate_sequence_input(key_value, config.hidden_size, "key_value");

    const int64_t head_dim = config.hidden_size / config.num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const LinearModule q_proj({config.hidden_size, config.hidden_size, config.use_bias, config.projection_precision});
    const LinearModule k_proj({config.hidden_size, config.hidden_size, config.use_bias, config.projection_precision});
    const LinearModule v_proj({config.hidden_size, config.hidden_size, config.use_bias, config.projection_precision});
    const LinearModule out_proj({config.hidden_size, config.hidden_size, config.use_bias, config.projection_precision});
    const MatMulModule matmul;

    auto q = q_proj.build(ctx, query, make_linear_weights(weights.q_weight, weights.q_bias));
    auto k = k_proj.build(ctx, key_value, make_linear_weights(weights.k_weight, weights.k_bias));
    auto v = v_proj.build(ctx, key_value, make_linear_weights(weights.v_weight, weights.v_bias));

    q = reshape_heads(ctx, q, config.num_heads, head_dim);
    k = reshape_heads(ctx, k, config.num_heads, head_dim);
    v = reshape_heads(ctx, v, config.num_heads, head_dim);

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});
    auto k_transposed = permute_tensor(ctx, k_heads, {0, 1, 3, 2});

    auto scores = matmul.build(ctx, q_heads, k_transposed);
    if (config.attention_precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(scores.tensor, config.attention_precision);
    }
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);

    auto context = matmul.build(ctx, attn, v_heads);
    if (config.attention_precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(context.tensor, config.attention_precision);
    }
    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = ensure_contiguous_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({query.shape.dims[0], query.shape.dims[1], config.hidden_size}));

    return out_proj.build(ctx, context, make_linear_weights(weights.out_weight, weights.out_bias));
}

inline NormConfig make_norm_config(int64_t hidden_size, float eps) {
    return NormConfig{hidden_size, eps, true, true};
}

}  // namespace engine::modules::attention::internal

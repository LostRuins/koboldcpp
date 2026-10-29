#include "engine/framework/modules/text_encoders/t5_gemma_encoder.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

int64_t attention_size(const T5GemmaEncoderConfig & config) {
    return config.attention_size > 0 ? config.attention_size : config.hidden_size;
}

void validate_config(const T5GemmaEncoderConfig & config) {
    if (config.hidden_size <= 0 || config.layers <= 0 || config.attention_heads <= 0 ||
        config.kv_heads <= 0 || config.head_dim <= 0 || config.intermediate_size <= 0 ||
        config.vocab_size <= 0) {
        throw std::runtime_error("T5GemmaEncoderConfig dimensions must be positive");
    }
    if (config.attention_heads * config.head_dim != attention_size(config)) {
        throw std::runtime_error("T5GemmaEncoderConfig attention_heads * head_dim must equal attention_size");
    }
    if (config.attention_heads % config.kv_heads != 0) {
        throw std::runtime_error("T5GemmaEncoderConfig attention_heads must be divisible by kv_heads");
    }
    if (!config.layer_rope_theta.empty() && static_cast<int64_t>(config.layer_rope_theta.size()) != config.layers) {
        throw std::runtime_error("T5GemmaEncoderConfig layer_rope_theta must match layer count");
    }
    if (!config.layer_rope_freq_scale.empty() && static_cast<int64_t>(config.layer_rope_freq_scale.size()) != config.layers) {
        throw std::runtime_error("T5GemmaEncoderConfig layer_rope_freq_scale must match layer count");
    }
    if (!(config.rope_theta > 0.0F) || !(config.rms_norm_eps > 0.0F) ||
        !(config.query_pre_attn_scalar > 0.0F)) {
        throw std::runtime_error("T5GemmaEncoderConfig scalar values must be positive");
    }
}

float layer_rope_theta(const T5GemmaEncoderConfig & config, int64_t layer_index) {
    return config.layer_rope_theta.empty()
        ? config.rope_theta
        : config.layer_rope_theta.at(static_cast<size_t>(layer_index));
}

float layer_rope_freq_scale(const T5GemmaEncoderConfig & config, int64_t layer_index) {
    return config.layer_rope_freq_scale.empty()
        ? config.rope_freq_scale
        : config.layer_rope_freq_scale.at(static_cast<size_t>(layer_index));
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

std::array<int, core::kMaxTensorRank> transpose_last_two_axes(size_t rank) {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    if (rank < 2) {
        throw std::runtime_error("transpose_last_two_axes requires rank >= 2");
    }
    std::swap(axes[rank - 2], axes[rank - 1]);
    return axes;
}

core::TensorValue matmul_f32(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    core::validate_rank_between(lhs, 2, core::kMaxTensorRank, "lhs");
    core::validate_rank_between(rhs, lhs.shape.rank, lhs.shape.rank, "rhs");
    const size_t rank = lhs.shape.rank;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (lhs.shape.dims[i] != rhs.shape.dims[i]) {
            throw std::runtime_error("T5GemmaEncoder matmul batch dimensions must match");
        }
    }
    if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2]) {
        throw std::runtime_error("T5GemmaEncoder matmul inner dimensions must match");
    }
    auto rhs_transposed = TransposeModule({transpose_last_two_axes(rank), rank}).build(ctx, rhs);
    rhs_transposed = ensure_contiguous(ctx, rhs_transposed);
    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    ggml_tensor * output = ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return core::wrap_tensor(output, output_shape, GGML_TYPE_F32);
}

core::TensorValue t5_gemma_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    T5GemmaRMSNormStyle style,
    float eps,
    int64_t hidden_size) {
    core::validate_last_dim(input, hidden_size, "T5GemmaEncoder RMSNorm input");
    core::validate_shape(weight, core::TensorShape::from_dims({hidden_size}), "T5GemmaEncoder RMSNorm weight");
    auto normalized = core::wrap_tensor(ggml_rms_norm(ctx.ggml, ensure_contiguous(ctx, input).tensor, eps), input.shape, GGML_TYPE_F32);
    if (style == T5GemmaRMSNormStyle::Direct) {
        return core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, weight.tensor), input.shape, GGML_TYPE_F32);
    }
    auto one_plus_weight = core::wrap_tensor(ggml_scale_bias(ctx.ggml, weight.tensor, 1.0F, 1.0F), weight.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, normalized.tensor, one_plus_weight.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    core::validate_rank_between(input, 4, 4, "T5GemmaEncoder repeat_kv_heads input");
    auto contiguous = ensure_contiguous(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
        .build(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const T5GemmaEncoderLayerWeights & weights,
    const T5GemmaEncoderConfig & config,
    int64_t layer_index) {
    const int64_t attn_size = attention_size(config);
    const LinearModule q_proj({config.hidden_size, config.attention_heads * config.head_dim, false, GGML_PREC_F32});
    const LinearModule k_proj({config.hidden_size, config.kv_heads * config.head_dim, false, GGML_PREC_F32});
    const LinearModule v_proj({config.hidden_size, config.kv_heads * config.head_dim, false, GGML_PREC_F32});
    const LinearModule o_proj({attn_size, config.hidden_size, false, GGML_PREC_F32});
    auto q = q_proj.build(ctx, input, weights.q_proj);
    auto k = k_proj.build(ctx, input, weights.k_proj);
    auto v = v_proj.build(ctx, input, weights.v_proj);
    q = reshape_heads(ctx, q, config.attention_heads, config.head_dim);
    k = reshape_heads(ctx, k, config.kv_heads, config.head_dim);
    v = reshape_heads(ctx, v, config.kv_heads, config.head_dim);
    if (config.use_qk_norm) {
        if (!weights.q_norm.weight.has_value() || !weights.k_norm.weight.has_value()) {
            throw std::runtime_error("T5GemmaEncoder q/k norm weights are required when use_qk_norm is enabled");
        }
        q = t5_gemma_rms_norm(ctx, q, *weights.q_norm.weight, config.rms_norm_style, config.rms_norm_eps, config.head_dim);
        k = t5_gemma_rms_norm(ctx, k, *weights.k_norm.weight, config.rms_norm_style, config.rms_norm_eps, config.head_dim);
    }
    q = RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, layer_rope_theta(config, layer_index), layer_rope_freq_scale(config, layer_index)}).build(ctx, q, positions);
    k = RoPEModule({config.head_dim, GGML_ROPE_TYPE_NEOX, layer_rope_theta(config, layer_index), layer_rope_freq_scale(config, layer_index)}).build(ctx, k, positions);
    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    const int64_t kv_repeats = config.attention_heads / config.kv_heads;
    k_heads = repeat_kv_heads(ctx, k_heads, kv_repeats);
    v_heads = repeat_kv_heads(ctx, v_heads, kv_repeats);
    auto k_heads_contiguous = ensure_contiguous(ctx, k_heads);
    auto scores_raw = ggml_mul_mat(ctx.ggml, k_heads_contiguous.tensor, q_heads.tensor);
    ggml_mul_mat_set_prec(scores_raw, GGML_PREC_F32);
    auto scores = core::wrap_tensor(
        scores_raw,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[1], q_heads.shape.dims[2], k_heads.shape.dims[2]}),
        GGML_TYPE_F32);
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(config.query_pre_attn_scalar)),
        scores.shape,
        GGML_TYPE_F32);
    if (config.attn_logit_softcap > 0.0F) {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, ensure_contiguous(ctx, scores).tensor, 1.0F / config.attn_logit_softcap),
            scores.shape,
            GGML_TYPE_F32);
        scores = TanhModule{}.build(ctx, scores);
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, ensure_contiguous(ctx, scores).tensor, config.attn_logit_softcap),
            scores.shape,
            GGML_TYPE_F32);
    }
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            ensure_contiguous(ctx, scores).tensor,
            additive_attention_mask.tensor,
            1.0F,
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = matmul_f32(ctx, attn, v_heads);
    context = TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], attn_size}));
    return o_proj.build(ctx, context, weights.o_proj);
}

core::TensorValue mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const T5GemmaEncoderLayerWeights & weights,
    const T5GemmaEncoderConfig & config) {
    const LinearModule gate_proj({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32});
    const LinearModule up_proj({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32});
    const LinearModule down_proj({config.intermediate_size, config.hidden_size, false, GGML_PREC_F32});
    auto gate = gate_proj.build(ctx, input, weights.gate_proj);
    gate = GeluModule({GeluApproximation::Tanh}).build(ctx, gate);
    auto up = up_proj.build(ctx, input, weights.up_proj);
    return down_proj.build(ctx, MulModule{}.build(ctx, gate, up), weights.down_proj);
}

core::TensorValue layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const T5GemmaEncoderLayerWeights & weights,
    const T5GemmaEncoderConfig & config,
    int64_t layer_index) {
    auto hidden = t5_gemma_rms_norm(ctx, input, weights.pre_self_attn_norm, config.rms_norm_style, config.rms_norm_eps, config.hidden_size);
    hidden = self_attention(ctx, hidden, positions, additive_attention_mask, weights, config, layer_index);
    hidden = t5_gemma_rms_norm(ctx, hidden, weights.post_self_attn_norm, config.rms_norm_style, config.rms_norm_eps, config.hidden_size);
    auto output = AddModule{}.build(ctx, input, hidden);
    hidden = t5_gemma_rms_norm(ctx, output, weights.pre_ff_norm, config.rms_norm_style, config.rms_norm_eps, config.hidden_size);
    hidden = mlp(ctx, hidden, weights, config);
    hidden = t5_gemma_rms_norm(ctx, hidden, weights.post_ff_norm, config.rms_norm_style, config.rms_norm_eps, config.hidden_size);
    return AddModule{}.build(ctx, output, hidden);
}

}  // namespace

T5GemmaEncoderModule::T5GemmaEncoderModule(T5GemmaEncoderConfig config) : config_(config) {
    validate_config(config_);
}

const T5GemmaEncoderConfig & T5GemmaEncoderModule::config() const noexcept {
    return config_;
}

core::TensorValue T5GemmaEncoderModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const T5GemmaEncoderWeights & weights) const {
    core::validate_rank_between(input_ids, 2, 2, "T5GemmaEncoder input_ids");
    core::validate_shape(positions, core::TensorShape::from_dims({input_ids.shape.dims[1]}), "T5GemmaEncoder positions");
    core::validate_shape(
        additive_attention_mask,
        core::TensorShape::from_dims({input_ids.shape.dims[0], config_.attention_heads, input_ids.shape.dims[1], input_ids.shape.dims[1]}),
        "T5GemmaEncoder additive_attention_mask");
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("T5GemmaEncoder layer count mismatch");
    }
    auto hidden = EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(ctx, input_ids, weights.embed_tokens);
    if (config_.scale_embeddings) {
        hidden = core::wrap_tensor(
            ggml_scale(ctx.ggml, ensure_contiguous(ctx, hidden).tensor, std::sqrt(static_cast<float>(config_.hidden_size))),
            hidden.shape,
            GGML_TYPE_F32);
    }
    for (int64_t i = 0; i < config_.layers; ++i) {
        hidden = layer(ctx, hidden, positions, additive_attention_mask, weights.layers[static_cast<size_t>(i)], config_, i);
    }
    return t5_gemma_rms_norm(ctx, hidden, weights.norm, config_.rms_norm_style, config_.rms_norm_eps, config_.hidden_size);
}

}  // namespace engine::modules

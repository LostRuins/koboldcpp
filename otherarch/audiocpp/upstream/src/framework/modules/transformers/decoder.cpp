#include "engine/framework/modules/transformers/decoder.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "../module_internal.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

using engine::modules::internal::concat_all;
using engine::modules::internal::validate_sequence_input;

inline const core::ModulePortSpec kDecoderInputs[] = {
    {"input", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kDecoderOutputs[] = {
    {"output", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kDecoderLayerSchema = {
    "QwenDecoderLayer",
    "nn.block",
    kDecoderInputs,
    1,
    kDecoderOutputs,
    1,
    "Qwen-style decoder block with grouped-query attention, RoPE, q/k RMSNorm, and SwiGLU MLP.",
};

int64_t require_head_dim(const DecoderLayerConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("QwenDecoderLayerConfig attention dimensions must be positive");
    }
    if (config.hidden_size <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("QwenDecoderLayerConfig hidden sizes must be positive");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("QwenDecoderLayerConfig.num_attention_heads must be divisible by num_key_value_heads");
    }
    if (config.activation_cast.enabled && config.activation_cast.type != GGML_TYPE_F32 &&
        config.activation_cast.type != GGML_TYPE_F16 && config.activation_cast.type != GGML_TYPE_BF16) {
        throw std::runtime_error("QwenDecoderLayerConfig activation cast supports only f32, f16, and bf16");
    }
    if (config.activation_cast.fused_round && config.activation_cast.type != GGML_TYPE_BF16) {
        throw std::runtime_error("QwenDecoderLayerConfig fused activation rounding requires bf16");
    }
    return config.head_dim;
}

core::TensorValue reshape_decoder_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
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
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask) {
    const MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    auto context = matmul.build(ctx, attn, v_heads);
    return TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

core::TensorValue attention_from_grouped_query_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    int64_t attention_heads,
    int64_t key_value_heads,
    const core::TensorValue & attention_mask) {
    const int64_t repeats = attention_heads / key_value_heads;
    std::vector<core::TensorValue> head_outputs;
    head_outputs.reserve(static_cast<size_t>(attention_heads));
    for (int64_t head = 0; head < attention_heads; ++head) {
        const int64_t key_value_head = head / repeats;
        auto q_head = SliceModule({1, head, 1}).build(ctx, q_heads);
        auto k_head = SliceModule({1, key_value_head, 1}).build(ctx, k_heads);
        auto v_head = SliceModule({1, key_value_head, 1}).build(ctx, v_heads);
        head_outputs.push_back(attention_from_heads(ctx, q_head, k_head, v_head, dim, attention_mask));
    }
    return concat_all(ctx, head_outputs, 2);
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    ggml_prec precision) {
    if (!core::has_backend_addressable_layout(q_heads.tensor) ||
        !core::has_backend_addressable_layout(k_heads.tensor) ||
        !core::has_backend_addressable_layout(v_heads.tensor)) {
        throw std::runtime_error("Qwen decoder flash attention expects contiguous Q/K/V heads");
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue flash_attention_from_grouped_heads_view_kv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask,
    ggml_prec precision) {
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_contiguous.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Qwen decoder cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, dim}),
        GGML_TYPE_F32);
}

void apply_batched_static_rope(
    core::ModuleBuildContext & ctx,
    const DecoderLayerConfig & config,
    const DecoderLayerWeights & weights,
    const core::TensorValue & positions,
    core::TensorValue & q,
    core::TensorValue & k,
    int64_t dim) {
    const core::TensorValue * rope_factors = weights.rope_frequency_factors.has_value()
        ? &*weights.rope_frequency_factors
        : nullptr;
    RoPEConfig rope{dim, config.rope_type, config.rope_theta};
    rope.intermediate_round_type = config.activation_cast.rope_intermediate_type;
    if (positions.shape.rank == 1 && positions.shape.dims[0] == q.shape.dims[1]) {
        q = RoPEModule(rope).build(ctx, q, positions, rope_factors);
        k = RoPEModule(rope).build(ctx, k, positions, rope_factors);
        return;
    }
    if (q.shape.dims[1] != 1 || positions.shape.rank != 1 || positions.shape.dims[0] != q.shape.dims[0]) {
        throw std::runtime_error("Qwen decoder batched static-cache RoPE positions must be [1] or [batch]");
    }
    std::vector<core::TensorValue> q_rows;
    std::vector<core::TensorValue> k_rows;
    q_rows.reserve(static_cast<size_t>(q.shape.dims[0]));
    k_rows.reserve(static_cast<size_t>(q.shape.dims[0]));
    for (int64_t batch = 0; batch < q.shape.dims[0]; ++batch) {
        auto q_row = SliceModule({0, batch, 1}).build(ctx, q);
        auto k_row = SliceModule({0, batch, 1}).build(ctx, k);
        auto pos_row = SliceModule({0, batch, 1}).build(ctx, positions);
        if (ctx.backend_type == core::BackendType::Vulkan) {
            // Vulkan RoPE cannot address a position view at a non-aligned byte offset.
            pos_row = core::ensure_backend_addressable_layout(ctx, pos_row);
        }
        q_rows.push_back(RoPEModule(rope).build(ctx, q_row, pos_row, rope_factors));
        k_rows.push_back(RoPEModule(rope).build(ctx, k_row, pos_row, rope_factors));
    }
    q = concat_all(ctx, q_rows, 0);
    k = concat_all(ctx, k_rows, 0);
}

LinearWeights require_linear(const LinearWeights & weights, bool use_bias, const char * name) {
    if (use_bias && !weights.bias.has_value()) {
        throw std::runtime_error(std::string(name) + " bias is required");
    }
    return weights;
}

core::TensorValue activation_cast(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DecoderActivationCastPolicy & policy) {
    if (policy.type == GGML_TYPE_F32) {
        return core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    }
    if (policy.fused_round && policy.type == GGML_TYPE_BF16 && ggml_is_contiguous_rows(input.tensor)) {
        // Fused single-kernel round-to-bf16: f32/f16/bf16 in, always f32 out,
        // values rounded to bf16. Numerically identical to the cast round trip
        // below (bf16 input is already rounded, so rounding is a widening no-op),
        // but avoids the intermediate bf16 tensor and one kernel launch.
        // kcpp: portable replacement for Audio.cpp's private BF16-round op.
        auto * round_input = input.tensor->type == GGML_TYPE_F32
            ? input.tensor
            : ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32);
        return core::wrap_tensor(
            ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, round_input, GGML_TYPE_BF16), GGML_TYPE_F32),
            input.shape,
            GGML_TYPE_F32);
    }
    auto * cast_input = policy.type == GGML_TYPE_BF16 && input.tensor->type != GGML_TYPE_F32
        ? ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32)
        : input.tensor;
    auto rounded = core::wrap_tensor(ggml_cast(ctx.ggml, cast_input, policy.type), input.shape, policy.type);
    return core::wrap_tensor(ggml_cast(ctx.ggml, rounded.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
}

struct QKVProjections {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

bool flash_branches_allowed(const DecoderLayerConfig & config) {
    return config.runtime.attention.allow_flash_attention;
}

QKVProjections build_qkv_projections(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DecoderLayerWeights & weights,
    const DecoderLayerConfig & config,
    int64_t dim) {
    const int64_t q_out = config.num_attention_heads * dim;
    const int64_t kv_out = config.num_key_value_heads * dim;
    if (config.qkv_layout == DecoderQKVLayout::PackedQKV) {
        if (!weights.self_attention.qkv_weight.has_value()) {
            throw std::runtime_error("Qwen packed QKV layout requires self_attention.qkv_weight");
        }
        auto qkv = LinearModule(
                       {
                           config.hidden_size,
                           q_out + kv_out * 2,
                           weights.self_attention.qkv_bias.has_value(),
                           config.projection_precision,
                       })
                       .build(ctx, input, {*weights.self_attention.qkv_weight, weights.self_attention.qkv_bias});
        return {
            SliceModule({2, 0, q_out}).build(ctx, qkv),
            SliceModule({2, q_out, kv_out}).build(ctx, qkv),
            SliceModule({2, q_out + kv_out, kv_out}).build(ctx, qkv),
        };
    }
    auto q = LinearModule(
                 {
                     config.hidden_size,
                     config.num_attention_heads * dim,
                     weights.self_attention.q_bias.has_value(),
                     config.projection_precision,
                 })
                 .build(
                     ctx,
                     input,
                     {weights.self_attention.q_weight, weights.self_attention.q_bias});
    auto k = LinearModule(
                 {
                     config.hidden_size,
                     config.num_key_value_heads * dim,
                     weights.self_attention.k_bias.has_value(),
                     config.projection_precision,
                 })
                 .build(
                     ctx,
                     input,
                     {weights.self_attention.k_weight, weights.self_attention.k_bias});
    auto v = LinearModule(
                 {
                     config.hidden_size,
                     config.num_key_value_heads * dim,
                     weights.self_attention.v_bias.has_value(),
                     config.projection_precision,
                 })
                 .build(
                     ctx,
                     input,
                     {weights.self_attention.v_weight, weights.self_attention.v_bias});
    return {q, k, v};
}

}  // namespace

DecoderLayerConfig decoder_layer_config_from_stack(const DecoderStackConfig & config) {
    DecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.num_key_value_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.rope_type = config.rope_type;
    out.position_encoding = config.position_encoding;
    out.attention_precision = config.attention_precision;
    out.projection_precision = config.projection_precision;
    out.qkv_layout = config.qkv_layout;
    out.use_qk_norm = config.use_qk_norm;
    out.activation_cast = config.activation_cast;
    out.runtime = config.runtime;
    return out;
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DecoderLayerConfig & config,
    const DecoderMLPWeights & weights) {
    if (config.runtime.mlp.mode == DecoderMLPMode::Exact) {
        auto gate = LinearModule(
                        {
                            config.hidden_size,
                            config.intermediate_size,
                            weights.gate_proj.bias.has_value(),
                            config.projection_precision,
                        })
                        .build(ctx, input, require_linear(weights.gate_proj, false, "QwenMLPWeights.gate_proj"));
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_projection) {
            gate = activation_cast(ctx, gate, config.activation_cast);
        }
        gate = SiluModule{}.build(ctx, gate);
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_silu) {
            gate = activation_cast(ctx, gate, config.activation_cast);
        }
        auto up = LinearModule(
                      {
                          config.hidden_size,
                          config.intermediate_size,
                          weights.up_proj.bias.has_value(),
                          config.projection_precision,
                      })
                      .build(ctx, input, require_linear(weights.up_proj, false, "QwenMLPWeights.up_proj"));
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_projection) {
            up = activation_cast(ctx, up, config.activation_cast);
        }
        auto gated = MulModule{}.build(ctx, gate, up);
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_mul) {
            gated = activation_cast(ctx, gated, config.activation_cast);
        }
        auto down = LinearModule(
                        {
                            config.intermediate_size,
                            config.hidden_size,
                            weights.down_proj.bias.has_value(),
                            config.projection_precision,
                        })
                        .build(ctx, gated, require_linear(weights.down_proj, false, "QwenMLPWeights.down_proj"));
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_projection) {
            down = activation_cast(ctx, down, config.activation_cast);
        }
        return down;
    }

    core::TensorValue gate;
    core::TensorValue up;
    std::optional<core::TensorValue> packed_gate_up;
    const auto mlp_mode = config.runtime.mlp.mode;
    if (mlp_mode == DecoderMLPMode::PackedGateUp) {
        if (!weights.gate_up_proj.has_value()) {
            throw std::runtime_error("QwenMLPWeights.gate_up_proj is required for packed gate/up mode");
        }
        auto gate_up = LinearModule(
                           {
                               config.hidden_size,
                               config.intermediate_size * 2,
                               weights.gate_up_proj->bias.has_value(),
                               config.projection_precision,
                           })
                           .build(
                               ctx,
                               input,
                               require_linear(*weights.gate_up_proj, false, "QwenMLPWeights.gate_up_proj"));
        packed_gate_up = gate_up;
        gate = SliceModule({2, 0, config.intermediate_size}).build(ctx, gate_up);
        up = SliceModule({2, config.intermediate_size, config.intermediate_size}).build(ctx, gate_up);
    } else {
        gate = LinearModule(
                   {
                       config.hidden_size,
                       config.intermediate_size,
                       weights.gate_proj.bias.has_value(),
                       config.projection_precision,
                   })
                   .build(ctx, input, require_linear(weights.gate_proj, false, "QwenMLPWeights.gate_proj"));
        up = LinearModule(
                 {
                     config.hidden_size,
                     config.intermediate_size,
                     weights.up_proj.bias.has_value(),
                     config.projection_precision,
                 })
                 .build(ctx, input, require_linear(weights.up_proj, false, "QwenMLPWeights.up_proj"));
    }
    const bool can_use_fused_swiglu =
        !config.activation_cast.enabled ||
        (!config.activation_cast.after_mlp_projection &&
         !config.activation_cast.after_mlp_silu &&
         !config.activation_cast.after_mlp_mul);
    core::TensorValue gated;
    if (can_use_fused_swiglu && mlp_mode == DecoderMLPMode::PackedGateUp) {
        gated = core::wrap_tensor(
            ggml_swiglu(ctx.ggml, packed_gate_up->tensor),
            core::TensorShape::from_dims({
                input.shape.dims[0],
                input.shape.dims[1],
                config.intermediate_size,
            }),
            packed_gate_up->type);
    } else if (can_use_fused_swiglu && mlp_mode == DecoderMLPMode::FusedSwiGLU) {
        gated = core::wrap_tensor(
            ggml_swiglu_split(ctx.ggml, gate.tensor, up.tensor),
            gate.shape,
            gate.type);
    } else {
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_projection) {
            gate = activation_cast(ctx, gate, config.activation_cast);
            up = activation_cast(ctx, up, config.activation_cast);
        }
        gate = SiluModule{}.build(ctx, gate);
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_silu) {
            gate = activation_cast(ctx, gate, config.activation_cast);
        }
        gated = MulModule{}.build(ctx, gate, up);
        if (config.activation_cast.enabled && config.activation_cast.after_mlp_mul) {
            gated = activation_cast(ctx, gated, config.activation_cast);
        }
    }
    auto down = LinearModule(
                    {
                        config.intermediate_size,
                        config.hidden_size,
                        weights.down_proj.bias.has_value(),
                        config.projection_precision,
                    })
                    .build(ctx, gated, require_linear(weights.down_proj, false, "QwenMLPWeights.down_proj"));
    if (config.activation_cast.enabled && config.activation_cast.after_mlp_projection) {
        down = activation_cast(ctx, down, config.activation_cast);
    }
    return down;
}

DecoderLayerModule::DecoderLayerModule(DecoderLayerConfig config) : config_(config) {
    require_head_dim(config_);
}

const DecoderLayerConfig & DecoderLayerModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & DecoderLayerModule::schema() const noexcept {
    return static_schema();
}

DecoderLayerOutputs DecoderLayerModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderLayerWeights & weights,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    if (prefix_key.has_value() != prefix_value.has_value()) {
        throw std::runtime_error("Qwen decoder layer requires both prefix_key and prefix_value or neither");
    }

    const int64_t dim = require_head_dim(config_);
    const int64_t kv_repeats = config_.num_attention_heads / config_.num_key_value_heads;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                                config_.activation_cast.norm_intermediate_type})
                      .build(ctx, input, weights.input_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_input_norm) {
        x_norm = activation_cast(ctx, x_norm, config_.activation_cast);
    }
    auto qkv = build_qkv_projections(ctx, x_norm, weights, config_, dim);
    if (config_.activation_cast.enabled && config_.activation_cast.after_qkv_projection) {
        qkv.q = activation_cast(ctx, qkv.q, config_.activation_cast);
        qkv.k = activation_cast(ctx, qkv.k, config_.activation_cast);
        qkv.v = activation_cast(ctx, qkv.v, config_.activation_cast);
    }

    auto q = reshape_decoder_heads(ctx, qkv.q, config_.num_attention_heads, dim);
    auto k = reshape_decoder_heads(ctx, qkv.k, config_.num_key_value_heads, dim);
    if (config_.use_qk_norm) {
        q = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, q, weights.q_norm);
        k = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, k, weights.k_norm);
        if (config_.activation_cast.enabled && config_.activation_cast.after_qk_norm) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    auto v = reshape_decoder_heads(ctx, qkv.v, config_.num_key_value_heads, dim);

    if (config_.position_encoding == DecoderPositionEncoding::Rotary) {
        const core::TensorValue * rope_factors = weights.rope_frequency_factors.has_value()
            ? &*weights.rope_frequency_factors
            : nullptr;
        RoPEConfig rope{dim, config_.rope_type, config_.rope_theta};
        rope.intermediate_round_type = config_.activation_cast.rope_intermediate_type;
        q = RoPEModule(rope).build(ctx, q, positions, rope_factors);
        k = RoPEModule(rope).build(ctx, k, positions, rope_factors);
        if (config_.activation_cast.enabled && config_.activation_cast.after_rope) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    const bool allow_flash = flash_branches_allowed(config_);
    const bool use_prefix_flash =
        allow_flash &&
        prefix_key.has_value() &&
        config_.runtime.attention.prefix_mode == DecoderPrefixAttentionMode::FlashWithPrefix &&
        config_.runtime.attention.prefill_mode == DecoderAttentionMode::FlashGroupedViewKV;
    core::TensorValue all_k = k;
    core::TensorValue all_v = v;
    // Cached prefix KV may be stored in a different dtype than the current
    // K/V (e.g. Higgs reference state); cast before concat on every path.
    // (The eager branch previously skipped this and died in ggml_concat.)
    if (prefix_key.has_value()) {
        auto attention_prefix_key = prefix_key;
        auto attention_prefix_value = prefix_value;
        if (attention_prefix_key->type != k.type) {
            attention_prefix_key = core::wrap_tensor(
                ggml_cast(ctx.ggml, attention_prefix_key->tensor, k.type),
                attention_prefix_key->shape,
                k.type);
        }
        if (attention_prefix_value->type != v.type) {
            attention_prefix_value = core::wrap_tensor(
                ggml_cast(ctx.ggml, attention_prefix_value->tensor, v.type),
                attention_prefix_value->shape,
                v.type);
        }
        all_k = ConcatModule({1}).build(ctx, *attention_prefix_key, k);
        all_v = ConcatModule({1}).build(ctx, *attention_prefix_value, v);
    }
    core::TensorValue context;
    if (allow_flash && !prefix_key.has_value() && attention_mask.has_value() &&
        config_.runtime.attention.prefill_mode == DecoderAttentionMode::FlashGroupedViewKV) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            *attention_mask,
            config_.attention_precision);
    } else if (allow_flash && attention_mask.has_value() && use_prefix_flash) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            *attention_mask,
            config_.attention_precision);
    } else if (allow_flash && attention_mask.has_value() && !prefix_key.has_value() &&
               config_.runtime.attention.prefill_mode == DecoderAttentionMode::FlashGrouped) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = flash_attention_from_grouped_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            *attention_mask,
            config_.attention_precision);
    } else {
        auto k_heads = repeat_kv_heads(
            ctx,
            TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k),
            kv_repeats);
        auto v_heads = repeat_kv_heads(
            ctx,
            TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v),
            kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    }
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention) {
        context = activation_cast(ctx, context, config_.activation_cast);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {
                            config_.num_attention_heads * dim,
                            config_.hidden_size,
                            weights.self_attention.out_bias.has_value(),
                            config_.projection_precision,
                        })
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, weights.self_attention.out_bias});
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention_output) {
        attn_out = activation_cast(ctx, attn_out, config_.activation_cast);
    }
    auto x = AddModule{}.build(ctx, input, attn_out);
    if (config_.activation_cast.enabled && config_.activation_cast.after_residual) {
        x = activation_cast(ctx, x, config_.activation_cast);
    }

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                               config_.activation_cast.norm_intermediate_type})
                     .build(ctx, x, weights.post_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_ffn_norm) {
        ff_in = activation_cast(ctx, ff_in, config_.activation_cast);
    }
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    auto output = AddModule{}.build(ctx, x, ff);
    if (config_.activation_cast.enabled && config_.activation_cast.after_output) {
        output = activation_cast(ctx, output, config_.activation_cast);
    }
    return {output, k, v};
}

DecoderLayerOutputs DecoderLayerModule::build_with_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderLayerWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const std::optional<core::TensorValue> & cache_slot,
    const core::TensorValue & attention_mask) const {
    return build_static_cache_impl(ctx, graph, input, positions, weights, cache_key, cache_value,
                                   cache_slot, attention_mask, false);
}

DecoderLayerOutputs DecoderLayerModule::build_with_static_cache_block(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderLayerWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const std::optional<core::TensorValue> & cache_slot,
    const core::TensorValue & attention_mask) const {
    return build_static_cache_impl(ctx, graph, input, positions, weights, cache_key, cache_value,
                                   cache_slot, attention_mask, true);
}

DecoderLayerOutputs DecoderLayerModule::build_static_cache_impl(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderLayerWeights & weights,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const std::optional<core::TensorValue> & cache_slot,
    const core::TensorValue & attention_mask,
    bool block) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    if (block && (input.shape.dims[0] != 1 ||
        config_.runtime.static_cache.update_mode != DecoderStaticCacheUpdateMode::DirectSetRows)) {
        throw std::runtime_error("Qwen static-cache blocks require a single sequence and DirectSetRows");
    }
    const int64_t dim = require_head_dim(config_);
    const int64_t kv_repeats = config_.num_attention_heads / config_.num_key_value_heads;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                                config_.activation_cast.norm_intermediate_type})
                      .build(ctx, input, weights.input_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_input_norm) {
        x_norm = activation_cast(ctx, x_norm, config_.activation_cast);
    }
    auto qkv = build_qkv_projections(ctx, x_norm, weights, config_, dim);
    if (config_.activation_cast.enabled && config_.activation_cast.after_qkv_projection) {
        qkv.q = activation_cast(ctx, qkv.q, config_.activation_cast);
        qkv.k = activation_cast(ctx, qkv.k, config_.activation_cast);
        qkv.v = activation_cast(ctx, qkv.v, config_.activation_cast);
    }

    auto q = reshape_decoder_heads(ctx, qkv.q, config_.num_attention_heads, dim);
    auto k = reshape_decoder_heads(ctx, qkv.k, config_.num_key_value_heads, dim);
    if (config_.use_qk_norm) {
        q = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, q, weights.q_norm);
        k = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, k, weights.k_norm);
        if (config_.activation_cast.enabled && config_.activation_cast.after_qk_norm) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    auto v = reshape_decoder_heads(ctx, qkv.v, config_.num_key_value_heads, dim);

    if (config_.position_encoding == DecoderPositionEncoding::Rotary) {
        const core::TensorValue * rope_factors = weights.rope_frequency_factors.has_value()
            ? &*weights.rope_frequency_factors
            : nullptr;
        RoPEConfig rope{dim, config_.rope_type, config_.rope_theta};
        rope.intermediate_round_type = config_.activation_cast.rope_intermediate_type;
        q = RoPEModule(rope).build(ctx, q, positions, rope_factors);
        k = RoPEModule(rope).build(ctx, k, positions, rope_factors);
        if (config_.activation_cast.enabled && config_.activation_cast.after_rope) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    core::TensorValue attention_key_cache = cache_key;
    core::TensorValue attention_value_cache = cache_value;
    core::TensorValue stored_key = k;
    core::TensorValue stored_value = v;
    if (config_.runtime.static_cache.update_mode == DecoderStaticCacheUpdateMode::DirectSetRows) {
        if (!cache_slot.has_value()) {
            throw std::runtime_error("Qwen decoder direct static-cache update requires cache_slot");
        }
        const FastKVSetRowsModule set_rows({
            config_.runtime.static_cache.set_rows_mode == DecoderStaticCacheSetRowsMode::BackendViewOptimized
                ? FastKVSetRowsMode::BackendViewOptimized
                : FastKVSetRowsMode::Exact,
        });
        attention_key_cache = block ? set_rows.build_block(ctx, cache_key, k, *cache_slot)
                                    : set_rows.build(ctx, cache_key, k, *cache_slot);
        attention_value_cache = block ? set_rows.build_block(ctx, cache_value, v, *cache_slot)
                                      : set_rows.build(ctx, cache_value, v, *cache_slot);
        if (config_.activation_cast.enabled && config_.activation_cast.after_static_cache_update) {
            attention_key_cache = activation_cast(ctx, attention_key_cache, config_.activation_cast);
            attention_value_cache = activation_cast(ctx, attention_value_cache, config_.activation_cast);
        }
    } else {
        const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
        stored_key = cache_view(ctx, cache_key, scratch_slot, 1, config_.num_key_value_heads, dim);
        stored_value = cache_view(ctx, cache_value, scratch_slot, 1, config_.num_key_value_heads, dim);
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, stored_key.tensor));
        ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, stored_value.tensor));
    }

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, attention_key_cache.shape.rank}).build(ctx, attention_key_cache);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, attention_value_cache.shape.rank}).build(ctx, attention_value_cache);
    core::TensorValue context;
    const bool allow_flash = flash_branches_allowed(config_);
    const bool use_grouped_query =
        config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeatThenGroupedQuery &&
        config_.runtime.attention.grouped_query_min_steps > 0 &&
        cache_key.shape.dims[1] >= config_.runtime.attention.grouped_query_min_steps &&
        kv_repeats > 1;
    if (use_grouped_query) {
        k_heads = core::ensure_backend_addressable_layout(ctx, k_heads);
        v_heads = core::ensure_backend_addressable_layout(ctx, v_heads);
        context = attention_from_grouped_query_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            config_.num_attention_heads,
            config_.num_key_value_heads,
            attention_mask);
    } else if (!allow_flash ||
               config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeat ||
               config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeatThenGroupedQuery) {
        k_heads = repeat_kv_heads(ctx, k_heads, kv_repeats);
        v_heads = repeat_kv_heads(ctx, v_heads, kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    } else if (config_.runtime.attention.static_mode == DecoderAttentionMode::FlashGroupedViewKV) {
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    } else {
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = flash_attention_from_grouped_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    }
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention) {
        context = activation_cast(ctx, context, config_.activation_cast);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({1, block ? input.shape.dims[1] : 1, config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {
                            config_.num_attention_heads * dim,
                            config_.hidden_size,
                            weights.self_attention.out_bias.has_value(),
                            config_.projection_precision,
                        })
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, weights.self_attention.out_bias});
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention_output) {
        attn_out = activation_cast(ctx, attn_out, config_.activation_cast);
    }
    auto x = AddModule{}.build(ctx, input, attn_out);
    if (config_.activation_cast.enabled && config_.activation_cast.after_residual) {
        x = activation_cast(ctx, x, config_.activation_cast);
    }

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                               config_.activation_cast.norm_intermediate_type})
                     .build(ctx, x, weights.post_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_ffn_norm) {
        ff_in = activation_cast(ctx, ff_in, config_.activation_cast);
    }
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    auto output = AddModule{}.build(ctx, x, ff);
    if (config_.activation_cast.enabled && config_.activation_cast.after_output) {
        output = activation_cast(ctx, output, config_.activation_cast);
    }
    return {output, stored_key, stored_value};
}

DecoderLayerOutputs DecoderLayerModule::build_with_static_cache_tail_batched(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderLayerWeights & weights,
    const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & cache_slot,
        const core::TensorValue & attention_mask) const {
    (void)graph;
    validate_sequence_input(input, config_.hidden_size, "input");
    if (input.shape.dims[0] <= 0 || input.shape.dims[1] != 1) {
        throw std::runtime_error("Qwen decoder batched static-cache update requires [batch, 1, hidden] input");
    }
    if (config_.runtime.static_cache.update_mode != DecoderStaticCacheUpdateMode::DirectSetRows) {
        throw std::runtime_error("Qwen decoder batched static-cache update supports only DirectSetRows");
    }
    const int64_t dim = require_head_dim(config_);
    const int64_t kv_repeats = config_.num_attention_heads / config_.num_key_value_heads;

    auto x_norm = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                                config_.activation_cast.norm_intermediate_type})
                      .build(ctx, input, weights.input_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_input_norm) {
        x_norm = activation_cast(ctx, x_norm, config_.activation_cast);
    }
    auto qkv = build_qkv_projections(ctx, x_norm, weights, config_, dim);
    if (config_.activation_cast.enabled && config_.activation_cast.after_qkv_projection) {
        qkv.q = activation_cast(ctx, qkv.q, config_.activation_cast);
        qkv.k = activation_cast(ctx, qkv.k, config_.activation_cast);
        qkv.v = activation_cast(ctx, qkv.v, config_.activation_cast);
    }

    auto q = reshape_decoder_heads(ctx, qkv.q, config_.num_attention_heads, dim);
    auto k = reshape_decoder_heads(ctx, qkv.k, config_.num_key_value_heads, dim);
    if (config_.use_qk_norm) {
        q = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, q, weights.q_norm);
        k = RMSNormModule({dim, config_.rms_norm_eps, true, false, false, false,
                          config_.activation_cast.norm_intermediate_type}).build(ctx, k, weights.k_norm);
        if (config_.activation_cast.enabled && config_.activation_cast.after_qk_norm) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    auto v = reshape_decoder_heads(ctx, qkv.v, config_.num_key_value_heads, dim);

    if (config_.position_encoding == DecoderPositionEncoding::Rotary) {
        apply_batched_static_rope(ctx, config_, weights, positions, q, k, dim);
        if (config_.activation_cast.enabled && config_.activation_cast.after_rope) {
            q = activation_cast(ctx, q, config_.activation_cast);
            k = activation_cast(ctx, k, config_.activation_cast);
        }
    }
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    const FastKVSetRowsModule set_rows({
        config_.runtime.static_cache.set_rows_mode == DecoderStaticCacheSetRowsMode::BackendViewOptimized
            ? FastKVSetRowsMode::BackendViewOptimized
            : FastKVSetRowsMode::Exact,
    });
    auto attention_key_cache = set_rows.build(ctx, cache_key, k, cache_slot);
    auto attention_value_cache = set_rows.build(ctx, cache_value, v, cache_slot);
    if (config_.activation_cast.enabled && config_.activation_cast.after_static_cache_update) {
        attention_key_cache = activation_cast(ctx, attention_key_cache, config_.activation_cast);
        attention_value_cache = activation_cast(ctx, attention_value_cache, config_.activation_cast);
    }

    auto q_heads = TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = TransposeModule({{0, 2, 1, 3}, attention_key_cache.shape.rank}).build(ctx, attention_key_cache);
    auto v_heads = TransposeModule({{0, 2, 1, 3}, attention_value_cache.shape.rank}).build(ctx, attention_value_cache);
    core::TensorValue context;
    const bool allow_flash = flash_branches_allowed(config_);
    const bool use_grouped_query =
        config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeatThenGroupedQuery &&
        config_.runtime.attention.grouped_query_min_steps > 0 &&
        cache_key.shape.dims[1] >= config_.runtime.attention.grouped_query_min_steps &&
        kv_repeats > 1;
    if (use_grouped_query) {
        k_heads = core::ensure_backend_addressable_layout(ctx, k_heads);
        v_heads = core::ensure_backend_addressable_layout(ctx, v_heads);
        context = attention_from_grouped_query_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            config_.num_attention_heads,
            config_.num_key_value_heads,
            attention_mask);
    } else if (!allow_flash ||
               config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeat ||
               config_.runtime.attention.static_mode == DecoderAttentionMode::ManualRepeatThenGroupedQuery) {
        k_heads = repeat_kv_heads(ctx, k_heads, kv_repeats);
        v_heads = repeat_kv_heads(ctx, v_heads, kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    } else if (config_.runtime.attention.static_mode == DecoderAttentionMode::FlashGroupedViewKV) {
        context = flash_attention_from_grouped_heads_view_kv(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    } else {
        k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
        v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
        context = flash_attention_from_grouped_heads(
            ctx,
            q_heads,
            k_heads,
            v_heads,
            dim,
            attention_mask,
            config_.attention_precision);
    }
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention) {
        context = activation_cast(ctx, context, config_.activation_cast);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.num_attention_heads * dim}));

    auto attn_out = LinearModule(
                        {
                            config_.num_attention_heads * dim,
                            config_.hidden_size,
                            weights.self_attention.out_bias.has_value(),
                            config_.projection_precision,
                        })
                        .build(
                            ctx,
                            context,
                            {weights.self_attention.out_weight, weights.self_attention.out_bias});
    if (config_.activation_cast.enabled && config_.activation_cast.after_attention_output) {
        attn_out = activation_cast(ctx, attn_out, config_.activation_cast);
    }
    auto x = AddModule{}.build(ctx, input, attn_out);
    if (config_.activation_cast.enabled && config_.activation_cast.after_residual) {
        x = activation_cast(ctx, x, config_.activation_cast);
    }

    auto ff_in = RMSNormModule({config_.hidden_size, config_.rms_norm_eps, true, false, false, false,
                               config_.activation_cast.norm_intermediate_type})
                     .build(ctx, x, weights.post_norm);
    if (config_.activation_cast.enabled && config_.activation_cast.after_ffn_norm) {
        ff_in = activation_cast(ctx, ff_in, config_.activation_cast);
    }
    auto ff = build_mlp(ctx, ff_in, config_, weights.mlp);
    auto output = AddModule{}.build(ctx, x, ff);
    if (config_.activation_cast.enabled && config_.activation_cast.after_output) {
        output = activation_cast(ctx, output, config_.activation_cast);
    }
    return {output, k, v};
}

const core::ModuleSchema & DecoderLayerModule::static_schema() noexcept {
    return kDecoderLayerSchema;
}

DecoderStackModule::DecoderStackModule(DecoderStackConfig config) : config_(config) {
    if (config_.layers <= 0) {
        throw std::runtime_error("QwenDecoderStackConfig.layers must be positive");
    }
    require_head_dim(decoder_layer_config_from_stack(config_));
}

const DecoderStackConfig & DecoderStackModule::config() const noexcept {
    return config_;
}

DecoderStackOutputs DecoderStackModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderStackWeights & weights,
    const std::optional<DecoderStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("QwenDecoderStackWeights layer count does not match config.layers");
    }
    if (prefix_state.has_value() && static_cast<int64_t>(prefix_state->layers.size()) != config_.layers) {
        throw std::runtime_error("QwenDecoderStackState layer count does not match config.layers");
    }

    auto output = input;
    DecoderStackState state;
    state.layers.reserve(weights.layers.size());
    const DecoderLayerModule layer_module(decoder_layer_config_from_stack(config_));
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto * layer_prefix = prefix_state.has_value() ? &prefix_state->layers[layer_index] : nullptr;
        auto layer = layer_module.build(
            ctx,
            output,
            positions,
            weights.layers[layer_index],
            layer_prefix != nullptr ? layer_prefix->key : std::nullopt,
            layer_prefix != nullptr ? layer_prefix->value : std::nullopt,
            attention_mask);
        output = layer.output;
        state.layers.push_back({layer.key, layer.value});
    }
    return {output, std::move(state)};
}

}  // namespace engine::modules

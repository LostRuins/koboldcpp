#pragma once

#include "engine/community_models/voxcpm1/minicpm.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::community_models::voxcpm1 {
namespace {

namespace binding = engine::modules::binding;

constexpr size_t kDefaultGraphNodes = 65536;

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

engine::core::TensorValue
ensure_contiguous(engine::core::ModuleBuildContext &ctx,
                  const engine::core::TensorValue &value) {
  return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue reshape_heads(engine::core::ModuleBuildContext &ctx,
                                        const engine::core::TensorValue &input,
                                        int64_t heads, int64_t dim) {
  return engine::core::reshape_tensor(
      ctx, ensure_contiguous(ctx, input),
      engine::core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

engine::core::TensorValue
repeat_kv_heads(engine::core::ModuleBuildContext &ctx,
                const engine::core::TensorValue &input, int64_t repeats) {
  if (repeats == 1) {
    return input;
  }
  std::vector<engine::core::TensorValue> heads;
  heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
  for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
    auto one = engine::modules::SliceModule({1, head, 1}).build(ctx, input);
    for (int64_t rep = 0; rep < repeats; ++rep) {
      heads.push_back(one);
    }
  }
  auto output = heads.front();
  for (size_t i = 1; i < heads.size(); ++i) {
    output = engine::modules::ConcatModule({1}).build(ctx, output, heads[i]);
  }
  return output;
}

engine::core::TensorValue scale_tensor(engine::core::ModuleBuildContext &ctx,
                                       const engine::core::TensorValue &input,
                                       float scale) {
  if (scale == 1.0F) {
    return input;
  }
  return engine::core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, scale),
                                   input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue
apply_minicpm_rope(engine::core::ModuleBuildContext &ctx,
                   const engine::core::TensorValue &input,
                   const engine::core::TensorValue &positions,
                   const VoxCPM1MiniCPMWeights &weights) {
  const auto &config = weights.config;
  if (config.no_rope) {
    return input;
  }
  if (!weights.rope_factors.has_value()) {
    throw std::runtime_error("VoxCPM1 MiniCPM graph missing RoPE factors");
  }
  const int64_t dim = head_dim(config);
  return engine::core::wrap_tensor(
      ggml_rope_ext(ctx.ggml, input.tensor, positions.tensor,
                    weights.rope_factors->tensor, static_cast<int>(dim),
                    GGML_ROPE_TYPE_NEOX,
                    static_cast<int>(
                        config.rope_scaling.original_max_position_embeddings),
                    config.rope_theta, 1.0F, 0.0F, weights.rope_attn_factor,
                    0.0F, 0.0F),
      input.shape, input.type);
}

engine::core::TensorValue
attention_from_heads(engine::core::ModuleBuildContext &ctx,
                     const engine::core::TensorValue &q_heads,
                     const engine::core::TensorValue &k_heads,
                     const engine::core::TensorValue &v_heads, int64_t dim,
                     bool is_causal) {
  const engine::modules::MatMulModule matmul;
  auto scores = matmul.build(
      ctx, q_heads,
      engine::modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank})
          .build(ctx, k_heads));
  scores = engine::core::wrap_tensor(
      ggml_scale(ctx.ggml, scores.tensor,
                 1.0F / std::sqrt(static_cast<float>(dim))),
      scores.shape, GGML_TYPE_F32);
  if (is_causal) {
    scores = engine::core::wrap_tensor(
        ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape,
        GGML_TYPE_F32);
  }
  scores = ensure_contiguous(ctx, scores);
  auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor),
                                        scores.shape, GGML_TYPE_F32);
  return matmul.build(ctx, attn, v_heads);
}

[[maybe_unused]] engine::core::TensorValue flash_attention_from_grouped_heads(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &q_heads,
    const engine::core::TensorValue &k_heads,
    const engine::core::TensorValue &v_heads, int64_t dim,
    const engine::core::TensorValue &attention_mask) {
  const auto q = ensure_contiguous(ctx, q_heads);
  const auto k = ensure_contiguous(ctx, k_heads);
  const auto v = ensure_contiguous(ctx, v_heads);
  auto *flash = ggml_flash_attn_ext(
      ctx.ggml, q.tensor, k.tensor, v.tensor, attention_mask.tensor,
      1.0F / std::sqrt(static_cast<float>(dim)), 0.0F, 0.0F);
  ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
  return engine::core::wrap_tensor(
      flash,
      engine::core::TensorShape::from_dims(
          {q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], dim}),
      GGML_TYPE_F32);
}

engine::core::TensorValue
minicpm_layer(engine::core::ModuleBuildContext &ctx,
              const engine::core::TensorValue &input,
              const engine::core::TensorValue &positions,
              const VoxCPM1MiniCPMLayerWeights &layer,
              const VoxCPM1MiniCPMWeights &weights, bool is_causal) {
  const auto &config = weights.config;
  const int64_t dim = head_dim(config);
  const int64_t kv_repeats =
      config.num_attention_heads / config.num_key_value_heads;
  const engine::modules::AddModule add;
  auto hidden = engine::modules::RMSNormModule(
                    {config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, layer.input_norm);
  auto q = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_attention_heads * dim, false))
               .build(ctx, hidden, layer.q_proj);
  auto k = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.k_proj);
  auto v = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.v_proj);
  q = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, q, config.num_attention_heads, dim),
                         positions, weights);
  k = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, k, config.num_key_value_heads, dim),
                         positions, weights);
  v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
  auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank})
                     .build(ctx, q);
  auto k_heads = repeat_kv_heads(
      ctx,
      engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank})
          .build(ctx, k),
      kv_repeats);
  auto v_heads = repeat_kv_heads(
      ctx,
      engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank})
          .build(ctx, v),
      kv_repeats);
  auto context =
      attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, is_causal);
  context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank})
                .build(ctx, context);
  context = engine::core::reshape_tensor(
      ctx, ensure_contiguous(ctx, context),
      engine::core::TensorShape::from_dims({input.shape.dims[0],
                                            input.shape.dims[1],
                                            config.num_attention_heads * dim}));
  auto attn = engine::modules::LinearModule(
                  binding::linear_config(config.num_attention_heads * dim,
                                         config.hidden_size, false))
                  .build(ctx, context, layer.o_proj);
  auto x =
      add.build(ctx, input,
                scale_tensor(ctx, attn,
                             config.use_mup ? config.scale_depth /
                                                  std::sqrt(static_cast<float>(
                                                      config.num_hidden_layers))
                                            : 1.0F));

  hidden = engine::modules::RMSNormModule(
               {config.hidden_size, config.rms_norm_eps, true, false})
               .build(ctx, x, layer.post_norm);
  auto gate = engine::modules::LinearModule(
                  binding::linear_config(config.hidden_size,
                                         config.intermediate_size, false))
                  .build(ctx, hidden, layer.gate_proj);
  gate = engine::modules::SiluModule{}.build(ctx, gate);
  auto up = engine::modules::LinearModule(
                binding::linear_config(config.hidden_size,
                                       config.intermediate_size, false))
                .build(ctx, hidden, layer.up_proj);
  auto gated = engine::modules::MulModule{}.build(ctx, gate, up);
  auto ff = engine::modules::LinearModule(
                binding::linear_config(config.intermediate_size,
                                       config.hidden_size, false))
                .build(ctx, gated, layer.down_proj);
  return add.build(
      ctx, x,
      scale_tensor(ctx, ff,
                   config.use_mup
                       ? config.scale_depth / std::sqrt(static_cast<float>(
                                                  config.num_hidden_layers))
                       : 1.0F));
}

[[maybe_unused]] engine::core::TensorValue
minicpm_transformer(engine::core::ModuleBuildContext &ctx,
                    engine::core::TensorValue input,
                    const engine::core::TensorValue &positions,
                    const VoxCPM1MiniCPMWeights &weights, bool is_causal) {
  for (size_t li = 0; li < weights.layers.size(); ++li) {
    input = minicpm_layer(ctx, input, positions, weights.layers[li], weights,
                          is_causal);
  }
  return engine::modules::RMSNormModule({weights.config.hidden_size,
                                         weights.config.rms_norm_eps, true,
                                         false})
      .build(ctx, input, weights.norm);
}

} // namespace
} // namespace engine::community_models::voxcpm1


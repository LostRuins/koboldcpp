#include "engine/framework/modules/speech_encoders/sanm.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace engine::modules {
namespace {

void validate_config(const SanmBlockConfig &config) {
  if (config.input_size <= 0 || config.model_size <= 0 ||
      config.num_heads <= 0 || config.ffn_size <= 0 ||
      config.fsmn_kernel_size <= 0) {
    throw std::runtime_error("SAN-M block dimensions must be positive");
  }
  if (config.model_size % config.num_heads != 0) {
    throw std::runtime_error(
        "SAN-M model size must be divisible by the head count");
  }
  if (config.fsmn_kernel_size % 2 == 0) {
    throw std::runtime_error("SAN-M FSMN kernel size must be odd");
  }
  if (!std::isfinite(config.layer_norm_eps) ||
      !(config.layer_norm_eps > 0.0F)) {
    throw std::runtime_error(
        "SAN-M layer norm epsilon must be finite and positive");
  }
}

core::TensorValue reshape_heads(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                int64_t num_heads, int64_t head_dim) {
  const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
  return core::reshape_tensor(
      ctx, contiguous,
      core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
}

struct SanmAttentionBranch {
  core::TensorValue output;
};

SanmAttentionBranch build_attention_branch(core::ModuleBuildContext &ctx,
                                           const core::TensorValue &normalized,
                                           const SanmBlockWeightsView &weights,
                                           const SanmBlockConfig &config) {
  const int64_t head_dim = config.model_size / config.num_heads;
  const LinearModule q_projection(
      {config.input_size, config.model_size, true, GGML_PREC_F32});
  const LinearModule k_projection(
      {config.input_size, config.model_size, true, GGML_PREC_F32});
  const LinearModule v_projection(
      {config.input_size, config.model_size, true, GGML_PREC_F32});
  const LinearModule output_projection(
      {config.model_size, config.model_size, true, GGML_PREC_F32});

  auto query = q_projection.build(ctx, normalized, weights.query_projection);
  auto key = k_projection.build(ctx, normalized, weights.key_projection);
  auto value = v_projection.build(ctx, normalized, weights.value_projection);

  query = reshape_heads(ctx, query, config.num_heads, head_dim);
  key = reshape_heads(ctx, key, config.num_heads, head_dim);
  auto value_heads = reshape_heads(ctx, value, config.num_heads, head_dim);
  query = TransposeModule({{0, 2, 1, 3}, query.shape.rank}).build(ctx, query);
  key = TransposeModule({{0, 2, 1, 3}, key.shape.rank}).build(ctx, key);
  value_heads = TransposeModule({{0, 2, 1, 3}, value_heads.shape.rank})
                    .build(ctx, value_heads);

  auto attention =
      ScaledDotProductAttentionModule({
                                          head_dim,
                                          config.attention_lowering,
                                          GGML_PREC_F32,
                                          AttentionCausality::NonCausal,
                                      })
          .build(ctx, query, key, value_heads);
  attention = core::ensure_backend_addressable_layout(ctx, attention);
  attention =
      core::reshape_tensor(ctx, attention,
                           core::TensorShape::from_dims(
                               {normalized.shape.dims[0],
                                normalized.shape.dims[1], config.model_size}));
  attention = output_projection.build(ctx, attention,
                                      weights.attention_output_projection);

  auto value_bct =
      TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
  value_bct = core::ensure_backend_addressable_layout(ctx, value_bct);
  auto fsmn = DepthwiseConv1dModule(
                  {
                      config.model_size,
                      config.fsmn_kernel_size,
                      1,
                      static_cast<int>((config.fsmn_kernel_size - 1) / 2),
                      1,
                      false,
                  })
                  .build(ctx, value_bct, {weights.fsmn_weight, std::nullopt});
  fsmn = TransposeModule({{0, 2, 1, 3}, fsmn.shape.rank}).build(ctx, fsmn);
  fsmn = AddModule{}.build(ctx, fsmn, value);
  return {AddModule{}.build(ctx, attention, fsmn)};
}

core::TensorValue build_ffn_residual(core::ModuleBuildContext &ctx,
                                     const core::TensorValue &residual,
                                     const SanmBlockWeightsView &weights,
                                     const SanmBlockConfig &config) {
  auto hidden =
      sanm_layer_norm(ctx, residual, weights.final_norm, config.layer_norm_eps);
  hidden =
      LinearModule({config.model_size, config.ffn_size, true, GGML_PREC_F32})
          .build(ctx, hidden, weights.ffn_input_projection);
  hidden = ReluModule{}.build(ctx, hidden);
  hidden =
      LinearModule({config.ffn_size, config.model_size, true, GGML_PREC_F32})
          .build(ctx, hidden, weights.ffn_output_projection);
  return AddModule{}.build(ctx, residual, hidden);
}

core::TensorValue build_block(core::ModuleBuildContext &ctx,
                              const core::TensorValue &input,
                              const SanmBlockWeightsView &weights,
                              const SanmBlockConfig &config,
                              bool add_input_residual) {
  validate_config(config);
  core::validate_rank_between(input, 3, 3, "SAN-M input");
  core::validate_last_dim(input, config.input_size, "SAN-M input");
  if (add_input_residual && config.input_size != config.model_size) {
    throw std::runtime_error(
        "SAN-M residual block requires input_size == model_size");
  }

  const auto normalized = sanm_layer_norm(
      ctx, input, weights.self_attention_norm, config.layer_norm_eps);
  auto residual =
      build_attention_branch(ctx, normalized, weights, config).output;
  if (add_input_residual) {
    residual = AddModule{}.build(ctx, input, residual);
  }
  return build_ffn_residual(ctx, residual, weights, config);
}

} // namespace

core::TensorValue sanm_layer_norm(core::ModuleBuildContext &ctx,
                                  const core::TensorValue &input,
                                  const NormWeights &weights, float epsilon) {
  core::validate_rank_between(input, 2, 4, "SAN-M layer norm input");
  if (!std::isfinite(epsilon) || !(epsilon > 0.0F)) {
    throw std::runtime_error(
        "SAN-M layer norm epsilon must be finite and positive");
  }
  return LayerNormModule({input.shape.last_dim(), epsilon, true, true})
      .build(ctx, input, weights);
}

core::TensorValue sanm_projection_block(core::ModuleBuildContext &ctx,
                                        const core::TensorValue &input,
                                        const SanmBlockWeightsView &weights,
                                        const SanmBlockConfig &config) {
  return build_block(ctx, input, weights, config, false);
}

core::TensorValue sanm_residual_block(core::ModuleBuildContext &ctx,
                                      const core::TensorValue &input,
                                      const SanmBlockWeightsView &weights,
                                      const SanmBlockConfig &config) {
  return build_block(ctx, input, weights, config, true);
}

} // namespace engine::modules

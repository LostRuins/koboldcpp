#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstdint>

namespace engine::modules {

struct SanmBlockWeightsView {
  NormWeights self_attention_norm;
  LinearWeights query_projection;
  LinearWeights key_projection;
  LinearWeights value_projection;
  LinearWeights attention_output_projection;
  core::TensorValue fsmn_weight;
  NormWeights final_norm;
  LinearWeights ffn_input_projection;
  LinearWeights ffn_output_projection;
};

struct SanmBlockConfig {
  int64_t input_size = 0;
  int64_t model_size = 0;
  int64_t num_heads = 0;
  int64_t ffn_size = 0;
  int64_t fsmn_kernel_size = 0;
  float layer_norm_eps = 1.0e-5F;
  ScaledDotProductAttentionLowering attention_lowering =
      ScaledDotProductAttentionLowering::Explicit;
};

core::TensorValue sanm_layer_norm(core::ModuleBuildContext &ctx,
                                  const core::TensorValue &input,
                                  const NormWeights &weights, float epsilon);

core::TensorValue sanm_projection_block(core::ModuleBuildContext &ctx,
                                        const core::TensorValue &input,
                                        const SanmBlockWeightsView &weights,
                                        const SanmBlockConfig &config);

core::TensorValue sanm_residual_block(core::ModuleBuildContext &ctx,
                                      const core::TensorValue &input,
                                      const SanmBlockWeightsView &weights,
                                      const SanmBlockConfig &config);

} // namespace engine::modules

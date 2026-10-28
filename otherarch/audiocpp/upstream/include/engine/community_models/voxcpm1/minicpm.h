#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/community_models/voxcpm1/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::community_models::voxcpm1 {

struct VoxCPM1MiniCPMLayerWeights {
  engine::modules::NormWeights input_norm;
  engine::modules::LinearWeights q_proj;
  engine::modules::LinearWeights k_proj;
  engine::modules::LinearWeights v_proj;
  engine::modules::LinearWeights o_proj;
  engine::modules::NormWeights post_norm;
  engine::modules::LinearWeights gate_proj;
  engine::modules::LinearWeights up_proj;
  engine::modules::LinearWeights down_proj;
};

struct VoxCPM1MiniCPMWeights {
  VoxCPM1MiniCPMConfig config;
  std::vector<VoxCPM1MiniCPMLayerWeights> layers;
  engine::modules::NormWeights norm;
  std::optional<engine::core::TensorValue> token_embedding;
  std::optional<engine::core::TensorValue> rope_factors;
  float rope_attn_factor = 1.0F;
};

struct VoxCPM1FeatEncoderWeights {
  engine::core::TensorValue special_token;
  engine::modules::LinearWeights in_proj;
  VoxCPM1MiniCPMWeights encoder;
};

struct VoxCPM1DiTWeights {
  engine::modules::LinearWeights in_proj;
  engine::modules::LinearWeights cond_proj;
  engine::modules::LinearWeights out_proj;
  engine::modules::LinearWeights time_mlp_1;
  engine::modules::LinearWeights time_mlp_2;
  engine::modules::LinearWeights delta_time_mlp_1;
  engine::modules::LinearWeights delta_time_mlp_2;
  VoxCPM1MiniCPMWeights decoder;
};

struct VoxCPM1ProjectionWeights {
  engine::modules::LinearWeights fsq_in_proj;
  engine::modules::LinearWeights fsq_out_proj;
  engine::modules::LinearWeights enc_to_lm_proj;
  engine::modules::LinearWeights lm_to_dit_proj;
  engine::modules::LinearWeights res_to_dit_proj;
  engine::modules::LinearWeights stop_proj;
  engine::modules::LinearWeights stop_head;
};

struct VoxCPM1ModelWeights {
  std::shared_ptr<engine::core::BackendWeightStore> store;
  VoxCPM1MiniCPMWeights base_lm;
  VoxCPM1MiniCPMWeights residual_lm;
  VoxCPM1FeatEncoderWeights feat_encoder;
  VoxCPM1DiTWeights dit;
  VoxCPM1ProjectionWeights projections;
};

int64_t head_dim(const VoxCPM1MiniCPMConfig &config);

enum class VoxCPM1MiniCPMKind {
  BaseLM,
  ResidualLM,
};

struct VoxCPM1MiniCPMStepOutput {
  std::vector<float> hidden;
  int64_t position = 0;
};

struct VoxCPM1PromptPrefillInput {
  std::vector<float> input_embeddings;
  std::vector<float> current_embeddings;
  std::vector<float> text_mask;
  std::vector<float> audio_mask;
  int64_t steps = 0;
};

struct VoxCPM1PromptPrefillOutput {
  std::vector<float> lm_hidden;
  std::vector<float> residual_hidden;
  engine::runtime::TransformerKVState base_state;
  engine::runtime::TransformerKVState residual_state;
};

class VoxCPM1WeightsRuntime final {
public:
  VoxCPM1WeightsRuntime(std::shared_ptr<const VoxCPM1Assets> assets,
                        engine::core::ExecutionContext &execution_context,
                        size_t weight_context_bytes,
                        engine::assets::TensorStorageType weight_storage_type);
  ~VoxCPM1WeightsRuntime();

  const VoxCPM1Assets &assets() const noexcept;
  const VoxCPM1ModelWeights &weights() const noexcept;
  ggml_backend_t backend() const noexcept;
  int threads() const noexcept;
  bool weights_uploaded() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1TextEmbeddingRuntime final {
public:
  VoxCPM1TextEmbeddingRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      size_t graph_context_bytes,
      bool mem_saver = false);
  ~VoxCPM1TextEmbeddingRuntime();

  std::vector<float> embed_token(int32_t token_id);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1PromptPrefillRuntime final {
public:
  VoxCPM1PromptPrefillRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      size_t graph_context_bytes,
      bool mem_saver = false);
  ~VoxCPM1PromptPrefillRuntime();

  VoxCPM1PromptPrefillOutput run(const VoxCPM1PromptPrefillInput &input);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM1MiniCPMStepRuntime final {
public:
  VoxCPM1MiniCPMStepRuntime(
      std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
      VoxCPM1MiniCPMKind kind, int64_t cache_steps,
      size_t graph_context_bytes);
  ~VoxCPM1MiniCPMStepRuntime();

  void reset();
  void import_state(const engine::runtime::TransformerKVState &state);
  engine::runtime::TransformerKVState export_state() const;
  VoxCPM1MiniCPMStepOutput run_step(const std::vector<float> &embedding);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::community_models::voxcpm1


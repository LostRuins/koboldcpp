#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/voxcpm2/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::voxcpm2 {

struct VoxCPM2MiniCPMLayerWeights {
  engine::modules::NormWeights input_norm;
  engine::modules::LinearWeights q_proj;
  engine::modules::LinearWeights k_proj;
  engine::modules::LinearWeights v_proj;
  engine::modules::LinearWeights o_proj;
  engine::modules::NormWeights post_norm;
  engine::modules::LinearWeights gate_proj;
  engine::modules::LinearWeights up_proj;
  engine::modules::LinearWeights down_proj;
  std::optional<engine::modules::LinearWeights> qkv_proj;
  std::optional<engine::modules::LinearWeights> gate_up_proj;
};

struct VoxCPM2MiniCPMWeights {
  VoxCPM2MiniCPMConfig config;
  std::vector<VoxCPM2MiniCPMLayerWeights> layers;
  engine::modules::NormWeights norm;
  std::optional<engine::core::TensorValue> token_embedding;
  std::optional<engine::core::TensorValue> rope_factors;
  float rope_attn_factor = 1.0F;
};

struct VoxCPM2FeatEncoderWeights {
  engine::core::TensorValue special_token;
  engine::modules::LinearWeights in_proj;
  VoxCPM2MiniCPMWeights encoder;
};

struct VoxCPM2DiTWeights {
  engine::modules::LinearWeights in_proj;
  engine::modules::LinearWeights cond_proj;
  engine::modules::LinearWeights out_proj;
  engine::modules::LinearWeights time_mlp_1;
  engine::modules::LinearWeights time_mlp_2;
  engine::modules::LinearWeights delta_time_mlp_1;
  engine::modules::LinearWeights delta_time_mlp_2;
  VoxCPM2MiniCPMWeights decoder;
};

struct VoxCPM2ProjectionWeights {
  engine::modules::LinearWeights fsq_in_proj;
  engine::modules::LinearWeights fsq_out_proj;
  engine::modules::LinearWeights enc_to_lm_proj;
  engine::modules::LinearWeights lm_to_dit_proj;
  engine::modules::LinearWeights res_to_dit_proj;
  engine::modules::LinearWeights fusion_concat_proj;
  engine::modules::LinearWeights stop_proj;
  engine::modules::LinearWeights stop_head;
};

struct VoxCPM2ModelWeights {
  std::shared_ptr<engine::core::BackendWeightStore> store;
  VoxCPM2MiniCPMWeights base_lm;
  VoxCPM2MiniCPMWeights residual_lm;
  VoxCPM2FeatEncoderWeights feat_encoder;
  VoxCPM2DiTWeights dit;
  VoxCPM2ProjectionWeights projections;
};

int64_t head_dim(const VoxCPM2MiniCPMConfig &config);

enum class VoxCPM2MiniCPMKind {
  BaseLM,
  ResidualLM,
};

struct VoxCPM2MiniCPMStepOutput {
  std::vector<float> hidden;
  int64_t position = 0;
};

struct VoxCPM2PromptPrefillInput {
  std::vector<float> input_embeddings;
  std::vector<float> current_embeddings;
  std::vector<float> text_mask;
  std::vector<float> audio_mask;
  int64_t steps = 0;
};

struct VoxCPM2PromptPrefillOutput {
  std::vector<float> lm_hidden;
  std::vector<float> residual_hidden;
  engine::runtime::TransformerKVState base_state;
  engine::runtime::TransformerKVState residual_state;
};

class VoxCPM2WeightsRuntime final {
public:
  VoxCPM2WeightsRuntime(std::shared_ptr<const VoxCPM2Assets> assets,
                        engine::core::ExecutionContext &execution_context,
                        size_t weight_context_bytes,
                        engine::assets::TensorStorageType weight_storage_type,
                        bool optimized = false);
  ~VoxCPM2WeightsRuntime();

  const VoxCPM2Assets &assets() const noexcept;
  const VoxCPM2ModelWeights &weights() const noexcept;
  ggml_backend_t backend() const noexcept;
  int threads() const noexcept;
  bool weights_uploaded() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM2TextEmbeddingRuntime final {
public:
  VoxCPM2TextEmbeddingRuntime(
      std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
      size_t graph_context_bytes,
      bool mem_saver = false);
  ~VoxCPM2TextEmbeddingRuntime();

  std::vector<float> embed_token(int32_t token_id);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM2PromptPrefillRuntime final {
public:
  VoxCPM2PromptPrefillRuntime(
      std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
      size_t graph_context_bytes);
  ~VoxCPM2PromptPrefillRuntime();

  VoxCPM2PromptPrefillOutput run(const VoxCPM2PromptPrefillInput &input);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VoxCPM2MiniCPMStepRuntime final {
public:
  VoxCPM2MiniCPMStepRuntime(
      std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
      VoxCPM2MiniCPMKind kind, int64_t cache_steps,
      size_t graph_context_bytes);
  ~VoxCPM2MiniCPMStepRuntime();

  void reset();
  void import_state(const engine::runtime::TransformerKVState &state);
  engine::runtime::TransformerKVState export_state() const;
  int64_t cache_steps() const;
  int64_t valid_steps() const;
  VoxCPM2MiniCPMStepOutput run_step(const std::vector<float> &embedding);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::voxcpm2


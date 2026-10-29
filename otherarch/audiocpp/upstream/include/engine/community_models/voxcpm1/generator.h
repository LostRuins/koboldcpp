#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/voxcpm1/types.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace engine::core {
class ExecutionContext;
}

namespace engine::community_models::voxcpm1 {

struct VoxCPM1Assets;

struct VoxCPM1FeatureGeneratorConfig {
  size_t weight_context_bytes = 3ull * 1024ull * 1024ull * 1024ull;
  size_t text_embedding_graph_context_bytes = 64ull * 1024ull * 1024ull;
  size_t lm_step_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t projection_graph_context_bytes = 256ull * 1024ull * 1024ull;
  size_t local_encoder_graph_context_bytes = 512ull * 1024ull * 1024ull;
  size_t dit_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t prompt_cache_slots = 1;
  bool mem_saver = false;
  engine::assets::TensorStorageType weight_storage_type =
      engine::assets::TensorStorageType::Native;
};

class VoxCPM1FeatureGeneratorRuntime final {
public:
  VoxCPM1FeatureGeneratorRuntime(
      std::shared_ptr<const VoxCPM1Assets> assets,
      engine::core::ExecutionContext &execution_context,
      VoxCPM1FeatureGeneratorConfig config = {});
  ~VoxCPM1FeatureGeneratorRuntime();

  VoxCPM1Result generate_zero_shot(const std::string &text,
                                   const VoxCPM1GenerationOptions &options);
  VoxCPM1Result generate(const std::string &text,
                         const VoxCPM1EncodedPrompt *prompt,
                         const VoxCPM1GenerationOptions &options);
  VoxCPM1StreamingResult
  generate_streaming(const std::string &text,
                     const VoxCPM1EncodedPrompt *prompt,
                     const VoxCPM1GenerationOptions &options,
                     const std::function<void(const VoxCPM1StreamingChunk &)>
                         &chunk_callback = nullptr);
  void release_runtime_memory();
  void release_text_length_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::community_models::voxcpm1


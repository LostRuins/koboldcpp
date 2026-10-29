#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/voxcpm2/types.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::voxcpm2 {

struct VoxCPM2Assets;

struct VoxCPM2FeatureGeneratorConfig {
  size_t weight_context_bytes = 3ull * 1024ull * 1024ull * 1024ull;
  size_t text_embedding_graph_context_bytes = 64ull * 1024ull * 1024ull;
  size_t lm_step_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t projection_graph_context_bytes = 256ull * 1024ull * 1024ull;
  size_t local_encoder_graph_context_bytes = 512ull * 1024ull * 1024ull;
  size_t dit_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t prompt_cache_slots = 1;
  bool mem_saver = true;
  engine::assets::TensorStorageType weight_storage_type =
      engine::assets::TensorStorageType::Native;
  bool optimized = false;
};

class VoxCPM2FeatureGeneratorRuntime final {
public:
  VoxCPM2FeatureGeneratorRuntime(
      std::shared_ptr<const VoxCPM2Assets> assets,
      engine::core::ExecutionContext &execution_context,
      VoxCPM2FeatureGeneratorConfig config = {});
  ~VoxCPM2FeatureGeneratorRuntime();

  VoxCPM2Result generate_zero_shot(const std::string &text,
                                   const VoxCPM2GenerationOptions &options);
  VoxCPM2Result generate(const std::string &text,
                         const VoxCPM2EncodedPrompt *prompt,
                         const VoxCPM2GenerationOptions &options);
  VoxCPM2StreamingResult
  generate_streaming(const std::string &text,
                     const VoxCPM2EncodedPrompt *prompt,
                     const VoxCPM2GenerationOptions &options,
                     const std::function<void(const VoxCPM2StreamingChunk &)>
                         &chunk_callback = nullptr);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::voxcpm2


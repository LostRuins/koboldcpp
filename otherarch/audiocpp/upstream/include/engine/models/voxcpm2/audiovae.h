#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/types.h"

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

struct VoxCPM2AudioVAEDecoderConfig {
  size_t weight_context_bytes = 768ull * 1024ull * 1024ull;
  size_t graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t encoder_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  int64_t latent_frame_capacity = 0;
  int64_t encoder_sample_capacity = 240000;
  engine::assets::TensorStorageType weight_storage_type =
      engine::assets::TensorStorageType::Native;
  int64_t tile_frames = 0;
};

class VoxCPM2AudioVAEDecoderRuntime final {
public:
  VoxCPM2AudioVAEDecoderRuntime(
      std::shared_ptr<const VoxCPM2Assets> assets,
      engine::core::ExecutionContext &execution_context,
      VoxCPM2AudioVAEDecoderConfig config = {});
  ~VoxCPM2AudioVAEDecoderRuntime();

  runtime::AudioBuffer decode_features(const std::vector<float> &features,
                                       int64_t patches);
  VoxCPM2EncodedPrompt encode_prompt_audio(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);
  void release_runtime_memory();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::voxcpm2


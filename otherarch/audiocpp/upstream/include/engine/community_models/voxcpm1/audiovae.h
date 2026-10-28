#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/voxcpm1/assets.h"
#include "engine/community_models/voxcpm1/types.h"

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

struct VoxCPM1AudioVAEDecoderConfig {
  size_t weight_context_bytes = 768ull * 1024ull * 1024ull;
  size_t graph_context_bytes = 1024ull * 1024ull * 1024ull;
  size_t encoder_graph_context_bytes = 1024ull * 1024ull * 1024ull;
  int64_t latent_frame_capacity = 0;
  int64_t encoder_sample_capacity = 240000;
  engine::assets::TensorStorageType weight_storage_type =
      engine::assets::TensorStorageType::F32;
};

// Streaming decode state for maintaining convolution state across patches
struct AudioVAEStreamingDecodeState {
  struct SlotSpec {
    int64_t frames = 0;
    int64_t channels = 0;
    std::string name;
  };

  struct Slot {
    int64_t frames = 0;
    int64_t channels = 0;
    ggml_tensor* tensor = nullptr;
    std::string name;
  };

  struct PendingUpdate {
    size_t slot_index = 0;
    ggml_tensor* tensor = nullptr;
  };

  AudioVAEStreamingDecodeState() = default;
  ~AudioVAEStreamingDecodeState();
  AudioVAEStreamingDecodeState(const AudioVAEStreamingDecodeState&) = delete;
  AudioVAEStreamingDecodeState& operator=(const AudioVAEStreamingDecodeState&) = delete;
  AudioVAEStreamingDecodeState(AudioVAEStreamingDecodeState&& other) noexcept;
  AudioVAEStreamingDecodeState& operator=(AudioVAEStreamingDecodeState&& other) noexcept;

  void reset();
  void clear();
  bool initialize(const std::vector<SlotSpec>& specs, core::ExecutionContext& execution_context);
  void begin_graph();
  ggml_tensor* take_slot(int64_t frames, int64_t channels, const std::string& name);
  void queue_update(ggml_tensor* tensor);
  void build_update_graph(ggml_cgraph* graph) const;
  void publish_updates(core::ExecutionContext& execution_context);

  bool is_initialized() const { return !slots_.empty() && ctx_ != nullptr; }
  ggml_context* get_context() const { return ctx_.get(); }

private:
  struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const noexcept {
      if (ctx != nullptr) {
        ggml_free(ctx);
      }
    }
  };

  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_backend_buffer_t buffer_ = nullptr;
  std::vector<Slot> slots_;
  std::vector<PendingUpdate> pending_updates_;
  size_t cursor_ = 0;
};

class VoxCPM1AudioVAEDecoderRuntime final {
public:
  VoxCPM1AudioVAEDecoderRuntime(
      std::shared_ptr<const VoxCPM1Assets> assets,
      engine::core::ExecutionContext &execution_context,
      VoxCPM1AudioVAEDecoderConfig config = {});
  ~VoxCPM1AudioVAEDecoderRuntime();

  runtime::AudioBuffer decode_features(const std::vector<float> &features,
                                       int64_t patches);
  
  // Streaming decode: initialize state once, then call decode_streaming_step per patch
  bool supports_streaming_decode() const;
  bool initialize_streaming_decode_state(AudioVAEStreamingDecodeState& state);
  runtime::AudioBuffer decode_streaming_step(const std::vector<float>& patch_features,
                                              AudioVAEStreamingDecodeState& state);

  VoxCPM1EncodedPrompt encode_prompt_audio(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);
  void release_runtime_memory();
  void release_encoder_graph();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::community_models::voxcpm1


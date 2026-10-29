#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::voxcpm2 {

// Upper bound for VoxCPM2GenerationOptions::stream_left_context. The decode
// window grows with it and every emitted patch re-decodes the whole window.
inline constexpr int64_t kVoxCPM2MaxStreamLeftContext = 8;

struct VoxCPM2GenerationOptions {
  int64_t min_tokens = 2;
  int64_t max_tokens = 4096;
  int64_t num_inference_steps = 10;
  float guidance_scale = 2.0F;
  bool retry_badcase = true;
  int64_t retry_badcase_max_times = 3;
  float retry_badcase_ratio_threshold = 6.0F;
  uint32_t seed = 1234;
  std::string cfm_noise_file;
  // Streaming: number of preceding patches decoded together with each emitted
  // patch as left context. The AudioVAE decoder is causal, but the streaming
  // path invokes it statelessly per patch, so its causal-convolution history
  // restarts from zero padding at every patch boundary and the seams click.
  // Decoding the preceding patches in the same window rebuilds that history;
  // the session trims their audio off again. The prompt's context rows count,
  // as in the offline decode. 0 decodes every patch alone. Range
  // [0, kVoxCPM2MaxStreamLeftContext].
  int64_t stream_left_context = 3;
};

struct VoxCPM2PromptAudio {
  runtime::AudioBuffer audio;
  std::string text;
};

struct VoxCPM2EncodedPrompt {
  std::string prompt_text;
  std::vector<float> prompt_features;
  int64_t prompt_patches = 0;
  std::vector<float> reference_features;
  int64_t reference_patches = 0;
};

struct VoxCPM2Request {
  std::string text;
  std::optional<VoxCPM2PromptAudio> prompt = std::nullopt;
  std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
  VoxCPM2GenerationOptions generation;
};

struct VoxCPM2TextPrompt {
  std::string text;
  std::vector<int32_t> input_ids;
};

struct VoxCPM2Result {
  runtime::AudioBuffer audio;
  std::vector<float> generated_features;
  int64_t generated_patches = 0;
  std::vector<float> decode_features;
  int64_t decode_patches = 0;
  int64_t decode_trim_patches = 0;
};

struct VoxCPM2StreamingChunk {
  std::vector<float> decode_features;
  int64_t decode_patches = 0;
  int64_t generated_patches = 0;
  // Left-context patches at the front of decode_features: decoded with the
  // chunk to rebuild the decoder's causal history, then dropped from its audio.
  int64_t context_patches = 0;
};

struct VoxCPM2StreamingResult {
  std::vector<VoxCPM2StreamingChunk> chunks;
  int64_t generated_patches = 0;
};

} // namespace engine::models::voxcpm2


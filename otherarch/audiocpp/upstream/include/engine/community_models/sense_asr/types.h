#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::sense_asr {

struct SenseAsrAudioFeatures {
  std::vector<float> values;
  int64_t frames = 0;
  int64_t feature_dim = 0;
};

struct SenseAsrEncoderOutput {
  std::vector<float> logits;
  int64_t frames = 0;
  int64_t vocab_size = 0;
};

struct SenseAsrDecodedTokens {
  std::vector<int32_t> ids;
  std::string text;
  std::vector<std::string> tags;
  std::string language;
};

struct SenseAsrTranscriptionOptions {
  std::string language = "auto";
  bool enable_itn = true;
  bool keep_tags = false;
};

} // namespace engine::community_models::sense_asr

#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::qwen3_asr {

struct Qwen3ASRGenerationOptions {
    int64_t max_new_tokens = 512;
    bool return_timestamps = false;
    bool clamp_timestamps_to_audio = false;
};

struct Qwen3ASRRequest {
    runtime::AudioBuffer audio;
    std::string context;
    std::string language;
    Qwen3ASRGenerationOptions generation;
};

struct Qwen3ASRResult {
    std::string text;
    std::string language;
    std::vector<runtime::WordTimestamp> word_timestamps;
};

struct Qwen3ASRPrompt {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> audio_token_positions;
    std::vector<int32_t> attention_mask;
};

using Qwen3ASRAudioFeatures = modules::Qwen3AudioFeatures;
using Qwen3ASRAudioEmbeddings = modules::Qwen3AudioEmbeddings;

struct Qwen3ASRGeneratedTokens {
    std::vector<int32_t> token_ids;
};

}  // namespace engine::models::qwen3_asr

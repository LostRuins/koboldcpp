#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoiceGenerationOptions {
    int64_t max_tokens = 0;
    float guidance_scale = 1.3F;
    float max_length_times = 2.0F;
    int64_t num_inference_steps = 10;
    uint32_t seed = 1234;
    bool do_sample = false;
    float temperature = 1.0F;
    int64_t top_k = 50;
    float top_p = 1.0F;
    std::string prompt_noise_file;
    std::string diffusion_noise_file;
};

struct VibeVoiceReferenceVoiceState {
    std::vector<float> acoustic_mean;
    int64_t frames = 0;
    int64_t dim = 0;
};

struct VibeVoiceSpeakerPrompt {
    runtime::AudioBuffer audio;
    std::optional<VibeVoiceReferenceVoiceState> reference_state = std::nullopt;
};

struct VibeVoiceRequest {
    std::string text;
    std::vector<VibeVoiceSpeakerPrompt> speakers;
    VibeVoiceGenerationOptions generation;
};

struct VibeVoiceResult {
    runtime::AudioBuffer audio;
    std::vector<int32_t> generated_tokens;
};

}  // namespace engine::models::vibevoice

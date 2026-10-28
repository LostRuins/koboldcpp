#pragma once

#include "engine/framework/text/chunking.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

enum class OmniVoicePromptMode {
    AutoVoice,
    VoiceClone,
    VoiceDesign,
};

struct OmniVoiceGenerationOptions {
    std::optional<uint32_t> seed = std::nullopt;
    int64_t num_inference_steps = 32;
    float guidance_scale = 2.0F;
    float speed = 1.0F;
    std::optional<float> duration_seconds = std::nullopt;
    float t_shift = 0.1F;
    bool denoise = true;
    bool postprocess_output = true;
    float layer_penalty_factor = 5.0F;
    float position_temperature = 5.0F;
    float class_temperature = 0.0F;
    bool preprocess_prompt = true;
    float audio_chunk_duration_seconds = 15.0F;
    float audio_chunk_threshold_seconds = 30.0F;
    std::optional<int64_t> text_chunk_size = std::nullopt;
    engine::text::TextChunkMode text_chunk_mode = engine::text::TextChunkMode::TagAware;
};

struct OmniVoiceAudioTokens {
    std::vector<int32_t> token_ids;
    int64_t frames = 0;
    int64_t codebooks = 0;
    float reference_rms = 0.0F;
};

struct OmniVoiceRequest {
    std::string text;
    std::string language;
    std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
    std::optional<OmniVoiceAudioTokens> reference_audio_tokens = std::nullopt;
    std::string reference_text;
    std::string instruct;
    float reference_rms = 0.0F;
    OmniVoiceGenerationOptions generation;
};

struct OmniVoicePrompt {
    OmniVoicePromptMode mode = OmniVoicePromptMode::AutoVoice;
    std::vector<int32_t> style_token_ids;
    std::vector<int32_t> text_token_ids;
    std::string text;
    std::string language;
    std::string instruct;
    std::string reference_text;
    std::optional<OmniVoiceAudioTokens> reference_audio_tokens = std::nullopt;
    int64_t target_audio_tokens = 0;
    float reference_rms = 0.0F;
};

struct OmniVoiceGeneratedAudioTokens {
    std::vector<int32_t> token_ids;
    int64_t frames = 0;
    int64_t codebooks = 0;
    bool graph_rebuilt = false;
    int64_t graph_total_token_capacity = 0;
    int64_t graph_target_frame_capacity = 0;
    double forward_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    double scoring_ms = 0.0;
    double update_ms = 0.0;
    int64_t decode_steps = 0;
};

struct OmniVoiceResult {
    runtime::AudioBuffer audio;
};

}  // namespace engine::models::omnivoice

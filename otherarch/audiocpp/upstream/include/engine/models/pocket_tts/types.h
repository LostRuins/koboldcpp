#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::pocket_tts {

enum class VoiceSourceKind {
    NamedPreset,
    PreparedEmbedding,
    CloneAudio,
};

enum class BackendKind {
    Cpu,
    Cuda,
    BestAvailable,
};

struct ExecutionConfig {
    BackendKind backend = BackendKind::Cpu;
    int device = 0;
    int threads = 1;
};

struct VoiceConfig {
    std::string preset_name;
    std::filesystem::path embedding_path;
    std::filesystem::path clone_audio_path;
    std::optional<runtime::AudioBuffer> clone_audio;
    std::string clone_prompt_text;
    bool truncate_clone_audio = false;
};

struct ModelConfig {
    std::filesystem::path model_dir;
    std::filesystem::path config_path;
    std::string language;
};

struct GenerationRequest {
    std::string text;
    VoiceConfig voice = {};
    int max_steps = 0;
    int max_tokens = 50;
    std::optional<int64_t> text_chunk_size = std::nullopt;
    int frames_after_eos = -1;
    float temperature = 0.7F;
    float noise_clamp = -1.0F;
    float eos_threshold = -4.0F;
    uint32_t seed = 1234;
    std::vector<float> noise_schedule;
    std::filesystem::path noise_schedule_path;
};

struct GenerationResult {
    int sample_rate = 24000;
    std::vector<float> audio;
};

struct VoiceConditioningPlan {
    VoiceSourceKind source = VoiceSourceKind::NamedPreset;
    std::filesystem::path asset_path;
    std::filesystem::path clone_audio_path;
    std::optional<runtime::AudioBuffer> clone_audio;
    std::string preset_name;
    std::string clone_prompt_text;
    bool truncate_clone_audio = false;
};

}  // namespace engine::models::pocket_tts

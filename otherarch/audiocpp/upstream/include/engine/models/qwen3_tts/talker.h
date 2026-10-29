#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::qwen3_tts {

enum class Qwen3TalkerPromptMode {
    VoiceClone,
    VoiceDesign,
    CustomVoice,
};

struct Qwen3TalkerPrefill {
    Qwen3TalkerPromptMode prompt_mode = Qwen3TalkerPromptMode::VoiceClone;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> instruct_ids;
    std::vector<int32_t> reference_ids;
    std::optional<Qwen3SpeechCodes> reference_codes = std::nullopt;
    std::optional<Qwen3SpeakerEmbedding> speaker_embedding = std::nullopt;
    std::string speaker;
    std::string language = "Auto";
    bool icl_mode = false;
    bool x_vector_only_mode = false;
};

struct Qwen3TalkerCodes {
    Qwen3SpeechCodes generated_codes;
    Qwen3SpeechCodes decoder_input_codes;
};

void validate_qwen3_talker_voice_clone_prefill(
    const Qwen3TalkerPrefill & prefill,
    int64_t hidden_size);

class Qwen3TalkerWeightsRuntime;
class Qwen3TalkerStepRuntime;

class Qwen3TalkerStepRuntime {
public:
    class Impl;
    explicit Qwen3TalkerStepRuntime(std::unique_ptr<Impl> impl);
    ~Qwen3TalkerStepRuntime();

    Qwen3TalkerCodes generate(
        const Qwen3TalkerPrefill & prefill,
        const Qwen3TTSGenerationOptions & options,
        float repetition_penalty = 1.05F,
        const std::function<void(const std::vector<int32_t> &)> & on_frame = {});
    int64_t release_cached_step_graph();

private:
    std::unique_ptr<Impl> impl_;
};

class Qwen3Talker {
public:
    explicit Qwen3Talker(Qwen3TTSTalkerConfig config);

    const Qwen3TTSTalkerConfig & config() const noexcept;

    std::shared_ptr<const Qwen3TalkerWeightsRuntime> create_weights_runtime(
        std::shared_ptr<const Qwen3TTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t graph_arena_bytes,
        size_t talker_constant_context_bytes,
        size_t code_predictor_constant_context_bytes,
        engine::assets::TensorStorageType weight_storage_type,
        Qwen3TTSPerfMode perf_mode,
        int64_t kv_cache_alignment = 1) const;

    std::shared_ptr<Qwen3TalkerStepRuntime> create_step_runtime(
        std::shared_ptr<const Qwen3TalkerWeightsRuntime> weights,
        int64_t prompt_capacity,
        int64_t generation_capacity) const;

private:
    Qwen3TTSTalkerConfig config_;
};

}  // namespace engine::models::qwen3_tts

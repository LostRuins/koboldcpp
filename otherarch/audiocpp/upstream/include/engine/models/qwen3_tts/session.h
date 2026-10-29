#pragma once

#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/prompt_tts_voice_clone.h"
#include "engine/models/qwen3_tts/speaker_encoder.h"
#include "engine/models/qwen3_tts/talker.h"
#include "engine/models/qwen3_tts/tokenizer_speech_decoder.h"
#include "engine/models/qwen3_tts/tokenizer_speech_encoder.h"
#include "engine/models/qwen3_tts/tokenizer_text.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::qwen3_tts {

class Qwen3TTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::StreamingTtsSessionBase {
public:
    Qwen3TTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Qwen3TTSAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override;

    struct VoicePromptCacheKey {
        std::string reference_text;
        Qwen3VoiceCloneMode mode = Qwen3VoiceCloneMode::Icl;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
    };

    struct VoicePromptCacheKeyEqual {
        bool operator()(const VoicePromptCacheKey & lhs, const VoicePromptCacheKey & rhs) const noexcept;
    };

    struct VoicePromptCacheEntry {
        Qwen3VoiceClonePrompt prompt;
    };

    Qwen3TTSRequest make_request(const runtime::TaskRequest & request) const;
    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming);
    runtime::AudioBuffer generate_audio(
        const Qwen3TalkerPrefill & prefill,
        const Qwen3TTSGenerationOptions & options,
        const Qwen3SpeechCodes * reference_codes,
        int64_t stream_frames,
        runtime::StreamingAudioChunkPolicy stream_policy,
        double & talker_ms,
        double & decoder_ms);
    const Qwen3VoiceClonePrompt & resolve_voice_prompt(
        const Qwen3VoiceCloneInput & input,
        const Qwen3TTSVoiceClonePromptBuilder & prompt_builder);

    runtime::TaskSpec task_;
    std::shared_ptr<const Qwen3TTSAssets> assets_;
    size_t talker_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    size_t speech_encoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    size_t speech_decoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    size_t speaker_encoder_graph_arena_bytes_ = 32ull * 1024ull * 1024ull;
    // No-alloc GGML context capacities for reusable constant tensor descriptors.
    // Generous on purpose; ConstantTensorCache fits them to the host when it has
    // to, so a small machine is not asked to reserve what it does not have.
    size_t talker_constant_context_bytes_ = 4ull * 1024ull * 1024ull;
    size_t code_predictor_constant_context_bytes_ = 4ull * 1024ull * 1024ull;
    size_t speech_decoder_constant_context_bytes_ = 4ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType talker_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType speech_encoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType speech_decoder_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::F32;
    Qwen3SpeechTokenizerDecoderRknn rknn_decoder_;
    bool mem_saver_ = false;
    // Voice the concatenated codes of all text chunks in one decode pass; false voices each chunk.
    bool stitch_decode_ = false;
    Qwen3TTSPerfMode perf_mode_ = Qwen3TTSPerfMode::Standard;
    Qwen3TextTokenizer text_tokenizer_;
    Qwen3Talker talker_;
    std::shared_ptr<const Qwen3TalkerWeightsRuntime> talker_weights_;
    std::shared_ptr<Qwen3TalkerStepRuntime> talker_step_;
    core::ExecutionContext voice_prompt_context_;
    std::unique_ptr<Qwen3SpeechTokenizerDecoderRuntime> speech_decoder_;
    std::unique_ptr<Qwen3SpeechTokenizerEncoderRuntime> speech_encoder_;
    std::unique_ptr<Qwen3TTSEcapaTdnnEncoderRuntime> speaker_encoder_;
    runtime::CacheSlots<VoicePromptCacheKey, VoicePromptCacheEntry, VoicePromptCacheKeyEqual> voice_prompt_cache_;
    std::optional<VoicePromptCacheEntry> uncached_voice_prompt_;
    runtime::StreamingAudioController<int32_t> stream_audio_;
};

}  // namespace engine::models::qwen3_tts

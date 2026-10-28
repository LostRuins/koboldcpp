#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/audio_tokenizer.h"
#include "engine/models/omnivoice/generator.h"
#include "engine/models/omnivoice/postprocess.h"
#include "engine/models/omnivoice/prompt_builder.h"
#include "engine/models/omnivoice/tokenizer_text.h"

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::omnivoice {

class OmniVoiceSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    OmniVoiceSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const OmniVoiceAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    struct SessionDefaults {
        std::optional<runtime::Transcript> text = std::nullopt;
        std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
        std::optional<std::string> reference_text = std::nullopt;
        std::optional<std::string> instruct = std::nullopt;
        std::unordered_map<std::string, std::string> options;
    };

    struct ReferencePromptCacheEntry {
        bool preprocess_prompt = true;
        bool reference_text_provided = false;
        int sample_rate = 0;
        int channels = 0;
        uint64_t sample_count = 0;
        uint64_t sample_hash = 0;
        OmniVoiceAudioTokens tokens;
    };

    OmniVoiceRequest make_request(const runtime::TaskRequest & request) const;
    std::optional<runtime::AudioBuffer> resolve_reference_audio(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_reference_text(const runtime::TaskRequest & request) const;
    std::optional<std::string> resolve_instruct(const runtime::TaskRequest & request) const;
    std::unordered_map<std::string, std::string> merged_request_options(const runtime::TaskRequest & request) const;
    OmniVoiceAudioTokens resolve_reference_audio_tokens(
        const runtime::AudioBuffer & audio,
        bool preprocess_prompt,
        bool reference_text_provided);
    std::vector<std::string> plan_text_chunks(const OmniVoiceRequest & request, const OmniVoicePrompt & prompt) const;
    void initialize_streaming_request(const runtime::TaskRequest & request);
    runtime::AudioBuffer synthesize_stream_chunk(size_t chunk_index);

    runtime::TaskSpec task_;
    std::shared_ptr<const OmniVoiceAssets> assets_;
    size_t audio_tokenizer_graph_arena_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_prefill_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t generator_decode_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    size_t audio_tokenizer_weight_context_bytes_ = 128ull * 1024ull * 1024ull;
    size_t generator_weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType audio_tokenizer_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType generator_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool mem_saver_ = false;
    OmniVoiceGeneratorPerfMode generator_perf_mode_ = OmniVoiceGeneratorPerfMode::Standard;
    OmniVoiceTextTokenizer tokenizer_;
    OmniVoiceHiggsAudioV2TokenizerRuntime audio_tokenizer_;
    OmniVoicePromptBuilder prompt_builder_;
    OmniVoiceQwen3DiffusionRuntime generator_;
    OmniVoicePostprocessor postprocessor_;
    SessionDefaults session_defaults_;
    std::optional<ReferencePromptCacheEntry> reference_prompt_cache_;
    std::optional<OmniVoiceRequest> stream_request_;
    std::vector<std::string> stream_text_chunks_;
    std::optional<OmniVoiceAudioTokens> stream_first_chunk_reference_;
    std::string stream_first_chunk_text_;
    runtime::AudioBuffer stream_merged_audio_;
    size_t stream_chunk_index_ = 0;
    int64_t stream_chunk_codebooks_ = 0;
    bool stream_started_ = false;
    bool stream_has_reference_audio_ = false;
    std::chrono::steady_clock::time_point stream_wall_start_;
};

}  // namespace engine::models::omnivoice

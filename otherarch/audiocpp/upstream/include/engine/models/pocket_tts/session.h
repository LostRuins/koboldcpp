#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/pocket_tts/acoustic_model.h"
#include "engine/models/pocket_tts/audio_decoder.h"
#include "engine/models/pocket_tts/flow_lm.h"
#include "engine/models/pocket_tts/text_conditioner.h"
#include "engine/models/pocket_tts/voice_conditioner.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace engine::models::pocket_tts {

struct PocketTTSGraphCapacityConfig {
    runtime::GraphCapacityMode prompt_mode = runtime::GraphCapacityMode::Double;
    runtime::GraphCapacityMode generation_mode = runtime::GraphCapacityMode::Double;
    int64_t prompt_capacity = 0;
    int generation_capacity = 0;
    size_t flow_weight_context_bytes = 64ull * 1024ull * 1024ull;
    size_t mimi_encoder_weight_context_bytes = 64ull * 1024ull * 1024ull;
    size_t mimi_decoder_weight_context_bytes = 64ull * 1024ull * 1024ull;
    size_t flow_weights_view_context_bytes = 256ull * 1024ull * 1024ull;
    size_t flow_step_graph_context_bytes = 256ull * 1024ull * 1024ull;
    size_t mimi_encoder_graph_context_bytes = 512ull * 1024ull * 1024ull;
    size_t mimi_conv_graph_context_bytes = 32ull * 1024ull * 1024ull;
    size_t mimi_transformer_graph_context_bytes = 96ull * 1024ull * 1024ull;
    size_t mimi_tail_graph_context_bytes = 512ull * 1024ull * 1024ull;
    int64_t mimi_full_chunk_frames = 90;
    int64_t mimi_stage2_chunk_frames = 900;
    bool mimi_use_full_sequence_path = true;
    engine::assets::TensorStorageType matmul_weight_storage_type = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type = engine::assets::TensorStorageType::Native;
};

class PocketTTSSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    PocketTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const PocketTTSAssets> manifest,
        std::filesystem::path model_dir);
    ~PocketTTSSession();

    PocketTTSSession(const PocketTTSSession &) = delete;
    PocketTTSSession & operator=(const PocketTTSSession &) = delete;

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

    void prepare_generation(const GenerationRequest & request);
    GenerationResult generate(const GenerationRequest & request);
    FlowLMState prepare_voice_state(const VoiceConfig & voice);
    void export_voice_state(const VoiceConfig & voice, const std::filesystem::path & destination);

private:
    struct AcousticCapacitySelection {
        int64_t prompt_capacity = 0;
        int generation_capacity = 0;
    };

    FlowLMState resolve_prepared_voice_state(const VoiceConditioningPlan & plan);
    PocketTTSGraphCapacityConfig resolve_graph_capacity_config() const;
    runtime::MappedGraphCapacityAdapter make_prompt_capacity_adapter() const;
    runtime::MappedGraphCapacityAdapter make_generation_capacity_adapter() const;
    AcousticCapacitySelection select_acoustic_capacities(int64_t prompt_steps, int max_steps) const;
    GenerationRequest effective_request_for_run(const runtime::TaskRequest & request) const;
    bool start_next_stream_text_chunk();
    std::vector<int64_t> prepared_prompt_capacities() const;
    std::vector<int64_t> prepared_generation_capacities() const;

    runtime::TaskSpec task_;
    std::shared_ptr<const PocketTTSAssets> manifest_;
    std::filesystem::path model_dir_;
    PocketTTSGraphCapacityConfig graph_capacity_;
    std::shared_ptr<const PocketTTSBackendWeights> weights_;
    TextConditioner text_conditioner_;
    VoiceConditioner voice_conditioner_;
    AcousticModel acoustic_model_;
    AudioDecoder audio_decoder_;
    runtime::CacheSlots<std::string, FlowLMState> cached_voice_states_;
    GenerationRequest prepared_session_request_;
    runtime::GraphCapacityController prompt_capacity_controller_;
    runtime::GraphCapacityController generation_capacity_controller_;

    GenerationRequest stream_request_;
    FlowLMState stream_voice_state_;
    std::vector<std::string> stream_text_chunks_;
    size_t stream_text_chunk_index_ = 0;
    std::optional<AcousticStreamState> stream_acoustic_state_;
    runtime::AudioBuffer stream_merged_audio_;
    runtime::StreamEventCallback stream_event_sink_;
    std::chrono::steady_clock::time_point stream_started_at_;
    size_t stream_audio_chunk_index_ = 0;
    bool stream_started_ = false;
};

}  // namespace engine::models::pocket_tts

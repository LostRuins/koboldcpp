#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/soprano_tts/assets.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::soprano_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_soprano_tts_loader();

struct SopranoRequest {
    std::string text;
    SopranoGenerationOptions generation;
};

class SopranoQwen3Generator;
class SopranoVocosRuntime;

class SopranoTTSOfflineSession final : public runtime::RuntimeSessionBase,
                                       public runtime::IOfflineVoiceTaskSession,
                                       public runtime::IStreamingVoiceTaskSession {
public:
    SopranoTTSOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SopranoTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~SopranoTTSOfflineSession() override;

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
    SopranoRequest make_request(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize(const SopranoRequest & request);
    // Streaming state (NeuTTS-style: start_stream parses the full request
    // once, next_stream_event consumes the stored per-chunk requests).
    std::vector<SopranoRequest> streaming_requests_;
    size_t streaming_index_ = 0;
    std::vector<runtime::AudioBuffer> streaming_chunks_;
    bool streaming_started_ = false;
    runtime::StreamEventCallback stream_sink_;

    runtime::TaskSpec task_;
    std::shared_ptr<const SopranoTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<SopranoQwen3Generator> generator_;
    std::unique_ptr<SopranoVocosRuntime> decoder_;
};

}  // namespace engine::community_models::soprano_tts

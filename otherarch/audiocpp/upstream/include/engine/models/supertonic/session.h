#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/supertonic/assets.h"
#include "engine/models/supertonic/tokenizer_text.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::supertonic {

class SupertonicRuntime;

struct SupertonicGenerationOptions {
    int num_inference_steps = 8;
    float speaking_rate = 1.05F;
    uint32_t seed = 1234U;
    std::string voice = "M1";
    std::string language = "en";
};

class SupertonicSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    SupertonicSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SupertonicAssets> assets);
    ~SupertonicSession() override;

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
    SupertonicGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) const;
    void validate_request(const runtime::TaskRequest & request) const;
    std::vector<runtime::TaskRequest> build_chunk_requests(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize_chunk(const runtime::TaskRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const SupertonicAssets> assets_;
    SupertonicTextTokenizer tokenizer_;
    assets::TensorStorageType weight_storage_type_ = assets::TensorStorageType::Native;
    std::size_t style_cache_slots_ = 4;
    std::unique_ptr<SupertonicRuntime> runtime_;
    std::vector<runtime::TaskRequest> stream_chunk_requests_;
    runtime::AudioBuffer stream_merged_audio_;
    std::size_t stream_chunk_index_ = 0;
    bool stream_started_ = false;
};

}  // namespace engine::models::supertonic

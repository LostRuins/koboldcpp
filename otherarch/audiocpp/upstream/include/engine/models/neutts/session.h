#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/models/neutts/ar.h"
#include "engine/models/neutts/assets.h"
#include "engine/models/neutts/codec.h"
#include "engine/models/neutts/prompt.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::neutts {

std::shared_ptr<runtime::IVoiceModelLoader> make_neutts_loader();

struct NeuTTSRequest {
    std::string text;
    std::string speaker = "emily";
    std::string emotion = "neutral";
    NeuTTSGenerationOptions generation;
};

class NeuTTSSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::StreamingTtsSessionBase {
public:
    NeuTTSSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const NeuTTSAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~NeuTTSSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    NeuTTSRequest parse_request(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize(const NeuTTSRequest & request, bool streaming = false);
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override;
    void reset_stream_state() override { stream_audio_.reset(); }

    runtime::TaskSpec task_;
    std::shared_ptr<const NeuTTSAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    NeuTTSPromptBuilder prompt_builder_;
    std::unique_ptr<NeuTTSQwen3ARRuntime> ar_;
    std::unique_ptr<NeuTTSNeuCodecDecoderRuntime> codec_;
    runtime::StreamingAudioController<int32_t> stream_audio_;
};

}  // namespace engine::models::neutts

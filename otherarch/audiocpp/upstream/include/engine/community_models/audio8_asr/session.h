#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/audio8_asr/assets.h"
#include "engine/community_models/audio8_asr/projector.h"
#include "engine/community_models/audio8_asr/thinker.h"
#include "engine/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.h"
#include "engine/models/qwen3_asr/frontend_whisper.h"

#include <memory>
#include <string>

namespace engine::community_models::audio8_asr {

class Audio8ASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    Audio8ASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Audio8ASRAssets> assets);
    ~Audio8ASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    std::string transcribe_clip(const runtime::AudioBuffer & audio);
    runtime::Transcript transcribe_audio(const runtime::AudioBuffer & audio);

    runtime::TaskSpec task_;
    std::shared_ptr<const Audio8ASRAssets> assets_;
    qwen3_asr::Qwen3ASRWhisperFrontend frontend_;
    modules::Qwen3AudioEncoderRuntime audio_encoder_;
    Audio8ProjectorRuntime projector_;
    Audio8Qwen2ThinkerRuntime thinker_;
};

class Audio8ASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    Audio8ASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Audio8ASRAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Audio8ASRAssets> assets_;
};

std::unique_ptr<Audio8ASRLoadedModel> load_audio8_asr_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_audio8_asr_loader();

}  // namespace engine::community_models::audio8_asr

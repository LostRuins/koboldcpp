#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/silero_vad/assets.h"
#include "engine/models/silero_vad/runtime.h"

#include <memory>
#include <string>

namespace engine::models::silero_vad {

class SileroVADSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    SileroVADSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SileroWeights> weights);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    runtime::TaskSpec task_;
    SileroVADConfig default_config_;
    engine::assets::TensorStorageType weight_storage_type_;
    SileroRuntime runtime_;
};

class SileroVADLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    SileroVADLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const SileroWeights> weights);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const SileroWeights> weights_;
};

std::unique_ptr<SileroVADLoadedModel> load_silero_vad_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_silero_vad_loader();

}  // namespace engine::models::silero_vad

#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/pocket_tts/model.h"

#include <memory>

namespace engine::models::pocket_tts {

class PocketTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    PocketTTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        PocketTTSModel model);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    PocketTTSModel model_;
};

std::unique_ptr<PocketTTSLoadedModel> load_pocket_tts_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_pocket_tts_loader();

}  // namespace engine::models::pocket_tts

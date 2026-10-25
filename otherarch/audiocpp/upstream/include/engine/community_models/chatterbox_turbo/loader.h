#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/chatterbox_turbo/assets.h"

#include <filesystem>
#include <memory>

namespace engine::community_models::chatterbox_turbo {

class ChatterboxTurboLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    ChatterboxTurboLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const ChatterboxTurboAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const ChatterboxTurboAssets> assets_;
};

std::unique_ptr<ChatterboxTurboLoadedModel> load_chatterbox_turbo_model(const std::filesystem::path & model_root);
std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_turbo_loader();

}  // namespace engine::community_models::chatterbox_turbo

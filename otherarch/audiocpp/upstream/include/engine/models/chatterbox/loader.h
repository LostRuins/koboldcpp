#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/chatterbox/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::chatterbox {

class ChatterboxLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    ChatterboxLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const ChatterboxAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const ChatterboxAssets> assets_;
};

std::unique_ptr<ChatterboxLoadedModel> load_chatterbox_model(const std::filesystem::path & model_root);
std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_loader();

}  // namespace engine::models::chatterbox

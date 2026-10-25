#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/fish_audio/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::fish_audio {

class FishAudioLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    FishAudioLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const FishAudioAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const FishAudioAssets> assets_;
};

std::unique_ptr<FishAudioLoadedModel> load_fish_audio_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_fish_audio_loader();

}  // namespace engine::models::fish_audio

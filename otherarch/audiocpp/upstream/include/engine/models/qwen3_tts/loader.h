#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/qwen3_tts/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::qwen3_tts {

class Qwen3TTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    Qwen3TTSLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Qwen3TTSAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Qwen3TTSAssets> assets_;
};

std::unique_ptr<Qwen3TTSLoadedModel> load_qwen3_tts_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_tts_loader();

}  // namespace engine::models::qwen3_tts

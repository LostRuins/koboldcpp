#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/qwen3_asr/assets.h"

#include <memory>

namespace engine::models::qwen3_asr {

class Qwen3ASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    Qwen3ASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const Qwen3ASRAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const Qwen3ASRAssets> assets_;
};

std::unique_ptr<Qwen3ASRLoadedModel> load_qwen3_asr_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_asr_loader();

}  // namespace engine::models::qwen3_asr

#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/voxcpm2/assets.h"

#include <filesystem>
#include <memory>

namespace engine::models::voxcpm2 {

class VoxCPM2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
  VoxCPM2LoadedModel(runtime::ModelMetadata metadata,
                     runtime::CapabilitySet capabilities,
                     std::shared_ptr<const VoxCPM2Assets> assets);

  const runtime::ModelMetadata &metadata() const noexcept override;
  const runtime::CapabilitySet &capabilities() const noexcept override;
  std::unique_ptr<runtime::IVoiceTaskSession>
  create_task_session(const runtime::TaskSpec &task,
                      const runtime::SessionOptions &options) const override;

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const VoxCPM2Assets> assets_;
};

std::unique_ptr<VoxCPM2LoadedModel>
load_voxcpm2_model(const std::filesystem::path &model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm2_loader();

} // namespace engine::models::voxcpm2


#pragma once

#include "engine/models/maya1/assets.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <memory>

namespace engine::models::maya1 {

std::shared_ptr<runtime::IVoiceModelLoader> make_maya1_loader();

class Maya1Session final : public runtime::RuntimeSessionBase,
                           public runtime::IOfflineVoiceTaskSession {
public:
  Maya1Session(runtime::TaskSpec task, runtime::SessionOptions options,
               std::shared_ptr<const Maya1Assets> assets,
               std::shared_ptr<const model_spec::ModelContract> contract);
  ~Maya1Session() override;

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

private:
  runtime::TaskSpec task_;
  std::shared_ptr<const Maya1Assets> assets_;
  std::shared_ptr<const model_spec::ModelContract> contract_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::maya1

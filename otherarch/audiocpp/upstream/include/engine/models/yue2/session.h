#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/pipeline.h"

#include <memory>

namespace engine::models::yue2 {

class Yue2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    Yue2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Yue2Assets> assets);
    ~Yue2Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const Yue2Assets> assets_;
    std::unique_ptr<Yue2PipelineRuntime> pipeline_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_yue2_loader();

}  // namespace engine::models::yue2

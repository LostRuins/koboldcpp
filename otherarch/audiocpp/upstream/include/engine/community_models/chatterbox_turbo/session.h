#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/chatterbox_turbo/assets.h"
#include "engine/community_models/chatterbox_turbo/tts.h"

#include <memory>
#include <string>

namespace engine::community_models::chatterbox_turbo {

class ChatterboxTurboSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ChatterboxTurboSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ChatterboxTurboAssets> assets);
    ~ChatterboxTurboSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const ChatterboxTurboAssets> assets_;
    std::unique_ptr<ChatterboxTurboTTSComponent> component_;
};

}  // namespace engine::community_models::chatterbox_turbo

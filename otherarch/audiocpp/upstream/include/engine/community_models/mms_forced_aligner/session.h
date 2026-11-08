#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/mms_forced_aligner/assets.h"
#include "engine/community_models/mms_forced_aligner/emissions.h"

#include <cstdint>
#include <memory>
#include <string>

namespace engine::community_models::mms_forced_aligner {

class MmsForcedAlignerSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    MmsForcedAlignerSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const MmsForcedAlignerAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const MmsForcedAlignerAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    int64_t max_alignment_cells_ = 50000000;
    int64_t max_target_tokens_ = 8192;
    std::unique_ptr<MmsWav2Vec2CtcRuntime> emission_runtime_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_mms_forced_aligner_loader();

}  // namespace engine::community_models::mms_forced_aligner

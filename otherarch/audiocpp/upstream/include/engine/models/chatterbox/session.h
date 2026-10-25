#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/models/chatterbox/assets.h"
#include "engine/models/chatterbox/conditionals.h"
#include "engine/models/chatterbox/tts.h"
#include "engine/models/chatterbox/vc.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::chatterbox {

enum class ChatterboxMultilingualT3Version {
    V2,
    V3,
};

struct ChatterboxConditionalsCacheKey {
    runtime::AudioBuffer reference_audio;
    float exaggeration = 0.0f;
    std::string language;
};

struct ChatterboxConditionalsCacheKeyEqual {
    bool operator()(
        const ChatterboxConditionalsCacheKey & lhs,
        const ChatterboxConditionalsCacheKey & rhs) const;
};

class ChatterboxSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ChatterboxSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ChatterboxAssets> assets);
    ~ChatterboxSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult run_voice_cloning(const runtime::TaskRequest & request);
    runtime::TaskResult run_voice_conversion(const runtime::TaskRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const ChatterboxAssets> assets_;
    engine::assets::TensorStorageType t3_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType component_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    ChatterboxMultilingualT3Version multilingual_t3_version_ = ChatterboxMultilingualT3Version::V2;
    bool mem_saver_ = false;
    std::unique_ptr<ChatterboxTTSComponent> component_;
    std::unique_ptr<ChatterboxVCComponent> vc_component_;
    std::optional<std::string> component_language_;
    std::optional<ChatterboxVoiceCloneConfig> voice_clone_config_;
    runtime::CacheSlots<
        ChatterboxConditionalsCacheKey,
        ChatterboxConditionalsOutputs,
        ChatterboxConditionalsCacheKeyEqual> conditionals_cache_;
    std::optional<ChatterboxConditionalsOutputs> cached_conditionals_;
    double cached_prompt_prep_ms_ = 0.0;
};

}  // namespace engine::models::chatterbox

#pragma once

#include "engine/models/pocket_tts/types.h"
#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
class PocketTTSSession;
struct PocketTTSGraphCapacityConfig;

class PocketTTSModel {
public:
    static PocketTTSModel load(const ModelConfig & config);

    PocketTTSModel() = default;
    ~PocketTTSModel() = default;
    PocketTTSModel(const PocketTTSModel &) = default;
    PocketTTSModel & operator=(const PocketTTSModel &) = default;
    PocketTTSModel(PocketTTSModel &&) noexcept = default;
    PocketTTSModel & operator=(PocketTTSModel &&) noexcept = default;

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const;

    const std::filesystem::path & model_dir() const noexcept;

private:
    friend std::unique_ptr<class PocketTTSLoadedModel> load_pocket_tts_model(const runtime::ModelLoadRequest & request);

    PocketTTSModel(std::filesystem::path model_dir, std::shared_ptr<const PocketTTSAssets> manifest);

    std::filesystem::path model_dir_;
    std::shared_ptr<const PocketTTSAssets> manifest_;
};

}  // namespace engine::models::pocket_tts

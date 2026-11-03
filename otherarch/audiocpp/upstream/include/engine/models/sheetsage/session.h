#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/models/sheetsage/runtime.h"
#include "engine/models/sheetsage/audio_frontend.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::models::sheetsage {

struct SheetSage2Assets {
    assets::ResourceBundle resources;
    SheetSage2DecoderConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    engine::audio::SparseMelFilterbank mel_filterbank;
    std::vector<float> mel_mean;
    std::vector<float> mel_std;
    std::vector<float> stft_window;
};

std::shared_ptr<const SheetSage2Assets> load_sheetsage2_assets(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_sheetsage2_loader();

class SheetSage2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SheetSage2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SheetSage2Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract);
    ~SheetSage2Session() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SheetSage2Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    Mert2EncoderRuntime encoder_;
    SheetSage2AudioFrontend audio_frontend_;
    SheetSage2DecoderRuntime decoder_;
};

}  // namespace engine::models::sheetsage

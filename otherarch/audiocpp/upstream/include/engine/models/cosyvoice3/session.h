#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/cosyvoice3/assets.h"
#include "engine/models/cosyvoice3/tokenizer_text.h"

#include <memory>

namespace engine::models::cosyvoice3 {

class CosyVoice3Frontend;
class CosyVoice3Qwen2ARRuntime;
class CosyVoice3DiTFlowRuntime;
class CosyVoice3HiftRuntime;

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_cosyvoice3_loader();

class CosyVoice3Session final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    CosyVoice3Session(
        engine::runtime::TaskSpec task,
        engine::runtime::SessionOptions options,
        std::shared_ptr<const CosyVoice3Assets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~CosyVoice3Session() override;

    std::string family() const override;
    engine::runtime::VoiceTaskKind task_kind() const override;
    engine::runtime::RunMode run_mode() const override;
    void prepare(const engine::runtime::SessionPreparationRequest & request) override;
    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override;

private:
    engine::runtime::TaskSpec task_;
    std::shared_ptr<const CosyVoice3Assets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::unique_ptr<CosyVoice3TextTokenizer> tokenizer_;
    std::unique_ptr<CosyVoice3Frontend> frontend_;
    std::unique_ptr<CosyVoice3Qwen2ARRuntime> ar_;
    std::unique_ptr<CosyVoice3DiTFlowRuntime> flow_;
    std::unique_ptr<CosyVoice3HiftRuntime> hift_;
    bool mem_saver_ = false;
};

}  // namespace engine::models::cosyvoice3

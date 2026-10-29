#include "engine/models/supertonic/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/supertonic/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::supertonic {
namespace {

runtime::CapabilitySet capabilities(const SupertonicAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = {
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
        "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
    };
    out.supports_style_condition = true;
    return out;
}

runtime::ModelMetadata metadata(const SupertonicAssets &) {
    runtime::ModelMetadata out;
    out.family = "supertonic";
    out.variant = "supertonic-3";
    out.description = "Supertonic 3 loaded from GGML safetensors assets.";
    return out;
}

runtime::ModelCliInterface cli(const SupertonicAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"voice_id", "M1|M2|M3|M4|M5|F1|F2|F3|F4|F5", "Preset voice style id, default M1; also exposed as --voice-id.", false, "M1"},
        {"num_inference_steps", "n", "Flow denoising steps, default 8.", false, "8", "1"},
        {"speed", "float", "Speech speed multiplier, default 1.05.", false, "1.05", "0"},
        {"seed", "n", "Noise seed, default 1234.", false, "1234", "0", "4294967295"},
        {"text_chunk_mode", "default|tag_aware|japanese|endline", "Long-form text chunking mode.", false, "default"},
        {"text_chunk_size", "int", "Framework text chunk size; defaults to 120 for Korean/Japanese, otherwise 300."},
    };
    out.session_options = {
        {"supertonic.weight_type", "native|f32|f16|bf16|q8_0", "Supertonic weight storage type.", false, "native"},
        {"supertonic.style_cache_slots", "n", "Preset voice style cache slots; default 4.", false, "4", "0"},
    };
    return out;
}

class SupertonicLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return "supertonic"; }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_supertonic_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_supertonic_model(request.model_path);
    }
};

}  // namespace

SupertonicLoadedModel::SupertonicLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const SupertonicAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & SupertonicLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & SupertonicLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> SupertonicLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Supertonic only supports offline and streaming sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Supertonic only supports the Tts task");
    }
    return std::make_unique<SupertonicSession>(task, options, assets_);
}

std::unique_ptr<SupertonicLoadedModel> load_supertonic_model(const std::filesystem::path & model_path) {
    auto assets = load_supertonic_assets(model_path);
    return std::make_unique<SupertonicLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_supertonic_loader() {
    return std::make_shared<SupertonicLoader>();
}

}  // namespace engine::models::supertonic

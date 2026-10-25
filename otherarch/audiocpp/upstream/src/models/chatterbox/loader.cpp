#include "engine/models/chatterbox/loader.h"

#include "engine/models/chatterbox/session.h"
#include "engine/models/chatterbox/text_tokenizer.h"

#include "engine/framework/model_spec/package.h"

#include <memory>
#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

runtime::CapabilitySet capabilities(const ChatterboxAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks.push_back({runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}});
    capabilities.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
    capabilities.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}});
    capabilities.languages = supported_chatterbox_language_codes();
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    return capabilities;
}

runtime::ModelMetadata metadata(const ChatterboxAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "chatterbox";
    out.variant = assets.resources.model_root().filename().string();
    out.description = "Chatterbox voice cloning and voice conversion loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli(const ChatterboxAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"exaggeration", "float", "Emotion exaggeration. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 0.5.", false, "0.5"},
        {"guidance_scale", "float", "T3 classifier-free guidance scale. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 0.5.", false, "0.5"},
        {"temperature", "float", "T3 sampling temperature. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 0.8.", false, "0.8"},
        {"repetition_penalty", "float", "T3 repetition penalty. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 1.2.", false, "1.2"},
        {"min_p", "float", "Minimum sampling probability. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 0.05.", false, "0.05"},
        {"top_p", "float", "Nucleus sampling probability. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 1.", false, "1"},
        {"flow_guidance_scale", "float", "S3Gen flow guidance scale. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 0.7.", false, "0.7"},
        {"max_tokens", "int", "Maximum generated speech tokens. Captured during clone preparation; changing it on the same prepared session is rejected. Default: 384.", false, "384", "1"},
        {"do_sample", "bool", "Enable sampling. Captured during clone preparation; changing it on the same prepared session is rejected. Default: true.", false, "true"},
        {"stop_on_eos", "bool", "Stop token generation on EOS. Captured during clone preparation; changing it on the same prepared session is rejected. Default: true.", false, "true"},
        {"num_inference_steps", "int", "Voice conversion flow steps; max_steps is also accepted.", false, "10", "1"},
        {"seed", "int", "Sampling seed; omitted chooses a random seed. Captured during clone preparation; changing it on the same prepared session is rejected.", false, std::nullopt, "0", "4294967295"},
        {"source_audio", "audio_path", "Voice conversion source WAV path; overrides audio_input.", false},
        {"target_voice", "audio_path", "Voice conversion target WAV path; overrides the speaker reference.", false},
        {"text_chunk_size", "int", "Framework long-form text chunk size.", false, "128"},
    };
    out.session_options = {
        {"chatterbox.weight_type", "native|f32|f16|bf16|q8_0", "Weight storage type; defaults to native.", false},
        {"chatterbox.t3_weight_type", "native|f32|f16|bf16|q8_0", "T3 weight storage; inherits weight_type.", false},
        {"chatterbox.multilingual_t3", "v2|v3", "Multilingual T3 checkpoint for non-English TTS; default v2.", false, "v2"},
        {"chatterbox.conditionals_cache_slots", "int", "Prepared voice-condition cache slots; default 1, set 0 to disable.", false, "1", "0"},
        {"chatterbox.mem_saver", "bool", "Free non-conditional runtime graphs after each request chunk; default false.", false, "false"},
        {"chatterbox.encoder_condition_samples", "int", "Maximum reference samples for the T3 tokenizer; the voice encoder still consumes the full 16 kHz reference. Default: 96000.", false, "96000"},
        {"chatterbox.decoder_condition_samples", "int", "S3Gen reference sample limit at 24 kHz.", false, "240000"},
        {"chatterbox.t3_speech_cond_prompt_len", "int", "T3 reference speech prompt token limit.", false, "150"},
    };
    return out;
}

class ChatterboxLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "chatterbox";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks.push_back({runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}});
        out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline}});
        out.supported_tasks.push_back({runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}});
        out.languages = supported_chatterbox_language_codes();
        out.supports_speaker_reference = true;
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
        const auto assets = load_chatterbox_assets(request.model_path);
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
        return load_chatterbox_model(request.model_path);
    }
};

}  // namespace

ChatterboxLoadedModel::ChatterboxLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const ChatterboxAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Chatterbox loaded model requires assets");
    }
}

const runtime::ModelMetadata & ChatterboxLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & ChatterboxLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> ChatterboxLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceCloning &&
        task.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("Chatterbox supports TTS, voice cloning, and voice conversion");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox only supports offline mode");
    }
    return std::make_unique<ChatterboxSession>(task, options, assets_);
}

std::unique_ptr<ChatterboxLoadedModel> load_chatterbox_model(const std::filesystem::path & model_root) {
    auto assets = load_chatterbox_assets(model_root);

    return std::make_unique<ChatterboxLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_chatterbox_loader() {
    return std::make_shared<ChatterboxLoader>();
}

}  // namespace engine::models::chatterbox

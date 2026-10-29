#include "engine/models/omnivoice/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/omnivoice/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::omnivoice {
namespace {

runtime::ModelMetadata metadata(const OmniVoiceAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "omnivoice";
    out.variant = assets.config.model_type;
    out.description = "OmniVoice multilingual TTS with voice clone and voice design pipelines.";
    return out;
}

runtime::CapabilitySet capabilities(const OmniVoiceAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    out.languages = assets.config.supported_languages;
    return out;
}

runtime::ModelCliInterface cli(const OmniVoiceAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"text_chunk_mode", "word_budget|tag_aware|japanese|endline", "Text chunking mode; default tag_aware.", false, "tag_aware"},
        {"language", "string", "Synthesis language code or label; omitted uses automatic language selection."},
        {"reference_text", "string", "Transcript for speaker reference audio used in voice cloning."},
        {"instruct", "string", "Voice-design instruction such as speaker attributes, accent, style, or delivery."},
        {"seed", "int", "Generation seed for reproducible output; omitted value keeps the session RNG state.", false, std::nullopt, "0", "4294967295"},
        {"num_inference_steps", "int", "Diffusion language-model inference steps; default 32.", false, "32", "1"},
        {"guidance_scale", "float", "Classifier-free guidance scale; default 2.0.", false, "2"},
        {"speed", "float", "Speech speed multiplier used for target-length estimation; default 1.0.", false, "1"},
        {"duration_sec", "float", "Target output duration in seconds; omitted uses the model's estimated duration.", false, std::nullopt, "0"},
        {"shift", "float", "Diffusion timestep schedule shift; default 0.1.", false, "0.1"},
        {"denoise", "bool", "Add the denoise control token when reference audio is present; default true.", false, "true"},
        {"preprocess_prompt", "bool", "Preprocess speaker reference audio before tokenization; default true.", false, "true"},
        {"postprocess_output", "bool", "Apply output postprocessing and trimming; default true.", false, "true"},
        {"layer_penalty_factor", "float", "Layer-order penalty used while selecting masked audio-codebook positions; default 5.0.", false, "5"},
        {"position_temperature", "float", "Gumbel temperature for masked-position selection; default 5.0.", false, "5"},
        {"class_temperature", "float", "Audio-token class sampling temperature; default 0 for greedy class selection.", false, "0", "0"},
        {"audio_chunk_duration_sec", "float", "Target chunk duration for automatic long-form splitting; default 15 seconds.", false, "15", "0"},
        {"audio_chunk_threshold_sec", "float", "Estimated duration above which long-form generation is split into chunks; default 30 seconds.", false, "30", "0"},
        {"text_chunk_size", "int", "Explicit long-form text chunk size in Unicode codepoints. Omitted value uses duration-based automatic chunking.", false, std::nullopt, "1"},
    };
    out.session_options = {
        {"omnivoice.mem_saver", "bool", "Release staged runtime graphs after request phases; default false. Cannot be combined with perf_mode=flash_attention.", false, "false"},
        {"omnivoice.perf_mode", "off|flash_attention", "Generator performance mode; default off keeps the exact-safe attention path. flash_attention requires mem_saver=false.", false, "off"},
        {"omnivoice.weight_type", "native|f32|f16|bf16|q8_0", "Shared generator weight storage type; default native.", false, "native"},
        {"omnivoice.generator_weight_type", "native|f32|f16|bf16|q8_0", "Generator weight storage type; defaults to weight_type, which defaults to native."},
        {"omnivoice.audio_tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Audio tokenizer weight storage type; default native.", false, "native"},
        {"omnivoice.audio_tokenizer_graph_arena_mb", "int", "Audio tokenizer graph arena size in MiB; default 128.", false, "128", "1"},
        {"omnivoice.generator_prefill_graph_arena_mb", "int", "Generator prefill graph arena size in MiB; default 256.", false, "256", "1"},
        {"omnivoice.generator_decode_graph_arena_mb", "int", "Generator decode graph arena size in MiB; default 256.", false, "256", "1"},
        {"omnivoice.audio_tokenizer_weight_context_mb", "int", "Audio tokenizer weight context size in MiB; default 128.", false, "128", "1"},
        {"omnivoice.generator_weight_context_mb", "int", "Generator weight context size in MiB; default 256.", false, "256", "1"},
    };
    return out;
}

class OmniVoiceLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "omnivoice";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    std::string advertised_instructions_policy() const override {
        return "soft_tags";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_omnivoice_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto package_spec = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            package_spec,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_omnivoice_model(request.model_path);
    }
};

}  // namespace

OmniVoiceLoadedModel::OmniVoiceLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const OmniVoiceAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & OmniVoiceLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & OmniVoiceLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> OmniVoiceLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<OmniVoiceSession>(task, options, assets_);
}

std::unique_ptr<OmniVoiceLoadedModel> load_omnivoice_model(const std::filesystem::path & model_path) {
    auto assets = load_omnivoice_assets(model_path);
    return std::make_unique<OmniVoiceLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_omnivoice_loader() {
    return std::make_shared<OmniVoiceLoader>();
}

}  // namespace engine::models::omnivoice

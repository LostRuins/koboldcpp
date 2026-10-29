#include "engine/models/vibevoice/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/vibevoice/lora.h"
#include "engine/models/vibevoice/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice {
namespace {

runtime::CapabilitySet capabilities(const VibeVoiceAssets &) {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    capabilities.languages = {"Auto"};
    capabilities.supports_speaker_reference = true;
    return capabilities;
}

runtime::ModelMetadata metadata(const VibeVoiceAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "vibevoice";
    metadata.variant = assets.config.model_type + "-qwen2.5";
    metadata.description = "VibeVoice loaded from local assets.";
    return metadata;
}

runtime::ModelCliInterface cli(const VibeVoiceAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"max_tokens", "n", "Maximum generated tokens; 0 selects an automatic budget.", false, "0", "0"},
        {"do_sample", "true|false", "Enable sampling instead of deterministic decoding.", false, "false"},
        {"temperature", "float", "Positive sampling temperature.", false, "1"},
        {"top_k", "n", "Top-k sampling limit; 0 disables filtering.", false, "50", "0"},
        {"top_p", "float", "Nucleus sampling probability in (0, 1].", false, "1", std::nullopt, "1"},
        {"seed", "int", "Generation seed.", false, "1234", "0", "4294967295"},
        {"voice_samples", "path[,path...]", "Comma-separated speaker reference WAV paths for multi-speaker prompts."},
        {"num_inference_steps", "n", "Diffusion inference steps."},
        {"guidance_scale", "float", "Classifier-free guidance scale."},
        {"max_length_times", "float", "Generation length multiplier."},
        {"prompt_noise_file", "path", "F32 prompt acoustic noise file."},
        {"diffusion_noise_file", "path", "F32 initial diffusion noise file."},
    };
    out.session_options = {
        {"vibevoice.weight_type", "native|f32|f16|bf16|q8_0", "Tokenizer, connector, decoder, and diffusion-head weight storage type."},
        {"vibevoice.tokenizer_weight_type", "native|f32|f16|bf16|q8_0", "Audio tokenizer weight storage type."},
        {"vibevoice.connector_weight_type", "native|f32|f16|bf16|q8_0", "Acoustic and semantic connector weight storage type."},
        {"vibevoice.decoder_weight_type", "native|f32|f16|bf16|q8_0", "Language decoder weight storage type."},
        {"vibevoice.diffusion_head_weight_type", "native|f32|f16|bf16|q8_0", "Diffusion prediction head weight storage type."},
        {"vibevoice.lora", "path", "Decoder adapter file or PEFT directory applied when creating the session; do not also supply the load option."},
        {"vibevoice.lora_scale", "float", "Session adapter scale override; omitted uses lora_alpha / r. Must be strictly positive when an adapter is supplied."},
    };
    out.load_options = {
        {"vibevoice.lora", "path", "Decoder adapter file or PEFT directory, with optional diffusion-head and connector sidecars, merged at load time."},
        {"vibevoice.lora_scale", "float", "LoRA scale override; omitted uses lora_alpha / r. Must be strictly positive when an adapter is supplied."},
    };
    return out;
}

class VibeVoiceLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vibevoice";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
        return out;
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
        const auto assets = load_vibevoice_assets(request.model_path);
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
        auto assets = apply_vibevoice_finetune_options(
            load_vibevoice_assets(request.model_path), request.options);
        return std::make_unique<VibeVoiceLoadedModel>(
            metadata(*assets), capabilities(*assets), std::move(assets));
    }
};

}  // namespace

VibeVoiceLoadedModel::VibeVoiceLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const VibeVoiceAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & VibeVoiceLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & VibeVoiceLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> VibeVoiceLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VibeVoice only supports offline and streaming sessions");
    }
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("VibeVoice only supports the Tts task");
    }
    return std::make_unique<VibeVoiceSession>(task, options, assets_);
}

std::unique_ptr<VibeVoiceLoadedModel> load_vibevoice_model(const std::filesystem::path & model_path) {
    auto assets = load_vibevoice_assets(model_path);
    return std::make_unique<VibeVoiceLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibevoice_loader() {
    return std::make_shared<VibeVoiceLoader>();
}

}  // namespace engine::models::vibevoice

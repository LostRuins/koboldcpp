#include "engine/models/fish_audio/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/fish_audio/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

runtime::ModelMetadata metadata(const FishAudioAssets &) {
    runtime::ModelMetadata out;
    out.family = "fish_audio";
    out.variant = "s2-pro";
    out.description = "Fish Audio S2-Pro loaded from prepared local assets.";
    out.config_candidates = {"config.json", "tokenizer_config.json", "tokenizer.json"};
    out.weight_candidates = {"model_audio_cpp.safetensors.index.json", "codec.safetensors", "model.gguf"};
    return out;
}

runtime::CapabilitySet capabilities(const FishAudioAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = {"en", "zh", "auto"};
    out.supports_speaker_reference = true;
    out.supports_style_condition = true;
    return out;
}

runtime::ModelCliInterface cli(const FishAudioAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"reference_text", "string", "Reference transcript; required with inline reference audio."},
        {"multi_reference_cond", "JSON", "Ordered Fish Audio reference conditioning pairs: [{\"audio\":\"ref.wav\",\"text\":\"transcript\"}, ...]."},
        {"max_tokens", "int", "Maximum Fish Audio semantic tokens to generate; default 1024. Zero selects the default token budget.", false, "1024", "0"},
        {"text_chunk_size", "N", "Long-form text chunk size; default 200.", false, "200", "1"},
        {"text_chunk_mode", "default|word_budget|tag_aware|japanese|endline", "Framework text chunking mode.", false, "word_budget"},
        {"top_p", "FLOAT", "Top-p sampling value in (0, 1]; default 0.8.", false, "0.8", "0.0", "1.0"},
        {"top_k", "N", "Top-k sampling value.", false, "30", "1"},
        {"temperature", "FLOAT", "Semantic-token sampling temperature in (0, 2); default 0.8.", false, "0.8", "0.0", "2.0"},
        {"seed", "N", "Sampling seed for reproducible output; omitted uses a random seed.", false, std::nullopt, "0"},
    };
    out.session_options = {
        {"fish_audio.lora", "path", "Prepared Fish Speech adapter checkpoint; merged into slow/fast decoder and embedding weights at session creation. Codec weights are unchanged.", false},
        {"fish_audio.lora_scale", "float", "Multiplier for the adapter's alpha/r scaling; default 1.", false, "1"},
        {"fish_audio.mem_saver", "true|false", "Release cached AR runtime graphs after each request; default false.", false, "false"},
        {"fish_audio.reference_cache_slots", "n", "Prepared reference-audio cache slots; default 1.", false, "1", "0"},
        {"fish_audio.weight_type", "native|f32|f16|bf16|q8_0", "AR matmul weight storage type; default native.", false, "native"},
        {"fish_audio.codec_weight_type", "native|f32|f16|q8_0", "Codec conv/matmul weight storage type; default native.", false, "native"},
        {"fish_audio.codec_encoder_bf16_compute", "true|false", "Use BF16 reference-encoder arithmetic on CUDA with F16/BF16 codec weights; default false. Decoding is unchanged.", false, "false"},
        {"fish_audio.ar_graph_arena_mb", "int", "AR runtime graph arena size in MiB; default 512.", false, "512", "1"},
        {"fish_audio.ar_weight_context_mb", "int", "AR weight context size in MiB; default 32.", false, "32", "1"},
        {"fish_audio.codec_graph_arena_mb", "int", "Codec encode/decode graph arena size in MiB; default 512.", false, "512", "1"},
        {"fish_audio.codec_weight_context_mb", "int", "Codec weight context size in MiB; default 32.", false, "32", "1"},
    };
    return out;
}

class FishAudioLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "fish_audio";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
        out.supports_style_condition = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto package_spec = engine::model_spec::default_spec_path(family());
            (void) engine::model_spec::load_resource_bundle(request.model_path, package_spec);
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_fish_audio_assets(request.model_path);
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
        return load_fish_audio_model(request.model_path);
    }
};

}  // namespace

FishAudioLoadedModel::FishAudioLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const FishAudioAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & FishAudioLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & FishAudioLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> FishAudioLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if ((task.task != runtime::VoiceTaskKind::Tts &&
         task.task != runtime::VoiceTaskKind::VoiceCloning) ||
        (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming)) {
        throw std::runtime_error("Fish Audio S2-Pro supports offline and streaming TTS and voice cloning sessions");
    }
    return std::make_unique<FishAudioSession>(task, options, assets_);
}

std::unique_ptr<FishAudioLoadedModel> load_fish_audio_model(const std::filesystem::path & model_path) {
    auto assets = load_fish_audio_assets(model_path);
    return std::make_unique<FishAudioLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_fish_audio_loader() {
    return std::make_shared<FishAudioLoader>();
}

}  // namespace engine::models::fish_audio

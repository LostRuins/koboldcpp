#include "engine/models/pocket_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/session.h"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

std::string requested_language(const runtime::ModelLoadRequest & request) {
    const auto options = runtime::apply_option_v1_compatibility(
        request.options, {{"language", "pocket_tts.language"}}, "PocketTTS", "load");
    return runtime::find_option(options, {"pocket_tts.language"}).value_or("english");
}

runtime::CapabilitySet capabilities(const PocketTTSAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = {assets.language};
    out.supports_speaker_reference = true;
    out.supports_style_condition = false;
    return out;
}

runtime::ModelMetadata metadata(const PocketTTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "pocket_tts";
    out.variant = assets.language;
    out.description = "PocketTTS loaded from local assets.";
    return out;
}

runtime::ModelCliInterface cli(const PocketTTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"frames_after_eos", "int", "Extra frames after EOS; -1 uses the model-derived value.", false, "-1", "-1"},
        {"max_frames", "int", "Maximum generated audio frames; zero estimates the cap from text length. Captured during session preparation.", false, "0", "0"},
        {"text_chunk_tokens", "int", "Tokenizer tokens per text chunk; must be positive, default 50. Set during preparation and overridable per request.", false, "50", "1"},
        {"temperature", "float", "Positive flow noise temperature; defaults to the model configuration. Captured during session preparation.", false},
        {"seed", "int", "Noise seed; omitted chooses a random seed. Captured during session preparation.", false, std::nullopt, "0", "4294967295"},
        {"noise_clamp", "float", "Noise clamp; a negative value disables clamping. Captured during session preparation.", false, "-1"},
        {"eos_threshold", "float", "EOS detection threshold. Captured during session preparation.", false, "-4"},
        {"noise_file", "path", "Precomputed noise schedule file. Captured during session preparation.", false},
        {"voice_embedding_file", "path", "Precomputed voice-state embedding path. Captured during session preparation.", false},
        {"truncate_clone_audio", "bool", "Truncate voice clone audio. Captured during session preparation.", false, "false"},
        {"text_chunk_size", "int", "Optional per-request character chunk size; otherwise use sentence/token budgeting.", false},
        {"export_voice_state_file", "path", "Export the prepared voice state during session preparation.", false},
    };
    out.session_options = {
        {"pocket_tts.voice_state_cache_slots", "int", "Prepared voice-state cache slots; default 4, set 0 to disable.", false, "4", "0"},
        {"pocket_tts.offline_graph_capacity_mode", "fixed|double|tiered|grow", "Offline graph capacity policy; CPU/Metal default tiered, other backends double.", false},
        {"pocket_tts.prompt_graph_capacity", "int", "Optional reserved prompt capacity.", false, std::nullopt, "1"},
        {"pocket_tts.generation_graph_capacity", "int", "Optional reserved generation capacity.", false, std::nullopt, "1"},
        {"pocket_tts.weight_type", "native|f32|f16|bf16|q8_0", "Shared matmul weight storage; component matmul settings inherit it. Convolution storage remains independently native unless overridden."},
        {"pocket_tts.matmul_weight_type", "native|f32|f16|bf16|q8_0", "Weight storage type; component-specific settings inherit weight_type.", false},
        {"pocket_tts.conv_weight_type", "native|f32|f16", "Convolution weight storage.", false, "native"},
        {"pocket_tts.weight_context_mb", "int", "Context or graph arena size in MiB.", false, "64", "1"},
        {"pocket_tts.flow_weight_context_mb", "int", "Flow weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.", false, std::nullopt, "1"},
        {"pocket_tts.mimi_encoder_weight_context_mb", "int", "Mimi encoder weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.", false, std::nullopt, "1"},
        {"pocket_tts.mimi_decoder_weight_context_mb", "int", "Mimi decoder weight context in MiB; inherits pocket_tts.weight_context_mb when omitted.", false, std::nullopt, "1"},
        {"pocket_tts.flow_weights_view_context_mb", "int", "Context or graph arena size in MiB.", false, "256", "1"},
        {"pocket_tts.flow_step_graph_arena_mb", "int", "Flow-step host graph-metadata arena size in MiB, not device workspace.", false, "256", "1"},
        {"pocket_tts.mimi_encoder_graph_arena_mb", "int", "Mimi encoder host graph-metadata arena size in MiB, not device workspace.", false, "512", "1"},
        {"pocket_tts.mimi_conv_graph_arena_mb", "int", "Mimi convolution host graph-metadata arena size in MiB, not device workspace.", false, "32", "1"},
        {"pocket_tts.mimi_transformer_graph_arena_mb", "int", "Mimi transformer host graph-metadata arena size in MiB, not device workspace.", false, "96", "1"},
        {"pocket_tts.mimi_tail_graph_arena_mb", "int", "Mimi tail host graph-metadata arena size in MiB, not device workspace.", false, "512", "1"},
        {"pocket_tts.mimi_full_chunk_frames", "int", "Full-sequence decoder chunk frames.", false, "90", "1"},
        {"pocket_tts.mimi_stage2_chunk_frames", "int", "Second-stage decoder chunk frames.", false, "900", "1"},
        {"pocket_tts.use_full_mimi", "bool", "Use full-sequence Mimi decoding.", false, "true"},
    };
    out.load_options = {
        {"pocket_tts.language", "string", "Model language selection.", false, "english"},
    };
    return out;
}

std::shared_ptr<const PocketTTSAssets> load_request_assets(const runtime::ModelLoadRequest & request) {
    return load_pocket_tts_assets(request.model_path, requested_language(request));
}

class PocketTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "pocket_tts";
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
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_request_assets(request);
        runtime::ModelInspection inspection;
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        inspection.model_root = assets->resources.model_root();
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
        return load_pocket_tts_model(request);
    }
};

}  // namespace

PocketTTSModel::PocketTTSModel(
    std::filesystem::path model_dir,
    std::shared_ptr<const PocketTTSAssets> manifest)
    : model_dir_(std::move(model_dir)),
      manifest_(std::move(manifest)) {}

PocketTTSModel PocketTTSModel::load(const ModelConfig & config) {
    if (config.model_dir.empty()) {
        throw std::runtime_error("PocketTTS model_dir is required");
    }
    runtime::ModelLoadRequest request;
    request.model_path = config.model_dir;
    if (!config.language.empty()) {
        request.options.emplace("language", config.language);
    }
    auto manifest = load_request_assets(request);
    auto voice_asset_root = manifest->voice_asset_root;
    return PocketTTSModel(std::move(voice_asset_root), std::move(manifest));
}

std::unique_ptr<runtime::IVoiceTaskSession> PocketTTSModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (!manifest_) {
        throw std::runtime_error("PocketTTS model is not loaded");
    }
    return std::make_unique<PocketTTSSession>(task, options, manifest_, model_dir_);
}

const std::filesystem::path & PocketTTSModel::model_dir() const noexcept {
    return model_dir_;
}

PocketTTSLoadedModel::PocketTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    PocketTTSModel model)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      model_(std::move(model)) {}

const runtime::ModelMetadata & PocketTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & PocketTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> PocketTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return model_.create_task_session(task, options);
}

std::unique_ptr<PocketTTSLoadedModel> load_pocket_tts_model(const runtime::ModelLoadRequest & request) {
    auto assets = load_request_assets(request);
    PocketTTSModel model(assets->voice_asset_root, assets);

    return std::make_unique<PocketTTSLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(model));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_pocket_tts_loader() {
    return std::make_shared<PocketTTSLoader>();
}

}  // namespace engine::models::pocket_tts

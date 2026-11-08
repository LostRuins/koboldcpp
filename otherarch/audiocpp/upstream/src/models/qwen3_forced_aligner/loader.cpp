#include "engine/models/qwen3_forced_aligner/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_forced_aligner/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_forced_aligner {
namespace {

std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> load_qwen3_forced_aligner_assets(
    const std::filesystem::path & model_path) {
    auto assets = engine::models::qwen3_asr::load_qwen3_asr_assets(model_path, "qwen3_forced_aligner");
    if (assets->config.thinker_model_type != "qwen3_forced_aligner") {
        throw std::runtime_error("Qwen3 forced aligner loader received non-aligner assets");
    }
    return assets;
}

runtime::ModelMetadata metadata(const engine::models::qwen3_asr::Qwen3ASRAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "qwen3_forced_aligner";
    out.variant = assets.config.model_size.empty() ? assets.config.thinker_model_type : assets.config.model_size;
    out.description = "Qwen3 forced aligner loaded from local assets.";
    return out;
}

runtime::CapabilitySet capabilities(const engine::models::qwen3_asr::Qwen3ASRAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Alignment, {runtime::RunMode::Offline}},
    };
    out.languages = assets.config.supported_languages;
    out.supports_timestamps = true;
    return out;
}

class Qwen3ForcedAlignerLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_forced_aligner";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Alignment, {runtime::RunMode::Offline}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            (void) load_qwen3_forced_aligner_assets(request.model_path);
            return !request.family_hint.has_value() || *request.family_hint == family();
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_qwen3_forced_aligner_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.cli.request_options = {
            {"clamp_timestamps_to_audio", "bool", "Opt-in guard for word timestamp output: keep repaired word spans inside the local audio input. Defaults to false to preserve existing timestamp repair behavior.", false, "false"},
        };
        inspection.cli.session_options = {
            {"qwen3_forced_aligner.weight_type", "native|f32|f16|bf16|q8_0", "Shared thinker matmul storage; does not override audio encoder storage. Default: native.", false, "native"},
            {"qwen3_forced_aligner.audio_encoder_weight_type", "native|f32|f16", "Audio encoder matmul storage. Default: native.", false, "native"},
            {"qwen3_forced_aligner.thinker_weight_type", "native|f32|f16|bf16|q8_0", "Thinker matmul storage; inherits weight_type when omitted."},
            {"qwen3_forced_aligner.audio_encoder_graph_arena_mb", "int", "Audio encoder graph metadata arena. Size in MiB; must be positive. Default: 128.", false, "128", "1"},
            {"qwen3_forced_aligner.thinker_prefill_graph_arena_mb", "int", "Thinker prefill graph metadata arena. Size in MiB; must be positive. Default: 256.", false, "256", "1"},
            {"qwen3_forced_aligner.thinker_decode_graph_arena_mb", "int", "Thinker cached-step graph metadata arena. Size in MiB; must be positive. Default: 256.", false, "256", "1"},
            {"qwen3_forced_aligner.thinker_weight_context_mb", "int", "Thinker weight metadata context. Size in MiB; must be positive. Default: 64.", false, "64", "1"},
        };
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
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
        return load_qwen3_forced_aligner_model(request.model_path);
    }
};

}  // namespace

Qwen3ForcedAlignerLoadedModel::Qwen3ForcedAlignerLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3ForcedAlignerLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3ForcedAlignerLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3ForcedAlignerLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Alignment) {
        throw std::runtime_error("Qwen3 forced aligner only supports the Alignment task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 forced aligner currently supports offline sessions");
    }
    return std::make_unique<Qwen3ForcedAlignerSession>(task, options, assets_);
}

std::unique_ptr<Qwen3ForcedAlignerLoadedModel> load_qwen3_forced_aligner_model(const std::filesystem::path & model_path) {
    auto assets = load_qwen3_forced_aligner_assets(model_path);
    return std::make_unique<Qwen3ForcedAlignerLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_forced_aligner_loader() {
    return std::make_shared<Qwen3ForcedAlignerLoader>();
}

}  // namespace engine::models::qwen3_forced_aligner

#include "engine/models/qwen3_asr/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/qwen3_asr/session.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

runtime::ModelMetadata metadata(const Qwen3ASRAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "qwen3_asr";
    out.variant = assets.config.model_size.empty() ? assets.config.model_type : assets.config.model_size;
    out.description = "Qwen3 ASR loaded from local assets.";
    return out;
}

runtime::CapabilitySet capabilities(const Qwen3ASRAssets & assets) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = assets.config.supported_languages;
    out.supports_timestamps = false;
    return out;
}

runtime::ModelCliInterface cli() {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"language", "string", "Recognition language code or label. Omit to detect language automatically; Auto also selects automatic detection.", false},
        {"max_tokens", "int", "Maximum generated transcript tokens; omitted uses the checkpoint generation_config.json (512 when absent).", false, std::nullopt, "1"},
        {"return_timestamps", "bool", "Return word-level timestamps by running Qwen3 ForcedAligner after ASR; default false and requires forced_aligner_path when enabled. Offline only; streaming rejects true even when an aligner is configured.", false, "false"},
        {"audio_chunk_mode", "auto|fixed|vad|none", "Audio chunking mode; default auto uses fixed chunks normally and VAD chunks when timestamps are requested. Offline only; streaming uses fixed-duration windows regardless of this selection.", false, "auto"},
        {"audio_chunk_duration_sec", "float", "Maximum audio chunk duration in seconds; must be > 0 when chunking or streaming. Ignored with audio_chunk_mode=none offline. Default 30 for streaming and offline transcription, or 15 for offline timestamps/VAD chunking.", false, std::nullopt, "0"},
        {"clamp_timestamps_to_audio", "bool", "Opt-in guard for word timestamp output: keep repaired forced-aligner word spans inside the local audio chunk. Defaults to false to preserve existing timestamp repair behavior.", false, "false"},
        {"preserve_punctuation", "bool", "Opt-in text output mode for timestamped chunked ASR: preserve ASR punctuation in text_output instead of rebuilding text from aligned words.", false, "false"},
    };
    out.session_options = {
        {"qwen3_asr.lora", "path", "PEFT LoRA adapter directory or checkpoint file; merged into matching encoder/decoder weights at session creation.", false},
        {"qwen3_asr.lora_scale", "float", "Multiplier for the adapter's alpha/r scaling; default 1.", false, "1"},
        {"qwen3_asr.forced_aligner_path", "path", "Optional Qwen3 ForcedAligner model path required for return_timestamps.", false},
        {"qwen3_asr.vad_model_path", "path", "Optional Silero VAD model path for VAD chunking; default assets/framework/models/silero_vad.", false, "assets/framework/models/silero_vad"},
        {"qwen3_asr.weight_type", "native|f32|f16|bf16|q8_0", "Shared text decoder weight storage fallback; default native.", false, "native"},
        {"qwen3_asr.audio_encoder_weight_type", "native|f32|f16", "Audio encoder weight storage type; default native.", false, "native"},
        {"qwen3_asr.text_decoder_weight_type", "native|f32|f16|bf16|q8_0", "Text decoder matmul weight storage type; defaults to weight_type when set, otherwise native.", false},
        {"qwen3_asr.audio_encoder_graph_arena_mb", "int", "Audio encoder graph arena size in MiB; default 128.", false, "128", "1"},
        {"qwen3_asr.text_decoder_prefill_graph_arena_mb", "int", "Text decoder prefill graph arena size in MiB; default 256.", false, "256", "1"},
        {"qwen3_asr.text_decoder_decode_graph_arena_mb", "int", "Text decoder cached-step graph arena size in MiB; default 256.", false, "256", "1"},
        {"qwen3_asr.text_decoder_weight_context_mb", "int", "Text decoder weight context size in MiB; default 64.", false, "64", "1"},
    };
    return out;
}

class Qwen3ASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_asr";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_timestamps = true;
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
        const auto assets = load_qwen3_asr_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli();
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
        return load_qwen3_asr_model(request.model_path);
    }

private:
};

}  // namespace

Qwen3ASRLoadedModel::Qwen3ASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Qwen3ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3ASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3ASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3ASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Qwen3 ASR only supports the Asr task");
    }
    if (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 ASR supports offline and streaming sessions");
    }
    return std::make_unique<Qwen3ASRSession>(task, options, assets_);
}

std::unique_ptr<Qwen3ASRLoadedModel> load_qwen3_asr_model(const std::filesystem::path & model_path) {
    auto assets = load_qwen3_asr_assets(model_path);
        return std::make_unique<Qwen3ASRLoadedModel>(
        metadata(*assets),
        capabilities(*assets),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_asr_loader() {
    return std::make_shared<Qwen3ASRLoader>();
}

}  // namespace engine::models::qwen3_asr

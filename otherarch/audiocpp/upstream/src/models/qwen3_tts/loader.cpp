#include "engine/models/qwen3_tts/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"
#include "engine/framework/rknn/engine_resources.h"
#include "engine/models/qwen3_tts/session.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

std::vector<std::string> supported_languages(const Qwen3TTSConfig & config) {
    std::vector<std::string> languages;
    languages.reserve(config.talker.codec_language_id.size() + 1);
    languages.push_back("Auto");
    for (const auto & [language, id] : config.talker.codec_language_id) {
        (void) id;
        languages.push_back(language);
    }
    std::sort(languages.begin() + 1, languages.end());
    return languages;
}

runtime::ModelMetadata metadata(const Qwen3TTSAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "qwen3_tts";
    out.variant = assets.config.tts_model_size + "-" + assets.config.tts_model_type;
    out.description = "Qwen3 TTS loaded from local extracted assets.";
    return out;
}

runtime::CapabilitySet capabilities(const Qwen3TTSAssets & assets) {
    runtime::CapabilitySet out;
    if (assets.config.variant == Qwen3TTSVariant::Base) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceCloning, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_speaker_reference = true;
    } else if (assets.config.variant == Qwen3TTSVariant::VoiceDesign) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
            {runtime::VoiceTaskKind::VoiceDesign, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_style_condition = true;
    } else if (assets.config.variant == Qwen3TTSVariant::CustomVoice) {
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_style_condition = true;
    }
    out.languages = supported_languages(assets.config);
    return out;
}

runtime::ModelCliInterface cli(const Qwen3TTSAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"stream_chunk_policy", "grow|fixed", "Streaming codec chunk schedule: grow starts with one frame and doubles to stream_frames_per_event; fixed uses that size throughout. Ignored offline.", false, "grow"},
        {"stream_frames_per_event", "int", "Codec frames per fixed streaming event, or the maximum for growing events. Smaller values reduce chunk latency but increase decoding overhead. Ignored in offline mode.", false, "12", "1"},
        {"language", "string", "Target synthesis language code or label; default auto lets the model infer from text and voice settings.", false, "auto"},
        {"max_tokens", "int", "Maximum generated codec frames per text chunk; omitted uses the checkpoint generation_config.json (2048 when absent).", false, std::nullopt, "1"},
        {"do_sample", "bool", "Sample the main codec stream instead of greedy decoding; default true.", false, "true"},
        {"temperature", "float", "Main codec sampling temperature; must be > 0 when do_sample=true. Ignored when do_sample=false; default 0.9.", false, "0.9", "0"},
        {"top_k", "int", "Main codec top-k sampling limit; default 50, 0 disables top-k filtering.", false, "50", "0"},
        {"top_p", "float", "Main codec nucleus sampling limit; default 1.0.", false, "1", "0", "1"},
        {"repetition_penalty", "float", "Main codec repetition penalty; must be > 0, default 1.05.", false, "1.05", "0"},
        {"subtalker_do_sample", "bool", "Enable subtalker sampling for the remaining audio codebooks instead of greedy decoding; default true.", false, "true"},
        {"subtalker_temperature", "float", "Subtalker sampling temperature; must be > 0 when subtalker_do_sample=true. Ignored when subtalker_do_sample=false; default 0.9.", false, "0.9", "0"},
        {"subtalker_top_k", "int", "Subtalker top-k sampling limit; default 50, 0 disables top-k filtering.", false, "50", "0"},
        {"subtalker_top_p", "float", "Subtalker nucleus sampling limit; default 1.0.", false, "1", "0", "1"},
        {"seed", "int", "Codec sampling seed; omitted requests choose a random seed.", false, std::nullopt, "0", "4294967295"},
        {"reference_text", "string", "Transcript matching the reference audio for Base voice cloning; required unless x_vector_only_mode=true.", false},
        {"x_vector_only_mode", "bool", "Use only the speaker embedding for Base voice cloning instead of ICL reference codec prompting; default false.", false, "false"},
        {"instruct", "string", "Optional natural-language voice or style instruction for VoiceDesign and CustomVoice variants; empty instructions are accepted.", false},
        {"voice_id", "aiden|dylan|eric|ono_anna|ryan|serena|sohee|uncle_fu|vivian", "Named preset speaker from the CustomVoice checkpoint. These are the official checkpoint's presets; available names are determined by the loaded checkpoint. Legacy speaker remains accepted.", false},
        {"text_chunk_size", "int", "Maximum UTF-8 codepoints per long-form text chunk; default 8192.", false, "8192", "1"},
    };
    out.session_options = {
        {"qwen3_tts.lora", "path", "PEFT LoRA adapter directory or checkpoint file; optional speaker_embedding and tts_config.json sidecars provide a trained custom voice.", false},
        {"qwen3_tts.lora_scale", "float", "Multiplier for the adapter's alpha/r scaling; default 1.", false, "1"},
        {"qwen3_tts.mem_saver", "bool", "Release the talker cached-step graph after each request; default false.", false, "false"},
        {"qwen3_tts.stitch_decode", "bool", "Voice the concatenated speech codes of all text chunks in one decoder pass instead of voicing each chunk separately; default false. Per-chunk voicing re-voices the voice-clone reference once per chunk and restarts the stateful vocoder cache at every chunk boundary; the stitched pass costs device memory proportional to the utterance.", false, "false"},
        {"qwen3_tts.perf_mode", "off|flash_attention|optimized", "Performance preset; default off. flash_attention enables flash attention; optimized also defaults convolution weights to f16, streaming decoding to stateful, and talker KV alignment to 256. Explicit session options override those defaults. May change audio and increase VRAM.", false, "off"},
        {"qwen3_tts.talker_kv_cache_alignment", "int", "Round talker KV capacity to this many rows. Defaults to 1 (no padding), or 256 with perf_mode=optimized. Explicit values override the preset. Larger alignments may use more memory and change numerical results.", false, "1", "1"},
        {"qwen3_tts.stream_decoder", "windowed|stateful", "GGML streaming decoder; stateful carries convolution and attention history and caches graphs for exact chunk lengths. Uses more graph memory and can change audio; default windowed preserves existing behavior.", false, "windowed"},
        {"qwen3_tts.voice_prompt_cache_slots", "int", "Voice prompt cache slots; default 1.", false, "1", "0"},
        {"qwen3_tts.weight_type", "native|f32|f16|bf16|q8_0", "Shared talker matmul weight storage type; default native.", false, "native"},
        {"qwen3_tts.talker_weight_type", "native|f32|f16|bf16|q8_0", "Talker weight storage type; defaults to weight_type when set, otherwise native.", false},
        {"qwen3_tts.speech_encoder_weight_type", "native|f32|f16|bf16|q8_0", "Reference speech encoder matmul weight storage type; default native.", false, "native"},
        {"qwen3_tts.speech_decoder_weight_type", "native|f32|f16|bf16|q8_0", "Speech decoder matmul weight storage type; default native.", false, "native"},
        {"qwen3_tts.conv_weight_type", "native|f32|f16", "Speech tokenizer and speaker-encoder convolution weight storage type; default f32.", false, "f32"},
        {"qwen3_tts.talker_graph_arena_mb", "int", "Talker graph arena size in MiB; default 32.", false, "32", "1"},
        {"qwen3_tts.speech_encoder_graph_arena_mb", "int", "Reference speech encoder graph arena size in MiB; default 32.", false, "32", "1"},
        {"qwen3_tts.speech_decoder_graph_arena_mb", "int", "Speech decoder graph arena size in MiB; default 32.", false, "32", "1"},
        {"qwen3_tts.speaker_encoder_graph_arena_mb", "int", "Speaker encoder graph arena size in MiB; default 32.", false, "32", "1"},
        {"qwen3_tts.talker_constant_context_mb", "int", "Talker constant tensor context size in MiB; default 4.", false, "4", "1"},
        {"qwen3_tts.code_predictor_constant_context_mb", "int", "Code predictor constant tensor context size in MiB; default 4.", false, "4", "1"},
        {"qwen3_tts.speech_decoder_constant_context_mb", "int", "Speech decoder constant tensor context size in MiB; default 4.", false, "4", "1"},
    };
    return out;
}

class Qwen3TTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "qwen3_tts";
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
        const auto assets = load_qwen3_tts_assets(request.model_path);
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
        return load_qwen3_tts_model(request.model_path);
    }

    runtime::ModelResourcePlan model_resources(
        const std::filesystem::path & model_path,
        const std::unordered_map<std::string, std::string> & session_options) const override {
        runtime::ModelResourcePlan plan;
        const auto option = session_options.find("qwen3_tts.exec_plan");
        if (option == session_options.end()) {
            return plan;
        }
        std::vector<std::string> paths;
        try {
            const auto value = io::json::parse(option->second);
            const auto * decoder = value.find("speech_decoder");
            const auto * impl = decoder ? decoder->find("impl") : nullptr;
            if (!impl || impl->as_string() != "rknn") {
                return plan;
            }
            const auto & artifacts = decoder->require("artifacts");
            paths.push_back(io::json::require_string(artifacts, "main"));
            for (const char * const role : {"first", "window"}) {
                if (const auto * artifact = artifacts.find(role)) {
                    paths.push_back(artifact->as_string());
                }
            }
        } catch (const std::runtime_error &) {
            // Session loading reports invalid execution plans.
            return plan;
        }
        return rknn::engine_resources(paths, model_path);
    }
};

}  // namespace

Qwen3TTSLoadedModel::Qwen3TTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Qwen3TTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Qwen3TTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Qwen3TTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Qwen3TTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (assets_->config.variant == Qwen3TTSVariant::Base &&
        task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Qwen3 base TTS model supports TTS and voice cloning");
    }
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign &&
        task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Qwen3 voice design model supports TTS and voice design");
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 custom voice model only supports the Tts task");
    }
    return std::make_unique<Qwen3TTSSession>(task, options, assets_);
}

std::unique_ptr<Qwen3TTSLoadedModel> load_qwen3_tts_model(const std::filesystem::path & model_path) {
    auto assets = load_qwen3_tts_assets(model_path);
    return std::make_unique<Qwen3TTSLoadedModel>(metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_qwen3_tts_loader() {
    return std::make_shared<Qwen3TTSLoader>();
}

}  // namespace engine::models::qwen3_tts

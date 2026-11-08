#include "engine/community_models/audio8_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::audio8_asr {
namespace json = engine::io::json;
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kWeightContextBytes = 64ull * 1024ull * 1024ull;
// Languages advertised by the loader and the loaded model; kept in one list.
const std::vector<std::string> kLanguages = {
    "Chinese", "English", "Cantonese", "French", "German", "Japanese", "Korean"};

std::filesystem::path default_spec_path() {
    return engine::model_spec::default_spec_path("audio8_asr");
}

std::shared_ptr<const Audio8ASRAssets> require_assets(std::shared_ptr<const Audio8ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Audio8 ASR session requires assets");
    }
    return assets;
}

void validate_audio_encoder_weight_storage(assets::TensorStorageType storage_type) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error("audio8_asr.audio_encoder_weight_type currently supports only native, f32, and f16");
}

void validate_matmul_weight_storage(assets::TensorStorageType storage_type) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error("audio8_asr.weight_type supports only native, f32, f16, bf16, and q8_0");
}

assets::TensorStorageType parse_encoder_weight_storage(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    // The encoder rejects on-the-fly quantized weights; already-quantized
    // GGUF tensors load as Native regardless of this option.
    const auto type = runtime::parse_tensor_storage_option(
        options,
        "audio8_asr.audio_encoder_weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32, assets::TensorStorageType::F16});
    validate_audio_encoder_weight_storage(type);
    return type;
}

assets::TensorStorageType parse_matmul_weight_storage(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    const auto type = runtime::parse_tensor_storage_option(
        options,
        "audio8_asr.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    validate_matmul_weight_storage(type);
    return type;
}

size_t encoder_graph_arena_bytes(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    return runtime::parse_size_mb_option(options, {"audio8_asr.encoder_graph_arena_mb"}, 128ull * 1024ull * 1024ull);
}

size_t projector_graph_arena_bytes(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    return runtime::parse_size_mb_option(options, {"audio8_asr.projector_graph_arena_mb"}, 256ull * 1024ull * 1024ull);
}

size_t prefill_graph_arena_bytes(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    return runtime::parse_size_mb_option(options, {"audio8_asr.prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull);
}

size_t decode_graph_arena_bytes(const runtime::SessionOptions & session_options) {
    const auto & options = session_options.options;
    return runtime::parse_size_mb_option(options, {"audio8_asr.decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull);
}

// The reference processor hands the model a bfloat16 mel tensor; replicate
// that rounding (audio8_asr_round_f32_to_bf16) before encoding.

// Join window transcripts. Whitespace is only inserted between ASCII words;
// CJK text has no word boundaries, so a byte >= 0x80 on either side means the
// windows are concatenated directly (the reference has no multi-window path;
// this mirrors how CJK text is written).
void append_clip_transcript(std::string & merged, std::string clip_text) {
    clip_text = engine::io::trim_ascii_whitespace(std::move(clip_text));
    if (clip_text.empty()) {
        return;
    }
    if (!merged.empty()) {
        const unsigned char last = static_cast<unsigned char>(merged.back());
        const unsigned char first = static_cast<unsigned char>(clip_text.front());
        if (last < 0x80 && first < 0x80 && last != ' ' && first != ' ') {
            merged.push_back(' ');
        }
    }
    merged += clip_text;
}

class Audio8ASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "audio8_asr";
    }

    std::vector<std::string> family_aliases() const override {
        return {"arkasr"};
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        out.languages = kLanguages;
        out.supports_timestamps = false;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value()) {
            const auto & hint = *request.family_hint;
            if (hint != family()) {
                const auto aliases = family_aliases();
                if (std::find(aliases.begin(), aliases.end(), hint) == aliases.end()) {
                    return false;
                }
            }
        }
        try {
            const auto resources = engine::model_spec::load_resource_bundle(request.model_path, default_spec_path());
            const auto config_root = resources.parse_json("config");
            return json::require_string(config_root, "model_type") == "arkasr";
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto resources = engine::model_spec::load_resource_bundle(
            request.model_path,
            default_spec_path());
        runtime::ModelInspection inspection;
        inspection.cli.session_options = {
            {"audio8_asr.weight_type", "native|f32|f16|bf16|q8_0", "Decoder and adapter weight storage type. Default: native.", false, "native"},
            {"audio8_asr.audio_encoder_weight_type", "native|f32|f16", "Audio encoder weight storage type (quantized storage is rejected for the encoder; already-quantized GGUF tensors load as native). Default: native.", false, "native"},
            {"audio8_asr.encoder_graph_arena_mb", "int", "Audio encoder graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Default: 128.", false, "128", "1"},
            {"audio8_asr.projector_graph_arena_mb", "int", "Adapter (MLP tower + pooling + projector) graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Default: 256.", false, "256", "1"},
            {"audio8_asr.prefill_graph_arena_mb", "int", "Decoder prefill graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Default: 256.", false, "256", "1"},
            {"audio8_asr.decode_graph_arena_mb", "int", "Decoder step decode graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Default: 256.", false, "256", "1"},
        };
        inspection.model_root = resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = "0.1b";
        inspection.metadata.description = "Audio8-ASR-0.1B multilingual ASR model.";
        inspection.capabilities = advertised_capabilities();
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            default_spec_path(),
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            default_spec_path(),
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_audio8_asr_model(request);
    }
};

}  // namespace

Audio8ASRSession::Audio8ASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Audio8ASRAssets> assets)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      frontend_(assets_->encoder_assets),
      audio_encoder_(
          assets_->encoder_assets->model_weights,
          assets_->encoder_assets->config.audio_encoder,
          execution_context(),
          {encoder_graph_arena_bytes(RuntimeSessionBase::options()),
           parse_encoder_weight_storage(RuntimeSessionBase::options()), "audio8_asr.audio_encoder"},
          assets_->encoder_assets->config.audio_encoder_binding),
      projector_(
          assets_->resources.open_tensor_source("weights"),
          assets_->config.tower,
          execution_context(),
          projector_graph_arena_bytes(RuntimeSessionBase::options()),
          kWeightContextBytes,
          parse_matmul_weight_storage(RuntimeSessionBase::options())),
      thinker_(
          assets_->resources.open_tensor_source("weights"),
          assets_->config.text_decoder,
          execution_context(),
          prefill_graph_arena_bytes(RuntimeSessionBase::options()),
          decode_graph_arena_bytes(RuntimeSessionBase::options()),
          kWeightContextBytes,
          parse_matmul_weight_storage(RuntimeSessionBase::options())) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Audio8 ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Audio8 ASR only supports offline sessions");
    }
    // All weight stores have uploaded by now; drop the resident file blob.
    assets_->resources.open_tensor_source("weights")->release_storage();
}

Audio8ASRSession::~Audio8ASRSession() = default;

std::string Audio8ASRSession::family() const {
    return "audio8_asr";
}

runtime::VoiceTaskKind Audio8ASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Audio8ASRSession::run_mode() const {
    return task_.mode;
}

void Audio8ASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

std::string Audio8ASRSession::transcribe_clip(const runtime::AudioBuffer & audio) {
    auto features = frontend_.extract(audio);
    // Match the reference mel dtype (bfloat16) before encoding.
    for (auto & value : features.values) {
        value = audio8_asr_round_f32_to_bf16(value);
    }
    const auto encoder_output = audio_encoder_.encode(features);

    const int64_t mel_frames = features.frames;
    const int64_t audio_tokens = audio8_asr_prompt_audio_token_count(mel_frames, assets_->config.merge_factor);
    const auto embeddings = projector_.project(encoder_output.values, encoder_output.tokens, audio_tokens);

    // Prompt: <|user|><|begin_of_audio|><|audio|>xN<|end_of_audio|>Please
    // transcribe this audio.<|assistant|>
    const auto & config = assets_->config;
    Audio8ASRPrompt prompt;
    prompt.input_ids.push_back(static_cast<int32_t>(config.user_token_id));
    prompt.input_ids.push_back(static_cast<int32_t>(config.begin_audio_token_id));
    for (int64_t i = 0; i < audio_tokens; ++i) {
        prompt.audio_token_positions.push_back(static_cast<int32_t>(prompt.input_ids.size()));
        prompt.input_ids.push_back(static_cast<int32_t>(config.text_decoder.audio_token_id));
    }
    prompt.input_ids.push_back(static_cast<int32_t>(config.end_audio_token_id));
    const auto instruction_ids = assets_->tokenizer->encode("Please transcribe this audio.", false);
    prompt.input_ids.insert(prompt.input_ids.end(), instruction_ids.begin(), instruction_ids.end());
    prompt.input_ids.push_back(static_cast<int32_t>(config.assistant_token_id));

    const Audio8ASRGenerationOptions generation{config.text_decoder.max_new_tokens};
    const auto generated = thinker_.generate(prompt, embeddings, generation);
    if (generated.token_ids.empty()) {
        return "";
    }
    std::vector<int32_t> filtered;
    filtered.reserve(generated.token_ids.size());
    for (const int32_t id : generated.token_ids) {
        if (id == static_cast<int32_t>(config.text_decoder.pad_token_id) ||
            std::find(
                config.text_decoder.eos_token_ids.begin(),
                config.text_decoder.eos_token_ids.end(),
                static_cast<int64_t>(id)) != config.text_decoder.eos_token_ids.end() ||
            assets_->tokenizer->is_control_token_id(id)) {
            continue;
        }
        filtered.push_back(id);
    }
    if (filtered.empty()) {
        return "";
    }
    return assets_->tokenizer->decode(filtered);
}

runtime::Transcript Audio8ASRSession::transcribe_audio(const runtime::AudioBuffer & audio) {
    if (audio.samples.empty()) {
        return {"", ""};
    }
    // The reference clips input at max_audio_samples (16 kHz samples, 30 s by
    // default); longer recordings are covered by transcribing fixed windows
    // and joining the text. The window is sized in the input sample domain
    // (interleaved values, input rate) so it always covers the same wall
    // time as the reference clip. A tail shorter than half a second folds
    // into the previous window instead of triggering a near-silent pass.
    const int64_t channels = std::max(1, audio.channels);
    const int64_t sample_rate = std::max(1, audio.sample_rate);
    const int64_t window_values =
        assets_->config.max_audio_samples * channels * sample_rate / 16000;
    const int64_t min_tail_values = sample_rate * channels / 2;
    std::string merged;
    int64_t offset = 0;
    const int64_t total = static_cast<int64_t>(audio.samples.size());
    while (offset < total) {
        int64_t span = std::min<int64_t>(window_values, total - offset);
        if (total - (offset + span) < min_tail_values) {
            span = total - offset;
        }
        runtime::AudioBuffer clip;
        clip.sample_rate = audio.sample_rate;
        clip.channels = audio.channels;
        clip.samples.assign(
            audio.samples.begin() + static_cast<ptrdiff_t>(offset),
            audio.samples.begin() + static_cast<ptrdiff_t>(offset + span));
        append_clip_transcript(merged, transcribe_clip(clip));
        offset += span;
    }
    return {engine::io::trim_ascii_whitespace(std::move(merged)), ""};
}

runtime::TaskResult Audio8ASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Audio8 ASR run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Audio8 ASR run() requires audio_input");
    }
    const auto wall_start = Clock::now();
    auto transcript = transcribe_audio(*request.audio_input);
    runtime::TaskResult result;
    result.text_output = std::move(transcript);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

Audio8ASRLoadedModel::Audio8ASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const Audio8ASRAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & Audio8ASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & Audio8ASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> Audio8ASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    // The session constructor validates the task/mode combination.
    return std::make_unique<Audio8ASRSession>(task, options, assets_);
}

std::unique_ptr<Audio8ASRLoadedModel> load_audio8_asr_model(const runtime::ModelLoadRequest & request) {
    auto assets = load_audio8_asr_assets(request.model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "audio8_asr";
    metadata.variant = "0.1b";
    metadata.description = "Audio8-ASR-0.1B multilingual ASR model loaded from local assets.";

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.languages = assets->config.supported_languages;
    capabilities.supports_timestamps = false;

    return std::make_unique<Audio8ASRLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_audio8_asr_loader() {
    return std::make_shared<Audio8ASRLoader>();
}

}  // namespace engine::community_models::audio8_asr

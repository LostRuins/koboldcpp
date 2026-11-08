#include "engine/models/silero_vad/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::models::silero_vad {
namespace {

std::filesystem::path resolve_weight_path(const std::filesystem::path & model_path) {
    return resolve_silero_assets(model_path).checkpoint_path;
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    if (engine::io::is_existing_file(request.model_path)) {
        return {{"default", std::filesystem::weakly_canonical(request.model_path)}};
    }
    return runtime::discover_named_assets(
        std::filesystem::weakly_canonical(request.model_path),
        {"silero_vad_16k.safetensors"});
}

SileroVADConfig silero_config_from_options(
    const std::unordered_map<std::string, std::string> & options,
    const SileroVADConfig * fallback = nullptr) {
    SileroVADConfig config = fallback != nullptr ? *fallback : SileroVADConfig{};
    if (const auto value = runtime::parse_float_option(options, {"threshold"})) {
        config.threshold = *value;
    }
    if (const auto value = runtime::parse_int_option(options, {"min_speech_duration_ms"})) {
        config.min_speech_duration_ms = *value;
    }
    if (const auto value = runtime::parse_int_option(options, {"min_silence_duration_ms"})) {
        config.min_silence_duration_ms = *value;
    }
    if (const auto value = runtime::parse_int_option(options, {"speech_pad_ms"})) {
        config.speech_pad_ms = *value;
    }
    if (const auto value = runtime::parse_float_option(options, {"max_speech_duration_sec", "max_speech_duration_s"})) {
        config.max_speech_duration_s = *value;
    }
    if (const auto value = runtime::parse_float_option(options, {"neg_threshold"})) {
        config.neg_threshold = *value;
    }
    if (const auto value = runtime::parse_int_option(options, {"min_silence_at_max_speech_ms"})) {
        config.min_silence_at_max_speech_ms = *value;
    }
    if (const auto value = runtime::find_option(options, {"use_max_poss_sil_at_max_speech"})) {
        config.use_max_poss_sil_at_max_speech = runtime::parse_bool_option(*value, "use_max_poss_sil_at_max_speech");
    }
    return config;
}

engine::assets::TensorStorageType silero_weight_type_from_options(const runtime::SessionOptions & options) {
    const auto it = options.options.find("silero_vad.weight_type");
    if (it == options.options.end()) {
        return engine::assets::TensorStorageType::Native;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("silero_vad.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

class SileroVADLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "silero_vad";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto assets = resolve_silero_assets(request.model_path);
            if (!request.family_hint.has_value() &&
                engine::io::is_existing_file(request.model_path) &&
                assets.checkpoint_path.filename() != "silero_vad_16k.safetensors") {
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Silero VAD does not expose selectable config assets");
        }
        const auto assets = resolve_silero_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = "16k";
        inspection.metadata.description = "Silero VAD loaded from safetensors weights.";
        inspection.metadata.weight_candidates = {"silero_vad_16k.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_silero_vad_model(request);
    }
};

}  // namespace

SileroVADSession::SileroVADSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SileroWeights> weights)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      default_config_(silero_config_from_options(options.options)),
      weight_storage_type_(silero_weight_type_from_options(options)),
      runtime_(std::move(weights), execution_context(), weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Vad) {
        throw std::runtime_error("Silero VAD only supports VoiceTaskKind::Vad");
    }
}

std::string SileroVADSession::family() const {
    return "silero_vad";
}

runtime::VoiceTaskKind SileroVADSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SileroVADSession::run_mode() const {
    return task_.mode;
}

void SileroVADSession::prepare(const runtime::SessionPreparationRequest & request) {
    int sample_rate = 16000;
    if (request.audio.has_value() && request.audio->sample_rate > 0) {
        sample_rate = request.audio->sample_rate;
    }
    runtime_.prepare(sample_rate);
    mark_prepared();
}

runtime::TaskResult SileroVADSession::run(const runtime::TaskRequest & request) {
    require_prepared("Silero VAD run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Silero VAD offline run called on non-offline session");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Silero VAD offline run requires audio_input");
    }
    const auto config = silero_config_from_options(request.options, &default_config_);
    const auto wall_start = std::chrono::steady_clock::now();
    auto result = runtime_.run_offline(*request.audio_input, config);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

void SileroVADSession::reset() {
    require_prepared("Silero VAD reset()");
    runtime_.reset(1);
}

runtime::StreamEvent SileroVADSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Silero VAD process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Silero VAD process_audio_chunk called on non-streaming session");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    auto event = runtime_.process_chunk(chunk, default_config_);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return event;
}

runtime::TaskResult SileroVADSession::finalize() {
    require_prepared("Silero VAD finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Silero VAD finalize called on non-streaming session");
    }
    return runtime_.finalize_stream(default_config_);
}

SileroVADLoadedModel::SileroVADLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const SileroWeights> weights)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      weights_(std::move(weights)) {}

const runtime::ModelMetadata & SileroVADLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & SileroVADLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> SileroVADLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<SileroVADSession>(task, options, weights_);
}

std::unique_ptr<SileroVADLoadedModel> load_silero_vad_model(const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::runtime_error("Silero VAD does not expose selectable config assets");
    }
    const auto weights = discover_weight_assets(request);
    const auto * selected_weight = runtime::select_named_asset(weights, request.weight_id, "weight");
    const auto weight_path = selected_weight != nullptr ? selected_weight->path : resolve_weight_path(request.model_path);
    const auto assets = resolve_silero_assets(weight_path);
    runtime::ModelMetadata metadata;
    metadata.family = "silero_vad";
    metadata.variant = "16k";
    metadata.description = "Silero VAD loaded from safetensors weights.";
    metadata.weight_candidates = {"silero_vad_16k.safetensors"};
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    capabilities.supports_timestamps = true;
    return std::make_unique<SileroVADLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        load_silero_weights_cached(assets.checkpoint_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_silero_vad_loader() {
    return std::make_shared<SileroVADLoader>();
}

}  // namespace engine::models::silero_vad

#include "engine/models/pocket_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "engine/models/pocket_tts/voice_state_assets.h"
#include "graph_common.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::pocket_tts {
namespace {

constexpr int64_t kCpuPromptCapacityFloor = 32;
constexpr int64_t kCpuGenerationCapacityFloor = 160;
constexpr int64_t kGenerationCapacityQuantum = 16;
constexpr int64_t kDefaultVoiceStateCacheSlots = 4;

int64_t next_power_of_two(int64_t value) {
    if (value <= 0) {
        throw std::runtime_error("PocketTTS capacity value must be positive");
    }
    int64_t capacity = 1;
    while (capacity < value) {
        if (capacity > (std::numeric_limits<int64_t>::max() / 2)) {
            throw std::runtime_error("PocketTTS capacity value is too large to round to power of two");
        }
        capacity *= 2;
    }
    return capacity;
}

int64_t round_up_to_multiple(int64_t value, int64_t multiple) {
    if (value <= 0 || multiple <= 0) {
        throw std::runtime_error("PocketTTS capacity rounding requires positive inputs");
    }
    const int64_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    if (value > std::numeric_limits<int64_t>::max() - (multiple - remainder)) {
        throw std::runtime_error("PocketTTS capacity value is too large to round");
    }
    return value + (multiple - remainder);
}

TextConditionerConfig make_text_config(const PocketTTSAssets & manifest) {
    return TextConditionerConfig{
        manifest.model_config.flow_dim,
    };
}

FlowLMConfig make_flow_config(const PocketTTSAssets & manifest) {
    return FlowLMConfig{
        manifest.model_config.flow_dim,
        manifest.model_config.flow_heads,
        manifest.model_config.flow_intermediate_size,
        manifest.model_config.latent_dim,
        manifest.model_config.flow_hidden_size,
        manifest.model_config.flow_layers,
        1.0e-5F,
        1.0e-6F,
        manifest.model_config.flow_depth,
    };
}

MimiDecoderConfig make_decoder_config(const PocketTTSAssets & manifest) {
    return MimiDecoderConfig{
        manifest.model_config.latent_dim,
        manifest.model_config.mimi_dim,
        manifest.model_config.mimi_heads,
        manifest.model_config.mimi_intermediate_size,
        manifest.model_config.mimi_layers,
        manifest.model_config.mimi_encoder_upsample_stride,
    };
}

bool has_voice_selection(const VoiceConfig & voice) {
    return !voice.preset_name.empty()
        || !voice.embedding_path.empty()
        || !voice.clone_audio_path.empty()
        || voice.clone_audio.has_value();
}

void validate_supported_style(const std::optional<runtime::VoiceCondition> & voice) {
    if (!voice.has_value() || !voice->style.has_value()) {
        return;
    }
    const auto & style = *voice->style;
    if (style.language.has_value() || style.emotion.has_value() || style.speaking_rate.has_value()
        || style.pitch_shift.has_value() || style.energy_scale.has_value() || !style.tags.empty()) {
        throw std::runtime_error("PocketTTS framework session does not support runtime style controls");
    }
}

void apply_voice_condition(const std::optional<runtime::VoiceCondition> & voice, VoiceConfig & voice_config) {
    if (voice.has_value() && voice->speaker.has_value()) {
        const auto & speaker = *voice->speaker;
        if (speaker.cached_voice_id.has_value()) {
            voice_config.preset_name = *speaker.cached_voice_id;
        }
        if (speaker.audio.has_value()) {
            voice_config.clone_audio = *speaker.audio;
        }
    }
}

void apply_generation_options(
    const std::unordered_map<std::string, std::string> & options,
    GenerationRequest & generation_request) {
    if (const auto embedding_path = runtime::find_option(options, {"voice_embedding_file", "pocket_tts.voice_embedding_file", "voice_embedding_path", "pocket_tts.voice_embedding_path"})) {
        generation_request.voice.embedding_path = *embedding_path;
    }
    if (const auto clone_text = runtime::find_option(options, {"voice_clone_text", "pocket_tts.voice_clone_text"})) {
        generation_request.voice.clone_prompt_text = *clone_text;
    }
    if (const auto truncate = runtime::find_option(options, {"truncate_clone_audio", "pocket_tts.truncate_clone_audio"})) {
        generation_request.voice.truncate_clone_audio = runtime::parse_bool_option(*truncate, "truncate_clone_audio");
    }
}

void apply_session_generation_options(
    const std::unordered_map<std::string, std::string> & options,
    GenerationRequest & generation_request) {
    apply_generation_options(options, generation_request);
    if (const auto max_steps = runtime::parse_int_option(options, {"max_frames", "max_steps", "pocket_tts.max_steps"})) {
        generation_request.max_steps = *max_steps;
    }
    if (const auto max_tokens = runtime::parse_int_option(
            options,
            {"text_chunk_tokens", "max_tokens", "pocket_tts.max_tokens"})) {
        generation_request.max_tokens = *max_tokens;
    }
    generation_request.text_chunk_size = engine::text::parse_text_chunk_size_override(options);
    if (const auto temperature = runtime::parse_float_option(options, {"temperature", "pocket_tts.temperature"})) {
        generation_request.temperature = *temperature;
    }
    if (const auto noise_clamp = runtime::parse_float_option(options, {"noise_clamp", "pocket_tts.noise_clamp"})) {
        generation_request.noise_clamp = *noise_clamp;
    }
    if (const auto eos_threshold = runtime::parse_float_option(options, {"eos_threshold", "pocket_tts.eos_threshold"})) {
        generation_request.eos_threshold = *eos_threshold;
    }
    generation_request.seed = runtime::parse_u32_option(options, {"seed", "pocket_tts.seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto noise_file = runtime::find_option(options, {"noise_file", "pocket_tts.noise_file"})) {
        generation_request.noise_schedule_path = *noise_file;
    }
}

void apply_request_generation_options(
    const std::unordered_map<std::string, std::string> & options,
    GenerationRequest & generation_request) {
    if (const auto frames_after_eos = runtime::parse_int_option(
            options,
            {"frames_after_eos", "pocket_tts.frames_after_eos"})) {
        generation_request.frames_after_eos = *frames_after_eos;
    }
    if (const auto max_tokens = runtime::parse_int_option(
            options,
            {"text_chunk_tokens", "max_tokens", "pocket_tts.max_tokens"})) {
        generation_request.max_tokens = *max_tokens;
    }
    generation_request.text_chunk_size = engine::text::parse_text_chunk_size_override(options);
}

GenerationRequest build_generation_request(const runtime::TaskRequest & request, float default_temperature) {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("PocketTTS run requires text_input");
    }
    validate_supported_style(request.voice);

    GenerationRequest generation_request;
    generation_request.temperature = default_temperature;
    generation_request.text = request.text_input->text;
    apply_voice_condition(request.voice, generation_request.voice);
    apply_request_generation_options(request.options, generation_request);
    return generation_request;
}

GenerationRequest build_preparation_generation_request(
    const runtime::SessionPreparationRequest & request,
    float default_temperature) {
    validate_supported_style(request.voice);

    GenerationRequest generation_request;
    generation_request.temperature = default_temperature;
    if (request.text.has_value()) {
        generation_request.text = request.text->text;
    }
    apply_voice_condition(request.voice, generation_request.voice);
    apply_session_generation_options(request.options, generation_request);
    return generation_request;
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, and f16");
}

}  // namespace

namespace {

void validate_generation_request(const GenerationRequest & request) {
    if (request.text.empty()) {
        throw std::runtime_error("PocketTTS text prompt is required");
    }
    if (request.max_steps < 0) {
        throw std::runtime_error("PocketTTS max_frames must be non-negative");
    }
    if (request.max_tokens <= 0) {
        throw std::runtime_error("PocketTTS text_chunk_tokens must be positive");
    }
    if (request.frames_after_eos < -1) {
        throw std::runtime_error("PocketTTS frames_after_eos must be -1 or non-negative");
    }
    if (request.temperature <= 0.0F) {
        throw std::runtime_error("PocketTTS temperature must be positive");
    }
}

// An explicit text_chunk_size keeps the character budget it always meant. Otherwise the text
// is cut like the reference cuts it: on sentence boundaries, at most max_tokens tokens per
// chunk. The streaming path used to pass max_tokens (50) as a CHARACTER budget, which cut
// nearly every sentence in two, and the offline path used 256 characters.
std::vector<std::string> split_generation_text(
    const TextConditioner & text_conditioner,
    const PocketTTSAssets & manifest,
    const GenerationRequest & request) {
    if (request.text_chunk_size.has_value()) {
        return engine::text::split_text_chunks(request.text, *request.text_chunk_size);
    }
    return text_conditioner.split_into_sentence_chunks(manifest, request.text, request.max_tokens);
}

void trace_text_chunking(const char * prefix, const GenerationRequest & request, size_t chunk_count) {
    const std::string key = std::string(prefix) +
        (request.text_chunk_size.has_value() ? "text_chunk_size" : "max_tokens");
    engine::debug::trace_log_scalar(
        key,
        request.text_chunk_size.has_value() ? *request.text_chunk_size : static_cast<int64_t>(request.max_tokens));
    engine::debug::trace_log_scalar(std::string(prefix) + "text_chunk_count", static_cast<int64_t>(chunk_count));
}

int estimate_max_steps(const PocketTTSAssets & manifest, int64_t token_count) {
    constexpr double kTokensPerSecondEstimate = 3.0;
    constexpr double kGenerationSecondsPadding = 2.0;
    if (token_count <= 0) {
        throw std::runtime_error("PocketTTS token_count must be positive when estimating max_steps");
    }
    const double frame_rate = static_cast<double>(manifest.model_config.mimi_frame_rate);
    const double generation_seconds = static_cast<double>(token_count) / kTokensPerSecondEstimate + kGenerationSecondsPadding;
    return static_cast<int>(std::ceil(generation_seconds * frame_rate));
}

int estimate_default_frames_after_eos(const PocketTTSAssets & manifest, std::string text) {
    if (manifest.model_config.model_recommended_frames_after_eos.has_value()) {
        return *manifest.model_config.model_recommended_frames_after_eos;
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    for (char & ch : text) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    for (size_t i = 1; i < text.size();) {
        if (text[i] == ' ' && text[i - 1] == ' ') {
            text.erase(text.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    if (manifest.model_config.remove_semicolons) {
        std::replace(text.begin(), text.end(), ';', ',');
    }
    int words = 0;
    bool in_word = false;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (in_word) {
                ++words;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if (in_word) {
        ++words;
    }
    const int guess = words <= 4 ? 3 : 1;
    return guess + 2;
}

VoiceConditioningPlan resolve_voice_conditioning_plan(
    const std::filesystem::path & model_dir,
    const GenerationRequest & request) {
    const auto & voice = request.voice;
    const auto has_string = [](const std::string & value) { return !value.empty(); };
    const auto has_path = [](const std::filesystem::path & value) { return !value.empty(); };
    const int provided = (has_string(voice.preset_name) ? 1 : 0)
        + (has_path(voice.embedding_path) ? 1 : 0)
        + (has_path(voice.clone_audio_path) ? 1 : 0)
        + (voice.clone_audio.has_value() ? 1 : 0);
    if (provided > 1) {
        throw std::runtime_error("PocketTTS voice config must choose only one of preset_name, embedding_path, clone_audio_path, or clone_audio");
    }
    if (has_string(voice.clone_prompt_text) && !has_path(voice.clone_audio_path) && !voice.clone_audio.has_value()) {
        throw std::runtime_error("PocketTTS clone_prompt_text requires clone_audio_path or clone_audio");
    }

    VoiceConditioningPlan plan;
    plan.truncate_clone_audio = voice.truncate_clone_audio;
    if (voice.clone_audio.has_value()) {
        plan.source = VoiceSourceKind::CloneAudio;
        plan.clone_audio = voice.clone_audio;
        plan.clone_prompt_text = voice.clone_prompt_text;
        return plan;
    }
    if (has_path(voice.clone_audio_path)) {
        plan.source = VoiceSourceKind::CloneAudio;
        plan.clone_audio_path = voice.clone_audio_path;
        plan.clone_prompt_text = voice.clone_prompt_text;
        return plan;
    }
    if (has_path(voice.embedding_path)) {
        plan.source = VoiceSourceKind::PreparedEmbedding;
        plan.asset_path = voice.embedding_path;
        return plan;
    }
    if (has_string(voice.preset_name)) {
        plan.source = VoiceSourceKind::NamedPreset;
        plan.preset_name = voice.preset_name;
        plan.asset_path = preset_embedding_path(model_dir, voice.preset_name);
        return plan;
    }
    throw std::runtime_error("PocketTTS voice config requires preset_name, embedding_path, clone_audio_path, or clone_audio");
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

std::string voice_state_cache_key(const VoiceConditioningPlan & plan) {
    switch (plan.source) {
        case VoiceSourceKind::NamedPreset:
            return "preset:" + plan.preset_name;
        case VoiceSourceKind::PreparedEmbedding:
            return "embedding:" + plan.asset_path.string();
        case VoiceSourceKind::CloneAudio:
            if (plan.clone_audio.has_value()) {
                const auto & audio = *plan.clone_audio;
                return "clone-buffer:" + std::to_string(audio.sample_rate)
                    + ":" + std::to_string(audio.channels)
                    + ":" + std::to_string(audio.samples.size())
                    + ":" + std::to_string(hash_audio_samples(audio))
                    + (plan.truncate_clone_audio ? ":truncate" : ":full");
            }
            return "clone:" + plan.clone_audio_path.string() + (plan.truncate_clone_audio ? ":truncate" : ":full");
    }
    throw std::runtime_error("PocketTTS voice state cache key received unknown voice source");
}

std::size_t resolve_voice_state_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"pocket_tts.voice_state_cache_slots", "voice_state_cache_slots"})
        .value_or(kDefaultVoiceStateCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("pocket_tts.voice_state_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("pocket_tts.voice_state_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

std::vector<float> load_noise_schedule_file(
    const std::filesystem::path & path,
    int64_t latent_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error("PocketTTS failed to open noise schedule file: " + path.string());
    }
    const size_t value_count = static_cast<size_t>(input.tellg()) / sizeof(float);
    std::vector<float> values(value_count);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(value_count * sizeof(float)));
    if (values.empty() || values.size() % static_cast<size_t>(latent_size) != 0) {
        throw std::runtime_error("PocketTTS noise schedule file has invalid shape: " + path.string());
    }
    return values;
}

AcousticGenerationConfig resolve_acoustic_generation_config(
    const PocketTTSAssets & manifest,
    const TextConditioningResult & text_state,
    const GenerationRequest & request,
    int64_t latent_size) {
    AcousticGenerationConfig acoustic_config;
    acoustic_config.max_steps =
        request.max_steps > 0 ? request.max_steps : estimate_max_steps(manifest, static_cast<int64_t>(text_state.tokens.size()));
    acoustic_config.frames_after_eos =
        request.frames_after_eos >= 0 ? request.frames_after_eos : estimate_default_frames_after_eos(manifest, text_state.prepared_text);
    acoustic_config.temperature = request.temperature;
    acoustic_config.noise_clamp = request.noise_clamp;
    acoustic_config.eos_threshold = request.eos_threshold;
    acoustic_config.seed = request.seed;
    acoustic_config.noise_schedule = request.noise_schedule;
    if (acoustic_config.noise_schedule.empty() && !request.noise_schedule_path.empty()) {
        acoustic_config.noise_schedule = load_noise_schedule_file(
            request.noise_schedule_path,
            latent_size);
    }
    return acoustic_config;
}

}  // namespace

PocketTTSSession::PocketTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const PocketTTSAssets> manifest,
    std::filesystem::path model_dir)
    : RuntimeSessionBase(options),
      task_(task),
      manifest_(std::move(manifest)),
      model_dir_(std::move(model_dir)),
      graph_capacity_(resolve_graph_capacity_config()),
      weights_(load_pocket_tts_backend_weights(
          *manifest_,
          execution_context().backend(),
          execution_context().backend_type(),
          graph_capacity_.matmul_weight_storage_type,
          graph_capacity_.conv_weight_storage_type,
          graph_capacity_.flow_weight_context_bytes,
          graph_capacity_.mimi_encoder_weight_context_bytes,
          graph_capacity_.mimi_decoder_weight_context_bytes)),
      text_conditioner_(make_text_config(*manifest_)),
      voice_conditioner_(make_flow_config(*manifest_)),
      acoustic_model_(make_flow_config(*manifest_)),
      audio_decoder_(make_decoder_config(*manifest_)),
      cached_voice_states_(resolve_voice_state_cache_slots(this->options())),
      prompt_capacity_controller_(graph_capacity_.prompt_mode),
      generation_capacity_controller_(graph_capacity_.generation_mode) {
    if (task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("PocketTTS supports TTS and voice cloning");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("PocketTTS only supports offline and streaming mode");
    }
    if (graph_capacity_.prompt_mode == runtime::GraphCapacityMode::Unsupported
        || graph_capacity_.generation_mode == runtime::GraphCapacityMode::Unsupported) {
        throw std::runtime_error("PocketTTS graph capacity mode=unsupported is not implemented");
    }
    if (graph_capacity_.prompt_mode == runtime::GraphCapacityMode::Fixed
        && graph_capacity_.prompt_capacity <= 0) {
        throw std::runtime_error("PocketTTS fixed prompt graph capacity must be positive");
    }
    if (graph_capacity_.generation_mode == runtime::GraphCapacityMode::Fixed
        && graph_capacity_.generation_capacity <= 0) {
        throw std::runtime_error("PocketTTS fixed generation graph capacity must be positive");
    }
}

PocketTTSSession::~PocketTTSSession() {
    acoustic_model_.clear_runtime_cache();
    audio_decoder_.clear_runtime_cache();
}

PocketTTSSession::AcousticCapacitySelection PocketTTSSession::select_acoustic_capacities(
    int64_t prompt_steps,
    int max_steps) const {
    const int64_t prompt_capacity = prompt_capacity_controller_.select_capacity_for_run(
        make_prompt_capacity_adapter(),
        prompt_steps);
    const int64_t generation_capacity = generation_capacity_controller_.select_capacity_for_run(
        make_generation_capacity_adapter(),
        max_steps);
    if (generation_capacity > std::numeric_limits<int>::max()) {
        throw std::runtime_error("PocketTTS generation graph capacity exceeds int range");
    }
    return {prompt_capacity, static_cast<int>(generation_capacity)};
}

FlowLMState PocketTTSSession::resolve_prepared_voice_state(const VoiceConditioningPlan & plan) {
    if (!manifest_) {
        throw std::runtime_error("PocketTTS session is missing model assets");
    }
    const auto & manifest = *manifest_;
    const std::string voice_key = voice_state_cache_key(plan);
    if (const auto * cached = cached_voice_states_.find(voice_key)) {
        engine::debug::trace_log_scalar("pocket_tts.voice_state.cache_hit", 1);
        engine::debug::trace_log_scalar(
            "pocket_tts.voice_state.cache_slots",
            static_cast<int64_t>(cached_voice_states_.capacity()));
        engine::debug::trace_log_scalar(
            "pocket_tts.voice_state.cache_entries",
            static_cast<int64_t>(cached_voice_states_.size()));
        engine::debug::trace_log_scalar("pocket_tts.voice_state.cache_evicted", 0);
        return *cached;
    }
    const bool will_evict =
        cached_voice_states_.capacity() > 0 &&
        cached_voice_states_.size() >= cached_voice_states_.capacity();
    const auto prepared =
        voice_conditioner_.prepare(
            plan,
            manifest,
            *weights_,
            execution_context().backend(),
            options().backend.threads,
            graph_capacity_.mimi_encoder_graph_context_bytes,
            graph_capacity_.flow_weights_view_context_bytes,
            graph_capacity_.flow_step_graph_context_bytes);
    FlowLMState acoustic_state = prepared.acoustic_state;
    cached_voice_states_.put(voice_key, std::move(acoustic_state));
    engine::debug::trace_log_scalar("pocket_tts.voice_state.cache_hit", 0);
    engine::debug::trace_log_scalar(
        "pocket_tts.voice_state.cache_slots",
        static_cast<int64_t>(cached_voice_states_.capacity()));
    engine::debug::trace_log_scalar(
        "pocket_tts.voice_state.cache_entries",
        static_cast<int64_t>(cached_voice_states_.size()));
    engine::debug::trace_log_scalar("pocket_tts.voice_state.cache_evicted", will_evict ? 1 : 0);
    return prepared.acoustic_state;
}

PocketTTSGraphCapacityConfig PocketTTSSession::resolve_graph_capacity_config() const {
    const runtime::GraphCapacityMode default_mode = core::requested_backend_uses_host_graph_plan(options().backend)
        ? runtime::GraphCapacityMode::Tiered
        : runtime::GraphCapacityMode::Double;
    const auto capacity_options = runtime::apply_option_v1_compatibility(
        options(), {{"offline_graph_capacity_mode", "pocket_tts.offline_graph_capacity_mode"}}, "PocketTTS");
    const auto mode = runtime::resolve_graph_capacity_mode(
        capacity_options,
        default_mode,
        {"pocket_tts.offline_graph_capacity_mode", "graph_capacity_mode"});
    PocketTTSGraphCapacityConfig config;
    config.prompt_mode = mode;
    config.generation_mode = mode;
    config.prompt_capacity = runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.prompt_graph_capacity"},
        0);
    const int64_t generation_capacity = runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.generation_graph_capacity"},
        0);
    if (generation_capacity > std::numeric_limits<int>::max()) {
        throw std::runtime_error("pocket_tts.generation_graph_capacity exceeds int range");
    }
    config.generation_capacity = static_cast<int>(generation_capacity);

    const int64_t weight_context_mb = runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.weight_context_mb"},
        64);
    config.flow_weight_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.flow_weight_context_mb"},
        weight_context_mb)) * 1024ull * 1024ull;
    config.mimi_encoder_weight_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_encoder_weight_context_mb"},
        weight_context_mb)) * 1024ull * 1024ull;
    config.mimi_decoder_weight_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_decoder_weight_context_mb"},
        weight_context_mb)) * 1024ull * 1024ull;
    config.flow_weights_view_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.flow_weights_view_context_mb"},
        256)) * 1024ull * 1024ull;
    config.flow_step_graph_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.flow_step_graph_arena_mb", "pocket_tts.flow_step_graph_context_mb"},
        256)) * 1024ull * 1024ull;
    config.mimi_encoder_graph_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_encoder_graph_arena_mb", "pocket_tts.mimi_encoder_graph_context_mb"},
        512)) * 1024ull * 1024ull;
    config.mimi_conv_graph_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_conv_graph_arena_mb", "pocket_tts.mimi_conv_graph_context_mb"},
        32)) * 1024ull * 1024ull;
    config.mimi_transformer_graph_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_transformer_graph_arena_mb", "pocket_tts.mimi_transformer_graph_context_mb"},
        96)) * 1024ull * 1024ull;
    config.mimi_tail_graph_context_bytes = static_cast<size_t>(runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_tail_graph_arena_mb", "pocket_tts.mimi_tail_graph_context_mb"},
        512)) * 1024ull * 1024ull;
    config.mimi_full_chunk_frames = runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_full_chunk_frames"},
        config.mimi_full_chunk_frames);
    config.mimi_stage2_chunk_frames = runtime::parse_positive_i64_option(
        options().options,
        {"pocket_tts.mimi_stage2_chunk_frames"},
        config.mimi_stage2_chunk_frames);
    if (const auto full_mimi = runtime::find_option(options().options, {"pocket_tts.use_full_mimi"})) {
        config.mimi_use_full_sequence_path = runtime::parse_bool_option(*full_mimi, "pocket_tts.use_full_mimi");
    }

    if (const auto weight_type = runtime::find_option(options().options, {"pocket_tts.weight_type"})) {
        config.matmul_weight_storage_type = assets::parse_tensor_storage_type(*weight_type);
        validate_matmul_weight_storage(config.matmul_weight_storage_type, "pocket_tts.weight_type");
    }
    if (const auto weight_type = runtime::find_option(options().options, {"pocket_tts.matmul_weight_type"})) {
        config.matmul_weight_storage_type = assets::parse_tensor_storage_type(*weight_type);
        validate_matmul_weight_storage(config.matmul_weight_storage_type, "pocket_tts.matmul_weight_type");
    }
    if (const auto weight_type = runtime::find_option(options().options, {"pocket_tts.conv_weight_type"})) {
        config.conv_weight_storage_type = assets::parse_tensor_storage_type(*weight_type);
        validate_conv_weight_storage(config.conv_weight_storage_type, "pocket_tts.conv_weight_type");
    }
    return config;
}

FlowLMState PocketTTSSession::prepare_voice_state(const VoiceConfig & voice) {
    GenerationRequest request;
    request.voice = voice;
    const auto plan = resolve_voice_conditioning_plan(model_dir_, request);
    return resolve_prepared_voice_state(plan);
}

void PocketTTSSession::export_voice_state(const VoiceConfig & voice, const std::filesystem::path & destination) {
    if (!manifest_) {
        throw std::runtime_error("PocketTTS session is missing model assets");
    }
    const auto state = prepare_voice_state(voice);
    const auto & config = manifest_->model_config;
    save_voice_state_assets(
        destination,
        state,
        config.flow_heads,
        config.flow_dim / config.flow_heads);
}

runtime::MappedGraphCapacityAdapter PocketTTSSession::make_prompt_capacity_adapter() const {
    const int64_t configured_capacity = graph_capacity_.prompt_capacity > 0
        ? graph_capacity_.prompt_capacity
        : 1;
    const bool host_graph_plan = execution_context().uses_host_graph_plan();
    return runtime::MappedGraphCapacityAdapter(
        configured_capacity,
        configured_capacity,
        [configured_capacity, host_graph_plan](int64_t request_size) {
            if (request_size <= 0) {
                throw std::runtime_error("PocketTTS prompt graph capacity request size must be positive");
            }
            if (!host_graph_plan) {
                return std::max(configured_capacity, request_size);
            }
            const int64_t target = std::max(configured_capacity, std::max(kCpuPromptCapacityFloor, request_size));
            return next_power_of_two(target);
        },
        [this]() { return prepared_prompt_capacities(); },
        [](int64_t) {
            throw std::runtime_error("PocketTTS prompt graph capacity cannot be prepared independently");
        });
}

runtime::MappedGraphCapacityAdapter PocketTTSSession::make_generation_capacity_adapter() const {
    const int64_t configured_capacity = graph_capacity_.generation_capacity > 0
        ? static_cast<int64_t>(graph_capacity_.generation_capacity)
        : 1;
    const bool host_graph_plan = execution_context().uses_host_graph_plan();
    return runtime::MappedGraphCapacityAdapter(
        configured_capacity,
        configured_capacity,
        [configured_capacity, host_graph_plan](int64_t request_size) {
            if (request_size <= 0) {
                throw std::runtime_error("PocketTTS generation graph capacity request size must be positive");
            }
            if (!host_graph_plan) {
                return std::max(configured_capacity, request_size);
            }
            const int64_t padded_request = request_size + (request_size / 2);
            const int64_t target = std::max(configured_capacity, std::max(kCpuGenerationCapacityFloor, padded_request));
            return round_up_to_multiple(target, kGenerationCapacityQuantum);
        },
        [this]() { return prepared_generation_capacities(); },
        [](int64_t) {
            throw std::runtime_error("PocketTTS generation graph capacity cannot be prepared independently");
        });
}

std::vector<int64_t> PocketTTSSession::prepared_prompt_capacities() const {
    const int64_t capacity = acoustic_model_.prepared_prompt_capacity();
    if (capacity <= 0) {
        return {};
    }
    return {capacity};
}

std::vector<int64_t> PocketTTSSession::prepared_generation_capacities() const {
    const int capacity = acoustic_model_.prepared_max_steps_capacity();
    if (capacity <= 0) {
        return {};
    }
    return {capacity};
}

std::string PocketTTSSession::family() const {
    return "pocket_tts";
}

runtime::VoiceTaskKind PocketTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode PocketTTSSession::run_mode() const {
    return task_.mode;
}

void PocketTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    prepared_session_request_ =
        build_preparation_generation_request(request, manifest_->model_config.default_temperature);
    if (!has_voice_selection(prepared_session_request_.voice)) {
        throw std::runtime_error(
            "PocketTTS session prepare() requires a session voice via --voice-id or --voice-ref");
    }
    if (const auto export_path = runtime::find_option(
            request.options,
            {"export_voice_state_file", "pocket_tts.export_voice_state_file", "pocket_tts.export_voice_state_path", "export_voice_state_path"})) {
        export_voice_state(prepared_session_request_.voice, *export_path);
    }
    prepare_generation(prepared_session_request_);
    mark_prepared();
}

runtime::TaskResult PocketTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("PocketTTS run()");
    audio_decoder_.clear_runtime_cache();
    const GenerationRequest generation_request = effective_request_for_run(request);
    const GenerationResult generated = generate(generation_request);
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        generated.sample_rate,
        1,
        generated.audio,
    };
    return result;
}

runtime::StreamingPolicy PocketTTSSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void PocketTTSSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("PocketTTS streaming");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("PocketTTS start_stream requires a streaming session");
    }
    reset();
    stream_request_ = effective_request_for_run(request);
    validate_generation_request(stream_request_);
    stream_text_chunks_ = split_generation_text(text_conditioner_, *manifest_, stream_request_);
    if (stream_text_chunks_.empty()) {
        throw std::runtime_error("PocketTTS streaming text chunking produced no segments");
    }
    const auto voice_plan = resolve_voice_conditioning_plan(model_dir_, stream_request_);
    stream_voice_state_ = resolve_prepared_voice_state(voice_plan);
    stream_merged_audio_ = runtime::AudioBuffer{manifest_->model_config.sample_rate, 1, {}};
    audio_decoder_.reset_streaming_state();
    stream_started_at_ = std::chrono::steady_clock::now();
    stream_started_ = true;
    trace_text_chunking("pocket_tts.streaming.", stream_request_, stream_text_chunks_.size());
}

std::optional<runtime::StreamEvent> PocketTTSSession::next_stream_event() {
    if (!stream_started_) {
        throw std::runtime_error("PocketTTS streaming has not been started");
    }
    while (true) {
        if (!stream_acoustic_state_.has_value()) {
            if (!start_next_stream_text_chunk()) {
                return std::nullopt;
            }
        }
        auto acoustic_step = acoustic_model_.next_stream_step(*stream_acoustic_state_);
        if (!acoustic_step.has_value()) {
            stream_acoustic_state_.reset();
            continue;
        }
        auto audio = audio_decoder_.decode_streaming_step(
            execution_context().backend(),
            options().backend.threads,
            *manifest_,
            *weights_,
            acoustic_step->next_latent,
            graph_capacity_.mimi_conv_graph_context_bytes,
            graph_capacity_.mimi_transformer_graph_context_bytes,
            graph_capacity_.mimi_tail_graph_context_bytes);
        runtime::AudioBuffer chunk{
            manifest_->model_config.sample_rate,
            1,
            std::move(audio),
        };
        runtime::append_audio_buffer(stream_merged_audio_, chunk);

        runtime::StreamEvent event;
        event.named_audio_outputs.push_back({
            "chunk_" + std::to_string(stream_audio_chunk_index_++),
            std::move(chunk),
            {},
        });
        return event;
    }
}

void PocketTTSSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

runtime::TaskResult PocketTTSSession::finish_stream() {
    if (!stream_started_) {
        throw std::runtime_error("PocketTTS streaming has not been started");
    }
    while (next_stream_event().has_value()) {
    }
    runtime::TaskResult result;
    result.audio_output = std::move(stream_merged_audio_);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(stream_started_at_));
    reset();
    return result;
}

void PocketTTSSession::reset() {
    stream_request_ = {};
    stream_voice_state_ = {};
    stream_text_chunks_.clear();
    stream_text_chunk_index_ = 0;
    stream_acoustic_state_.reset();
    audio_decoder_.reset_streaming_state();
    stream_merged_audio_ = {};
    stream_audio_chunk_index_ = 0;
    stream_started_ = false;
}

runtime::StreamEvent PocketTTSSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void) chunk;
    throw std::runtime_error("PocketTTS streaming does not accept audio chunks");
}

runtime::TaskResult PocketTTSSession::finalize() {
    return finish_stream();
}

GenerationRequest PocketTTSSession::effective_request_for_run(const runtime::TaskRequest & request) const {
    runtime::TaskRequest effective_request = request;
    effective_request.voice.reset();
    GenerationRequest generation_request =
        build_generation_request(effective_request, manifest_->model_config.default_temperature);
    generation_request.max_steps = prepared_session_request_.max_steps;
    generation_request.max_tokens = prepared_session_request_.max_tokens;
    generation_request.temperature = prepared_session_request_.temperature;
    generation_request.noise_clamp = prepared_session_request_.noise_clamp;
    generation_request.eos_threshold = prepared_session_request_.eos_threshold;
    generation_request.seed = prepared_session_request_.seed;
    generation_request.noise_schedule = prepared_session_request_.noise_schedule;
    generation_request.noise_schedule_path = prepared_session_request_.noise_schedule_path;
    generation_request.voice = prepared_session_request_.voice;
    return generation_request;
}

bool PocketTTSSession::start_next_stream_text_chunk() {
    if (stream_text_chunk_index_ >= stream_text_chunks_.size()) {
        return false;
    }
    const auto & chunk = stream_text_chunks_[stream_text_chunk_index_++];
    const TextConditioningResult text_state = text_conditioner_.prepare(*manifest_, weights_->host, chunk);
    const AcousticGenerationConfig acoustic_config = resolve_acoustic_generation_config(
        *manifest_,
        text_state,
        stream_request_,
        acoustic_model_.config().latent_size);
    const int64_t prompt_steps = static_cast<int64_t>(
        text_state.text_embeddings.size() / static_cast<size_t>(acoustic_model_.config().hidden_size));
    const AcousticCapacitySelection capacities = select_acoustic_capacities(prompt_steps, acoustic_config.max_steps);
    AcousticPreparedRuntime acoustic_runtime = acoustic_model_.prepare_runtime(
        execution_context().backend(),
        options().backend.threads,
        *manifest_,
        *weights_,
        text_state.text_embeddings,
        stream_voice_state_,
        acoustic_config,
        capacities.prompt_capacity,
        stream_voice_state_.current_end,
        capacities.generation_capacity,
        graph_capacity_.flow_weights_view_context_bytes,
        graph_capacity_.flow_step_graph_context_bytes);
    stream_acoustic_state_ = acoustic_model_.start_stream(
        acoustic_runtime,
        *manifest_,
        *weights_,
        text_state.text_embeddings,
        stream_voice_state_,
        acoustic_config);
    return true;
}

void PocketTTSSession::prepare_generation(const GenerationRequest & request) {
    if (!manifest_) {
        throw std::runtime_error("PocketTTS session is missing model assets");
    }
    const auto voice_plan = resolve_voice_conditioning_plan(model_dir_, request);
    const FlowLMState voice_state = resolve_prepared_voice_state(voice_plan);
    if (request.text.empty()) {
        return;
    }
    const auto & manifest = *manifest_;
    const auto chunks = split_generation_text(text_conditioner_, manifest, request);
    for (const auto & chunk : chunks) {
        const TextConditioningResult text_state = text_conditioner_.prepare(manifest, weights_->host, chunk);
        const AcousticGenerationConfig acoustic_config = resolve_acoustic_generation_config(
            manifest,
            text_state,
            request,
            acoustic_model_.config().latent_size);
        const int64_t prompt_steps = static_cast<int64_t>(
            text_state.text_embeddings.size() / static_cast<size_t>(acoustic_model_.config().hidden_size));
        const AcousticCapacitySelection capacities = select_acoustic_capacities(prompt_steps, acoustic_config.max_steps);
        (void) acoustic_model_.prepare_runtime(
            execution_context().backend(),
            options().backend.threads,
            manifest,
            *weights_,
            text_state.text_embeddings,
            voice_state,
            acoustic_config,
            capacities.prompt_capacity,
            voice_state.current_end,
            capacities.generation_capacity,
            graph_capacity_.flow_weights_view_context_bytes,
            graph_capacity_.flow_step_graph_context_bytes);
    }
}

GenerationResult PocketTTSSession::generate(const GenerationRequest & request) {
    validate_generation_request(request);
    if (!manifest_) {
        throw std::runtime_error("PocketTTS session is missing model assets");
    }
    const auto & manifest = *manifest_;

    const auto started_inference = std::chrono::steady_clock::now();
    const auto chunks = split_generation_text(text_conditioner_, manifest, request);
    trace_text_chunking("pocket_tts.", request, chunks.size());
    const auto voice_plan = resolve_voice_conditioning_plan(model_dir_, request);

    VoiceConditioningResult voice_state;
    const double voice_conditioner_ms = engine::debug::measure_ms([&]() {
        voice_state.plan = voice_plan;
        voice_state.acoustic_state = resolve_prepared_voice_state(voice_state.plan);
    });

    std::vector<float> audio;
    double text_conditioner_ms = 0.0;
    double acoustic_prepare_ms = 0.0;
    double acoustic_generate_ms = 0.0;
    double audio_decode_ms = 0.0;

    for (const auto & chunk : chunks) {
        TextConditioningResult text_state;
        text_conditioner_ms += engine::debug::measure_ms([&]() {
            text_state = text_conditioner_.prepare(manifest, weights_->host, chunk);
        });
        const AcousticGenerationConfig acoustic_config = resolve_acoustic_generation_config(
            manifest,
            text_state,
            request,
            acoustic_model_.config().latent_size);
        const int64_t prompt_steps = static_cast<int64_t>(
            text_state.text_embeddings.size() / static_cast<size_t>(acoustic_model_.config().hidden_size));
        const AcousticCapacitySelection capacities = select_acoustic_capacities(prompt_steps, acoustic_config.max_steps);

        AcousticPreparedRuntime acoustic_runtime;
        acoustic_prepare_ms += engine::debug::measure_ms([&]() {
            acoustic_runtime = acoustic_model_.prepare_runtime(
                execution_context().backend(),
                options().backend.threads,
                manifest,
                *weights_,
                text_state.text_embeddings,
                voice_state.acoustic_state,
                acoustic_config,
                capacities.prompt_capacity,
                voice_state.acoustic_state.current_end,
                capacities.generation_capacity,
                graph_capacity_.flow_weights_view_context_bytes,
                graph_capacity_.flow_step_graph_context_bytes);
        });
        AcousticModelResult acoustic;
        acoustic_generate_ms += engine::debug::measure_ms([&]() {
            acoustic = acoustic_model_.generate(
                acoustic_runtime,
                manifest,
                *weights_,
                text_state.text_embeddings,
                voice_state.acoustic_state,
                acoustic_config);
        });
        std::vector<float> chunk_audio;
        audio_decode_ms += engine::debug::measure_ms([&]() {
            chunk_audio = audio_decoder_.decode(
                execution_context().backend(),
                options().backend.threads,
                manifest,
                *weights_,
                acoustic.latents,
                acoustic.generated_steps,
                graph_capacity_.mimi_conv_graph_context_bytes,
                graph_capacity_.mimi_transformer_graph_context_bytes,
                graph_capacity_.mimi_tail_graph_context_bytes,
                graph_capacity_.mimi_full_chunk_frames,
                graph_capacity_.mimi_stage2_chunk_frames,
                graph_capacity_.mimi_use_full_sequence_path);
        });
        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
    }

    const auto ended_inference = std::chrono::steady_clock::now();
    const double inference_ms = engine::debug::elapsed_ms(started_inference, ended_inference);
    GenerationResult result;
    result.sample_rate = manifest.model_config.sample_rate;
    result.audio = audio;
    engine::debug::timing_log_scalar("pocket_tts.text_conditioner_ms", text_conditioner_ms);
    engine::debug::timing_log_scalar("pocket_tts.voice_conditioner_ms", voice_conditioner_ms);
    engine::debug::timing_log_scalar("pocket_tts.acoustic_prepare_ms", acoustic_prepare_ms);
    engine::debug::timing_log_scalar("pocket_tts.acoustic_generate_ms", acoustic_generate_ms);
    engine::debug::timing_log_scalar("pocket_tts.audio_decode_ms", audio_decode_ms);
    engine::debug::timing_log_scalar("session.wall_ms", inference_ms);
    return result;
}

}  // namespace engine::models::pocket_tts

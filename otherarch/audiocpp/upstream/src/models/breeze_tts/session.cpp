#include "engine/models/breeze_tts/session.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/breeze_tts/generator.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::breeze_tts {
namespace {

constexpr const char * kFamily = "breeze_tts";
constexpr const char * kModelName = "BreezeTTS";
constexpr int64_t kDefaultTextChunkSize = 600;
constexpr int64_t kDefaultReferenceCacheSlots = 1;

std::shared_ptr<const BreezeTTSAssets> require_assets(std::shared_ptr<const BreezeTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("BreezeTTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("BreezeTTS session requires a model contract");
    }
    return contract;
}

std::vector<runtime::TaskRequest> split_request(const runtime::TaskRequest & request) {
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    if (text_chunk_size <= 0) {
        throw std::runtime_error("BreezeTTS text_chunk_size must be positive");
    }
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    return runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
}

core::AttentionPreference attention_preference_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"breeze_tts.attention"})) {
        return core::parse_attention_preference(*value, "breeze_tts.attention");
    }
    return core::AttentionPreference::Auto;
}

Bf16ActivationMode bf16_activation_mode_from_options(const runtime::SessionOptions & options) {
    const auto value = runtime::find_option(options.options, {"bf16_activations", "breeze_tts.bf16_activations"});
    if (!value.has_value()) {
        return Bf16ActivationMode::Auto;
    }
    if (*value == "auto") {
        return Bf16ActivationMode::Auto;
    }
    if (*value == "on") {
        return Bf16ActivationMode::On;
    }
    if (*value == "off") {
        return Bf16ActivationMode::Off;
    }
    throw std::runtime_error("BreezeTTS bf16_activations must be auto, on or off (got '" + *value + "')");
}

void trace_attention_preference(core::AttentionPreference preference) {
    const char * name = "auto";
    if (preference == core::AttentionPreference::Flash) {
        name = "flash";
    } else if (preference == core::AttentionPreference::Eager) {
        name = "eager";
    }
    engine::debug::trace_log_scalar("breeze_tts.attention.preference", std::string_view(name));
}

void validate_session_options(
    const runtime::SessionOptions & options,
    const engine::model_spec::ModelContract & contract) {
    auto validation_options = options;
    // Older standalone GGUF packages embed a v1 contract that predates this
    // backend-compatibility option; keep them usable while still validating
    // the option value in attention_preference_from_options().
    // bf16_activations is handled the same way.
    for (const char * key : {"breeze_tts.attention", "bf16_activations", "breeze_tts.bf16_activations"}) {
        if (contract.session_option_keys.find(key) == contract.session_option_keys.end()) {
            validation_options.options.erase(key);
        }
    }
    runtime::validate_spec_backed_session_options(validation_options, contract, kFamily, kModelName);
}

void validate_request_options(
    const std::unordered_map<std::string, std::string> & options,
    const engine::model_spec::ModelContract & contract) {
    auto validation_options = options;
    validation_options.erase("instruct");
    validation_options.erase("instruction");
    (void)runtime::find_option(options, {"instruct", "instruction"});
    for (const char * key : {"stream_frames_per_event", "stream_lookahead_margin", "stream_chunk_policy"}) {
        if (contract.request_option_keys.find(key) == contract.request_option_keys.end()) {
            validation_options.erase(key);
        }
    }
    runtime::validate_spec_backed_request_options(validation_options, contract, kModelName);
}

std::size_t reference_cache_slots_from_options(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"breeze_tts.reference_cache_slots", "reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("breeze_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<uint64_t>(slots) > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("breeze_tts.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
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

std::unique_ptr<runtime::IVoiceTaskSession> create_breeze_tts_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const BreezeTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<BreezeTTSSession>(task, options, std::move(assets), std::move(contract));
}

}  // namespace

BreezeTTSSession::BreezeTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const BreezeTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      reference_cache_(reference_cache_slots_from_options(options)) {
    validate_session_options(options, *contract_);
    if (task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceCloning &&
        task_.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("BreezeTTS supports tts, clone, and voice design tasks");
    }
    if (task_.mode != runtime::RunMode::Offline &&
        task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("BreezeTTS supports offline and streaming sessions");
    }
    using T = engine::assets::TensorStorageType;
    const auto storage_option = runtime::find_option_match(
        options.options, {"breeze_tts.weight_type", "weight_type"});
    const auto storage_type = runtime::parse_tensor_storage_option(
        options.options,
        storage_option ? storage_option->key : "breeze_tts.weight_type",
        T::Native,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0, T::Q4_0, T::Q4_K});
    const auto graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"breeze_tts.graph_arena_mb", "graph_arena_mb"},
        32ull * 1024ull * 1024ull);
    const auto weight_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"breeze_tts.weight_context_mb", "weight_context_mb"},
        4ull * 1024ull * 1024ull);
    const auto attention_preference = attention_preference_from_options(options);
    trace_attention_preference(attention_preference);
    const auto bf16_activations = bf16_activation_mode_from_options(options);
    generator_ = std::make_unique<BreezeGeneratorRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type,
        attention_preference,
        bf16_activations);
}

BreezeTTSSession::~BreezeTTSSession() = default;

std::string BreezeTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind BreezeTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode BreezeTTSSession::run_mode() const {
    return task_.mode;
}

void BreezeTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    validate_request_options(request.options, *contract_);
    mark_prepared();
}

BreezeSpeechCodes BreezeTTSSession::resolve_reference_codes(const runtime::AudioBuffer & audio) {
    ReferenceCacheKey key;
    key.sample_rate = audio.sample_rate;
    key.channels = audio.channels;
    key.sample_count = static_cast<uint64_t>(audio.samples.size());
    key.sample_hash = hash_audio_samples(audio);
    if (const auto * cached = reference_cache_.find(key)) {
        engine::debug::trace_log_scalar("breeze_tts.reference_cache.hit", 1);
        engine::debug::trace_log_scalar("breeze_tts.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
        engine::debug::trace_log_scalar("breeze_tts.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
        return cached->codes;
    }
    const auto start = std::chrono::steady_clock::now();
    ReferenceCacheEntry entry;
    entry.codes = generator_->encode_reference(audio);
    engine::debug::trace_log_scalar("breeze_tts.reference.frames", entry.codes.frames);
    engine::debug::trace_log_scalar("breeze_tts.reference.codebooks", entry.codes.code_groups);
    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(entry);
    } else {
        reference_cache_.put(key, std::move(entry));
    }
    engine::debug::trace_log_scalar("breeze_tts.reference_cache.hit", 0);
    engine::debug::trace_log_scalar("breeze_tts.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
    engine::debug::trace_log_scalar("breeze_tts.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
    engine::debug::timing_log_scalar("breeze_tts.reference_encode_ms", engine::debug::elapsed_ms(start));
    if (reference_cache_.capacity() == 0) {
        return uncached_reference_->codes;
    }
    const auto * cached = reference_cache_.find(key);
    if (cached == nullptr) {
        throw std::runtime_error("BreezeTTS reference cache insert failed");
    }
    return cached->codes;
}

runtime::TaskResult BreezeTTSSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    validate_request_options(request.options, *contract_);
    require_prepared("BreezeTTS run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("BreezeTTS run requires an offline session");
    }
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("BreezeTTS requires text input");
    }
    runtime::AudioBuffer merged;
    auto chunks = split_request(request);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    engine::debug::trace_log_scalar("breeze_tts.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("breeze_tts.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("breeze_tts.text.chunk_count", static_cast<int64_t>(chunks.size()));
    std::optional<BreezeSpeechCodes> reference_codes;
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        reference_codes = resolve_reference_codes(*request.voice->speaker->audio);
    }
    const bool anchor_voice = chunks.size() > 1 && !reference_codes.has_value();
    for (size_t index = 0; index < chunks.size(); ++index) {
        auto generation = build_generation_request(chunks[index], reference_codes, index);
        if (anchor_voice && index > 0) {
            generation.reference_text = chunks.front().text_input->text;
        }
        BreezeSpeechCodes first_codes;
        runtime::append_audio_buffer(merged, generator_->generate(
            generation, anchor_voice && index == 0 ? &first_codes : nullptr));
        if (anchor_voice && index == 0) {
            reference_codes = std::move(first_codes);
        }
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::TaskResult BreezeTTSSession::generate_stream(const runtime::TaskRequest & request) {
    validate_request_options(request.options, *contract_);
    require_prepared("BreezeTTS streaming");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("BreezeTTS streaming requires text input");
    }
    const auto started_at = std::chrono::steady_clock::now();
    const auto chunks = split_request(request);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    engine::debug::trace_log_scalar("breeze_tts.streaming.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("breeze_tts.streaming.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("breeze_tts.streaming.text.chunk_count", static_cast<int64_t>(chunks.size()));
    std::optional<BreezeSpeechCodes> reference_codes;
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        reference_codes = resolve_reference_codes(*request.voice->speaker->audio);
    }
    const auto frames_per_event = runtime::parse_i64_option(request.options, {"stream_frames_per_event"})
        .value_or(16);
    if (frames_per_event <= 0) {
        throw std::runtime_error("BreezeTTS stream_frames_per_event must be positive");
    }
    const auto lookahead = runtime::parse_i64_option(request.options, {"stream_lookahead_margin"}).value_or(0);
    if (lookahead < 0) {
        throw std::runtime_error("BreezeTTS stream_lookahead_margin must be non-negative");
    }
    runtime::StreamingAudioConfig config;
    config.frames_per_chunk = static_cast<size_t>(frames_per_event);
    config.right_context_frames = static_cast<size_t>(lookahead);
    config.policy = runtime::parse_streaming_audio_chunk_policy(
        runtime::find_option(request.options, {"stream_chunk_policy"}).value_or("grow"));
    engine::debug::trace_log_scalar(
        "breeze_tts.streaming.frames_per_event", frames_per_event);
    engine::debug::trace_log_scalar("breeze_tts.streaming.lookahead_margin", lookahead);
    runtime::AudioBuffer merged{24000, 1, {}};
    const bool anchor_voice = chunks.size() > 1 && !reference_codes.has_value();
    for (size_t index = 0; index < chunks.size(); ++index) {
        auto generation = build_generation_request(chunks[index], reference_codes, index);
        if (anchor_voice && index > 0) {
            generation.reference_text = chunks.front().text_input->text;
        }
        BreezeSpeechCodes first_codes;
        runtime::append_audio_buffer(merged, generator_->generate_stream(
            generation, config, stream_event_sink(), anchor_voice && index == 0 ? &first_codes : nullptr));
        if (anchor_voice && index == 0) {
            reference_codes = std::move(first_codes);
        }
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started_at));
    return result;
}

void BreezeTTSSession::reset_stream_state() {
    generator_->end_stream();
}

BreezeGenerationRequest BreezeTTSSession::build_generation_request(
    const runtime::TaskRequest & request,
    const std::optional<BreezeSpeechCodes> & reference_codes,
    size_t chunk_index) const {
    BreezeGenerationRequest generation;
    generation.text = request.text_input->text;
    generation.instruction = runtime::find_option(request.options, {"instruct", "instruction"}).value_or("");
    generation.reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
    generation.guidance_scale = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(generation.guidance_scale);
    if (generation.guidance_scale < 0.0F) {
        throw std::runtime_error("BreezeTTS guidance_scale must be non-negative");
    }
    generation.temperature = runtime::parse_positive_finite_float_option(request.options, {"temperature"}).value_or(generation.temperature);
    generation.depth_temperature = runtime::parse_positive_finite_float_option(request.options, {"depth_temperature"}).value_or(generation.depth_temperature);
    generation.top_k = runtime::parse_i64_option(request.options, {"top_k"}).value_or(generation.top_k);
    generation.top_p = runtime::parse_positive_finite_float_option(request.options, {"top_p"}).value_or(generation.top_p);
    generation.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, generation.max_tokens);
    generation.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(generation.seed);
    generation.reference_codes = reference_codes;
    if (chunk_index > 0) {
        ++generation.seed;
    }
    return generation;
}

bool BreezeTTSSession::ReferenceCacheKeyEqual::operator()(
    const ReferenceCacheKey & lhs,
    const ReferenceCacheKey & rhs) const noexcept {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_breeze_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<BreezeTTSAssets> config;
    config.family = kFamily;
    config.load_assets = load_breeze_tts_assets;
    config.create_session = create_breeze_tts_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::breeze_tts

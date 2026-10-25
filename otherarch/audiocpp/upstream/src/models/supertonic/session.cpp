#include "engine/models/supertonic/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/supertonic/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::supertonic {
namespace {

std::shared_ptr<const SupertonicAssets> require_assets(std::shared_ptr<const SupertonicAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Supertonic session requires assets");
    }
    return assets;
}

void validate_weight_storage(assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

void validate_session_options(const runtime::SessionOptions & options) {
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("supertonic.", 0) == 0) {
            if (key == "supertonic.weight_type" ||
                key == "supertonic.style_cache_slots") {
                continue;
            }
            throw std::runtime_error("unknown Supertonic session option: " + key);
        }
    }
}

std::size_t resolve_style_cache_slots(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultStyleCacheSlots = 4;
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"supertonic.style_cache_slots", "style_cache_slots"})
        .value_or(kDefaultStyleCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("supertonic.style_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("supertonic.style_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

}  // namespace

SupertonicSession::SupertonicSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SupertonicAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_) {
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Supertonic only supports offline and streaming sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Supertonic only supports the Tts task");
    }
    if (const auto it = options.options.find("supertonic.weight_type"); it != options.options.end()) {
        weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
        validate_weight_storage(weight_storage_type_, "supertonic.weight_type");
    }
    style_cache_slots_ = resolve_style_cache_slots(options);
    validate_session_options(options);
    runtime_ = std::make_unique<SupertonicRuntime>(assets_, options.backend, weight_storage_type_, style_cache_slots_);
}

SupertonicSession::~SupertonicSession() = default;

std::string SupertonicSession::family() const {
    return "supertonic";
}

runtime::VoiceTaskKind SupertonicSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SupertonicSession::run_mode() const {
    return task_.mode;
}

void SupertonicSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

runtime::TaskResult SupertonicSession::run(const runtime::TaskRequest & request) {
    require_prepared("Supertonic run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Supertonic run requires an offline session");
    }
    const auto chunk_requests = build_chunk_requests(request);
    runtime::TaskResult result;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        runtime::append_audio_buffer(merged_audio, synthesize_chunk(chunk_request));
    }
    result.audio_output = std::move(merged_audio);
    return result;
}

runtime::StreamingPolicy SupertonicSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void SupertonicSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Supertonic streaming");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Supertonic start_stream requires a streaming session");
    }
    reset();
    stream_chunk_requests_ = build_chunk_requests(request);
    stream_started_ = true;
}

std::optional<runtime::StreamEvent> SupertonicSession::next_stream_event() {
    if (!stream_started_) {
        throw std::runtime_error("Supertonic streaming has not been started");
    }
    if (stream_chunk_index_ >= stream_chunk_requests_.size()) {
        return std::nullopt;
    }
    const std::size_t chunk_index = stream_chunk_index_++;
    auto chunk_audio = synthesize_chunk(stream_chunk_requests_[chunk_index]);
    runtime::append_audio_buffer(stream_merged_audio_, chunk_audio);
    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({
        "chunk_" + std::to_string(chunk_index),
        std::move(chunk_audio),
        {},
    });
    return event;
}

void SupertonicSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    (void)sink;
}

runtime::TaskResult SupertonicSession::finish_stream() {
    if (!stream_started_) {
        throw std::runtime_error("Supertonic streaming has not been started");
    }
    runtime::TaskResult result;
    result.audio_output = std::move(stream_merged_audio_);
    reset();
    return result;
}

void SupertonicSession::reset() {
    stream_chunk_requests_.clear();
    stream_merged_audio_ = runtime::AudioBuffer{};
    stream_chunk_index_ = 0;
    stream_started_ = false;
}

runtime::StreamEvent SupertonicSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("Supertonic streaming does not consume audio chunks");
}

runtime::TaskResult SupertonicSession::finalize() {
    return finish_stream();
}

SupertonicGenerationOptions SupertonicSession::generation_options_from_request(const runtime::TaskRequest & request) const {
    const auto request_options = runtime::apply_option_v1_compatibility(
        request.options,
        {{"voice", "voice_id"}, {"supertonic.voice", "voice_id"}, {"speaking_rate", "speed"}},
        "Supertonic", "request");
    SupertonicGenerationOptions options;
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        options.language = request.text_input->language;
    }
    if (request.voice.has_value() && request.voice->style.has_value() && request.voice->style->language.has_value()) {
        options.language = *request.voice->style->language;
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->cached_voice_id.has_value()) {
        options.voice = *request.voice->speaker->cached_voice_id;
    } else {
        // WebUI model_params send the picked preset as a plain request option
        // (see webui/configs/model_params.json). Accept it as a fallback so the
        // voice picker takes effect; unknown ids are rejected downstream by the
        // voice-style loader.
        if (const auto preset = runtime::find_option(request_options, {"voice_id"})) {
            if (!preset->empty()) {
                options.voice = *preset;
            }
        }
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        if (*value <= 0) {
            throw std::runtime_error("Supertonic num_inference_steps must be positive");
        }
        options.num_inference_steps = static_cast<int>(*value);
    }
    if (const auto value = runtime::parse_positive_finite_float_option(request_options, {"speed"})) {
        options.speaking_rate = *value;
    }
    if (request.voice.has_value() && request.voice->style.has_value() &&
        request.voice->style->speaking_rate.has_value()) {
        const float rate = *request.voice->style->speaking_rate;
        if (!std::isfinite(rate) || rate <= 0.0f) {
            throw std::runtime_error("Supertonic speaking_rate must be positive and finite");
        }
        options.speaking_rate = rate;
    }
    if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
        options.seed = *value;
    }
    return options;
}

void SupertonicSession::validate_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Supertonic requires --text input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Supertonic does not accept audio input");
    }
}

std::vector<runtime::TaskRequest> SupertonicSession::build_chunk_requests(const runtime::TaskRequest & request) const {
    validate_request(request);
    const auto options = generation_options_from_request(request);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options)
            .value_or((options.language == "ko" || options.language == "ja") ? 120 : 300);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options)
            .value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
    engine::debug::trace_log_scalar("supertonic.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("supertonic.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("supertonic.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    return chunk_requests;
}

runtime::AudioBuffer SupertonicSession::synthesize_chunk(const runtime::TaskRequest & request) {
    const auto options = generation_options_from_request(request);
    return runtime_->synthesize(request.text_input->text, options, tokenizer_);
}

}  // namespace engine::models::supertonic

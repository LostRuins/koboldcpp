#include "engine/community_models/chatterbox_turbo/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {

namespace {

// Chatterbox Turbo has no CFG-based prefix cache to amortize across chunks (unlike base
// Chatterbox), but chunking still matters here: the S3Gen flow encoder's attention buffer scales
// with token count, and without a bound a long paragraph can exceed VRAM on small GPUs even
// though the model weights themselves fit comfortably. Matches base Chatterbox's default
// (src/models/chatterbox/session.cpp).
constexpr int64_t kDefaultTextChunkSize = 128;

ChatterboxTurboGenerateConfig make_generate_config(const std::unordered_map<std::string, std::string> & options) {
    ChatterboxTurboGenerateConfig config;
    if (const auto value = runtime::parse_float_option(options, {"temperature"})) {
        config.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(options, {"top_p"})) {
        config.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(options, {"repetition_penalty"})) {
        config.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_u32_option(options, {"top_k"})) {
        config.top_k = *value;
    }
    if (const auto value = runtime::parse_u32_option(options, {"max_tokens", "max_new_tokens"})) {
        config.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_u32_option(options, {"seed"})) {
        config.seed = *value;
    }
    // exaggeration/cfg_weight/min_p are accepted elsewhere in the request surface but have no
    // effect on Turbo, matching upstream tts_turbo.py's logger.warning(...ignored...).
    return config;
}

}  // namespace

ChatterboxTurboSession::ChatterboxTurboSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ChatterboxTurboAssets> assets)
    : RuntimeSessionBase(options), task_(std::move(task)), assets_(std::move(assets)) {
    if (!assets_) {
        throw std::runtime_error("Chatterbox Turbo session requires assets");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Chatterbox Turbo session supports --task tts only");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Chatterbox Turbo session only supports offline mode");
    }
}

ChatterboxTurboSession::~ChatterboxTurboSession() = default;

std::string ChatterboxTurboSession::family() const {
    return "chatterbox_turbo";
}

runtime::VoiceTaskKind ChatterboxTurboSession::task_kind() const {
    return task_.task;
}

runtime::RunMode ChatterboxTurboSession::run_mode() const {
    return task_.mode;
}

void ChatterboxTurboSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.text.has_value() || request.text->text.empty()) {
        throw std::runtime_error("Chatterbox Turbo prepare requires text input");
    }
    if (request.voice.has_value() && request.voice->speaker.has_value() && request.voice->speaker->audio.has_value()) {
        throw std::runtime_error(
            "Chatterbox Turbo does not support custom voice cloning (only the built-in default voice) "
            "-- omit the speaker reference audio to use the built-in voice");
    }
    if (!component_) {
        component_ = std::make_unique<ChatterboxTurboTTSComponent>(assets_, execution_context());
    }
    mark_prepared();
}

runtime::TaskResult ChatterboxTurboSession::run(const runtime::TaskRequest & request) {
    require_prepared("Chatterbox Turbo run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Chatterbox Turbo run requires text input");
    }
    const auto config = make_generate_config(request.options);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);

    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const auto outputs = component_->generate(chunk_request.text_input->text, config);
        runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{24000, 1, outputs.waveform});
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    return result;
}

}  // namespace engine::community_models::chatterbox_turbo

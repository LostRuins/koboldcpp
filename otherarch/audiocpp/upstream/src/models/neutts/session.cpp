#include "engine/models/neutts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::neutts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "neutts";
constexpr int64_t kDefaultTextChunkSize = 600;
constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 1024ull * 1024ull * 1024ull;

uint64_t fnv1a_i32(const std::vector<int32_t> & values) {
    uint64_t hash = 1469598103934665603ull;
    for (const int32_t value : values) {
        uint32_t bytes = static_cast<uint32_t>(value);
        for (int i = 0; i < 4; ++i) {
            hash ^= static_cast<uint8_t>(bytes & 0xffu);
            hash *= 1099511628211ull;
            bytes >>= 8;
        }
    }
    return hash;
}

std::shared_ptr<const NeuTTSAssets> require_assets(
    std::shared_ptr<const NeuTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("NeuTTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("NeuTTS session requires a model contract");
    }
    return contract;
}

std::string request_text(const runtime::TaskRequest & request) {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("NeuTTS requires non-empty text input");
    }
    return request.text_input->text;
}

std::string request_speaker(const runtime::TaskRequest & request) {
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->cached_voice_id.has_value()) {
        return *request.voice->speaker->cached_voice_id;
    }
    return runtime::find_option(request.options, {"voice_id"}).value_or("emily");
}

std::string request_emotion(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"emotion"})) {
        return *value;
    }
    if (request.voice.has_value() && request.voice->style.has_value() &&
        request.voice->style->emotion.has_value()) {
        return *request.voice->style->emotion;
    }
    return "neutral";
}

NeuTTSGenerationOptions request_generation_options(const runtime::TaskRequest & request) {
    NeuTTSGenerationOptions out;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        out.max_tokens = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"min_tokens"})) {
        out.min_tokens = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    out.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());
    if (out.max_tokens < 0) {
        throw std::runtime_error("NeuTTS max_tokens must be non-negative");
    }
    if (out.min_tokens < 0 || (out.max_tokens > 0 && out.min_tokens > out.max_tokens)) {
        throw std::runtime_error("NeuTTS min_tokens must be non-negative and not exceed max_tokens");
    }
    if (out.temperature <= 0.0F) {
        throw std::runtime_error("NeuTTS temperature must be positive");
    }
    if (out.top_k <= 0 || out.top_k > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NeuTTS top_k must be a positive integer");
    }
    return out;
}

std::vector<runtime::TaskRequest> split_neutts_request(const runtime::TaskRequest & request) {
    const auto text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    engine::debug::trace_log_scalar("neutts.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("neutts.text_chunk_size", text_chunk_size);
    return runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
}

runtime::AudioBuffer merge_chunks(const std::vector<runtime::AudioBuffer> & chunks) {
    runtime::AudioBuffer out;
    for (const auto & chunk : chunks) {
        if (chunk.samples.empty()) {
            continue;
        }
        if (out.samples.empty()) {
            out = chunk;
        } else {
            runtime::append_audio_buffer(out, chunk);
        }
    }
    return out;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_neutts_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const NeuTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<NeuTTSSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

NeuTTSSession::NeuTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NeuTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      prompt_builder_(assets_) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "NeuTTS");
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("NeuTTS supports the TTS task");
    }
    const size_t graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"neutts.runtime_graph_arena_mb"},
        kDefaultGraphArenaBytes);
    const auto shared_weight_type = runtime::parse_tensor_storage_option(
        options.options,
        "neutts.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto backbone_default_weight_type =
        runtime::find_option(options.options, {"neutts.weight_type"}).has_value()
            ? shared_weight_type
            : (execution_context().backend_type() == core::BackendType::Cpu
                   ? assets::TensorStorageType::F32
                   : shared_weight_type);
    const auto backbone_weight_type = runtime::parse_tensor_storage_option(
        options.options,
        "neutts.generator_weight_type",
        backbone_default_weight_type,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto codec_weight_type = runtime::parse_tensor_storage_option(
        options.options,
        "neutts.codec_weight_type",
        shared_weight_type,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_weight_type = runtime::parse_tensor_storage_option(
        options.options,
        "neutts.codec_conv_weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16});
    ar_ = std::make_unique<NeuTTSQwen3ARRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        graph_arena_bytes,
        kDefaultWeightContextBytes,
        backbone_weight_type);
    codec_ = std::make_unique<NeuTTSNeuCodecDecoderRuntime>(
        make_neutts_fsq_audio_codec_config(assets_->codec),
        assets_->codec_weights,
        execution_context(),
        graph_arena_bytes,
        kDefaultWeightContextBytes,
        codec_weight_type,
        conv_weight_type);
}

NeuTTSSession::~NeuTTSSession() = default;

std::string NeuTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind NeuTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode NeuTTSSession::run_mode() const {
    return task_.mode;
}

void NeuTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "NeuTTS");
    mark_prepared();
}

NeuTTSRequest NeuTTSSession::parse_request(const runtime::TaskRequest & request) const {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "NeuTTS");
    NeuTTSRequest out;
    out.text = request_text(request);
    out.speaker = request_speaker(request);
    out.emotion = request_emotion(request);
    out.generation = request_generation_options(request);
    return out;
}

runtime::AudioBuffer NeuTTSSession::synthesize(const NeuTTSRequest & request, bool streaming) {
    const auto start = Clock::now();
    auto timing_start = Clock::now();
    const auto prompt = prompt_builder_.build(request.text, request.speaker, request.emotion);
    debug::timing_log_scalar("neutts.prompt.build_ms", engine::debug::elapsed_ms(timing_start));
    debug::trace_log_scalar("neutts.speaker", prompt.speaker);
    debug::trace_log_scalar("neutts.emotion", prompt.emotion);
    debug::trace_log_scalar("neutts.prompt.tokens", static_cast<int64_t>(prompt.token_ids.size()));
    debug::trace_log_scalar("neutts.prompt.token_ids.fnv1a", fnv1a_i32(prompt.token_ids));
    debug::trace_log_scalar("neutts.generation.max_tokens", request.generation.max_tokens);
    debug::trace_log_scalar("neutts.generation.min_tokens", request.generation.min_tokens);
    debug::trace_log_scalar("neutts.generation.temperature", request.generation.temperature);
    debug::trace_log_scalar("neutts.generation.top_k", request.generation.top_k);
    debug::trace_log_scalar("neutts.generation.seed", request.generation.seed);

    std::function<void(int32_t)> on_speech_code;
    std::vector<float> overlap;
    if (streaming) {
        runtime::StreamingAudioConfig config;
        config.policy = runtime::StreamingAudioChunkPolicy::Fixed;
        config.frames_per_chunk = 25;
        config.left_context_frames = 51;
        config.right_context_frames = 5;
        std::vector<int32_t> reference;
        for (const auto token : prompt.token_ids) {
            if (token >= prompt.speech_token_start && token <= prompt.speech_token_end) {
                reference.push_back(token - prompt.speech_token_start);
            }
        }
        stream_audio_.begin(config, [&](const runtime::StreamingAudioWindow<int32_t> & window) {
            const auto hop = static_cast<size_t>(assets_->codec.hop_length);
            auto decoded = codec_->decode_audio(std::vector<int32_t>(window.data, window.data + window.frames()));
            const size_t begin = window.left_frames * hop;
            const size_t count = window.new_frames * hop;
            const size_t blend = std::min(overlap.size(), count);
            for (size_t i = 0; i < blend; ++i) {
                const float alpha = blend > 1 ? static_cast<float>(i) / static_cast<float>(blend - 1) : 1.0F;
                decoded[begin + i] = overlap[i] * (1.0F - alpha) + decoded[begin + i] * alpha;
            }
            const size_t tail = std::min(size_t{2}, window.right_frames) * hop;
            overlap.assign(decoded.begin() + begin + count, decoded.begin() + begin + count + tail);
            return runtime::AudioBuffer{static_cast<int>(assets_->codec.output_sample_rate), 1,
                std::vector<float>(decoded.begin() + begin, decoded.begin() + begin + count)};
        }, stream_event_sink(), reference);
        on_speech_code = [this](int32_t code) { stream_audio_.push({code}); };
    }

    timing_start = Clock::now();
    auto codes = ar_->generate(
        prompt.token_ids,
        prompt.speech_token_start,
        prompt.speech_token_end,
        prompt.speech_generation_end,
        request.generation,
        on_speech_code);
    const double ar_total_ms = engine::debug::elapsed_ms(timing_start);
    debug::timing_log_scalar("neutts.ar.total_ms", streaming
        ? ar_total_ms - stream_audio_.decode_ms() - stream_audio_.publish_ms()
        : ar_total_ms);
    if (codes.speech_codes.empty()) {
        throw std::runtime_error("NeuTTS generated no speech codes");
    }

    timing_start = Clock::now();
    auto audio = streaming ? stream_audio_.finish().samples : codec_->decode_audio(codes.speech_codes);
    debug::timing_log_scalar("neutts.codec.total_ms", streaming
        ? stream_audio_.decode_ms() : engine::debug::elapsed_ms(timing_start));
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(assets_->codec.output_sample_rate);
    out.channels = 1;
    out.samples = std::move(audio);
    debug::trace_log_scalar("neutts.output_samples", static_cast<int64_t>(out.samples.size()));
    debug::timing_log_scalar("neutts.segment.total_ms", engine::debug::elapsed_ms(start));
    return out;
}

runtime::TaskResult NeuTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("NeuTTS run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("NeuTTS run requires an offline session");
    }
    const auto wall_start = Clock::now();
    std::vector<runtime::AudioBuffer> chunks;
    const auto chunk_requests = split_neutts_request(request);
    chunks.reserve(chunk_requests.size());
    for (size_t i = 0; i < chunk_requests.size(); ++i) {
        auto parsed = parse_request(chunk_requests[i]);
        if (i > 0) {
            parsed.generation.seed += static_cast<uint64_t>(i);
        }
        chunks.push_back(synthesize(parsed));
    }
    runtime::TaskResult result;
    result.audio_output = merge_chunks(chunks);
    debug::trace_log_scalar("neutts.text.chunk_count", static_cast<int64_t>(chunks.size()));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::TaskResult NeuTTSSession::generate_stream(const runtime::TaskRequest & request) {
    require_prepared("NeuTTS streaming");
    const auto start = Clock::now();
    const auto chunk_requests = split_neutts_request(request);
    runtime::AudioBuffer audio;
    for (size_t i = 0; i < chunk_requests.size(); ++i) {
        auto parsed = parse_request(chunk_requests[i]);
        if (i > 0) {
            parsed.generation.seed += static_cast<uint64_t>(i);
        }
        runtime::append_audio_buffer(audio, synthesize(parsed, true));
    }
    if (chunk_requests.empty()) {
        throw std::runtime_error("NeuTTS streaming text chunking produced no segments");
    }
    debug::trace_log_scalar("neutts.streaming.chunk_count", static_cast<int64_t>(chunk_requests.size()));
    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_neutts_loader() {
    runtime::SpecBackedVoiceModelConfig<NeuTTSAssets> config;
    config.family = kFamily;
    config.load_assets = load_neutts_assets;
    config.create_session = create_neutts_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::neutts

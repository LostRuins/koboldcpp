#include "engine/community_models/soprano_tts/session.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/community_models/soprano_tts/generator.h"
#include "engine/community_models/soprano_tts/tokenizer_text.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/text/chunking.h"
#include "engine/community_models/soprano_tts/vocoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace engine::community_models::soprano_tts {
namespace {

constexpr const char * kFamily = "soprano_tts";
constexpr size_t kDefaultGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 256ull * 1024ull * 1024ull;

std::shared_ptr<const SopranoTTSAssets> require_assets(
    std::shared_ptr<const SopranoTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Soprano session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Soprano session requires a model contract");
    }
    return contract;
}

std::string request_text(const runtime::TaskRequest & request) {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Soprano requires non-empty text input");
    }
    return request.text_input->text;
}

SopranoGenerationOptions request_generation_options(const runtime::TaskRequest & request) {
    SopranoGenerationOptions out;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        out.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("Soprano temperature must be positive");
        }
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        out.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"eos_bias"})) {
        out.eos_bias = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.has_seed = true;
    }
    if (!out.has_seed) {
        out.seed = runtime::random_u64_seed();
    }
    if (out.max_new_tokens < 1) {
        throw std::runtime_error("Soprano max_tokens must be positive");
    }
    return out;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_soprano_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const SopranoTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<SopranoTTSOfflineSession>(
        task, options, std::move(assets), std::move(contract));
}

}  // namespace

SopranoTTSOfflineSession::SopranoTTSOfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SopranoTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    const auto normalized_options = runtime::apply_option_v1_compatibility(
        options,
        {{"backbone_weight_type", "soprano_tts.backbone_weight_type"},
         {"decoder_weight_type", "soprano_tts.decoder_weight_type"},
         {"text_chunk_size", "soprano_tts.text_chunk_size"}},
        "Soprano");
    auto validation_options = normalized_options;
    // Older embedded specs classify the weight controls as load options.
    validation_options.options.erase("soprano_tts.backbone_weight_type");
    validation_options.options.erase("soprano_tts.decoder_weight_type");
    validation_options.options.erase("soprano_tts.text_chunk_size");
    runtime::validate_spec_backed_session_options(validation_options, *contract_, kFamily, "Soprano");
    core::ExecutionContext & execution = execution_context();
    const auto backbone_storage = runtime::parse_tensor_storage_option(
        normalized_options.options,
        "soprano_tts.backbone_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto decoder_storage = runtime::parse_tensor_storage_option(
        normalized_options.options,
        "soprano_tts.decoder_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16});
    generator_ = std::make_unique<SopranoQwen3Generator>(
        *assets_, execution, kDefaultGraphArenaBytes, kDefaultGraphArenaBytes,
        kDefaultWeightContextBytes, backbone_storage);
    decoder_ = std::make_unique<SopranoVocosRuntime>(
        *assets_, execution, kDefaultWeightContextBytes, kDefaultGraphArenaBytes,
        decoder_storage, decoder_storage);
}

SopranoTTSOfflineSession::~SopranoTTSOfflineSession() = default;

std::string SopranoTTSOfflineSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind SopranoTTSOfflineSession::task_kind() const {
    return runtime::VoiceTaskKind::Tts;
}

runtime::RunMode SopranoTTSOfflineSession::run_mode() const {
    return task_.mode;
}

SopranoRequest SopranoTTSOfflineSession::make_request(const runtime::TaskRequest & request) const {
    SopranoRequest out;
    out.text = request_text(request);
    out.generation = request_generation_options(request);
    return out;
}
void SopranoTTSOfflineSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Soprano");
    mark_prepared();
}


runtime::TaskResult SopranoTTSOfflineSession::run(const runtime::TaskRequest & request) {
    require_prepared("Soprano run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Soprano run requires an offline session");
    }
    const SopranoRequest req = make_request(request);
    const auto audio = synthesize(req);

    runtime::TaskResult result;
    result.audio_output = audio;
    return result;
}

runtime::AudioBuffer SopranoTTSOfflineSession::synthesize(const SopranoRequest & request) {
    const std::filesystem::path tokenizer_path =
        assets_->resources.require_file("tokenizer_json");
    SopranoTextTokenizer tokenizer(tokenizer_path);
    const int64_t chunk_codepoints = runtime::parse_i64_option(
        options().options, {"soprano_tts.text_chunk_size", "text_chunk_size"})
        .value_or(200);
    const auto chunks = engine::text::split_text_chunks(
        request.text, chunk_codepoints, engine::text::TextChunkMode::Default);
    runtime::AudioBuffer out;
    for (const auto & chunk : chunks) {
        const auto prompt_ids = tokenizer.encode_text(chunk);
        const auto generate_start = std::chrono::steady_clock::now();
        const auto generated = generator_->generate(prompt_ids, request.generation);
        const auto generate_end = std::chrono::steady_clock::now();
        engine::debug::timing_log_scalar(
            "soprano_tts.lm.generate_ms", engine::debug::elapsed_ms(generate_start, generate_end));
        engine::debug::trace_log_scalar("soprano_tts.lm.frames", generated.frames);
        auto audio = decoder_->decode(generated.features, generated.frames);
        engine::debug::timing_log_scalar(
            "soprano_tts.decoder.decode_ms",
            engine::debug::elapsed_ms(generate_end, std::chrono::steady_clock::now()));
        if (out.sample_rate == 0) {
            out.sample_rate = audio.sample_rate;
            out.channels = audio.channels;
        } else if (out.sample_rate != audio.sample_rate || out.channels != audio.channels) {
            throw std::runtime_error("Soprano chunk audio format mismatch");
        }
        out.samples.insert(out.samples.end(), audio.samples.begin(), audio.samples.end());
    }
    if (out.sample_rate == 0) {
        throw std::runtime_error("Soprano produced no audio chunks");
    }
    return out;
}


// --------------------------------------------------------------------------- //
// Streaming interface
// --------------------------------------------------------------------------- //
runtime::StreamingPolicy SopranoTTSOfflineSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}
void SopranoTTSOfflineSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Soprano start_stream");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Soprano");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Soprano start_stream requires a streaming session");
    }
    reset();
    // Parse the request (and its sampling options) once, then keep one full
    // request per text chunk so next_stream_event honors the user options.
    const auto chunk_codepoints = runtime::parse_i64_option(
        options().options, {"soprano_tts.text_chunk_size", "text_chunk_size"}).value_or(200);
    const auto chunks = engine::text::split_text_chunks(
        request_text(request), chunk_codepoints, engine::text::TextChunkMode::Default);
    SopranoRequest parsed;
    parsed.generation = request_generation_options(request);
    streaming_requests_.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        SopranoRequest chunk_request;
        chunk_request.text = chunk;
        chunk_request.generation = parsed.generation;
        streaming_requests_.push_back(std::move(chunk_request));
    }
    if (streaming_requests_.empty()) {
        throw std::runtime_error("Soprano streaming text chunking produced no segments");
    }
    streaming_started_ = true;
}
std::optional<runtime::StreamEvent> SopranoTTSOfflineSession::next_stream_event() {
    if (!streaming_started_) {
        throw std::runtime_error("Soprano streaming has not been started");
    }
    if (streaming_index_ >= streaming_requests_.size()) {
        return std::nullopt;
    }
    auto audio = synthesize(streaming_requests_[streaming_index_]);
    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({
        "chunk_" + std::to_string(streaming_index_),
        audio,
        {},
    });
    streaming_chunks_.push_back(std::move(audio));
    if (stream_sink_) {
        stream_sink_(event);
    }
    ++streaming_index_;
    return event;
}
void SopranoTTSOfflineSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}
runtime::TaskResult SopranoTTSOfflineSession::finish_stream() {
    if (!streaming_started_) {
        throw std::runtime_error("Soprano streaming has not been started");
    }
    runtime::TaskResult result;
    runtime::AudioBuffer merged;
    for (const auto & chunk_audio : streaming_chunks_) {
        if (merged.sample_rate == 0) {
            merged = chunk_audio;
        } else {
            runtime::append_audio_buffer(merged, chunk_audio);
        }
    }
    result.audio_output = std::move(merged);
    reset();
    return result;
}
void SopranoTTSOfflineSession::reset() {
    streaming_requests_.clear();
    streaming_index_ = 0;
    streaming_chunks_.clear();
    streaming_started_ = false;
}
runtime::StreamEvent SopranoTTSOfflineSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("Soprano is a TTS model and does not accept audio input");
}
runtime::TaskResult SopranoTTSOfflineSession::finalize() {
    return runtime::TaskResult{};
}


std::shared_ptr<runtime::IVoiceModelLoader> make_soprano_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<SopranoTTSAssets> config;
    config.family = kFamily;
    config.load_assets = load_soprano_tts_assets;
    config.create_session = create_soprano_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::soprano_tts

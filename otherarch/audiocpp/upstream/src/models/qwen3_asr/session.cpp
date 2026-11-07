#include "engine/models/qwen3_asr/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/qwen3_forced_aligner/session.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kTimestampFixedChunkContextSeconds = 1.0;
constexpr double kPreferredStreamingFeedSeconds = 1.0;
constexpr float kDefaultStreamingWindowSeconds = 30.0F;

std::shared_ptr<const Qwen3ASRAssets> require_assets(
    std::shared_ptr<const Qwen3ASRAssets> assets, const runtime::SessionOptions & options) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 ASR session requires assets");
    }
    const auto path = runtime::find_option(options.options, {"qwen3_asr.lora"});
    if (!path || path->empty()) {
        return assets;
    }
    engine::assets::WeightAdapterReadOptions read_options;
    read_options.require_scaling_metadata = true;
    read_options.strength = runtime::parse_finite_float_option(
        options.options, {"qwen3_asr.lora_scale"}).value_or(1.0F);
    auto adapter = engine::assets::read_weight_adapter(std::filesystem::u8path(*path), read_options);
    engine::assets::WeightAdapterBindings bindings;
    for (const auto & tensor : assets->model_weights->tensors()) {
        bindings.add(tensor.name, tensor.name);
        if (tensor.name.rfind("thinker.", 0) == 0) {
            bindings.add(tensor.name.substr(8), tensor.name);
        }
        if (assets->config.hf_transformers_layout) {
            if (tensor.name.rfind("model.language_model.", 0) == 0) {
                bindings.add("model." + tensor.name.substr(21), tensor.name);
            } else if (tensor.name.rfind("model.audio_tower.", 0) == 0) {
                bindings.add(tensor.name.substr(6), tensor.name);
            }
        }
    }
    auto plan = engine::assets::bind_weight_adapter(
        *assets->model_weights, std::move(adapter), bindings, engine::assets::WeightMergeMode::DeltaF32);
    auto adapted = std::make_shared<Qwen3ASRAssets>(*assets);
    adapted->model_weights = engine::assets::make_weight_adapter_source(
        assets->model_weights, std::move(plan), "qwen3_asr.lora");
    return adapted;
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

void validate_audio_encoder_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error("qwen3_asr.audio_encoder_weight_type currently supports only native, f32, and f16");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    std::initializer_list<std::string_view> keys,
    engine::assets::TensorStorageType default_value) {
    if (const auto value = runtime::find_option(options.options, keys)) {
        return engine::assets::parse_tensor_storage_type(*value);
    }
    return default_value;
}

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("Qwen3 ASR audio chunking requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Qwen3 ASR audio samples must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

bool request_return_timestamps(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"return_timestamps"})) {
        return runtime::parse_bool_option(*value, "return_timestamps");
    }
    return false;
}

bool request_clamp_timestamps_to_audio(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"clamp_timestamps_to_audio"})) {
        return runtime::parse_bool_option(*value, "clamp_timestamps_to_audio");
    }
    return false;
}

bool request_preserve_punctuation(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(
            request.options,
            {"qwen3_asr.preserve_punctuation", "preserve_punctuation"})) {
        return runtime::parse_bool_option(*value, "qwen3_asr.preserve_punctuation");
    }
    return false;
}

void log_chunk_word_diagnostics(
    int64_t chunk_index,
    int64_t chunk_count,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span,
    const runtime::TaskResult & item,
    size_t merged_before,
    size_t merged_after,
    int64_t source_sample_rate,
    int64_t timestamp_sample_rate) {
    if (!debug::log_enabled()) {
        return;
    }
    const int64_t source_samples = source_sample_rate == timestamp_sample_rate
        ? source_span.end_sample - source_span.start_sample
        : static_cast<int64_t>(std::llround(
            static_cast<double>(source_span.end_sample - source_span.start_sample) *
            static_cast<double>(timestamp_sample_rate) /
            static_cast<double>(source_sample_rate)));
    int64_t outside_count = 0;
    int64_t first_outside = -1;
    int64_t max_word_end = 0;
    for (size_t i = 0; i < item.word_timestamps.size(); ++i) {
        const auto & word = item.word_timestamps[i];
        max_word_end = std::max(max_word_end, word.span.end_sample);
        const int64_t local_start = std::max<int64_t>(word.span.start_sample, 0);
        const int64_t local_end = std::min<int64_t>(word.span.end_sample, source_samples);
        if (local_start >= local_end) {
            if (first_outside < 0) {
                first_outside = static_cast<int64_t>(i);
            }
            ++outside_count;
        }
    }
    std::ostringstream out;
    out << "qwen3_asr.words_chunk"
        << " index=" << chunk_index
        << " count=" << chunk_count
        << " source_start=" << source_span.start_sample
        << " source_end=" << source_span.end_sample
        << " keep_start=" << keep_span.start_sample
        << " keep_end=" << keep_span.end_sample
        << " source_sample_rate=" << source_sample_rate
        << " timestamp_sample_rate=" << timestamp_sample_rate
        << " raw_words=" << item.word_timestamps.size()
        << " merged_delta=" << (merged_after - merged_before)
        << " outside_words=" << outside_count
        << " source_samples=" << source_samples
        << " max_word_end=" << max_word_end;
    if (item.text_output.has_value()) {
        out << " text_chars=" << item.text_output->text.size();
    }
    if (!item.word_timestamps.empty()) {
        const auto & first = item.word_timestamps.front();
        const auto & last = item.word_timestamps.back();
        out << " first_word=\"" << first.word << "\""
            << " first_start=" << first.span.start_sample
            << " first_end=" << first.span.end_sample
            << " last_word=\"" << last.word << "\""
            << " last_start=" << last.span.start_sample
            << " last_end=" << last.span.end_sample;
    }
    if (first_outside >= 0) {
        const auto index = static_cast<size_t>(first_outside);
        const auto append = [&](const char * prefix, int64_t word_index, const runtime::WordTimestamp & word) {
            out << ' ' << prefix << "_index=" << word_index
                << ' ' << prefix << "_word=\"" << word.word << "\""
                << ' ' << prefix << "_start=" << word.span.start_sample
                << ' ' << prefix << "_end=" << word.span.end_sample;
        };
        if (index > 0) {
            append("prev", first_outside - 1, item.word_timestamps[index - 1]);
        }
        append("bad", first_outside, item.word_timestamps[index]);
        if (index + 1 < item.word_timestamps.size()) {
            append("next", first_outside + 1, item.word_timestamps[index + 1]);
        }
    }
    debug::log_message(debug::LogLevel::Info, "qwen3_asr", out.str());
}

}  // namespace

Qwen3ASRSession::Qwen3ASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3ASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets), options)),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.audio_encoder_graph_arena_mb"}, 128ull * 1024ull * 1024ull)),
      thinker_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.text_decoder_prefill_graph_arena_mb", "qwen3_asr.thinker_prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.text_decoder_decode_graph_arena_mb", "qwen3_asr.thinker_decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.text_decoder_weight_context_mb", "qwen3_asr.thinker_weight_context_mb"}, 64ull * 1024ull * 1024ull)),
      audio_encoder_weight_storage_type_(option_weight_type(options, {"qwen3_asr.audio_encoder_weight_type"}, engine::assets::TensorStorageType::Native)),
      thinker_weight_storage_type_(option_weight_type(
          options,
          {"qwen3_asr.text_decoder_weight_type", "qwen3_asr.thinker_weight_type"},
          option_weight_type(options, {"qwen3_asr.weight_type"}, engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_->model_weights, assets_->config.audio_encoder,
          execution_context(), {audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_, "qwen3_asr.audio_encoder"},
          assets_->config.audio_encoder_binding),
      thinker_(
          assets_,
          execution_context(),
          thinker_prefill_graph_arena_bytes_,
          thinker_decode_graph_arena_bytes_,
          thinker_weight_context_bytes_,
          thinker_weight_storage_type_),
      prompt_builder_(tokenizer_),
      postprocessor_(tokenizer_),
      vad_model_path_(runtime::find_option(options.options, {"qwen3_asr.vad_model_path", "qwen3_asr.vad_path"}).value_or(default_vad_model_path().string())) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Qwen3 ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 ASR supports offline and streaming sessions");
    }
    validate_audio_encoder_weight_storage(audio_encoder_weight_storage_type_);
    validate_matmul_weight_storage(thinker_weight_storage_type_, "qwen3_asr.thinker_weight_type");
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("qwen3_asr.", 0) == 0 &&
            key != "qwen3_asr.audio_encoder_graph_arena_mb" &&
            key != "qwen3_asr.thinker_prefill_graph_arena_mb" &&
            key != "qwen3_asr.thinker_decode_graph_arena_mb" &&
            key != "qwen3_asr.thinker_weight_context_mb" &&
            key != "qwen3_asr.audio_encoder_weight_type" &&
            key != "qwen3_asr.thinker_weight_type" &&
            key != "qwen3_asr.text_decoder_prefill_graph_arena_mb" &&
            key != "qwen3_asr.text_decoder_decode_graph_arena_mb" &&
            key != "qwen3_asr.text_decoder_weight_context_mb" &&
            key != "qwen3_asr.text_decoder_weight_type" &&
            key != "qwen3_asr.forced_aligner_path" &&
            key != "qwen3_asr.vad_path" &&
            key != "qwen3_asr.weight_type" &&
            key != "qwen3_asr.lora" &&
            key != "qwen3_asr.lora_scale" &&
            key != "qwen3_asr.forced_aligner_model_path" &&
            key != "qwen3_asr.aligner_model_path" &&
            key != "qwen3_asr.vad_model_path") {
            throw std::runtime_error("unknown Qwen3 ASR session option: " + key);
        }
    }
    if (const auto aligner_path = runtime::find_option(
            options.options,
            {"qwen3_asr.forced_aligner_path", "qwen3_asr.forced_aligner_model_path", "qwen3_asr.aligner_model_path"})) {
        runtime::SessionOptions aligner_options;
        aligner_options.backend = options.backend;
        for (const auto & [key, value] : options.options) {
            if (key.rfind("qwen3_forced_aligner.", 0) == 0) {
                aligner_options.options.emplace(key, value);
            }
        }
        forced_aligner_session_ = std::make_unique<engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession>(
            runtime::TaskSpec{runtime::VoiceTaskKind::Alignment, runtime::RunMode::Offline},
            aligner_options,
            load_qwen3_asr_assets(std::filesystem::path(*aligner_path), "qwen3_forced_aligner"));
    }
    assets_->model_weights->release_storage();
}

Qwen3ASRSession::~Qwen3ASRSession() = default;

std::string Qwen3ASRSession::family() const {
    return "qwen3_asr";
}

runtime::VoiceTaskKind Qwen3ASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3ASRSession::run_mode() const {
    return task_.mode;
}

void Qwen3ASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Qwen3 ASR prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult Qwen3ASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 ASR run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 ASR run() requires an offline session");
    }
    const auto chunks = audio_chunk_plan(request);
    if (chunks.empty()) {
        const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
        if (request.audio_input.has_value() &&
            (mode == engine::audio::AudioChunkMode::Vad ||
             (mode == engine::audio::AudioChunkMode::Auto && request_return_timestamps(request)))) {
            runtime::TaskResult empty;
            empty.text_output = runtime::Transcript{"", request.text_input.has_value() ? request.text_input->language : ""};
            return empty;
        }
        return run_single(make_request(request));
    }
    const auto & audio = *request.audio_input;
    if (chunks.size() <= 1) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
        auto item = run_single(make_request(item_request));
        std::vector<runtime::WordTimestamp> merged_words;
        const size_t merged_before = merged_words.size();
        engine::audio::append_chunk_word_timestamps(
            merged_words,
            item.word_timestamps,
            chunks.front().source_span,
            chunks.front().keep_span,
            audio.sample_rate,
            assets_->config.sample_rate);
        log_chunk_word_diagnostics(
            0,
            1,
            chunks.front().source_span,
            chunks.front().keep_span,
            item,
            merged_before,
            merged_words.size(),
            audio.sample_rate,
            assets_->config.sample_rate);
        item.word_timestamps = std::move(merged_words);
        return item;
    }
    runtime::TaskResult merged;
    std::ostringstream text;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const auto & chunk = chunks[chunk_index];
        runtime::TaskRequest item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunk.source_span);
        auto item = run_single(make_request(item_request));
        if (item.text_output.has_value() && !item.text_output->text.empty()) {
            if (text.tellp() > 0) {
                text << ' ';
            }
            text << item.text_output->text;
            if (!merged.text_output.has_value()) {
                merged.text_output = runtime::Transcript{"", item.text_output->language};
            } else if (merged.text_output->language.empty()) {
                merged.text_output->language = item.text_output->language;
            }
        }
        const size_t merged_before = merged.word_timestamps.size();
        engine::audio::append_chunk_word_timestamps(
            merged.word_timestamps,
            item.word_timestamps,
            chunk.source_span,
            chunk.keep_span,
            audio.sample_rate,
            assets_->config.sample_rate);
        log_chunk_word_diagnostics(
            static_cast<int64_t>(chunk_index),
            static_cast<int64_t>(chunks.size()),
            chunk.source_span,
            chunk.keep_span,
            item,
            merged_before,
            merged.word_timestamps.size(),
            audio.sample_rate,
            assets_->config.sample_rate);
    }
    if (merged.text_output.has_value()) {
        if (request_return_timestamps(request) && !merged.word_timestamps.empty() &&
            !request_preserve_punctuation(request)) {
            std::ostringstream word_text;
            for (const auto & word : merged.word_timestamps) {
                if (word_text.tellp() > 0) {
                    word_text << ' ';
                }
                word_text << word.word;
            }
            merged.text_output->text = word_text.str();
        } else {
            merged.text_output->text = text.str();
        }
    }
    return merged;
}

runtime::StreamingPolicy Qwen3ASRSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_seconds = kPreferredStreamingFeedSeconds;
    return policy;
}

void Qwen3ASRSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 ASR start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 ASR start_stream() requires a streaming session");
    }
    if (request_return_timestamps(request)) {
        throw std::runtime_error("Qwen3 ASR streaming does not support return_timestamps");
    }
    reset();
    streaming_request_ = request;
    if (streaming_request_.audio_input.has_value()) {
        streaming_request_.audio_input->samples.clear();
    }
    stream_started_ = true;
    stream_wall_start_ = Clock::now();
}

void Qwen3ASRSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void Qwen3ASRSession::reset() {
    require_prepared("Qwen3 ASR reset()");
    streaming_request_ = runtime::TaskRequest{};
    streaming_result_ = runtime::TaskResult{};
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_audio_offset_values_ = 0;
    streaming_text_.clear();
    streaming_published_bytes_ = 0;
    streaming_windows_processed_ = 0;
    stream_started_ = false;
    stream_wall_start_ = {};
}

runtime::StreamEvent Qwen3ASRSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Qwen3 ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 ASR process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Qwen3 ASR process_audio_chunk() requires start_stream");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    if (audio.channels <= 0 || audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Qwen3 ASR streaming audio chunk has invalid channel layout");
    }
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_audio_offset_values_ > 0) {
        streaming_audio_.samples.clear();
        streaming_audio_offset_values_ = 0;
    }
    runtime::append_audio_buffer(streaming_audio_, audio);
    return process_available_stream_chunks(false);
}

runtime::TaskResult Qwen3ASRSession::finish_stream() {
    return finalize();
}

runtime::TaskResult Qwen3ASRSession::finalize() {
    const auto finalize_start = Clock::now();
    require_prepared("Qwen3 ASR finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Qwen3 ASR finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Qwen3 ASR finalize() requires start_stream");
    }
    if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
        throw std::runtime_error("Qwen3 ASR streaming pending audio offset is out of range");
    }
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_windows_processed_ == 0) {
        throw std::runtime_error("Qwen3 ASR finalize() requires streamed audio");
    }
    (void) process_available_stream_chunks(true);
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output =
            runtime::Transcript{"", streaming_request_.text_input.has_value() ? streaming_request_.text_input->language : ""};
    }
    stream_started_ = false;
    if (stream_event_sink_ != nullptr) {
        runtime::StreamEvent event;
        event.is_final = true;
        stream_event_sink_(event);
    }
    engine::debug::timing_log_scalar("qwen3_asr.session.stream.windows", streaming_windows_processed_);
    engine::debug::timing_log_scalar(
        "qwen3_asr.session.stream.finalize_ms",
        engine::debug::elapsed_ms(finalize_start));
    if (stream_wall_start_ != std::chrono::steady_clock::time_point{}) {
        engine::debug::timing_log_scalar(
            "qwen3_asr.session.stream.wall_ms",
            engine::debug::elapsed_ms(stream_wall_start_));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(stream_wall_start_));
    }
    return streaming_result_;
}

runtime::TaskResult Qwen3ASRSession::run_single(const Qwen3ASRRequest & asr_request) {
    const auto wall_start = Clock::now();
    if (asr_request.generation.return_timestamps && forced_aligner_session_ == nullptr) {
        throw std::runtime_error(
            "Qwen3 ASR timestamp output requires --session-option "
            "qwen3_asr.forced_aligner_model_path=<path-to-Qwen3-ForcedAligner-0.6B>");
    }
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(asr_request.audio);
    const auto frontend_end = Clock::now();
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(asr_request, features.encoder_tokens);
    const auto prompt_end = Clock::now();
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto encoder_end = Clock::now();
    const auto thinker_start = Clock::now();
    const auto tokens = thinker_.generate(prompt, audio_embeddings, asr_request.generation);
    const auto thinker_end = Clock::now();
    const auto postprocess_start = Clock::now();
    const auto decoded = postprocessor_.decode(tokens, asr_request);
    const auto postprocess_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, decoded.language};
    result.word_timestamps = decoded.word_timestamps;
    if (asr_request.generation.return_timestamps) {
        if (!decoded.text.empty() &&
            engine::models::qwen3_forced_aligner::has_alignable_words(decoded.text, decoded.language)) {
            if (decoded.language.empty()) {
                throw std::runtime_error("Qwen3 ASR timestamp output requires a requested or detected language");
            }
            runtime::TaskRequest align_request;
            align_request.audio_input = asr_request.audio;
            align_request.text_input = runtime::Transcript{decoded.text, decoded.language};
            align_request.options["audio_chunk_mode"] = "none";
            if (asr_request.generation.clamp_timestamps_to_audio) {
                align_request.options["clamp_timestamps_to_audio"] = "true";
            }
            forced_aligner_session_->prepare(runtime::build_preparation_request(align_request));
            auto aligned = forced_aligner_session_->run(align_request);
            result.word_timestamps = std::move(aligned.word_timestamps);
        }
    }
    const auto wall_end = Clock::now();
    debug::timing_log_scalar("qwen3_asr.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("qwen3_asr.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("qwen3_asr.audio_encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
    debug::timing_log_scalar("qwen3_asr.thinker_ms", engine::debug::elapsed_ms(thinker_start, thinker_end));
    debug::timing_log_scalar("qwen3_asr.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::trace_log_scalar("qwen3_asr.prompt_tokens", prompt.input_ids.size());
    debug::trace_log_scalar("qwen3_asr.audio_frames", features.frames);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    return result;
}

runtime::StreamEvent Qwen3ASRSession::process_available_stream_chunks(bool final) {
    runtime::StreamEvent last_event;
    last_event.is_final = false;
    if (streaming_audio_.sample_rate <= 0 || streaming_audio_.channels <= 0) {
        return last_event;
    }
    if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
        throw std::runtime_error("Qwen3 ASR streaming pending audio offset is out of range");
    }
    if (streaming_audio_.samples.size() % static_cast<size_t>(streaming_audio_.channels) != 0 ||
        streaming_audio_offset_values_ % static_cast<size_t>(streaming_audio_.channels) != 0) {
        throw std::runtime_error("Qwen3 ASR streaming pending audio has invalid channel layout");
    }
    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(streaming_request_.options)
        .value_or(kDefaultStreamingWindowSeconds);
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("Qwen3 ASR streaming audio_chunk_seconds must be positive");
    }
    const int64_t window_frames =
        static_cast<int64_t>(std::llround(static_cast<double>(seconds) * static_cast<double>(streaming_audio_.sample_rate)));
    if (window_frames <= 0) {
        throw std::runtime_error("Qwen3 ASR streaming audio_chunk_seconds produced an empty chunk");
    }

    int64_t processed_chunks = 0;
    while (true) {
        const int64_t pending_frames = static_cast<int64_t>(
            (streaming_audio_.samples.size() - streaming_audio_offset_values_) /
            static_cast<size_t>(streaming_audio_.channels));
        if (pending_frames <= 0 || (!final && pending_frames < window_frames)) {
            break;
        }
        const int64_t take_frames = final ? std::min<int64_t>(pending_frames, window_frames) : window_frames;
        const size_t take_values = static_cast<size_t>(take_frames * streaming_audio_.channels);
        runtime::AudioBuffer chunk;
        chunk.sample_rate = streaming_audio_.sample_rate;
        chunk.channels = streaming_audio_.channels;
        const auto begin = streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_);
        chunk.samples.assign(begin, begin + static_cast<std::ptrdiff_t>(take_values));
        streaming_audio_offset_values_ += take_values;
        last_event = process_one_stream_chunk(chunk);
        ++streaming_windows_processed_;
        ++processed_chunks;
        if (stream_event_sink_ != nullptr && last_event.partial_text.has_value()) {
            stream_event_sink_(last_event);
            last_event.partial_text.reset();
        }
    }
    if (processed_chunks > 0) {
        if (streaming_audio_offset_values_ == streaming_audio_.samples.size()) {
            streaming_audio_.samples.clear();
            streaming_audio_offset_values_ = 0;
        } else if (streaming_audio_offset_values_ > 1ull * 1024ull * 1024ull &&
                   streaming_audio_offset_values_ * 2 > streaming_audio_.samples.size()) {
            streaming_audio_.samples.erase(
                streaming_audio_.samples.begin(),
                streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_));
            streaming_audio_offset_values_ = 0;
        }
    }
    return last_event;
}

runtime::StreamEvent Qwen3ASRSession::process_one_stream_chunk(const runtime::AudioBuffer & audio) {
    runtime::TaskRequest item_request = streaming_request_;
    item_request.audio_input = audio;
    item_request.options["audio_chunk_mode"] = "none";
    auto item = run_single(make_request(item_request));

    runtime::StreamEvent event;
    event.is_final = false;
    if (!item.text_output.has_value() || item.text_output->text.empty()) {
        return event;
    }
    const std::string delta = streaming_text_.empty()
        ? item.text_output->text
        : " " + item.text_output->text;
    streaming_text_ += delta;
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output = runtime::Transcript{"", item.text_output->language};
    } else if (streaming_result_.text_output->language.empty()) {
        streaming_result_.text_output->language = item.text_output->language;
    }
    streaming_result_.text_output->text = streaming_text_;
    if (streaming_published_bytes_ < streaming_text_.size()) {
        event.partial_text = runtime::Transcript{
            streaming_text_.substr(streaming_published_bytes_),
            streaming_result_.text_output->language,
        };
        streaming_published_bytes_ = streaming_text_.size();
    }
    return event;
}

std::vector<Qwen3ASRSession::AudioChunkPlan> Qwen3ASRSession::audio_chunk_plan(const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        return {};
    }
    const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return {};
    }
    if (mode == engine::audio::AudioChunkMode::QuietEnergy) {
        throw std::runtime_error("Qwen3 ASR does not support audio_chunk_mode=quiet_energy");
    }
    const bool return_timestamps = request_return_timestamps(request);
    const auto & audio = *request.audio_input;
    const int64_t frames = audio_frame_count(audio);
    if (mode == engine::audio::AudioChunkMode::Vad ||
        (mode == engine::audio::AudioChunkMode::Auto && return_timestamps)) {
        const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options).value_or(15.0F);
        if (!(seconds > 0.0F)) {
            throw std::runtime_error("Qwen3 ASR audio_chunk_seconds must be positive");
        }
        const auto options = engine::audio::VadAudioChunkOptions{
            static_cast<int64_t>(std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.5 * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.25 * static_cast<double>(audio.sample_rate))),
        };
        if (options.max_chunk_samples <= 0) {
            throw std::runtime_error("Qwen3 ASR audio_chunk_seconds produced an empty chunk");
        }
        const auto spans = engine::audio::plan_vad_audio_chunks(audio, vad_session(), options);
        std::vector<AudioChunkPlan> plan;
        plan.reserve(spans.size());
        for (const auto & span : spans) {
            plan.push_back(AudioChunkPlan{span, span});
        }
        return plan;
    }
    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options).value_or(return_timestamps ? 15.0F : 30.0F);
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("Qwen3 ASR audio_chunk_seconds must be positive");
    }
    const int64_t samples = static_cast<int64_t>(
        std::llround(static_cast<double>(seconds) * static_cast<double>(request.audio_input->sample_rate)));
    if (samples <= 0) {
        throw std::runtime_error("Qwen3 ASR audio_chunk_seconds produced an empty chunk");
    }
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        engine::audio::AudioChunkSpec{
            samples,
            samples,
            engine::audio::AudioChunkPadMode::Zero,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        });
    const int64_t context_samples = return_timestamps
        ? std::min<int64_t>(
            samples / 2,
            static_cast<int64_t>(std::llround(kTimestampFixedChunkContextSeconds * static_cast<double>(audio.sample_rate))))
        : 0;
    std::vector<AudioChunkPlan> plan;
    plan.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        const runtime::TimeSpan keep_span{
            chunk.output_start_sample,
            chunk.output_start_sample + chunk.valid_samples,
        };
        const runtime::TimeSpan source_span{
            std::max<int64_t>(0, keep_span.start_sample - context_samples),
            std::min<int64_t>(frames, keep_span.end_sample + context_samples),
        };
        plan.push_back(AudioChunkPlan{source_span, keep_span});
    }
    return plan;
}

runtime::IOfflineVoiceTaskSession & Qwen3ASRSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Qwen3 ASR internal VAD session does not support offline execution");
        }
        session.release();
        vad_session_.reset(offline);
    }
    return *vad_session_;
}

Qwen3ASRRequest Qwen3ASRSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Qwen3 ASR run() requires audio_input");
    }
    Qwen3ASRRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = assets_->config.max_new_tokens;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto value = runtime::find_option(request.options, {"language"})) {
        out.language = *value == "Auto" ? "" : *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
        if (out.generation.max_new_tokens <= 0) {
            throw std::runtime_error("Qwen3 ASR max_tokens must be positive");
        }
    }
    if (const auto value = runtime::find_option(request.options, {"return_timestamps"})) {
        out.generation.return_timestamps = runtime::parse_bool_option(*value, "return_timestamps");
    }
    out.generation.clamp_timestamps_to_audio = request_clamp_timestamps_to_audio(request);
    return out;
}

}  // namespace engine::models::qwen3_asr

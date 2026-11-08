#include "engine/models/qwen3_forced_aligner/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_forced_aligner {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultAudioEncoderGraphArenaBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultThinkerPrefillGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultThinkerDecodeGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultThinkerWeightContextBytes = 64ull * 1024ull * 1024ull;

bool request_clamp_timestamps_to_audio(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"clamp_timestamps_to_audio"})) {
        return runtime::parse_bool_option(*value, "clamp_timestamps_to_audio");
    }
    return false;
}

std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> require_assets(
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 forced aligner session requires assets");
    }
    if (assets->config.thinker_model_type != "qwen3_forced_aligner") {
        throw std::runtime_error("Qwen3 forced aligner requires qwen3_forced_aligner thinker config");
    }
    if (assets->config.timestamp_token_id <= 0 || assets->config.timestamp_segment_time_ms <= 0) {
        throw std::runtime_error("Qwen3 forced aligner model is missing timestamp config");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
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
    throw std::runtime_error("qwen3_forced_aligner.audio_encoder_weight_type currently supports only native, f32, and f16");
}

}  // namespace

Qwen3ForcedAlignerSession::Qwen3ForcedAlignerSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options,
          {"qwen3_forced_aligner.audio_encoder_graph_arena_mb"},
          kDefaultAudioEncoderGraphArenaBytes)),
      thinker_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options,
          {"qwen3_forced_aligner.thinker_prefill_graph_arena_mb"},
          kDefaultThinkerPrefillGraphArenaBytes)),
      thinker_decode_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options,
          {"qwen3_forced_aligner.thinker_decode_graph_arena_mb"},
          kDefaultThinkerDecodeGraphArenaBytes)),
      thinker_weight_context_bytes_(runtime::parse_size_mb_option(
          options.options,
          {"qwen3_forced_aligner.thinker_weight_context_mb"},
          kDefaultThinkerWeightContextBytes)),
      audio_encoder_weight_storage_type_(option_weight_type(
          options,
          "qwen3_forced_aligner.audio_encoder_weight_type",
          engine::assets::TensorStorageType::Native)),
      thinker_weight_storage_type_(option_weight_type(
          options,
          "qwen3_forced_aligner.thinker_weight_type",
          option_weight_type(options, "qwen3_forced_aligner.weight_type", engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_->model_weights, assets_->config.audio_encoder,
          execution_context(), {audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_, "qwen3_forced_aligner.audio_encoder"},
          assets_->config.audio_encoder_binding),
      thinker_(
          assets_,
          execution_context(),
          thinker_prefill_graph_arena_bytes_,
          thinker_decode_graph_arena_bytes_,
          thinker_weight_context_bytes_,
          thinker_weight_storage_type_),
      processor_(*assets_, tokenizer_) {
    if (task_.task != runtime::VoiceTaskKind::Alignment) {
        throw std::runtime_error("Qwen3 forced aligner only supports VoiceTaskKind::Alignment");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 forced aligner currently supports offline sessions");
    }
    validate_audio_encoder_weight_storage(audio_encoder_weight_storage_type_);
    validate_matmul_weight_storage(thinker_weight_storage_type_, "qwen3_forced_aligner.thinker_weight_type");
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("qwen3_forced_aligner.", 0) == 0 &&
            key != "qwen3_forced_aligner.audio_encoder_graph_arena_mb" &&
            key != "qwen3_forced_aligner.thinker_prefill_graph_arena_mb" &&
            key != "qwen3_forced_aligner.thinker_decode_graph_arena_mb" &&
            key != "qwen3_forced_aligner.thinker_weight_context_mb" &&
            key != "qwen3_forced_aligner.audio_encoder_weight_type" &&
            key != "qwen3_forced_aligner.thinker_weight_type" &&
            key != "qwen3_forced_aligner.weight_type") {
            throw std::runtime_error("unknown Qwen3 forced aligner session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

std::string Qwen3ForcedAlignerSession::family() const {
    return "qwen3_forced_aligner";
}

runtime::VoiceTaskKind Qwen3ForcedAlignerSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3ForcedAlignerSession::run_mode() const {
    return task_.mode;
}

void Qwen3ForcedAlignerSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value() || !request.text.has_value()) {
        throw std::runtime_error("Qwen3 forced aligner prepare() requires audio and transcript contracts");
    }
    mark_prepared();
}

runtime::TaskResult Qwen3ForcedAlignerSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 forced aligner run()");
    const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (mode == engine::audio::AudioChunkMode::Fixed ||
        mode == engine::audio::AudioChunkMode::QuietEnergy ||
        mode == engine::audio::AudioChunkMode::Vad) {
        throw std::runtime_error(
            "Qwen3 forced aligner does not support standalone audio chunking; "
            "chunk audio before ASR so each aligner request receives matching audio and transcript");
    }
    return run_single(request);
}

runtime::TaskResult Qwen3ForcedAlignerSession::run_single(const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Qwen3 forced aligner run() requires audio_input");
    }
    if (!request.text_input.has_value() || request.text_input->text.empty() || request.text_input->language.empty()) {
        throw std::runtime_error("Qwen3 forced aligner run() requires transcript text and language");
    }
    const int64_t run_index = run_index_++;
    if (request.audio_input->channels <= 0) {
        throw std::runtime_error("Qwen3 forced aligner audio requires positive channels");
    }
    if (request.audio_input->samples.size() % static_cast<size_t>(request.audio_input->channels) != 0) {
        throw std::runtime_error("Qwen3 forced aligner audio samples must be divisible by channel count");
    }
    const int64_t audio_frames =
        static_cast<int64_t>(request.audio_input->samples.size() / static_cast<size_t>(request.audio_input->channels));
    const auto wall_start = Clock::now();
    if (!has_alignable_words(request.text_input->text, request.text_input->language)) {
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{request.text_input->text, request.text_input->language};
        const auto wall_end = Clock::now();
        if (debug::log_enabled()) {
            std::ostringstream out;
            out << "qwen3_forced_aligner.run"
                << " index=" << run_index
                << " alignable_words=0"
                << " audio_frames=" << audio_frames
                << " audio_sample_rate=" << request.audio_input->sample_rate
                << " text_chars=" << request.text_input->text.size()
                << " language=\"" << request.text_input->language << "\"";
            debug::log_message(debug::LogLevel::Info, "qwen3_forced_aligner", out.str());
        }
        debug::trace_log_scalar("qwen3_forced_aligner.alignable_words", 0);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
        return result;
    }
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(*request.audio_input);
    const auto frontend_end = Clock::now();
    const auto prompt_start = Clock::now();
    const auto align_prompt = processor_.build_prompt(request.text_input->text, request.text_input->language, features.encoder_tokens);
    const auto prompt_end = Clock::now();
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto encoder_end = Clock::now();
    const auto thinker_start = Clock::now();
    const auto output_ids = thinker_.classify_prompt(align_prompt.prompt, audio_embeddings);
    const auto thinker_end = Clock::now();

    std::vector<int32_t> timestamp_ids;
    timestamp_ids.reserve(align_prompt.timestamp_positions.size());
    int32_t min_timestamp_id = std::numeric_limits<int32_t>::max();
    int32_t max_timestamp_id = std::numeric_limits<int32_t>::min();
    for (const int32_t position : align_prompt.timestamp_positions) {
        if (position < 0 || position >= static_cast<int32_t>(output_ids.size())) {
            throw std::runtime_error("Qwen3 forced aligner timestamp position out of range");
        }
        const int32_t id = output_ids[static_cast<size_t>(position)];
        min_timestamp_id = std::min(min_timestamp_id, id);
        max_timestamp_id = std::max(max_timestamp_id, id);
        timestamp_ids.push_back(id);
    }
    const auto postprocess_start = Clock::now();
    auto timestamps = processor_.parse_timestamps(
        align_prompt.words,
        timestamp_ids,
        assets_->config.sample_rate,
        audio_frames,
        request_clamp_timestamps_to_audio(request));
    const auto postprocess_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{request.text_input->text, request.text_input->language};
    result.word_timestamps = std::move(timestamps);
    const auto wall_end = Clock::now();
    if (debug::log_enabled()) {
        int64_t out_of_audio_count = 0;
        int64_t first_out_of_audio = -1;
        int64_t max_word_end = 0;
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            const auto & word = result.word_timestamps[i];
            max_word_end = std::max(max_word_end, word.span.end_sample);
            if (word.span.start_sample >= audio_frames || word.span.end_sample <= 0) {
                if (first_out_of_audio < 0) {
                    first_out_of_audio = static_cast<int64_t>(i);
                }
                ++out_of_audio_count;
            }
        }
        std::ostringstream out;
        out << "qwen3_forced_aligner.run"
            << " index=" << run_index
            << " alignable_words=" << align_prompt.words.size()
            << " audio_frames=" << audio_frames
            << " audio_sample_rate=" << request.audio_input->sample_rate
            << " feature_frames=" << features.frames
            << " encoder_tokens=" << features.encoder_tokens
            << " prompt_tokens=" << align_prompt.prompt.input_ids.size()
            << " timestamp_positions=" << align_prompt.timestamp_positions.size()
            << " timestamp_id_min=" << (timestamp_ids.empty() ? 0 : min_timestamp_id)
            << " timestamp_id_max=" << (timestamp_ids.empty() ? 0 : max_timestamp_id)
            << " max_word_end=" << max_word_end
            << " out_of_audio_words=" << out_of_audio_count
            << " text_chars=" << request.text_input->text.size()
            << " language=\"" << request.text_input->language << "\"";
        if (first_out_of_audio >= 0) {
            const auto append_word_span = [&](const char * prefix, int64_t word_index, const runtime::WordTimestamp & word) {
                out << ' ' << prefix << "_index=" << word_index
                    << ' ' << prefix << "_word=\"" << word.word << "\""
                    << ' ' << prefix << "_start=" << word.span.start_sample
                    << ' ' << prefix << "_end=" << word.span.end_sample;
            };
            const auto index = static_cast<size_t>(first_out_of_audio);
            if (index > 0) {
                append_word_span("prev", first_out_of_audio - 1, result.word_timestamps[index - 1]);
            }
            append_word_span("bad", first_out_of_audio, result.word_timestamps[index]);
            if (index + 1 < result.word_timestamps.size()) {
                append_word_span("next", first_out_of_audio + 1, result.word_timestamps[index + 1]);
            }
        }
        debug::log_message(debug::LogLevel::Info, "qwen3_forced_aligner", out.str());
        if (out_of_audio_count > 0) {
            for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
                const auto & word = result.word_timestamps[i];
                std::ostringstream word_out;
                word_out << "qwen3_forced_aligner.word"
                    << " run_index=" << run_index
                    << " word_index=" << i
                    << " word=\"" << word.word << "\""
                    << " start=" << word.span.start_sample
                    << " end=" << word.span.end_sample;
                if (i * 2 + 1 < timestamp_ids.size()) {
                    word_out << " timestamp_start_id=" << timestamp_ids[i * 2]
                        << " timestamp_end_id=" << timestamp_ids[i * 2 + 1];
                }
                debug::log_message(debug::LogLevel::Info, "qwen3_forced_aligner", word_out.str());
            }
        }
    }
    debug::timing_log_scalar("qwen3_forced_aligner.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("qwen3_forced_aligner.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("qwen3_forced_aligner.audio_encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
    debug::timing_log_scalar("qwen3_forced_aligner.thinker_ms", engine::debug::elapsed_ms(thinker_start, thinker_end));
    debug::timing_log_scalar("qwen3_forced_aligner.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::trace_log_scalar("qwen3_forced_aligner.prompt_tokens", align_prompt.prompt.input_ids.size());
    debug::trace_log_scalar("qwen3_forced_aligner.audio_frames", features.frames);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    return result;
}

}  // namespace engine::models::qwen3_forced_aligner

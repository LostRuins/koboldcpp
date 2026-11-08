#include "engine/community_models/sense_asr/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sense_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float kDefaultChunkSeconds = 30.0F;
constexpr float kDefaultStreamingWindowSeconds = 30.0F;
constexpr double kPreferredStreamingFeedSeconds = 1.0;

std::shared_ptr<const SenseAsrAssets>
require_assets(std::shared_ptr<const SenseAsrAssets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("SenseVoice session requires assets");
  }
  return assets;
}

const engine::model_spec::ModelContract &require_contract(
    const std::shared_ptr<const engine::model_spec::ModelContract> &contract) {
  if (contract == nullptr) {
    throw std::runtime_error("SenseVoice session requires a model contract");
  }
  return *contract;
}

void validate_weight_storage(assets::TensorStorageType storage,
                             const std::string &option) {
  if (storage == assets::TensorStorageType::Native ||
      storage == assets::TensorStorageType::F32 ||
      storage == assets::TensorStorageType::F16 ||
      storage == assets::TensorStorageType::BF16 ||
      storage == assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(option +
                           " supports only native, f32, f16, bf16, and q8_0");
}

assets::TensorStorageType
option_weight_type(const runtime::SessionOptions &options, const char *key,
                   assets::TensorStorageType fallback) {
  const auto value = options.options.find(key);
  return value == options.options.end()
             ? fallback
             : assets::parse_tensor_storage_type(value->second);
}

runtime::SessionOptions
validate_session_setup(const runtime::TaskSpec &task,
                       runtime::SessionOptions options,
                       const engine::model_spec::ModelContract &contract) {
  if (task.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("SenseVoice only supports VoiceTaskKind::Asr");
  }
  if (task.mode != runtime::RunMode::Offline &&
      task.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "SenseVoice supports offline and streaming sessions");
  }
  const auto shared = option_weight_type(options, "sense_asr.weight_type",
                                         assets::TensorStorageType::Native);
  validate_weight_storage(shared, "sense_asr.weight_type");
  runtime::validate_spec_backed_session_options(options, contract, "sense_asr",
                                                "SenseVoice");
  return options;
}

int64_t audio_frame_count(const runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0) {
    throw std::runtime_error(
        "SenseVoice audio requires a positive sample rate");
  }
  if (audio.channels <= 0) {
    throw std::runtime_error("SenseVoice audio requires positive channels");
  }
  if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::runtime_error(
        "SenseVoice audio samples must be divisible by channel count");
  }
  return static_cast<int64_t>(audio.samples.size() /
                              static_cast<size_t>(audio.channels));
}

std::unordered_map<std::string, std::string>
normalize_request_options(std::unordered_map<std::string, std::string> options,
                          const engine::model_spec::ModelContract &contract) {
  options = runtime::apply_option_v1_compatibility(
      std::move(options),
      {
          {"audio_chunk_seconds", "audio_chunk_duration_sec"},
          {"audio_chunk_duration_seconds", "audio_chunk_duration_sec"},
          {"audio_chunk_duration", "audio_chunk_duration_sec"},
      },
      "SenseVoice", "request");
  runtime::validate_spec_backed_request_options(options, contract,
                                                "SenseVoice");
  return options;
}

bool is_language_tag(const std::string &tag) {
  static const std::vector<std::string> kLanguages = {
      "auto", "zh", "en", "yue", "ja", "ko", "nospeech", "pt",
      "ru",   "es", "it", "fr",  "de", "nl", "pl",       "tr",
      "ar",   "hi", "vi", "th",  "id", "ms", "fa"};
  return std::find(kLanguages.begin(), kLanguages.end(), tag) !=
         kLanguages.end();
}

std::string sv_trim(const std::string &s) {
  const size_t first = s.find_first_not_of(' ');
  if (first == std::string::npos) {
    return "";
  }
  const size_t last = s.find_last_not_of(' ');
  return s.substr(first, last - first + 1);
}

SenseAsrDecodedTokens decode_ctc(const std::vector<float> &logits,
                                 int64_t frames, int64_t vocab_size,
                                 int64_t blank_id,
                                 const std::vector<std::string> &vocab,
                                 bool keep_tags) {
  SenseAsrDecodedTokens decoded;
  if (frames <= 0 || vocab_size <= 0) {
    return decoded;
  }
  std::vector<int32_t> ids;
  ids.reserve(static_cast<size_t>(frames));
  int32_t previous = -1;
  for (int64_t frame = 0; frame < frames; ++frame) {
    const float *column = logits.data() + static_cast<size_t>(frame) *
                                              static_cast<size_t>(vocab_size);
    int32_t argmax = 0;
    float best = column[0];
    for (int64_t token = 1; token < vocab_size; ++token) {
      if (column[token] > best) {
        best = column[token];
        argmax = static_cast<int32_t>(token);
      }
    }
    if (argmax != previous && argmax != blank_id) {
      ids.push_back(argmax);
    }
    previous = argmax;
  }
  if (std::getenv("SENSE_ASR_DUMP_IDS") != nullptr) {
    std::string s = "ids(" + std::to_string(frames) + "):";
    for (int32_t id : ids)
      s += " " + std::to_string(id);
    s += "\n";
    (void)!std::fwrite(s.data(), 1, s.size(), stderr);
  }
  if (std::getenv("SENSE_ASR_DUMP_LOGITS") != nullptr) {
    int start = std::atoi(std::getenv("SENSE_ASR_DUMP_LOGITS"));
    if (start < 0)
      start = 0;
    for (int64_t frame = start; frame < std::min<int64_t>(frames, start + 50);
         ++frame) {
      const float *column = logits.data() + static_cast<size_t>(frame) *
                                                static_cast<size_t>(vocab_size);
      int32_t argmax = 0;
      float best = column[0];
      for (int64_t token = 1; token < vocab_size; ++token) {
        if (column[token] > best) {
          best = column[token];
          argmax = static_cast<int32_t>(token);
        }
      }
      fprintf(stderr, "frame %lld: argmax=%d (%.4f)\n", (long long)frame,
              argmax, best);
    }
  }
  decoded.ids = std::move(ids);

  if (vocab.empty()) {
    return decoded;
  }
  std::string text;
  for (int32_t id : decoded.ids) {
    if (id < 0 || id >= static_cast<int32_t>(vocab.size())) {
      continue;
    }
    const std::string &piece = vocab[static_cast<size_t>(id)];
    if (piece.size() >= 2 && piece[0] == '<' && piece[1] == '|') {
      const std::string tag = piece.substr(2, piece.size() - 4);
      if (is_language_tag(tag) && decoded.language.empty()) {
        decoded.language = tag;
      }
      decoded.tags.push_back(tag);
      if (keep_tags) {
        text += piece;
      }
      continue;
    }
    text += piece;
  }
  const std::string lb = "\xe2\x96\x81";
  size_t position = 0;
  while ((position = text.find(lb, position)) != std::string::npos) {
    text.replace(position, 3, " ");
    position += 1;
  }
  decoded.text = sv_trim(text);
  return decoded;
}

bool ascii_word_boundary(const std::string &text, bool front) {
  if (text.empty()) {
    return false;
  }
  const unsigned char value =
      static_cast<unsigned char>(front ? text.front() : text.back());
  return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z');
}

void append_chunk_text(std::string &merged, std::string chunk) {
  chunk = engine::io::trim_ascii_whitespace(std::move(chunk));
  if (chunk.empty()) {
    return;
  }
  if (!merged.empty() && ascii_word_boundary(merged, false) &&
      ascii_word_boundary(chunk, true)) {
    merged.push_back(' ');
  }
  merged += chunk;
}

int32_t language_query_token(const std::string &language) {
  static const std::vector<std::pair<std::string, int32_t>> kLidTokens = {
      {"auto", 0}, {"zh", 3},  {"en", 4},       {"yue", 7},
      {"ja", 11},  {"ko", 12}, {"nospeech", 13}};
  for (const auto &[tag, token] : kLidTokens) {
    if (tag == language) {
      return token;
    }
  }
  return 0;
}

std::vector<int32_t>
query_tokens(const SenseAsrTranscriptionOptions &transcription,
             const std::vector<int32_t> &default_tokens) {
  constexpr int32_t kEventToken = 1;
  constexpr int32_t kEmotionToken = 2;
  constexpr int32_t kWithItnToken = 14;
  constexpr int32_t kWithoutItnToken = 15;
  std::vector<int32_t> tokens = default_tokens;
  if (tokens.size() >= 4) {
    tokens[0] = language_query_token(transcription.language);
    tokens[1] = kEventToken;
    tokens[2] = kEmotionToken;
    tokens[3] = transcription.enable_itn ? kWithItnToken : kWithoutItnToken;
  }
  return tokens;
}

std::filesystem::path default_vad_model_path() {
  return std::filesystem::path("assets") / "framework" / "models" /
         "silero_vad";
}

} // namespace

SenseAsrSession::SenseAsrSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const SenseAsrAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(
          validate_session_setup(task, options, require_contract(contract))),
      task_(task), assets_(require_assets(std::move(assets))),
      contract_(std::move(contract)),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"sense_asr.encoder_graph_arena_mb"},
          1024ull * 1024ull * 1024ull)),
      weight_storage_type_(option_weight_type(
          options, "sense_asr.weight_type", assets::TensorStorageType::Native)),
      frontend_(assets_->config.frontend),
      encoder_(assets_, execution_context(), encoder_graph_arena_bytes_,
               weight_storage_type_),
      vad_model_path_(
          runtime::find_option(options.options, {"sense_asr.vad_model_path"})
              .value_or(default_vad_model_path().string())) {
  encoder_.set_query_tokens(assets_->config.encoder.query_tokens);
  assets_->model_weights->release_storage();
}

SenseAsrSession::~SenseAsrSession() = default;

std::string SenseAsrSession::family() const { return "sense_asr"; }

runtime::VoiceTaskKind SenseAsrSession::task_kind() const { return task_.task; }

runtime::RunMode SenseAsrSession::run_mode() const { return task_.mode; }

void SenseAsrSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  if (!request.audio.has_value()) {
    throw std::runtime_error("SenseVoice prepare() requires an audio contract");
  }
  mark_prepared();
}

runtime::TaskResult SenseAsrSession::run(const runtime::TaskRequest &request) {
  require_prepared("SenseVoice run()");
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("SenseVoice run() requires an offline session");
  }
  auto normalized_request = request;
  normalized_request.options =
      normalize_request_options(request.options, *contract_);
  const auto chunks = audio_chunk_plan(normalized_request);
  if (chunks.empty()) {
    return run_single(make_request(normalized_request));
  }
  const auto &audio = *normalized_request.audio_input;
  if (chunks.size() == 1) {
    auto item = normalized_request;
    item.audio_input =
        engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
    return run_single(make_request(item));
  }

  runtime::TaskResult merged;
  std::string text;
  for (const auto &chunk : chunks) {
    auto item = normalized_request;
    item.audio_input =
        engine::audio::slice_audio_buffer(audio, chunk.source_span);
    const auto result = run_single(make_request(item));
    if (result.text_output.has_value()) {
      append_chunk_text(text, result.text_output->text);
      if (!merged.text_output.has_value()) {
        merged.text_output =
            runtime::Transcript{"", result.text_output->language};
      } else if (merged.text_output->language.empty()) {
        merged.text_output->language = result.text_output->language;
      }
    }
  }
  if (!merged.text_output.has_value()) {
    merged.text_output = runtime::Transcript{};
  }
  merged.text_output->text = std::move(text);
  return merged;
}

runtime::StreamingPolicy SenseAsrSession::streaming_policy() const {
  runtime::StreamingPolicy policy;
  policy.input = runtime::StreamingInputKind::AudioChunks;
  policy.output = runtime::StreamingOutputKind::FinalResult;
  policy.preferred_audio_chunk_seconds = kPreferredStreamingFeedSeconds;
  return policy;
}

void SenseAsrSession::begin_stream(const runtime::TaskRequest &request) {
  require_prepared("SenseVoice start_stream()");
  streaming_request_ = request;
  streaming_request_.options =
      normalize_request_options(request.options, *contract_);
  if (streaming_request_.audio_input.has_value()) {
    streaming_request_.audio_input->samples.clear();
  }
}

void SenseAsrSession::reset_stream_state() {
  require_prepared("SenseVoice reset()");
  streaming_request_ = runtime::TaskRequest{};
  streaming_audio_ = runtime::AudioBuffer{};
  streaming_audio_offset_values_ = 0;
  streaming_windows_processed_ = 0;
}

void
SenseAsrSession::consume_audio(const runtime::AudioChunk &chunk) {
  require_prepared("SenseVoice process_audio_chunk()");
  runtime::AudioBuffer audio;
  audio.sample_rate = chunk.sample_rate;
  audio.channels = chunk.channels;
  audio.samples = chunk.samples;
  if (audio.channels <= 0 ||
      audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::runtime_error(
        "SenseVoice streaming audio chunk has invalid channel layout");
  }
  if (streaming_audio_offset_values_ == streaming_audio_.samples.size() &&
      streaming_audio_offset_values_ > 0) {
    streaming_audio_.samples.clear();
    streaming_audio_offset_values_ = 0;
  }
  runtime::append_audio_buffer(streaming_audio_, audio);
  process_available_stream_chunks(false);
}

void SenseAsrSession::complete_stream(runtime::TaskResult &) {
  const auto finalize_start = Clock::now();
  require_prepared("SenseVoice finalize()");
  if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
    throw std::runtime_error(
        "SenseVoice streaming pending audio offset is out of range");
  }
  if (streaming_audio_offset_values_ == streaming_audio_.samples.size() &&
      streaming_windows_processed_ == 0) {
    throw std::runtime_error("SenseVoice finalize() requires streamed audio");
  }
  process_available_stream_chunks(true);
  runtime::AsrStreamMetadata event;
  event.is_final = true;
  publish_metadata(std::move(event));
  engine::debug::timing_log_scalar("sense_asr.session.stream.windows",
                                   streaming_windows_processed_);
  engine::debug::timing_log_scalar("sense_asr.session.stream.finalize_ms",
                                   engine::debug::elapsed_ms(finalize_start));
  engine::debug::timing_log_scalar("sense_asr.session.stream.wall_ms", stream_wall_ms());
}

SenseAsrSession::AsrRequest
SenseAsrSession::make_request(const runtime::TaskRequest &request) const {
  if (!request.audio_input.has_value()) {
    throw std::runtime_error("SenseVoice run() requires audio_input");
  }
  (void)audio_frame_count(*request.audio_input);
  AsrRequest out;
  out.audio = *request.audio_input;
  if (const auto value = runtime::find_option(request.options, {"language"})) {
    out.transcription.language = *value;
  }
  if (const auto value =
          runtime::find_option(request.options, {"enable_itn"})) {
    out.transcription.enable_itn =
        runtime::parse_bool_option(*value, "enable_itn");
  }
  if (const auto value = runtime::find_option(request.options, {"keep_tags"})) {
    out.transcription.keep_tags =
        runtime::parse_bool_option(*value, "keep_tags");
  }
  return out;
}

std::vector<SenseAsrSession::AudioChunkPlan>
SenseAsrSession::audio_chunk_plan(const runtime::TaskRequest &request) {
  if (!request.audio_input.has_value()) {
    return {};
  }
  const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
  if (mode == engine::audio::AudioChunkMode::None) {
    return {};
  }
  if (mode == engine::audio::AudioChunkMode::QuietEnergy) {
    throw std::runtime_error(
        "SenseVoice supports audio_chunk_mode=auto, fixed, or none");
  }
  const auto &audio = *request.audio_input;
  const int64_t frames = audio_frame_count(audio);
  if (mode == engine::audio::AudioChunkMode::Vad ||
      mode == engine::audio::AudioChunkMode::Auto) {
    const auto seconds =
        engine::audio::parse_audio_chunk_seconds_override(request.options)
            .value_or(kDefaultChunkSeconds);
    if (!(seconds > 0.0F)) {
      throw std::runtime_error(
          "SenseVoice audio_chunk_duration_sec must be positive");
    }
    const auto vad_options = engine::audio::VadAudioChunkOptions{
        static_cast<int64_t>(
            std::llround(static_cast<double>(seconds) *
                         static_cast<double>(audio.sample_rate))),
        static_cast<int64_t>(
            std::llround(0.5 * static_cast<double>(audio.sample_rate))),
        static_cast<int64_t>(
            std::llround(0.25 * static_cast<double>(audio.sample_rate))),
    };
    if (vad_options.max_chunk_samples <= 0) {
      throw std::runtime_error(
          "SenseVoice audio_chunk_duration_sec produced an empty chunk");
    }
    const auto spans =
        engine::audio::plan_vad_audio_chunks(audio, vad_session(), vad_options);
    std::vector<AudioChunkPlan> plan;
    plan.reserve(spans.size());
    for (const auto &span : spans) {
      plan.push_back(AudioChunkPlan{span});
    }
    return plan;
  }
  const auto seconds =
      engine::audio::parse_audio_chunk_seconds_override(request.options)
          .value_or(kDefaultChunkSeconds);
  if (!(seconds > 0.0F)) {
    throw std::runtime_error(
        "SenseVoice audio_chunk_duration_sec must be positive");
  }
  const double sample_count =
      static_cast<double>(seconds) * static_cast<double>(audio.sample_rate);
  if (sample_count >=
      static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "SenseVoice audio_chunk_duration_sec is too large");
  }
  const int64_t samples = static_cast<int64_t>(std::llround(sample_count));
  if (samples <= 0) {
    throw std::runtime_error(
        "SenseVoice audio_chunk_duration_sec produced an empty chunk");
  }
  const auto chunks = engine::audio::plan_audio_chunks(
      frames, {samples, samples, engine::audio::AudioChunkPadMode::Zero,
               engine::audio::AudioChunkTailAlignment::Start, 0});
  std::vector<AudioChunkPlan> plan;
  plan.reserve(chunks.size());
  for (const auto &chunk : chunks) {
    plan.push_back({{chunk.output_start_sample,
                     chunk.output_start_sample + chunk.valid_samples}});
  }
  return plan;
}

runtime::IOfflineVoiceTaskSession &SenseAsrSession::vad_session() {
  if (vad_session_ == nullptr) {
    runtime::ModelLoadRequest load_request;
    load_request.model_path = vad_model_path_;
    vad_model_ =
        engine::models::silero_vad::load_silero_vad_model(load_request);
    auto session = vad_model_->create_task_session(
        runtime::TaskSpec{runtime::VoiceTaskKind::Vad,
                          runtime::RunMode::Offline},
        runtime::SessionOptions{options().backend, {}});
    auto *offline =
        dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
      throw std::runtime_error(
          "SenseVoice internal VAD session does not support offline execution");
    }
    session.release();
    vad_session_.reset(offline);
  }
  return *vad_session_;
}

runtime::TaskResult SenseAsrSession::run_single(const AsrRequest &request) {
  const auto wall_start = Clock::now();

  const auto resample_start = Clock::now();
  const auto mono =
      engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
          request.audio.samples, request.audio.sample_rate,
          request.audio.channels, assets_->config.frontend.sample_rate);
  const auto resample_end = Clock::now();

  const auto frontend_start = Clock::now();
  const auto features =
      frontend_.extract(mono, assets_->config.frontend.sample_rate);
  const auto frontend_end = Clock::now();

  const auto encoder_start = Clock::now();
  encoder_.set_query_tokens(query_tokens(request.transcription,
                                         assets_->config.encoder.query_tokens));
  const auto encoded = encoder_.encode(features);
  const auto encoder_end = Clock::now();

  const auto decode_start = Clock::now();
  const auto decoded =
      decode_ctc(encoded.logits, encoded.frames, encoded.vocab_size,
                 assets_->config.encoder.blank_id, assets_->config.vocab,
                 request.transcription.keep_tags);
  const auto decode_end = Clock::now();

  runtime::TaskResult result;
  result.text_output = runtime::Transcript{decoded.text, decoded.language};

  debug::timing_log_scalar("sense_asr.resample_ms",
                           debug::elapsed_ms(resample_start, resample_end));
  debug::timing_log_scalar("sense_asr.frontend_ms",
                           debug::elapsed_ms(frontend_start, frontend_end));
  debug::timing_log_scalar("sense_asr.encoder_ms",
                           debug::elapsed_ms(encoder_start, encoder_end));
  debug::timing_log_scalar("sense_asr.decode_ms",
                           debug::elapsed_ms(decode_start, decode_end));
  debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(wall_start));
  debug::trace_log_scalar("sense_asr.audio_input_frames",
                          audio_frame_count(request.audio));
  debug::trace_log_scalar("sense_asr.frontend_frames", features.frames);
  debug::trace_log_scalar("sense_asr.encoder_frames", encoded.frames);
  debug::trace_log_scalar("sense_asr.decoded_tokens", decoded.ids.size());
  return result;
}

void
SenseAsrSession::process_available_stream_chunks(bool final) {
  if (streaming_audio_.sample_rate <= 0 || streaming_audio_.channels <= 0) {
    return;
  }
  if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
    throw std::runtime_error(
        "SenseVoice streaming pending audio offset is out of range");
  }
  if (streaming_audio_.samples.size() %
              static_cast<size_t>(streaming_audio_.channels) !=
          0 ||
      streaming_audio_offset_values_ %
              static_cast<size_t>(streaming_audio_.channels) !=
          0) {
    throw std::runtime_error(
        "SenseVoice streaming pending audio has invalid channel layout");
  }
  const auto seconds = engine::audio::parse_audio_chunk_seconds_override(
                           streaming_request_.options)
                           .value_or(kDefaultStreamingWindowSeconds);
  if (!(seconds > 0.0F)) {
    throw std::runtime_error(
        "SenseVoice streaming audio_chunk_duration_sec must be positive");
  }
  const int64_t window_frames = static_cast<int64_t>(
      std::llround(static_cast<double>(seconds) *
                   static_cast<double>(streaming_audio_.sample_rate)));
  if (window_frames <= 0) {
    throw std::runtime_error("SenseVoice streaming audio_chunk_duration_sec "
                             "produced an empty chunk");
  }

  int64_t processed_chunks = 0;
  while (true) {
    const int64_t pending_frames = static_cast<int64_t>(
        (streaming_audio_.samples.size() - streaming_audio_offset_values_) /
        static_cast<size_t>(streaming_audio_.channels));
    if (pending_frames <= 0 || (!final && pending_frames < window_frames)) {
      break;
    }
    const int64_t take_frames =
        final ? std::min<int64_t>(pending_frames, window_frames)
              : window_frames;
    const size_t take_values =
        static_cast<size_t>(take_frames * streaming_audio_.channels);
    runtime::AudioBuffer chunk;
    chunk.sample_rate = streaming_audio_.sample_rate;
    chunk.channels = streaming_audio_.channels;
    const auto begin =
        streaming_audio_.samples.begin() +
        static_cast<std::ptrdiff_t>(streaming_audio_offset_values_);
    chunk.samples.assign(begin,
                         begin + static_cast<std::ptrdiff_t>(take_values));
    streaming_audio_offset_values_ += take_values;
    process_one_stream_chunk(chunk);
    ++streaming_windows_processed_;
    ++processed_chunks;
  }
  if (processed_chunks > 0) {
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size()) {
      streaming_audio_.samples.clear();
      streaming_audio_offset_values_ = 0;
    } else if (streaming_audio_offset_values_ > 1ull * 1024ull * 1024ull &&
               streaming_audio_offset_values_ * 2 >
                   streaming_audio_.samples.size()) {
      streaming_audio_.samples.erase(
          streaming_audio_.samples.begin(),
          streaming_audio_.samples.begin() +
              static_cast<std::ptrdiff_t>(streaming_audio_offset_values_));
      streaming_audio_offset_values_ = 0;
    }
  }
}

void
SenseAsrSession::process_one_stream_chunk(const runtime::AudioBuffer &audio) {
  runtime::TaskRequest item_request = streaming_request_;
  item_request.audio_input = audio;
  item_request.options["audio_chunk_mode"] = "none";
  auto item = run_single(make_request(item_request));

  if (!item.text_output.has_value() || item.text_output->text.empty()) {
    return;
  }
  auto transcript = committed_transcript();
  if (transcript.text.empty()) {
    transcript.text = item.text_output->text;
  } else {
    append_chunk_text(transcript.text, item.text_output->text);
  }
  if (transcript.language.empty()) {
    transcript.language = item.text_output->language;
  }
  publish_transcript(std::move(transcript));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sense_asr_loader() {
  runtime::SpecBackedVoiceModelConfig<SenseAsrAssets> config;
  config.family = "sense_asr";
  config.load_assets = [](const std::filesystem::path &model_path) {
    return load_sense_asr_assets(model_path);
  };
  config.create_session =
      [](const runtime::TaskSpec &task, const runtime::SessionOptions &options,
         std::shared_ptr<const SenseAsrAssets> assets,
         std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<SenseAsrSession>(
            task, options, std::move(assets), std::move(contract));
      };
  return runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::community_models::sense_asr

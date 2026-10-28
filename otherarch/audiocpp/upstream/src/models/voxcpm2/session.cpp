#include "engine/models/voxcpm2/session.h"

#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::voxcpm2 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 2048;

enum class ChunkStrategy {
  Continuation,
  Stateless,
};

ChunkStrategy parse_chunk_strategy(
    const std::unordered_map<std::string, std::string> &options) {
  const auto value =
      runtime::find_option(options, {"voxcpm2.chunk_strategy", "chunk_strategy"})
          .value_or("continuation");
  if (value == "continuation") {
    return ChunkStrategy::Continuation;
  }
  if (value == "stateless") {
    return ChunkStrategy::Stateless;
  }
  throw std::runtime_error(
      "VoxCPM2 chunk_strategy must be continuation or stateless");
}

const char *chunk_strategy_name(ChunkStrategy strategy) {
  switch (strategy) {
  case ChunkStrategy::Continuation:
    return "continuation";
  case ChunkStrategy::Stateless:
    return "stateless";
  }
  return "unknown";
}

std::shared_ptr<const VoxCPM2Assets>
require_assets(std::shared_ptr<const VoxCPM2Assets> assets,
               const runtime::SessionOptions & options) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM2 session requires assets");
  }
  const auto path = runtime::find_option(options.options, {"voxcpm2.lora"});
  if (!path || path->empty()) {
    return assets;
  }
  const auto adapter_path = std::filesystem::u8path(*path);
  auto file = adapter_path;
  if (io::is_existing_directory(adapter_path)) {
    const auto resolved = engine::assets::find_tensor_file(adapter_path, "lora_weights");
    if (!resolved) throw std::runtime_error("VoxCPM2 lora_weights checkpoint is missing in " + adapter_path.string());
    file = *resolved;
  }
  const auto config = io::json::parse_file(file.parent_path() / "lora_config.json");
  const auto & lora = config.require("lora_config");
  const auto rank = io::json::require_i64(lora, "r");
  if (rank <= 0) {
    throw std::runtime_error("VoxCPM2 LoRA rank must be positive");
  }
  engine::assets::WeightAdapterReadOptions read_options;
  read_options.scale_override = io::json::require_f32(lora, "alpha") / static_cast<float>(rank);
  read_options.strength = runtime::parse_finite_float_option(
      options.options, {"voxcpm2.lora_scale"}).value_or(1.0F);
  auto adapter = engine::assets::read_weight_adapter(file, read_options);
  for (const auto & [name, update] : adapter.updates) {
    const auto * factors = std::get_if<engine::assets::LowRankWeightUpdate>(&update.factors);
    if (factors == nullptr || factors->r != rank) {
      throw std::runtime_error("VoxCPM2 LoRA factors do not match lora_config.json: " + name);
    }
  }
  engine::assets::WeightAdapterBindings bindings;
  for (const auto & tensor : assets->model_weights->tensors()) {
    bindings.add(tensor.name, tensor.name);
  }
  auto plan = engine::assets::bind_weight_adapter(
      *assets->model_weights, std::move(adapter), bindings, engine::assets::WeightMergeMode::DeltaF32);
  auto adapted = std::make_shared<VoxCPM2Assets>(*assets);
  adapted->model_weights = engine::assets::make_weight_adapter_source(
      assets->model_weights, std::move(plan), "voxcpm2.lora");
  return adapted;
}

void reject_enabled_denoise(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  const auto match = runtime::find_option_match(options, keys);
  if (match.has_value() &&
      runtime::parse_bool_option(match->value, match->key)) {
    throw std::runtime_error(
        "VoxCPM2 denoise is disabled in this implementation");
  }
}

void reject_denoiser_option(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  if (runtime::find_option_match(options, keys).has_value()) {
    throw std::runtime_error(
        "VoxCPM2 denoise is disabled in this implementation");
  }
}

bool audio_buffer_equal(const runtime::AudioBuffer &lhs,
                        const runtime::AudioBuffer &rhs) {
  return lhs.sample_rate == rhs.sample_rate && lhs.channels == rhs.channels &&
         lhs.samples == rhs.samples;
}

bool optional_audio_equal(const std::optional<runtime::AudioBuffer> &lhs,
                          const std::optional<runtime::AudioBuffer> &rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs.has_value() || audio_buffer_equal(*lhs, *rhs);
}

std::optional<std::pair<std::string, std::string>>
split_leading_voice_tag(const std::string &text) {
  const std::string trimmed = engine::io::trim_ascii_whitespace(text);
  if (trimmed.empty() || trimmed.front() != '(') {
    return std::nullopt;
  }
  const size_t close = trimmed.find(')');
  if (close == std::string::npos || close + 1 >= trimmed.size()) {
    return std::nullopt;
  }
  const std::string tag =
      engine::io::trim_ascii_whitespace(trimmed.substr(0, close + 1));
  const std::string body =
      engine::io::trim_ascii_whitespace(trimmed.substr(close + 1));
  if (tag.empty() || body.empty()) {
    return std::nullopt;
  }
  return std::make_pair(tag, body);
}

std::vector<runtime::TaskRequest>
chunk_voxcpm2_request(const runtime::TaskRequest &request,
                      int64_t text_chunk_size,
                      engine::text::TextChunkMode text_chunk_mode) {
  if (!request.text_input.has_value() ||
      text_chunk_mode != engine::text::TextChunkMode::TagAware) {
    return runtime::chunk_text_request(request, text_chunk_size,
                                       text_chunk_mode);
  }
  const auto split = split_leading_voice_tag(request.text_input->text);
  if (!split.has_value()) {
    return runtime::chunk_text_request(request, text_chunk_size,
                                       text_chunk_mode);
  }
  const auto body_chunks = engine::text::split_text_chunks(
      split->second, text_chunk_size, text_chunk_mode);
  if (body_chunks.empty()) {
    return {request};
  }
  std::vector<runtime::TaskRequest> out;
  out.reserve(body_chunks.size());
  for (size_t i = 0; i < body_chunks.size(); ++i) {
    runtime::TaskRequest item = request;
    item.text_input->text =
        i == 0 ? split->first + " " + body_chunks[i] : body_chunks[i];
    out.push_back(std::move(item));
  }
  return out;
}

size_t prompt_cache_slots_from_options(
    const std::unordered_map<std::string, std::string> &options) {
  constexpr int64_t kDefaultPromptCacheSlots = 1;
  const int64_t slots =
      runtime::parse_i64_option(options, {"voxcpm2.prompt_cache_slots"})
          .value_or(kDefaultPromptCacheSlots);
  if (slots < 0) {
    throw std::runtime_error("voxcpm2.prompt_cache_slots must be non-negative");
  }
  return static_cast<size_t>(slots);
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type,
                             const char *option_name) {
  if (storage_type == engine::assets::TensorStorageType::Native ||
      storage_type == engine::assets::TensorStorageType::F32 ||
      storage_type == engine::assets::TensorStorageType::F16 ||
      storage_type == engine::assets::TensorStorageType::BF16 ||
      storage_type == engine::assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(std::string(option_name) +
                           " supports only native, f32, f16, bf16, and q8_0");
}

void parse_weight_type(
    const std::unordered_map<std::string, std::string> &options,
    const char *key, engine::assets::TensorStorageType &storage_type) {
  const auto it = options.find(key);
  if (it == options.end()) {
    return;
  }
  storage_type = engine::assets::parse_tensor_storage_type(it->second);
  validate_weight_storage(storage_type, key);
}

void validate_session_options(
    const std::unordered_map<std::string, std::string> &options) {
  for (const auto &[key, value] : options) {
    (void)value;
    if (key.rfind("voxcpm2.", 0) != 0) {
      continue;
    }
    if (key == "voxcpm2.lora" || key == "voxcpm2.lora_scale" ||
        key == "voxcpm2.weight_context_mb" ||
        key == "voxcpm2.text_embedding_graph_context_mb" ||
        key == "voxcpm2.text_embedding_graph_arena_mb" ||
        key == "voxcpm2.lm_step_graph_context_mb" ||
        key == "voxcpm2.lm_step_graph_arena_mb" ||
        key == "voxcpm2.projection_graph_context_mb" ||
        key == "voxcpm2.projection_graph_arena_mb" ||
        key == "voxcpm2.local_encoder_graph_context_mb" ||
        key == "voxcpm2.local_encoder_graph_arena_mb" ||
        key == "voxcpm2.dit_graph_context_mb" ||
        key == "voxcpm2.dit_graph_arena_mb" ||
        key == "voxcpm2.audiovae_weight_context_mb" ||
        key == "voxcpm2.audiovae_graph_context_mb" ||
        key == "voxcpm2.audiovae_graph_arena_mb" ||
        key == "voxcpm2.audiovae_encoder_graph_context_mb" ||
        key == "voxcpm2.audiovae_encoder_graph_arena_mb" ||
        key == "voxcpm2.audiovae_latent_capacity" ||
        key == "voxcpm2.audiovae_tile_frames" ||
        key == "voxcpm2.exec_plan" ||
        key == "voxcpm2.audiovae_encoder_sample_capacity" ||
        key == "voxcpm2.weight_type" ||
        key == "voxcpm2.audiovae_weight_type" ||
        key == "voxcpm2.prompt_cache_slots" ||
        key == "voxcpm2.mem_saver" ||
        key == "voxcpm2.perf_mode" ||
        key == "voxcpm2.denoise" || key == "voxcpm2.load_denoiser") {
      continue;
    }
    throw std::runtime_error("unknown VoxCPM2 session option: " + key);
  }
}

int64_t product(const std::vector<int64_t> &values) {
  int64_t out = 1;
  for (const int64_t value : values) {
    if (value <= 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE decoder rate is invalid");
    }
    out *= value;
  }
  return out;
}

} // namespace

bool VoxCPM2SessionBase::EncodedPromptCacheKeyEqual::operator()(
    const EncodedPromptCacheKey &lhs,
    const EncodedPromptCacheKey &rhs) const {
  return lhs.prompt_text == rhs.prompt_text &&
         optional_audio_equal(lhs.prompt_audio, rhs.prompt_audio) &&
         optional_audio_equal(lhs.reference_audio, rhs.reference_audio);
}

VoxCPM2SessionBase::VoxCPM2SessionBase(runtime::TaskSpec task,
                                       runtime::SessionOptions options,
                                       std::shared_ptr<const VoxCPM2Assets> assets)
    : RuntimeSessionBase(options), task_(task),
      assets_(require_assets(std::move(assets), options)),
      encoded_prompt_cache_(prompt_cache_slots_from_options(options.options)) {
  if (task_.mode != runtime::RunMode::Offline &&
      task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 only supports offline and streaming sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts &&
      task_.task != runtime::VoiceTaskKind::VoiceCloning &&
      task_.task != runtime::VoiceTaskKind::VoiceDesign) {
    throw std::runtime_error("VoxCPM2 supports TTS, voice cloning, and voice design");
  }

  reject_enabled_denoise(options.options, {"voxcpm2.denoise"});
  reject_enabled_denoise(options.options, {"voxcpm2.load_denoiser"});
  reject_denoiser_option(options.options, {"voxcpm2.denoiser"});
  validate_session_options(options.options);

  generator_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.weight_context_mb"},
      generator_config_.weight_context_bytes);
  generator_config_.text_embedding_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.text_embedding_graph_arena_mb", "voxcpm2.text_embedding_graph_context_mb"},
          generator_config_.text_embedding_graph_context_bytes);
  generator_config_.lm_step_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.lm_step_graph_arena_mb", "voxcpm2.lm_step_graph_context_mb"},
      generator_config_.lm_step_graph_context_bytes);
  generator_config_.projection_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.projection_graph_arena_mb", "voxcpm2.projection_graph_context_mb"},
          generator_config_.projection_graph_context_bytes);
  generator_config_.local_encoder_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.local_encoder_graph_arena_mb", "voxcpm2.local_encoder_graph_context_mb"},
          generator_config_.local_encoder_graph_context_bytes);
  generator_config_.dit_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.dit_graph_arena_mb", "voxcpm2.dit_graph_context_mb"},
      generator_config_.dit_graph_context_bytes);
  generator_config_.prompt_cache_slots = encoded_prompt_cache_.capacity();
  const auto perf_mode =
      runtime::find_option(options.options, {"voxcpm2.perf_mode"})
          .value_or("off");
  if (perf_mode != "off" && perf_mode != "optimized") {
    throw std::runtime_error("voxcpm2.perf_mode must be off or optimized");
  }
  generator_config_.optimized = perf_mode == "optimized";
  debug::trace_log_scalar("voxcpm2.perf_mode", perf_mode);
  decoder_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_weight_context_mb"},
      decoder_config_.weight_context_bytes);
  decoder_config_.graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_graph_arena_mb", "voxcpm2.audiovae_graph_context_mb"},
      decoder_config_.graph_context_bytes);
  decoder_config_.encoder_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_encoder_graph_arena_mb", "voxcpm2.audiovae_encoder_graph_context_mb"},
      decoder_config_.encoder_graph_context_bytes);
  decoder_config_.latent_frame_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm2.audiovae_latent_capacity"},
      decoder_config_.latent_frame_capacity);
  decoder_config_.tile_frames = runtime::parse_i64_option(
      options.options, {"voxcpm2.audiovae_tile_frames"})
      .value_or(decoder_config_.tile_frames);
  decoder_config_.encoder_sample_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm2.audiovae_encoder_sample_capacity"},
      decoder_config_.encoder_sample_capacity);
  parse_weight_type(options.options, "voxcpm2.weight_type",
                    generator_config_.weight_storage_type);
  parse_weight_type(options.options, "voxcpm2.audiovae_weight_type",
                    decoder_config_.weight_storage_type);
  if (const auto mem_saver =
          runtime::find_option(options.options, {"voxcpm2.mem_saver"})) {
    generator_config_.mem_saver =
        runtime::parse_bool_option(*mem_saver, "voxcpm2.mem_saver");
  }

  const auto exec_plan = runtime::parse_exec_plan(
      options.options, "voxcpm2.exec_plan",
      {{"audiovae", "ggml",
        {{"ggml", {execution_context().backend_type()}, {}, {}, {"threads"}}}}});
  const auto &audiovae_plan = exec_plan.at("audiovae");
  if (const auto threads = audiovae_plan.params.find("threads");
      threads != audiovae_plan.params.end()) {
    const auto &value = threads->second;
    if (!value.is_number() || !std::isfinite(value.as_number()) ||
        value.as_number() < 0 ||
        value.as_number() > std::numeric_limits<int>::max() ||
        std::floor(value.as_number()) != value.as_number()) {
      throw std::runtime_error(
          "voxcpm2.exec_plan.audiovae.params.threads must be a nonnegative int");
    }
    const auto audiovae_threads = static_cast<int>(value.as_i64());
    if (audiovae_threads > 0) {
      if (execution_context().backend_type() != core::BackendType::Cpu) {
        throw std::runtime_error(
            "voxcpm2.exec_plan.audiovae.params.threads > 0 requires a CPU session");
      }
      core::BackendConfig decoder_backend = execution_context().config();
      decoder_backend.threads = audiovae_threads;
      decoder_execution_context_ =
          std::make_unique<core::ExecutionContext>(decoder_backend);
    }
  }

  generator_ = std::make_unique<VoxCPM2FeatureGeneratorRuntime>(
      assets_, execution_context(), generator_config_);
  decoder_ = std::make_unique<VoxCPM2AudioVAEDecoderRuntime>(
      assets_,
      decoder_execution_context_ ? *decoder_execution_context_
                                 : execution_context(),
      decoder_config_);
}

VoxCPM2SessionBase::~VoxCPM2SessionBase() = default;

std::string VoxCPM2SessionBase::family_impl() const { return "voxcpm2"; }

runtime::VoiceTaskKind VoxCPM2SessionBase::task_kind_impl() const { return task_.task; }

runtime::RunMode VoxCPM2SessionBase::run_mode_impl() const { return task_.mode; }

void VoxCPM2SessionBase::prepare_impl(
    const runtime::SessionPreparationRequest &request) {
  (void)request;
  mark_prepared();
}

runtime::TaskResult VoxCPM2SessionBase::run_offline_request(const runtime::TaskRequest &request) {
  require_prepared("VoxCPM2 run");
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("VoxCPM2 run requires an offline session");
  }
  validate_request(request);
  auto release_runtime_memory = [this](VoxCPM2SessionBase *self) {
    if (self != nullptr) {
      self->release_request_runtime_memory();
    }
  };
  std::unique_ptr<VoxCPM2SessionBase, decltype(release_runtime_memory)>
      release_guard(generator_config_.mem_saver ? this : nullptr,
                    release_runtime_memory);

  const auto wall_start = Clock::now();
  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
  const auto text_chunk_mode =
      engine::text::parse_text_chunk_mode_override(request.options)
          .value_or(engine::text::TextChunkMode::TagAware);
  const auto chunk_requests =
      chunk_voxcpm2_request(request, text_chunk_size, text_chunk_mode);
  const auto chunk_strategy = parse_chunk_strategy(request.options);
  const auto generation_options = generation_options_from_request(request);
  const auto prompt_text =
      runtime::find_option(request.options, {"voxcpm2.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM2EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);

  runtime::TaskResult result;
  double generator_ms = 0.0;
  double decoder_ms = 0.0;
  runtime::AudioBuffer merged_audio;
  std::optional<VoxCPM2EncodedPrompt> continuation_prompt;
  const VoxCPM2EncodedPrompt *active_prompt = prompt;
  for (const auto & chunk_request : chunk_requests) {
    const auto generator_start = Clock::now();
    const auto generated = generator_->generate(
        chunk_request.text_input->text, active_prompt, generation_options);
    generator_ms += engine::debug::elapsed_ms(generator_start, Clock::now());

    const auto decoder_start = Clock::now();
    auto audio = decoder_->decode_features(generated.decode_features,
                                           generated.decode_patches);
    if (generated.decode_trim_patches > 0) {
      const int64_t trim_samples =
          generated.decode_trim_patches * assets_->config.patch_size *
          product(assets_->config.audio_vae.decoder_rates);
      if (trim_samples > static_cast<int64_t>(audio.samples.size())) {
        throw std::runtime_error(
            "VoxCPM2 decoded continuation trim exceeds audio length");
      }
      audio.samples.erase(
          audio.samples.begin(),
          audio.samples.begin() + static_cast<std::ptrdiff_t>(trim_samples));
    }
    decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    runtime::append_audio_buffer(merged_audio, audio);

    if (chunk_strategy == ChunkStrategy::Continuation &&
        chunk_requests.size() > 1 && generated.generated_patches > 0) {
      VoxCPM2EncodedPrompt next_prompt;
      if (prompt != nullptr) {
        next_prompt.reference_features = prompt->reference_features;
        next_prompt.reference_patches = prompt->reference_patches;
      }
      next_prompt.prompt_text = chunk_request.text_input->text;
      next_prompt.prompt_features = generated.generated_features;
      next_prompt.prompt_patches = generated.generated_patches;
      continuation_prompt = std::move(next_prompt);
      active_prompt = &*continuation_prompt;
    }
  }
  result.audio_output = std::move(merged_audio);

  const auto wall_end = Clock::now();
  debug::trace_log_scalar("voxcpm2.text_chunk_size", text_chunk_size);
  debug::trace_log_scalar("voxcpm2.text_chunk_mode",
                          engine::text::text_chunk_mode_name(text_chunk_mode));
  debug::trace_log_scalar("voxcpm2.text_chunk_count",
                          static_cast<int64_t>(chunk_requests.size()));
  debug::trace_log_scalar("voxcpm2.chunk_strategy",
                          std::string_view(chunk_strategy_name(chunk_strategy)));
  debug::timing_log_scalar("voxcpm2.generator_ms", generator_ms);
  debug::timing_log_scalar("voxcpm2.audiovae_decoder_ms", decoder_ms);
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

runtime::TaskResult
VoxCPM2SessionBase::run_streaming_request(
    const runtime::TaskRequest &request,
    const runtime::StreamEventCallback &stream_event_sink) {
  require_prepared("VoxCPM2 run_streaming");
  if (task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 run_streaming requires a streaming session");
  }
  validate_request(request);
  auto release_runtime_memory = [this](VoxCPM2SessionBase *self) {
    if (self != nullptr) {
      self->release_request_runtime_memory();
    }
  };
  std::unique_ptr<VoxCPM2SessionBase, decltype(release_runtime_memory)>
      release_guard(generator_config_.mem_saver ? this : nullptr,
                    release_runtime_memory);

  const auto wall_start = Clock::now();
  auto generation_options = generation_options_from_request(request);
  const auto prompt_text =
      runtime::find_option(request.options, {"voxcpm2.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM2EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);

  runtime::TaskResult result;
  runtime::StreamingAudioConfig stream_config;
  stream_config.frames_per_chunk = static_cast<size_t>(runtime::parse_positive_i64_option(
      request.options, {"stream_frames_per_event"}, 4));
  stream_config.policy = runtime::parse_streaming_audio_chunk_policy(
      runtime::find_option(request.options, {"stream_chunk_policy"}).value_or("grow"));
  stream_config.frame_width = static_cast<size_t>(assets_->config.patch_size * assets_->config.feat_dim);
  stream_config.left_context_frames = static_cast<size_t>(generation_options.stream_left_context);
  const size_t samples_per_patch = static_cast<size_t>(assets_->config.patch_size *
      product(assets_->config.audio_vae.decoder_rates));
  size_t emitted_chunks = 0;
  size_t emitted_patches = 0;
  auto publish = [&](const runtime::StreamEvent &audio_event) {
    emitted_patches += audio_event.audio_output->samples.size() / samples_per_patch;
    runtime::NamedAudioBuffer named;
    named.id = "chunk_" + std::to_string(emitted_chunks);
    named.audio = *audio_event.audio_output;
    named.meta.insert_or_assign(
        "generated_patches", std::to_string(emitted_patches));
    if (stream_event_sink) {
      runtime::StreamEvent event;
      event.named_audio_outputs.push_back(named);
      stream_event_sink(event);
    }
    result.named_audio_outputs.push_back(std::move(named));
    ++emitted_chunks;
  };
  auto decode = [&](const runtime::StreamingAudioWindow<float> &window) {
    auto audio = decoder_->decode_features(
        std::vector<float>(window.data, window.data + window.frames() * window.frame_width),
        static_cast<int64_t>(window.frames()));
    const size_t context_samples = window.left_frames * samples_per_patch;
    if (context_samples > audio.samples.size()) {
      throw std::runtime_error("VoxCPM2 streaming context trim exceeds chunk audio length");
    }
    audio.samples.erase(audio.samples.begin(), audio.samples.begin() + context_samples);
    return audio;
  };
  bool started = false;
  auto feed = [&](const std::vector<float> &reference,
                  const std::vector<float> &frames) {
    if (!started) {
      streaming_audio_.begin(stream_config, decode, publish, reference);
      started = true;
    }
    streaming_audio_.push(frames);
  };

  // With a dedicated AudioVAE backend, decoding (and publishing) moves to a
  // worker thread fed in patch order, so the generator never waits for the
  // decoder. Only the worker touches streaming_audio_ and the sink, and it is
  // joined before this function returns, so callbacks stay sequential and
  // never outlive the request.
  const bool async_decode = decoder_execution_context_ != nullptr;
  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  struct PendingFrames {
    std::vector<float> reference;
    std::vector<float> frames;
  };
  std::deque<PendingFrames> queue;
  bool generation_done = false;
  std::exception_ptr worker_error;
  std::thread worker;
  if (async_decode) {
    worker = std::thread([&] {
      try {
        for (;;) {
          PendingFrames item;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&] { return generation_done || !queue.empty(); });
            if (queue.empty()) {
              break;
            }
            item = std::move(queue.front());
            queue.pop_front();
          }
          feed(item.reference, item.frames);
        }
        result.audio_output = streaming_audio_.finish();
      } catch (...) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        worker_error = std::current_exception();
        queue.clear();
      }
    });
  }
  auto stop_worker = [&] {
    if (!worker.joinable()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      generation_done = true;
    }
    queue_cv.notify_one();
    worker.join();
  };
  struct WorkerJoin {
    decltype(stop_worker) &stop;
    ~WorkerJoin() { stop(); }
  } worker_join{stop_worker};

  bool queued_reference = false;
  auto emit_chunk = [&](const VoxCPM2StreamingChunk &chunk) {
    const auto new_begin = chunk.decode_features.begin() +
        static_cast<std::ptrdiff_t>(chunk.context_patches * stream_config.frame_width);
    std::vector<float> reference;
    // In async mode `started` belongs to the worker; the generator side only
    // tracks whether it already queued the first (reference-carrying) item.
    if (async_decode ? !queued_reference : !started) {
      reference.assign(chunk.decode_features.begin(), new_begin);
    }
    std::vector<float> frames(new_begin, chunk.decode_features.end());
    if (!async_decode) {
      feed(reference, frames);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (worker_error) {
        // The decoder or the sink failed (e.g. the client went away): stop
        // generating; the error is rethrown after the join.
        throw std::runtime_error("VoxCPM2 streaming decoder stopped");
      }
      queue.push_back({std::move(reference), std::move(frames)});
      queued_reference = true;
    }
    queue_cv.notify_one();
  };

  const auto generator_start = Clock::now();
  double decode_wait_ms = 0.0;
  try {
    (void)generator_->generate_streaming(request.text_input->text, prompt,
                                         generation_options, emit_chunk);
  } catch (...) {
    stop_worker();
    if (worker_error) {
      std::rethrow_exception(worker_error);
    }
    throw;
  }
  const auto generation_end = Clock::now();
  if (async_decode) {
    stop_worker();
    decode_wait_ms = engine::debug::elapsed_ms(generation_end, Clock::now());
    if (worker_error) {
      std::rethrow_exception(worker_error);
    }
  } else {
    result.audio_output = streaming_audio_.finish();
  }
  const auto generator_end = Clock::now();
  const double generator_with_callbacks_ms =
      engine::debug::elapsed_ms(generator_start, generator_end);

  const auto wall_end = Clock::now();
  const double decoder_ms = streaming_audio_.decode_ms();
  if (async_decode) {
    debug::timing_log_scalar(
        "voxcpm2.generator_ms",
        engine::debug::elapsed_ms(generator_start, generation_end));
    debug::timing_log_scalar("voxcpm2.audiovae_decode_wait_ms", decode_wait_ms);
  } else {
    debug::timing_log_scalar(
        "voxcpm2.generator_ms",
        std::max(0.0, generator_with_callbacks_ms - decoder_ms));
  }
  debug::timing_log_scalar("voxcpm2.generator_streaming_callbacks_ms",
                           generator_with_callbacks_ms);
  debug::timing_log_scalar("voxcpm2.audiovae_decoder_ms", decoder_ms);
  debug::timing_log_scalar("voxcpm2.streaming_chunks",
                           static_cast<double>(emitted_chunks));
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

void VoxCPM2SessionBase::release_request_runtime_memory() {
  if (!generator_config_.mem_saver) {
    return;
  }
  generator_->release_runtime_memory();
  decoder_->release_runtime_memory();
}

const VoxCPM2EncodedPrompt *VoxCPM2SessionBase::encoded_prompt_for_request(
    const std::optional<runtime::AudioBuffer> &prompt_audio,
    const std::string &prompt_text,
    const std::optional<runtime::AudioBuffer> &reference_audio) {
  if (!prompt_audio.has_value() && !reference_audio.has_value()) {
    return nullptr;
  }
  EncodedPromptCacheKey key;
  key.prompt_text = prompt_text;
  key.prompt_audio = prompt_audio;
  key.reference_audio = reference_audio;
  if (auto *cached = encoded_prompt_cache_.find(key)) {
    debug::trace_log_scalar("voxcpm2.prompt_cache.hit", 1);
    debug::trace_log_scalar("voxcpm2.prompt_cache.slots",
                            static_cast<int64_t>(
                                encoded_prompt_cache_.capacity()));
    debug::trace_log_scalar("voxcpm2.prompt_cache.entries",
                            static_cast<int64_t>(encoded_prompt_cache_.size()));
    debug::trace_log_scalar("voxcpm2.prompt_cache.evicted", 0);
    debug::timing_log_scalar("voxcpm2.prompt_encode_ms", 0.0);
    return &cached->encoded;
  }

  const auto encode_start = Clock::now();
  EncodedPromptCacheEntry entry;
  entry.encoded =
      decoder_->encode_prompt_audio(prompt_audio, prompt_text, reference_audio);
  const double encode_ms = engine::debug::elapsed_ms(encode_start);
  if (encoded_prompt_cache_.capacity() == 0) {
    uncached_encoded_prompt_ = std::move(entry);
    debug::trace_log_scalar("voxcpm2.prompt_cache.hit", 0);
    debug::trace_log_scalar("voxcpm2.prompt_cache.slots", 0);
    debug::trace_log_scalar("voxcpm2.prompt_cache.entries", 0);
    debug::trace_log_scalar("voxcpm2.prompt_cache.evicted", 0);
    debug::timing_log_scalar("voxcpm2.prompt_encode_ms", encode_ms);
    return &uncached_encoded_prompt_->encoded;
  }
  const bool will_evict =
      encoded_prompt_cache_.size() >= encoded_prompt_cache_.capacity();
  encoded_prompt_cache_.put(std::move(key), std::move(entry));
  EncodedPromptCacheKey lookup;
  lookup.prompt_text = prompt_text;
  lookup.prompt_audio = prompt_audio;
  lookup.reference_audio = reference_audio;
  auto *cached = encoded_prompt_cache_.find(lookup);
  if (cached == nullptr) {
    throw std::runtime_error("VoxCPM2 prompt cache insert failed");
  }
  debug::trace_log_scalar("voxcpm2.prompt_cache.hit", 0);
  debug::trace_log_scalar("voxcpm2.prompt_cache.slots",
                          static_cast<int64_t>(
                              encoded_prompt_cache_.capacity()));
  debug::trace_log_scalar("voxcpm2.prompt_cache.entries",
                          static_cast<int64_t>(encoded_prompt_cache_.size()));
  debug::trace_log_scalar("voxcpm2.prompt_cache.evicted", will_evict ? 1 : 0);
  debug::timing_log_scalar("voxcpm2.prompt_encode_ms",
                           encode_ms);
  return &cached->encoded;
}

VoxCPM2GenerationOptions VoxCPM2SessionBase::generation_options_from_request(
    const runtime::TaskRequest &request) const {
  VoxCPM2GenerationOptions options;
  if (const auto value = runtime::parse_i64_option(
          request.options, {"voxcpm2.min_tokens", "min_tokens"})) {
    options.min_tokens = *value;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options, {"max_tokens", "voxcpm2.max_tokens"})) {
    options.max_tokens = *value;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"num_inference_steps", "voxcpm2.num_inference_steps"})) {
    options.num_inference_steps = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options, {"guidance_scale", "voxcpm2.guidance_scale"})) {
    options.guidance_scale = *value;
  }
  if (const auto match = runtime::find_option_match(
          request.options, {"voxcpm2.retry_badcase", "retry_badcase"})) {
    options.retry_badcase =
        runtime::parse_bool_option(match->value, match->key);
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"voxcpm2.retry_badcase_max_times", "retry_badcase_max_times"})) {
    options.retry_badcase_max_times = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options, {"voxcpm2.retry_badcase_ratio_threshold",
                            "retry_badcase_ratio_threshold"})) {
    options.retry_badcase_ratio_threshold = *value;
  }
  if (const auto value = runtime::parse_u32_option(request.options,
                                                   {"voxcpm2.seed", "seed"})) {
    options.seed = *value;
  }
  options.cfm_noise_file =
      runtime::find_option(request.options,
                           {"voxcpm2.cfm_noise_file", "cfm_noise_file"})
          .value_or("");
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"voxcpm2.stream_left_context", "stream_left_context"})) {
    options.stream_left_context = *value;
  }
  if (options.stream_left_context < 0 ||
      options.stream_left_context > kVoxCPM2MaxStreamLeftContext) {
    throw std::runtime_error(
        "VoxCPM2 stream_left_context must be between 0 and " +
        std::to_string(kVoxCPM2MaxStreamLeftContext));
  }
  if (options.min_tokens < 0) {
    throw std::runtime_error("VoxCPM2 min_tokens must be non-negative");
  }
  if (options.max_tokens < 0) {
    throw std::runtime_error("VoxCPM2 max_tokens must be non-negative");
  }
  if (options.max_tokens == 0) {
    options.max_tokens = assets_->config.max_length;
  }
  if (options.min_tokens > options.max_tokens) {
    throw std::runtime_error("VoxCPM2 min_tokens must not exceed max_tokens");
  }
  if (options.max_tokens > assets_->config.max_length) {
    throw std::runtime_error(
        "VoxCPM2 max_tokens exceeds model config max_length");
  }
  if (options.num_inference_steps <= 0) {
    throw std::runtime_error(
        "VoxCPM2 num_inference_steps must be positive");
  }
  if (options.guidance_scale < 0.0F) {
    throw std::runtime_error("VoxCPM2 guidance_scale must be non-negative");
  }
  if (options.retry_badcase_max_times <= 0) {
    throw std::runtime_error(
        "VoxCPM2 retry_badcase_max_times must be positive");
  }
  if (options.retry_badcase_ratio_threshold <= 0.0F) {
    throw std::runtime_error(
        "VoxCPM2 retry_badcase_ratio_threshold must be positive");
  }
  reject_enabled_denoise(request.options, {"voxcpm2.denoise", "denoise"});
  reject_enabled_denoise(request.options,
                         {"voxcpm2.load_denoiser", "load_denoiser"});
  reject_denoiser_option(request.options, {"voxcpm2.denoiser", "denoiser"});
  return options;
}

void VoxCPM2SessionBase::validate_request(
    const runtime::TaskRequest &request) const {
  if (!request.text_input.has_value()) {
    throw std::runtime_error("VoxCPM2 requires text input");
  }
  if (request.text_input->text.empty()) {
    throw std::runtime_error("VoxCPM2 text input must not be empty");
  }
  if (request.voice.has_value()) {
    if (request.voice->style.has_value()) {
      throw std::runtime_error(
          "VoxCPM2 C++ session does not consume style conditions");
    }
    if (request.voice->speaker.has_value()) {
      const auto &speaker = *request.voice->speaker;
      if (speaker.cached_voice_id.has_value()) {
        throw std::runtime_error("VoxCPM2 C++ session requires speaker "
                                 "reference audio, not a cached voice id");
      }
      if (!speaker.audio.has_value()) {
        throw std::runtime_error(
            "VoxCPM2 C++ session speaker condition requires audio");
      }
    }
  }
  if (!request.input_artifacts.empty()) {
    throw runtime::InvalidRequestError(
        "VoxCPM2 C++ session does not consume input artifacts");
  }
}

VoxCPM2OfflineSession::VoxCPM2OfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VoxCPM2Assets> assets)
    : VoxCPM2SessionBase(task, std::move(options), std::move(assets)) {}

std::string VoxCPM2OfflineSession::family() const { return family_impl(); }

runtime::VoiceTaskKind VoxCPM2OfflineSession::task_kind() const {
  return task_kind_impl();
}

runtime::RunMode VoxCPM2OfflineSession::run_mode() const {
  return run_mode_impl();
}

void VoxCPM2OfflineSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  prepare_impl(request);
}

runtime::TaskResult
VoxCPM2OfflineSession::run(const runtime::TaskRequest &request) {
  return run_offline_request(request);
}

VoxCPM2StreamingSession::VoxCPM2StreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VoxCPM2Assets> assets)
    : VoxCPM2SessionBase(task, std::move(options), std::move(assets)) {}

std::string VoxCPM2StreamingSession::family() const { return family_impl(); }

runtime::VoiceTaskKind VoxCPM2StreamingSession::task_kind() const {
  return task_kind_impl();
}

runtime::RunMode VoxCPM2StreamingSession::run_mode() const {
  return run_mode_impl();
}

void VoxCPM2StreamingSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  prepare_impl(request);
}

runtime::TaskResult VoxCPM2StreamingSession::generate_stream(const runtime::TaskRequest &request) {
  return run_streaming_request(request, stream_event_sink());
}

} // namespace engine::models::voxcpm2


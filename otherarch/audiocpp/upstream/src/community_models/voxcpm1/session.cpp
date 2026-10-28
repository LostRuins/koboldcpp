#include "engine/community_models/voxcpm1/session.h"

#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chinese_variant.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::community_models::voxcpm1 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 2048;

std::shared_ptr<const VoxCPM1Assets>
require_assets(std::shared_ptr<const VoxCPM1Assets> assets,
               const runtime::SessionOptions & options) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM1 session requires assets");
  }
  const auto path = runtime::find_option(options.options, {"voxcpm1.lora"});
  if (!path || path->empty()) {
    return assets;
  }
  const auto adapter_path = std::filesystem::u8path(*path);
  auto file = adapter_path;
  if (io::is_existing_directory(adapter_path)) {
    const auto resolved = engine::assets::find_tensor_file(adapter_path, "lora_weights");
    if (!resolved) throw std::runtime_error("VoxCPM1 lora_weights checkpoint is missing in " + adapter_path.string());
    file = *resolved;
  }
  const auto config = io::json::parse_file(file.parent_path() / "lora_config.json");
  const auto & lora = config.require("lora_config");
  const auto rank = io::json::require_i64(lora, "r");
  if (rank <= 0) {
    throw std::runtime_error("VoxCPM1 LoRA rank must be positive");
  }
  engine::assets::WeightAdapterReadOptions read_options;
  read_options.scale_override = io::json::require_f32(lora, "alpha") / static_cast<float>(rank);
  read_options.strength = runtime::parse_finite_float_option(
      options.options, {"voxcpm1.lora_scale"}).value_or(1.0F);
  auto adapter = engine::assets::read_weight_adapter(file, read_options);
  for (const auto & [name, update] : adapter.updates) {
    const auto * factors = std::get_if<engine::assets::LowRankWeightUpdate>(&update.factors);
    if (factors == nullptr || factors->r != rank) {
      throw std::runtime_error("VoxCPM1 LoRA factors do not match lora_config.json: " + name);
    }
  }
  engine::assets::WeightAdapterBindings bindings;
  const std::regex layer_pattern(R"(((residual_lm|locenc|locdit)\.)?blk\.([0-9]+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight)");
  const std::unordered_map<std::string, std::string> prefixes{
      {"", "base_lm"}, {"residual_lm", "residual_lm"},
      {"locenc", "feat_encoder.encoder"}, {"locdit", "feat_decoder.estimator.decoder"}};
  const std::unordered_map<std::string, std::string> projections{
      {"attn_q", "self_attn.q_proj"}, {"attn_k", "self_attn.k_proj"},
      {"attn_v", "self_attn.v_proj"}, {"attn_output", "self_attn.o_proj"},
      {"ffn_gate", "mlp.gate_proj"}, {"ffn_up", "mlp.up_proj"}, {"ffn_down", "mlp.down_proj"}};
  for (const auto & tensor : assets->model_weights->tensors()) {
    bindings.add(tensor.name, tensor.name);
    std::smatch match;
    if (std::regex_match(tensor.name, match, layer_pattern)) {
      bindings.add(prefixes.at(match[2].str()) + ".layers." + match[3].str() + "." +
          projections.at(match[4].str()), tensor.name);
    }
  }
  for (const auto & entry : std::initializer_list<std::pair<const char *, const char *>>{
           {"enc_to_lm_proj", "proj.enc_to_lm.weight"},
           {"lm_to_dit_proj", "proj.lm_to_dit.weight"},
           {"res_to_dit_proj", "proj.res_to_dit.weight"},
           {"feat_encoder.in_proj", "locenc.in_proj.weight"},
           {"feat_decoder.estimator.in_proj", "locdit.in_proj.weight"},
           {"feat_decoder.estimator.cond_proj", "locdit.cond_proj.weight"},
           {"feat_decoder.estimator.out_proj", "locdit.out_proj.weight"}}) {
    bindings.add(entry.first, entry.second);
  }
  auto plan = engine::assets::bind_weight_adapter(
      *assets->model_weights, std::move(adapter), bindings, engine::assets::WeightMergeMode::DeltaF32);
  auto adapted = std::make_shared<VoxCPM1Assets>(*assets);
  adapted->model_weights = engine::assets::make_weight_adapter_source(
      assets->model_weights, std::move(plan), "voxcpm1.lora");
  return adapted;
}

void reject_enabled_denoise(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  const auto match = runtime::find_option_match(options, keys);
  if (match.has_value() &&
      runtime::parse_bool_option(match->value, match->key)) {
    throw std::runtime_error(
        "VoxCPM1 denoise is disabled in this implementation");
  }
}

void reject_denoiser_option(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  if (runtime::find_option_match(options, keys).has_value()) {
    throw std::runtime_error(
        "VoxCPM1 denoise is disabled in this implementation");
  }
}

std::unordered_map<std::string, std::string> normalize_v1_session_options(
    std::unordered_map<std::string, std::string> options) {
  return options;
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

size_t prompt_cache_slots_from_options(
    const std::unordered_map<std::string, std::string> &options) {
  constexpr int64_t kDefaultPromptCacheSlots = 1;
  const int64_t slots = runtime::parse_i64_option(
      options, {"voxcpm1.prompt_cache_slots", "voxcpm1.prompt_cache_slots"})
      .value_or(kDefaultPromptCacheSlots);
  if (slots < 0) {
    throw std::runtime_error("voxcpm1.prompt_cache_slots must be non-negative");
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
    if (key.rfind("voxcpm1.", 0) != 0) {
      continue;
    }
    if (key == "voxcpm1.lora" || key == "voxcpm1.lora_scale" ||
        key == "voxcpm1.weight_context_mb" ||
        key == "voxcpm1.text_embedding_graph_context_mb" ||
        key == "voxcpm1.text_embedding_graph_arena_mb" ||
        key == "voxcpm1.lm_step_graph_context_mb" ||
        key == "voxcpm1.lm_step_graph_arena_mb" ||
        key == "voxcpm1.projection_graph_context_mb" ||
        key == "voxcpm1.projection_graph_arena_mb" ||
        key == "voxcpm1.local_encoder_graph_context_mb" ||
        key == "voxcpm1.local_encoder_graph_arena_mb" ||
        key == "voxcpm1.dit_graph_context_mb" ||
        key == "voxcpm1.dit_graph_arena_mb" ||
        key == "voxcpm1.audiovae_weight_context_mb" ||
        key == "voxcpm1.audiovae_graph_context_mb" ||
        key == "voxcpm1.audiovae_graph_arena_mb" ||
        key == "voxcpm1.audiovae_encoder_graph_context_mb" ||
        key == "voxcpm1.audiovae_encoder_graph_arena_mb" ||
        key == "voxcpm1.audiovae_latent_capacity" ||
        key == "voxcpm1.audiovae_encoder_sample_capacity" ||
        key == "voxcpm1.weight_type" ||
        key == "voxcpm1.audiovae_weight_type" ||
        key == "voxcpm1.prompt_cache_slots" ||
        key == "voxcpm1.mem_saver" ||
        key == "voxcpm1.denoise" || key == "voxcpm1.load_denoiser") {
      continue;
    }
    throw std::runtime_error("unknown VoxCPM1 session option: " + key);
  }
}

int64_t product(const std::vector<int64_t> &values) {
  int64_t out = 1;
  for (const int64_t value : values) {
    if (value <= 0) {
      throw std::runtime_error("VoxCPM1 AudioVAE decoder rate is invalid");
    }
    out *= value;
  }
  return out;
}

std::optional<std::string> extract_request_language(const runtime::TaskRequest &request) {
  if (request.text_input.has_value() && !request.text_input->language.empty()) {
    return request.text_input->language;
  }
  if (request.voice.has_value() && request.voice->style.has_value() &&
      request.voice->style->language.has_value()) {
    return request.voice->style->language;
  }
  if (const auto lang = runtime::find_option(request.options, {"language", "lang"})) {
    return *lang;
  }
  return std::nullopt;
}

} // namespace

bool VoxCPM1SessionBase::EncodedPromptCacheKeyEqual::operator()(
    const EncodedPromptCacheKey &lhs,
    const EncodedPromptCacheKey &rhs) const {
  return lhs.prompt_text == rhs.prompt_text &&
         optional_audio_equal(lhs.prompt_audio, rhs.prompt_audio) &&
         optional_audio_equal(lhs.reference_audio, rhs.reference_audio);
}

VoxCPM1SessionBase::VoxCPM1SessionBase(runtime::TaskSpec task,
                                       runtime::SessionOptions options,
                                       std::shared_ptr<const VoxCPM1Assets> assets)
    : RuntimeSessionBase(options), task_(task),
      assets_(require_assets(std::move(assets), options)),
      encoded_prompt_cache_(prompt_cache_slots_from_options(options.options)) {
  if (task_.mode != runtime::RunMode::Offline &&
      task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        std::string("VoxCPM1") +
        " only supports offline and streaming sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts &&
      task_.task != runtime::VoiceTaskKind::VoiceCloning) {
    throw std::runtime_error(
        std::string("VoxCPM1") +
        " supports TTS and voice cloning");
  }

  options.options = normalize_v1_session_options(std::move(options.options));

  reject_enabled_denoise(options.options, {"voxcpm1.denoise"});
  reject_enabled_denoise(options.options, {"voxcpm1.load_denoiser"});
  reject_denoiser_option(options.options, {"voxcpm1.denoiser"});
  validate_session_options(options.options);

  generator_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.weight_context_mb"},
      generator_config_.weight_context_bytes);
  generator_config_.text_embedding_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm1.text_embedding_graph_arena_mb", "voxcpm1.text_embedding_graph_context_mb"},
          generator_config_.text_embedding_graph_context_bytes);
  generator_config_.lm_step_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.lm_step_graph_arena_mb", "voxcpm1.lm_step_graph_context_mb"},
      generator_config_.lm_step_graph_context_bytes);
  generator_config_.projection_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm1.projection_graph_arena_mb", "voxcpm1.projection_graph_context_mb"},
          generator_config_.projection_graph_context_bytes);
  generator_config_.local_encoder_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm1.local_encoder_graph_arena_mb", "voxcpm1.local_encoder_graph_context_mb"},
          generator_config_.local_encoder_graph_context_bytes);
  generator_config_.dit_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.dit_graph_arena_mb", "voxcpm1.dit_graph_context_mb"},
      generator_config_.dit_graph_context_bytes);
  generator_config_.prompt_cache_slots = encoded_prompt_cache_.capacity();
  decoder_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.audiovae_weight_context_mb"},
      decoder_config_.weight_context_bytes);
  decoder_config_.graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.audiovae_graph_arena_mb", "voxcpm1.audiovae_graph_context_mb"},
      decoder_config_.graph_context_bytes);
  decoder_config_.encoder_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm1.audiovae_encoder_graph_arena_mb", "voxcpm1.audiovae_encoder_graph_context_mb"},
      decoder_config_.encoder_graph_context_bytes);
  decoder_config_.latent_frame_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm1.audiovae_latent_capacity"},
      decoder_config_.latent_frame_capacity);
  decoder_config_.encoder_sample_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm1.audiovae_encoder_sample_capacity"},
      decoder_config_.encoder_sample_capacity);
  parse_weight_type(options.options, "voxcpm1.weight_type",
                    generator_config_.weight_storage_type);
  parse_weight_type(options.options, "voxcpm1.audiovae_weight_type",
                    decoder_config_.weight_storage_type);
  if (const auto mem_saver =
          runtime::find_option(options.options, {"voxcpm1.mem_saver"})) {
    generator_config_.mem_saver =
        runtime::parse_bool_option(*mem_saver, "voxcpm1.mem_saver");
  }

  generator_ = std::make_unique<VoxCPM1FeatureGeneratorRuntime>(
      assets_, execution_context(), generator_config_);
  decoder_ = std::make_unique<VoxCPM1AudioVAEDecoderRuntime>(
      assets_, execution_context(), decoder_config_);
}

VoxCPM1SessionBase::~VoxCPM1SessionBase() = default;

std::string VoxCPM1SessionBase::family_impl() const {
  return "voxcpm1";
}

runtime::VoiceTaskKind VoxCPM1SessionBase::task_kind_impl() const { return task_.task; }

runtime::RunMode VoxCPM1SessionBase::run_mode_impl() const { return task_.mode; }

void VoxCPM1SessionBase::prepare_impl(
    const runtime::SessionPreparationRequest &request) {
  (void)request;
  mark_prepared();
}

runtime::TaskResult VoxCPM1SessionBase::run_offline_request(const runtime::TaskRequest &request) {
  require_prepared("VoxCPM1 run");
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("VoxCPM1 run requires an offline session");
  }
  validate_request(request);
  auto release_runtime_memory = [this](VoxCPM1SessionBase *self) {
    if (self != nullptr) {
      self->release_request_runtime_memory();
    }
  };
  std::unique_ptr<VoxCPM1SessionBase, decltype(release_runtime_memory)>
      release_guard(this, release_runtime_memory);

  const auto wall_start = Clock::now();
  // Traditional -> Simplified conversion (OpenCC TSCharacters) unless language is Cantonese/Yue.
  // Mirrors audio8_tts fix: keep Traditional only for yue/cantonese/zh-HK/zh-MO.
  const auto language = extract_request_language(request);
  auto maybe_convert = [&](std::string_view text) -> std::string {
    return engine::text::maybe_convert_traditional_to_simplified_opt(text, language);
  };
  const auto prompt_text_raw =
      runtime::find_option(request.options, {"voxcpm1.prompt_text",
                                             "voxcpm1.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  const std::string prompt_text = maybe_convert(prompt_text_raw);
  // Apply conversion to TTS text before chunking so word-budget chunking operates on converted text.
  runtime::TaskRequest converted_request = request;
  if (converted_request.text_input.has_value()) {
    converted_request.text_input->text = maybe_convert(converted_request.text_input->text);
  }
  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
  const auto text_chunk_mode =
      engine::text::parse_text_chunk_mode_override(request.options)
          .value_or(engine::text::TextChunkMode::TagAware);
  const auto chunk_requests =
      runtime::chunk_text_request(converted_request, text_chunk_size, text_chunk_mode);
  const auto generation_options = generation_options_from_request(request);
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM1EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);
  // The encoded prompt (voice-clone conditioning) is cached as host-side
  // vectors; the VAE encoder graph that produced it is not needed again until
  // a different voice is encoded. Drop it before the generator runs so the
  // generator and decoder phases never coexist with the encoder graph.
  decoder_->release_encoder_graph();

  runtime::TaskResult result;
  double generator_ms = 0.0;
  double decoder_ms = 0.0;
  runtime::AudioBuffer merged_audio;
  const bool mem_saver = generator_config_.mem_saver;
  for (size_t chunk_index = 0; chunk_index < chunk_requests.size();
       ++chunk_index) {
    const auto &chunk_request = chunk_requests[chunk_index];
    const auto generator_start = Clock::now();
    const auto generated = generator_->generate(
        chunk_request.text_input->text, prompt, generation_options);
    generator_ms += engine::debug::elapsed_ms(generator_start, Clock::now());

    if (mem_saver && chunk_index + 1 == chunk_requests.size()) {
      // Last chunk: free the generator graphs before the AudioVAE decode so
      // the final decode peaks at weight + decoder graph instead of weight +
      // generator + decoder. Graphs rebuild lazily on the next request.
      generator_->release_runtime_memory();
    }

    const auto decoder_start = Clock::now();
    auto audio = decoder_->decode_features(generated.decode_features,
                                           generated.decode_patches);
    if (generated.decode_trim_patches > 0) {
      const int64_t trim_samples =
          generated.decode_trim_patches * assets_->config.patch_size *
          product(assets_->config.audio_vae.decoder_rates);
      if (trim_samples > static_cast<int64_t>(audio.samples.size())) {
        throw std::runtime_error(
            "VoxCPM1 decoded continuation trim exceeds audio length");
      }
      audio.samples.erase(
          audio.samples.begin(),
          audio.samples.begin() + static_cast<std::ptrdiff_t>(trim_samples));
    }
    decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    runtime::append_audio_buffer(merged_audio, audio);
  }
  result.audio_output = std::move(merged_audio);

  const auto wall_end = Clock::now();
  debug::trace_log_scalar("voxcpm1.text_chunk_size", text_chunk_size);
  debug::trace_log_scalar("voxcpm1.text_chunk_mode",
                          engine::text::text_chunk_mode_name(text_chunk_mode));
  debug::trace_log_scalar("voxcpm1.text_chunk_count",
                          static_cast<int64_t>(chunk_requests.size()));
  debug::timing_log_scalar("voxcpm1.generator_ms", generator_ms);
  debug::timing_log_scalar("voxcpm1.audiovae_decoder_ms", decoder_ms);
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

runtime::TaskResult
VoxCPM1SessionBase::run_streaming_request(
    const runtime::TaskRequest &request,
    const runtime::StreamEventCallback &stream_event_sink) {
  require_prepared("VoxCPM1 run_streaming");
  if (task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM1 run_streaming requires a streaming session");
  }
  validate_request(request);
  // Declared before release_guard on purpose: the decoder's streaming graph is
  // built against this state's tensors, and release_guard drops that graph via
  // decoder_->release_runtime_memory(). Reverse destruction order then frees
  // the graph before the state it points into, on return and on exceptions.
  AudioVAEStreamingDecodeState streaming_state;
  auto release_runtime_memory = [this](VoxCPM1SessionBase *self) {
    if (self != nullptr) {
      self->release_request_runtime_memory();
    }
  };
  std::unique_ptr<VoxCPM1SessionBase, decltype(release_runtime_memory)>
      release_guard(this, release_runtime_memory);

  const auto wall_start = Clock::now();
  const auto language = extract_request_language(request);
  auto maybe_convert = [&](std::string_view text) -> std::string {
    return engine::text::maybe_convert_traditional_to_simplified_opt(text, language);
  };
  auto generation_options = generation_options_from_request(request);
  const auto prompt_text_raw =
      runtime::find_option(request.options, {"voxcpm1.prompt_text",
                                             "voxcpm1.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  const std::string prompt_text = maybe_convert(prompt_text_raw);
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM1EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);
  // Same host-side clone-conditioning cache invariant as the offline path:
  // the encoder graph is only needed to produce the cached vectors, so free
  // it before the streaming generation starts.
  decoder_->release_encoder_graph();

  runtime::TaskResult result;
  runtime::AudioBuffer merged;
  merged.sample_rate = assets_->config.audio_vae.output_sample_rate;
  merged.channels = 1;
  double decoder_ms = 0.0;
  size_t emitted_chunks = 0;

  // Carry AudioVAE decoder convolution state across streamed patches so the
  // boundary between consecutive chunks stays continuous instead of resetting.
  bool use_streaming_decode = decoder_->supports_streaming_decode();
  if (use_streaming_decode) {
    use_streaming_decode =
        decoder_->initialize_streaming_decode_state(streaming_state);
  }

  auto emit_chunk = [&](const VoxCPM1StreamingChunk &chunk) {
    const auto decoder_start = Clock::now();
    auto audio = use_streaming_decode
        ? decoder_->decode_streaming_step(chunk.decode_features,
                                          streaming_state)
        : decoder_->decode_features(chunk.decode_features,
                                    chunk.decode_patches);
    decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    if (emitted_chunks == 0) {
      merged.sample_rate = audio.sample_rate;
      merged.channels = audio.channels;
    } else if (audio.sample_rate != merged.sample_rate ||
               audio.channels != merged.channels) {
      throw std::runtime_error(
          "VoxCPM1 streaming decoder chunk format changed");
    }
    merged.samples.insert(merged.samples.end(), audio.samples.begin(),
                          audio.samples.end());
    runtime::NamedAudioBuffer named;
    named.id = "chunk_" + std::to_string(emitted_chunks);
    named.audio = std::move(audio);
    named.meta.insert_or_assign(
        "generated_patches", std::to_string(chunk.generated_patches));
    if (stream_event_sink) {
      runtime::StreamEvent event;
      event.named_audio_outputs.push_back(named);
      stream_event_sink(event);
    }
    result.named_audio_outputs.push_back(std::move(named));
    ++emitted_chunks;
  };

  const auto generator_start = Clock::now();
  const std::string streaming_text = maybe_convert(request.text_input->text);
  (void)generator_->generate_streaming(streaming_text, prompt,
                                       generation_options, emit_chunk);
  const auto generator_end = Clock::now();
  const double generator_with_callbacks_ms =
      engine::debug::elapsed_ms(generator_start, generator_end);

  result.audio_output = std::move(merged);

  const auto wall_end = Clock::now();
  debug::timing_log_scalar(
      "voxcpm1.generator_ms",
      std::max(0.0, generator_with_callbacks_ms - decoder_ms));
  debug::timing_log_scalar("voxcpm1.generator_streaming_callbacks_ms",
                           generator_with_callbacks_ms);
  debug::timing_log_scalar("voxcpm1.audiovae_decoder_ms", decoder_ms);
  debug::timing_log_scalar("voxcpm1.streaming_chunks",
                           static_cast<double>(emitted_chunks));
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

void VoxCPM1SessionBase::release_request_runtime_memory() {
  // Only the cloned voice is cached across requests (host-side encoded
  // vectors in encoded_prompt_cache_). Every graph whose size follows the
  // request text/audio length (prompt prefill, VAE encoder/decoder) is
  // dropped so a long-lived server session returns to baseline VRAM and
  // reallocates fresh buffers sized to the next request.
  generator_->release_text_length_memory();
  decoder_->release_runtime_memory();
  if (generator_config_.mem_saver) {
    // mem_saver additionally drops the fixed-size generator graphs so the
    // session idles at weight-only VRAM.
    generator_->release_runtime_memory();
  }
}

const VoxCPM1EncodedPrompt *VoxCPM1SessionBase::encoded_prompt_for_request(
    const std::optional<runtime::AudioBuffer> &prompt_audio,
    const std::string &prompt_text,
    const std::optional<runtime::AudioBuffer> &reference_audio) {
  if (!prompt_audio.has_value() && !reference_audio.has_value()) {
    return nullptr;
  }
  // VoxCPM1 clones only via prompt-continuation mode (golden VoxCPM.cpp
  // uses --prompt-audio + --prompt-text); the V2 reference-mode path wraps
  // audio in tokens 103/104, which the V1 LM was never trained on. Route a
  // V1 reference audio through the prompt path so --voice-ref clones like
  // --audio.
  std::optional<runtime::AudioBuffer> effective_prompt_audio = prompt_audio;
  std::optional<runtime::AudioBuffer> effective_reference_audio = reference_audio;
  if (assets_->config.v1 && !effective_prompt_audio.has_value() &&
      effective_reference_audio.has_value()) {
    effective_prompt_audio = effective_reference_audio;
    effective_reference_audio.reset();
  }
  EncodedPromptCacheKey key;
  key.prompt_text = prompt_text;
  key.prompt_audio = effective_prompt_audio;
  key.reference_audio = effective_reference_audio;
  if (auto *cached = encoded_prompt_cache_.find(key)) {
    debug::trace_log_scalar("voxcpm1.prompt_cache.hit", 1);
    debug::trace_log_scalar("voxcpm1.prompt_cache.slots",
                            static_cast<int64_t>(
                                encoded_prompt_cache_.capacity()));
    debug::trace_log_scalar("voxcpm1.prompt_cache.entries",
                            static_cast<int64_t>(encoded_prompt_cache_.size()));
    debug::trace_log_scalar("voxcpm1.prompt_cache.evicted", 0);
    debug::timing_log_scalar("voxcpm1.prompt_encode_ms", 0.0);
    return &cached->encoded;
  }

  const auto encode_start = Clock::now();
  EncodedPromptCacheEntry entry;
  entry.encoded = decoder_->encode_prompt_audio(
      effective_prompt_audio, prompt_text, effective_reference_audio);
  const double encode_ms = engine::debug::elapsed_ms(encode_start);
  if (encoded_prompt_cache_.capacity() == 0) {
    uncached_encoded_prompt_ = std::move(entry);
    debug::trace_log_scalar("voxcpm1.prompt_cache.hit", 0);
    debug::trace_log_scalar("voxcpm1.prompt_cache.slots", 0);
    debug::trace_log_scalar("voxcpm1.prompt_cache.entries", 0);
    debug::trace_log_scalar("voxcpm1.prompt_cache.evicted", 0);
    debug::timing_log_scalar("voxcpm1.prompt_encode_ms", encode_ms);
    return &uncached_encoded_prompt_->encoded;
  }
  const bool will_evict =
      encoded_prompt_cache_.size() >= encoded_prompt_cache_.capacity();
  encoded_prompt_cache_.put(std::move(key), std::move(entry));
  EncodedPromptCacheKey lookup;
  lookup.prompt_text = prompt_text;
  lookup.prompt_audio = effective_prompt_audio;
  lookup.reference_audio = effective_reference_audio;
  auto *cached = encoded_prompt_cache_.find(lookup);
  if (cached == nullptr) {
    throw std::runtime_error("VoxCPM1 prompt cache insert failed");
  }
  debug::trace_log_scalar("voxcpm1.prompt_cache.hit", 0);
  debug::trace_log_scalar("voxcpm1.prompt_cache.slots",
                          static_cast<int64_t>(
                              encoded_prompt_cache_.capacity()));
  debug::trace_log_scalar("voxcpm1.prompt_cache.entries",
                          static_cast<int64_t>(encoded_prompt_cache_.size()));
  debug::trace_log_scalar("voxcpm1.prompt_cache.evicted", will_evict ? 1 : 0);
  debug::timing_log_scalar("voxcpm1.prompt_encode_ms",
                           encode_ms);
  return &cached->encoded;
}

VoxCPM1GenerationOptions VoxCPM1SessionBase::generation_options_from_request(
    const runtime::TaskRequest &request) const {
  VoxCPM1GenerationOptions options;
  bool min_tokens_explicit = false;
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"voxcpm1.min_tokens", "voxcpm1.min_tokens", "min_tokens"})) {
    options.min_tokens = *value;
    min_tokens_explicit = true;
  }
  // Set V1-specific default min_tokens if not explicitly provided
  if (!min_tokens_explicit && assets_->config.v1) {
    // Reference VoxCPM.cpp uses kMinLen=2 (stop may fire from the 4th patch);
    // the decode loop gates on `index > min_tokens`, which is the same check.
    // A higher floor (e.g. 20) forces ~1.6 s of audio and pads short
    // utterances with trailing silence after the stop predictor fires.
    options.min_tokens = 2;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"max_tokens", "voxcpm1.max_tokens", "voxcpm1.max_tokens"})) {
    options.max_tokens = *value;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"num_inference_steps", "voxcpm1.num_inference_steps",
           "voxcpm1.num_inference_steps"})) {
    options.num_inference_steps = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options,
          {"guidance_scale", "voxcpm1.guidance_scale",
           "voxcpm1.guidance_scale"})) {
    options.guidance_scale = *value;
  }
  bool retry_badcase_explicit = false;
  if (const auto match = runtime::find_option_match(
          request.options,
          {"voxcpm1.retry_badcase", "voxcpm1.retry_badcase",
           "retry_badcase"})) {
    options.retry_badcase =
        runtime::parse_bool_option(match->value, match->key);
    retry_badcase_explicit = true;
  }
  // Streaming emits decoded chunks to the client in real time, so a bad-case
  // retry (regenerate from scratch, discard earlier output) is impossible by
  // construction. The struct default retry_badcase=true exists for the
  // offline path; it must not leak into the streaming path and block every
  // streaming request. Relax it to false unless the caller explicitly asked
  // for retry, which the generator accepts and warns about.
  if (!retry_badcase_explicit && task_.mode == runtime::RunMode::Streaming) {
    options.retry_badcase = false;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"voxcpm1.retry_badcase_max_times",
           "voxcpm1.retry_badcase_max_times", "retry_badcase_max_times"})) {
    options.retry_badcase_max_times = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options,
          {"voxcpm1.retry_badcase_ratio_threshold",
           "voxcpm1.retry_badcase_ratio_threshold",
           "retry_badcase_ratio_threshold"})) {
    options.retry_badcase_ratio_threshold = *value;
  }
  if (const auto value = runtime::parse_u32_option(
          request.options, {"voxcpm1.seed", "voxcpm1.seed", "seed"})) {
    options.seed = *value;
  }
  options.cfm_noise_file =
      runtime::find_option(request.options,
                           {"voxcpm1.cfm_noise_file", "voxcpm1.cfm_noise_file",
                            "cfm_noise_file"})
          .value_or("");
  if (options.min_tokens < 0) {
    throw std::runtime_error("VoxCPM1 min_tokens must be non-negative");
  }
  if (options.max_tokens < 0) {
    throw std::runtime_error("VoxCPM1 max_tokens must be non-negative");
  }
  if (options.max_tokens == 0) {
    options.max_tokens = assets_->config.max_length;
  }
  if (options.min_tokens > options.max_tokens) {
    throw std::runtime_error("VoxCPM1 min_tokens must not exceed max_tokens");
  }
  if (options.max_tokens > assets_->config.max_length) {
    throw std::runtime_error(
        "VoxCPM1 max_tokens exceeds model config max_length");
  }
  if (options.num_inference_steps <= 0) {
    throw std::runtime_error(
        "VoxCPM1 num_inference_steps must be positive");
  }
  if (options.guidance_scale < 0.0F) {
    throw std::runtime_error("VoxCPM1 guidance_scale must be non-negative");
  }
  if (options.retry_badcase_max_times <= 0) {
    throw std::runtime_error(
        "VoxCPM1 retry_badcase_max_times must be positive");
  }
  if (options.retry_badcase_ratio_threshold <= 0.0F) {
    throw std::runtime_error(
        "VoxCPM1 retry_badcase_ratio_threshold must be positive");
  }
  reject_enabled_denoise(request.options,
                         {"voxcpm1.denoise", "voxcpm1.denoise", "denoise"});
  reject_enabled_denoise(request.options,
                         {"voxcpm1.load_denoiser", "voxcpm1.load_denoiser",
                          "load_denoiser"});
  reject_denoiser_option(request.options,
                         {"voxcpm1.denoiser", "voxcpm1.denoiser", "denoiser"});
  return options;
}

void VoxCPM1SessionBase::validate_request(
    const runtime::TaskRequest &request) const {
  if (!request.text_input.has_value()) {
    throw std::runtime_error("VoxCPM1 requires text input");
  }
  if (request.text_input->text.empty()) {
    throw std::runtime_error("VoxCPM1 text input must not be empty");
  }
  if (request.voice.has_value()) {
    if (request.voice->style.has_value()) {
      throw std::runtime_error(
          "VoxCPM1 C++ session does not consume style conditions");
    }
    if (request.voice->speaker.has_value()) {
      const auto &speaker = *request.voice->speaker;
      if (speaker.cached_voice_id.has_value()) {
        throw std::runtime_error("VoxCPM1 C++ session requires speaker "
                                 "reference audio, not a cached voice id");
      }
      if (!speaker.audio.has_value()) {
        throw std::runtime_error(
            "VoxCPM1 C++ session speaker condition requires audio");
      }
    }
  }
  if (!request.input_artifacts.empty()) {
    throw runtime::InvalidRequestError(
        "VoxCPM1 C++ session does not consume input artifacts");
  }
}

VoxCPM1OfflineSession::VoxCPM1OfflineSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VoxCPM1Assets> assets)
    : VoxCPM1SessionBase(task, std::move(options), std::move(assets)) {}

std::string VoxCPM1OfflineSession::family() const { return family_impl(); }

runtime::VoiceTaskKind VoxCPM1OfflineSession::task_kind() const {
  return task_kind_impl();
}

runtime::RunMode VoxCPM1OfflineSession::run_mode() const {
  return run_mode_impl();
}

void VoxCPM1OfflineSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  prepare_impl(request);
}

runtime::TaskResult
VoxCPM1OfflineSession::run(const runtime::TaskRequest &request) {
  return run_offline_request(request);
}

VoxCPM1StreamingSession::VoxCPM1StreamingSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VoxCPM1Assets> assets)
    : VoxCPM1SessionBase(task, std::move(options), std::move(assets)) {}

std::string VoxCPM1StreamingSession::family() const { return family_impl(); }

runtime::VoiceTaskKind VoxCPM1StreamingSession::task_kind() const {
  return task_kind_impl();
}

runtime::RunMode VoxCPM1StreamingSession::run_mode() const {
  return run_mode_impl();
}

void VoxCPM1StreamingSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  prepare_impl(request);
}

runtime::StreamingPolicy VoxCPM1StreamingSession::streaming_policy() const {
  runtime::StreamingPolicy policy;
  policy.input = runtime::StreamingInputKind::None;
  policy.output = runtime::StreamingOutputKind::FinalResult;
  return policy;
}

void VoxCPM1StreamingSession::start_stream(const runtime::TaskRequest &request) {
  reset();
  result_ = run_streaming_request(request, stream_event_sink_);
  started_ = true;
}

void VoxCPM1StreamingSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
  stream_event_sink_ = std::move(sink);
}

std::optional<runtime::StreamEvent> VoxCPM1StreamingSession::next_stream_event() {
  if (!started_) {
    throw std::runtime_error("VoxCPM1 streaming has not been started");
  }
  if (next_chunk_index_ >= result_.named_audio_outputs.size()) {
    return std::nullopt;
  }
  const auto & named = result_.named_audio_outputs[next_chunk_index_++];
  runtime::StreamEvent event;
  event.named_audio_outputs.push_back(named);
  return event;
}

runtime::TaskResult VoxCPM1StreamingSession::finish_stream() {
  if (!started_) {
    throw std::runtime_error("VoxCPM1 streaming has not been started");
  }
  started_ = false;
  next_chunk_index_ = 0;
  return std::move(result_);
}

void VoxCPM1StreamingSession::reset() {
  result_ = runtime::TaskResult{};
  next_chunk_index_ = 0;
  started_ = false;
}

runtime::StreamEvent VoxCPM1StreamingSession::process_audio_chunk(
    const runtime::AudioChunk &chunk) {
  (void)chunk;
  throw std::runtime_error("VoxCPM1 streaming does not consume audio chunks");
}

runtime::TaskResult VoxCPM1StreamingSession::finalize() {
  return finish_stream();
}

namespace {

runtime::ModelMetadata metadata_v1(const VoxCPM1Assets &assets) {
  runtime::ModelMetadata out;
  out.family = "voxcpm1";
  out.variant = assets.config.architecture;
  out.description = "VoxCPM1 loaded from GGUF assets.";
  return out;
}

runtime::CapabilitySet capabilities_v1(const VoxCPM1Assets &) {
  runtime::CapabilitySet out;
  out.supported_tasks = {
      {runtime::VoiceTaskKind::Tts,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
      {runtime::VoiceTaskKind::VoiceCloning,
       {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
  };
  out.languages = {"Auto"};
  out.supports_speaker_reference = true;
  return out;
}

runtime::ModelCliInterface cli_v1(const VoxCPM1Assets &) {
  runtime::ModelCliInterface out;
  out.request_options = {
      {"language", "code", "Text language hint used for text normalization; also exposed as --language."},
      {"text_chunk_mode", "word_budget|tag_aware|japanese|endline", "Text chunking mode; default tag_aware.", false, "tag_aware"},
      {"seed", "int", "Random seed for MiniCPM and diffusion sampling.", false},
      {"max_tokens", "int", "Maximum MiniCPM output tokens; default 4096. Zero selects model max_length; negative counts are rejected.", false, "4096"},
      {"min_tokens", "int", "Minimum MiniCPM output tokens before an EOS stop is honored; default 2.", false, "2"},
      {"num_inference_steps", "int", "CFM diffusion sampling steps; default 10. Must be positive.", false, "10"},
      {"guidance_scale", "float", "CFM classifier-free guidance rate. Must be nonnegative. Default: 2.", false, "2"},
      {"retry_badcase", "bool", "Retry when generation is detected as a bad case; defaults to true offline and false in streaming. Streaming ignores an explicit true value with a warning.", false},
      {"retry_badcase_max_times", "int", "Maximum total generation attempts when retrying bad cases, including the initial attempt; default 3. Must be positive.", false, "3"},
      {"retry_badcase_ratio_threshold", "float", "Bad-case ratio threshold for retry decisions. Must be positive."},
      {"prompt_text", "string", "Text prompt for prompt-continuation voice cloning.", false},
      {"reference_text", "string", "Alias for prompt_text; text prompt for prompt-continuation voice cloning.", false},
      {"text_chunk_size", "int", "Maximum characters per offline text chunk; must be positive. Does not control streaming chunk cadence. Default: 2048.", false, "2048", "1"},
  };
  out.session_options = {
      {"voxcpm1.lora", "path", "Native LoRA directory or checkpoint file with adjacent lora_config.json; merged into generator weights at session creation.", false},
      {"voxcpm1.lora_scale", "float", "Multiplier for the adapter's configured alpha/r scaling; default 1.", false, "1"},
      {"voxcpm1.mem_saver", "bool", "Use tighter graph workspaces and release request runtime graphs; default false.", false, "false"},
      {"voxcpm1.prompt_cache_slots", "int", "Prompt and prompt-audio embedding cache slots; default 1.", false, "1"},
      {"voxcpm1.weight_type", "native|f32|f16|bf16|q8_0", "Model weight storage type.", false, "native"},
      {"voxcpm1.audiovae_weight_type", "native|f32|f16|bf16|q8_0", "AudioVAE weight storage type.", false, "native"},
      {"voxcpm1.weight_context_mb", "int", "Model weight graph context size in MB.", false},
      {"voxcpm1.text_embedding_graph_arena_mb", "int", "Text embedding host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.lm_step_graph_arena_mb", "int", "LM step host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.projection_graph_arena_mb", "int", "Projection host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.local_encoder_graph_arena_mb", "int", "Local encoder host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.dit_graph_arena_mb", "int", "DiT estimator host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.audiovae_weight_context_mb", "int", "AudioVAE weight graph context size in MB.", false},
      {"voxcpm1.audiovae_graph_arena_mb", "int", "AudioVAE decoder host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.audiovae_encoder_graph_arena_mb", "int", "AudioVAE encoder host graph-metadata arena size in MiB, not device workspace.", false},
      {"voxcpm1.audiovae_latent_capacity", "int", "AudioVAE decoder latent frame capacity.", false},
      {"voxcpm1.audiovae_encoder_sample_capacity", "int", "AudioVAE encoder sample capacity.", false},
  };
  return out;
}

class VoxCPM1LoadedModel final : public runtime::ILoadedVoiceModel {
public:
  VoxCPM1LoadedModel(runtime::ModelMetadata metadata,
                     runtime::CapabilitySet capabilities,
                     std::shared_ptr<const VoxCPM1Assets> assets)
      : metadata_(std::move(metadata)),
        capabilities_(std::move(capabilities)),
        assets_(std::move(assets)) {}

  const runtime::ModelMetadata &metadata() const noexcept override {
    return metadata_;
  }

  const runtime::CapabilitySet &capabilities() const noexcept override {
    return capabilities_;
  }

  std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
      const runtime::TaskSpec &task,
      const runtime::SessionOptions &options) const override {
    if (task.task != runtime::VoiceTaskKind::Tts &&
        task.task != runtime::VoiceTaskKind::VoiceCloning) {
      throw std::runtime_error("VoxCPM1 supports TTS and voice cloning");
    }
    if (task.mode != runtime::RunMode::Offline &&
        task.mode != runtime::RunMode::Streaming) {
      throw std::runtime_error(
          "VoxCPM1 only supports offline and streaming sessions");
    }
    if (task.mode == runtime::RunMode::Streaming) {
      return std::make_unique<VoxCPM1StreamingSession>(
          task, options, assets_);
    }
    return std::make_unique<VoxCPM1OfflineSession>(task, options, assets_);
  }

private:
  runtime::ModelMetadata metadata_;
  runtime::CapabilitySet capabilities_;
  std::shared_ptr<const VoxCPM1Assets> assets_;
};

class VoxCPM1Loader final : public runtime::IVoiceModelLoader {
public:
  std::string family() const override { return "voxcpm1"; }

  runtime::CapabilitySet advertised_capabilities() const override {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        {runtime::VoiceTaskKind::VoiceCloning,
         {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_speaker_reference = true;
    return out;
  }

  std::string advertised_instructions_policy() const override {
    return "text_prefix";
  }

  bool can_load(const runtime::ModelLoadRequest &request) const override {
    try {
      (void)engine::model_spec::load_resource_bundle(
          request.model_path,
          engine::model_spec::default_spec_path(family()));
      return !request.family_hint.has_value() ||
             *request.family_hint == family();
    } catch (...) {
      return false;
    }
  }

  runtime::ModelInspection inspect(
      const runtime::ModelLoadRequest &request) const override {
    const auto assets = load_voxcpm1_assets(request.model_path);
    runtime::ModelInspection inspection;
    inspection.model_root = assets->resources.model_root();
    inspection.metadata = metadata_v1(*assets);
    inspection.capabilities = capabilities_v1(*assets);
    inspection.cli = cli_v1(*assets);
    const auto spec_path = engine::model_spec::default_spec_path(family());
    inspection.discovered_configs =
        runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
    inspection.discovered_weights =
        runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
    return inspection;
  }

  std::unique_ptr<runtime::ILoadedVoiceModel> load(
      const runtime::ModelLoadRequest &request) const override {
    auto assets = load_voxcpm1_assets(request.model_path);
    return std::make_unique<VoxCPM1LoadedModel>(
        metadata_v1(*assets), capabilities_v1(*assets), std::move(assets));
  }
};

}

std::shared_ptr<runtime::IVoiceModelLoader> make_voxcpm1_loader() {
  return std::make_shared<VoxCPM1Loader>();
}

} // namespace engine::community_models::voxcpm1


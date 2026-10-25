#include "engine/models/maya1/session.h"

#include "engine/framework/codecs/snac_decoder_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/maya1/generator.h"
#include "engine/models/maya1/tokenizer.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::maya1 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kFamily = "maya1";
constexpr size_t kArWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kArPrefillGraphBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kArDecodeGraphBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kCodecWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kCodecGraphContextBytes = 4ull * 1024ull * 1024ull;

codecs::SnacCodes unpack_snac_codes(const std::vector<int32_t> &tokens) {
  const size_t frames = tokens.size() / 7;
  codecs::SnacCodes out;
  out.codebooks.resize(3);
  out.codebooks[0].reserve(frames);
  out.codebooks[1].reserve(frames * 2);
  out.codebooks[2].reserve(frames * 4);
  for (size_t frame = 0; frame < frames; ++frame) {
    const auto code = [&](size_t slot) {
      return (tokens[frame * 7 + slot] - 128266) % 4096;
    };
    out.codebooks[0].push_back(code(0));
    out.codebooks[1].push_back(code(1));
    out.codebooks[1].push_back(code(4));
    out.codebooks[2].push_back(code(2));
    out.codebooks[2].push_back(code(3));
    out.codebooks[2].push_back(code(5));
    out.codebooks[2].push_back(code(6));
  }
  return out;
}

codecs::SnacDecoderConfig maya1_snac_config() {
  codecs::SnacDecoderConfig config;
  config.trace_name = "maya1.snac";
  return config;
}

codecs::SnacDecoderRuntimeOptions
maya1_snac_options(assets::TensorStorageType storage_type) {
  codecs::SnacDecoderRuntimeOptions options;
  options.weight_context_bytes = kCodecWeightContextBytes;
  options.graph_context_bytes = kCodecGraphContextBytes;
  options.weight_storage_type = storage_type;
  return options;
}

codecs::SnacDecoderWeightBinding maya1_snac_weight_binding() {
  codecs::SnacDecoderWeightBinding binding;
  binding.quantizer_prefix = "codec.quantizer.quantizers.";
  binding.decoder_prefix = "codec.decoder.model.";
  return binding;
}

Maya1GenerationOptions
generation_options(const runtime::TaskRequest &request,
                   const Maya1GenerationConfig &defaults) {
  Maya1GenerationOptions out;
  out.max_tokens = runtime::parse_i64_option(request.options, {"max_tokens"})
                       .value_or(defaults.max_tokens);
  out.min_tokens = runtime::parse_i64_option(request.options, {"min_tokens"})
                       .value_or(defaults.min_tokens);
  out.temperature =
      runtime::parse_finite_float_option(request.options, {"temperature"})
          .value_or(defaults.temperature);
  out.top_p = runtime::parse_finite_float_option(request.options, {"top_p"})
                  .value_or(defaults.top_p);
  out.repetition_penalty = runtime::parse_finite_float_option(
                               request.options, {"repetition_penalty"})
                               .value_or(defaults.repetition_penalty);
  out.seed = runtime::parse_u64_option(request.options, {"seed"})
                 .value_or(runtime::random_u64_seed());
  if (out.max_tokens <= 0 || out.min_tokens < 0 ||
      out.min_tokens > out.max_tokens) {
    throw std::runtime_error("Maya1 token limits are invalid");
  }
  if (out.temperature <= 0.0F || out.top_p <= 0.0F || out.top_p > 1.0F ||
      out.repetition_penalty <= 0.0F) {
    throw std::runtime_error("Maya1 sampling options are invalid");
  }
  return out;
}

std::unique_ptr<runtime::IVoiceTaskSession>
create_session(const runtime::TaskSpec &task,
               const runtime::SessionOptions &options,
               std::shared_ptr<const Maya1Assets> assets,
               std::shared_ptr<const model_spec::ModelContract> contract) {
  return std::make_unique<Maya1Session>(task, options, std::move(assets),
                                        std::move(contract));
}

} // namespace

class Maya1Session::Impl {
public:
  Impl(std::shared_ptr<const Maya1Assets> assets,
       core::ExecutionContext &execution, assets::TensorStorageType ar_storage,
       assets::TensorStorageType codec_storage)
      : tokenizer(assets),
        generator(assets, execution, kArPrefillGraphBytes, kArDecodeGraphBytes,
                  kArWeightContextBytes, ar_storage),
        codec(maya1_snac_config(), assets->codec_weights, execution,
              maya1_snac_options(codec_storage), maya1_snac_weight_binding()) {}

  Maya1Tokenizer tokenizer;
  Maya1LlamaGenerator generator;
  codecs::SnacDecoderRuntime codec;
};

Maya1Session::Maya1Session(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const Maya1Assets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)),
      contract_(std::move(contract)) {
  if (assets_ == nullptr || contract_ == nullptr) {
    throw std::runtime_error("Maya1 requires assets and a model contract");
  }
  runtime::validate_spec_backed_session_options(options, *contract_, kFamily,
                                                "Maya1");
  if ((task_.task != runtime::VoiceTaskKind::Tts &&
       task_.task != runtime::VoiceTaskKind::VoiceDesign) ||
      task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Maya1 currently implements offline TTS and voice design");
  }
  auto shared_storage = assets::TensorStorageType::Native;
  if (const auto value =
          runtime::find_option(options.options, {"maya1.weight_type", "weight_type"})) {
    shared_storage = assets::parse_tensor_storage_type(*value);
  }
  auto ar_storage = shared_storage;
  if (const auto value =
          runtime::find_option(options.options, {"maya1.ar_weight_type", "ar_weight_type"})) {
    ar_storage = assets::parse_tensor_storage_type(*value);
  }
  auto codec_storage = shared_storage;
  if (const auto value =
          runtime::find_option(options.options, {"maya1.codec_weight_type", "codec_weight_type"})) {
    codec_storage = assets::parse_tensor_storage_type(*value);
  }
  if (codec_storage != assets::TensorStorageType::Native &&
      codec_storage != assets::TensorStorageType::F32) {
    throw std::runtime_error(
        "Maya1 codec_weight_type supports only native or f32; when setting "
        "weight_type for the AR model, set maya1.codec_weight_type=f32 explicitly");
  }
  impl_ = std::make_unique<Impl>(assets_, execution_context(), ar_storage,
                                 codec_storage);
}

Maya1Session::~Maya1Session() = default;

std::string Maya1Session::family() const { return std::string(kFamily); }

runtime::VoiceTaskKind Maya1Session::task_kind() const {
  return task_.task;
}

runtime::RunMode Maya1Session::run_mode() const { return task_.mode; }

void Maya1Session::prepare(const runtime::SessionPreparationRequest &request) {
  runtime::validate_spec_backed_request_options(request.options, *contract_,
                                                "Maya1");
  mark_prepared();
}

runtime::TaskResult Maya1Session::run(const runtime::TaskRequest &request) {
  const auto wall_start = Clock::now();
  require_prepared("Maya1 run");
  runtime::validate_spec_backed_request_options(request.options, *contract_,
                                                "Maya1");
  if (!request.text_input.has_value() || request.text_input->text.empty()) {
    throw std::runtime_error("Maya1 requires non-empty text");
  }
  const auto instruct = runtime::find_option(request.options, {"instruct"});
  if (!instruct.has_value() || instruct->empty()) {
    throw std::runtime_error("Maya1 requires an instruct voice description");
  }
  auto options = generation_options(request, assets_->generation);
  debug::trace_log_scalar("maya1.instruct", *instruct);
  debug::trace_log_scalar("maya1.seed", options.seed);
  debug::trace_log_scalar("maya1.max_tokens", options.max_tokens);
  debug::trace_log_scalar("maya1.min_tokens", options.min_tokens);
  debug::trace_log_scalar("maya1.temperature", options.temperature);
  debug::trace_log_scalar("maya1.top_p", options.top_p);
  debug::trace_log_scalar("maya1.repetition_penalty",
                          options.repetition_penalty);

  const auto chunk_size =
      text::parse_text_chunk_size_override(request.options).value_or(300);
  const auto chunk_mode = text::parse_text_chunk_mode_override(request.options)
                              .value_or(text::TextChunkMode::TagAware);
  auto chunks = runtime::chunk_text_request(request, chunk_size, chunk_mode);
  debug::trace_log_scalar("maya1.text_chunk_mode",
                          text::text_chunk_mode_name(chunk_mode));
  debug::trace_log_scalar("maya1.text_chunk_size", chunk_size);
  debug::trace_log_scalar("maya1.text_chunk_count",
                          static_cast<int64_t>(chunks.size()));
  std::vector<runtime::AudioBuffer> output_chunks;
  output_chunks.reserve(chunks.size());
  for (const auto &chunk : chunks) {
    const auto prompt =
        impl_->tokenizer.build_prompt(*instruct, chunk.text_input->text);
    auto generated = impl_->generator.generate(prompt, options);
    output_chunks.push_back(impl_->codec.decode(
        unpack_snac_codes(generated.snac_tokens), options.seed));
    ++options.seed;
  }

  runtime::AudioBuffer audio;
  for (const auto &chunk : output_chunks) {
    if (audio.samples.empty()) {
      audio = chunk;
    } else {
      runtime::append_audio_buffer(audio, chunk);
    }
  }
  runtime::TaskResult result;
  result.audio_output = std::move(audio);
  debug::timing_log_scalar("session.wall_ms",
                           debug::elapsed_ms(wall_start, Clock::now()));
  return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_maya1_loader() {
  runtime::SpecBackedVoiceModelConfig<Maya1Assets> config;
  config.family = kFamily;
  config.load_assets = load_maya1_assets;
  config.create_session = create_session;
  return runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::models::maya1

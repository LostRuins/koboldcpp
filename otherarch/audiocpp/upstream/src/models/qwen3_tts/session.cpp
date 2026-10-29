#include "engine/models/qwen3_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/rknn/engine.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"
#include "engine/models/qwen3_tts/prompt_tts_voice_design.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 8192;

// Text chunking splits the talker, not the vocoder: with `qwen3_tts.stitch_decode=true` every
// chunk's codes are concatenated and voiced in one decode pass. Voicing per chunk re-voices the
// whole voice-clone reference once per chunk (`decode_and_trim_reference` prepends it and trims it
// back off), pads every chunk's tail up to a full 64-token group, and zeroes the stateful
// conv/KV cache at every chunk boundary. Measured on RK3588, 300 Russian characters, voice-clone
// reference 22.5 s: voicing per chunk at `--text-chunk-size 60` runs the vocoder 7 times for
// 159.9 s, voicing the concatenated codes once runs it once for 57.5 s, and the unchunked path
// runs it once for 48.3 s. One decode call builds a vocoder graph sized by its frame count, so the
// stitched pass costs device memory proportional to the utterance (~0.5 MiB per speech frame on
// CUDA, measured on an RTX 4070 Ti) - the reason it is opt-in.
void append_speech_codes(Qwen3SpeechCodes & dst, const Qwen3SpeechCodes & src) {
    if (src.frames == 0) {
        return;
    }
    if (dst.frames == 0) {
        dst = src;
        return;
    }
    if (dst.code_groups != src.code_groups) {
        throw std::runtime_error("Qwen3 TTS text chunks produced incompatible code shapes");
    }
    dst.codes.insert(dst.codes.end(), src.codes.begin(), src.codes.end());
    dst.frames += src.frames;
}

runtime::AudioBuffer decode_stitched(const Qwen3SpeechTokenizerDecoderRuntime & decoder,
                                     const Qwen3SpeechCodes * reference_codes,
                                     const Qwen3SpeechCodes & codes) {
    return reference_codes != nullptr
        ? decoder.decode_and_trim_reference(*reference_codes, codes)
        : decoder.decode(codes);
}

std::shared_ptr<const Qwen3TTSAssets> require_assets(std::shared_ptr<const Qwen3TTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 TTS session requires assets");
    }
    return assets;
}

Qwen3TTSGenerationOptions generation_options_from_request_impl(
    const runtime::TaskRequest & request,
    const Qwen3TTSConfig & config) {
    Qwen3TTSGenerationOptions options;
    options.max_new_tokens = config.max_new_tokens;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("Qwen3 TTS max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::find_option(request.options, {"subtalker_do_sample"})) {
        options.subtalker_do_sample = runtime::parse_bool_option(*value, "subtalker_do_sample");
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"repetition_penalty"})) {
        options.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"subtalker_temperature"})) {
        options.subtalker_temperature = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"subtalker_top_k"})) {
        options.subtalker_top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"subtalker_top_p"})) {
        options.subtalker_top_p = *value;
    }
    if (options.do_sample && options.temperature <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS temperature must be positive when sampling");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("Qwen3 TTS top_k must be non-negative");
    }
    if (options.top_p < 0.0F || options.top_p > 1.0F) {
        throw std::runtime_error("Qwen3 TTS top_p must be in [0, 1]");
    }
    if (options.repetition_penalty <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS repetition_penalty must be positive");
    }
    if (options.subtalker_do_sample && options.subtalker_temperature <= 0.0F) {
        throw std::runtime_error("Qwen3 TTS subtalker_temperature must be positive when sampling");
    }
    if (options.subtalker_top_k < 0) {
        throw std::runtime_error("Qwen3 TTS subtalker_top_k must be non-negative");
    }
    if (options.subtalker_top_p < 0.0F || options.subtalker_top_p > 1.0F) {
        throw std::runtime_error("Qwen3 TTS subtalker_top_p must be in [0, 1]");
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    return options;
}

uint64_t fnv1a_mix(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_mix(hash, &bits, sizeof(bits));
    }
    return hash;
}

core::BackendConfig voice_prompt_backend_config(const runtime::SessionOptions & options) {
    core::BackendConfig config = options.backend;
    // Voice-clone prompt codes are discrete argmax outputs; keep this stage on CPU so CUDA TF32 math
    // cannot change the reference prompt that the main talker conditions on.
    config.type = core::BackendType::Cpu;
    return config;
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.mem_saver");
    }
    return false;
}

bool stitch_decode_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.stitch_decode"})) {
        return runtime::parse_bool_option(*value, "qwen3_tts.stitch_decode");
    }
    return false;
}

Qwen3TTSPerfMode perf_mode_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"qwen3_tts.perf_mode"})) {
        if (*value == "off" || *value == "standard") {
            return Qwen3TTSPerfMode::Standard;
        }
        if (*value == "flash_attention" || *value == "optimized") {
            return Qwen3TTSPerfMode::FlashAttention;
        }
        throw std::runtime_error("Invalid qwen3_tts.perf_mode: " + *value);
    }
    return Qwen3TTSPerfMode::Standard;
}

std::size_t voice_prompt_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 1;
    const int64_t slots = runtime::parse_i64_option(options.options, {"qwen3_tts.voice_prompt_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("qwen3_tts.voice_prompt_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

void validate_talker_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error("Qwen3 TTS talker_weight_type currently supports only native, f32, f16, bf16, and q8_0");
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

void validate_conv_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, and f16");
}

class TalkerCachedStepReleaseGuard {
public:
    TalkerCachedStepReleaseGuard(Qwen3TalkerStepRuntime * runtime, bool enabled)
        : runtime_(runtime), enabled_(enabled) {}

    ~TalkerCachedStepReleaseGuard() noexcept {
        try {
            release();
        } catch (...) {
            // Cleanup must not replace the request exception during stack unwinding.
        }
    }

    void release() {
        if (!enabled_ || released_ || runtime_ == nullptr) {
            return;
        }
        const auto release_start = Clock::now();
        const int64_t released_steps = runtime_->release_cached_step_graph();
        debug::timing_log_scalar(
            "qwen3_tts.talker.cached_step_release_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
        debug::timing_log_scalar("qwen3_tts.talker.cached_step_released_steps", released_steps);
        released_ = true;
    }

private:
    Qwen3TalkerStepRuntime * runtime_ = nullptr;
    bool enabled_ = false;
    bool released_ = false;
};

}  // namespace

Qwen3TTSGenerationOptions qwen3_tts_generation_options_from_request(
    const runtime::TaskRequest & request,
    const Qwen3TTSConfig & config) {
    return generation_options_from_request_impl(request, config);
}

bool Qwen3TTSSession::VoicePromptCacheKeyEqual::operator()(
    const VoicePromptCacheKey & lhs,
    const VoicePromptCacheKey & rhs) const noexcept {
    return lhs.reference_text == rhs.reference_text &&
        lhs.mode == rhs.mode &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

Qwen3TTSSession::Qwen3TTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3TTSAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(apply_qwen3_tts_lora(require_assets(std::move(assets)), options.options)),
      mem_saver_(mem_saver_from_options(options)),
      stitch_decode_(stitch_decode_from_options(options)),
      perf_mode_(perf_mode_from_options(options)),
      text_tokenizer_(assets_),
      talker_(assets_->config.talker),
      voice_prompt_context_(voice_prompt_backend_config(options)),
      voice_prompt_cache_(voice_prompt_cache_slots_from_options(options)) {
    const bool optimized = runtime::find_option(options.options, {"qwen3_tts.perf_mode"}).value_or("off") == "optimized";
    if (optimized) {
        conv_weight_storage_type_ = engine::assets::TensorStorageType::F16;
    }
    talker_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_graph_arena_mb"}, talker_graph_arena_bytes_);
    speech_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_encoder_graph_arena_mb"}, speech_encoder_graph_arena_bytes_);
    speech_decoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_graph_arena_mb"}, speech_decoder_graph_arena_bytes_);
    speaker_encoder_graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speaker_encoder_graph_arena_mb"}, speaker_encoder_graph_arena_bytes_);
    talker_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.talker_constant_context_mb"}, talker_constant_context_bytes_);
    code_predictor_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.code_predictor_constant_context_mb"}, code_predictor_constant_context_bytes_);
    speech_decoder_constant_context_bytes_ = runtime::parse_size_mb_option(
        options.options, {"qwen3_tts.speech_decoder_constant_context_mb"}, speech_decoder_constant_context_bytes_);
    if (const auto it = options.options.find("qwen3_tts.weight_type"); it != options.options.end()) {
        const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(storage_type, "qwen3_tts.weight_type");
        validate_talker_weight_storage(storage_type);
        talker_weight_storage_type_ = storage_type;
    }
    if (const auto it = options.options.find("qwen3_tts.conv_weight_type"); it != options.options.end()) {
        conv_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_conv_weight_storage(conv_weight_storage_type_, "qwen3_tts.conv_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.talker_weight_type"); it != options.options.end()) {
        talker_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_talker_weight_storage(talker_weight_storage_type_);
    }
    if (const auto it = options.options.find("qwen3_tts.speech_encoder_weight_type"); it != options.options.end()) {
        speech_encoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_encoder_weight_storage_type_, "qwen3_tts.speech_encoder_weight_type");
    }
    if (const auto it = options.options.find("qwen3_tts.speech_decoder_weight_type"); it != options.options.end()) {
        speech_decoder_weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_matmul_weight_storage(speech_decoder_weight_storage_type_, "qwen3_tts.speech_decoder_weight_type");
    }
    const auto exec_plan = runtime::parse_exec_plan(
        options.options,
        "qwen3_tts.exec_plan",
        {{"speech_decoder", "ggml",
          {{"ggml", {options.backend.type}, {}, {}},
           {"rknn", {}, {"main"}, {"first", "window"}}}}});
    const auto & decoder_plan = exec_plan.at("speech_decoder");
    if (decoder_plan.impl == "rknn") {
        const auto & model_dir = assets_->resources.model_directory();
        rknn_decoder_.model_path = engine::rknn::resolve_engine_path(
            decoder_plan.artifacts.at("main"), model_dir);
        if (const auto first = decoder_plan.artifacts.find("first"); first != decoder_plan.artifacts.end()) {
            rknn_decoder_.first_model_path = engine::rknn::resolve_engine_path(first->second, model_dir);
        }
        if (const auto window = decoder_plan.artifacts.find("window");
            window != decoder_plan.artifacts.end()) {
            rknn_decoder_.window_path = engine::rknn::resolve_engine_path(window->second, model_dir);
        }
    }
    for (const auto & [key, _] : options.options) {
        if (key.rfind("qwen3_tts.", 0) == 0 &&
            key != "qwen3_tts.talker_graph_arena_mb" &&
            key != "qwen3_tts.speech_encoder_graph_arena_mb" &&
            key != "qwen3_tts.speech_decoder_graph_arena_mb" &&
            key != "qwen3_tts.speaker_encoder_graph_arena_mb" &&
            key != "qwen3_tts.talker_constant_context_mb" &&
            key != "qwen3_tts.code_predictor_constant_context_mb" &&
            key != "qwen3_tts.speech_decoder_constant_context_mb" &&
            key != "qwen3_tts.weight_type" &&
            key != "qwen3_tts.lora" &&
            key != "qwen3_tts.lora_scale" &&
            key != "qwen3_tts.conv_weight_type" &&
            key != "qwen3_tts.talker_weight_type" &&
            key != "qwen3_tts.speech_encoder_weight_type" &&
            key != "qwen3_tts.speech_decoder_weight_type" &&
            key != "qwen3_tts.voice_prompt_cache_slots" &&
            key != "qwen3_tts.exec_plan" &&
            key != "qwen3_tts.perf_mode" &&
            key != "qwen3_tts.talker_kv_cache_alignment" &&
            key != "qwen3_tts.stream_decoder" &&
            key != "qwen3_tts.stitch_decode" &&
            key != "qwen3_tts.mem_saver") {
            throw std::runtime_error("unknown Qwen3 TTS session option: " + key);
        }
    }
    const int64_t talker_kv_cache_alignment = runtime::parse_i64_option(options.options, {"qwen3_tts.talker_kv_cache_alignment"}).value_or(optimized ? 256 : 1);
    if (talker_kv_cache_alignment <= 0) throw std::runtime_error("qwen3_tts.talker_kv_cache_alignment must be positive");
    debug::trace_log_scalar("qwen3_tts.talker_kv_cache_alignment", talker_kv_cache_alignment);
    talker_weights_ = talker_.create_weights_runtime(
        assets_,
        options.backend.type,
        options.backend.device,
        std::max(1, options.backend.threads),
        talker_graph_arena_bytes_,
        talker_constant_context_bytes_,
        code_predictor_constant_context_bytes_,
        talker_weight_storage_type_,
        perf_mode_,
        talker_kv_cache_alignment);
    talker_step_ = talker_.create_step_runtime(
        talker_weights_,
        assets_->config.talker.max_position_embeddings,
        assets_->config.max_new_tokens);
    const auto stream_decoder = runtime::find_option(options.options, {"qwen3_tts.stream_decoder"}).value_or(optimized ? "stateful" : "windowed");
    if (stream_decoder != "windowed" && stream_decoder != "stateful") {
        throw std::runtime_error("qwen3_tts.stream_decoder must be windowed or stateful");
    }
    debug::trace_log_scalar("qwen3_tts.stream_decoder", stream_decoder);
    speech_decoder_ = std::make_unique<Qwen3SpeechTokenizerDecoderRuntime>(
        assets_,
        execution_context(),
        speech_decoder_graph_arena_bytes_,
        speech_decoder_constant_context_bytes_,
        speech_decoder_weight_storage_type_,
        conv_weight_storage_type_,
        perf_mode_,
        rknn_decoder_,
        stream_decoder == "stateful");
    if (assets_->config.variant == Qwen3TTSVariant::Base &&
        task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("Qwen3 base TTS model supports TTS and voice cloning");
    }
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign &&
        task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceDesign) {
        throw std::runtime_error("Qwen3 voice design model supports TTS and voice design");
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice && task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Qwen3 custom voice model only supports the Tts task");
    }
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        speech_encoder_ = std::make_unique<Qwen3SpeechTokenizerEncoderRuntime>(
            assets_,
            voice_prompt_context_,
            speech_encoder_graph_arena_bytes_,
            speech_encoder_weight_storage_type_,
            conv_weight_storage_type_,
            perf_mode_);
        speaker_encoder_ = std::make_unique<Qwen3TTSEcapaTdnnEncoderRuntime>(
            assets_,
            voice_prompt_context_,
            speaker_encoder_graph_arena_bytes_,
            conv_weight_storage_type_);
    }
    if (!rknn_decoder_.model_path.empty()) {
        // The NPU decoder voices the speech decoder, so the ggml decoder's weights are
        // never loaded. The speech-tokenizer tensor source is one resident blob opened for
        // the whole model, and a source frees only the ranges it hands out, so the
        // decoder's half of it would stay resident for the session's lifetime. Every
        // consumer of this source uploads its weights in its own constructor, so by here
        // the blob is dead.
        assets_->speech_tokenizer_weights->release_storage();
    }
}

std::string Qwen3TTSSession::family() const {
    return "qwen3_tts";
}

runtime::VoiceTaskKind Qwen3TTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3TTSSession::run_mode() const {
    return task_.mode;
}

void Qwen3TTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult Qwen3TTSSession::run(const runtime::TaskRequest & request) {
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 TTS run requires an offline session");
    }
    return synthesize(request, false);
}

runtime::TaskResult Qwen3TTSSession::generate_stream(const runtime::TaskRequest & request) {
    return synthesize(request, true);
}

void Qwen3TTSSession::reset_stream_state() {
    stream_audio_.reset();
}

runtime::AudioBuffer Qwen3TTSSession::generate_audio(
    const Qwen3TalkerPrefill & prefill,
    const Qwen3TTSGenerationOptions & options,
    const Qwen3SpeechCodes * reference_codes,
    int64_t stream_frames,
    runtime::StreamingAudioChunkPolicy stream_policy,
    double & talker_ms,
    double & decoder_ms) {
    if (stream_frames == 0) {
        const auto talker_start = Clock::now();
        const auto codes = talker_step_->generate(prefill, options, options.repetition_penalty);
        talker_ms += engine::debug::elapsed_ms(talker_start);
        const auto decoder_start = Clock::now();
        auto audio = reference_codes != nullptr
            ? speech_decoder_->decode_and_trim_reference(*reference_codes, codes.generated_codes)
            : speech_decoder_->decode(codes.generated_codes);
        decoder_ms += engine::debug::elapsed_ms(decoder_start);
        return audio;
    }

    runtime::StreamingAudioConfig config;
    config.policy = stream_policy;
    config.frames_per_chunk = static_cast<size_t>(stream_frames);
    config.frame_width = static_cast<size_t>(assets_->config.talker.num_code_groups);
    config.left_context_frames = Qwen3SpeechTokenizerDecoderRuntime::kLeftContextFrames;
    stream_audio_.begin(config,
        [&](const runtime::StreamingAudioWindow<int32_t> & window) {
            Qwen3SpeechCodes codes;
            codes.code_groups = static_cast<int64_t>(window.frame_width);
            codes.frames = static_cast<int64_t>(window.frames());
            codes.codes.assign(window.data, window.data + window.frames() * window.frame_width);
            return speech_decoder_->decode_stream_chunk(codes, window.left_frames, stream_frames);
        }, stream_event_sink(), reference_codes ? reference_codes->codes : std::vector<int32_t>{},
        // Stateful decoders carry history in a cache instead of a left
        // context window: the controller resets it per utterance and drains the
        // sub-group tail once at the end. No-ops on the windowed GGML path.
        [this] { speech_decoder_->reset_stream(); },
        [this] { return speech_decoder_->flush_stream(); });
    const auto talker_start = Clock::now();
    talker_step_->generate(prefill, options, options.repetition_penalty,
        [&](const std::vector<int32_t> & frame) { stream_audio_.push(frame); });
    talker_ms += engine::debug::elapsed_ms(talker_start) - stream_audio_.decode_ms() - stream_audio_.publish_ms();
    auto audio = stream_audio_.finish();
    decoder_ms += stream_audio_.decode_ms();
    if (audio.samples.empty()) {
        throw std::runtime_error("Qwen3 TTS generated no audio frames");
    }
    return audio;
}

runtime::TaskResult Qwen3TTSSession::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("Qwen3 TTS run");
    const auto wall_start = Clock::now();
    const int64_t stream_frames = streaming
        ? runtime::parse_int_option(request.options, {"stream_frames_per_event"}).value_or(12) : 0;
    if (streaming && stream_frames <= 0) {
        throw std::runtime_error("Qwen3 TTS stream_frames_per_event must be positive");
    }
    const auto stream_policy = streaming
        ? runtime::parse_streaming_audio_chunk_policy(
              runtime::find_option(request.options, {"stream_chunk_policy"}).value_or("grow"))
        : runtime::StreamingAudioChunkPolicy::Grow;
    TalkerCachedStepReleaseGuard release_talker_cached_step_graph(talker_step_.get(), mem_saver_);
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3TTSVoiceDesignPromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        Qwen3SpeechCodes codes;
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            const auto prefill_start = Clock::now();
            const auto prefill = prompt_builder.build_prefill(qwen_request);
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            if (stitch_decode_ && stream_frames == 0) {
                const auto talker_start = Clock::now();
                auto generated = talker_step_->generate(
                    prefill, qwen_request.generation, qwen_request.generation.repetition_penalty);
                talker_ms += engine::debug::elapsed_ms(talker_start);
                append_speech_codes(codes, generated.generated_codes);
            } else {
                runtime::append_audio_buffer(
                    merged_audio,
                    generate_audio(prefill, qwen_request.generation, nullptr, stream_frames, stream_policy, talker_ms, decoder_ms));
            }
        }
        if (stitch_decode_ && stream_frames == 0) {
            const auto decoder_start = Clock::now();
            merged_audio = decode_stitched(*speech_decoder_, nullptr, codes);
            decoder_ms += engine::debug::elapsed_ms(decoder_start);
        }
        release_talker_cached_step_graph.release();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.voice_design_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        Qwen3TTSCustomVoicePromptBuilder prompt_builder(
            text_tokenizer_,
            assets_->config.talker.max_position_embeddings,
            assets_->config.talker.max_position_embeddings);
        double prefill_ms = 0.0;
        double talker_ms = 0.0;
        double decoder_ms = 0.0;
        runtime::AudioBuffer merged_audio;
        Qwen3SpeechCodes codes;
        for (const auto & chunk_request : chunk_requests) {
            const Qwen3TTSRequest qwen_request = make_request(chunk_request);
            const auto prefill_start = Clock::now();
            const auto prefill = prompt_builder.build_prefill(qwen_request);
            prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
            if (stitch_decode_ && stream_frames == 0) {
                const auto talker_start = Clock::now();
                auto generated = talker_step_->generate(
                    prefill, qwen_request.generation, qwen_request.generation.repetition_penalty);
                talker_ms += engine::debug::elapsed_ms(talker_start);
                append_speech_codes(codes, generated.generated_codes);
            } else {
                runtime::append_audio_buffer(
                    merged_audio,
                    generate_audio(prefill, qwen_request.generation, nullptr, stream_frames, stream_policy, talker_ms, decoder_ms));
            }
        }
        if (stitch_decode_ && stream_frames == 0) {
            const auto decoder_start = Clock::now();
            merged_audio = decode_stitched(*speech_decoder_, nullptr, codes);
            decoder_ms += engine::debug::elapsed_ms(decoder_start);
        }
        release_talker_cached_step_graph.release();
        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        debug::timing_log_scalar("qwen3_tts.custom_voice_prefill_build_ms", prefill_ms);
        debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
        debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
        return result;
    }
    const Qwen3TTSRequest first_request = make_request(chunk_requests.front());
    if (!first_request.voice_clone.has_value()) {
        throw std::runtime_error("Qwen3 base TTS requires voice clone reference audio");
    }
    if (speech_encoder_ == nullptr || speaker_encoder_ == nullptr) {
        throw std::runtime_error("Qwen3 base TTS session is missing voice clone runtimes");
    }
    Qwen3TTSVoiceClonePromptBuilder prompt_builder(
        text_tokenizer_,
        *speech_encoder_,
        *speaker_encoder_,
        assets_->config.talker.max_position_embeddings);
    double prompt_ms = 0.0;
    double prefill_ms = 0.0;
    double talker_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    Qwen3SpeechCodes codes;
    std::optional<Qwen3SpeechCodes> reference_codes;
    for (const auto & chunk_request : chunk_requests) {
        const Qwen3TTSRequest qwen_request = make_request(chunk_request);
        const auto prompt_start = Clock::now();
        const auto & voice_prompt = resolve_voice_prompt(*qwen_request.voice_clone, prompt_builder);
        prompt_ms += engine::debug::elapsed_ms(prompt_start, Clock::now());
        const auto prefill_start = Clock::now();
        const auto prefill = prompt_builder.build_prefill(qwen_request, voice_prompt);
        prefill_ms += engine::debug::elapsed_ms(prefill_start, Clock::now());
        if (stitch_decode_ && stream_frames == 0) {
            const auto talker_start = Clock::now();
            auto generated = talker_step_->generate(
                prefill, qwen_request.generation, qwen_request.generation.repetition_penalty);
            talker_ms += engine::debug::elapsed_ms(talker_start);
            append_speech_codes(codes, generated.generated_codes);
            if (!reference_codes && voice_prompt.reference_codes) {
                reference_codes = *voice_prompt.reference_codes;
            }
        } else {
            runtime::append_audio_buffer(merged_audio, generate_audio(
                prefill, qwen_request.generation,
                voice_prompt.reference_codes ? &*voice_prompt.reference_codes : nullptr,
                stream_frames, stream_policy, talker_ms, decoder_ms));
        }
    }
    if (stitch_decode_ && stream_frames == 0) {
        // One decode pass over the concatenated codes: the reference clip is voiced once for the
        // whole utterance instead of once per text chunk, and the stateful vocoder cache runs
        // continuously across the chunk boundaries.
        const auto decoder_start = Clock::now();
        merged_audio = decode_stitched(*speech_decoder_,
                                       reference_codes ? &*reference_codes : nullptr,
                                       codes);
        decoder_ms += engine::debug::elapsed_ms(decoder_start);
    }
    release_talker_cached_step_graph.release();
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    debug::timing_log_scalar("qwen3_tts.voice_prompt_ms", prompt_ms);
    debug::timing_log_scalar("qwen3_tts.prefill_build_ms", prefill_ms);
    debug::timing_log_scalar("qwen3_tts.talker_ms", talker_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder_ms", decoder_ms);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

const Qwen3VoiceClonePrompt & Qwen3TTSSession::resolve_voice_prompt(
    const Qwen3VoiceCloneInput & input,
    const Qwen3TTSVoiceClonePromptBuilder & prompt_builder) {
    const uint64_t sample_count = static_cast<uint64_t>(input.reference_audio.samples.size());
    const uint64_t sample_hash = hash_audio_samples(input.reference_audio);
    VoicePromptCacheKey key;
    key.reference_text = input.reference_text;
    key.mode = input.mode;
    key.sample_rate = input.reference_audio.sample_rate;
    key.channels = input.reference_audio.channels;
    key.sample_count = sample_count;
    key.sample_hash = sample_hash;
    if (auto * cached = voice_prompt_cache_.find(key)) {
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 1);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return cached->prompt;
    }

    VoicePromptCacheEntry entry;
    entry.prompt = prompt_builder.build_voice_prompt(input);
    if (voice_prompt_cache_.capacity() == 0) {
        uncached_voice_prompt_ = std::move(entry);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", 0);
        debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", 0);
        return uncached_voice_prompt_->prompt;
    }
    const bool will_evict = voice_prompt_cache_.size() >= voice_prompt_cache_.capacity();
    voice_prompt_cache_.put(std::move(key), std::move(entry));
    auto * cached = voice_prompt_cache_.find(VoicePromptCacheKey{
        input.reference_text,
        input.mode,
        input.reference_audio.sample_rate,
        input.reference_audio.channels,
        sample_count,
        sample_hash,
    });
    if (cached == nullptr) {
        throw std::runtime_error("Qwen3 TTS voice prompt cache insert failed");
    }
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.hit", 0);
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.slots", static_cast<int64_t>(voice_prompt_cache_.capacity()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.entries", static_cast<int64_t>(voice_prompt_cache_.size()));
    debug::trace_log_scalar("qwen3_tts.voice_prompt_cache.evicted", will_evict ? 1 : 0);
    return cached->prompt;
}

Qwen3TTSRequest Qwen3TTSSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Qwen3 TTS requires text input");
    }
    Qwen3TTSRequest out;
    out.text = request.text_input->text;
    out.language = !request.text_input->language.empty() ? request.text_input->language : "Auto";
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        out.language = *language;
    }
    out.generation = qwen3_tts_generation_options_from_request(request, assets_->config);
    if (assets_->config.variant == Qwen3TTSVariant::Base) {
        const runtime::AudioBuffer * reference_audio = nullptr;
        if (request.voice.has_value()
            && request.voice->speaker.has_value()
            && request.voice->speaker->audio.has_value()) {
            reference_audio = &*request.voice->speaker->audio;
        } else if (request.audio_input.has_value()) {
            reference_audio = &*request.audio_input;
        }
        if (reference_audio != nullptr) {
            Qwen3VoiceCloneInput voice_clone;
            voice_clone.reference_audio = *reference_audio;
            if (const auto reference_text = runtime::find_option(
                    request.options,
                    {"reference_text"})) {
                voice_clone.reference_text = *reference_text;
            }
            bool x_vector_only = false;
            if (const auto value = runtime::find_option(
                    request.options,
                    {"x_vector_only_mode"})) {
                x_vector_only = runtime::parse_bool_option(*value, "x_vector_only_mode");
            }
            voice_clone.mode = x_vector_only
                ? Qwen3VoiceCloneMode::SpeakerEmbeddingOnly
                : Qwen3VoiceCloneMode::Icl;
            out.voice_clone = std::move(voice_clone);
        }
    } else if (assets_->config.variant == Qwen3TTSVariant::VoiceDesign) {
        Qwen3VoiceDesignInput voice_design;
        voice_design.instruct = runtime::find_option(
            request.options,
            {"instruction", "instruct"})
            .value_or("");
        if (voice_design.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                voice_design.instruct = tag->second;
            }
        }
        out.voice_design = std::move(voice_design);
    } else if (assets_->config.variant == Qwen3TTSVariant::CustomVoice) {
        Qwen3CustomVoiceInput custom_voice;
        custom_voice.speaker = runtime::find_option(request.options, {"voice_id", "speaker"}).value_or("");
        if (custom_voice.speaker.empty() && request.voice.has_value() && request.voice->speaker.has_value()) {
            custom_voice.speaker = request.voice->speaker->cached_voice_id.value_or("");
        }
        custom_voice.instruct = runtime::find_option(
            request.options,
            {"instruction", "instruct"})
            .value_or("");
        if (custom_voice.instruct.empty()
            && request.voice.has_value()
            && request.voice->style.has_value()) {
            const auto tag = request.voice->style->tags.find("instruct");
            if (tag != request.voice->style->tags.end()) {
                custom_voice.instruct = tag->second;
            }
        }
        out.custom_voice = std::move(custom_voice);
    }
    return out;
}

}  // namespace engine::models::qwen3_tts

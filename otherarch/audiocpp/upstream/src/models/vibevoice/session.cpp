#include "engine/models/vibevoice/session.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/vibevoice/lora.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxReferenceVoiceStates = 4;
constexpr int64_t kCudaVoicePromptMaxSeconds = 30;
constexpr int64_t kDefaultVoicePromptMaxSeconds = 10;

std::shared_ptr<const VibeVoiceAssets> require_assets(std::shared_ptr<const VibeVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice session requires assets");
    }
    return assets;
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

const runtime::SessionOptions & require_supported_backend_options(const runtime::SessionOptions & options) {
    if (options.backend.type != engine::core::BackendType::Cpu &&
        options.backend.type != engine::core::BackendType::Cuda &&
        options.backend.type != engine::core::BackendType::Hip &&
        options.backend.type != engine::core::BackendType::Vulkan &&
        options.backend.type != engine::core::BackendType::Metal) {
        throw std::runtime_error("VibeVoice session supports only CPU, CUDA, HIP, Vulkan, and Metal backends");
    }
    for (const auto & [key, value] : options.options) {
        if (key == "vibevoice.weight_type" ||
            key == "vibevoice.tokenizer_weight_type" ||
            key == "vibevoice.connector_weight_type" ||
            key == "vibevoice.decoder_weight_type" ||
            key == "vibevoice.diffusion_head_weight_type") {
            validate_weight_storage(engine::assets::parse_tensor_storage_type(value), key.c_str());
        } else if (key == "vibevoice.lora" || key == "vibevoice.lora_scale") {
            // Validated where the adapter is loaded.
        } else if (key.rfind("vibevoice.", 0) == 0) {
            throw std::runtime_error("unknown VibeVoice session option: " + key);
        }
    }
    return options;
}

VibeVoiceGenerationOptions generation_options_from_request(
    const runtime::TaskRequest & request,
    const VibeVoiceAssets & assets) {
    VibeVoiceGenerationOptions options;
    options.num_inference_steps = assets.config.diffusion_head.ddpm_num_inference_steps;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("VibeVoice max_tokens must be non-negative");
        }
        options.max_tokens = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"guidance_scale"})) {
        if (*value < 0.0F) {
            throw std::runtime_error("VibeVoice guidance_scale must be non-negative");
        }
        options.guidance_scale = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"max_length_times"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("VibeVoice max_length_times must be positive");
        }
        options.max_length_times = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        if (*value <= 0) {
            throw std::runtime_error("VibeVoice num_inference_steps must be positive");
        }
        options.num_inference_steps = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        if (*value <= 0.0F) {
            throw std::runtime_error("VibeVoice temperature must be positive");
        }
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        if (*value < 0) {
            throw std::runtime_error("VibeVoice top_k must be non-negative");
        }
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        if (*value <= 0.0F || *value > 1.0F) {
            throw std::runtime_error("VibeVoice top_p must be in (0, 1]");
        }
        options.top_p = *value;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(options.seed);
    options.prompt_noise_file =
        runtime::find_option(request.options, {"prompt_noise_file"})
            .value_or("");
    options.diffusion_noise_file =
        runtime::find_option(request.options, {"diffusion_noise_file"})
            .value_or("");
    return options;
}

std::vector<std::string> split_voice_sample_paths(std::string value) {
    std::vector<std::string> paths;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(',', start);
        auto item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) == 0;
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return std::isspace(ch) == 0;
        }).base(), item.end());
        if (!item.empty()) {
            paths.push_back(std::move(item));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return paths;
}

runtime::AudioBuffer read_voice_sample(const std::string & path) {
    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
        throw std::runtime_error("VibeVoice voice sample is empty: " + path);
    }
    return runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

runtime::AudioBuffer cap_voice_sample_duration(
    runtime::AudioBuffer audio,
    core::BackendType backend_type,
    const std::string & path) {
    const int64_t max_seconds = (backend_type == core::BackendType::Cuda || backend_type == core::BackendType::Hip)
        ? kCudaVoicePromptMaxSeconds
        : kDefaultVoicePromptMaxSeconds;
    const int64_t max_frames = static_cast<int64_t>(audio.sample_rate) * max_seconds;
    const auto max_samples = static_cast<size_t>(max_frames * static_cast<int64_t>(audio.channels));
    engine::debug::trace_log_scalar("vibevoice.reference_voice.prompt_path", path);
    engine::debug::trace_log_scalar("vibevoice.reference_voice.prompt_sample_rate", audio.sample_rate);
    engine::debug::trace_log_scalar("vibevoice.reference_voice.prompt_channels", audio.channels);
    engine::debug::trace_log_scalar(
        "vibevoice.reference_voice.prompt_input_samples",
        static_cast<int64_t>(audio.samples.size()));
    engine::debug::trace_log_scalar("vibevoice.reference_voice.prompt_max_seconds", max_seconds);
    if (audio.samples.size() > max_samples) {
        audio.samples.resize(max_samples);
    }
    engine::debug::trace_log_scalar(
        "vibevoice.reference_voice.prompt_used_samples",
        static_cast<int64_t>(audio.samples.size()));
    return audio;
}

VibeVoiceReferenceVoiceState reference_state_from_latents(const VibeVoiceTokenizerLatents & latents) {
    if (latents.frames <= 0 || latents.dim <= 0 ||
        latents.values.size() != static_cast<size_t>(latents.frames * latents.dim)) {
        throw std::runtime_error("VibeVoice reference voice acoustic encoder returned invalid shape");
    }
    VibeVoiceReferenceVoiceState state;
    state.acoustic_mean = latents.values;
    state.frames = latents.frames;
    state.dim = latents.dim;
    return state;
}

}  // namespace

VibeVoiceSession::VibeVoiceSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VibeVoiceAssets> assets)
    : runtime::RuntimeSessionBase(require_supported_backend_options(options)),
      task_(task),
      assets_(apply_vibevoice_finetune_options(require_assets(std::move(assets)), options.options)),
      text_tokenizer_(assets_),
      audio_tokenizer_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          256ull * 1024ull * 1024ull,
          128ull * 1024ull * 1024ull,
          options.options.find("vibevoice.tokenizer_weight_type") != options.options.end()
              ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.tokenizer_weight_type"))
              : (options.options.find("vibevoice.weight_type") != options.options.end()
                      ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.weight_type"))
                      : engine::assets::TensorStorageType::Native)),
      connector_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          64ull * 1024ull * 1024ull,
          32ull * 1024ull * 1024ull,
          options.options.find("vibevoice.connector_weight_type") != options.options.end()
              ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.connector_weight_type"))
              : (options.options.find("vibevoice.weight_type") != options.options.end()
                      ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.weight_type"))
                      : engine::assets::TensorStorageType::Native)),
      decoder_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          256ull * 1024ull * 1024ull,
          128ull * 1024ull * 1024ull,
          options.options.find("vibevoice.decoder_weight_type") != options.options.end()
              ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.decoder_weight_type"))
              : (options.options.find("vibevoice.weight_type") != options.options.end()
                      ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.weight_type"))
                      : engine::assets::TensorStorageType::Native)),
      diffusion_head_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          64ull * 1024ull * 1024ull,
          64ull * 1024ull * 1024ull,
          options.options.find("vibevoice.diffusion_head_weight_type") != options.options.end()
              ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.diffusion_head_weight_type"))
              : (options.options.find("vibevoice.weight_type") != options.options.end()
                      ? engine::assets::parse_tensor_storage_type(options.options.at("vibevoice.weight_type"))
                      : engine::assets::TensorStorageType::Native)) {}

std::string VibeVoiceSession::family() const {
    return "vibevoice";
}

runtime::VoiceTaskKind VibeVoiceSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VibeVoiceSession::run_mode() const {
    return task_.mode;
}

void VibeVoiceSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult VibeVoiceSession::run(const runtime::TaskRequest & request) {
    return synthesize(request, false);
}

runtime::TaskResult VibeVoiceSession::generate_stream(const runtime::TaskRequest & request) {
    return synthesize(request, true);
}

runtime::TaskResult VibeVoiceSession::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("VibeVoice run");
    const auto wall_start = Clock::now();
    auto vibevoice_request = make_request(request);
    std::function<void(const runtime::AudioBuffer &)> on_audio;
    if (streaming && stream_event_sink()) {
        on_audio = [this](const runtime::AudioBuffer & audio) {
            runtime::StreamEvent event;
            event.audio_output = audio;
            stream_event_sink()(event);
        };
    }
    auto result = generate_vibevoice(
        vibevoice_request,
        text_tokenizer_,
        audio_tokenizer_,
        connector_,
        decoder_,
        diffusion_head_,
        positive_decoder_cache_,
        negative_decoder_cache_,
        on_audio);
    runtime::TaskResult out;
    out.audio_output = std::move(result.audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return out;
}

VibeVoiceRequest VibeVoiceSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("VibeVoice requires non-empty text input");
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        throw std::runtime_error("VibeVoice C++ session does not consume style conditions");
    }
    if (!request.input_artifacts.empty()) {
        throw runtime::InvalidRequestError("VibeVoice C++ session does not consume input artifacts");
    }
    VibeVoiceRequest out;
    out.text = request.text_input->text;
    out.generation = generation_options_from_request(request, *assets_);
    std::vector<std::string> voice_sample_paths;
    if (const auto voice_samples = runtime::find_option(request.options, {"vibevoice.voice_samples", "voice_samples"})) {
        voice_sample_paths = split_voice_sample_paths(*voice_samples);
    }
    const bool has_voice_ref = request.voice.has_value() && request.voice->speaker.has_value();
    if (!voice_sample_paths.empty() && has_voice_ref) {
        throw std::runtime_error("VibeVoice request cannot combine voice_samples option with voice_ref");
    }
    if (!voice_sample_paths.empty()) {
        out.speakers = resolve_voice_sample_prompts(voice_sample_paths);
    }
    if (has_voice_ref) {
        const auto & speaker = *request.voice->speaker;
        if (speaker.cached_voice_id.has_value()) {
            throw std::runtime_error("VibeVoice C++ session requires speaker reference audio, not a cached voice id");
        }
        if (!speaker.audio.has_value()) {
            throw std::runtime_error("VibeVoice C++ session speaker condition requires audio");
        }
        out.speakers.push_back(VibeVoiceSpeakerPrompt{*speaker.audio});
    }
    return out;
}

std::vector<VibeVoiceSpeakerPrompt> VibeVoiceSession::resolve_voice_sample_prompts(
    const std::vector<std::string> & sample_paths) const {
    if (sample_paths.size() > kMaxReferenceVoiceStates) {
        throw std::runtime_error("VibeVoice 1.5B supports at most 4 reference speakers");
    }
    auto cached_speakers = [&]() {
        size_t total = 0;
        for (const auto & item : reference_voice_state_cache_) {
            total += item.states.size();
        }
        return total;
    };
    int64_t cache_evictions = 0;
    const auto found = std::find_if(
        reference_voice_state_cache_.begin(),
        reference_voice_state_cache_.end(),
        [&](const ReferenceVoiceStateCacheEntry & entry) {
            return entry.sample_paths == sample_paths;
        });
    if (found != reference_voice_state_cache_.end()) {
        std::vector<VibeVoiceSpeakerPrompt> prompts;
        prompts.reserve(found->states.size());
        for (size_t index = 0; index < found->states.size(); ++index) {
            VibeVoiceSpeakerPrompt prompt;
            prompt.audio = found->audio[index];
            prompt.reference_state = found->states[index];
            prompts.push_back(std::move(prompt));
        }
        auto entry = std::move(*found);
        reference_voice_state_cache_.erase(found);
        reference_voice_state_cache_.push_back(std::move(entry));
        engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_hits", static_cast<int64_t>(sample_paths.size()));
        engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_misses", 0);
        engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_evictions", cache_evictions);
        engine::debug::timing_log_scalar(
            "vibevoice.reference_voice.cache_entries",
            static_cast<int64_t>(reference_voice_state_cache_.size()));
        engine::debug::timing_log_scalar(
            "vibevoice.reference_voice.cached_speakers",
            static_cast<int64_t>(cached_speakers()));
        return prompts;
    }

    std::vector<runtime::AudioBuffer> audio;
    audio.reserve(sample_paths.size());
    for (const auto & path : sample_paths) {
        audio.push_back(cap_voice_sample_duration(read_voice_sample(path), options().backend.type, path));
    }
    auto acoustic_means = audio_tokenizer_.encode_acoustic_batch(audio);
    if (acoustic_means.size() != sample_paths.size()) {
        throw std::runtime_error("VibeVoice reference voice acoustic encoder returned unexpected batch size");
    }

    ReferenceVoiceStateCacheEntry entry;
    entry.sample_paths = sample_paths;
    entry.audio = audio;
    entry.states.reserve(acoustic_means.size());
    std::vector<VibeVoiceSpeakerPrompt> prompts;
    prompts.reserve(acoustic_means.size());
    for (size_t index = 0; index < acoustic_means.size(); ++index) {
        auto state = reference_state_from_latents(acoustic_means[index]);
        VibeVoiceSpeakerPrompt prompt;
        prompt.audio = audio[index];
        prompt.reference_state = state;
        prompts.push_back(std::move(prompt));
        entry.states.push_back(std::move(state));
    }

    while (cached_speakers() + entry.states.size() > kMaxReferenceVoiceStates && !reference_voice_state_cache_.empty()) {
        reference_voice_state_cache_.erase(reference_voice_state_cache_.begin());
        ++cache_evictions;
    }
    reference_voice_state_cache_.push_back(std::move(entry));
    engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_hits", 0);
    engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_misses", static_cast<int64_t>(sample_paths.size()));
    engine::debug::timing_log_scalar("vibevoice.reference_voice.cache_evictions", cache_evictions);
    engine::debug::timing_log_scalar(
        "vibevoice.reference_voice.cache_entries",
        static_cast<int64_t>(reference_voice_state_cache_.size()));
    engine::debug::timing_log_scalar(
        "vibevoice.reference_voice.cached_speakers",
        static_cast<int64_t>(cached_speakers()));
    return prompts;
}

}  // namespace engine::models::vibevoice

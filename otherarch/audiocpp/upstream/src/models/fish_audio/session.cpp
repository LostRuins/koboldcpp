#include "engine/models/fish_audio/session.h"

#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/fish_audio/ar.h"
#include "engine/models/fish_audio/generator.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::fish_audio {
namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr size_t kDefaultArGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultArWeightContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecWeightContextBytes = 32ull * 1024ull * 1024ull;
constexpr int64_t kDefaultReferenceCacheSlots = 1;
constexpr const char * kReferenceTextOption = "reference_text";
constexpr const char * kMultiReferenceCondOption = "multi_reference_cond";

std::shared_ptr<const FishAudioAssets> require_assets(
    std::shared_ptr<const FishAudioAssets> assets, const runtime::SessionOptions & options) {
    if (assets == nullptr) {
        throw std::runtime_error("Fish Audio session requires assets");
    }
    const auto path = runtime::find_option(options.options, {"fish_audio.lora"});
    if (!path || path->empty()) {
        return assets;
    }
    engine::assets::WeightAdapterReadOptions read_options;
    read_options.require_scaling_metadata = true;
    read_options.strength = runtime::parse_finite_float_option(
        options.options, {"fish_audio.lora_scale"}).value_or(1.0F);
    auto adapter = engine::assets::read_weight_adapter(std::filesystem::u8path(*path), read_options);
    engine::assets::WeightAdapterBindings bindings;
    const std::string suffix = ".q_proj.weight";
    for (const auto & tensor : assets->model_weights->tensors()) {
        bindings.add(tensor.name, tensor.name);
        bindings.add("model." + tensor.name, tensor.name);
        if (tensor.name.size() < suffix.size() ||
            tensor.name.compare(tensor.name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const auto prefix = tensor.name.substr(0, tensor.name.size() - suffix.size());
        const auto q_rows = tensor.shape.at(0);
        const auto k_name = prefix + ".k_proj.weight";
        const auto v_name = prefix + ".v_proj.weight";
        const auto k_rows = assets->model_weights->require_metadata(k_name).shape.at(0);
        const auto v_rows = assets->model_weights->require_metadata(v_name).shape.at(0);
        const std::vector<engine::assets::WeightAdapterTarget> targets{
            {tensor.name, 0, q_rows}, {k_name, q_rows, k_rows}, {v_name, q_rows + k_rows, v_rows}};
        bindings.add(prefix + ".wqkv", targets);
        bindings.add("model." + prefix + ".wqkv", targets);
    }
    auto plan = engine::assets::bind_weight_adapter(
        *assets->model_weights, std::move(adapter), bindings, engine::assets::WeightMergeMode::DeltaF32);
    auto adapted = std::make_shared<FishAudioAssets>(*assets);
    adapted->model_weights = engine::assets::make_weight_adapter_source(
        assets->model_weights, std::move(plan), "fish_audio.lora");
    return adapted;
}

assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return assets::parse_tensor_storage_type(it->second);
}

void validate_ar_weight_storage(assets::TensorStorageType type, const char * option_name) {
    if (type == assets::TensorStorageType::Native ||
        type == assets::TensorStorageType::F32 ||
        type == assets::TensorStorageType::F16 ||
        type == assets::TensorStorageType::BF16 ||
        type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports native/f32/f16/bf16/q8_0");
}

void validate_codec_weight_storage(assets::TensorStorageType type, const char * option_name) {
    if (type == assets::TensorStorageType::Native ||
        type == assets::TensorStorageType::F32 ||
        type == assets::TensorStorageType::F16 ||
        type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports native/f32/f16/q8_0");
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"fish_audio.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "fish_audio.mem_saver");
    }
    return false;
}

std::size_t resolve_reference_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"fish_audio.reference_cache_slots", "reference_cache_slots"})
        .value_or(kDefaultReferenceCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("fish_audio.reference_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("fish_audio.reference_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

uint64_t mix_reference_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_reference_key(key, static_cast<uint64_t>(bits));
    }
    return key;
}

FishAudioGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) {
    FishAudioGenerationOptions options;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_new_tokens", "max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("Fish Audio max_new_tokens must be non-negative");
        }
        if (*value > 0) {
            options.max_new_tokens = *value;
        }
    }
    options.text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(options.text_chunk_size);
    options.top_p = runtime::parse_float_option(request.options, {"top_p"}).value_or(options.top_p);
    options.top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(options.top_k);
    options.temperature = runtime::parse_float_option(request.options, {"temperature"}).value_or(options.temperature);
    options.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(runtime::random_u32_seed());
    if (options.max_new_tokens <= 0) {
        throw std::runtime_error("Fish Audio max_new_tokens must be positive after default resolution");
    }
    if (options.text_chunk_size <= 0) {
        throw std::runtime_error("Fish Audio text_chunk_size must be positive");
    }
    if (!(options.top_p > 0.0F && options.top_p <= 1.0F)) {
        throw std::runtime_error("Fish Audio top_p must be in (0, 1]");
    }
    if (options.top_k <= 0) {
        throw std::runtime_error("Fish Audio top_k must be positive");
    }
    if (!(options.temperature > 0.0F && options.temperature < 2.0F)) {
        throw std::runtime_error("Fish Audio temperature must be in (0, 2)");
    }
    return options;
}

std::string lower_ascii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool valid_reference_id_char(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == ' ';
}

void validate_reference_id(const std::string & id) {
    if (id.empty() || id.size() > 255) {
        throw std::runtime_error(
            "Fish Audio cached_voice_id must be 1-255 characters");
    }
    for (const unsigned char ch : id) {
        if (!valid_reference_id_char(ch)) {
            throw std::runtime_error(
                "Fish Audio cached_voice_id may only contain alphanumeric characters, hyphens, underscores, and spaces");
        }
    }
}

bool is_supported_saved_reference_audio(const fs::path & path) {
    return lower_ascii(path.extension().string()) == ".wav";
}

std::vector<fs::path> collect_saved_reference_audio_files(const fs::path & directory) {
    std::vector<fs::path> files;
    for (const auto & entry : fs::recursive_directory_iterator(directory)) {
        if (!entry.is_regular_file() || !is_supported_saved_reference_audio(entry.path())) {
            continue;
        }
        auto lab_path = entry.path();
        lab_path.replace_extension(".lab");
        if (engine::io::is_existing_file(lab_path)) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

runtime::AudioBuffer read_saved_reference_audio(const fs::path & path) {
    auto wav = engine::audio::read_wav_f32(path);
    return runtime::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
}

std::vector<FishAudioReference> load_saved_references(
    const FishAudioAssets & assets,
    const std::string & reference_id) {
    validate_reference_id(reference_id);
    const auto reference_dir = engine::io::require_directory(
        assets.resources.model_root() / "references" / reference_id,
        "Fish Audio cached voice reference");
    const auto audio_files = collect_saved_reference_audio_files(reference_dir);
    if (audio_files.empty()) {
        throw std::runtime_error(
            "Fish Audio cached_voice_id '" + reference_id +
            "' requires at least one WAV reference with a matching .lab file under " +
            reference_dir.string());
    }
    std::vector<FishAudioReference> references;
    references.reserve(audio_files.size());
    for (const auto & audio_file : audio_files) {
        auto lab_path = audio_file;
        lab_path.replace_extension(".lab");
        references.push_back(FishAudioReference{
            read_saved_reference_audio(audio_file),
            engine::io::read_text_file(lab_path),
            reference_id + "\n" + audio_file.lexically_normal().string()});
    }
    return references;
}

std::string reference_cache_id_from_voice(const std::optional<runtime::VoiceCondition> & voice) {
    if (voice.has_value() &&
        voice->speaker.has_value() &&
        voice->speaker->cached_voice_id.has_value() &&
        !voice->speaker->cached_voice_id->empty()) {
        return *voice->speaker->cached_voice_id;
    }
    return {};
}

bool has_reference_selector(const std::optional<runtime::VoiceCondition> & voice) {
    if (!voice.has_value() || !voice->speaker.has_value()) {
        return false;
    }
    const auto & speaker = *voice->speaker;
    return speaker.audio.has_value() ||
        (speaker.cached_voice_id.has_value() && !speaker.cached_voice_id->empty());
}

std::vector<FishAudioReference> references_from_voice(
    const FishAudioAssets & assets,
    const std::optional<runtime::VoiceCondition> & voice,
    const std::unordered_map<std::string, std::string> & options,
    const char * role) {
    if (!has_reference_selector(voice)) {
        return {};
    }
    const auto & speaker = *voice->speaker;
    if (speaker.audio.has_value()) {
        auto reference_text = runtime::find_option(options, {kReferenceTextOption});
        if (!reference_text.has_value()) {
            throw std::runtime_error(
                std::string(role) + " with inline reference audio requires reference_text option");
        }
        return {FishAudioReference{
            speaker.audio,
            *reference_text,
            reference_cache_id_from_voice(voice)}};
    }
    return load_saved_references(assets, *speaker.cached_voice_id);
}

std::vector<FishAudioReference> multi_reference_cond_from_options(
    const std::unordered_map<std::string, std::string> & options) {
    const auto value = runtime::find_option(options, {kMultiReferenceCondOption});
    if (!value.has_value()) {
        return {};
    }
    const auto root = engine::io::json::parse(*value);
    if (!root.is_array()) {
        throw std::runtime_error("Fish Audio multi_reference_cond must be a JSON array");
    }
    std::vector<FishAudioReference> references;
    references.reserve(root.as_array().size());
    for (const auto & item : root.as_array()) {
        if (!item.is_object()) {
            throw std::runtime_error("Fish Audio multi_reference_cond entries must be objects");
        }
        const auto audio_path = engine::io::json::require_string(item, "audio");
        const auto text = engine::io::json::require_string(item, "text");
        if (audio_path.empty()) {
            throw std::runtime_error("Fish Audio multi_reference_cond audio path must not be empty");
        }
        if (text.empty()) {
            throw std::runtime_error("Fish Audio multi_reference_cond text must not be empty");
        }
        references.push_back(FishAudioReference{
            read_saved_reference_audio(audio_path),
            text,
            {}});
    }
    return references;
}

}  // namespace

FishAudioSession::FishAudioSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const FishAudioAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets), options)),
      reference_cache_(resolve_reference_cache_slots(this->options())) {
    if ((task_.task != runtime::VoiceTaskKind::Tts &&
         task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
        (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming)) {
        throw std::runtime_error("Fish Audio supports offline and streaming TTS and voice cloning sessions");
    }
    const auto ar_weight_type =
        option_weight_type(options, "fish_audio.weight_type", assets::TensorStorageType::Native);
    const auto codec_weight_type =
        option_weight_type(options, "fish_audio.codec_weight_type", assets::TensorStorageType::Native);
    validate_ar_weight_storage(ar_weight_type, "fish_audio.weight_type");
    validate_codec_weight_storage(codec_weight_type, "fish_audio.codec_weight_type");
    const int threads = options.backend.threads > 0 ? options.backend.threads : 1;
    auto ar = std::make_unique<FishAudioDualARRuntime>(
        assets_,
        options.backend,
        threads,
        runtime::parse_size_mb_option(options.options, {"fish_audio.ar_graph_arena_mb"}, kDefaultArGraphArenaBytes),
        runtime::parse_size_mb_option(options.options, {"fish_audio.ar_weight_context_mb"}, kDefaultArWeightContextBytes),
        ar_weight_type);
    engine::codecs::FishDacCodecRuntimeOptions codec_options;
    codec_options.graph_arena_bytes =
        runtime::parse_size_mb_option(options.options, {"fish_audio.codec_graph_arena_mb"}, kDefaultCodecGraphArenaBytes);
    codec_options.weight_context_bytes =
        runtime::parse_size_mb_option(options.options, {"fish_audio.codec_weight_context_mb"}, kDefaultCodecWeightContextBytes);
    codec_options.matmul_weight_storage_type = codec_weight_type;
    codec_options.conv_weight_storage_type = codec_weight_type;
    if (const auto value = runtime::find_option(options.options, {"fish_audio.codec_encoder_bf16_compute"})) {
        codec_options.encoder_bf16_compute = runtime::parse_bool_option(*value, "fish_audio.codec_encoder_bf16_compute");
    }
    core::ExecutionContext codec_execution(options.backend);
    auto codec_component = engine::codecs::FishDacCodecComponent::load_from_tensor_source(
        assets_->codec_weights,
        assets_->config.codec,
        engine::codecs::FishDacCodecWeightBinding{},
        codec_execution.backend(),
        codec_execution.backend_type(),
        codec_options);
    auto codec = std::make_unique<engine::codecs::FishDacCodecRuntime>(
        std::move(codec_component),
        options.backend,
        threads,
        codec_options.graph_arena_bytes);
    generator_ = std::make_unique<FishAudioGenerator>(
        assets_,
        std::move(ar),
        std::move(codec));
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

FishAudioSession::~FishAudioSession() = default;

std::string FishAudioSession::family() const {
    return "fish_audio";
}

runtime::VoiceTaskKind FishAudioSession::task_kind() const {
    return task_.task;
}

runtime::RunMode FishAudioSession::run_mode() const {
    return task_.mode;
}

bool FishAudioSession::ReferenceCacheKeyEqual::operator()(
    const ReferenceCacheKey & lhs,
    const ReferenceCacheKey & rhs) const {
    return lhs.source_id == rhs.source_id &&
        lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

void FishAudioSession::prepare(const runtime::SessionPreparationRequest & request) {
    defaults_.reset();
    FishAudioRequest defaults;
    bool has_defaults = false;
    if (request.text.has_value()) {
        defaults.text = request.text->text;
        has_defaults = true;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"max_new_tokens", "max_tokens"})) {
        if (*value < 0) {
            throw std::runtime_error("Fish Audio max_new_tokens must be non-negative");
        }
        if (*value > 0) {
            defaults.generation.max_new_tokens = *value;
        }
    }
    if (auto references = references_from_voice(*assets_, request.voice, request.options, "Fish Audio prepare");
        !references.empty()) {
        defaults.references = std::move(references);
        has_defaults = true;
    } else if (auto references = multi_reference_cond_from_options(request.options);
               !references.empty()) {
        defaults.references = std::move(references);
        has_defaults = true;
    }
    if (has_defaults) {
        defaults_ = std::move(defaults);
    }
    mark_prepared();
}

FishAudioRequest FishAudioSession::make_request(const runtime::TaskRequest & request) const {
    FishAudioRequest out = defaults_.value_or(FishAudioRequest{});
    if (request.text_input.has_value()) {
        out.text = request.text_input->text;
    }
    out.generation = generation_options_from_request(request);
    if (auto references = references_from_voice(*assets_, request.voice, request.options, "Fish Audio request");
        !references.empty()) {
        out.references = std::move(references);
    } else if (auto references = multi_reference_cond_from_options(request.options);
               !references.empty()) {
        out.references = std::move(references);
    } else if (request.text_input.has_value()) {
        out.references.clear();
    }
    if (out.text.empty()) {
        throw std::runtime_error("Fish Audio request text must not be empty");
    }
    return out;
}

const engine::codecs::FishDacCodes & FishAudioSession::resolve_reference_codes(const FishAudioReference & reference) {
    ReferenceCacheKey key;
    key.source_id = reference.cache_id;
    if (reference.cache_id.empty() && !reference.audio.has_value()) {
        throw std::runtime_error("Fish Audio cached reference requires reference audio or a reference id");
    }
    if (reference.audio.has_value() && reference.cache_id.empty()) {
        key.sample_rate = reference.audio->sample_rate;
        key.channels = reference.audio->channels;
        key.sample_count = static_cast<uint64_t>(reference.audio->samples.size());
        key.sample_hash = hash_audio_samples(*reference.audio);
    }
    if (const auto * cached = reference_cache_.find(key)) {
        engine::debug::trace_log_scalar("fish_audio.reference_cache.hit", 1);
        engine::debug::trace_log_scalar("fish_audio.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
        engine::debug::trace_log_scalar("fish_audio.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
        engine::debug::trace_log_scalar("fish_audio.reference_cache.evicted", 0);
        return cached->codes;
    }
    if (!reference.audio.has_value()) {
        throw std::runtime_error("Fish Audio reference id is not cached and no reference audio was provided");
    }
    const bool will_evict = reference_cache_.capacity() > 0 && reference_cache_.size() >= reference_cache_.capacity();
    const auto start = Clock::now();
    ReferenceCacheEntry entry;
    entry.codes = generator_->encode_reference(*reference.audio);
    engine::debug::trace_log_scalar("fish_audio.reference.frames", entry.codes.frames);
    engine::debug::trace_log_scalar("fish_audio.reference.codebooks", entry.codes.codebooks);
    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(entry);
    } else {
        reference_cache_.put(key, std::move(entry));
    }
    engine::debug::trace_log_scalar("fish_audio.reference_cache.hit", 0);
    engine::debug::trace_log_scalar("fish_audio.reference_cache.slots", static_cast<int64_t>(reference_cache_.capacity()));
    engine::debug::trace_log_scalar("fish_audio.reference_cache.entries", static_cast<int64_t>(reference_cache_.size()));
    engine::debug::trace_log_scalar("fish_audio.reference_cache.evicted", will_evict ? 1 : 0);
    engine::debug::timing_log_scalar("fish_audio.reference_encode_ms", engine::debug::elapsed_ms(start, Clock::now()));
    if (reference_cache_.capacity() == 0) {
        return uncached_reference_->codes;
    }
    const auto * cached = reference_cache_.find(key);
    if (cached == nullptr) {
        throw std::runtime_error("Fish Audio reference cache insert failed");
    }
    return cached->codes;
}

runtime::TaskResult FishAudioSession::run(const runtime::TaskRequest & request) {
    return synthesize(request, false);
}

runtime::TaskResult FishAudioSession::generate_stream(const runtime::TaskRequest & request) {
    return synthesize(request, true);
}

runtime::TaskResult FishAudioSession::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("Fish Audio run()");
    const auto wall_start = Clock::now();
    const bool mem_saver = mem_saver_from_options(options());
    const auto request_options = generation_options_from_request(request);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, request_options.text_chunk_size, text_chunk_mode);
    engine::debug::trace_log_scalar("fish_audio.text_chunk_size", request_options.text_chunk_size);
    engine::debug::trace_log_scalar("fish_audio.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("fish_audio.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));

    runtime::AudioBuffer merged_audio;
    std::vector<engine::codecs::FishDacCodes> reference_codes;
    std::optional<FishAudioConversationTurn> previous_turn = std::nullopt;
    for (size_t chunk_index = 0; chunk_index < chunk_requests.size(); ++chunk_index) {
        const auto & chunk_request = chunk_requests[chunk_index];
        auto fish_request = make_request(chunk_request);
        if (!fish_request.references.empty() && reference_codes.empty()) {
            reference_codes.reserve(fish_request.references.size());
            for (const auto & reference : fish_request.references) {
                reference_codes.push_back(resolve_reference_codes(reference));
            }
        }
        std::function<void(const runtime::AudioBuffer &)> on_audio;
        if (streaming && stream_event_sink()) {
            on_audio = [&](const runtime::AudioBuffer & audio) {
                runtime::StreamEvent event;
                event.audio_output = audio;
                stream_event_sink()(event);
            };
        }
        auto generated = generator_->generate(fish_request, reference_codes, previous_turn, mem_saver, streaming, on_audio);
        if (streaming && generated.audio.samples.empty()) {
            continue;
        }
        runtime::append_audio_buffer(merged_audio, generated.audio);
        if (chunk_requests.size() > 1) {
            previous_turn = FishAudioConversationTurn{fish_request.text, std::move(generated.codes)};
        }
    }
    if (streaming && merged_audio.samples.empty()) {
        throw std::runtime_error(
            "Fish Audio streaming produced no audio across all text chunks. "
            "Check max_tokens / max_new_tokens: a value of 1 leaves no confirmed audio frames. "
            "Try increasing the token limit; generation may also have ended immediately.");
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

}  // namespace engine::models::fish_audio

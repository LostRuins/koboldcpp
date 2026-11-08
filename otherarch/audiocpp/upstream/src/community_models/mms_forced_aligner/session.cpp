#include "engine/community_models/mms_forced_aligner/session.h"

#include "engine/community_models/mms_forced_aligner/ctc_alignment.h"
#include "engine/community_models/mms_forced_aligner/text_processor.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::mms_forced_aligner {

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "mms_forced_aligner";
constexpr double kSampleRate16k = 16000.0;
constexpr int64_t kFrameStrideSamples = 320;
constexpr double kDefaultEmissionWindowSec = 30.0;
constexpr double kDefaultEmissionContextSec = 2.0;
constexpr int64_t kDefaultMaxAlignmentCells = 50000000;
constexpr int64_t kDefaultMaxTargetTokens = 8192;

std::shared_ptr<const MmsForcedAlignerAssets> require_assets(
    std::shared_ptr<const MmsForcedAlignerAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MMS forced aligner session requires assets");
    }
    if (assets->model_weights == nullptr) {
        throw std::runtime_error("MMS forced aligner tensor source must not be null");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MMS forced aligner session requires a model contract");
    }
    return contract;
}

runtime::SessionOptions normalize_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    runtime::validate_spec_backed_session_options(
        options,
        *require_contract(contract),
        kFamily,
        "MMS forced aligner");
    return options;
}

engine::assets::TensorStorageType parse_weight_storage_type(
    const runtime::SessionOptions & options) {
    return runtime::parse_tensor_storage_option(
        options.options,
        "mms_forced_aligner.weight_type",
        engine::assets::TensorStorageType::Native,
        {engine::assets::TensorStorageType::Native,
         engine::assets::TensorStorageType::F32,
         engine::assets::TensorStorageType::F16});
}

MmsTextNormalization parse_text_normalization(const std::string & value) {
    if (value == "latin") {
        return MmsTextNormalization::Latin;
    }
    if (value == "pre_romanized") {
        return MmsTextNormalization::PreRomanized;
    }
    throw std::runtime_error("MMS forced aligner text_normalization must be latin or pre_romanized, got '" + value + "'");
}

MmsStarFrequency parse_star_frequency(const std::string & value) {
    if (value == "segment") {
        return MmsStarFrequency::Segment;
    }
    if (value == "edges") {
        return MmsStarFrequency::Edges;
    }
    throw std::runtime_error("MMS forced aligner star_frequency must be segment or edges, got '" + value + "'");
}

float parse_merge_threshold_sec(const std::unordered_map<std::string, std::string> & options) {
    const auto value = runtime::parse_finite_float_option(options, {"merge_threshold_sec"});
    if (!value.has_value()) {
        return 0.0F;
    }
    if (*value < 0.0F) {
        throw std::runtime_error("MMS forced aligner merge_threshold_sec must be non-negative");
    }
    return *value;
}

int64_t source_audio_samples(const runtime::TaskRequest & request) {
    return static_cast<int64_t>(request.audio_input->samples.size()) / request.audio_input->channels;
}

// Rescales a 16 kHz sample coordinate to the source audio rate.
int64_t rescale_sample(int64_t sample_16k, int64_t source_rate) {
    return static_cast<int64_t>(std::llround(
        static_cast<double>(sample_16k) * static_cast<double>(source_rate) / kSampleRate16k));
}

// nl/nld and en/eng are alias pairs; any other code stands for itself, so
// pre-romanized mode accepts arbitrary codes as literals.
bool same_language(const std::string & a, const std::string & b) {
    return (a == b) ||
           ((a == "nl" || a == "nld") && (b == "nl" || b == "nld")) ||
           ((a == "en" || a == "eng") && (b == "en" || b == "eng"));
}

}  // namespace

MmsForcedAlignerSession::MmsForcedAlignerSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MmsForcedAlignerAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(normalize_session_options(std::move(options), contract)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::Alignment) {
        throw std::runtime_error("MMS forced aligner only supports VoiceTaskKind::Alignment");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MMS forced aligner currently supports offline sessions");
    }
    const auto & session_options = RuntimeSessionBase::options().options;
    const double window_sec = runtime::parse_positive_finite_float_option(
        session_options, {"mms_forced_aligner.emission_window_sec"}).value_or(kDefaultEmissionWindowSec);
    const double context_sec = runtime::parse_finite_float_option(
        session_options, {"mms_forced_aligner.emission_context_sec"}).value_or(kDefaultEmissionContextSec);
    max_alignment_cells_ = runtime::parse_positive_i64_option(
        session_options, {"mms_forced_aligner.max_alignment_cells"}, kDefaultMaxAlignmentCells);
    max_target_tokens_ = runtime::parse_positive_i64_option(
        session_options, {"mms_forced_aligner.max_target_tokens"}, kDefaultMaxTargetTokens);
    const auto weight_storage_type =
        parse_weight_storage_type(RuntimeSessionBase::options());

    MmsWav2Vec2CtcConfig emission_config;
    emission_config.window_sec = window_sec;
    emission_config.context_sec = context_sec;
    emission_runtime_ = std::make_unique<MmsWav2Vec2CtcRuntime>(
        assets_,
        RuntimeSessionBase::options().backend,
        weight_storage_type,
        emission_config);
}

std::string MmsForcedAlignerSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MmsForcedAlignerSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MmsForcedAlignerSession::run_mode() const {
    return task_.mode;
}

void MmsForcedAlignerSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value() || !request.text.has_value()) {
        throw std::runtime_error("MMS forced aligner prepare() requires audio and transcript contracts");
    }
    mark_prepared();
}

runtime::TaskResult MmsForcedAlignerSession::run(const runtime::TaskRequest & request) {
    require_prepared("MMS forced aligner run()");
    const auto chunk_mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (chunk_mode == engine::audio::AudioChunkMode::Fixed ||
        chunk_mode == engine::audio::AudioChunkMode::QuietEnergy ||
        chunk_mode == engine::audio::AudioChunkMode::Vad) {
        throw std::runtime_error(
            "MMS forced aligner does not support standalone audio chunking; "
            "split audio and transcript into matching blocks before alignment "
            "so each request receives one audio block and its transcript block");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MMS forced aligner run() requires audio_input");
    }
    if (!request.text_input.has_value() || request.text_input->text.empty() || request.text_input->language.empty()) {
        throw std::runtime_error("MMS forced aligner run() requires transcript text and language");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MMS forced aligner");
    const auto wall_start = Clock::now();

    const std::string & language = request.text_input->language;
    if (const auto it = request.options.find("language"); it != request.options.end() && !it->second.empty()) {
        if (!same_language(it->second, language)) {
            throw std::runtime_error(
                "MMS forced aligner request language '" + language + "' conflicts with option language '" +
                it->second + "'");
        }
    }

    const auto text_start = Clock::now();
    MmsTextProcessorOptions processor_options;
    if (const auto it = request.options.find("text_normalization"); it != request.options.end()) {
        processor_options.normalization = parse_text_normalization(it->second);
    }
    if (const auto it = request.options.find("star_frequency"); it != request.options.end()) {
        processor_options.star_frequency = parse_star_frequency(it->second);
    }
    const float merge_threshold_sec = parse_merge_threshold_sec(request.options);
    const auto prepared = prepare_mms_text(
        assets_->vocabulary,
        request.text_input->text,
        language,
        processor_options);
    if (static_cast<int64_t>(prepared.target_ids.size()) > max_target_tokens_) {
        throw std::runtime_error(
            "MMS forced aligner transcript has " + std::to_string(prepared.target_ids.size()) +
            " target tokens, exceeding max_target_tokens=" + std::to_string(max_target_tokens_));
    }
    const auto text_end = Clock::now();

    const auto emission_start = Clock::now();
    const auto emissions = emission_runtime_->compute(*request.audio_input);
    const auto emission_end = Clock::now();

    const auto alignment_start = Clock::now();
    CtcAlignmentLimits limits;
    limits.max_alignment_cells = max_alignment_cells_;
    limits.max_target_tokens = max_target_tokens_;
    MmsLogProbabilities log_probs{emissions.log_probs.data(), emissions.frames, emissions.classes};
    const auto alignment = ctc_forced_align(
        log_probs,
        prepared.target_ids,
        assets_->vocabulary.blank_id,
        limits);
    const auto alignment_end = Clock::now();

    const auto postprocess_start = Clock::now();
    const auto spans = mms_word_spans_from_alignment(
        alignment,
        prepared,
        assets_->vocabulary.blank_id,
        merge_threshold_sec);
    const int64_t source_rate = request.audio_input->sample_rate;
    const int64_t source_samples = source_audio_samples(request);
    const int64_t source_max_sample = source_samples > 0 ? source_samples : 0;
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{request.text_input->text, prepared.canonical_language};
    result.word_timestamps.reserve(spans.size());
    int64_t previous_end = 0;
    for (const auto & span : spans) {
        const int64_t start_16k = span.start_frame * kFrameStrideSamples;
        const int64_t end_16k = (span.end_frame + 1) * kFrameStrideSamples;
        int64_t start_sample = rescale_sample(start_16k, source_rate);
        int64_t end_sample = rescale_sample(end_16k, source_rate);
        start_sample = std::max<int64_t>(0, std::min(start_sample, source_max_sample));
        end_sample = std::max<int64_t>(0, std::min(end_sample, source_max_sample));
        if (end_sample < start_sample) {
            end_sample = start_sample;
        }
        if (start_sample < previous_end && previous_end <= source_max_sample) {
            start_sample = previous_end;
        }
        runtime::WordTimestamp timestamp;
        timestamp.span.start_sample = start_sample;
        timestamp.span.end_sample = end_sample;
        timestamp.word = prepared.original_words[static_cast<size_t>(span.word_index)];
        timestamp.confidence = span.log_prob;
        result.word_timestamps.push_back(std::move(timestamp));
        previous_end = end_sample;
    }
    const auto postprocess_end = Clock::now();

    const auto wall_end = Clock::now();
    debug::timing_log_scalar("mms_forced_aligner.text_ms", engine::debug::elapsed_ms(text_start, text_end));
    debug::timing_log_scalar("mms_forced_aligner.emission_ms", engine::debug::elapsed_ms(emission_start, emission_end));
    debug::timing_log_scalar("mms_forced_aligner.alignment_ms", engine::debug::elapsed_ms(alignment_start, alignment_end));
    debug::timing_log_scalar("mms_forced_aligner.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    debug::trace_log_scalar("mms_forced_aligner.audio_frames", emissions.frames);
    debug::trace_log_scalar("mms_forced_aligner.target_tokens", prepared.target_ids.size());
    debug::trace_log_scalar(
        "mms_forced_aligner.alignment_cells",
        emissions.frames * static_cast<int64_t>(2 * prepared.target_ids.size() + 1));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_mms_forced_aligner_loader() {
    runtime::SpecBackedVoiceModelConfig<MmsForcedAlignerAssets> config;
    config.family = kFamily;
    config.load_assets = load_mms_forced_aligner_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                               const runtime::SessionOptions & options,
                               std::shared_ptr<const MmsForcedAlignerAssets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<MmsForcedAlignerSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::mms_forced_aligner

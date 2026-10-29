#include "engine/models/omnivoice/audio_tokenizer.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

using Clock = std::chrono::steady_clock;

namespace assets_ns = engine::assets;
namespace modules = engine::modules;
constexpr float kTargetReferenceRms = 0.1F;
constexpr float kSilenceThresholdDb = -50.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t checked_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("OmniVoice audio tokenizer expected positive ") + name);
    }
    return value;
}

void validate_weight_storage_type(assets_ns::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets_ns::TensorStorageType::Native:
        case assets_ns::TensorStorageType::F32:
        case assets_ns::TensorStorageType::F16:
        case assets_ns::TensorStorageType::BF16:
        case assets_ns::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "OmniVoice audio tokenizer weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

struct NormalizedReferenceAudio {
    std::vector<float> acoustic_samples_24k;
    std::vector<float> semantic_samples_16k_padded;
    float reference_rms = 0.0F;
    int64_t frames = 0;
};

std::vector<float> to_mono(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("OmniVoice reference audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("OmniVoice reference audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("OmniVoice reference audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("OmniVoice interleaved reference audio has invalid channel layout");
    }
    return engine::audio::mixdown_interleaved_to_mono_average(
        audio.samples,
        audio.channels,
        engine::audio::MonoMixAccumulation::Float64);
}

int64_t samples_to_ms(size_t samples, int sample_rate) {
    return static_cast<int64_t>(
        std::llround(1000.0 * static_cast<double>(samples) / static_cast<double>(sample_rate)));
}

size_t ms_to_samples(int64_t ms, int sample_rate, size_t total_samples) {
    if (ms <= 0) {
        return 0;
    }
    const int64_t samples = static_cast<int64_t>(
        static_cast<double>(ms) * static_cast<double>(sample_rate) / 1000.0);
    return static_cast<size_t>(std::clamp<int64_t>(samples, 0, static_cast<int64_t>(total_samples)));
}

double db_to_amplitude(float dbfs) {
    return std::pow(10.0, static_cast<double>(dbfs) / 20.0) * 32768.0;
}

int rms_pcm16_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int64_t start_ms,
    int64_t end_ms) {
    const size_t start = ms_to_samples(start_ms, sample_rate, pcm.size());
    const size_t end = ms_to_samples(end_ms, sample_rate, pcm.size());
    if (start >= end) {
        return 0;
    }
    double sum = 0.0;
    for (size_t i = start; i < end; ++i) {
        const double value = static_cast<double>(pcm[i]);
        sum += value * value;
    }
    return static_cast<int>(std::sqrt(sum / static_cast<double>(end - start)));
}

std::vector<float> resample_sinc_hann_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate,
    int lowpass_filter_width = 6,
    double rolloff = 0.99) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("OmniVoice resample sample rates must be positive");
    }
    if (input.empty() || input_sample_rate == output_sample_rate) {
        return input;
    }
    const int rate_gcd = std::gcd(input_sample_rate, output_sample_rate);
    const int orig = input_sample_rate / rate_gcd;
    const int next = output_sample_rate / rate_gcd;
    if (orig <= 0 || next <= 0) {
        throw std::runtime_error("OmniVoice resample reduced rates must be positive");
    }

    const double base_freq = static_cast<double>(std::min(orig, next)) * rolloff;
    const int64_t width = static_cast<int64_t>(
        std::ceil(static_cast<double>(lowpass_filter_width * orig) / base_freq));
    const int64_t kernel_size = (2 * width) + orig;
    const double scale = base_freq / static_cast<double>(orig);
    const double lowpass = static_cast<double>(lowpass_filter_width);
    const double pi = std::acos(-1.0);

    std::vector<double> kernels(static_cast<size_t>(next * kernel_size), 0.0);
    for (int phase = 0; phase < next; ++phase) {
        const double phase_offset = -static_cast<double>(phase) / static_cast<double>(next);
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            const double idx = static_cast<double>(tap - width) / static_cast<double>(orig);
            double t = (phase_offset + idx) * base_freq;
            t = std::clamp(t, -lowpass, lowpass);
            const double window = std::pow(std::cos(t * pi / lowpass / 2.0), 2.0);
            const double angle = t * pi;
            const double sinc = std::abs(angle) < 1.0e-12 ? 1.0 : std::sin(angle) / angle;
            kernels[static_cast<size_t>(phase * kernel_size + tap)] = sinc * window * scale;
        }
    }

    std::vector<float> padded(static_cast<size_t>(width) + input.size() + static_cast<size_t>(width + orig), 0.0F);
    std::copy(input.begin(), input.end(), padded.begin() + static_cast<std::ptrdiff_t>(width));

    const int64_t steps =
        (static_cast<int64_t>(padded.size()) - kernel_size) / static_cast<int64_t>(orig) + 1;
    const int64_t target_length = static_cast<int64_t>(
        std::ceil(static_cast<double>(next) * static_cast<double>(input.size()) / static_cast<double>(orig)));
    std::vector<float> output(static_cast<size_t>(target_length), 0.0F);
    for (int64_t index = 0; index < target_length; ++index) {
        const int phase = static_cast<int>(index % next);
        const int64_t step = index / next;
        if (step >= steps) {
            break;
        }
        const int64_t start = step * static_cast<int64_t>(orig);
        double sum = 0.0;
        const double * kernel = kernels.data() + static_cast<std::ptrdiff_t>(phase * kernel_size);
        for (int64_t tap = 0; tap < kernel_size; ++tap) {
            sum += static_cast<double>(padded[static_cast<size_t>(start + tap)]) * kernel[tap];
        }
        output[static_cast<size_t>(index)] = static_cast<float>(sum);
    }
    return output;
}

std::vector<std::pair<int64_t, int64_t>> detect_silence_ranges_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int min_silence_len_ms,
    float silence_threshold_db,
    int seek_step_ms) {
    const int64_t segment_len_ms = samples_to_ms(pcm.size(), sample_rate);
    if (segment_len_ms < min_silence_len_ms) {
        return {};
    }
    const double silence_threshold = db_to_amplitude(silence_threshold_db);
    std::vector<int64_t> silence_starts;
    const int64_t last_slice_start = segment_len_ms - min_silence_len_ms;
    for (int64_t start_ms = 0; start_ms <= last_slice_start; start_ms += seek_step_ms) {
        if (static_cast<double>(rms_pcm16_ms(pcm, sample_rate, start_ms, start_ms + min_silence_len_ms)) <=
            silence_threshold) {
            silence_starts.push_back(start_ms);
        }
    }
    if ((last_slice_start % seek_step_ms) != 0) {
        if (static_cast<double>(
                rms_pcm16_ms(pcm, sample_rate, last_slice_start, last_slice_start + min_silence_len_ms)) <=
            silence_threshold) {
            silence_starts.push_back(last_slice_start);
        }
    }
    if (silence_starts.empty()) {
        return {};
    }

    std::vector<std::pair<int64_t, int64_t>> silent_ranges;
    int64_t prev = silence_starts.front();
    int64_t current_start = prev;
    for (size_t i = 1; i < silence_starts.size(); ++i) {
        const int64_t start_ms = silence_starts[i];
        const bool continuous = (start_ms == prev + seek_step_ms);
        const bool has_gap = start_ms > (prev + min_silence_len_ms);
        if (!continuous && has_gap) {
            silent_ranges.emplace_back(current_start, prev + min_silence_len_ms);
            current_start = start_ms;
        }
        prev = start_ms;
    }
    silent_ranges.emplace_back(current_start, prev + min_silence_len_ms);
    return silent_ranges;
}

std::vector<std::pair<int64_t, int64_t>> detect_nonsilent_ranges_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int min_silence_ms,
    float silence_threshold_db,
    int seek_step_ms) {
    const auto silent_ranges =
        detect_silence_ranges_ms(pcm, sample_rate, min_silence_ms, silence_threshold_db, seek_step_ms);
    const int64_t segment_len_ms = samples_to_ms(pcm.size(), sample_rate);
    if (silent_ranges.empty()) {
        return {{0, segment_len_ms}};
    }
    if (silent_ranges.front().first == 0 && silent_ranges.front().second == segment_len_ms) {
        return {};
    }
    std::vector<std::pair<int64_t, int64_t>> nonsilent_ranges;
    int64_t prev_end = 0;
    for (const auto & range : silent_ranges) {
        nonsilent_ranges.emplace_back(prev_end, range.first);
        prev_end = range.second;
    }
    if (prev_end != segment_len_ms) {
        nonsilent_ranges.emplace_back(prev_end, segment_len_ms);
    }
    if (!nonsilent_ranges.empty() && nonsilent_ranges.front() == std::pair<int64_t, int64_t>{0, 0}) {
        nonsilent_ranges.erase(nonsilent_ranges.begin());
    }
    return nonsilent_ranges;
}

std::vector<int16_t> slice_pcm16_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int64_t start_ms,
    int64_t end_ms) {
    const size_t start = ms_to_samples(start_ms, sample_rate, pcm.size());
    const size_t end = ms_to_samples(end_ms, sample_rate, pcm.size());
    if (start >= end) {
        return {};
    }
    return std::vector<int16_t>(
        pcm.begin() + static_cast<std::ptrdiff_t>(start),
        pcm.begin() + static_cast<std::ptrdiff_t>(end));
}

int64_t detect_leading_silence_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    float silence_threshold_db = -50.0F,
    int chunk_size_ms = 10) {
    int64_t trim_ms = 0;
    const int64_t segment_len_ms = samples_to_ms(pcm.size(), sample_rate);
    while (trim_ms < segment_len_ms &&
           rms_pcm16_ms(pcm, sample_rate, trim_ms, trim_ms + chunk_size_ms) <= db_to_amplitude(silence_threshold_db)) {
        trim_ms += chunk_size_ms;
    }
    return std::min(trim_ms, segment_len_ms);
}

std::vector<int16_t> trim_edges_pcm16(
    const std::vector<int16_t> & mono,
    int sample_rate,
    int keep_lead_ms,
    int keep_trail_ms,
    float silence_threshold_db) {
    if (mono.empty()) {
        return mono;
    }
    const int64_t lead_trim_ms = detect_leading_silence_ms(mono, sample_rate, silence_threshold_db);
    auto trimmed = slice_pcm16_ms(
        mono,
        sample_rate,
        std::max<int64_t>(0, lead_trim_ms - keep_lead_ms),
        samples_to_ms(mono.size(), sample_rate));
    std::reverse(trimmed.begin(), trimmed.end());
    const int64_t trail_trim_ms = detect_leading_silence_ms(trimmed, sample_rate, silence_threshold_db);
    trimmed = slice_pcm16_ms(
        trimmed,
        sample_rate,
        std::max<int64_t>(0, trail_trim_ms - keep_trail_ms),
        samples_to_ms(trimmed.size(), sample_rate));
    std::reverse(trimmed.begin(), trimmed.end());
    return trimmed;
}

std::vector<int16_t> remove_mid_silence_pcm16(
    const std::vector<int16_t> & mono,
    int sample_rate,
    int min_silence_ms,
    int keep_silence_ms,
    float silence_threshold_db) {
    if (mono.empty() || min_silence_ms <= 0) {
        return mono;
    }
    auto nonsilent = detect_nonsilent_ranges_ms(mono, sample_rate, min_silence_ms, silence_threshold_db, 10);
    if (nonsilent.empty()) {
        return {};
    }
    std::vector<std::pair<int64_t, int64_t>> output_ranges;
    output_ranges.reserve(nonsilent.size());
    for (const auto & range : nonsilent) {
        output_ranges.emplace_back(range.first - keep_silence_ms, range.second + keep_silence_ms);
    }
    for (size_t i = 0; i + 1 < output_ranges.size(); ++i) {
        auto & current = output_ranges[i];
        auto & next = output_ranges[i + 1];
        if (next.first < current.second) {
            current.second = (current.second + next.first) / 2;
            next.first = current.second;
        }
    }
    std::vector<int16_t> output;
    for (const auto & range : output_ranges) {
        auto chunk = slice_pcm16_ms(
            mono,
            sample_rate,
            std::max<int64_t>(0, range.first),
            std::min<int64_t>(range.second, samples_to_ms(mono.size(), sample_rate)));
        output.insert(output.end(), chunk.begin(), chunk.end());
    }
    return output;
}

std::vector<float> preprocess_reference_mono(
    std::vector<float> mono,
    int sample_rate,
    const OmniVoiceReferenceAudioOptions & options) {
    if (!options.preprocess_prompt) {
        return mono;
    }
    auto mono_pcm16 = engine::audio::float_to_pcm16_clipped(
        mono,
        engine::audio::Pcm16QuantizeMode::TruncateTowardZero);
    if (!options.has_reference_text) {
        const double duration = static_cast<double>(mono_pcm16.size()) / static_cast<double>(sample_rate);
        if (duration > 20.0) {
            auto nonsilent = detect_nonsilent_ranges_ms(mono_pcm16, sample_rate, 100, -40.0F, 10);
            if (!nonsilent.empty()) {
                const int64_t max_ms = 15000;
                const int64_t min_ms = 3000;
                int64_t best_split_ms = 0;
                for (const auto & range : nonsilent) {
                    if (range.first > best_split_ms && range.first <= max_ms) {
                        best_split_ms = range.first;
                    }
                    if (range.second > max_ms) {
                        break;
                    }
                }
                if (best_split_ms < min_ms) {
                    best_split_ms = std::min<int64_t>(max_ms, samples_to_ms(mono_pcm16.size(), sample_rate));
                }
                mono_pcm16 = slice_pcm16_ms(mono_pcm16, sample_rate, 0, best_split_ms);
            }
        }
    }
    mono_pcm16 = remove_mid_silence_pcm16(mono_pcm16, sample_rate, 200, 200, kSilenceThresholdDb);
    mono_pcm16 = trim_edges_pcm16(mono_pcm16, sample_rate, 100, 200, kSilenceThresholdDb);
    if (mono_pcm16.empty()) {
        throw std::runtime_error(
            "OmniVoice reference audio is empty after silence removal; try preprocess_prompt=false");
    }
    return engine::audio::pcm16_to_float_unit_range(mono_pcm16);
}

NormalizedReferenceAudio normalize_reference_audio(
    const runtime::AudioBuffer & audio,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config,
    const OmniVoiceReferenceAudioOptions & options) {
    auto mono = to_mono(audio);
    if (audio.sample_rate != config.sample_rate) {
        mono = resample_sinc_hann_mono(mono, audio.sample_rate, config.sample_rate);
    }
    NormalizedReferenceAudio normalized = {};
    normalized.reference_rms = engine::audio::root_mean_square_or_throw(mono);
    if (normalized.reference_rms > 0.0F && normalized.reference_rms < kTargetReferenceRms) {
        const float scale = kTargetReferenceRms / normalized.reference_rms;
        for (float & sample : mono) {
            sample *= scale;
        }
    }
    mono = preprocess_reference_mono(std::move(mono), config.sample_rate, options);

    const int64_t hop_length = checked_positive(config.hop_length, "hop_length");
    const int64_t remainder = static_cast<int64_t>(mono.size()) % hop_length;
    if (remainder > 0) {
        mono.resize(mono.size() - static_cast<size_t>(remainder));
    }
    if (mono.empty()) {
        throw std::runtime_error("OmniVoice reference audio becomes empty after hop-length alignment");
    }

    normalized.frames = static_cast<int64_t>(mono.size()) / hop_length;
    normalized.acoustic_samples_24k = mono;
    auto semantic = resample_sinc_hann_mono(mono, config.sample_rate, config.semantic_sample_rate);
    semantic.insert(semantic.begin(), 160, 0.0F);
    semantic.insert(semantic.end(), 160, 0.0F);
    normalized.semantic_samples_16k_padded = std::move(semantic);
    return normalized;
}

int64_t conv1d_output_length(int64_t input, int64_t kernel, int stride, int padding, int dilation) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0) {
        throw std::runtime_error("OmniVoice conv1d_output_length received an invalid shape");
    }
    return ((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
}

int64_t acoustic_encoder_output_length(int64_t input_samples, const OmniVoiceHiggsAudioV2TokenizerConfig & config) {
    int64_t length = input_samples;
    length = conv1d_output_length(length, 7, 1, 3, 1);
    for (const int64_t stride_value : config.acoustic_model.downsampling_ratios) {
        const int stride = static_cast<int>(checked_positive(stride_value, "downsampling ratio"));
        length = conv1d_output_length(length, 2 * stride, stride, (stride + 1) / 2, 1);
    }
    length = conv1d_output_length(length, 3, 1, 1, 1);
    return length;
}

int64_t semantic_downsample_factor(const OmniVoiceHiggsAudioV2TokenizerConfig & config) {
    const double factor =
        static_cast<double>(checked_positive(config.hop_length, "hop_length")) /
        (static_cast<double>(checked_positive(config.sample_rate, "sample_rate")) /
         static_cast<double>(checked_positive(config.semantic_sample_rate, "semantic_sample_rate"))) /
        static_cast<double>(checked_positive(config.downsample_factor, "downsample_factor"));
    const auto rounded = static_cast<int64_t>(std::llround(factor));
    if (rounded <= 0 || std::fabs(factor - static_cast<double>(rounded)) > 1.0e-6) {
        throw std::runtime_error("OmniVoice semantic_downsample_factor is not an integer");
    }
    return rounded;
}

int64_t semantic_feature_frames(const OmniVoiceHiggsAudioV2TokenizerConfig & config, int64_t semantic_samples) {
    const auto & semantic = config.semantic_model;
    if (semantic.conv_dim.empty() || semantic.conv_dim.size() != semantic.conv_kernel.size() ||
        semantic.conv_dim.size() != semantic.conv_stride.size()) {
        throw std::runtime_error("OmniVoice HuBERT feature extractor config is incomplete");
    }
    int64_t length = semantic_samples;
    int64_t in_channels = 1;
    for (size_t i = 0; i < semantic.conv_dim.size(); ++i) {
        const int64_t out_channels = semantic.conv_dim[i];
        const int64_t kernel = semantic.conv_kernel[i];
        const int stride = static_cast<int>(semantic.conv_stride[i]);
        length = conv1d_output_length(length, kernel, stride, 0, 1);
        in_channels = out_channels;
    }
    (void) in_channels;
    return length;
}

std::vector<float> pad_acoustic_input_if_needed(
    const std::vector<float> & input,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config,
    int64_t frame_count) {
    const int64_t output_frames = acoustic_encoder_output_length(static_cast<int64_t>(input.size()), config);
    if (output_frames == frame_count) {
        return input;
    }
    if (output_frames != frame_count - 1) {
        throw std::runtime_error("OmniVoice acoustic encoder frame alignment is unsupported");
    }
    std::vector<float> padded(static_cast<size_t>(input.size() + config.hop_length), 0.0F);
    std::copy(input.begin(), input.end(), padded.begin() + config.hop_length / 2);
    if (acoustic_encoder_output_length(static_cast<int64_t>(padded.size()), config) != frame_count) {
        throw std::runtime_error("OmniVoice acoustic encoder pad heuristic does not match reference frame count");
    }
    return padded;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue transpose_bct_to_btc(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1}, value.shape.rank}).build(ctx, value);
}

core::TensorValue transpose_btc_to_bct(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1}, value.shape.rank}).build(ctx, value);
}

core::TensorValue group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const modules::NormWeights & weights,
    core::DeferredTensorWriter * writer = nullptr) {
    core::TensorValue output;
    if (input.shape.rank == 3 && groups == input.shape.dims[1]) {
        if (writer == nullptr) {
            throw std::runtime_error("OmniVoice exact group norm requires a tensor writer");
        }
        auto input_f32 = input.type == GGML_TYPE_F32
            ? input
            : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
        auto mean = modules::ReduceMeanModule({2}).build(ctx, input_f32);
        auto mean_rep = modules::RepeatModule(modules::RepeatConfig{input_f32.shape}).build(ctx, mean);
        auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
        auto centered_sq = modules::MulModule{}.build(ctx, centered, centered);
        auto variance = modules::ReduceMeanModule({2}).build(ctx, centered_sq);

        auto eps_tensor = writer->make_f32_tensor(ctx, core::TensorShape::from_dims({1, 1, 1}), {eps});
        auto eps_rep = modules::RepeatModule(modules::RepeatConfig{variance.shape}).build(ctx, eps_tensor);
        auto std = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, variance, eps_rep));
        auto std_rep = modules::RepeatModule(modules::RepeatConfig{input_f32.shape}).build(ctx, std);
        output = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, std_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    } else {
        output = core::wrap_tensor(ggml_group_norm(ctx.ggml, input.tensor, groups, eps), input.shape, GGML_TYPE_F32);
    }
    if (weights.weight.has_value()) {
        auto weight = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

struct SnakeActivationWeights {
    core::TensorValue alpha;
};

core::TensorValue ensure_f32_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue build_snake1d_exact(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const SnakeActivationWeights & weights,
    int64_t channels) {
    modules::Snake1dWeights snake_weights = {};
    snake_weights.alpha = weights.alpha;
    return modules::Snake1dModule({channels}).build(ctx, input, snake_weights);
}

struct FlatConv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel_size = 0;
};

struct FlatConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
};

core::TensorValue build_conv1d_im2col_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConv1dWeights & weights,
    int stride,
    int padding,
    int dilation);

core::TensorValue apply_bct_sequence_mask(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask) {
    auto repeated = modules::RepeatModule({
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2]}),
    }).build(ctx, mask);
    return modules::MulModule{}.build(ctx, input, repeated);
}

struct ResidualUnitWeights {
    SnakeActivationWeights snake1;
    FlatConv1dWeights conv1;
    SnakeActivationWeights snake2;
    FlatConv1dWeights conv2;
};

struct SemanticEncoderBlockWeights {
    std::vector<FlatConv1dWeights> residual_conv1;
    std::vector<FlatConv1dWeights> residual_conv2;
    FlatConv1dWeights conv;
};

struct AcousticEncoderBlockWeights {
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
    ResidualUnitWeights res3;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv;
};

struct AcousticDecoderBlockWeights {
    SnakeActivationWeights snake;
    FlatConvTranspose1dWeights conv_t;
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
    ResidualUnitWeights res3;
};

struct HubertFeatureExtractorWeights {
    std::vector<FlatConv1dWeights> convs;
    modules::NormWeights first_group_norm;
};

struct HubertFeatureProjectionWeights {
    modules::NormWeights layer_norm;
    modules::LinearWeights projection;
};

struct HubertPositionConvGroupWeights {
    FlatConv1dWeights conv;
};

struct HubertEncoderLayerWeights {
    modules::AttentionWeights attention;
    modules::NormWeights layer_norm;
    modules::FeedForwardWeights feed_forward;
    modules::NormWeights final_layer_norm;
};

struct SemanticEncoderWeights {
    FlatConv1dWeights conv;
    std::vector<SemanticEncoderBlockWeights> blocks;
};

struct AcousticEncoderWeights {
    FlatConv1dWeights conv1;
    std::vector<AcousticEncoderBlockWeights> blocks;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv2;
};

struct AcousticDecoderWeights {
    FlatConv1dWeights conv1;
    std::vector<AcousticDecoderBlockWeights> blocks;
    SnakeActivationWeights snake;
    FlatConv1dWeights conv2;
};

struct QuantizerWeights {
    modules::LinearWeights project_in;
    modules::LinearWeights score;
    core::TensorValue codebook;
    modules::LinearWeights project_out;
};

struct HiggsAudioV2TokenizerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    HubertFeatureExtractorWeights feature_extractor;
    HubertFeatureProjectionWeights feature_projection;
    std::vector<HubertPositionConvGroupWeights> positional_conv_groups;
    modules::NormWeights encoder_input_layer_norm;
    std::vector<HubertEncoderLayerWeights> hubert_layers;
    SemanticEncoderWeights semantic_encoder;
    AcousticEncoderWeights acoustic_encoder;
    modules::LinearWeights fc;
    std::vector<QuantizerWeights> quantizers;
    modules::LinearWeights fc2;
    AcousticDecoderWeights acoustic_decoder;
};

FlatConv1dWeights load_flat_conv1d(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    assets_ns::TensorStorageType storage_type,
    bool use_bias) {
    FlatConv1dWeights weights = {};
    weights.out_channels = out_channels;
    weights.in_channels = in_channels;
    weights.kernel_size = kernel_size;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        {out_channels, in_channels, kernel_size},
        core::TensorShape::from_dims({out_channels, in_channels * kernel_size}));
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets_ns::TensorStorageType::F32, {out_channels});
    }
    return weights;
}

FlatConv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    assets_ns::TensorStorageType storage_type,
    bool use_bias) {
    FlatConv1dWeights weights = {};
    weights.out_channels = out_channels;
    weights.in_channels = in_channels;
    weights.kernel_size = kernel_size;
    weights.weight = store.load_tensor(
        source,
        prefix + ".weight",
        storage_type,
        {out_channels, in_channels, kernel_size});
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets_ns::TensorStorageType::F32, {out_channels});
    }
    return weights;
}

FlatConvTranspose1dWeights load_flat_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    assets_ns::TensorStorageType storage_type,
    bool use_bias) {
    FlatConvTranspose1dWeights weights = {};
    weights.in_channels = in_channels;
    weights.out_channels = out_channels;
    weights.kernel_size = kernel_size;
    weights.weight = store.load_tensor_as_shape(
        source,
        prefix + ".weight",
        storage_type,
        {in_channels, out_channels, kernel_size},
        core::TensorShape::from_dims({in_channels, out_channels * kernel_size}));
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets_ns::TensorStorageType::F32, {out_channels});
    }
    return weights;
}

SnakeActivationWeights load_snake_activation(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    const auto alpha_values = source.require_f32(name, {1, channels, 1});
    SnakeActivationWeights weights = {};
    weights.alpha = store.make_from_f32(
        core::TensorShape::from_dims({channels}),
        assets_ns::TensorStorageType::F32,
        alpha_values);
    return weights;
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    assets_ns::TensorStorageType storage_type,
    bool use_bias) {
    modules::LinearWeights weights = {};
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets_ns::TensorStorageType::F32, {out_features});
    }
    return weights;
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_tensor(source, prefix + ".weight", assets_ns::TensorStorageType::F32, {hidden_size}),
        store.load_tensor(source, prefix + ".bias", assets_ns::TensorStorageType::F32, {hidden_size}),
    };
}

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    assets_ns::TensorStorageType storage_type) {
    modules::AttentionWeights weights = {};
    weights.q_weight = store.load_tensor(source, prefix + ".q_proj.weight", storage_type, {hidden_size, hidden_size});
    weights.q_bias = store.load_tensor(source, prefix + ".q_proj.bias", assets_ns::TensorStorageType::F32, {hidden_size});
    weights.k_weight = store.load_tensor(source, prefix + ".k_proj.weight", storage_type, {hidden_size, hidden_size});
    weights.k_bias = store.load_tensor(source, prefix + ".k_proj.bias", assets_ns::TensorStorageType::F32, {hidden_size});
    weights.v_weight = store.load_tensor(source, prefix + ".v_proj.weight", storage_type, {hidden_size, hidden_size});
    weights.v_bias = store.load_tensor(source, prefix + ".v_proj.bias", assets_ns::TensorStorageType::F32, {hidden_size});
    weights.out_weight = store.load_tensor(source, prefix + ".out_proj.weight", storage_type, {hidden_size, hidden_size});
    weights.out_bias = store.load_tensor(source, prefix + ".out_proj.bias", assets_ns::TensorStorageType::F32, {hidden_size});
    return weights;
}

FlatConv1dWeights reshape_conv1d_weight(core::ModuleBuildContext & ctx, const FlatConv1dWeights & flat) {
    if (flat.weight.shape.rank == 3 &&
        flat.weight.shape.dims[0] == flat.out_channels &&
        flat.weight.shape.dims[1] == flat.in_channels &&
        flat.weight.shape.dims[2] == flat.kernel_size) {
        return flat;
    }
    FlatConv1dWeights reshaped = flat;
    reshaped.weight = core::reshape_tensor(
        ctx,
        flat.weight,
        core::TensorShape::from_dims({flat.out_channels, flat.in_channels, flat.kernel_size}));
    return reshaped;
}

FlatConvTranspose1dWeights reshape_conv_transpose1d_weight(core::ModuleBuildContext & ctx, const FlatConvTranspose1dWeights & flat) {
    FlatConvTranspose1dWeights reshaped = flat;
    reshaped.weight = core::reshape_tensor(
        ctx,
        flat.weight,
        core::TensorShape::from_dims({flat.in_channels, flat.out_channels, flat.kernel_size}));
    return reshaped;
}

modules::Conv1dWeights make_conv1d_weights(core::ModuleBuildContext & ctx, const FlatConv1dWeights & flat) {
    const auto reshaped = reshape_conv1d_weight(ctx, flat);
    return {reshaped.weight, reshaped.bias};
}

modules::ConvTranspose1dWeights make_conv_transpose1d_weights(
    core::ModuleBuildContext & ctx,
    const FlatConvTranspose1dWeights & flat) {
    const auto reshaped = reshape_conv_transpose1d_weight(ctx, flat);
    return {reshaped.weight, reshaped.bias};
}

std::vector<float> apply_positional_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    if (static_cast<int64_t>(g.size()) != kernel_size ||
        static_cast<int64_t>(v.size()) != out_channels * in_channels * kernel_size) {
        throw std::runtime_error("OmniVoice positional weight-norm tensor shape mismatch");
    }
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
        double squared_norm = 0.0;
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t offset = static_cast<size_t>(((out * in_channels) + in) * kernel_size + kernel_index);
                const double value = static_cast<double>(v[offset]);
                squared_norm += value * value;
            }
        }
        const double scale = static_cast<double>(g[static_cast<size_t>(kernel_index)]) /
            std::sqrt(std::max(squared_norm, 1.0e-20));
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t offset = static_cast<size_t>(((out * in_channels) + in) * kernel_size + kernel_index);
                weight[offset] = static_cast<float>(static_cast<double>(v[offset]) * scale);
            }
        }
    }
    return weight;
}

std::vector<float> select_group_kernel(
    const std::vector<float> & values,
    int64_t total_out,
    int64_t total_in_per_group,
    int64_t kernel_size,
    int64_t groups,
    int64_t group_index) {
    const int64_t out_per_group = total_out / groups;
    std::vector<float> group_values(static_cast<size_t>(out_per_group * total_in_per_group * kernel_size), 0.0F);
    for (int64_t out = 0; out < out_per_group; ++out) {
        const int64_t global_out = group_index * out_per_group + out;
        const size_t src_offset = static_cast<size_t>(global_out * total_in_per_group * kernel_size);
        const size_t dst_offset = static_cast<size_t>(out * total_in_per_group * kernel_size);
        std::copy_n(
            values.begin() + static_cast<std::ptrdiff_t>(src_offset),
            total_in_per_group * kernel_size,
            group_values.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }
    return group_values;
}

std::vector<float> select_group_bias(
    const std::vector<float> & bias,
    int64_t groups,
    int64_t group_index) {
    const int64_t out_per_group = static_cast<int64_t>(bias.size()) / groups;
    std::vector<float> group_bias(static_cast<size_t>(out_per_group), 0.0F);
    std::copy_n(
        bias.begin() + static_cast<std::ptrdiff_t>(group_index * out_per_group),
        out_per_group,
        group_bias.begin());
    return group_bias;
}

modules::FeedForwardWeights load_feed_forward(
    core::BackendWeightStore & store,
    const assets_ns::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    int64_t intermediate_size,
    assets_ns::TensorStorageType storage_type) {
    return {
        store.load_tensor(source, prefix + ".intermediate_dense.weight", storage_type, {intermediate_size, hidden_size}),
        store.load_tensor(source, prefix + ".intermediate_dense.bias", assets_ns::TensorStorageType::F32, {intermediate_size}),
        store.load_tensor(source, prefix + ".output_dense.weight", storage_type, {hidden_size, intermediate_size}),
        store.load_tensor(source, prefix + ".output_dense.bias", assets_ns::TensorStorageType::F32, {hidden_size}),
    };
}

std::shared_ptr<const HiggsAudioV2TokenizerWeights> load_weights(
    const OmniVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets_ns::TensorStorageType storage_type) {
    const auto & config = assets.config.audio_tokenizer;
    const auto & source = *assets.audio_tokenizer_weights;
    auto weights = std::make_shared<HiggsAudioV2TokenizerWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "omnivoice.audio_tokenizer.weights",
        weight_context_bytes);

    const auto & semantic = config.semantic_model;
    weights->feature_extractor.convs.reserve(semantic.conv_dim.size());
    int64_t feature_in_channels = 1;
    for (size_t i = 0; i < semantic.conv_dim.size(); ++i) {
        const std::string prefix = "semantic_model.feature_extractor.conv_layers." + std::to_string(i) + ".conv";
        weights->feature_extractor.convs.push_back(load_conv1d(
            *weights->store,
            source,
            prefix,
            semantic.conv_dim[i],
            feature_in_channels,
            semantic.conv_kernel[i],
            storage_type,
            false));
        feature_in_channels = semantic.conv_dim[i];
    }
    weights->feature_extractor.first_group_norm = {
        weights->store->load_tensor(
            source,
            "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight",
            assets_ns::TensorStorageType::F32,
            {semantic.conv_dim.front()}),
        weights->store->load_tensor(
            source,
            "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias",
            assets_ns::TensorStorageType::F32,
            {semantic.conv_dim.front()}),
    };

    weights->feature_projection.layer_norm = {
        weights->store->load_tensor(
            source,
            "semantic_model.feature_projection.layer_norm.weight",
            assets_ns::TensorStorageType::F32,
            {semantic.conv_dim.back()}),
        weights->store->load_tensor(
            source,
            "semantic_model.feature_projection.layer_norm.bias",
            assets_ns::TensorStorageType::F32,
            {semantic.conv_dim.back()}),
    };
    weights->feature_projection.projection = load_linear(
        *weights->store,
        source,
        "semantic_model.feature_projection.projection",
        semantic.hidden_size,
        semantic.conv_dim.back(),
        assets_ns::TensorStorageType::F32,
        true);

    if (semantic.do_stable_layer_norm) {
        throw std::runtime_error("OmniVoice HuBERT stable-layer-norm variant is not implemented");
    }
    if (semantic.hidden_size % semantic.num_conv_pos_embedding_groups != 0) {
        throw std::runtime_error("OmniVoice HuBERT positional conv groups do not divide hidden_size");
    }
    const auto pos_g = source.require_f32(
        "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        {1, 1, semantic.num_conv_pos_embeddings});
    const auto pos_v = source.require_f32(
        "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        {semantic.hidden_size, semantic.hidden_size / semantic.num_conv_pos_embedding_groups, semantic.num_conv_pos_embeddings});
    const auto pos_bias = source.require_f32(
        "semantic_model.encoder.pos_conv_embed.conv.bias",
        {semantic.hidden_size});
    const auto pos_weight = apply_positional_weight_norm(
        pos_g,
        pos_v,
        semantic.hidden_size,
        semantic.hidden_size / semantic.num_conv_pos_embedding_groups,
        semantic.num_conv_pos_embeddings);
    const int64_t pos_groups = semantic.num_conv_pos_embedding_groups;
    const int64_t pos_channels_per_group = semantic.hidden_size / pos_groups;
    weights->positional_conv_groups.reserve(static_cast<size_t>(pos_groups));
    for (int64_t group_index = 0; group_index < pos_groups; ++group_index) {
        const auto group_weight = select_group_kernel(
            pos_weight,
            semantic.hidden_size,
            pos_channels_per_group,
            semantic.num_conv_pos_embeddings,
            pos_groups,
            group_index);
        const auto group_bias = select_group_bias(pos_bias, pos_groups, group_index);
        HubertPositionConvGroupWeights group = {};
        group.conv.out_channels = pos_channels_per_group;
        group.conv.in_channels = pos_channels_per_group;
        group.conv.kernel_size = semantic.num_conv_pos_embeddings;
        group.conv.weight = weights->store->make_from_f32(
            core::TensorShape::from_dims({pos_channels_per_group, pos_channels_per_group * semantic.num_conv_pos_embeddings}),
            storage_type,
            group_weight);
        group.conv.bias = weights->store->make_from_f32(
            core::TensorShape::from_dims({pos_channels_per_group}),
            assets_ns::TensorStorageType::F32,
            group_bias);
        weights->positional_conv_groups.push_back(std::move(group));
    }

    weights->encoder_input_layer_norm = load_norm(
        *weights->store,
        source,
        "semantic_model.encoder.layer_norm",
        semantic.hidden_size);
    weights->hubert_layers.reserve(static_cast<size_t>(semantic.num_hidden_layers));
    for (int64_t layer = 0; layer < semantic.num_hidden_layers; ++layer) {
        const std::string prefix = "semantic_model.encoder.layers." + std::to_string(layer);
        HubertEncoderLayerWeights layer_weights = {};
        layer_weights.attention = load_attention(
            *weights->store,
            source,
            prefix + ".attention",
            semantic.hidden_size,
            storage_type);
        layer_weights.layer_norm = load_norm(*weights->store, source, prefix + ".layer_norm", semantic.hidden_size);
        layer_weights.feed_forward = load_feed_forward(
            *weights->store,
            source,
            prefix + ".feed_forward",
            semantic.hidden_size,
            semantic.intermediate_size,
            storage_type);
        layer_weights.final_layer_norm = load_norm(
            *weights->store,
            source,
            prefix + ".final_layer_norm",
            semantic.hidden_size);
        weights->hubert_layers.push_back(std::move(layer_weights));
    }

    weights->semantic_encoder.conv = load_flat_conv1d(
        *weights->store,
        source,
        "encoder_semantic.conv",
        semantic.hidden_size,
        semantic.hidden_size,
        config.kernel_size,
        storage_type,
        false);
    weights->semantic_encoder.blocks.reserve(config.strides.size());
    int64_t semantic_channels = semantic.hidden_size;
    for (size_t block_index = 0; block_index < config.strides.size(); ++block_index) {
        SemanticEncoderBlockWeights block = {};
        block.residual_conv1.reserve(config.block_dilations.size());
        block.residual_conv2.reserve(config.block_dilations.size());
        for (size_t residual_index = 0; residual_index < config.block_dilations.size(); ++residual_index) {
            const std::string prefix =
                "encoder_semantic.conv_blocks." + std::to_string(block_index) + ".res_units." + std::to_string(residual_index);
            block.residual_conv1.push_back(load_flat_conv1d(
                *weights->store,
                source,
                prefix + ".conv1",
                semantic_channels,
                semantic_channels,
                config.unit_kernel_size,
                storage_type,
                false));
            block.residual_conv2.push_back(load_flat_conv1d(
                *weights->store,
                source,
                prefix + ".conv2",
                semantic_channels,
                semantic_channels,
                1,
                storage_type,
                false));
        }
        block.conv = load_flat_conv1d(
            *weights->store,
            source,
            "encoder_semantic.conv_blocks." + std::to_string(block_index) + ".conv",
            semantic_channels,
            semantic_channels,
            config.strides[block_index] == 1 ? 3 : (2 * config.strides[block_index]),
            storage_type,
            true);
        weights->semantic_encoder.blocks.push_back(std::move(block));
    }

    const auto & acoustic = config.acoustic_model;
    weights->acoustic_encoder.conv1 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_encoder.conv1",
        acoustic.encoder_hidden_size,
        1,
        7,
        storage_type,
        true);
    weights->acoustic_encoder.blocks.reserve(acoustic.downsampling_ratios.size());
    int64_t encoder_block_input = acoustic.encoder_hidden_size;
    for (size_t block_index = 0; block_index < acoustic.downsampling_ratios.size(); ++block_index) {
        const int64_t stride = acoustic.downsampling_ratios[block_index];
        const int64_t output_channels = acoustic.encoder_hidden_size * (1ll << (static_cast<int64_t>(block_index) + 1));
        const std::string block_prefix = "acoustic_encoder.block." + std::to_string(block_index);
        auto load_residual_unit = [&](const std::string & prefix, int64_t channels) {
            ResidualUnitWeights unit = {};
            unit.snake1 = load_snake_activation(*weights->store, source, prefix + ".snake1.alpha", channels);
            unit.conv1 = load_flat_conv1d(*weights->store, source, prefix + ".conv1", channels, channels, 7, storage_type, true);
            unit.snake2 = load_snake_activation(*weights->store, source, prefix + ".snake2.alpha", channels);
            unit.conv2 = load_flat_conv1d(*weights->store, source, prefix + ".conv2", channels, channels, 1, storage_type, true);
            return unit;
        };
        AcousticEncoderBlockWeights block = {};
        block.res1 = load_residual_unit(block_prefix + ".res_unit1", encoder_block_input);
        block.res2 = load_residual_unit(block_prefix + ".res_unit2", encoder_block_input);
        block.res3 = load_residual_unit(block_prefix + ".res_unit3", encoder_block_input);
        block.snake = load_snake_activation(*weights->store, source, block_prefix + ".snake1.alpha", encoder_block_input);
        block.conv = load_flat_conv1d(
            *weights->store,
            source,
            block_prefix + ".conv1",
            output_channels,
            encoder_block_input,
            2 * stride,
            storage_type,
            true);
        weights->acoustic_encoder.blocks.push_back(std::move(block));
        encoder_block_input = output_channels;
    }
    weights->acoustic_encoder.snake =
        load_snake_activation(*weights->store, source, "acoustic_encoder.snake1.alpha", encoder_block_input);
    weights->acoustic_encoder.conv2 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_encoder.conv2",
        acoustic.hidden_size,
        encoder_block_input,
        3,
        storage_type,
        true);

    weights->fc = load_linear(
        *weights->store,
        source,
        "fc",
        config.hidden_size,
        config.hidden_size,
        storage_type,
        true);

    weights->quantizers.reserve(static_cast<size_t>(config.num_codebooks));
    for (int64_t quantizer_index = 0; quantizer_index < config.num_codebooks; ++quantizer_index) {
        const std::string prefix = "quantizer.quantizers." + std::to_string(quantizer_index);
        const auto codebook = source.require_f32(prefix + ".codebook.embed", {config.codebook_size, config.codebook_dim});
        std::vector<float> score_weight(static_cast<size_t>(config.codebook_size * config.codebook_dim), 0.0F);
        std::vector<float> score_bias(static_cast<size_t>(config.codebook_size), 0.0F);
        for (int64_t row = 0; row < config.codebook_size; ++row) {
            double norm = 0.0;
            for (int64_t col = 0; col < config.codebook_dim; ++col) {
                const float value = codebook[static_cast<size_t>(row * config.codebook_dim + col)];
                score_weight[static_cast<size_t>(row * config.codebook_dim + col)] = 2.0F * value;
                norm += static_cast<double>(value) * static_cast<double>(value);
            }
            score_bias[static_cast<size_t>(row)] = static_cast<float>(-norm);
        }
        QuantizerWeights quantizer = {};
        quantizer.project_in = load_linear(
            *weights->store,
            source,
            prefix + ".project_in",
            config.codebook_dim,
            config.hidden_size,
            storage_type,
            true);
        quantizer.score = {
            weights->store->make_from_f32(
                core::TensorShape::from_dims({config.codebook_size, config.codebook_dim}),
                assets_ns::TensorStorageType::F32,
                score_weight),
            weights->store->make_from_f32(
                core::TensorShape::from_dims({config.codebook_size}),
                assets_ns::TensorStorageType::F32,
                score_bias),
        };
        quantizer.codebook = weights->store->load_tensor(
            source,
            prefix + ".codebook.embed",
            assets_ns::TensorStorageType::F32,
            {config.codebook_size, config.codebook_dim});
        quantizer.project_out = load_linear(
            *weights->store,
            source,
            prefix + ".project_out",
            config.hidden_size,
            config.codebook_dim,
            storage_type,
            true);
        weights->quantizers.push_back(std::move(quantizer));
    }

    weights->fc2 = load_linear(
        *weights->store,
        source,
        "fc2",
        acoustic.hidden_size,
        config.hidden_size,
        assets_ns::TensorStorageType::F32,
        true);

    weights->acoustic_decoder.conv1 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_decoder.conv1",
        acoustic.decoder_hidden_size,
        acoustic.hidden_size,
        7,
        storage_type,
        true);
    weights->acoustic_decoder.blocks.reserve(acoustic.upsampling_ratios.size());
    for (size_t block_index = 0; block_index < acoustic.upsampling_ratios.size(); ++block_index) {
        const int64_t stride = acoustic.upsampling_ratios[block_index];
        const int64_t input_channels = acoustic.decoder_hidden_size / (1ll << static_cast<int64_t>(block_index));
        const int64_t output_channels = acoustic.decoder_hidden_size / (1ll << (static_cast<int64_t>(block_index) + 1));
        const std::string block_prefix = "acoustic_decoder.block." + std::to_string(block_index);
        auto load_residual_unit = [&](const std::string & prefix, int64_t channels) {
            ResidualUnitWeights unit = {};
            unit.snake1 = load_snake_activation(*weights->store, source, prefix + ".snake1.alpha", channels);
            unit.conv1 = load_flat_conv1d(*weights->store, source, prefix + ".conv1", channels, channels, 7, storage_type, true);
            unit.snake2 = load_snake_activation(*weights->store, source, prefix + ".snake2.alpha", channels);
            unit.conv2 = load_flat_conv1d(*weights->store, source, prefix + ".conv2", channels, channels, 1, storage_type, true);
            return unit;
        };
        AcousticDecoderBlockWeights block = {};
        block.snake = load_snake_activation(*weights->store, source, block_prefix + ".snake1.alpha", input_channels);
        block.conv_t = load_flat_conv_transpose1d(
            *weights->store,
            source,
            block_prefix + ".conv_t1",
            input_channels,
            output_channels,
            2 * stride,
            storage_type,
            true);
        block.res1 = load_residual_unit(block_prefix + ".res_unit1", output_channels);
        block.res2 = load_residual_unit(block_prefix + ".res_unit2", output_channels);
        block.res3 = load_residual_unit(block_prefix + ".res_unit3", output_channels);
        weights->acoustic_decoder.blocks.push_back(std::move(block));
    }
    const int64_t final_decoder_channels =
        acoustic.decoder_hidden_size / (1ll << static_cast<int64_t>(acoustic.upsampling_ratios.size()));
    weights->acoustic_decoder.snake =
        load_snake_activation(*weights->store, source, "acoustic_decoder.snake1.alpha", final_decoder_channels);
    weights->acoustic_decoder.conv2 = load_flat_conv1d(
        *weights->store,
        source,
        "acoustic_decoder.conv2",
        1,
        final_decoder_channels,
        7,
        storage_type,
        true);

    weights->store->upload();
    return weights;
}

core::TensorValue build_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels,
    int dilation) {
    auto x = build_snake1d_exact(ctx, input, weights.snake1, channels);
    x = modules::Conv1dModule({channels, channels, 7, 1, 3 * dilation, dilation, true})
            .build(ctx, x, make_conv1d_weights(ctx, weights.conv1));
    x = build_snake1d_exact(ctx, x, weights.snake2, channels);
    x = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true})
            .build(ctx, x, make_conv1d_weights(ctx, weights.conv2));
    return modules::AddModule().build(ctx, input, x);
}

core::TensorValue build_semantic_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConv1dWeights & conv1,
    const FlatConv1dWeights & conv2,
    int64_t channels,
    int dilation) {
    auto x = modules::EluModule().build(ctx, input);
    const int padding = static_cast<int>(((conv1.kernel_size - 1) / 2) * static_cast<int64_t>(dilation));
    x = modules::Conv1dModule({channels, channels, conv1.kernel_size, 1, padding, dilation, false})
            .build(ctx, x, make_conv1d_weights(ctx, conv1));
    x = modules::EluModule().build(ctx, x);
    x = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, false})
            .build(ctx, x, make_conv1d_weights(ctx, conv2));
    return modules::AddModule().build(ctx, input, x);
}

core::TensorValue build_conv_transpose_with_crop(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConvTranspose1dWeights & weights,
    int stride,
    int padding,
    int output_padding) {
    auto output = modules::ConvTranspose1dModule({
        weights.in_channels,
        weights.out_channels,
        weights.kernel_size,
        stride,
        0,
        1,
        weights.bias.has_value(),
    }).build(ctx, input, make_conv_transpose1d_weights(ctx, weights));
    const int64_t cropped_length =
        (input.shape.dims[2] - 1) * stride - 2 * padding + weights.kernel_size + output_padding;
    auto cropped = core::wrap_tensor(
        ggml_cont(
            ctx.ggml,
            ggml_view_3d(
                ctx.ggml,
                output.tensor,
                cropped_length,
                weights.out_channels,
                1,
                output.tensor->nb[1],
                output.tensor->nb[2],
                static_cast<size_t>(padding) * output.tensor->nb[0])),
        core::TensorShape::from_dims({1, weights.out_channels, cropped_length}),
        GGML_TYPE_F32);
    return cropped;
}

core::TensorValue build_conv1d_im2col_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FlatConv1dWeights & weights,
    int stride,
    int padding,
    int dilation) {
    core::validate_rank_between(input, 3, 3, "input");
    const int64_t output_frames =
        (input.shape.dims[2] + 2 * padding - dilation * (weights.kernel_size - 1) - 1) / stride + 1;
    if (output_frames <= 0) {
        throw std::runtime_error("OmniVoice exact conv1d computed non-positive output length");
    }
    auto input_f32 = ensure_f32_tensor(ctx, ensure_contiguous(ctx, input));
    auto reshaped = reshape_conv1d_weight(ctx, weights);
    auto weight_f32 = ensure_f32_tensor(ctx, ensure_contiguous(ctx, reshaped.weight));
    ggml_tensor * im2col = ggml_im2col(
        ctx.ggml,
        weight_f32.tensor,
        input_f32.tensor,
        stride,
        0,
        padding,
        0,
        dilation,
        0,
        false,
        GGML_TYPE_F32);
    auto im2col_btk = core::wrap_tensor(
        im2col,
        core::TensorShape::from_dims({input.shape.dims[0], output_frames, weights.in_channels * weights.kernel_size}),
        GGML_TYPE_F32);
    im2col_btk = ensure_contiguous(ctx, im2col_btk);
    modules::LinearWeights linear_weights = {};
    linear_weights.weight = core::reshape_tensor(
        ctx,
        reshaped.weight,
        core::TensorShape::from_dims({weights.out_channels, weights.in_channels * weights.kernel_size}));
    if (weights.bias.has_value()) {
        linear_weights.bias = *weights.bias;
    }
    auto output_bto = modules::LinearModule({
        weights.in_channels * weights.kernel_size,
        weights.out_channels,
        weights.bias.has_value(),
    }).build(ctx, im2col_btk, linear_weights);
    return transpose_btc_to_bct(ctx, output_bto);
}

core::TensorValue build_linear_btc_as_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features,
    bool use_bias) {
    auto input_bct = transpose_btc_to_bct(ctx, input);
    modules::Conv1dWeights conv_weights = {
        core::reshape_tensor(ctx, weights.weight, core::TensorShape::from_dims({out_features, in_features, 1})),
        weights.bias,
    };
    auto output_bct = modules::Conv1dModule({
        in_features,
        out_features,
        1,
        1,
        0,
        1,
        use_bias,
    }).build(ctx, input_bct, conv_weights);
    return transpose_bct_to_btc(ctx, output_bct);
}

core::TensorValue build_grouped_positional_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::vector<HubertPositionConvGroupWeights> & groups,
    int64_t total_channels,
    int64_t kernel_size) {
    if (groups.empty()) {
        throw std::runtime_error("OmniVoice positional conv group weights are empty");
    }
    const int64_t channels_per_group = total_channels / static_cast<int64_t>(groups.size());
    std::optional<core::TensorValue> output;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto input_slice = modules::SliceModule({
            1,
            static_cast<int64_t>(group_index) * channels_per_group,
            channels_per_group,
        }).build(ctx, input);
        auto conv = modules::Conv1dModule({
            channels_per_group,
            channels_per_group,
            kernel_size,
            1,
            static_cast<int>(kernel_size / 2),
            1,
            true,
        }).build(ctx, input_slice, make_conv1d_weights(ctx, groups[group_index].conv));
        output = output.has_value() ? modules::ConcatModule({1}).build(ctx, *output, conv) : conv;
    }
    auto combined = *output;
    if (kernel_size % 2 == 0) {
        combined = modules::SliceModule({2, 0, input.shape.dims[2]}).build(ctx, combined);
    }
    return modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, combined);
}

core::TensorValue build_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::AttentionWeights & weights,
    int64_t hidden_size,
    int64_t num_heads) {
    if (hidden_size <= 0 || num_heads <= 0 || (hidden_size % num_heads) != 0) {
        throw std::runtime_error("OmniVoice HuBERT flash self-attention received an invalid hidden shape");
    }
    const int64_t head_dim = hidden_size / num_heads;
    const modules::LinearModule q_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule k_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule v_proj({hidden_size, hidden_size, true, GGML_PREC_F32});
    const modules::LinearModule out_proj({hidden_size, hidden_size, true, GGML_PREC_F32});

    auto q = q_proj.build(ctx, input, {
        weights.q_weight,
        weights.q_bias,
    });
    auto k = k_proj.build(ctx, input, {
        weights.k_weight,
        weights.k_bias,
    });
    auto v = v_proj.build(ctx, input, {
        weights.v_weight,
        weights.v_bias,
    });

    q = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, q),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    k = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, k),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    v = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, v),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    scores = ensure_contiguous(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context)),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    return out_proj.build(ctx, context, {
        weights.out_weight,
        weights.out_bias,
    });
}

core::TensorValue build_hubert_sequence_mean(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const HiggsAudioV2TokenizerWeights & weights,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config) {
    const auto & semantic = config.semantic_model;
    auto x = hidden;
    auto summed = x;
    const modules::LayerNormModule layer_norm({semantic.hidden_size, semantic.layer_norm_eps, true, true});
    const modules::FeedForwardModule feed_forward({
        semantic.hidden_size,
        semantic.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
    });

    for (const auto & layer_weights : weights.hubert_layers) {
        auto attn = build_self_attention(
            ctx,
            x,
            layer_weights.attention,
            semantic.hidden_size,
            semantic.num_attention_heads);
        x = modules::AddModule().build(ctx, x, attn);
        x = layer_norm.build(ctx, x, layer_weights.layer_norm);
        auto ff = feed_forward.build(ctx, x, layer_weights.feed_forward);
        x = modules::AddModule().build(ctx, x, ff);
        x = layer_norm.build(ctx, x, layer_weights.final_layer_norm);
        summed = modules::AddModule().build(ctx, summed, x);
    }
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, summed.tensor, 1.0F / static_cast<float>(weights.hubert_layers.size() + 1)),
        summed.shape,
        GGML_TYPE_F32);
}

core::TensorValue build_semantic_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioV2TokenizerWeights & weights,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config) {
    const auto & semantic = config.semantic_model;
    auto x = modules::Conv1dModule({
        semantic.hidden_size,
        semantic.hidden_size,
        config.kernel_size,
        1,
        static_cast<int>(config.kernel_size / 2),
        1,
        false,
    }).build(ctx, input_bct, make_conv1d_weights(ctx, weights.semantic_encoder.conv));
    for (size_t block_index = 0; block_index < weights.semantic_encoder.blocks.size(); ++block_index) {
        const auto & block = weights.semantic_encoder.blocks[block_index];
        for (size_t residual_index = 0; residual_index < block.residual_conv1.size(); ++residual_index) {
            x = build_semantic_residual_unit(
                ctx,
                x,
                block.residual_conv1[residual_index],
                block.residual_conv2[residual_index],
                semantic.hidden_size,
                static_cast<int>(config.block_dilations[residual_index]));
        }
        const int stride = static_cast<int>(config.strides[block_index]);
        const int kernel = stride == 1 ? 3 : static_cast<int>(2 * config.strides[block_index]);
        const int padding = (kernel - 1) / 2;
        x = modules::Conv1dModule({
            semantic.hidden_size,
            semantic.hidden_size,
            kernel,
            stride,
            padding,
            1,
            true,
        }).build(ctx, x, make_conv1d_weights(ctx, block.conv));
    }
    return x;
}

core::TensorValue build_acoustic_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioV2TokenizerWeights & weights,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config) {
    const auto & acoustic = config.acoustic_model;
    auto x = modules::Conv1dModule({1, acoustic.encoder_hidden_size, 7, 1, 3, 1, true})
                 .build(ctx, input_bct, make_conv1d_weights(ctx, weights.acoustic_encoder.conv1));
    int64_t channels = acoustic.encoder_hidden_size;
    for (size_t block_index = 0; block_index < weights.acoustic_encoder.blocks.size(); ++block_index) {
        const auto & block = weights.acoustic_encoder.blocks[block_index];
        x = build_residual_unit(ctx, x, block.res1, channels, 1);
        x = build_residual_unit(ctx, x, block.res2, channels, 3);
        x = build_residual_unit(ctx, x, block.res3, channels, 9);
        x = build_snake1d_exact(ctx, x, block.snake, channels);
        const int stride = static_cast<int>(acoustic.downsampling_ratios[block_index]);
        const int64_t next_channels = acoustic.encoder_hidden_size * (1ll << (static_cast<int64_t>(block_index) + 1));
        x = modules::Conv1dModule({
            channels,
            next_channels,
            2 * stride,
            stride,
            (stride + 1) / 2,
            1,
            true,
        }).build(ctx, x, make_conv1d_weights(ctx, block.conv));
        channels = next_channels;
    }
    x = build_snake1d_exact(ctx, x, weights.acoustic_encoder.snake, channels);
    return modules::Conv1dModule({channels, acoustic.hidden_size, 3, 1, 1, 1, true})
        .build(ctx, x, make_conv1d_weights(ctx, weights.acoustic_encoder.conv2));
}

core::TensorValue build_acoustic_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const HiggsAudioV2TokenizerWeights & weights,
    const OmniVoiceHiggsAudioV2TokenizerConfig & config,
    const core::TensorValue * frame_mask,
    const std::vector<core::TensorValue> * block_masks) {
    const auto & acoustic = config.acoustic_model;
    auto x = modules::Conv1dModule({
        acoustic.hidden_size,
        acoustic.decoder_hidden_size,
        7,
        1,
        3,
        1,
        true,
    }).build(ctx, input_bct, make_conv1d_weights(ctx, weights.acoustic_decoder.conv1));
    if (frame_mask != nullptr) {
        x = apply_bct_sequence_mask(ctx, x, *frame_mask);
    }
    int64_t channels = acoustic.decoder_hidden_size;
    for (size_t block_index = 0; block_index < weights.acoustic_decoder.blocks.size(); ++block_index) {
        const auto & block = weights.acoustic_decoder.blocks[block_index];
        const core::TensorValue * block_mask = nullptr;
        if (block_masks != nullptr) {
            block_mask = &block_masks->at(block_index);
        }
        const int stride = static_cast<int>(acoustic.upsampling_ratios[block_index]);
        const int padding = (stride + 1) / 2;
        const int output_padding = stride % 2;
        x = build_snake1d_exact(ctx, x, block.snake, channels);
        x = build_conv_transpose_with_crop(ctx, x, block.conv_t, stride, padding, output_padding);
        if (block_mask != nullptr) {
            x = apply_bct_sequence_mask(ctx, x, *block_mask);
        }
        channels = block.conv_t.out_channels;
        x = build_residual_unit(ctx, x, block.res1, channels, 1);
        if (block_mask != nullptr) {
            x = apply_bct_sequence_mask(ctx, x, *block_mask);
        }
        x = build_residual_unit(ctx, x, block.res2, channels, 3);
        if (block_mask != nullptr) {
            x = apply_bct_sequence_mask(ctx, x, *block_mask);
        }
        x = build_residual_unit(ctx, x, block.res3, channels, 9);
        if (block_mask != nullptr) {
            x = apply_bct_sequence_mask(ctx, x, *block_mask);
        }
    }
    x = build_snake1d_exact(ctx, x, weights.acoustic_decoder.snake, channels);
    return modules::Conv1dModule({channels, 1, 7, 1, 3, 1, true})
        .build(ctx, x, make_conv1d_weights(ctx, weights.acoustic_decoder.conv2));
}

core::TensorValue build_quantizer_decode_sequence(
    core::ModuleBuildContext & ctx,
    const std::vector<ggml_tensor *> & code_inputs,
    const HiggsAudioV2TokenizerWeights & weights,
    int64_t frames) {
    std::optional<core::TensorValue> latent;
    for (size_t quantizer_index = 0; quantizer_index < code_inputs.size(); ++quantizer_index) {
        const auto codes = core::wrap_tensor(
            code_inputs[quantizer_index],
            core::TensorShape::from_dims({1, frames}),
            GGML_TYPE_I32);
        auto embedded = modules::CodebookLookupModule({
            static_cast<int64_t>(weights.quantizers[quantizer_index].codebook.shape.dims[0]),
            static_cast<int64_t>(weights.quantizers[quantizer_index].codebook.shape.dims[1]),
        }).build(ctx, codes, weights.quantizers[quantizer_index].codebook);
        auto projected = modules::LinearModule({
            embedded.shape.dims[2],
            weights.quantizers[quantizer_index].project_out.weight.shape.dims[0],
            true,
        }).build(ctx, embedded, weights.quantizers[quantizer_index].project_out);
        auto projected_bct = transpose_btc_to_bct(ctx, projected);
        latent = latent.has_value() ? modules::AddModule().build(ctx, *latent, projected_bct) : projected_bct;
    }
    if (!latent.has_value()) {
        throw std::runtime_error("OmniVoice quantizer decode requires at least one codebook");
    }
    return *latent;
}

struct HiggsAudioV2EncoderGraph {
    HiggsAudioV2EncoderGraph(
        std::shared_ptr<const OmniVoiceAssets> assets,
        std::shared_ptr<const HiggsAudioV2TokenizerWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t acoustic_samples,
        int64_t semantic_samples,
        int64_t frames)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          compute_threads_(std::max(1, execution_context.config().threads)),
          acoustic_sample_capacity_(acoustic_samples),
          semantic_sample_capacity_(semantic_samples),
          frame_capacity_(frames) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice audio-tokenizer encoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.audio_tokenizer.encode", backend_type_};
        const auto & config = assets_->config.audio_tokenizer;
        const auto & semantic = config.semantic_model;
        semantic_downsample_factor_ = semantic_downsample_factor(config);

        acoustic_input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, acoustic_sample_capacity_}));
        semantic_input_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, semantic_sample_capacity_}));
        downsample_indices_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frame_capacity_}));
        ggml_set_input(acoustic_input_.tensor);
        ggml_set_input(semantic_input_.tensor);
        ggml_set_input(downsample_indices_.tensor);

        auto hubert = semantic_input_;
        for (size_t i = 0; i < weights_->feature_extractor.convs.size(); ++i) {
            const auto & conv_weights = weights_->feature_extractor.convs[i];
            hubert = build_conv1d_im2col_f32(
                ctx,
                hubert,
                conv_weights,
                static_cast<int>(semantic.conv_stride[i]),
                0,
                1);
            hubert = ensure_contiguous(ctx, hubert);
            if (i == 0) {
                hubert = group_norm_affine(
                    ctx,
                    hubert,
                    semantic.conv_dim.front(),
                    semantic.layer_norm_eps,
                    weights_->feature_extractor.first_group_norm,
                    &tensor_writer_);
            }
            hubert = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hubert);
        }
        hubert = transpose_bct_to_btc(ctx, hubert);
        if (semantic.feat_proj_layer_norm) {
            hubert = modules::LayerNormModule({
                semantic.conv_dim.back(),
                semantic.layer_norm_eps,
                true,
                true,
            }).build(ctx, hubert, weights_->feature_projection.layer_norm);
        }
        hubert = build_linear_btc_as_conv1d(
            ctx,
            hubert,
            weights_->feature_projection.projection,
            semantic.conv_dim.back(),
            semantic.hidden_size,
            true);

        auto position = build_grouped_positional_conv(ctx, transpose_btc_to_bct(ctx, hubert), weights_->positional_conv_groups, semantic.hidden_size, semantic.num_conv_pos_embeddings);
        hubert = modules::AddModule().build(ctx, hubert, transpose_bct_to_btc(ctx, position));
        hubert = modules::LayerNormModule({
            semantic.hidden_size,
            semantic.layer_norm_eps,
            true,
            true,
        }).build(ctx, hubert, weights_->encoder_input_layer_norm);
        hubert = build_hubert_sequence_mean(ctx, hubert, *weights_, config);

        auto hubert_flat = core::reshape_tensor(ctx, ensure_contiguous(ctx, hubert), core::TensorShape::from_dims({hubert.shape.dims[1], hubert.shape.dims[2]}));
        auto semantic_downsampled = modules::EmbeddingModule({
            hubert.shape.dims[1],
            hubert.shape.dims[2],
        }).build(ctx, downsample_indices_, hubert_flat);
        semantic_downsampled = core::reshape_tensor(
            ctx,
            semantic_downsampled,
            core::TensorShape::from_dims({1, frame_capacity_, semantic.hidden_size}));
        auto semantic_encoded = build_semantic_encoder(ctx, transpose_btc_to_bct(ctx, semantic_downsampled), *weights_, config);
        auto acoustic_encoded = build_acoustic_encoder(ctx, acoustic_input_, *weights_, config);
        if (semantic_encoded.shape.dims[2] != acoustic_encoded.shape.dims[2]) {
            throw std::runtime_error("OmniVoice semantic/acoustic encoder frame counts do not match");
        }
        auto embeddings_bct = modules::ConcatModule({1}).build(ctx, acoustic_encoded, semantic_encoded);
        auto embeddings_btc = transpose_bct_to_btc(ctx, embeddings_bct);
        embeddings_btc = modules::LinearModule({
            config.hidden_size,
            config.hidden_size,
            true,
            GGML_PREC_F32,
        }).build(ctx, embeddings_btc, weights_->fc);
        auto residual_bct = transpose_btc_to_bct(ctx, embeddings_btc);

        code_outputs_.reserve(static_cast<size_t>(config.num_codebooks));
        for (int64_t quantizer_index = 0; quantizer_index < config.num_codebooks; ++quantizer_index) {
            const auto & quantizer = weights_->quantizers[static_cast<size_t>(quantizer_index)];
            auto residual_btc = transpose_bct_to_btc(ctx, residual_bct);
            auto projected = modules::LinearModule({
                config.hidden_size,
                config.codebook_dim,
                true,
                GGML_PREC_F32,
            }).build(ctx, residual_btc, quantizer.project_in);
            auto logits = modules::LinearModule({
                config.codebook_dim,
                config.codebook_size,
                true,
                GGML_PREC_F32,
            }).build(ctx, projected, quantizer.score);
            auto logits_flat = core::reshape_tensor(
                ctx,
                ensure_contiguous(ctx, logits),
                core::TensorShape::from_dims({frame_capacity_, config.codebook_size}));
            auto ids_raw = ggml_argmax(ctx.ggml, logits_flat.tensor);
            ggml_set_output(ids_raw);
            code_outputs_.push_back(ids_raw);
            auto ids = core::reshape_tensor(
                ctx,
                core::wrap_tensor(ids_raw, core::TensorShape::from_dims({frame_capacity_}), GGML_TYPE_I32),
                core::TensorShape::from_dims({1, frame_capacity_}));
            auto embedded = modules::CodebookLookupModule({
                config.codebook_size,
                config.codebook_dim,
            }).build(ctx, ids, quantizer.codebook);
            auto quantized = modules::LinearModule({
                config.codebook_dim,
                config.hidden_size,
                true,
                GGML_PREC_F32,
            }).build(ctx, embedded, quantizer.project_out);
            auto quantized_bct = transpose_btc_to_bct(ctx, quantized);
            residual_bct = core::wrap_tensor(
                ggml_sub(ctx.ggml, residual_bct.tensor, quantized_bct.tensor),
                residual_bct.shape,
                GGML_TYPE_F32);
        }

        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (ggml_tensor * code_output : code_outputs_) {
            ggml_build_forward_expand(graph_, code_output);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate OmniVoice audio-tokenizer encoder graph");
        }
        tensor_writer_.flush();
    }

    ~HiggsAudioV2EncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        int64_t acoustic_samples,
        int64_t semantic_samples,
        int64_t frames,
        ggml_backend_t backend,
        int threads) const {
        return acoustic_sample_capacity_ >= acoustic_samples &&
            semantic_sample_capacity_ >= semantic_samples &&
            frame_capacity_ >= frames &&
            backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    OmniVoiceAudioTokens run(const NormalizedReferenceAudio & audio) {
        const int64_t actual_frames = audio.frames;
        if (static_cast<int64_t>(audio.acoustic_samples_24k.size()) > acoustic_sample_capacity_ ||
            static_cast<int64_t>(audio.semantic_samples_16k_padded.size()) > semantic_sample_capacity_ ||
            actual_frames > frame_capacity_) {
            throw std::runtime_error("OmniVoice audio-tokenizer encoder request exceeds prepared capacity");
        }
        std::vector<float> acoustic_padded(static_cast<size_t>(acoustic_sample_capacity_), 0.0F);
        std::copy(audio.acoustic_samples_24k.begin(), audio.acoustic_samples_24k.end(), acoustic_padded.begin());
        core::write_tensor_f32(acoustic_input_, acoustic_padded);
        std::vector<float> semantic_padded(static_cast<size_t>(semantic_sample_capacity_), 0.0F);
        std::copy(audio.semantic_samples_16k_padded.begin(), audio.semantic_samples_16k_padded.end(), semantic_padded.begin());
        core::write_tensor_f32(semantic_input_, semantic_padded);
        std::vector<int32_t> downsample(static_cast<size_t>(frame_capacity_), 0);
        for (int64_t i = 0; i < actual_frames; ++i) {
            downsample[static_cast<size_t>(i)] = static_cast<int32_t>(i * semantic_downsample_factor_);
        }
        core::write_tensor_i32(downsample_indices_, downsample);
        tensor_writer_.flush();
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice audio-tokenizer encoder graph compute failed");
        }
        OmniVoiceAudioTokens tokens = {};
        tokens.frames = actual_frames;
        tokens.codebooks = static_cast<int64_t>(code_outputs_.size());
        tokens.reference_rms = audio.reference_rms;
        tokens.token_ids.assign(static_cast<size_t>(actual_frames * tokens.codebooks), 0);
        for (size_t codebook = 0; codebook < code_outputs_.size(); ++codebook) {
            std::vector<int32_t> values(static_cast<size_t>(frame_capacity_), 0);
            ggml_backend_tensor_get(
                code_outputs_[codebook],
                values.data(),
                0,
                values.size() * sizeof(int32_t));
            for (int64_t frame = 0; frame < actual_frames; ++frame) {
                tokens.token_ids[static_cast<size_t>(frame * tokens.codebooks + static_cast<int64_t>(codebook))] =
                    values[static_cast<size_t>(frame)];
            }
        }
        return tokens;
    }

    int64_t frame_capacity() const noexcept { return frame_capacity_; }
    int64_t acoustic_sample_capacity() const noexcept { return acoustic_sample_capacity_; }
    int64_t semantic_sample_capacity() const noexcept { return semantic_sample_capacity_; }

private:
    std::shared_ptr<const OmniVoiceAssets> assets_;
    std::shared_ptr<const HiggsAudioV2TokenizerWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t acoustic_sample_capacity_ = 0;
    int64_t semantic_sample_capacity_ = 0;
    int64_t frame_capacity_ = 0;
    int64_t semantic_downsample_factor_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::DeferredTensorWriter tensor_writer_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    core::TensorValue acoustic_input_;
    core::TensorValue semantic_input_;
    core::TensorValue downsample_indices_;
    std::vector<ggml_tensor *> code_outputs_;
};

struct HiggsAudioV2DecoderGraph {
    HiggsAudioV2DecoderGraph(
        std::shared_ptr<const OmniVoiceAssets> assets,
        std::shared_ptr<const HiggsAudioV2TokenizerWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t frames,
        int64_t codebooks)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          compute_threads_(std::max(1, execution_context.config().threads)),
          graph_arena_bytes_(graph_arena_bytes) {
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice audio-tokenizer decoder graph allocator");
        }
        rebuild(frames, codebooks);
    }

    void rebuild(int64_t frames, int64_t codebooks) {
        clear_graph();
        frame_capacity_ = frames;
        codebook_capacity_ = codebooks;
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice audio-tokenizer decoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.audio_tokenizer.decode", backend_type_};
        code_inputs_.reserve(static_cast<size_t>(codebook_capacity_));
        for (int64_t i = 0; i < codebook_capacity_; ++i) {
            auto input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frame_capacity_}));
            ggml_set_input(input.tensor);
            code_inputs_.push_back(input.tensor);
        }
        decoder_frame_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, frame_capacity_}));
        ggml_set_input(decoder_frame_mask_.tensor);
        auto latent_bct = build_quantizer_decode_sequence(ctx, code_inputs_, *weights_, frame_capacity_);
        latent_bct = apply_bct_sequence_mask(ctx, latent_bct, decoder_frame_mask_);
        auto acoustic = build_linear_btc_as_conv1d(
            ctx,
            transpose_bct_to_btc(ctx, latent_bct),
            weights_->fc2,
            assets_->config.audio_tokenizer.hidden_size,
            assets_->config.audio_tokenizer.acoustic_model.hidden_size,
            true);
        auto acoustic_bct = transpose_btc_to_bct(ctx, acoustic);
        acoustic_bct = apply_bct_sequence_mask(ctx, acoustic_bct, decoder_frame_mask_);
        int64_t current_length = frame_capacity_;
        decoder_block_masks_.reserve(assets_->config.audio_tokenizer.acoustic_model.upsampling_ratios.size());
        for (const int64_t stride : assets_->config.audio_tokenizer.acoustic_model.upsampling_ratios) {
            current_length *= stride;
            auto mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, current_length}));
            ggml_set_input(mask.tensor);
            decoder_block_masks_.push_back(mask);
        }
        auto audio = build_acoustic_decoder(
            ctx,
            acoustic_bct,
            *weights_,
            assets_->config.audio_tokenizer,
            &decoder_frame_mask_,
            &decoder_block_masks_);
        const int64_t expected_samples = frame_capacity_ * assets_->config.audio_tokenizer.hop_length;
        if (audio.shape.dims[2] < expected_samples) {
            throw std::runtime_error("OmniVoice decoder produced fewer samples than expected");
        }
        if (audio.shape.dims[2] != expected_samples) {
            const int64_t trim = audio.shape.dims[2] - expected_samples;
            if ((trim % 2) != 0) {
                throw std::runtime_error("OmniVoice decoder output trim is not symmetric");
            }
            audio = modules::SliceModule({2, trim / 2, expected_samples}).build(ctx, audio);
        }
        output_ = audio.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph_, output_);
        if (!ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate OmniVoice audio-tokenizer decoder graph");
        }
        code_input_host_.resize(static_cast<size_t>(codebook_capacity_));
        for (auto & ids : code_input_host_) {
            ids.assign(static_cast<size_t>(frame_capacity_), 0);
        }
        decoder_frame_mask_host_.assign(static_cast<size_t>(frame_capacity_), 0.0F);
        decoder_block_mask_host_.resize(decoder_block_masks_.size());
        for (size_t i = 0; i < decoder_block_masks_.size(); ++i) {
            decoder_block_mask_host_[i].assign(static_cast<size_t>(decoder_block_masks_[i].shape.dims[2]), 0.0F);
        }
    }

    ~HiggsAudioV2DecoderGraph() {
        clear_graph();
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t frames, int64_t codebooks, ggml_backend_t backend, int threads) const {
        return frame_capacity_ >= frames && codebook_capacity_ >= codebooks && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    runtime::AudioBuffer run(const OmniVoiceGeneratedAudioTokens & codes) {
        if (codes.frames > frame_capacity_ || codes.codebooks > codebook_capacity_) {
            throw std::runtime_error("OmniVoice audio-tokenizer decoder request exceeds prepared capacity");
        }
        for (int64_t codebook = 0; codebook < codebook_capacity_; ++codebook) {
            auto & ids = code_input_host_[static_cast<size_t>(codebook)];
            std::fill(ids.begin(), ids.end(), 0);
            for (int64_t frame = 0; frame < codes.frames; ++frame) {
                ids[static_cast<size_t>(frame)] =
                    codebook < codes.codebooks
                        ? codes.token_ids[static_cast<size_t>(frame * codes.codebooks + codebook)]
                        : 0;
            }
            if (frame_capacity_ > 0) {
                ggml_backend_tensor_set(
                    code_inputs_[static_cast<size_t>(codebook)],
                    ids.data(),
                    0,
                    static_cast<size_t>(frame_capacity_) * sizeof(int32_t));
            }
        }
        std::fill(decoder_frame_mask_host_.begin(), decoder_frame_mask_host_.end(), 0.0F);
        std::fill_n(decoder_frame_mask_host_.begin(), static_cast<size_t>(codes.frames), 1.0F);
        if (frame_capacity_ > 0) {
            ggml_backend_tensor_set(
                decoder_frame_mask_.tensor,
                decoder_frame_mask_host_.data(),
                0,
                static_cast<size_t>(frame_capacity_) * sizeof(float));
        }
        int64_t current_length = codes.frames;
        for (size_t i = 0; i < decoder_block_masks_.size(); ++i) {
            current_length *= assets_->config.audio_tokenizer.acoustic_model.upsampling_ratios[i];
            auto & block_mask = decoder_block_mask_host_[i];
            std::fill(block_mask.begin(), block_mask.end(), 0.0F);
            std::fill_n(block_mask.begin(), static_cast<size_t>(current_length), 1.0F);
            ggml_backend_tensor_set(
                decoder_block_masks_[i].tensor,
                block_mask.data(),
                0,
                static_cast<size_t>(decoder_block_masks_[i].shape.dims[2]) * sizeof(float));
        }
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice audio-tokenizer decoder graph compute failed");
        }
        runtime::AudioBuffer audio = {};
        audio.sample_rate = assets_->config.audio_tokenizer.sample_rate;
        audio.channels = 1;
        const int64_t actual_samples = codes.frames * assets_->config.audio_tokenizer.hop_length;
        audio.samples.resize(static_cast<size_t>(actual_samples));
        if (actual_samples > 0) {
            ggml_backend_tensor_get(
                output_,
                audio.samples.data(),
                0,
                static_cast<size_t>(actual_samples) * sizeof(float));
        }
        return audio;
    }

    int64_t frame_capacity() const noexcept { return frame_capacity_; }
    int64_t codebook_capacity() const noexcept { return codebook_capacity_; }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        output_ = nullptr;
        code_inputs_.clear();
        decoder_block_masks_.clear();
        ctx_.reset();
    }

    std::shared_ptr<const OmniVoiceAssets> assets_;
    std::shared_ptr<const HiggsAudioV2TokenizerWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    int64_t frame_capacity_ = 0;
    int64_t codebook_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<ggml_tensor *> code_inputs_;
    core::TensorValue decoder_frame_mask_;
    std::vector<core::TensorValue> decoder_block_masks_;
    std::vector<std::vector<int32_t>> code_input_host_;
    std::vector<float> decoder_frame_mask_host_;
    std::vector<std::vector<float>> decoder_block_mask_host_;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

struct OmniVoiceHiggsAudioV2TokenizerRuntime::Impl {
    std::shared_ptr<const OmniVoiceAssets> assets;
    std::shared_ptr<const HiggsAudioV2TokenizerWeights> weights;
    core::ExecutionContext * execution_context = nullptr;
    size_t graph_arena_bytes = 0;
    std::unique_ptr<HiggsAudioV2EncoderGraph> encoder_graph;
    std::unique_ptr<HiggsAudioV2DecoderGraph> decoder_graph;
    OmniVoiceHiggsAudioV2TokenizerRuntimeStats last_stats = {};
};

OmniVoiceHiggsAudioV2TokenizerRuntime::OmniVoiceHiggsAudioV2TokenizerRuntime(
    std::shared_ptr<const OmniVoiceAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr) {
        throw std::runtime_error("OmniVoice audio tokenizer requires assets");
    }
    validate_weight_storage_type(weight_storage_type);
    impl_->assets = std::move(assets);
    impl_->execution_context = &execution_context;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->weights = load_weights(
        *impl_->assets,
        execution_context.backend(),
        execution_context.backend_type(),
        weight_context_bytes,
        weight_storage_type);
}

OmniVoiceHiggsAudioV2TokenizerRuntime::~OmniVoiceHiggsAudioV2TokenizerRuntime() = default;

const OmniVoiceHiggsAudioV2TokenizerRuntimeStats & OmniVoiceHiggsAudioV2TokenizerRuntime::last_stats() const noexcept {
    return impl_->last_stats;
}

OmniVoiceAudioTokens OmniVoiceHiggsAudioV2TokenizerRuntime::encode_reference_audio(
    const runtime::AudioBuffer & audio,
    const OmniVoiceReferenceAudioOptions & options) {
    const auto normalized = normalize_reference_audio(audio, impl_->assets->config.audio_tokenizer, options);
    const auto acoustic_samples =
        pad_acoustic_input_if_needed(normalized.acoustic_samples_24k, impl_->assets->config.audio_tokenizer, normalized.frames);
    const int64_t semantic_frames = semantic_feature_frames(
        impl_->assets->config.audio_tokenizer,
        static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()));
    const int64_t downsample_factor = semantic_downsample_factor(impl_->assets->config.audio_tokenizer);
    if (semantic_frames != normalized.frames * downsample_factor) {
        throw std::runtime_error("OmniVoice semantic feature extractor frame count does not match reference downsample ratio");
    }
    impl_->last_stats.encoder_graph_rebuilt = false;
    impl_->last_stats.encoder_rebuild_ms = 0.0;
    if (impl_->encoder_graph == nullptr ||
        !impl_->encoder_graph->matches(
            static_cast<int64_t>(acoustic_samples.size()),
            static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()),
            normalized.frames,
            impl_->execution_context->backend(),
            impl_->execution_context->config().threads)) {
        const auto rebuild_start = Clock::now();
        impl_->encoder_graph.reset();
        impl_->encoder_graph = std::make_unique<HiggsAudioV2EncoderGraph>(
            impl_->assets,
            impl_->weights,
            *impl_->execution_context,
            impl_->graph_arena_bytes,
            static_cast<int64_t>(acoustic_samples.size()),
            static_cast<int64_t>(normalized.semantic_samples_16k_padded.size()),
            normalized.frames);
        const auto rebuild_end = Clock::now();
        impl_->last_stats.encoder_graph_rebuilt = true;
        impl_->last_stats.encoder_rebuild_ms = engine::debug::elapsed_ms(rebuild_start, rebuild_end);
    }
    impl_->last_stats.encoder_frame_capacity = impl_->encoder_graph->frame_capacity();
    impl_->last_stats.encoder_acoustic_sample_capacity = impl_->encoder_graph->acoustic_sample_capacity();
    impl_->last_stats.encoder_semantic_sample_capacity = impl_->encoder_graph->semantic_sample_capacity();
    NormalizedReferenceAudio graph_audio = normalized;
    graph_audio.acoustic_samples_24k = acoustic_samples;
    return impl_->encoder_graph->run(graph_audio);
}

runtime::AudioBuffer OmniVoiceHiggsAudioV2TokenizerRuntime::decode_audio_tokens(
    const OmniVoiceGeneratedAudioTokens & audio_tokens) {
    if (audio_tokens.frames <= 0 || audio_tokens.codebooks <= 0) {
        throw std::runtime_error("OmniVoice generated audio tokens are empty");
    }
    if (audio_tokens.codebooks > impl_->assets->config.audio_tokenizer.num_codebooks) {
        throw std::runtime_error("OmniVoice generated audio token codebook count exceeds tokenizer configuration");
    }
    if (static_cast<int64_t>(audio_tokens.token_ids.size()) != audio_tokens.frames * audio_tokens.codebooks) {
        throw std::runtime_error("OmniVoice generated audio token shape is invalid");
    }
    impl_->last_stats.decoder_graph_rebuilt = false;
    impl_->last_stats.decoder_rebuild_ms = 0.0;
    const bool needs_rebuild =
        impl_->decoder_graph == nullptr ||
        !impl_->decoder_graph->matches(
            audio_tokens.frames,
            audio_tokens.codebooks,
            impl_->execution_context->backend(),
            impl_->execution_context->config().threads) ||
            impl_->decoder_graph->frame_capacity() != audio_tokens.frames ||
        impl_->decoder_graph->codebook_capacity() != audio_tokens.codebooks;
    if (needs_rebuild) {
        const auto rebuild_start = Clock::now();
        if (impl_->decoder_graph == nullptr) {
            impl_->decoder_graph = std::make_unique<HiggsAudioV2DecoderGraph>(
                impl_->assets,
                impl_->weights,
                *impl_->execution_context,
                impl_->graph_arena_bytes,
                audio_tokens.frames,
                audio_tokens.codebooks);
        } else {
            impl_->decoder_graph->rebuild(audio_tokens.frames, audio_tokens.codebooks);
        }
        const auto rebuild_end = Clock::now();
        impl_->last_stats.decoder_graph_rebuilt = true;
        impl_->last_stats.decoder_rebuild_ms = engine::debug::elapsed_ms(rebuild_start, rebuild_end);
    }
    impl_->last_stats.decoder_frame_capacity = impl_->decoder_graph->frame_capacity();
    impl_->last_stats.decoder_codebook_capacity = impl_->decoder_graph->codebook_capacity();
    return impl_->decoder_graph->run(audio_tokens);
}

void OmniVoiceHiggsAudioV2TokenizerRuntime::release_runtime_graphs() {
    impl_->encoder_graph.reset();
    impl_->decoder_graph.reset();
}

}  // namespace engine::models::omnivoice

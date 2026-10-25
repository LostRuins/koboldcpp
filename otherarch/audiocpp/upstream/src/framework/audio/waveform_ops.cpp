#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::audio {

void apply_preemphasis_in_place(float * mono_samples, size_t sample_count, float coefficient) {
    if (mono_samples == nullptr && sample_count > 0) {
        throw std::runtime_error("preemphasis requires a valid sample buffer");
    }
    if (coefficient == 0.0F || sample_count <= 1) {
        return;
    }
    for (size_t i = sample_count - 1; i > 0; --i) {
        mono_samples[i] -= coefficient * mono_samples[i - 1];
    }
}

void apply_preemphasis_in_place(std::vector<float> & mono_samples, float coefficient) {
    apply_preemphasis_in_place(mono_samples.data(), mono_samples.size(), coefficient);
}

std::vector<float> apply_preemphasis(std::vector<float> mono_samples, float coefficient) {
    apply_preemphasis_in_place(mono_samples, coefficient);
    return mono_samples;
}

std::vector<float> copy_or_zero_pad_samples_to_count(
    const std::vector<float> & samples,
    size_t output_sample_count) {
    std::vector<float> out(output_sample_count, 0.0F);
    const size_t copy_count = std::min(out.size(), samples.size());
    std::copy_n(samples.begin(), copy_count, out.begin());
    return out;
}

std::vector<float> zero_pad_samples_to_multiple(
    const std::vector<float> & samples,
    int64_t sample_multiple) {
    if (sample_multiple <= 0) {
        throw std::runtime_error("audio sample padding multiple must be positive");
    }
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    const int64_t multiples = (sample_count + sample_multiple - 1) / sample_multiple;
    const int64_t padded_samples = multiples * sample_multiple;
    std::vector<float> out = samples;
    out.resize(static_cast<size_t>(padded_samples), 0.0F);
    return out;
}

void truncate_samples_to_count(std::vector<float> & samples, size_t max_sample_count) {
    if (samples.size() > max_sample_count) {
        samples.resize(max_sample_count);
    }
}

void normalize_peak_to_unit_range_and_clamp_in_place(std::vector<float> & samples) {
    float peak = 0.0F;
    for (const float sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }
    if (peak > 1.0F) {
        for (float & sample : samples) {
            sample /= peak;
        }
    }
    for (float & sample : samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
}

std::vector<float> trim_mono_librosa_effects(
    const std::vector<float> & mono_samples,
    const LibrosaEffectsTrimOptions & options) {
    if (options.frame_length <= 0 || options.hop_length <= 0) {
        throw std::runtime_error("librosa effects trim frame and hop lengths must be positive");
    }
    if (!std::isfinite(options.top_db) || options.top_db < 0.0F) {
        throw std::runtime_error("librosa effects trim top_db must be finite and non-negative");
    }
    if (!std::isfinite(options.amin) || options.amin <= 0.0F) {
        throw std::runtime_error("librosa effects trim amin must be finite and positive");
    }
    if (mono_samples.empty()) {
        return mono_samples;
    }

    const int64_t sample_count = static_cast<int64_t>(mono_samples.size());
    const int64_t frames = 1 + sample_count / options.hop_length;
    std::vector<float> rms(static_cast<size_t>(frames), 0.0F);
    float ref = options.amin;
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t start = frame * options.hop_length - options.frame_length / 2;
        double power = 0.0;
        for (int64_t i = 0; i < options.frame_length; ++i) {
            const int64_t sample_index = start + i;
            const float sample =
                sample_index >= 0 && sample_index < sample_count
                    ? mono_samples[static_cast<size_t>(sample_index)]
                    : 0.0F;
            power += static_cast<double>(sample) * static_cast<double>(sample);
        }
        const float value = std::sqrt(static_cast<float>(power / static_cast<double>(options.frame_length)));
        rms[static_cast<size_t>(frame)] = value;
        ref = std::max(ref, value);
    }

    const float threshold = ref * std::pow(10.0F, -options.top_db / 20.0F);
    int64_t first = -1;
    int64_t last = -1;
    for (int64_t frame = 0; frame < frames; ++frame) {
        if (std::max(rms[static_cast<size_t>(frame)], options.amin) > threshold) {
            if (first < 0) {
                first = frame;
            }
            last = frame;
        }
    }
    if (first < 0) {
        return {};
    }

    const int64_t start = std::min<int64_t>(sample_count, first * options.hop_length);
    const int64_t end = std::min<int64_t>(sample_count, (last + 1) * options.hop_length);
    return std::vector<float>(
        mono_samples.begin() + static_cast<std::ptrdiff_t>(start),
        mono_samples.begin() + static_cast<std::ptrdiff_t>(end));
}

std::vector<float> reflect_pad_samples(
    const std::vector<float> & samples,
    int64_t left_pad_samples,
    int64_t right_pad_samples) {
    if (samples.empty()) {
        throw std::runtime_error("reflect padding requires non-empty samples");
    }
    if (left_pad_samples < 0 || right_pad_samples < 0) {
        throw std::runtime_error("reflect padding size is invalid for sample count");
    }
    if (samples.size() == 1 && (left_pad_samples > 0 || right_pad_samples > 0)) {
        throw std::runtime_error("reflect padding requires at least two samples when padding is non-zero");
    }
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    std::vector<float> out(static_cast<size_t>(sample_count + left_pad_samples + right_pad_samples), 0.0F);
    for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        int64_t index = i - left_pad_samples;
        while (index < 0 || index >= sample_count) {
            if (index < 0) {
                index = -index;
            }
            if (index >= sample_count) {
                index = 2 * sample_count - index - 2;
            }
        }
        out[static_cast<size_t>(i)] = samples[static_cast<size_t>(index)];
    }
    return out;
}

float root_mean_square_or_throw(const std::vector<float> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("RMS input samples must not be empty");
    }
    double sum = 0.0;
    for (const float sample : samples) {
        sum += static_cast<double>(sample) * static_cast<double>(sample);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(samples.size())));
}

std::vector<int16_t> float_to_pcm16_clipped(
    const std::vector<float> & samples,
    Pcm16QuantizeMode quantize_mode) {
    std::vector<int16_t> pcm(samples.size(), 0);
    for (size_t i = 0; i < samples.size(); ++i) {
        const float clipped = std::clamp(samples[i] * 32768.0F, -32768.0F, 32767.0F);
        if (quantize_mode == Pcm16QuantizeMode::RoundToNearest) {
            pcm[i] = static_cast<int16_t>(std::lrint(clipped));
        } else {
            pcm[i] = static_cast<int16_t>(clipped);
        }
    }
    return pcm;
}

std::vector<float> pcm16_to_float_unit_range(const std::vector<int16_t> & pcm_samples) {
    std::vector<float> samples(pcm_samples.size(), 0.0F);
    for (size_t i = 0; i < pcm_samples.size(); ++i) {
        samples[i] = static_cast<float>(pcm_samples[i]) / 32768.0F;
    }
    return samples;
}

}  // namespace engine::audio

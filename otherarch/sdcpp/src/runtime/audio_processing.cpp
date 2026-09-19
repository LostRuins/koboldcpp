#include "audio_processing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace sd::audio {

    // Match torchaudio's Hann-windowed sinc resampler.
    std::vector<float> resample_audio(const float* samples,
                                      uint64_t sample_count,
                                      uint32_t orig_sample_rate,
                                      uint32_t target_sample_rate) {
        if (samples == nullptr || sample_count == 0 || orig_sample_rate == 0 || target_sample_rate == 0) {
            return {};
        }
        if (orig_sample_rate == target_sample_rate) {
            return std::vector<float>(samples, samples + sample_count);
        }

        constexpr int kLowpassFilterWidth = 6;
        constexpr double kRolloff         = 0.99;
        constexpr double kPi              = 3.14159265358979323846;

        const uint64_t gcd        = std::gcd(static_cast<uint64_t>(orig_sample_rate),
                                             static_cast<uint64_t>(target_sample_rate));
        const int64_t orig_freq   = static_cast<int64_t>(orig_sample_rate / gcd);
        const int64_t new_freq    = static_cast<int64_t>(target_sample_rate / gcd);
        const double base_freq    = static_cast<double>(std::min(orig_freq, new_freq)) * kRolloff;
        const int64_t width       = static_cast<int64_t>(std::ceil(kLowpassFilterWidth * orig_freq / base_freq));
        const int64_t kernel_size = 2 * width + orig_freq;

        std::vector<double> kernel(static_cast<size_t>(new_freq) * kernel_size);
        for (int64_t j = 0; j < new_freq; ++j) {
            for (int64_t i = 0; i < kernel_size; ++i) {
                double t = -static_cast<double>(j) / new_freq + static_cast<double>(i - width) / orig_freq;
                t *= base_freq;
                t                           = std::clamp(t, -static_cast<double>(kLowpassFilterWidth), static_cast<double>(kLowpassFilterWidth));
                const double cos_arg        = std::cos(t * kPi / kLowpassFilterWidth / 2);
                const double window         = cos_arg * cos_arg;
                double s                    = t * kPi;
                const double sinc           = (s == 0.0) ? 1.0 : std::sin(s) / s;
                kernel[j * kernel_size + i] = sinc * window * (base_freq / orig_freq);
            }
        }

        const uint64_t num_phases    = static_cast<uint64_t>(sample_count / orig_freq) + 1;
        const uint64_t target_length = (static_cast<uint64_t>(new_freq) * sample_count +
                                        static_cast<uint64_t>(orig_freq) - 1) /
                                       static_cast<uint64_t>(orig_freq);
        std::vector<float> out(target_length);
        for (uint64_t phase = 0; phase < num_phases; ++phase) {
            const int64_t src_base = static_cast<int64_t>(phase * orig_freq) - width;
            for (int64_t j = 0; j < new_freq; ++j) {
                const uint64_t out_index = phase * new_freq + j;
                if (out_index >= target_length) {
                    break;
                }
                const double* k = &kernel[j * kernel_size];
                double acc      = 0.0;
                for (int64_t i = 0; i < kernel_size; ++i) {
                    const int64_t src = src_base + i;
                    if (src >= 0 && src < static_cast<int64_t>(sample_count)) {
                        acc += samples[src] * k[i];
                    }
                }
                out[out_index] = static_cast<float>(acc);
            }
        }
        return out;
    }

    std::vector<float> downmix_to_mono(const float* interleaved_samples,
                                       uint64_t sample_count,
                                       uint32_t channels) {
        std::vector<float> mono;
        if (interleaved_samples == nullptr || sample_count == 0 || channels == 0) {
            return mono;
        }
        mono.resize(static_cast<size_t>(sample_count));
        if (channels == 1) {
            std::memcpy(mono.data(), interleaved_samples, static_cast<size_t>(sample_count) * sizeof(float));
            return mono;
        }
        const float scale = 1.0f / static_cast<float>(channels);
        for (uint64_t i = 0; i < sample_count; ++i) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; ++c) {
                sum += interleaved_samples[i * channels + c];
            }
            mono[static_cast<size_t>(i)] = sum * scale;
        }
        return mono;
    }

}  // namespace sd::audio

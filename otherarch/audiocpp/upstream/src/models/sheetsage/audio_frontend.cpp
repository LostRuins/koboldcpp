#include "engine/models/sheetsage/audio_frontend.h"

#include "engine/framework/audio/conversion.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace engine::models::sheetsage {

std::vector<float> SheetSage2AudioFrontend::prepare(
    const std::vector<float> & interleaved,
    int source_rate,
    int channels,
    int target_rate,
    int threads) {
    if (source_rate <= 0 || target_rate <= 0 || channels < 1 || channels > 8 || threads < 1 ||
        interleaved.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("SheetSage2 requires positive rates and 1-8 interleaved channels");
    }
    // Default layouts: mono, stereo, 2.1, 4.0, 5.0, 5.1, 6.1, 7.1.
    // Float mono output preserves front/surround mix gains without peak normalization.
    constexpr float side = 0.7071067811865475244F;
    constexpr std::array<std::array<float, 8>, 8> gains{{
        {{1}},
        {{side, side}},
        {{side, side, 0}},
        {{side, side, 1, 0.5F}},
        {{side, side, 1, 0.5F, 0.5F}},
        {{side, side, 1, 0, 0.5F, 0.5F}},
        {{side, side, 1, 0, 0.5F, 0.5F, 0.5F}},
        {{side, side, 1, 0, 0.5F, 0.5F, 0.5F, 0.5F}},
    }};
    const int64_t frames = static_cast<int64_t>(interleaved.size() / static_cast<size_t>(channels));
    std::vector<float> mono;
    // Match the file frontend's operation order, including FP32 rounding.
    if (channels > 1 && source_rate != target_rate) {
        for (int channel = 0; channel < channels; ++channel) {
            const auto resampled = prepare(
                engine::audio::extract_interleaved_channel(interleaved, channels, channel),
                source_rate, 1, target_rate, threads);
            if (channel == 0) mono.resize(resampled.size(), 0.0F);
            const float gain = gains[static_cast<size_t>(channels - 1)][static_cast<size_t>(channel)];
            for (size_t frame = 0; frame < resampled.size(); ++frame) {
                mono[frame] += resampled[frame] * gain;
            }
        }
        return mono;
    }
    if (channels == 1) {
        mono = engine::audio::mixdown_interleaved_to_mono_average(interleaved, channels);
    } else {
        mono.resize(static_cast<size_t>(frames));
        const auto & row = gains[static_cast<size_t>(channels - 1)];
        for (int64_t frame = 0; frame < frames; ++frame) {
            float value = 0.0F;
            for (int channel = 0; channel < channels; ++channel) {
                value += interleaved[static_cast<size_t>(frame * channels + channel)] * row[static_cast<size_t>(channel)];
            }
            mono[static_cast<size_t>(frame)] = value;
        }
    }
    if (source_rate == target_rate || mono.empty()) {
        return mono;
    }

    // Python's file frontend uses a 32-lobe, beta=9 Kaiser sinc with 0.97 cutoff.
    // Keep exact rational phases where possible, otherwise interpolate a 1024-phase bank.
    constexpr double pi = 3.14159265358979323846264338327950288;
    const double cutoff = std::min(0.97 * target_rate / source_rate, 1.0);
    const int taps = (static_cast<int>(std::ceil(32.0 / cutoff)) + 1) & ~1;
    const int stride = (taps + 7) & ~7;
    const int center = (taps - 1) / 2;
    const int phases = std::min(1024, target_rate / std::gcd(source_rate, target_rate));
    const auto key = std::make_pair(source_rate, target_rate);
    auto found = filters_.find(key);
    if (found == filters_.end()) {
        std::vector<float> bank(static_cast<size_t>((phases + 1) * stride));
        std::vector<double> coefficients(static_cast<size_t>(phases * taps));
        double normalization = 0.0;
        for (int phase = 0; phase < phases; ++phase) {
            for (int tap = 0; tap < taps; ++tap) {
                const double distance = tap - center - static_cast<double>(phase) / phases;
                const double angle = pi * distance * cutoff;
                const double radius = 2.0 * distance / taps;
                const double x = 9.0 * std::sqrt(std::max(0.0, 1.0 - radius * radius));
                const double quarter_square = x * x / 4.0;
                // I0(x) series for 0 <= x <= 9; the tail after k=32 is below 2e-31.
                // Avoid special functions unavailable in Apple's libc++.
                double window = 1.0;
                double term = 1.0;
                for (int k = 1; k <= 32; ++k) {
                    term *= quarter_square / (k * k);
                    window += term;
                }
                const double value = (angle == 0.0 ? 1.0 : std::sin(angle) / angle) * window;
                if (phase == 0) {
                    normalization += value;
                }
                coefficients[static_cast<size_t>(phase * taps + tap)] = value;
            }
        }
        for (int phase = 0; phase < phases; ++phase) {
            for (int tap = 0; tap < taps; ++tap) {
                bank[static_cast<size_t>(phase * stride + tap)] = static_cast<float>(
                    coefficients[static_cast<size_t>(phase * taps + tap)] / normalization);
            }
        }
        // Phase 1 is phase 0 shifted by one input sample.
        bank[static_cast<size_t>(phases * stride)] = bank[static_cast<size_t>(stride - 1)];
        std::copy_n(bank.begin(), stride - 1, bank.begin() + phases * stride + 1);
        found = filters_.emplace(key, std::move(bank)).first;
    }
    const auto & bank = found->second;
    // Flush only the reflected tail available after the unpadded convolution.
    const int64_t unpadded_frames = std::max<int64_t>(0,
        ((frames - taps / 2) * target_rate + source_rate - 1) / source_rate);
    const int64_t remaining = frames - unpadded_frames * source_rate / target_rate + center;
    const int64_t reflection = (std::min<int64_t>(remaining, taps) + 1) / 2;
    const int64_t output_frames = std::max<int64_t>(0,
        ((frames + reflection - taps / 2) * target_rate + source_rate - 1) / source_rate);
    std::vector<float> output(static_cast<size_t>(output_frames));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if (output_frames >= 4096)
#endif
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        const int64_t numerator = frame * source_rate;
        const int64_t sample = numerator / target_rate;
        const int64_t phase_numerator = (numerator % target_rate) * phases;
        const int phase = static_cast<int>(phase_numerator / target_rate);
        const double fraction = static_cast<double>(phase_numerator % target_rate) / target_rate;
        const float * weights = bank.data() + phase * stride;
        // Fixed FP32 lanes keep the reduction independent of the inference backend.
        std::array<float, 8> sums{};
        std::array<float, 8> next_sums{};
        for (int tap = 0; tap < stride; ++tap) {
            int64_t index = sample - center + tap;
            // Reflect about the first sample; repeat the last sample at the right edge.
            while (index < 0 || index >= frames) {
                index = index < 0 ? -index : 2 * frames - 1 - index;
            }
            const float input = mono[static_cast<size_t>(index)];
            const size_t lane = static_cast<size_t>(tap % 8);
            sums[lane] = std::fma(input, weights[tap], sums[lane]);
            if (fraction != 0.0) {
                next_sums[lane] = std::fma(input, weights[stride + tap], next_sums[lane]);
            }
        }
        const float value = ((sums[0] + sums[4]) + (sums[2] + sums[6])) +
                            ((sums[1] + sums[5]) + (sums[3] + sums[7]));
        const float next_value = ((next_sums[0] + next_sums[4]) + (next_sums[2] + next_sums[6])) +
                                 ((next_sums[1] + next_sums[5]) + (next_sums[3] + next_sums[7]));
        output[static_cast<size_t>(frame)] = static_cast<float>(
            fraction == 0.0 ? value : value + (next_value - value) * fraction);
    }
    return output;
}

}  // namespace engine::models::sheetsage

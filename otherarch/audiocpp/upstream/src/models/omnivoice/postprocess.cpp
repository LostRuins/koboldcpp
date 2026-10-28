#include "engine/models/omnivoice/postprocess.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::omnivoice {
namespace {

int64_t ms_to_samples(int ms, int sample_rate) {
    return static_cast<int64_t>(ms) * static_cast<int64_t>(sample_rate) / 1000;
}

int64_t len_ms(const std::vector<int16_t> & pcm, int sample_rate) {
    if (sample_rate <= 0) {
        throw std::runtime_error("OmniVoice silence utilities require positive sample_rate");
    }
    return static_cast<int64_t>(pcm.size()) * 1000 / sample_rate;
}

float silence_threshold_amplitude(float silence_threshold_db) {
    return std::pow(10.0F, silence_threshold_db / 20.0F) * 32768.0F;
}

float rms_pcm16(const std::vector<int16_t> & pcm, int64_t start_ms, int64_t end_ms, int sample_rate) {
    const int64_t start_sample = std::clamp<int64_t>(
        ms_to_samples(static_cast<int>(start_ms), sample_rate),
        0,
        static_cast<int64_t>(pcm.size()));
    const int64_t end_sample = std::clamp<int64_t>(
        ms_to_samples(static_cast<int>(end_ms), sample_rate),
        start_sample,
        static_cast<int64_t>(pcm.size()));
    if (end_sample <= start_sample) {
        return 0.0F;
    }
    double sum = 0.0;
    for (int64_t i = start_sample; i < end_sample; ++i) {
        const double value = static_cast<double>(pcm[static_cast<size_t>(i)]);
        sum += value * value;
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(end_sample - start_sample)));
}

std::vector<std::pair<int64_t, int64_t>> detect_silence_ranges_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int min_silence_len_ms,
    float silence_threshold_db,
    int seek_step_ms) {
    const int64_t seg_len_ms = len_ms(pcm, sample_rate);
    if (seg_len_ms < min_silence_len_ms) {
        return {};
    }
    const float threshold = silence_threshold_amplitude(silence_threshold_db);
    std::vector<int64_t> silence_starts;
    const int64_t last_slice_start = seg_len_ms - min_silence_len_ms;
    for (int64_t start_ms = 0; start_ms <= last_slice_start; start_ms += seek_step_ms) {
        if (rms_pcm16(pcm, start_ms, start_ms + min_silence_len_ms, sample_rate) <= threshold) {
            silence_starts.push_back(start_ms);
        }
    }
    if (last_slice_start % seek_step_ms != 0) {
        if (rms_pcm16(pcm, last_slice_start, last_slice_start + min_silence_len_ms, sample_rate) <= threshold) {
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
        const bool continuous = start_ms == prev + seek_step_ms;
        const bool has_gap = start_ms > prev + min_silence_len_ms;
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
    int min_silence_len_ms,
    float silence_threshold_db,
    int seek_step_ms) {
    const auto silent_ranges =
        detect_silence_ranges_ms(pcm, sample_rate, min_silence_len_ms, silence_threshold_db, seek_step_ms);
    const int64_t seg_len_ms = len_ms(pcm, sample_rate);
    if (silent_ranges.empty()) {
        return {{0, seg_len_ms}};
    }
    if (silent_ranges.front().first == 0 && silent_ranges.front().second == seg_len_ms) {
        return {};
    }
    std::vector<std::pair<int64_t, int64_t>> nonsilent_ranges;
    int64_t prev_end = 0;
    for (const auto & [start_ms, end_ms] : silent_ranges) {
        nonsilent_ranges.emplace_back(prev_end, start_ms);
        prev_end = end_ms;
    }
    if (silent_ranges.back().second != seg_len_ms) {
        nonsilent_ranges.emplace_back(prev_end, seg_len_ms);
    }
    if (!nonsilent_ranges.empty() && nonsilent_ranges.front().first == 0 && nonsilent_ranges.front().second == 0) {
        nonsilent_ranges.erase(nonsilent_ranges.begin());
    }
    return nonsilent_ranges;
}

std::vector<int16_t> slice_pcm16_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int64_t start_ms,
    int64_t end_ms) {
    const int64_t start_sample = std::clamp<int64_t>(
        ms_to_samples(static_cast<int>(start_ms), sample_rate),
        0,
        static_cast<int64_t>(pcm.size()));
    const int64_t end_sample = std::clamp<int64_t>(
        ms_to_samples(static_cast<int>(end_ms), sample_rate),
        start_sample,
        static_cast<int64_t>(pcm.size()));
    return std::vector<int16_t>(
        pcm.begin() + static_cast<std::ptrdiff_t>(start_sample),
        pcm.begin() + static_cast<std::ptrdiff_t>(end_sample));
}

std::vector<int16_t> split_on_silence_pcm16(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int min_silence_len_ms,
    float silence_threshold_db,
    int keep_silence_ms,
    int seek_step_ms) {
    const auto nonsilent =
        detect_nonsilent_ranges_ms(pcm, sample_rate, min_silence_len_ms, silence_threshold_db, seek_step_ms);
    if (nonsilent.empty()) {
        return {};
    }
    std::vector<std::pair<int64_t, int64_t>> ranges;
    ranges.reserve(nonsilent.size());
    for (const auto & [start_ms, end_ms] : nonsilent) {
        ranges.emplace_back(start_ms - keep_silence_ms, end_ms + keep_silence_ms);
    }
    for (size_t i = 0; i + 1 < ranges.size(); ++i) {
        const int64_t last_end = ranges[i].second;
        const int64_t next_start = ranges[i + 1].first;
        if (next_start < last_end) {
            ranges[i].second = (last_end + next_start) / 2;
            ranges[i + 1].first = ranges[i].second;
        }
    }

    std::vector<int16_t> out;
    for (const auto & [start_ms_raw, end_ms_raw] : ranges) {
        const int64_t start_ms = std::max<int64_t>(0, start_ms_raw);
        const int64_t end_ms = std::min<int64_t>(end_ms_raw, len_ms(pcm, sample_rate));
        auto part = slice_pcm16_ms(pcm, sample_rate, start_ms, end_ms);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

int64_t detect_leading_silence_ms(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    float silence_threshold_db,
    int chunk_size_ms = 10) {
    const float threshold = silence_threshold_amplitude(silence_threshold_db);
    int64_t trim_ms = 0;
    const int64_t total_ms = len_ms(pcm, sample_rate);
    while (trim_ms < total_ms &&
           rms_pcm16(pcm, trim_ms, std::min<int64_t>(trim_ms + chunk_size_ms, total_ms), sample_rate) < threshold) {
        trim_ms += chunk_size_ms;
    }
    return std::min<int64_t>(trim_ms, total_ms);
}

std::vector<int16_t> trim_edges_pcm16(
    const std::vector<int16_t> & pcm,
    int sample_rate,
    int keep_lead_ms,
    int keep_trail_ms,
    float silence_threshold_db) {
    if (pcm.empty()) {
        return {};
    }
    const int64_t total_ms = len_ms(pcm, sample_rate);
    int64_t start_ms = detect_leading_silence_ms(pcm, sample_rate, silence_threshold_db, 10);
    start_ms = std::max<int64_t>(0, start_ms - keep_lead_ms);

    std::vector<int16_t> reversed(pcm.rbegin(), pcm.rend());
    int64_t trail_trim_ms = detect_leading_silence_ms(reversed, sample_rate, silence_threshold_db, 10);
    trail_trim_ms = std::max<int64_t>(0, trail_trim_ms - keep_trail_ms);
    const int64_t end_ms = std::max<int64_t>(start_ms, total_ms - trail_trim_ms);
    return slice_pcm16_ms(pcm, sample_rate, start_ms, end_ms);
}

std::vector<float> fade_and_pad_audio(
    const std::vector<float> & mono,
    int sample_rate,
    float pad_duration_seconds = 0.1F,
    float fade_duration_seconds = 0.1F) {
    if (mono.empty()) {
        return mono;
    }

    const size_t fade_samples = static_cast<size_t>(fade_duration_seconds * static_cast<float>(sample_rate));
    const size_t pad_samples = static_cast<size_t>(pad_duration_seconds * static_cast<float>(sample_rate));
    std::vector<float> processed = mono;

    if (fade_samples > 0) {
        const size_t k = std::min(fade_samples, processed.size() / 2);
        if (k > 0) {
            for (size_t i = 0; i < k; ++i) {
                const float fade_in = k == 1 ? 0.0F : static_cast<float>(i) / static_cast<float>(k - 1);
                const float fade_out = k == 1 ? 1.0F : 1.0F - static_cast<float>(i) / static_cast<float>(k - 1);
                processed[i] *= fade_in;
                processed[processed.size() - k + i] *= fade_out;
            }
        }
    }

    if (pad_samples > 0) {
        processed.insert(processed.begin(), pad_samples, 0.0F);
        processed.insert(processed.end(), pad_samples, 0.0F);
    }
    return processed;
}

std::vector<float> mono_from_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        throw std::runtime_error("OmniVoice postprocess requires valid audio metadata");
    }
    if (audio.samples.empty()) {
        return {};
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("OmniVoice postprocess received invalid interleaved audio");
    }
    return engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

runtime::AudioBuffer make_audio_buffer(std::vector<float> mono, int sample_rate) {
    runtime::AudioBuffer out;
    out.sample_rate = sample_rate;
    out.channels = 1;
    out.samples = std::move(mono);
    return out;
}

}  // namespace

OmniVoiceResult OmniVoicePostprocessor::finalize(
    const runtime::AudioBuffer & audio,
    const OmniVoiceRequest & request) const {
    auto mono = mono_from_audio(audio);
    if (request.generation.postprocess_output) {
        auto pcm = engine::audio::float_to_pcm16_clipped(
            mono,
            engine::audio::Pcm16QuantizeMode::RoundToNearest);
        pcm = split_on_silence_pcm16(pcm, audio.sample_rate, 500, -50.0F, 500, 10);
        pcm = trim_edges_pcm16(pcm, audio.sample_rate, 100, 100, -50.0F);
        mono = engine::audio::pcm16_to_float_unit_range(pcm);
    }

    if (request.reference_audio_tokens.has_value() && request.reference_rms < 0.1F && request.reference_rms > 0.0F) {
        const float scale = request.reference_rms / 0.1F;
        for (float & sample : mono) {
            sample *= scale;
        }
    } else if (!request.reference_audio_tokens.has_value()) {
        float peak = 0.0F;
        for (const float sample : mono) {
            peak = std::max(peak, std::abs(sample));
        }
        if (peak > 1.0e-6F) {
            const float scale = 0.5F / peak;
            for (float & sample : mono) {
                sample *= scale;
            }
        }
    }

    mono = fade_and_pad_audio(mono, audio.sample_rate);
    OmniVoiceResult result;
    result.audio = make_audio_buffer(std::move(mono), audio.sample_rate);
    return result;
}

}  // namespace engine::models::omnivoice

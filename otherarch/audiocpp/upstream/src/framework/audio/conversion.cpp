#include "engine/framework/audio/conversion.h"

#include "engine/framework/audio/resampling.h"

#include <algorithm>
#include <stdexcept>

namespace engine::audio {

std::vector<float> mixdown_interleaved_to_mono_average(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    MonoMixAccumulation accumulation) {
    if (channel_count <= 0) {
        throw std::runtime_error("audio channel count must be positive");
    }
    if (interleaved_samples.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("interleaved audio sample count must be divisible by channel count");
    }
    if (channel_count == 1) {
        return interleaved_samples;
    }
    const size_t frames = interleaved_samples.size() / static_cast<size_t>(channel_count);
    std::vector<float> mono(frames, 0.0F);
    for (size_t frame = 0; frame < frames; ++frame) {
        if (accumulation == MonoMixAccumulation::Float64) {
            double sum = 0.0;
            for (int channel = 0; channel < channel_count; ++channel) {
                sum += interleaved_samples[frame * static_cast<size_t>(channel_count) + static_cast<size_t>(channel)];
            }
            mono[frame] = static_cast<float>(sum / static_cast<double>(channel_count));
        } else {
            float sum = 0.0F;
            for (int channel = 0; channel < channel_count; ++channel) {
                sum += interleaved_samples[frame * static_cast<size_t>(channel_count) + static_cast<size_t>(channel)];
            }
            mono[frame] = sum / static_cast<float>(channel_count);
        }
    }
    return mono;
}

std::vector<float> duplicate_mono_to_interleaved_channels(
    const std::vector<float> & mono_samples,
    int target_channel_count) {
    if (target_channel_count <= 0) {
        throw std::runtime_error("target channel count must be positive");
    }
    std::vector<float> interleaved(mono_samples.size() * static_cast<size_t>(target_channel_count), 0.0F);
    for (size_t frame = 0; frame < mono_samples.size(); ++frame) {
        for (int channel = 0; channel < target_channel_count; ++channel) {
            interleaved[frame * static_cast<size_t>(target_channel_count) + static_cast<size_t>(channel)] =
                mono_samples[frame];
        }
    }
    return interleaved;
}

std::vector<float> deinterleave_to_planar_channels(
    const std::vector<float> & interleaved_samples,
    int channel_count) {
    if (channel_count <= 0) {
        throw std::runtime_error("audio channel count must be positive");
    }
    if (interleaved_samples.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("interleaved audio sample count must be divisible by channel count");
    }
    const int64_t frames = static_cast<int64_t>(interleaved_samples.size() / static_cast<size_t>(channel_count));
    std::vector<float> planar(static_cast<size_t>(frames * channel_count), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * channel_count >= 1 << 14)
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channel_count; ++channel) {
            planar[static_cast<size_t>(channel * frames + frame)] =
                interleaved_samples[static_cast<size_t>(frame * channel_count + channel)];
        }
    }
    return planar;
}

std::vector<float> interleave_planar_channels(
    const std::vector<float> & planar_samples,
    int channel_count,
    int64_t frame_count) {
    if (channel_count <= 0 || frame_count < 0) {
        throw std::runtime_error("planar audio interleave shape is invalid");
    }
    if (planar_samples.size() != static_cast<size_t>(channel_count * frame_count)) {
        throw std::runtime_error("planar audio sample count does not match channel/frame shape");
    }
    std::vector<float> interleaved(static_cast<size_t>(frame_count * channel_count), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frame_count * channel_count >= 1 << 14)
#endif
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        for (int channel = 0; channel < channel_count; ++channel) {
            interleaved[static_cast<size_t>(frame * channel_count + channel)] =
                planar_samples[static_cast<size_t>(channel * frame_count + frame)];
        }
    }
    return interleaved;
}

std::vector<float> extract_interleaved_channel(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    int channel_index) {
    if (channel_count <= 0 || channel_index < 0 || channel_index >= channel_count) {
        throw std::runtime_error("interleaved audio channel selection is invalid");
    }
    if (interleaved_samples.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("interleaved audio sample count must be divisible by channel count");
    }
    const size_t frames = interleaved_samples.size() / static_cast<size_t>(channel_count);
    std::vector<float> channel(frames, 0.0F);
    for (size_t frame = 0; frame < frames; ++frame) {
        channel[frame] = interleaved_samples[frame * static_cast<size_t>(channel_count) + static_cast<size_t>(channel_index)];
    }
    return channel;
}

std::vector<float> convert_wav_to_mono_linear_resampled(
    const WavData & wav,
    int target_sample_rate_hz) {
    if (wav.sample_rate <= 0 || target_sample_rate_hz <= 0) {
        throw std::runtime_error("audio sample rates must be positive");
    }
    auto mono = mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
    if (wav.sample_rate != target_sample_rate_hz) {
        mono = resample_mono_linear(mono, wav.sample_rate, target_sample_rate_hz);
    }
    return mono;
}

std::vector<float> convert_interleaved_audio_to_mono_linear_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz) {
    return convert_wav_to_mono_linear_resampled(
        WavData{sample_rate_hz, channel_count, interleaved_samples},
        target_sample_rate_hz);
}

std::vector<float> convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options,
    MonoMixAccumulation accumulation) {
    if (sample_rate_hz <= 0 || target_sample_rate_hz <= 0) {
        throw std::runtime_error("audio sample rates must be positive");
    }
    auto mono = mixdown_interleaved_to_mono_average(interleaved_samples, channel_count, accumulation);
    if (sample_rate_hz != target_sample_rate_hz) {
        mono = resample_mono_torchaudio_sinc_hann(mono, sample_rate_hz, target_sample_rate_hz, options);
    }
    return mono;
}

std::vector<float> read_wav_f32_as_mono_linear_resampled(
    const std::filesystem::path & path,
    int target_sample_rate_hz) {
    return convert_wav_to_mono_linear_resampled(read_wav_f32(path), target_sample_rate_hz);
}

}  // namespace engine::audio

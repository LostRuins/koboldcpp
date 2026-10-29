#pragma once

#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/wav_reader.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::audio {

enum class MonoMixAccumulation {
    Float32,
    Float64,
};

std::vector<float> mixdown_interleaved_to_mono_average(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    MonoMixAccumulation accumulation = MonoMixAccumulation::Float32);

std::vector<float> duplicate_mono_to_interleaved_channels(
    const std::vector<float> & mono_samples,
    int target_channel_count);

std::vector<float> deinterleave_to_planar_channels(
    const std::vector<float> & interleaved_samples,
    int channel_count);

std::vector<float> interleave_planar_channels(
    const std::vector<float> & planar_samples,
    int channel_count,
    int64_t frame_count);

std::vector<float> extract_interleaved_channel(
    const std::vector<float> & interleaved_samples,
    int channel_count,
    int channel_index);

std::vector<float> convert_wav_to_mono_linear_resampled(
    const WavData & wav,
    int target_sample_rate_hz);

std::vector<float> convert_interleaved_audio_to_mono_linear_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz);

std::vector<float> convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
    const std::vector<float> & interleaved_samples,
    int sample_rate_hz,
    int channel_count,
    int target_sample_rate_hz,
    const TorchaudioSincHannResampleOptions & options = {},
    MonoMixAccumulation accumulation = MonoMixAccumulation::Float32);

std::vector<float> read_wav_f32_as_mono_linear_resampled(
    const std::filesystem::path & path,
    int target_sample_rate_hz);

}  // namespace engine::audio

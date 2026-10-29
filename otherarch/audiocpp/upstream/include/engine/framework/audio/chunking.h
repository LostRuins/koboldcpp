#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::audio {

enum class AudioChunkPadMode {
    Zero,
    Reflect,
};

enum class AudioChunkTailAlignment {
    Start,
    Center,
};

enum class AudioChunkCounterMode {
    SharedAcrossLanes,
    PerLane,
};

enum class AudioChunkMode {
    Auto,
    Fixed,
    QuietEnergy,
    Vad,
    None,
    Silence,  // Recording-relative RMS threshold and silence-gap midpoints.
};

struct AudioChunkSpec {
    int64_t chunk_samples = 0;
    int64_t hop_samples = 0;
    AudioChunkPadMode pad_mode = AudioChunkPadMode::Zero;
    AudioChunkTailAlignment tail_alignment = AudioChunkTailAlignment::Start;
    int64_t reflect_min_valid_samples = 0;
};

struct AudioChunkSpan {
    int64_t index = 0;
    int64_t output_start_sample = 0;
    int64_t valid_samples = 0;
    int64_t copy_start_sample = 0;
    int64_t valid_start_in_chunk = 0;
};

struct VadAudioChunkOptions {
    int64_t max_chunk_samples = 0;
    int64_t merge_gap_samples = 0;
    int64_t padding_samples = 0;
};

struct QuietEnergyAudioChunkOptions {
    int64_t chunk_samples = 0;
    int64_t boundary_context_samples = 0;
    int64_t min_energy_window_samples = 0;
};

struct SilenceAudioChunkOptions {
    int64_t max_chunk_samples = 0;
    // Inputs at or below this length stay whole, even if above max_chunk_samples.
    int64_t trigger_samples = 0;
};

// Librosa-style silence splitting: centered 2048-sample RMS, hop 512,
// 30 dB below peak RMS. Preserve all audio and cut at silence-gap midpoints,
// falling back to max_chunk_samples when no gap is available.
std::vector<runtime::TimeSpan> plan_silence_audio_chunks(
    const std::vector<float> & mono_samples,
    const SilenceAudioChunkOptions & options);

std::vector<AudioChunkSpan> plan_audio_chunks(int64_t input_samples, const AudioChunkSpec & spec);

AudioChunkMode parse_audio_chunk_mode(
    const std::unordered_map<std::string, std::string> & options);

std::optional<float> parse_audio_chunk_seconds_override(
    const std::unordered_map<std::string, std::string> & options);

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const std::vector<runtime::SpeechSegment> & segments,
    int64_t audio_samples,
    const VadAudioChunkOptions & options);

std::vector<runtime::TimeSpan> plan_vad_audio_chunks(
    const runtime::AudioBuffer & audio,
    runtime::IOfflineVoiceTaskSession & vad_session,
    const VadAudioChunkOptions & options);

std::vector<runtime::TimeSpan> plan_quiet_energy_audio_chunks(
    const std::vector<float> & mono_samples,
    const QuietEnergyAudioChunkOptions & options);

runtime::AudioBuffer slice_audio_buffer(
    const runtime::AudioBuffer & audio,
    const runtime::TimeSpan & span);

std::vector<float> make_triangular_overlap_window(int64_t chunk_samples);
std::vector<float> make_linear_fade_window(int64_t chunk_samples, int64_t fade_samples);

void copy_interleaved_chunk_to_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & input_interleaved,
    int64_t channels,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec);

void copy_planar_chunk(
    std::vector<float> & output_planar,
    const std::vector<float> & input_planar,
    int64_t lanes,
    int64_t input_frames,
    const AudioChunkSpan & span,
    const AudioChunkSpec & spec);

void overlap_add_planar_chunk(
    std::vector<float> & output_planar,
    std::vector<float> & weights,
    const std::vector<float> & chunk_planar,
    int64_t lanes,
    int64_t output_frames,
    const AudioChunkSpan & span,
    const std::vector<float> & window,
    AudioChunkCounterMode counter_mode);

void normalize_overlap_added_planar(
    std::vector<float> & output_planar,
    const std::vector<float> & weights,
    int64_t lanes,
    int64_t output_frames,
    AudioChunkCounterMode counter_mode);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & chunk_span);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span);

void append_chunk_word_timestamps(
    std::vector<runtime::WordTimestamp> & output,
    const std::vector<runtime::WordTimestamp> & chunk_words,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span,
    int64_t source_sample_rate,
    int64_t timestamp_sample_rate);

void append_chunk_speech_metadata(
    runtime::TaskResult & output,
    const runtime::TaskResult & chunk_result,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span,
    int64_t source_sample_rate,
    int64_t timestamp_sample_rate);

}  // namespace engine::audio

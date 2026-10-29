#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace engine::audio {

// Output sample formats the WAV writer can emit. The reader in wav_reader.h
// decodes PCM 8/16/24/32, float 32/64, A-law and mu-law; these are the three
// worth writing. 16-bit is the historical behaviour and stays the default so no
// existing caller changes.
enum class WavSampleFormat {
    Pcm16,
    Pcm24,
    Float32,
};

// Dither is a deliberate choice, not a universal improvement. It trades raw SNR
// for decorrelated quantisation error: measured on a -60 dBFS 997 Hz sine,
// undithered rounding leaves harmonics at -54.6 dBc while 2 LSB peak-to-peak
// TPDF pushes them to -66.1 dBc, at the cost of 4.6 dB of SNR. It must be
// applied exactly once, at the final quantisation to an integer format, so it
// stays opt-in: a chain that writes an intermediate 16-bit file and reads it
// back would otherwise dither twice.
enum class WavDitherMode {
    None,
    TriangularPdf,
};

// What to do with samples outside +/-1.0. HardClip is the historical behaviour:
// a per-sample clamp, which on a tone peaking 1 dB over full scale puts 30 % of
// samples on the rail and measures -26.7 dB THD+N. LookaheadLimit instead
// applies a smoothed broadband gain envelope that dips before the overshoot
// arrives, which on the same signal reaches the ceiling with 0 % of samples
// railed and -113 dB THD+N, at the cost of about 1 dB of level.
//
// A memoryless soft-clip waveshaper was tried first and rejected: shaping the
// top of the waveform measured -26.7 dB THD+N, indistinguishable from the hard
// clamp it replaced. Overshoot is a gain problem, not a curve problem.
enum class WavPeakPolicy {
    HardClip,
    LookaheadLimit,
};

struct LookaheadLimiterOptions {
    // Just under full scale, so the int quantiser below never rounds up onto
    // the rail.
    float ceiling = 0.995F;
    // Lookahead and release half-window. 5 ms is long enough to ride over a
    // 200 Hz cycle without audible pumping and short enough that a single
    // transient does not duck a whole bar.
    double window_seconds = 0.005;
};

struct WavWriteOptions {
    WavSampleFormat format = WavSampleFormat::Pcm16;
    WavDitherMode dither = WavDitherMode::None;
    WavPeakPolicy peak_policy = WavPeakPolicy::HardClip;
    LookaheadLimiterOptions limiter{};
    // Dither is generated from a deterministic per-call sequence so a written
    // file is reproducible; change the seed to decorrelate repeated renders.
    uint64_t dither_seed = 0x9E3779B97F4A7C15ULL;
};

// Smoothed lookahead peak limiter, for callers that need to manage peaks before
// the output stage (a summed mix, for example). Channels are gain-linked from a
// per-frame peak so the stereo image does not shift. Returns the largest gain
// reduction applied, in dB; returns 0 and leaves the buffer bit-identical when
// nothing exceeded the ceiling.
float apply_lookahead_limiter_in_place(
    std::vector<float> & samples,
    int channel_count,
    int sample_rate,
    const LookaheadLimiterOptions & options = {});

int wav_sample_format_bit_depth(WavSampleFormat format);
const char * wav_sample_format_name(WavSampleFormat format);

void write_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio,
    const WavWriteOptions & options = {});

// 16-bit PCM, hard clip, no dither. Unchanged behaviour for the ~70 call sites
// that use it; equivalent to write_wav with default options.
void write_pcm16_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio);

}  // namespace engine::audio

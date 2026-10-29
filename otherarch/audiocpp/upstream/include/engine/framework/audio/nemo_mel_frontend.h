#pragma once

#include "engine/framework/audio/dsp.h"

#include <cstdint>
#include <vector>

namespace engine::audio {

enum class MelInputRate { Resample, RequireMatch };
enum class WaveScale { None, DivideByMaxPlusEps };
enum class DitherMethod { None, Normal, BoxMuller16 };
enum class MelWindow { SymmetricPrecise, PeriodicF64, StftHann, FromArgument };
enum class MelBank { Slaney, HTK, FromArgument };
enum class MelPath { SparsePowerLn, LogMelSpectrogram, ComplexPowerLn, DensePowerLog10 };
enum class MelNorm { None, PerBinF32, PerBinF64, PerBinMixed };
enum class MelLayout { TimeMajor, FeatureMajor };
enum class ValidFrameRule { StftFrames, FloorHops, CeilHops, FullWindows };
enum class PadBasis { StftFrames, ValidFrames };

struct NemoMelFrontendConfig {
    int64_t sample_rate = 0;
    int64_t n_mels = 0;
    STFTConfig stft;
    MelInputRate input_rate = MelInputRate::Resample;
    WaveScale wave_scale = WaveScale::None;
    float preemphasis = 0.0f;
    float dither_stddev = 0.0f;
    DitherMethod dither_method = DitherMethod::None;
    MelWindow window = MelWindow::StftHann;
    MelBank mel_bank = MelBank::Slaney;
    MelPath mel_path = MelPath::SparsePowerLn;
    float log_zero_guard = 0x1p-24f;
    MelNorm norm = MelNorm::None;
    MelLayout layout = MelLayout::TimeMajor;
    int64_t frame_multiple = 1;
    PadBasis pad_basis = PadBasis::StftFrames;
    int64_t stack_frames = 1;
    bool add_deltas = false;
    float max_log_drop = 0.0f;
    float log_scale = 1.0f;
    float log_bias = 0.0f;
};

struct NemoMelRunConfig {
    bool center = true;
    ValidFrameRule valid_frame_rule = ValidFrameRule::FloorHops;
    // A nonnegative value overrides the calculated valid-frame count, as needed
    // by a streaming scheduler. A negative value means no override.
    int64_t valid_frames_override = -1;
};

struct NemoMelFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t raw_frames = 0;
    int64_t valid_frames = 0;
    int64_t feature_size = 0;
    double mel_ms = 0.0;
    double normalize_ms = 0.0;
};

class NemoMelFrontend {
public:
    // FromArgument uses caller-supplied values with canonical runtime shapes:
    // window [win_length] and filterbank [n_mels, n_fft / 2 + 1].
    NemoMelFrontend(
        NemoMelFrontendConfig config,
        std::vector<float> window_values = {},
        AudioTensor filterbank_values = {});

    NemoMelFeatures extract_audio(
        const std::vector<float> & interleaved,
        int sample_rate,
        int channels,
        const NemoMelRunConfig & run,
        size_t threads = 0) const;

    std::vector<float> prepare_audio(
        const std::vector<float> & interleaved,
        int sample_rate,
        int channels) const;

    NemoMelFeatures extract_mono(
        std::vector<float> mono,
        const NemoMelRunConfig & run,
        size_t threads = 0) const;

private:
    NemoMelFrontendConfig config_;
    std::vector<float> window_;
    AudioTensor dense_filterbank_;
    SparseMelFilterbank sparse_filterbank_;
};

}  // namespace engine::audio

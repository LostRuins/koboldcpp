#pragma once

#include "engine/framework/audio/dsp.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::audio {

enum class MelWaveformPadding { None, Reflect, ReflectOrRepeatSingleton };
enum class MelFilterbankProjection { DenseF32, SparseF32, DenseF64, DenseLongDouble };
enum class MelSpectrumMode { Magnitude, PowerBeforeProjection, PowerDuringProjection };
enum class MelOutputLayout { FeatureMajor, TimeMajor };
enum class MelResampleMode { RequireMatch, Linear, TorchaudioSincF64, SoxrQualityActualLength };
enum class MelLogPrecision { F32, F64 };
enum class MelValueTransform { Ln, Log10, None };
enum class MelHannWindow { Symmetric, Periodic };
enum class MelFilterbankNormalization { None, Slaney };

struct MelSpectrogramFrontendConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t win_length = 0;
    int64_t n_mels = 0;
    float mel_fmin = 0.0f;
    float mel_fmax = 0.0f;
    MelFilterbankNormalization filterbank_normalization = MelFilterbankNormalization::Slaney;
    STFTPadMode stft_pad_mode = STFTPadMode::Reflect;
    MelHannWindow window = MelHannWindow::Periodic;
    bool stft_center = false;
    MelWaveformPadding waveform_padding = MelWaveformPadding::Reflect;
    MelFilterbankProjection filterbank_projection = MelFilterbankProjection::DenseF32;
    MelSpectrumMode spectrum_mode = MelSpectrumMode::Magnitude;
    float magnitude_epsilon = 0.0f;
    double log_floor = 1.0e-5;
    MelLogPrecision log_precision = MelLogPrecision::F32;
    MelValueTransform value_transform = MelValueTransform::Ln;
    MelOutputLayout layout = MelOutputLayout::FeatureMajor;
    MelResampleMode resample_mode = MelResampleMode::RequireMatch;
    bool require_mono = false;
    int64_t minimum_samples = 0;
    int64_t drop_last_frames = 0;
    int64_t max_frames = 0;
    int64_t frame_multiple = 1;
    float frame_pad_value = 0.0f;
    float log_dynamic_range = 0.0f;
    float log_shift = 0.0f;
    float log_divisor = 1.0f;

    bool operator==(const MelSpectrogramFrontendConfig & other) const noexcept;
};

struct MelSpectrogramFeatures {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t mel_bins = 0;
};

class MelSpectrogramFrontend {
public:
    explicit MelSpectrogramFrontend(MelSpectrogramFrontendConfig config);

    MelSpectrogramFeatures extract_mono(const std::vector<float> & mono, size_t threads = 0) const;
    MelSpectrogramFeatures extract_planar(
        const std::vector<float> & planar, int64_t channels, size_t threads = 0) const;
    MelSpectrogramFeatures extract_audio(
        const std::vector<float> & interleaved, int sample_rate, int channels,
        size_t threads = 0) const;

private:
    MelSpectrogramFrontendConfig config_;
    AudioTensor filterbank_;
    SparseMelFilterbank sparse_filterbank_;
};

std::shared_ptr<const MelSpectrogramFrontend> get_cached_mel_spectrogram_frontend(
    const MelSpectrogramFrontendConfig & config);

}  // namespace engine::audio

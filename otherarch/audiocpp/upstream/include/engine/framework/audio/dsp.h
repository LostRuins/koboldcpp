#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::audio {

struct AudioTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

enum class STFTPadMode {
    Reflect,
    Constant,
};

enum class STFTFamily {
    Default,
    Kokoro,
    // Periodic Hann with a float32 angular step and float32 cosine.
    PeriodicF32,
};

struct STFTConfig {
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t win_length = 0;
    bool center = true;
    STFTPadMode pad_mode = STFTPadMode::Reflect;
    STFTFamily family = STFTFamily::Default;
};

const std::vector<float> & get_cached_stft_window(const STFTConfig & config);

// Speech-focused STFT surface used by voice models in this repo.
// It assumes real float input and returns one-sided complex or magnitude output.
class STFT {
public:
    AudioTensor compute_magnitude(
        const std::vector<float> & waveform,
        const std::vector<float> & window,
        int64_t batch,
        int64_t samples,
        const STFTConfig & config,
        size_t threads = 0) const;

    AudioTensor compute_complex(
        const std::vector<float> & waveform,
        const std::vector<float> & window,
        int64_t batch,
        int64_t samples,
        const STFTConfig & config,
        size_t threads = 0) const;
};

class ISTFT {
public:
    AudioTensor compute(
        const std::vector<float> & complex_spec,
        const std::vector<float> & window,
        int64_t batch,
        int64_t freq_bins,
        int64_t frames,
        int64_t samples,
        const STFTConfig & config,
        size_t threads = 0) const;
};

struct MelFilterbankConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t n_mels = 0;
    float lowfreq = 0.0f;
    float highfreq = 0.0f;
    bool slaney_norm = true;
};

struct SparseMelFilterbank {
    AudioTensor dense;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
};

struct WhisperLogMelConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 400;
    int64_t hop_length = 160;
    int64_t feature_size = 80;
    STFTFamily stft_family = STFTFamily::Default;
};

struct WhisperLogMelFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

class MelFilterbank {
public:
    AudioTensor build(const MelFilterbankConfig & config) const;
    SparseMelFilterbank build_sparse(const MelFilterbankConfig & config) const;
    SparseMelFilterbank prepare_sparse(const AudioTensor & filterbank) const;

    AudioTensor compute(
        const std::vector<float> & power_spec,
        int64_t batch,
        int64_t freq_bins,
        int64_t frames,
        const MelFilterbankConfig & config) const;

    AudioTensor compute_custom(
        const std::vector<float> & power_spec,
        int64_t batch,
        int64_t freq_bins,
        int64_t frames,
        const AudioTensor & filterbank) const;

    AudioTensor compute_custom_sparse_from_magnitude(
        const std::vector<float> & magnitude,
        int64_t batch,
        int64_t freq_bins,
        int64_t stft_frames,
        int64_t output_frames,
        const SparseMelFilterbank & filterbank) const;
};

class WhisperLogMelExtractor {
public:
    explicit WhisperLogMelExtractor(WhisperLogMelConfig config);

    const WhisperLogMelConfig & config() const noexcept;
    WhisperLogMelFeatures compute(
        const std::vector<float> & samples,
        size_t threads = 0) const;

private:
    WhisperLogMelConfig config_;
    SparseMelFilterbank filterbank_;
};

class MelSpectrogram {
public:
    AudioTensor compute(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        int64_t sample_rate,
        const STFTConfig & config,
        int64_t n_mels,
        size_t threads = 0) const;
};

enum class FeatureNormalizeType {
    None,
    PerFeature,
    AllFeatures,
};

struct FeatureNormalizeOutput {
    AudioTensor normalized;
    AudioTensor mean;
    AudioTensor stddev;
};

class FeatureNormalizer {
public:
    FeatureNormalizeOutput compute(
        const std::vector<float> & features,
        const std::vector<int64_t> & seq_len,
        int64_t batch,
        int64_t feature_dim,
        int64_t frames,
        FeatureNormalizeType normalize_type) const;
};

class LogMelSpectrogram {
public:
    AudioTensor compute(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        int64_t sample_rate,
        const STFTConfig & config,
        int64_t n_mels,
        size_t threads = 0) const;

    AudioTensor compute(
        const std::vector<float> & waveform,
        const std::vector<float> & window,
        int64_t batch,
        int64_t samples,
        const STFTConfig & config,
        const AudioTensor & filterbank,
        size_t threads = 0) const;
};

}  // namespace engine::audio

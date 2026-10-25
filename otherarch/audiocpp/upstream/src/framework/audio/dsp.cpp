#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace engine::audio {

namespace {

constexpr long double kPi = 3.14159265358979323846264338327950288L;

template <typename Fn>
double measure_ms(Fn && fn) {
    const auto started = std::chrono::steady_clock::now();
    std::forward<Fn>(fn)();
    const auto ended = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(ended - started).count();
}

int64_t checked_product(std::initializer_list<int64_t> dims) {
    int64_t result = 1;
    for (const int64_t dim : dims) {
        result *= dim;
    }
    return result;
}

int64_t reflect_index(int64_t index, int64_t length) {
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

float hz_to_mel(float hz) {
    constexpr float kFsp = 200.0f / 3.0f;
    constexpr float kMinLogHz = 1000.0f;
    constexpr float kMinLogMel = kMinLogHz / kFsp;
    constexpr float kLogstep = 0.06875177742094912f;  // ln(6.4) / 27
    if (hz >= kMinLogHz) {
        return kMinLogMel + std::log(hz / kMinLogHz) / kLogstep;
    }
    return hz / kFsp;
}

float mel_to_hz(float mel) {
    constexpr float kFsp = 200.0f / 3.0f;
    constexpr float kMinLogHz = 1000.0f;
    constexpr float kMinLogMel = kMinLogHz / kFsp;
    constexpr float kLogstep = 0.06875177742094912f;  // ln(6.4) / 27
    if (mel >= kMinLogMel) {
        return kMinLogHz * std::exp(kLogstep * (mel - kMinLogMel));
    }
    return mel * kFsp;
}

std::vector<float> make_hann_window(int64_t win_length, STFTFamily family) {
    std::vector<float> window(static_cast<size_t>(win_length), 0.0f);
    if (win_length <= 1) {
        if (win_length == 1) {
            window[0] = 1.0f;
        }
        return window;
    }
    if (family == STFTFamily::PeriodicF32) {
        const float step = static_cast<float>(2.0 * static_cast<double>(kPi) / win_length);
        for (int64_t i = 0; i < win_length; ++i) {
            window[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos(step * static_cast<float>(i));
        }
        return window;
    }
    const float denom = family == STFTFamily::Kokoro
                            ? static_cast<float>(win_length)
                            : static_cast<float>(win_length - 1);
    for (int64_t i = 0; i < win_length; ++i) {
        window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / denom);
    }
    return window;
}

struct STFTWindowKey {
    int64_t win_length = 0;
    STFTFamily family = STFTFamily::Default;

    bool operator==(const STFTWindowKey & other) const noexcept {
        return win_length == other.win_length && family == other.family;
    }
};

struct STFTWindowKeyHash {
    size_t operator()(const STFTWindowKey & key) const noexcept {
        size_t hash = static_cast<size_t>(key.win_length);
        hash = hash * 1315423911ULL + static_cast<size_t>(key.family);
        return hash;
    }
};

struct MelFilterbankKey {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t n_mels = 0;
    float lowfreq = 0.0f;
    float highfreq = 0.0f;
    bool slaney_norm = true;

    bool operator==(const MelFilterbankKey & other) const noexcept {
        return sample_rate == other.sample_rate &&
               n_fft == other.n_fft &&
               n_mels == other.n_mels &&
               lowfreq == other.lowfreq &&
               highfreq == other.highfreq &&
               slaney_norm == other.slaney_norm;
    }
};

struct MelFilterbankKeyHash {
    size_t operator()(const MelFilterbankKey & key) const noexcept {
        size_t hash = static_cast<size_t>(key.sample_rate);
        hash = hash * 1315423911ULL + static_cast<size_t>(key.n_fft);
        hash = hash * 1315423911ULL + static_cast<size_t>(key.n_mels);
        hash = hash * 1315423911ULL + std::hash<float>{}(key.lowfreq);
        hash = hash * 1315423911ULL + std::hash<float>{}(key.highfreq);
        hash = hash * 1315423911ULL + static_cast<size_t>(key.slaney_norm);
        return hash;
    }
};

const std::vector<float> & get_cached_window_impl(const STFTConfig & config) {
    static std::mutex mutex;
    static std::unordered_map<STFTWindowKey, std::vector<float>, STFTWindowKeyHash> cache;

    const STFTWindowKey key{
        config.win_length,
        config.family,
    };

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(key, make_hann_window(config.win_length, config.family)).first->second;
}

std::vector<float> make_mel_filterbank(const MelFilterbankConfig & config) {
    const float mel_min = hz_to_mel(config.lowfreq);
    const float mel_max = hz_to_mel(config.highfreq > 0.0f ? config.highfreq : static_cast<float>(config.sample_rate) / 2.0f);

    std::vector<float> mel_points(static_cast<size_t>(config.n_mels + 2));
    for (int64_t i = 0; i < config.n_mels + 2; ++i) {
        mel_points[static_cast<size_t>(i)] =
            mel_min + (mel_max - mel_min) * static_cast<float>(i) / static_cast<float>(config.n_mels + 1);
    }

    std::vector<float> hz_points(static_cast<size_t>(config.n_mels + 2));
    for (int64_t i = 0; i < config.n_mels + 2; ++i) {
        const float hz = mel_to_hz(mel_points[static_cast<size_t>(i)]);
        hz_points[static_cast<size_t>(i)] = hz;
    }

    const int64_t freq_bins = (config.n_fft / 2) + 1;
    std::vector<float> filterbank(static_cast<size_t>(config.n_mels * freq_bins), 0.0f);
    std::vector<float> fftfreqs(static_cast<size_t>(freq_bins), 0.0f);
    for (int64_t i = 0; i < freq_bins; ++i) {
        fftfreqs[static_cast<size_t>(i)] =
            static_cast<float>(config.sample_rate) * 0.5f * static_cast<float>(i) / static_cast<float>(freq_bins - 1);
    }
    for (int64_t mel = 0; mel < config.n_mels; ++mel) {
        const float left = hz_points[static_cast<size_t>(mel)];
        const float center = hz_points[static_cast<size_t>(mel + 1)];
        const float right = hz_points[static_cast<size_t>(mel + 2)];
        const float lower_width = std::max(center - left, 1e-12f);
        const float upper_width = std::max(right - center, 1e-12f);
        for (int64_t i = 0; i < freq_bins; ++i) {
            const float freq = fftfreqs[static_cast<size_t>(i)];
            const float lower = (freq - left) / lower_width;
            const float upper = (right - freq) / upper_width;
            filterbank[static_cast<size_t>(mel * freq_bins + i)] = std::max(0.0f, std::min(lower, upper));
        }
        if (config.slaney_norm) {
            const float enorm = 2.0f / std::max(hz_points[static_cast<size_t>(mel + 2)] - hz_points[static_cast<size_t>(mel)], 1e-12f);
            for (int64_t i = 0; i < freq_bins; ++i) {
                filterbank[static_cast<size_t>(mel * freq_bins + i)] *= enorm;
            }
        }
    }
    return filterbank;
}

const std::vector<float> & get_cached_mel_filterbank(const MelFilterbankConfig & config) {
    static std::mutex mutex;
    static std::unordered_map<MelFilterbankKey, std::vector<float>, MelFilterbankKeyHash> cache;

    const MelFilterbankKey key{
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.lowfreq,
        config.highfreq,
        config.slaney_norm,
    };

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(key, make_mel_filterbank(config)).first->second;
}

std::vector<int64_t> sparse_mel_starts(const AudioTensor & filterbank) {
    const int64_t n_mels = filterbank.shape[0];
    const int64_t freq_bins = filterbank.shape[1];
    std::vector<int64_t> starts(static_cast<size_t>(n_mels), 0);
    for (int64_t mel = 0; mel < n_mels; ++mel) {
        int64_t start = 0;
        while (start < freq_bins && filterbank.values[static_cast<size_t>(mel * freq_bins + start)] == 0.0f) {
            ++start;
        }
        starts[static_cast<size_t>(mel)] = start;
    }
    return starts;
}

std::vector<int64_t> sparse_mel_ends(const AudioTensor & filterbank) {
    const int64_t n_mels = filterbank.shape[0];
    const int64_t freq_bins = filterbank.shape[1];
    std::vector<int64_t> ends(static_cast<size_t>(n_mels), freq_bins);
    for (int64_t mel = 0; mel < n_mels; ++mel) {
        int64_t end = freq_bins;
        while (end > 0 && filterbank.values[static_cast<size_t>(mel * freq_bins + end - 1)] == 0.0f) {
            --end;
        }
        ends[static_cast<size_t>(mel)] = end;
    }
    return ends;
}

FeatureNormalizeOutput normalize_batch_impl(
    const std::vector<float> & features,
    const std::vector<int64_t> & seq_len,
    int64_t batch,
    int64_t feature_dim,
    int64_t frames,
    FeatureNormalizeType normalize_type) {
    if (static_cast<int64_t>(features.size()) != checked_product({batch, feature_dim, frames})) {
        throw std::runtime_error("FeatureNormalizer input size mismatch");
    }
    if (static_cast<int64_t>(seq_len.size()) != batch) {
        throw std::runtime_error("FeatureNormalizer seq_len size mismatch");
    }

    FeatureNormalizeOutput output;
    output.normalized.shape = {batch, feature_dim, frames};
    output.normalized.values = features;

    if (normalize_type == FeatureNormalizeType::None) {
        output.mean.shape = {batch};
        output.mean.values.assign(static_cast<size_t>(batch), 0.0f);
        output.stddev.shape = {batch};
        output.stddev.values.assign(static_cast<size_t>(batch), 0.0f);
        return output;
    }

    if (normalize_type == FeatureNormalizeType::PerFeature) {
        output.mean.shape = {batch, feature_dim};
        output.mean.values.assign(static_cast<size_t>(batch * feature_dim), 0.0f);
        output.stddev.shape = {batch, feature_dim};
        output.stddev.values.assign(static_cast<size_t>(batch * feature_dim), 0.0f);

#ifdef _OPENMP
        #pragma omp parallel for if(batch * feature_dim >= 32)
#endif
        for (int64_t b = 0; b < batch; ++b) {
            const int64_t valid = std::clamp<int64_t>(seq_len[static_cast<size_t>(b)], 0, frames);
            for (int64_t f = 0; f < feature_dim; ++f) {
                float mean = 0.0f;
                for (int64_t t = 0; t < valid; ++t) {
                    mean += features[static_cast<size_t>(((b * feature_dim) + f) * frames + t)];
                }
                if (valid > 0) {
                    mean /= static_cast<float>(valid);
                }
                output.mean.values[static_cast<size_t>(b * feature_dim + f)] = mean;

                float variance_sum = 0.0f;
                for (int64_t t = 0; t < valid; ++t) {
                    const float delta = features[static_cast<size_t>(((b * feature_dim) + f) * frames + t)] - mean;
                    variance_sum += delta * delta;
                }
                float stddev = 0.0f;
                if (valid > 1) {
                    stddev = std::sqrt(variance_sum / static_cast<float>(valid - 1));
                }
                stddev += 1e-5f;
                output.stddev.values[static_cast<size_t>(b * feature_dim + f)] = stddev;

                for (int64_t t = 0; t < frames; ++t) {
                    const size_t idx = static_cast<size_t>(((b * feature_dim) + f) * frames + t);
                    output.normalized.values[idx] = t < valid ? (features[idx] - mean) / stddev : 0.0f;
                }
            }
        }
        return output;
    }

    output.mean.shape = {batch};
    output.mean.values.assign(static_cast<size_t>(batch), 0.0f);
    output.stddev.shape = {batch};
    output.stddev.values.assign(static_cast<size_t>(batch), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for if(batch >= 4)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t valid = std::clamp<int64_t>(seq_len[static_cast<size_t>(b)], 0, frames);
        const int64_t count = valid * feature_dim;
        float mean = 0.0f;
        for (int64_t f = 0; f < feature_dim; ++f) {
            for (int64_t t = 0; t < valid; ++t) {
                mean += features[static_cast<size_t>(((b * feature_dim) + f) * frames + t)];
            }
        }
        if (count > 0) {
            mean /= static_cast<float>(count);
        }
        output.mean.values[static_cast<size_t>(b)] = mean;

        float variance_sum = 0.0f;
        for (int64_t f = 0; f < feature_dim; ++f) {
            for (int64_t t = 0; t < valid; ++t) {
                const float delta = features[static_cast<size_t>(((b * feature_dim) + f) * frames + t)] - mean;
                variance_sum += delta * delta;
            }
        }
        float stddev = 0.0f;
        if (count > 1) {
            stddev = std::sqrt(variance_sum / static_cast<float>(count - 1));
        }
        stddev += 1e-5f;
        output.stddev.values[static_cast<size_t>(b)] = stddev;

        for (int64_t f = 0; f < feature_dim; ++f) {
            for (int64_t t = 0; t < frames; ++t) {
                const size_t idx = static_cast<size_t>(((b * feature_dim) + f) * frames + t);
                output.normalized.values[idx] = (features[idx] - mean) / stddev;
            }
        }
    }
    return output;
}

}

AudioTensor STFT::compute_magnitude(
    const std::vector<float> & waveform,
    const std::vector<float> & window,
    int64_t batch,
    int64_t samples,
    const STFTConfig & config,
    size_t threads) const {
    auto complex_result = compute_complex(waveform, window, batch, samples, config, threads);
    AudioTensor result;
    result.shape = {complex_result.shape[0], complex_result.shape[1], complex_result.shape[2]};
    result.values.resize(static_cast<size_t>(checked_product({result.shape[0], result.shape[1], result.shape[2]})));
    for (size_t i = 0, out = 0; i < complex_result.values.size(); i += 2, ++out) {
        const float re = complex_result.values[i];
        const float im = complex_result.values[i + 1];
        result.values[out] = std::sqrt(re * re + im * im);
    }
    return result;
}

AudioTensor STFT::compute_complex(
    const std::vector<float> & waveform,
    const std::vector<float> & window,
    int64_t batch,
    int64_t samples,
    const STFTConfig & config,
    size_t threads) const {
    if (static_cast<int64_t>(waveform.size()) != checked_product({batch, samples}) ||
        static_cast<int64_t>(window.size()) != config.win_length) {
        throw std::runtime_error("STFT input size mismatch");
    }

    const int64_t pad = config.center ? config.n_fft / 2 : 0;
    const int64_t frames = 1 + (samples + 2 * pad - config.n_fft) / config.hop_length;
    const int64_t freq_bins = (config.n_fft / 2) + 1;
    const int64_t window_offset = (config.n_fft - config.win_length) / 2;

    std::vector<float> framed(static_cast<size_t>(checked_product({batch, frames, config.n_fft})), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * frames >= 8)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            const float * signal = waveform.data() + static_cast<size_t>(b * samples);
            const int64_t start = frame_index * config.hop_length - pad;
            float * frame = framed.data() + static_cast<size_t>((b * frames + frame_index) * config.n_fft);
            for (int64_t i = 0; i < config.win_length; ++i) {
                const int64_t sample_index = start + window_offset + i;
                float sample = 0.0f;
                if (sample_index >= 0 && sample_index < samples) {
                    sample = signal[sample_index];
                } else if (config.pad_mode == STFTPadMode::Reflect) {
                    sample = signal[reflect_index(sample_index, samples)];
                }
                frame[window_offset + i] = sample * window[static_cast<size_t>(i)];
            }
        }
    }

    TensorShape shape_in{
        static_cast<size_t>(batch),
        static_cast<size_t>(frames),
        static_cast<size_t>(config.n_fft),
    };
    TensorStrideBytes stride_in{
        static_cast<std::ptrdiff_t>(frames * config.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };
    TensorStrideBytes stride_out{
        static_cast<std::ptrdiff_t>(frames * freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };

    std::vector<std::complex<float>> spectrum(
        static_cast<size_t>(checked_product({batch, frames, freq_bins})),
        std::complex<float>(0.0f, 0.0f));
    real_fft_forward(
        shape_in,
        stride_in,
        stride_out,
        static_cast<size_t>(2),
        framed.data(),
        spectrum.data(),
        1.0f,
        threads);

    AudioTensor result;
    result.shape = {batch, freq_bins, frames, 2};
    result.values.assign(static_cast<size_t>(checked_product({batch, freq_bins, frames, 2})), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * freq_bins * frames >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < freq_bins; ++f) {
            for (int64_t t = 0; t < frames; ++t) {
                const auto value = spectrum[static_cast<size_t>((b * frames + t) * freq_bins + f)];
                const size_t base = static_cast<size_t>((((b * freq_bins) + f) * frames + t) * 2);
                result.values[base] = value.real();
                result.values[base + 1] = value.imag();
            }
        }
    }
    return result;
}

AudioTensor ISTFT::compute(
    const std::vector<float> & complex_spec,
    const std::vector<float> & window,
    int64_t batch,
    int64_t freq_bins,
    int64_t frames,
    int64_t samples,
    const STFTConfig & config,
    size_t threads) const {
    if (static_cast<int64_t>(complex_spec.size()) != checked_product({batch, freq_bins, frames, 2}) ||
        static_cast<int64_t>(window.size()) != config.win_length) {
        throw std::runtime_error("ISTFT input size mismatch");
    }

    const int64_t pad = config.n_fft / 2;
    const int64_t padded_samples = samples + 2 * pad;
    const int64_t usable_window = std::min<int64_t>(config.win_length, config.n_fft);

    std::vector<std::complex<float>> spectrum(
        static_cast<size_t>(checked_product({batch, frames, freq_bins})),
        std::complex<float>(0.0f, 0.0f));
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * frames * freq_bins >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            for (int64_t k = 0; k < freq_bins; ++k) {
                const size_t src = static_cast<size_t>((((b * freq_bins) + k) * frames + frame_index) * 2);
                spectrum[static_cast<size_t>((b * frames + frame_index) * freq_bins + k)] = {
                    complex_spec[src],
                    complex_spec[src + 1],
                };
            }
        }
    }

    TensorShape output_shape{
        static_cast<size_t>(batch),
        static_cast<size_t>(frames),
        static_cast<size_t>(config.n_fft),
    };
    TensorStrideBytes input_strides{
        static_cast<std::ptrdiff_t>(frames * freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(freq_bins * static_cast<int64_t>(sizeof(std::complex<float>))),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
    };
    TensorStrideBytes output_strides{
        static_cast<std::ptrdiff_t>(frames * config.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(config.n_fft * static_cast<int64_t>(sizeof(float))),
        static_cast<std::ptrdiff_t>(sizeof(float)),
    };

    std::vector<float> framed(static_cast<size_t>(checked_product({batch, frames, config.n_fft})), 0.0f);
    real_fft_inverse(
        output_shape,
        input_strides,
        output_strides,
        2,
        spectrum.data(),
        framed.data(),
        1.0f / static_cast<float>(config.n_fft),
        threads);

    std::vector<float> window_sq(static_cast<size_t>(usable_window), 0.0f);
    for (int64_t i = 0; i < usable_window; ++i) {
        const float w = window[static_cast<size_t>(i)];
        window_sq[static_cast<size_t>(i)] = w * w;
    }

    std::vector<float> accum(static_cast<size_t>(batch * padded_samples), 0.0f);
    std::vector<float> window_sums(static_cast<size_t>(batch * padded_samples), 0.0f);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            const int64_t start = frame_index * config.hop_length;
            const float * frame = framed.data() + static_cast<size_t>((b * frames + frame_index) * config.n_fft);
            float * accum_row = accum.data() + static_cast<size_t>(b * padded_samples + start);
            float * window_row = window_sums.data() + static_cast<size_t>(b * padded_samples + start);
            for (int64_t n = 0; n < usable_window; ++n) {
                accum_row[n] += frame[n] * window[static_cast<size_t>(n)];
                window_row[n] += window_sq[static_cast<size_t>(n)];
            }
        }
    }

    AudioTensor result;
    result.shape = {batch, samples};
    result.values.assign(static_cast<size_t>(batch * samples), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < samples; ++i) {
            const size_t padded_index = static_cast<size_t>(b * padded_samples + i + pad);
            const float denom = window_sums[padded_index] > 1e-8f ? window_sums[padded_index] : 1.0f;
            result.values[static_cast<size_t>(b * samples + i)] = accum[padded_index] / denom;
        }
    }
    return result;
}

AudioTensor MelFilterbank::build(const MelFilterbankConfig & config) const {
    if (config.sample_rate <= 0 || config.n_fft <= 0 || config.n_mels <= 0) {
        throw std::runtime_error("MelFilterbank config values must be positive");
    }
    AudioTensor result;
    result.shape = {config.n_mels, (config.n_fft / 2) + 1};
    result.values = get_cached_mel_filterbank(config);
    return result;
}

SparseMelFilterbank MelFilterbank::build_sparse(const MelFilterbankConfig & config) const {
    return prepare_sparse(build(config));
}

SparseMelFilterbank MelFilterbank::prepare_sparse(const AudioTensor & filterbank) const {
    if (filterbank.shape.size() != 2) {
        throw std::runtime_error("SparseMelFilterbank requires rank-2 filterbank");
    }
    const int64_t n_mels = filterbank.shape[0];
    const int64_t freq_bins = filterbank.shape[1];
    if (n_mels <= 0 || freq_bins <= 0 ||
        static_cast<int64_t>(filterbank.values.size()) != checked_product({n_mels, freq_bins})) {
        throw std::runtime_error("SparseMelFilterbank filterbank shape mismatch");
    }
    SparseMelFilterbank sparse;
    sparse.dense = filterbank;
    sparse.starts = sparse_mel_starts(sparse.dense);
    sparse.ends = sparse_mel_ends(sparse.dense);
    return sparse;
}

AudioTensor MelFilterbank::compute(
    const std::vector<float> & power_spec,
    int64_t batch,
    int64_t freq_bins,
    int64_t frames,
    const MelFilterbankConfig & config) const {
    if (static_cast<int64_t>(power_spec.size()) != checked_product({batch, freq_bins, frames})) {
        throw std::runtime_error("MelFilterbank input size mismatch");
    }
    const auto filterbank = build(config);
    if (filterbank.shape[1] != freq_bins) {
        throw std::runtime_error("MelFilterbank freq_bins mismatch");
    }
    AudioTensor result;
    result.shape = {batch, config.n_mels, frames};
    result.values.assign(static_cast<size_t>(checked_product({batch, config.n_mels, frames})), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * config.n_mels * frames >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t m = 0; m < config.n_mels; ++m) {
            for (int64_t t = 0; t < frames; ++t) {
                long double sum = 0.0;
                for (int64_t f = 0; f < freq_bins; ++f) {
                    sum += static_cast<long double>(filterbank.values[static_cast<size_t>(m * freq_bins + f)]) *
                           static_cast<long double>(power_spec[static_cast<size_t>(((b * freq_bins) + f) * frames + t)]);
                }
                result.values[static_cast<size_t>(((b * config.n_mels) + m) * frames + t)] = static_cast<float>(sum);
            }
        }
    }
    return result;
}

AudioTensor MelFilterbank::compute_custom(
    const std::vector<float> & power_spec,
    int64_t batch,
    int64_t freq_bins,
    int64_t frames,
    const AudioTensor & filterbank) const {
    if (static_cast<int64_t>(power_spec.size()) != checked_product({batch, freq_bins, frames})) {
        throw std::runtime_error("MelFilterbank input size mismatch");
    }
    if (filterbank.shape.size() != 2 || filterbank.shape[1] != freq_bins) {
        throw std::runtime_error("MelFilterbank custom filterbank shape mismatch");
    }
    const int64_t n_mels = filterbank.shape[0];
    if (static_cast<int64_t>(filterbank.values.size()) != checked_product({n_mels, freq_bins})) {
        throw std::runtime_error("MelFilterbank custom filterbank value count mismatch");
    }

    AudioTensor result;
    result.shape = {batch, n_mels, frames};
    result.values.assign(static_cast<size_t>(checked_product({batch, n_mels, frames})), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * n_mels * frames >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t m = 0; m < n_mels; ++m) {
            for (int64_t t = 0; t < frames; ++t) {
                long double sum = 0.0;
                for (int64_t f = 0; f < freq_bins; ++f) {
                    sum += static_cast<long double>(filterbank.values[static_cast<size_t>(m * freq_bins + f)]) *
                           static_cast<long double>(power_spec[static_cast<size_t>(((b * freq_bins) + f) * frames + t)]);
                }
                result.values[static_cast<size_t>(((b * n_mels) + m) * frames + t)] = static_cast<float>(sum);
            }
        }
    }
    return result;
}

AudioTensor MelFilterbank::compute_custom_sparse_from_magnitude(
    const std::vector<float> & magnitude,
    int64_t batch,
    int64_t freq_bins,
    int64_t stft_frames,
    int64_t output_frames,
    const SparseMelFilterbank & filterbank) const {
    if (batch <= 0 || freq_bins <= 0 || stft_frames <= 0 || output_frames <= 0 || output_frames > stft_frames ||
        static_cast<int64_t>(magnitude.size()) != checked_product({batch, freq_bins, stft_frames})) {
        throw std::runtime_error("SparseMelFilterbank magnitude input size mismatch");
    }
    if (filterbank.dense.shape.size() != 2 || filterbank.dense.shape[1] != freq_bins) {
        throw std::runtime_error("SparseMelFilterbank filterbank shape mismatch");
    }
    const int64_t n_mels = filterbank.dense.shape[0];
    if (static_cast<int64_t>(filterbank.dense.values.size()) != checked_product({n_mels, freq_bins}) ||
        static_cast<int64_t>(filterbank.starts.size()) != n_mels ||
        static_cast<int64_t>(filterbank.ends.size()) != n_mels) {
        throw std::runtime_error("SparseMelFilterbank metadata shape mismatch");
    }
    for (int64_t m = 0; m < n_mels; ++m) {
        const int64_t start = filterbank.starts[static_cast<size_t>(m)];
        const int64_t end = filterbank.ends[static_cast<size_t>(m)];
        if (start < 0 || end < start || end > freq_bins) {
            throw std::runtime_error("SparseMelFilterbank metadata range mismatch");
        }
    }

    AudioTensor result;
    result.shape = {batch, n_mels, output_frames};
    result.values.assign(static_cast<size_t>(checked_product({batch, n_mels, output_frames})), 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * n_mels * output_frames >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t m = 0; m < n_mels; ++m) {
            for (int64_t t = 0; t < output_frames; ++t) {
                double sum = 0.0;
                int64_t f = filterbank.starts[static_cast<size_t>(m)];
                const int64_t end = filterbank.ends[static_cast<size_t>(m)];
                for (; f + 3 < end; f += 4) {
                    const size_t filter_base = static_cast<size_t>(m * freq_bins + f);
                    const size_t mag_base = static_cast<size_t>(((b * freq_bins) + f) * stft_frames + t);
                    const float mag0 = magnitude[mag_base];
                    const float mag1 = magnitude[mag_base + static_cast<size_t>(stft_frames)];
                    const float mag2 = magnitude[mag_base + static_cast<size_t>(2 * stft_frames)];
                    const float mag3 = magnitude[mag_base + static_cast<size_t>(3 * stft_frames)];
                    sum += static_cast<double>(mag0 * mag0) *
                           static_cast<double>(filterbank.dense.values[filter_base]);
                    sum += static_cast<double>(mag1 * mag1) *
                           static_cast<double>(filterbank.dense.values[filter_base + 1]);
                    sum += static_cast<double>(mag2 * mag2) *
                           static_cast<double>(filterbank.dense.values[filter_base + 2]);
                    sum += static_cast<double>(mag3 * mag3) *
                           static_cast<double>(filterbank.dense.values[filter_base + 3]);
                }
                for (; f < end; ++f) {
                    const float mag = magnitude[static_cast<size_t>(((b * freq_bins) + f) * stft_frames + t)];
                    sum += static_cast<double>(mag * mag) *
                           static_cast<double>(filterbank.dense.values[static_cast<size_t>(m * freq_bins + f)]);
                }
                result.values[static_cast<size_t>(((b * n_mels) + m) * output_frames + t)] = static_cast<float>(sum);
            }
        }
    }
    return result;
}

WhisperLogMelExtractor::WhisperLogMelExtractor(WhisperLogMelConfig config)
    : config_(config),
      filterbank_(MelFilterbank().build_sparse(
          MelFilterbankConfig{
              config_.sample_rate,
              config_.n_fft,
              config_.feature_size,
              0.0F,
              static_cast<float>(config_.sample_rate) / 2.0F,
              true,
          })) {}

const WhisperLogMelConfig & WhisperLogMelExtractor::config() const noexcept {
    return config_;
}

WhisperLogMelFeatures WhisperLogMelExtractor::compute(
    const std::vector<float> & samples,
    size_t threads) const {
    if (config_.sample_rate <= 0 || config_.feature_size <= 0 || config_.hop_length <= 0 || config_.n_fft <= 0) {
        throw std::runtime_error("Whisper log-mel config is invalid");
    }
    if (samples.empty()) {
        throw std::runtime_error("Whisper log-mel input is empty");
    }
    const STFTConfig stft_config{
        config_.n_fft,
        config_.hop_length,
        config_.n_fft,
        true,
        STFTPadMode::Reflect,
        config_.stft_family,
    };
    const auto & window = get_cached_stft_window(stft_config);
    auto magnitude = STFT().compute_magnitude(
        samples,
        window,
        1,
        static_cast<int64_t>(samples.size()),
        stft_config,
        threads);
    if (magnitude.shape.size() != 3) {
        throw std::runtime_error("Whisper log-mel STFT returned unexpected rank");
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (stft_frames <= 1) {
        throw std::runtime_error("Whisper log-mel STFT produced too few frames");
    }
    WhisperLogMelFeatures out;
    out.mel_bins = config_.feature_size;
    out.frames = stft_frames - 1;
    auto mel = MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        stft_frames,
        out.frames,
        filterbank_);
    if (mel.shape.size() != 3 || mel.shape[1] != out.mel_bins || mel.shape[2] != out.frames) {
        throw std::runtime_error("Whisper log-mel frontend returned unexpected shape");
    }
    float max_log = -INFINITY;
    for (float & value : mel.values) {
        value = std::log10(std::max(value, 1.0e-10F));
        max_log = std::max(max_log, value);
    }
    const float floor = max_log - 8.0F;
    for (float & value : mel.values) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    out.values = std::move(mel.values);
    return out;
}

AudioTensor MelSpectrogram::compute(
    const std::vector<float> & waveform,
    int64_t batch,
    int64_t samples,
    int64_t sample_rate,
    const STFTConfig & config,
    int64_t n_mels,
    size_t threads) const {
    const auto & window = get_cached_window_impl(config);
    AudioTensor magnitude;
    const double stft_ms = measure_ms([&]() {
        magnitude = STFT().compute_magnitude(waveform, window, batch, samples, config, threads);
    });
    if (debug::timing_log_enabled()) {
        debug::timing_log_scalar("audio.stft_ms", stft_ms);
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    std::vector<float> power_spec(magnitude.values.size(), 0.0f);
    for (size_t i = 0; i < magnitude.values.size(); ++i) {
        power_spec[i] = magnitude.values[i] * magnitude.values[i];
    }

    return MelFilterbank().compute(
        power_spec,
        batch,
        freq_bins,
        frames,
        MelFilterbankConfig{sample_rate, config.n_fft, n_mels, 0.0f, 0.0f, true});
}

AudioTensor LogMelSpectrogram::compute(
    const std::vector<float> & waveform,
    int64_t batch,
    int64_t samples,
    int64_t sample_rate,
    const STFTConfig & config,
    int64_t n_mels,
    size_t threads) const {
    const auto & window = get_cached_window_impl(config);
    AudioTensor magnitude;
    const double stft_ms = measure_ms([&]() {
        magnitude = STFT().compute_magnitude(waveform, window, batch, samples, config, threads);
    });
    if (debug::timing_log_enabled()) {
        debug::timing_log_scalar("audio.stft_ms", stft_ms);
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    const auto filterbank = MelFilterbank().build(
        MelFilterbankConfig{sample_rate, config.n_fft, n_mels, 0.0f, 0.0f, true});

    AudioTensor result;
    result.shape = {batch, n_mels, frames};
    result.values.assign(static_cast<size_t>(checked_product({batch, n_mels, frames})), 0.0f);

    const int64_t work_items = batch * frames;
    const unsigned thread_count = static_cast<unsigned>(std::max<size_t>(1, threads > 0 ? threads : std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    workers.reserve(thread_count > 0 ? thread_count - 1 : 0);

    auto worker = [&](unsigned ith) {
        for (int64_t work = static_cast<int64_t>(ith); work < work_items; work += static_cast<int64_t>(thread_count)) {
            const int64_t b = work / frames;
            const int64_t t = work % frames;
            for (int64_t m = 0; m < n_mels; ++m) {
                double sum = 0.0;
                int64_t f = 0;
                for (; f + 3 < freq_bins; f += 4) {
                    const size_t filter_base = static_cast<size_t>(m * freq_bins + f);
                    const size_t mag_base = static_cast<size_t>(((b * freq_bins) + f) * frames + t);
                    const float mag0 = magnitude.values[mag_base];
                    const float mag1 = magnitude.values[mag_base + static_cast<size_t>(frames)];
                    const float mag2 = magnitude.values[mag_base + static_cast<size_t>(2 * frames)];
                    const float mag3 = magnitude.values[mag_base + static_cast<size_t>(3 * frames)];
                    sum += static_cast<double>(mag0 * mag0) * static_cast<double>(filterbank.values[filter_base + 0]);
                    sum += static_cast<double>(mag1 * mag1) * static_cast<double>(filterbank.values[filter_base + 1]);
                    sum += static_cast<double>(mag2 * mag2) * static_cast<double>(filterbank.values[filter_base + 2]);
                    sum += static_cast<double>(mag3 * mag3) * static_cast<double>(filterbank.values[filter_base + 3]);
                }
                for (; f < freq_bins; ++f) {
                    const float mag = magnitude.values[static_cast<size_t>(((b * freq_bins) + f) * frames + t)];
                    sum += static_cast<double>(mag * mag) *
                           static_cast<double>(filterbank.values[static_cast<size_t>(m * freq_bins + f)]);
                }
                // Caution: this log guard is Based on NeMo's default log-mel path:
                // log(x + 2^-24). This is correct for Sortformer parity, but other
                // frontends may require different log bases or guard semantics.
                constexpr float kLogZeroGuard = 5.960464477539063e-8f;  // 2^-24
                result.values[static_cast<size_t>(((b * n_mels) + m) * frames + t)] =
                    std::log(static_cast<float>(sum) + kLogZeroGuard);
            }
        }
    };

    for (unsigned ith = 1; ith < thread_count; ++ith) {
        workers.emplace_back(worker, ith);
    }
    worker(0);
    for (auto & worker_thread : workers) {
        worker_thread.join();
    }

    return result;
}

AudioTensor LogMelSpectrogram::compute(
    const std::vector<float> & waveform,
    const std::vector<float> & window,
    int64_t batch,
    int64_t samples,
    const STFTConfig & config,
    const AudioTensor & filterbank,
    size_t threads) const {
    AudioTensor magnitude;
    const double stft_ms = measure_ms([&]() {
        magnitude = STFT().compute_magnitude(waveform, window, batch, samples, config, threads);
    });
    if (debug::timing_log_enabled()) {
        debug::timing_log_scalar("audio.stft_ms", stft_ms);
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    std::vector<float> power_spec(magnitude.values.size(), 0.0f);
    for (size_t i = 0; i < magnitude.values.size(); ++i) {
        power_spec[i] = magnitude.values[i] * magnitude.values[i];
    }

    auto result = MelFilterbank().compute_custom(power_spec, batch, freq_bins, frames, filterbank);
    constexpr float kLogZeroGuard = 5.960464477539063e-8f;  // 2^-24
    for (float & value : result.values) {
        value = std::log(value + kLogZeroGuard);
    }
    return result;
}

FeatureNormalizeOutput FeatureNormalizer::compute(
    const std::vector<float> & features,
    const std::vector<int64_t> & seq_len,
    int64_t batch,
    int64_t feature_dim,
    int64_t frames,
    FeatureNormalizeType normalize_type) const {
    return normalize_batch_impl(features, seq_len, batch, feature_dim, frames, normalize_type);
}

const std::vector<float> & get_cached_stft_window(const STFTConfig & config) {
    return get_cached_window_impl(config);
}

}  // namespace engine::audio

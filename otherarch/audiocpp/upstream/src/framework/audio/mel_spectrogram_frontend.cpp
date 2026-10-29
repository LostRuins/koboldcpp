#include "engine/framework/audio/mel_spectrogram_frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace engine::audio {
namespace {

template <MelFilterbankProjection Projection, bool PowerDuringProjection>
void project_mel_filterbank(
    const MelSpectrogramFrontendConfig & config,
    const AudioTensor & magnitude,
    const AudioTensor & filterbank,
    const SparseMelFilterbank & sparse_filterbank,
    int64_t channels,
    int64_t freq_bins,
    int64_t stft_frames,
    int64_t frames,
    MelSpectrogramFeatures & out) {
    using Accumulator = std::conditional_t<
        Projection == MelFilterbankProjection::DenseLongDouble,
        long double,
        std::conditional_t<Projection == MelFilterbankProjection::DenseF64, double, float>>;
    const int64_t projection_work_items = channels * config.n_mels * frames;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) if(projection_work_items >= 4096)
#endif
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t mel = 0; mel < config.n_mels; ++mel) {
            for (int64_t frame = 0; frame < frames; ++frame) {
                Accumulator sum = 0;
                const int64_t start = Projection == MelFilterbankProjection::SparseF32
                    ? sparse_filterbank.starts[static_cast<size_t>(mel)] : 0;
                const int64_t end = Projection == MelFilterbankProjection::SparseF32
                    ? sparse_filterbank.ends[static_cast<size_t>(mel)] : freq_bins;
                for (int64_t freq = start; freq < end; ++freq) {
                    const float value = magnitude.values[static_cast<size_t>(
                        ((channel * freq_bins + freq) * stft_frames) + frame)];
                    const float weight = filterbank.values[
                        static_cast<size_t>(mel * freq_bins + freq)];
                    if constexpr (Projection == MelFilterbankProjection::DenseLongDouble) {
                        sum += static_cast<long double>(weight) *
                               static_cast<long double>(value);
                    } else if constexpr (Projection == MelFilterbankProjection::DenseF64) {
                        sum += static_cast<double>(weight) * static_cast<double>(value);
                    } else if constexpr (PowerDuringProjection) {
                        sum += weight * value * value;
                    } else {
                        sum += weight * value;
                    }
                }
                const size_t output_index = config.layout == MelOutputLayout::FeatureMajor
                    ? static_cast<size_t>(((channel * config.n_mels + mel) * out.frames) + frame)
                    : static_cast<size_t>(((channel * out.frames + frame) * config.n_mels) + mel);
                if constexpr (Projection == MelFilterbankProjection::DenseF64) {
                    out.values[output_index] = config.value_transform == MelValueTransform::None
                        ? static_cast<float>(sum)
                        : config.value_transform == MelValueTransform::Log10
                            ? static_cast<float>(std::log10(std::max(sum, config.log_floor)))
                            : static_cast<float>(std::log(std::max(sum, config.log_floor)));
                } else {
                    const float float_sum = static_cast<float>(sum);
                    out.values[output_index] = config.value_transform == MelValueTransform::None
                        ? float_sum
                        : config.value_transform == MelValueTransform::Log10
                            ? config.log_precision == MelLogPrecision::F64
                                ? static_cast<float>(std::log10(std::max(
                                      static_cast<double>(float_sum), config.log_floor)))
                                : std::log10(std::max(
                                      float_sum, static_cast<float>(config.log_floor)))
                            : config.log_precision == MelLogPrecision::F64
                                ? static_cast<float>(std::log(std::max(
                                      static_cast<double>(float_sum), config.log_floor)))
                                : std::log(std::max(
                                      float_sum, static_cast<float>(config.log_floor)));
                }
            }
        }
    }
}

}  // namespace

bool MelSpectrogramFrontendConfig::operator==(const MelSpectrogramFrontendConfig & other) const noexcept {
    return sample_rate == other.sample_rate && n_fft == other.n_fft &&
           hop_length == other.hop_length && win_length == other.win_length &&
           n_mels == other.n_mels && mel_fmin == other.mel_fmin &&
           mel_fmax == other.mel_fmax && filterbank_normalization == other.filterbank_normalization &&
           stft_pad_mode == other.stft_pad_mode && window == other.window &&
           stft_center == other.stft_center && waveform_padding == other.waveform_padding &&
           filterbank_projection == other.filterbank_projection && spectrum_mode == other.spectrum_mode &&
           magnitude_epsilon == other.magnitude_epsilon &&
           log_floor == other.log_floor && log_precision == other.log_precision &&
           value_transform == other.value_transform &&
           layout == other.layout && resample_mode == other.resample_mode && require_mono == other.require_mono &&
           minimum_samples == other.minimum_samples && drop_last_frames == other.drop_last_frames &&
           max_frames == other.max_frames && frame_multiple == other.frame_multiple &&
           frame_pad_value == other.frame_pad_value && log_dynamic_range == other.log_dynamic_range &&
           log_shift == other.log_shift && log_divisor == other.log_divisor;
}

MelSpectrogramFrontend::MelSpectrogramFrontend(MelSpectrogramFrontendConfig config)
    : config_(std::move(config)) {
    if (config_.sample_rate <= 0 || config_.n_fft <= 0 || config_.hop_length <= 0 ||
        config_.win_length <= 0 || config_.n_mels <= 0 || config_.win_length > config_.n_fft ||
        config_.mel_fmin < 0.0f ||
        (config_.mel_fmax != 0.0f && config_.mel_fmax <= config_.mel_fmin) ||
        config_.magnitude_epsilon < 0.0f || config_.log_floor <= 0.0 ||
        config_.minimum_samples < 0 || config_.drop_last_frames < 0 ||
        config_.max_frames < 0 || config_.frame_multiple <= 0 ||
        config_.log_dynamic_range < 0.0f || config_.log_divisor <= 0.0f) {
        throw std::runtime_error("MelSpectrogramFrontend invalid config");
    }
    if (config_.spectrum_mode == MelSpectrumMode::PowerDuringProjection &&
        config_.filterbank_projection != MelFilterbankProjection::DenseF32 &&
        config_.filterbank_projection != MelFilterbankProjection::SparseF32) {
        throw std::runtime_error("MelSpectrogramFrontend unsupported power accumulation");
    }
    filterbank_ = MelFilterbank().build({config_.sample_rate, config_.n_fft, config_.n_mels,
                                        config_.mel_fmin, config_.mel_fmax,
                                        config_.filterbank_normalization == MelFilterbankNormalization::Slaney});
    if (config_.filterbank_projection == MelFilterbankProjection::SparseF32) {
        sparse_filterbank_ = MelFilterbank().prepare_sparse(filterbank_);
    }
}

MelSpectrogramFeatures MelSpectrogramFrontend::extract_audio(
    const std::vector<float> & interleaved, int sample_rate, int channels, size_t threads) const {
    if (sample_rate <= 0 || channels <= 0 || interleaved.empty() ||
        interleaved.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("MelSpectrogramFrontend invalid audio input");
    }
    if (config_.require_mono && channels != 1) {
        throw std::runtime_error("MelSpectrogramFrontend expects mono audio");
    }
    if (config_.resample_mode == MelResampleMode::RequireMatch && sample_rate != config_.sample_rate) {
        throw std::runtime_error("MelSpectrogramFrontend unexpected sample rate");
    }
    std::vector<float> mono;
    if (config_.resample_mode == MelResampleMode::Linear) {
        mono = convert_interleaved_audio_to_mono_linear_resampled(
            interleaved, sample_rate, channels, static_cast<int>(config_.sample_rate));
    } else {
        mono = mixdown_interleaved_to_mono_average(interleaved, channels);
        if (config_.resample_mode == MelResampleMode::TorchaudioSincF64 && sample_rate != config_.sample_rate) {
            TorchaudioSincHannResampleOptions options;
            options.kernel_mode = TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat64;
            mono = resample_mono_torchaudio_sinc_hann(
                mono, sample_rate, static_cast<int>(config_.sample_rate), options);
        } else if (config_.resample_mode == MelResampleMode::SoxrQualityActualLength &&
                   sample_rate != config_.sample_rate) {
            SoxrResampleOptions options;
            options.profile = SoxrResampleProfile::QualityOnly;
            options.output_length_policy = SoxrOutputLengthPolicy::ActualOutput;
            options.output_padding = 256;
            options.reject_empty_output = true;
            options.warning_context = "mel spectrogram frontend";
            options.fallback_description = "linear resampling";
            mono = resample_mono_soxr_or_linear(
                mono, sample_rate, static_cast<int>(config_.sample_rate), options);
        }
    }
    return extract_mono(mono, threads);
}

MelSpectrogramFeatures MelSpectrogramFrontend::extract_mono(
    const std::vector<float> & mono, size_t threads) const {
    return extract_planar(mono, 1, threads);
}

MelSpectrogramFeatures MelSpectrogramFrontend::extract_planar(
    const std::vector<float> & planar, int64_t channels, size_t threads) const {
    if (channels <= 0 || planar.empty() || planar.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("MelSpectrogramFrontend invalid planar waveform");
    }
    const int64_t input_samples = static_cast<int64_t>(planar.size()) / channels;
    const int64_t samples = std::max(input_samples, config_.minimum_samples);
    const int64_t pad = config_.waveform_padding != MelWaveformPadding::None
        ? (config_.n_fft - config_.hop_length) / 2 : 0;
    const int64_t padded_samples = samples + 2 * pad;
    std::vector<float> padded(static_cast<size_t>(channels * padded_samples), 0.0f);
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::vector<float> input(static_cast<size_t>(samples), 0.0f);
        std::copy_n(planar.begin() + channel * input_samples, input_samples, input.begin());
        std::vector<float> channel_padded;
        if (pad > 0 && input.size() == 1 &&
            config_.waveform_padding == MelWaveformPadding::ReflectOrRepeatSingleton) {
            channel_padded.assign(static_cast<size_t>(padded_samples), input.front());
        } else {
            channel_padded = pad > 0 ? reflect_pad_samples(input, pad, pad) : std::move(input);
        }
        std::copy(channel_padded.begin(), channel_padded.end(),
                  padded.begin() + channel * padded_samples);
    }
    const STFTFamily window_family = config_.window == MelHannWindow::Periodic
        ? STFTFamily::Kokoro : STFTFamily::Default;
    const STFTConfig stft_config{
        config_.n_fft, config_.hop_length, config_.win_length,
        config_.stft_center, config_.stft_pad_mode, window_family};
    const auto & window = get_cached_stft_window(stft_config);
    auto magnitude = STFT().compute_magnitude(
        padded, window, channels, padded_samples, stft_config, threads);
    if (magnitude.shape.size() != 3 || magnitude.shape[0] != channels ||
        magnitude.shape[1] != config_.n_fft / 2 + 1 || magnitude.shape[2] <= 0) {
        throw std::runtime_error("MelSpectrogramFrontend STFT shape mismatch");
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (config_.drop_last_frames >= stft_frames) {
        throw std::runtime_error("MelSpectrogramFrontend has no frames after trim");
    }
    const int64_t frames = config_.max_frames > 0
        ? std::min(stft_frames - config_.drop_last_frames, config_.max_frames)
        : stft_frames - config_.drop_last_frames;
    MelSpectrogramFeatures out;
    out.channels = channels;
    out.frames = ((frames + config_.frame_multiple - 1) / config_.frame_multiple) *
                 config_.frame_multiple;
    out.mel_bins = config_.n_mels;
    out.values.assign(static_cast<size_t>(channels * out.frames * config_.n_mels),
                      config_.frame_pad_value);

    if (config_.magnitude_epsilon > 0.0f ||
        config_.spectrum_mode == MelSpectrumMode::PowerBeforeProjection) {
#ifdef _OPENMP
#pragma omp parallel for if(magnitude.values.size() >= 4096)
#endif
        for (int64_t index = 0;
             index < static_cast<int64_t>(magnitude.values.size()); ++index) {
            float value = magnitude.values[static_cast<size_t>(index)];
            if (config_.magnitude_epsilon > 0.0f) {
                value = std::sqrt(value * value + config_.magnitude_epsilon);
            }
            if (config_.spectrum_mode == MelSpectrumMode::PowerBeforeProjection) {
                value *= value;
            }
            magnitude.values[static_cast<size_t>(index)] = value;
        }
    }

    const auto run_projection = [&](auto projection_tag) {
        constexpr MelFilterbankProjection projection = decltype(projection_tag)::value;
        if (config_.spectrum_mode == MelSpectrumMode::PowerDuringProjection) {
            project_mel_filterbank<projection, true>(
                config_, magnitude, filterbank_, sparse_filterbank_,
                channels, freq_bins, stft_frames, frames, out);
        } else {
            project_mel_filterbank<projection, false>(
                config_, magnitude, filterbank_, sparse_filterbank_,
                channels, freq_bins, stft_frames, frames, out);
        }
    };
    switch (config_.filterbank_projection) {
        case MelFilterbankProjection::DenseF32:
            run_projection(std::integral_constant<MelFilterbankProjection, MelFilterbankProjection::DenseF32>{});
            break;
        case MelFilterbankProjection::SparseF32:
            run_projection(std::integral_constant<MelFilterbankProjection, MelFilterbankProjection::SparseF32>{});
            break;
        case MelFilterbankProjection::DenseF64:
            run_projection(std::integral_constant<MelFilterbankProjection, MelFilterbankProjection::DenseF64>{});
            break;
        case MelFilterbankProjection::DenseLongDouble:
            run_projection(std::integral_constant<MelFilterbankProjection, MelFilterbankProjection::DenseLongDouble>{});
            break;
    }
    if (config_.log_dynamic_range > 0.0f) {
        const float peak = *std::max_element(out.values.begin(), out.values.end());
        const float floor = peak - config_.log_dynamic_range;
        for (float & value : out.values) {
            value = (std::max(value, floor) + config_.log_shift) / config_.log_divisor;
        }
    }
    return out;
}

std::shared_ptr<const MelSpectrogramFrontend> get_cached_mel_spectrogram_frontend(
    const MelSpectrogramFrontendConfig & config) {
    static std::mutex mutex;
    static std::vector<std::pair<MelSpectrogramFrontendConfig,
                                 std::shared_ptr<const MelSpectrogramFrontend>>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & entry : cache) {
        if (entry.first == config) return entry.second;
    }
    auto frontend = std::make_shared<MelSpectrogramFrontend>(config);
    cache.emplace_back(config, frontend);
    return frontend;
}

}  // namespace engine::audio

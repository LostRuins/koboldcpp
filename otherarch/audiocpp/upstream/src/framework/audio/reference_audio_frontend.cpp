#include "engine/framework/audio/reference_audio_frontend.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::audio {
namespace {

struct RealDftTables {
    std::vector<double> cos;
    std::vector<double> sin;
};

const RealDftTables & cached_real_dft_tables_512() {
    static const RealDftTables tables = [] {
        constexpr int64_t kFft = 512;
        constexpr int64_t kFreqBins = kFft / 2 + 1;
        constexpr double kPi = 3.14159265358979323846264338327950288;
        RealDftTables out;
        out.cos.resize(static_cast<size_t>(kFreqBins * kFft));
        out.sin.resize(static_cast<size_t>(kFreqBins * kFft));
        for (int64_t freq = 0; freq < kFreqBins; ++freq) {
            for (int64_t n = 0; n < kFft; ++n) {
                const double angle = 2.0 * kPi * static_cast<double>(freq * n) / static_cast<double>(kFft);
                out.cos[static_cast<size_t>(freq * kFft + n)] = std::cos(angle);
                out.sin[static_cast<size_t>(freq * kFft + n)] = std::sin(angle);
            }
        }
        return out;
    }();
    return tables;
}

std::vector<float> require_mono_samples(const std::vector<float> & samples, int channels) {
    if (channels <= 0) {
        throw std::runtime_error("reference audio channel count must be positive");
    }
    if (samples.empty()) {
        throw std::runtime_error("reference audio must not be empty");
    }
    if (samples.size() % static_cast<size_t>(channels) != 0) {
        throw std::runtime_error("reference audio sample count must be divisible by channel count");
    }
    if (channels == 1) {
        return samples;
    }
    return engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
}

std::vector<float> resample_mono(const std::vector<float> & input, int input_sample_rate, int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("reference audio resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float32ComputationStoredAsFloat32;
    options.accumulation = engine::audio::TorchaudioSincHannAccumulation::Float32;
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate, options);
}

std::vector<float> resample_mono_librosa(const std::vector<float> & input, int input_sample_rate, int output_sample_rate) {
    if (input_sample_rate <= 0 || output_sample_rate <= 0) {
        throw std::runtime_error("reference audio resampling requires positive sample rates");
    }
    if (input_sample_rate == output_sample_rate || input.empty()) {
        return input;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.require_full_input = true;
    if (auto output = engine::audio::try_resample_mono_soxr(input, input_sample_rate, output_sample_rate, options)) {
        return *output;
    }
    return engine::audio::resample_mono_torchaudio_sinc_hann(input, input_sample_rate, output_sample_rate);
}

}  // namespace

ReferenceAudioFbankFeatures ReferenceAudioFrontend::campplus_fbank_16k(const std::vector<float> & waveform_16k) {
    constexpr int64_t kWindowSize = 400;
    constexpr int64_t kWindowShift = 160;
    constexpr int64_t kPaddedWindowSize = 512;
    constexpr int64_t kNumMels = 80;
    constexpr float kPreemphasis = 0.97F;
    constexpr float kEpsilon = std::numeric_limits<float>::epsilon();

    if (static_cast<int64_t>(waveform_16k.size()) < kWindowSize) {
        throw std::runtime_error("reference CAMPPlus fbank requires at least one 25 ms frame");
    }

    const int64_t frames = 1 + (static_cast<int64_t>(waveform_16k.size()) - kWindowSize) / kWindowShift;
    const auto & window = cached_kaldi_povey_window(kWindowSize);
    const auto & mel_filterbank = cached_kaldi_campplus_mel_filterbank_16k();

    std::vector<float> frame(static_cast<size_t>(kWindowSize), 0.0F);
    std::vector<float> stft_batch(static_cast<size_t>(frames * kPaddedWindowSize), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * kWindowShift;
        float mean = 0.0F;
        for (int64_t i = 0; i < kWindowSize; ++i) {
            const float sample = waveform_16k[static_cast<size_t>(start + i)];
            frame[static_cast<size_t>(i)] = sample;
            mean += sample;
        }
        mean /= static_cast<float>(kWindowSize);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            frame[static_cast<size_t>(i)] -= mean;
        }
        for (int64_t i = kWindowSize - 1; i > 0; --i) {
            frame[static_cast<size_t>(i)] -= kPreemphasis * frame[static_cast<size_t>(i - 1)];
        }
        frame[0] -= kPreemphasis * frame[0];
        for (int64_t i = 0; i < kWindowSize; ++i) {
            stft_batch[static_cast<size_t>(frame_index * kPaddedWindowSize + i)] =
                frame[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }
    }

    std::vector<float> stft_window(static_cast<size_t>(kPaddedWindowSize), 1.0F);
    const engine::audio::STFTConfig stft_config{
        kPaddedWindowSize,
        kPaddedWindowSize,
        kPaddedWindowSize,
        false,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        stft_batch,
        stft_window,
        frames,
        kPaddedWindowSize,
        stft_config);

    const int64_t freq_bins = (kPaddedWindowSize / 2) + 1;
    ReferenceAudioFbankFeatures output;
    output.frames = frames;
    output.feature_dim = kNumMels;
    output.values.assign(static_cast<size_t>(frames * kNumMels), 0.0F);
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
            float energy = 0.0F;
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(frame_index * freq_bins + freq)];
                energy += (mag * mag) * mel_filterbank[static_cast<size_t>(mel_bin * freq_bins + freq)];
            }
            output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] =
                std::log(std::max(energy, kEpsilon));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
        float mean = 0.0F;
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            mean += output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)];
        }
        mean /= static_cast<float>(frames);
        for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
            output.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] -= mean;
        }
    }
    return output;
}

ReferenceAudioSemanticFeatures ReferenceAudioFrontend::semantic_features_16k(const std::vector<float> & waveform_16k) {
    constexpr int64_t kWindowSize = 400;
    constexpr int64_t kWindowShift = 160;
    constexpr int64_t kFftSize = 512;
    constexpr int64_t kFreqBins = kFftSize / 2 + 1;
    constexpr int64_t kNumMels = 80;
    constexpr float kPreemphasis = 0.97F;
    constexpr double kInputScale = 32768.0;
    constexpr double kMelFloor = 1.192092955078125e-07;

    if (static_cast<int64_t>(waveform_16k.size()) < kWindowSize) {
        throw std::runtime_error("reference semantic fbank requires at least one 25 ms frame");
    }

    const int64_t frames = 1 + (static_cast<int64_t>(waveform_16k.size()) - kWindowSize) / kWindowShift;
    const auto & window = cached_kaldi_povey_window(kWindowSize);
    const auto & mel_filterbank = cached_kaldi_campplus_mel_filterbank_16k();
    const auto & dft = cached_real_dft_tables_512();

    ReferenceAudioFbankFeatures fbank;
    fbank.frames = frames;
    fbank.feature_dim = kNumMels;
    fbank.values.assign(static_cast<size_t>(frames * kNumMels), 0.0F);

#ifdef _OPENMP
#pragma omp parallel for if(frames >= 8)
#endif
    for (int64_t frame_index = 0; frame_index < frames; ++frame_index) {
        const int64_t start = frame_index * kWindowShift;
        double buffer[kFftSize] = {};
        double mean = 0.0;
        for (int64_t i = 0; i < kWindowSize; ++i) {
            const double sample = static_cast<double>(waveform_16k[static_cast<size_t>(start + i)]) * kInputScale;
            buffer[i] = sample;
            mean += sample;
        }
        mean /= static_cast<double>(kWindowSize);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            buffer[i] -= mean;
        }
        for (int64_t i = kWindowSize - 1; i > 0; --i) {
            buffer[i] -= kPreemphasis * buffer[i - 1];
        }
        buffer[0] *= (1.0 - kPreemphasis);
        for (int64_t i = 0; i < kWindowSize; ++i) {
            buffer[i] *= static_cast<double>(window[static_cast<size_t>(i)]);
        }

        double power[kFreqBins] = {};
        for (int64_t freq = 0; freq < kFreqBins; ++freq) {
            double re = 0.0;
            double im = 0.0;
            const size_t table_offset = static_cast<size_t>(freq * kFftSize);
            for (int64_t n = 0; n < kFftSize; ++n) {
                const double sample = buffer[n];
                re += sample * dft.cos[table_offset + static_cast<size_t>(n)];
                im -= sample * dft.sin[table_offset + static_cast<size_t>(n)];
            }
            power[freq] = re * re + im * im;
        }

        for (int64_t mel_bin = 0; mel_bin < kNumMels; ++mel_bin) {
            double energy = 0.0;
            for (int64_t freq = 0; freq < kFreqBins; ++freq) {
                energy += static_cast<double>(mel_filterbank[static_cast<size_t>(mel_bin * kFreqBins + freq)]) *
                    power[freq];
            }
            fbank.values[static_cast<size_t>(frame_index * kNumMels + mel_bin)] =
                static_cast<float>(std::log(std::max(energy, kMelFloor)));
        }
    }

    for (int64_t mel_bin = 0; mel_bin < fbank.feature_dim; ++mel_bin) {
        double mean = 0.0;
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            mean += static_cast<double>(fbank.values[static_cast<size_t>(frame * fbank.feature_dim + mel_bin)]);
        }
        mean /= static_cast<double>(fbank.frames);

        double variance = 0.0;
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            const double diff =
                static_cast<double>(fbank.values[static_cast<size_t>(frame * fbank.feature_dim + mel_bin)]) - mean;
            variance += diff * diff;
        }
        variance = fbank.frames > 1 ? variance / static_cast<double>(fbank.frames - 1) : 0.0;
        const float scale = static_cast<float>(1.0 / std::sqrt(variance + 1.0e-7));
        for (int64_t frame = 0; frame < fbank.frames; ++frame) {
            float & value = fbank.values[static_cast<size_t>(frame * fbank.feature_dim + mel_bin)];
            value = (value - static_cast<float>(mean)) * scale;
        }
    }

    const int64_t padded_frames = fbank.frames + (fbank.frames % 2);
    std::vector<float> padded(static_cast<size_t>(padded_frames * fbank.feature_dim), 1.0F);
    std::copy(fbank.values.begin(), fbank.values.end(), padded.begin());

    ReferenceAudioSemanticFeatures output;
    output.frames = padded_frames / 2;
    output.feature_dim = fbank.feature_dim * 2;
    output.values.assign(static_cast<size_t>(output.frames * output.feature_dim), 0.0F);
    output.attention_mask.assign(static_cast<size_t>(output.frames), 0);
    for (int64_t pair = 0; pair < output.frames; ++pair) {
        const int64_t first_frame = pair * 2;
        const int64_t second_frame = first_frame + 1;
        std::copy_n(
            padded.data() + static_cast<size_t>(first_frame * fbank.feature_dim),
            static_cast<size_t>(fbank.feature_dim),
            output.values.data() + static_cast<size_t>(pair * output.feature_dim));
        std::copy_n(
            padded.data() + static_cast<size_t>(second_frame * fbank.feature_dim),
            static_cast<size_t>(fbank.feature_dim),
            output.values.data() + static_cast<size_t>(pair * output.feature_dim + fbank.feature_dim));
        output.attention_mask[static_cast<size_t>(pair)] = second_frame < fbank.frames ? 1 : 0;
    }
    return output;
}

ReferenceAudioFrontend::ReferenceAudioFrontend(ReferenceAudioFrontendConfig config)
    : config_(std::move(config)) {
    if (config_.speaker_sample_rate <= 0 || config_.max_duration_seconds < 0) {
        throw std::runtime_error("ReferenceAudioFrontend invalid config");
    }
}

ReferenceAudioFeatures ReferenceAudioFrontend::extract(
    const std::vector<float> & samples, int sample_rate, int channels, size_t threads) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("ReferenceAudioFrontend invalid sample rate");
    }
    auto mono = require_mono_samples(samples, channels);
    if (config_.max_duration_seconds > 0) {
        truncate_samples_to_count(
            mono,
            static_cast<size_t>(static_cast<int64_t>(sample_rate) * config_.max_duration_seconds));
    }
    const auto resample = [](const std::vector<float> & input, int input_rate, int output_rate,
                             ReferenceAudioResampleMode mode) {
        return mode == ReferenceAudioResampleMode::SoxrF32ExactLength
            ? resample_mono_librosa(input, input_rate, output_rate)
            : resample_mono(input, input_rate, output_rate);
    };
    ReferenceAudioFeatures out;
    out.mel_waveform = resample(
        mono, sample_rate, static_cast<int>(config_.mel.sample_rate), config_.mel_resample);
    const auto & speaker_source = config_.speaker_source == SpeakerWaveformSource::MelWaveform
        ? out.mel_waveform : mono;
    const int speaker_source_rate = config_.speaker_source == SpeakerWaveformSource::MelWaveform
        ? static_cast<int>(config_.mel.sample_rate) : sample_rate;
    out.speaker_waveform = resample(
        speaker_source, speaker_source_rate, config_.speaker_sample_rate, config_.speaker_resample);
    out.mel = get_cached_mel_spectrogram_frontend(config_.mel)->extract_mono(out.mel_waveform, threads);
    out.campplus = campplus_fbank_16k(out.speaker_waveform);
    out.semantic = semantic_features_16k(out.speaker_waveform);
    return out;
}

}  // namespace engine::audio

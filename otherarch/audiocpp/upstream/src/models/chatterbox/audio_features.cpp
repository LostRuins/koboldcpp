#include "components/component_weights.h"
#include "components/audio_processing.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/audio/resampling.h"

#include <stdexcept>
#include <utility>

namespace engine::models::chatterbox::components {
std::vector<float> resample_component_torchaudio_hann_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::TorchaudioSincHannResampleOptions options;
    options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::Float64ComputationStoredAsFloat64;
    return engine::audio::resample_mono_torchaudio_sinc_hann(
        input,
        input_sample_rate,
        output_sample_rate,
        options);
}

S3TokenizerLogMelOutputs compute_s3tokenizer_log_mel(const runtime::AudioBuffer & audio) {
    engine::audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.win_length = 400;
    config.n_mels = 128;
    config.mel_fmax = 8000.0f;
    config.window = engine::audio::MelHannWindow::Periodic;
    config.stft_center = true;
    config.waveform_padding = engine::audio::MelWaveformPadding::None;
    config.spectrum_mode = engine::audio::MelSpectrumMode::PowerBeforeProjection;
    config.value_transform = engine::audio::MelValueTransform::Log10;
    config.log_floor = 1.0e-10;
    config.drop_last_frames = 1;
    config.log_dynamic_range = 8.0f;
    config.log_shift = 4.0f;
    config.log_divisor = 4.0f;
    config.resample_mode = engine::audio::MelResampleMode::SoxrQualityActualLength;
    config.require_mono = true;
    auto features = engine::audio::get_cached_mel_spectrogram_frontend(config)
        ->extract_audio(audio.samples, audio.sample_rate, audio.channels);
    return {std::move(features.values), features.mel_bins, features.frames};
}

CampplusFbankOutputs compute_campplus_fbank(const runtime::AudioBuffer & audio) {
    if (audio.channels != 1) {
        throw std::runtime_error("S3 speaker fbank expects mono audio");
    }

    std::vector<float> mono = audio.sample_rate == 16000
        ? audio.samples
        : resample_component_torchaudio_hann_mono(audio.samples, audio.sample_rate, 16000);

    if (mono.size() < 400) {
        return {};
    }
    auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(mono);
    return {std::move(features.values), features.frames, features.feature_dim};
}


}  // namespace engine::models::chatterbox::components

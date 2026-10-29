#include "components/audio_processing.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"

#include "components/component_weights.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace engine::models::chatterbox::components {

std::vector<float> resample_component_mono(
    const std::vector<float> & input,
    int input_sample_rate,
    int output_sample_rate) {
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ActualOutput;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = "Chatterbox S3 prompt mel";
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(input, input_sample_rate, output_sample_rate, options);
}

S3PromptMelOutputs compute_s3_prompt_mel(const runtime::AudioBuffer & audio) {
    engine::audio::MelSpectrogramFrontendConfig config;
    config.sample_rate = 24000;
    config.n_fft = 1920;
    config.hop_length = 480;
    config.win_length = 1920;
    config.n_mels = 80;
    config.mel_fmax = 8000.0f;
    config.stft_pad_mode = engine::audio::STFTPadMode::Constant;
    config.filterbank_projection = engine::audio::MelFilterbankProjection::DenseF64;
    config.magnitude_epsilon = 1.0e-9f;
    config.resample_mode = engine::audio::MelResampleMode::SoxrQualityActualLength;
    config.require_mono = true;
    auto features = engine::audio::get_cached_mel_spectrogram_frontend(config)
        ->extract_audio(audio.samples, audio.sample_rate, audio.channels);
    return {std::move(features.values), features.mel_bins, features.frames};
}

}  // namespace engine::models::chatterbox::components

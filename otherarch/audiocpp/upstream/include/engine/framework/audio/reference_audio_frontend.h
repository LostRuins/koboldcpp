#pragma once

#include "engine/framework/audio/mel_spectrogram_frontend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::audio {

enum class ReferenceAudioResampleMode { TorchaudioF32, SoxrF32ExactLength };
enum class SpeakerWaveformSource { OriginalInput, MelWaveform };

struct ReferenceAudioFrontendConfig {
    MelSpectrogramFrontendConfig mel;
    int speaker_sample_rate = 16000;
    int64_t max_duration_seconds = 0;
    ReferenceAudioResampleMode mel_resample = ReferenceAudioResampleMode::TorchaudioF32;
    ReferenceAudioResampleMode speaker_resample = ReferenceAudioResampleMode::TorchaudioF32;
    SpeakerWaveformSource speaker_source = SpeakerWaveformSource::OriginalInput;
};

struct ReferenceAudioFbankFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t feature_dim = 0;
};

struct ReferenceAudioSemanticFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t frames = 0;
    int64_t feature_dim = 0;
};

struct ReferenceAudioFeatures {
    std::vector<float> mel_waveform;
    std::vector<float> speaker_waveform;
    MelSpectrogramFeatures mel;
    ReferenceAudioFbankFeatures campplus;
    ReferenceAudioSemanticFeatures semantic;
};

class ReferenceAudioFrontend {
public:
    explicit ReferenceAudioFrontend(ReferenceAudioFrontendConfig config);

    ReferenceAudioFeatures extract(const std::vector<float> & samples,
                                   int sample_rate, int channels, size_t threads) const;

    static ReferenceAudioFbankFeatures campplus_fbank_16k(const std::vector<float> & waveform);
    static ReferenceAudioSemanticFeatures semantic_features_16k(const std::vector<float> & waveform);

private:
    ReferenceAudioFrontendConfig config_;
};

}  // namespace engine::audio

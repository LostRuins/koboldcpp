#pragma once

#include "engine/framework/audio/wav_reader.h"

#include <filesystem>
#include <optional>
#include <vector>

namespace engine::audio {

struct AudioMixInput {
    WavData audio;
    float gain = 1.0F;
    std::optional<WavData> activity_reference = std::nullopt;
    float activity_threshold_dbfs = -40.0F;
    double activity_window_seconds = 0.1;
    double activity_margin_seconds = 0.35;
    double activity_fade_seconds = 0.03;
};

WavData mix_audio_to_reference_shape(
    const std::vector<AudioMixInput> & inputs,
    bool normalize_peak);

WavData read_and_mix_wavs_to_reference_shape(
    const std::vector<std::pair<std::filesystem::path, float>> & inputs,
    bool normalize_peak);

}  // namespace engine::audio

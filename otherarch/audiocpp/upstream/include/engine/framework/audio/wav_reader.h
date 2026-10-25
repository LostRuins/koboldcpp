#pragma once

#include <filesystem>
#include <istream>
#include <string_view>
#include <vector>

namespace engine::audio {

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

WavData read_wav_f32(std::istream & input);
WavData read_wav_f32(std::string_view input);
WavData read_wav_f32(const std::filesystem::path & path);

}  // namespace engine::audio

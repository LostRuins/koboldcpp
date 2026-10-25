#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::codecs {

struct LatentChunk {
    int frame_count = 0;
    int channel_count = 0;
    std::vector<float> values;
};

struct AudioChunk {
    int sample_rate = 0;
    int channel_count = 0;
    std::vector<float> samples;
};

class IAudioCodec {
public:
    virtual ~IAudioCodec() = default;

    virtual std::string family() const = 0;
    virtual AudioChunk decode(const LatentChunk & latents) = 0;
};

}  // namespace engine::codecs

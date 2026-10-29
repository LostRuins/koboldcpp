#pragma once

#include "engine/framework/audio/wav_writer.h"

#include <filesystem>
#include <string>
#include <vector>

namespace engine::audio {

struct AudioBuffer {
    int sample_rate = 0;
    int channel_count = 1;
    std::vector<float> samples;
};

class IAudioSink {
public:
    virtual ~IAudioSink() = default;
    virtual std::string family() const = 0;
    virtual void write(const std::filesystem::path & path, const AudioBuffer & audio) const = 0;
};

// 16-bit PCM, hard clip, no dither. Kept as its own type because it is the
// default sink for every existing route.
class WavPcm16Sink final : public IAudioSink {
public:
    std::string family() const override;
    void write(const std::filesystem::path & path, const AudioBuffer & audio) const override;
};

// Bit-depth-selectable sink. Default-constructed it is byte-for-byte the same
// output as WavPcm16Sink; construct it with WavWriteOptions to emit 24-bit PCM
// or float32, or to enable dither or the soft peak limiter.
class WavSink final : public IAudioSink {
public:
    WavSink() = default;
    explicit WavSink(const WavWriteOptions & options)
        : options_(options) {}

    const WavWriteOptions & options() const noexcept {
        return options_;
    }

    std::string family() const override;
    void write(const std::filesystem::path & path, const AudioBuffer & audio) const override;

private:
    WavWriteOptions options_{};
};

}  // namespace engine::audio

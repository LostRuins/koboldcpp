#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::audio {

struct RnnoiseWeights;

struct RnnoiseConfig {
    int64_t feature_size = 65;
    int64_t conv1_channels = 128;
    int64_t conv2_channels = 384;
    int64_t gru_size = 384;
    int64_t gain_bands = 32;
};

struct RnnoiseFrameOutput {
    std::vector<float> gains;
    float vad = 0.0f;
};

struct RnnoiseSequenceOutput {
    std::vector<float> gains;
    std::vector<float> vad;
    int64_t frames = 0;
    int64_t gain_bands = 0;
};

struct RnnoiseWaveformOutput {
    int sample_rate = 48000;
    std::vector<float> samples;
    std::vector<float> vad;
};

class RnnoiseModel {
public:
    static RnnoiseModel load_from_safetensors(const std::filesystem::path & checkpoint_path);
    static RnnoiseModel load_from_safetensors(const std::filesystem::path & checkpoint_path, const core::BackendConfig & backend_config);

    RnnoiseModel();
    ~RnnoiseModel();
    RnnoiseModel(RnnoiseModel &&) noexcept;
    RnnoiseModel & operator=(RnnoiseModel &&) noexcept;
    RnnoiseModel(const RnnoiseModel &) = delete;
    RnnoiseModel & operator=(const RnnoiseModel &) = delete;

    const RnnoiseConfig & config() const noexcept;
    const std::filesystem::path & source_path() const noexcept;

    RnnoiseSequenceOutput infer_features(
        const std::vector<float> & features,
        int64_t frames,
        int64_t feature_size) const;
    RnnoiseWaveformOutput process_mono_48k(const std::vector<float> & waveform) const;

    std::unique_ptr<class RnnoiseStreamingSession> create_streaming_session() const;

private:
    explicit RnnoiseModel(std::shared_ptr<const RnnoiseWeights> weights);

    std::shared_ptr<const RnnoiseWeights> weights_;
};

class RnnoiseStreamingSession {
public:
    explicit RnnoiseStreamingSession(std::shared_ptr<const RnnoiseWeights> weights);
    ~RnnoiseStreamingSession();
    RnnoiseStreamingSession(RnnoiseStreamingSession &&) noexcept;
    RnnoiseStreamingSession & operator=(RnnoiseStreamingSession &&) noexcept;
    RnnoiseStreamingSession(const RnnoiseStreamingSession &) = delete;
    RnnoiseStreamingSession & operator=(const RnnoiseStreamingSession &) = delete;

    void reset();
    RnnoiseFrameOutput process_frame(const float * features, int64_t feature_size);
    float process_audio_frame(const float * input, float * output, int64_t samples);

private:
    struct State;

    std::shared_ptr<const RnnoiseWeights> weights_;
    std::unique_ptr<State> state_;
};

}  // namespace engine::audio

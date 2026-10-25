#pragma once

#include "engine/framework/core/backend.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct GTCRNModelState;

struct GTCRNWaveformOutput {
    int sample_rate = 16000;
    std::vector<float> samples;
};

class GTCRNModel {
public:
    static GTCRNModel load_from_safetensors(const std::filesystem::path & checkpoint_path);
    static GTCRNModel load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        const core::BackendConfig & backend_config);

    GTCRNModel();
    ~GTCRNModel();
    GTCRNModel(GTCRNModel &&) noexcept;
    GTCRNModel & operator=(GTCRNModel &&) noexcept;
    GTCRNModel(const GTCRNModel &) = delete;
    GTCRNModel & operator=(const GTCRNModel &) = delete;

    GTCRNWaveformOutput denoise_mono_16k(const std::vector<float> & waveform) const;
    std::unique_ptr<class GTCRNStreamingSession> create_streaming_session() const;

private:
    explicit GTCRNModel(std::shared_ptr<const GTCRNModelState> state);

    std::shared_ptr<const GTCRNModelState> state_;
};

class GTCRNStreamingSession {
public:
    explicit GTCRNStreamingSession(std::shared_ptr<const GTCRNModelState> state);
    ~GTCRNStreamingSession();
    GTCRNStreamingSession(GTCRNStreamingSession &&) noexcept;
    GTCRNStreamingSession & operator=(GTCRNStreamingSession &&) noexcept;
    GTCRNStreamingSession(const GTCRNStreamingSession &) = delete;
    GTCRNStreamingSession & operator=(const GTCRNStreamingSession &) = delete;

    void reset();
    void process_stft_frame(const float * spec_frame_257x2, float * out_spec_frame_257x2);

private:
    struct State;

    std::shared_ptr<const GTCRNModelState> model_;
    std::unique_ptr<State> state_;
};

}  // namespace engine::audio

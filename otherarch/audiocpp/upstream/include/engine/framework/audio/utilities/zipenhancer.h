#pragma once

#include "engine/framework/core/backend.h"

#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct ZipEnhancerModelState;

struct ZipEnhancerWaveformOutput {
    int sample_rate = 16000;
    std::vector<float> samples;
};

class ZipEnhancerModel {
public:
    static ZipEnhancerModel load_from_directory(const std::filesystem::path & model_dir);
    static ZipEnhancerModel load_from_directory(const std::filesystem::path & model_dir, const core::BackendConfig & backend_config);

    ZipEnhancerModel();
    ~ZipEnhancerModel();
    ZipEnhancerModel(ZipEnhancerModel &&) noexcept;
    ZipEnhancerModel & operator=(ZipEnhancerModel &&) noexcept;
    ZipEnhancerModel(const ZipEnhancerModel &) = delete;
    ZipEnhancerModel & operator=(const ZipEnhancerModel &) = delete;

    ZipEnhancerWaveformOutput denoise_mono_16k(const std::vector<float> & waveform) const;

private:
    explicit ZipEnhancerModel(std::shared_ptr<const ZipEnhancerModelState> state);

    std::shared_ptr<const ZipEnhancerModelState> state_;
};

}  // namespace engine::audio

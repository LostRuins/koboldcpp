#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::audio {

struct DeepFilterNet2ModelState;

struct DeepFilterNet2Tensor {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

struct DeepFilterNet2Output {
    DeepFilterNet2Tensor erb_mask;
    DeepFilterNet2Tensor df_coefs;
    DeepFilterNet2Tensor enc_lsnr;
    DeepFilterNet2Tensor df_alpha;
};

struct DeepFilterNet2WaveformOutput {
    int sample_rate = 48000;
    std::vector<float> samples;
};

class DeepFilterNet2Model {
public:
    static DeepFilterNet2Model load_from_directory(const std::filesystem::path & model_dir);
    static DeepFilterNet2Model load_from_directory(const std::filesystem::path & model_dir, const core::BackendConfig & backend_config);

    DeepFilterNet2Model();
    ~DeepFilterNet2Model();
    DeepFilterNet2Model(DeepFilterNet2Model &&) noexcept;
    DeepFilterNet2Model & operator=(DeepFilterNet2Model &&) noexcept;
    DeepFilterNet2Model(const DeepFilterNet2Model &) = delete;
    DeepFilterNet2Model & operator=(const DeepFilterNet2Model &) = delete;

    DeepFilterNet2Output run_features(
        const std::vector<float> & feat_erb,
        const std::vector<int64_t> & feat_erb_shape,
        const std::vector<float> & feat_spec,
        const std::vector<int64_t> & feat_spec_shape) const;

    DeepFilterNet2WaveformOutput run_mono_48k(const std::vector<float> & waveform) const;

private:
    explicit DeepFilterNet2Model(std::shared_ptr<const DeepFilterNet2ModelState> state);

    std::shared_ptr<const DeepFilterNet2ModelState> state_;
};

}  // namespace engine::audio

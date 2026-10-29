#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

enum class HifiGanResBlockKind {
    PairedConv,
    SingleConv,
};

struct HifiGanSourceConditioningConfig {
    bool enabled = false;
    int64_t harmonic_num = 0;
    float amplitude = 0.1F;
    float noise_std = 0.003F;
    float voiced_threshold = 0.0F;
    std::string linear_prefix = "m_source.l_linear";
    std::string noise_conv_prefix = "noise_convs";
};

struct HifiGanGlobalConditioningConfig {
    int64_t channels = 0;
    std::string prefix = "cond";
    bool use_bias = true;
};

struct HifiGanVocoderConfig {
    int64_t sampling_rate = 0;
    int64_t num_mels = 0;
    int64_t upsample_initial_channel = 0;
    int64_t output_channels = 1;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> resblock_dilation_sizes;
    HifiGanResBlockKind resblock_kind = HifiGanResBlockKind::PairedConv;
    float leaky_relu_slope = 0.1F;
    float post_leaky_relu_slope = 0.1F;
    bool conv_post_use_bias = true;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    std::string tensor_prefix;
    HifiGanSourceConditioningConfig source;
    HifiGanGlobalConditioningConfig global_conditioning;
    bool release_source_storage_after_load = false;
    bool lower_padded_conv_transpose_as_crop = false;
};

struct HifiGanVocoderWeights {
    struct ResBlockWeights {
        std::vector<Conv1dWeights> convs1;
        std::vector<Conv1dWeights> convs2;
    };

    struct LinearWeights {
        core::TensorValue weight;
        std::optional<core::TensorValue> bias;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = true;
    };

    HifiGanVocoderConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dWeights conv_pre;
    std::vector<ConvTranspose1dWeights> ups;
    std::optional<Conv1dWeights> cond;
    std::vector<Conv1dWeights> noise_convs;
    std::vector<ResBlockWeights> resblocks;
    Conv1dWeights conv_post;
    LinearWeights source_linear;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

struct HifiGanVocoderOutput {
    std::vector<float> waveform;
    int64_t batch = 0;
    int64_t samples = 0;
    int64_t sample_rate = 0;
};

struct HifiGanVocoderRequest {
    const std::vector<float> * mel = nullptr;
    int64_t frames = 0;
    const std::vector<float> * f0 = nullptr;
    const std::vector<float> * conditioning = nullptr;
    int64_t conditioning_frames = 0;
    uint64_t seed = 1234;
    uint64_t prior_noise_values = 0;
};

class HifiGanVocoderComponent {
public:
    static HifiGanVocoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        HifiGanVocoderConfig config);

    HifiGanVocoderComponent() = default;
    HifiGanVocoderComponent(
        std::shared_ptr<const HifiGanVocoderWeights> weights,
        core::BackendConfig backend);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const HifiGanVocoderWeights> & weights() const noexcept;
    int64_t sample_rate() const noexcept;
    int64_t num_mels() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;

    HifiGanVocoderOutput synthesize(const HifiGanVocoderRequest & request) const;
    HifiGanVocoderOutput synthesize(const std::vector<float> & mel, int64_t frames) const;
    HifiGanVocoderOutput synthesize_with_f0(
        const std::vector<float> & mel,
        const std::vector<float> & f0,
        int64_t frames,
        uint64_t seed = 1234,
        uint64_t prior_noise_values = 0) const;
    void release_runtime_graph() const;

private:
    struct State;

    std::shared_ptr<const HifiGanVocoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules

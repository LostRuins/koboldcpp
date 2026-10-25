#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::modules {

enum class HiftVocoderWeightLayout {
    Canonical,
    TorchParametrizedWeightNorm,
};

enum class HiftVocoderUpsampleMode {
    ConvTranspose1d,
    CausalConv1dNearest,
};

enum class HiftVocoderSourceMode {
    Standard,
    CausalSineGen2,
    SineGen2,
};

struct HiftVocoderConfig {
    int64_t in_channels = 0;
    int64_t base_channels = 0;
    int64_t nb_harmonics = 0;
    int64_t sampling_rate = 0;
    float nsf_alpha = 0.1F;
    float nsf_sigma = 0.003F;
    float nsf_voiced_threshold = 10.0F;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    int64_t istft_n_fft = 0;
    int64_t istft_hop = 0;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> resblock_dilation_sizes;
    std::vector<int64_t> source_resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> source_resblock_dilation_sizes;
    float lrelu_slope = 0.1F;
    float audio_limit = 0.99F;
    int64_t conv_pre_kernel_size = 7;
    bool causal_convolutions = false;
    int64_t f0_num_class = 0;
    int64_t f0_in_channels = 0;
    int64_t f0_cond_channels = 0;
    std::vector<int64_t> f0_condnet_kernel_sizes;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    std::string tensor_prefix;
    HiftVocoderWeightLayout weight_layout = HiftVocoderWeightLayout::Canonical;
    HiftVocoderUpsampleMode upsample_mode = HiftVocoderUpsampleMode::ConvTranspose1d;
    HiftVocoderSourceMode source_mode = HiftVocoderSourceMode::Standard;
};

struct HiftVocoderWeights {
    struct Conv1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        core::TensorValue weight_tensor;
        core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t pad_left = 0;
        int64_t pad_right = 0;
        int64_t dilation = 1;
        bool use_bias = false;
    };

    struct ConvTranspose1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        core::TensorValue weight_tensor;
        core::TensorValue bias_tensor;
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        bool use_bias = false;
    };

    struct LinearWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        core::TensorValue weight_tensor;
        core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct SnakeWeights {
        std::vector<float> alpha;
        core::TensorValue alpha_tensor;
        core::TensorValue inv_alpha_tensor;
    };

    struct ResBlockWeights {
        std::vector<Conv1dWeights> convs1;
        std::vector<Conv1dWeights> convs2;
        std::vector<SnakeWeights> activations1;
        std::vector<SnakeWeights> activations2;
    };

    struct F0PredictorWeights {
        std::vector<Conv1dWeights> condnet;
        LinearWeights classifier;
    };

    HiftVocoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    F0PredictorWeights f0_predictor;
    Conv1dWeights conv_pre;
    std::vector<ConvTranspose1dWeights> ups;
    std::vector<Conv1dWeights> up_convs;
    std::vector<Conv1dWeights> source_downs;
    std::vector<ResBlockWeights> source_resblocks;
    std::vector<ResBlockWeights> resblocks;
    Conv1dWeights conv_post;
    LinearWeights source_linear;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

struct HiftVocoderOutput {
    std::vector<float> waveform;
    int64_t batch = 0;
    int64_t samples = 0;
    int64_t sample_rate = 0;
};

class HiftVocoderComponent {
public:
    static HiftVocoderComponent load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        HiftVocoderConfig config);
    static HiftVocoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        HiftVocoderConfig config);

    HiftVocoderComponent() = default;
    HiftVocoderComponent(
        std::shared_ptr<const HiftVocoderWeights> weights,
        core::BackendConfig backend);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const HiftVocoderWeights> & weights() const noexcept;
    int64_t sample_rate() const noexcept;
    int64_t num_mels() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;

    HiftVocoderOutput synthesize(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed = 1234,
        uint64_t prior_noise_values = 0,
        const std::vector<float> * source_random_values = nullptr) const;
    std::vector<float> predict_f0(const std::vector<float> & mel, int64_t frames) const;
    void release_runtime_cache() const;

private:
    struct State;

    std::shared_ptr<const HiftVocoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules

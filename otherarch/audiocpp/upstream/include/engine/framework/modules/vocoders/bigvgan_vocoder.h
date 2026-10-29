#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <ggml.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

struct BigVganVocoderConfig {
    int64_t sampling_rate = 0;
    int64_t num_mels = 0;
    int64_t n_fft = 0;
    int64_t hop_size = 0;
    int64_t win_size = 0;
    int64_t upsample_initial_channel = 0;
    bool snake_logscale = false;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

enum class BigVganActivationLayout {
    InterleavedPairs,
    GroupedByStage,
};

struct BigVganGraphOptions {
    bool apply_final_activation = true;
    bool use_tanh_at_final = false;
    BigVganActivationLayout activation_layout = BigVganActivationLayout::InterleavedPairs;
    bool use_depthwise_transpose_module = false;
    bool lower_padded_conv_transpose_as_crop = false;
};

struct BigVganVocoderWeights {
    struct Conv1dWeights {
        core::TensorValue weight;
        std::optional<core::TensorValue> bias;
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool use_bias = true;
    };

    struct ConvTranspose1dWeights {
        core::TensorValue conv1d_weight;
        std::optional<core::TensorValue> transpose_weight;
        std::optional<core::TensorValue> bias;
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        bool use_bias = true;
    };

    struct ActivationWeights {
        core::TensorValue alpha;
        core::TensorValue inv_beta;
        core::TensorValue up_filter;
        core::TensorValue down_filter;
    };

    struct ResBlockWeights {
        std::vector<Conv1dWeights> convs1;
        std::vector<Conv1dWeights> convs2;
        std::vector<ActivationWeights> activations;
    };

    BigVganVocoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dWeights conv_pre;
    std::vector<ConvTranspose1dWeights> ups;
    std::vector<ResBlockWeights> resblocks;
    ActivationWeights activation_post;
    Conv1dWeights conv_post;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

BigVganVocoderWeights load_direct_bigvgan_from_tensor_source(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    BigVganVocoderConfig config,
    BigVganActivationLayout activation_layout = BigVganActivationLayout::GroupedByStage);

BigVganVocoderWeights load_mono_module_list_bigvgan_from_tensor_source(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    BigVganVocoderConfig config,
    bool torch_parametrized_weight_norm,
    BigVganActivationLayout activation_layout = BigVganActivationLayout::GroupedByStage);

ggml_tensor * build_bigvgan_graph(
    ggml_context * ctx,
    core::BackendType backend_type,
    const BigVganVocoderWeights & weights,
    ggml_tensor * mel,
    const BigVganGraphOptions & options = {});

struct BigVganVocoderOutput {
    std::vector<float> waveform;
    int64_t batch = 0;
    int64_t samples = 0;
    int64_t sample_rate = 0;
};

class BigVganVocoderComponent {
public:
    static BigVganVocoderComponent load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        BigVganVocoderConfig config);
    static BigVganVocoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        BigVganVocoderConfig config);

    BigVganVocoderComponent() = default;
    BigVganVocoderComponent(
        std::shared_ptr<const BigVganVocoderWeights> weights,
        core::BackendConfig backend);

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const BigVganVocoderWeights> & weights() const noexcept;
    int64_t sample_rate() const noexcept;
    int64_t num_mels() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t parameter_count() const noexcept;

    BigVganVocoderOutput synthesize(
        const std::vector<float> & mel,
        int64_t frames) const;

    BigVganVocoderOutput synthesize_chunked(
        const std::vector<float> & mel,
        int64_t frames,
        int64_t chunk_frames,
        int64_t overlap_frames) const;
    void release_runtime_graph();

private:
    struct State;

    std::shared_ptr<const BigVganVocoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules

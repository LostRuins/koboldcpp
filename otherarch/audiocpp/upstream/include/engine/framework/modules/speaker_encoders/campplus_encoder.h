#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

enum class CampplusEncoderWeightLayout {
    Unfused,
    Fused,
};

struct CampplusEncoderConfig {
    int64_t feat_dim = 80;
    int64_t embedding_size = 192;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    // Preserve the established shared CAMPPlus pooling by default. GLM-TTS
    // opts into normalization by the complete segment width to match its
    // published ONNX graph for a final partial segment.
    bool normalize_partial_segment_by_full_length = false;
    std::string tensor_prefix = "speaker_encoder";
    float stats_variance_floor = 0.0F;
    CampplusEncoderWeightLayout weight_layout = CampplusEncoderWeightLayout::Unfused;
};

struct CampplusEncoderOutputs {
    std::vector<float> embedding;
    int64_t embedding_size = 0;
};

struct CampplusEncoderWeights {
    struct BatchNorm1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        core::TensorValue scale_tensor;
        core::TensorValue shift_tensor;
        bool affine = true;
    };

    struct BatchNorm2dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        core::TensorValue scale_tensor;
        core::TensorValue shift_tensor;
    };

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
        int64_t dilation = 1;
        bool use_bias = false;
    };

    struct Conv2dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        core::TensorValue weight_tensor;
        core::TensorValue bias_tensor;
        bool use_bias = false;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel_h = 0;
        int64_t kernel_w = 0;
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        int64_t padding_h = 0;
        int64_t padding_w = 0;
    };

    struct BasicResBlockWeights {
        Conv2dWeights conv1_folded;
        Conv2dWeights conv2_folded;
        bool use_shortcut = false;
        Conv2dWeights shortcut_conv_folded;
    };

    struct CAMLayerWeights {
        Conv1dWeights linear_local;
        Conv1dWeights linear1;
        Conv1dWeights linear2;
    };

    struct CAMDenseTDNNLayerWeights {
        BatchNorm1dWeights nonlinear1_bn;
        Conv1dWeights linear1;
        BatchNorm1dWeights nonlinear2_bn;
        CAMLayerWeights cam_layer;
    };

    struct CAMDenseTDNNBlockWeights {
        std::vector<CAMDenseTDNNLayerWeights> layers;
    };

    struct TransitLayerWeights {
        BatchNorm1dWeights nonlinear_bn;
        Conv1dWeights linear;
    };

    CampplusEncoderConfig config;
    std::filesystem::path source_path;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;

    Conv2dWeights head_conv1_folded;
    std::vector<BasicResBlockWeights> head_layer1;
    std::vector<BasicResBlockWeights> head_layer2;
    Conv2dWeights head_conv2_folded;
    Conv1dWeights tdnn_linear_folded;
    std::vector<CAMDenseTDNNBlockWeights> blocks;
    std::vector<TransitLayerWeights> transits;
    BatchNorm1dWeights out_nonlinear_bn;
    Conv1dWeights dense_linear_folded;

    int64_t loaded_tensor_count = 0;
    int64_t skipped_buffer_count = 0;
    int64_t parameter_count = 0;
};

class CampplusEncoderComponent {
public:
    static CampplusEncoderComponent load_from_safetensors(
        const std::filesystem::path & checkpoint_path,
        core::BackendConfig backend,
        CampplusEncoderConfig config = {});
    static CampplusEncoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        CampplusEncoderConfig config = {});

    CampplusEncoderComponent() = default;
    CampplusEncoderComponent(
        std::shared_ptr<const CampplusEncoderWeights> weights,
        core::BackendConfig backend);
    ~CampplusEncoderComponent();
    CampplusEncoderComponent(CampplusEncoderComponent &&) noexcept;
    CampplusEncoderComponent & operator=(CampplusEncoderComponent &&) noexcept;
    CampplusEncoderComponent(const CampplusEncoderComponent &) = delete;
    CampplusEncoderComponent & operator=(const CampplusEncoderComponent &) = delete;

    const core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const CampplusEncoderWeights> & weights() const noexcept;
    int64_t embedding_size() const noexcept;
    int64_t loaded_tensor_count() const noexcept;
    int64_t skipped_buffer_count() const noexcept;
    int64_t parameter_count() const noexcept;

    CampplusEncoderOutputs embed_from_features(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;
    void release_runtime_graph();

private:
    struct State;

    std::shared_ptr<const CampplusEncoderWeights> weights_;
    core::BackendConfig backend_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::modules

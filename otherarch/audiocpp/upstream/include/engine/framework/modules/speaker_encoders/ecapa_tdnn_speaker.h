#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules::ecapa_tdnn {

enum class Conv1dPaddingMode {
    Zero,
    Reflect,
};

enum class AttentivePoolingKind {
    GlobalContext,
    Simple,
};

struct BatchNorm1dWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

struct Conv1dWeights {
    std::string weight_name;
    std::vector<int64_t> weight_source_shape;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t dilation = 1;
    bool use_bias = true;
    Conv1dPaddingMode padding_mode = Conv1dPaddingMode::Reflect;
    std::optional<std::string> bias_name;
};

struct TDNNBlockWeights {
    Conv1dWeights conv;
    BatchNorm1dWeights norm;
};

struct Res2NetBlockWeights {
    std::vector<TDNNBlockWeights> blocks;
    int64_t scale = 8;
    int64_t width = 0;
    bool first_chunk_passthrough = true;
};

struct SEBlockWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
};

struct SERes2NetBlockWeights {
    TDNNBlockWeights tdnn1;
    Res2NetBlockWeights res2net;
    TDNNBlockWeights tdnn2;
    SEBlockWeights se;
    bool use_shortcut = false;
    Conv1dWeights shortcut;
};

struct AspWeights {
    TDNNBlockWeights tdnn;
    Conv1dWeights conv;
    AttentivePoolingKind kind = AttentivePoolingKind::GlobalContext;
    bool tdnn_use_batch_norm = true;
};

struct EcapaWeights {
    std::shared_ptr<const assets::TensorSource> source;
    int64_t feature_dim = 80;
    int64_t embedding_dim = 192;
    float stats_eps = 1.0e-12f;
    size_t weight_context_bytes = 256ull * 1024ull * 1024ull;
    size_t graph_context_bytes = 256ull * 1024ull * 1024ull;
    TDNNBlockWeights block0;
    std::vector<SERes2NetBlockWeights> se_blocks;
    TDNNBlockWeights mfa;
    bool mfa_use_batch_norm = true;
    AspWeights asp;
    BatchNorm1dWeights asp_bn;
    Conv1dWeights fc;
    std::vector<float> classifier_weight;
    std::vector<float> embedding_global_mean;
};

}  // namespace engine::modules::ecapa_tdnn

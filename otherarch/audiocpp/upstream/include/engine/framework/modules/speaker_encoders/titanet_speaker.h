#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules::titanet {

struct JasperBlockConfig {
    int64_t filters = 0;
    int64_t repeat = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    bool residual = false;
    bool separable = false;
    bool se = false;
};

struct TitaNetConfig {
    int64_t sample_rate = 0;
    int64_t n_mels = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t win_length = 0;
    int64_t embedding_size = 0;
    int64_t num_classes = 0;
    float preemph = 0.97f;
    std::string normalize;
    std::string pool_mode;
    std::vector<JasperBlockConfig> jasper;
};

struct BatchNorm1dWeights {
    int64_t channels = 0;
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

struct Conv1dWeights {
    std::string weight_name;
    std::vector<int64_t> weight_source_shape;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    int64_t groups = 1;
    bool use_bias = false;
    std::optional<std::string> bias_name;
};

struct SeparableConvBn {
    Conv1dWeights depthwise;
    Conv1dWeights pointwise;
    BatchNorm1dWeights bn;
};

struct SqueezeExciteWeights {
    std::string fc0_weight_name;
    std::string fc2_weight_name;
    int64_t channels = 0;
    int64_t hidden = 0;
};

struct JasperBlockWeights {
    std::vector<SeparableConvBn> repeats;
    bool has_residual = false;
    Conv1dWeights residual_conv;
    BatchNorm1dWeights residual_bn;
    SqueezeExciteWeights se;
};

struct AttentionPoolWeights {
    Conv1dWeights tdnn_conv;
    BatchNorm1dWeights tdnn_bn;
    Conv1dWeights out_conv;
};

struct EmbeddingHeadWeights {
    BatchNorm1dWeights bn;
    Conv1dWeights conv;
};

struct TitaNetWeights {
    TitaNetConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<float> window;
    std::vector<float> fb;
    std::vector<JasperBlockWeights> blocks;
    AttentionPoolWeights pool;
    EmbeddingHeadWeights emb;
    std::vector<float> classifier_weight;
    int64_t num_classes = 0;
};

}  // namespace engine::modules::titanet

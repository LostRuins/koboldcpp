#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct Wav2Vec2BertEncoderConfig {
    int64_t feature_dim = 160;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t num_hidden_layers = 24;
    int64_t output_hidden_layer = 17;
    int64_t num_attention_heads = 16;
    int64_t relative_positions = 73;
    int64_t relative_left = 64;
    int64_t relative_right = 8;
    int64_t conv_kernel = 31;
    float layer_norm_eps = 1.0e-5F;
    bool apply_semantic_normalization = true;
    bool mask_padded_frames = true;
    bool project_relative_keys_first = false;
    bool pointwise_conv_as_linear = false;
};

struct Wav2Vec2BertAttentionWeightNames {
    std::string q = "linear_q";
    std::string k = "linear_k";
    std::string v = "linear_v";
    std::string out = "linear_out";
    std::string distance_embedding = "distance_embedding";
};

struct Wav2Vec2BertConvWeightNames {
    std::string layer_norm = "layer_norm";
    std::string pointwise_in = "pointwise_conv1";
    std::string depthwise = "depthwise_conv";
    std::string depthwise_layer_norm = "depthwise_layer_norm";
    std::string pointwise_out = "pointwise_conv2";
};

struct Wav2Vec2BertLayerWeightNames {
    std::string ffn1_norm = "ffn1_layer_norm";
    std::string ffn1_in = "ffn1.intermediate_dense";
    std::string ffn1_out = "ffn1.output_dense";
    std::string self_attn_norm = "self_attn_layer_norm";
    std::string self_attn = "self_attn";
    Wav2Vec2BertAttentionWeightNames attention;
    std::string conv = "conv_module";
    Wav2Vec2BertConvWeightNames conv_module;
    std::string ffn2_norm = "ffn2_layer_norm";
    std::string ffn2_in = "ffn2.intermediate_dense";
    std::string ffn2_out = "ffn2.output_dense";
    std::string final_norm = "final_layer_norm";
};

struct Wav2Vec2BertEncoderWeightBinding {
    std::string feature_norm = "feature_projection.layer_norm";
    std::string feature_projection = "feature_projection.projection";
    std::string encoder_layers = "encoder.layers";
    Wav2Vec2BertLayerWeightNames layer;
    std::string stats_mean = "mean";
    std::string stats_variance = "var";
    assets::TensorStorageType matmul_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType conv_storage_type = assets::TensorStorageType::F32;
    assets::TensorStorageType stats_storage_type = assets::TensorStorageType::F32;
};

struct Wav2Vec2BertAttentionWeights {
    LinearWeights q;
    LinearWeights k;
    LinearWeights v;
    LinearWeights out;
    core::TensorValue distance_embedding;
};

struct Wav2Vec2BertConvWeights {
    NormWeights layer_norm;
    Conv1dWeights pointwise_in;
    DepthwiseConv1dWeights depthwise;
    NormWeights depthwise_layer_norm;
    Conv1dWeights pointwise_out;
};

struct Wav2Vec2BertLayerWeights {
    NormWeights ffn1_norm;
    LinearWeights ffn1_in;
    LinearWeights ffn1_out;
    NormWeights self_attn_norm;
    Wav2Vec2BertAttentionWeights self_attn;
    Wav2Vec2BertConvWeights conv;
    NormWeights ffn2_norm;
    LinearWeights ffn2_in;
    LinearWeights ffn2_out;
    NormWeights final_norm;
};

struct Wav2Vec2BertEncoderWeights {
    Wav2Vec2BertEncoderConfig config;
    std::shared_ptr<core::BackendWeightStore> store;
    NormWeights feature_norm;
    LinearWeights feature_projection;
    std::vector<Wav2Vec2BertLayerWeights> layers;
    core::TensorValue semantic_mean;
    core::TensorValue semantic_std;
    core::TensorValue half_scale;
    core::TensorValue relative_scale;
    core::TensorValue left_conv_pad;
};

struct Wav2Vec2BertEncoderInput {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct Wav2Vec2BertEncoderOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

class Wav2Vec2BertEncoderComponent {
public:
    static Wav2Vec2BertEncoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> model_source,
        std::shared_ptr<const assets::TensorSource> stats_source,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        Wav2Vec2BertEncoderConfig config = {});
    static Wav2Vec2BertEncoderComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> model_source,
        std::shared_ptr<const assets::TensorSource> stats_source,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        Wav2Vec2BertEncoderConfig config,
        Wav2Vec2BertEncoderWeightBinding binding);

    Wav2Vec2BertEncoderComponent();
    Wav2Vec2BertEncoderComponent(
        std::shared_ptr<const Wav2Vec2BertEncoderWeights> weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes);
    ~Wav2Vec2BertEncoderComponent();

    Wav2Vec2BertEncoderComponent(const Wav2Vec2BertEncoderComponent &) = delete;
    Wav2Vec2BertEncoderComponent & operator=(const Wav2Vec2BertEncoderComponent &) = delete;
    Wav2Vec2BertEncoderComponent(Wav2Vec2BertEncoderComponent &&) noexcept;
    Wav2Vec2BertEncoderComponent & operator=(Wav2Vec2BertEncoderComponent &&) noexcept;

    void prepare(int64_t frames);
    Wav2Vec2BertEncoderOutput encode(const Wav2Vec2BertEncoderInput & features);
    void release_runtime_graph();

private:
    class Graph;

    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const Wav2Vec2BertEncoderWeights> weights_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::modules

#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace engine::modules {

enum class T5BaseFeedForwardKind {
    Relu,
    GatedGeluTanh,
};

struct T5BaseEncoderConfig {
    int64_t hidden_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t vocab_size = 0;
    int64_t relative_attention_num_buckets = 32;
    int64_t relative_attention_max_distance = 128;
    float rms_norm_eps = 1.0e-6F;
    T5BaseFeedForwardKind feed_forward_kind = T5BaseFeedForwardKind::Relu;
    bool shared_relative_position_bias = true;
    bool flash_attention = false;
};

struct T5BaseEncoderLayerWeights {
    core::TensorValue self_attention_layer_norm;
    core::TensorValue ffn_layer_norm;
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
    LinearWeights wi_proj;
    LinearWeights wo_proj;
    LinearWeights gate_proj;
    std::optional<core::TensorValue> relative_attention_bias;
};

struct T5BaseEncoderWeights {
    core::TensorValue embed_tokens;
    core::TensorValue relative_attention_bias;
    std::vector<T5BaseEncoderLayerWeights> layers;
    core::TensorValue final_layer_norm;
};

std::vector<int32_t> t5_base_relative_position_buckets(
    int64_t query_length,
    int64_t key_length,
    int64_t num_buckets,
    int64_t max_distance,
    bool bidirectional = true);

class T5BaseEncoderModule {
public:
    explicit T5BaseEncoderModule(T5BaseEncoderConfig config);

    const T5BaseEncoderConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_ids,
        const core::TensorValue & relative_position_buckets,
        const core::TensorValue & additive_attention_mask,
        const T5BaseEncoderWeights & weights) const;
private:
    T5BaseEncoderConfig config_;
};

}  // namespace engine::modules

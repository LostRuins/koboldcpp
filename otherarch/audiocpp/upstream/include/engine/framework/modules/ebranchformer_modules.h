#pragma once

#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

namespace engine::modules {

struct EBranchformerBlockConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    float norm_eps = 1.0e-12F;
    int gate_kernel_size = 31;
    int merge_kernel_size = 31;
    bool use_packed_qkv = true;
    bool use_packed_kv = true;
};

struct EBranchformerCgMLPWeights {
    LinearWeights input_projection;
    NormWeights gate_norm;
    DepthwiseConv1dWeights gate_conv;
    LinearWeights output_projection;
};

struct EBranchformerBlockWeights {
    FeedForwardWeights macaron_ffn;
    NormWeights macaron_norm;
    NormWeights attention_norm;
    AttentionWeights attention;
    NormWeights cgmlp_norm;
    EBranchformerCgMLPWeights cgmlp;
    DepthwiseConv1dWeights merge_conv;
    LinearWeights merge_projection;
    FeedForwardWeights final_ffn;
    NormWeights final_ffn_norm;
    NormWeights output_norm;
    NormWeights cross_attention_norm;
    AttentionWeights cross_attention;
};

class EBranchformerBlockModule {
public:
    explicit EBranchformerBlockModule(EBranchformerBlockConfig config) : config_(config) {}

    // Optional cross-attention precedes the final normalization.
    core::TensorValue build(
        core::ModuleBuildContext & ctx, core::TensorValue input,
        const EBranchformerBlockWeights & weights, const core::TensorValue & half_scale,
        const core::TensorValue * memory = nullptr,
        const core::TensorValue * memory_mask = nullptr) const;

private:
    EBranchformerBlockConfig config_;
};

}  // namespace engine::modules

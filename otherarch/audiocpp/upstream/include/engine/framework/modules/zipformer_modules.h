#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <functional>
#include <vector>

namespace engine::modules {

using ZipformerConstantFactory =
    std::function<core::TensorValue(const core::TensorShape &, std::vector<float>)>;

struct ZipformerLinearWeights {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct ZipformerRelativeAttentionConfig {
    int64_t num_heads = 0;
    int64_t query_head_dim = 0;
    int64_t position_dim = 48;
    int64_t position_head_dim = 4;
};

struct ZipformerRelativeAttentionWeights {
    ZipformerLinearWeights in_proj;
    std::vector<float> linear_pos_weight;
};

struct ZipformerRelativeAttentionOutputs {
    std::vector<core::TensorValue> weights;
    core::TensorValue key_state;
};

class ZipformerRelativeAttentionModule {
public:
    explicit ZipformerRelativeAttentionModule(ZipformerRelativeAttentionConfig config);

    ZipformerRelativeAttentionOutputs build(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_key,
        const core::TensorValue & padding_mask,
        const ZipformerRelativeAttentionWeights & weights) const;

private:
    ZipformerRelativeAttentionConfig config_;
};

struct ZipformerNonlinearAttentionWeights {
    ZipformerLinearWeights in_proj;
    ZipformerLinearWeights out_proj;
};

struct ZipformerStatefulOutputs {
    core::TensorValue output;
    core::TensorValue state;
};

class ZipformerNonlinearAttentionModule {
public:
    ZipformerStatefulOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & attention,
        const core::TensorValue & cached_value,
        const ZipformerNonlinearAttentionWeights & weights) const;
};

struct ZipformerConvolutionConfig {
    int64_t kernel_size = 0;
};

struct ZipformerConvolutionWeights {
    ZipformerLinearWeights in_proj;
    DepthwiseConv1dWeights causal_conv;
    DepthwiseConv1dWeights chunkwise_conv;
    std::vector<float> chunk_scale_left;
    std::vector<float> chunk_scale_right;
    ZipformerLinearWeights out_proj;
};

class ZipformerConvolutionModule {
public:
    explicit ZipformerConvolutionModule(ZipformerConvolutionConfig config);

    ZipformerStatefulOutputs build(
        core::ModuleBuildContext & ctx,
        ZipformerConstantFactory & constant_factory,
        const core::TensorValue & input,
        const core::TensorValue & cached_value,
        const ZipformerConvolutionWeights & weights) const;

private:
    ZipformerConvolutionConfig config_;
};

}  // namespace engine::modules

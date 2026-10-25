#pragma once
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

namespace engine::modules {
// Vocos/F5-style non-causal ConvNeXt, logical [batch, frames, channels].
struct ConvNeXt1dWeights {
    DepthwiseConv1dWeights depthwise;
    NormWeights norm;
    LinearWeights expansion;
    LinearWeights projection;
    core::TensorValue gamma;
};
// Optional BF16 autocast preserves F32 normalization and residual arithmetic.
core::TensorValue build_convnext1d(
    core::ModuleBuildContext & ctx, const core::TensorValue & input,
    const ConvNeXt1dWeights & weights, bool bf16_autocast = false);
}

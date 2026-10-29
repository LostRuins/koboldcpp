#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"

#include <cstdint>
#include <optional>

namespace engine::modules {

struct DepthwiseConv1dConfig {
    int64_t channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
};

struct DepthwiseConv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class DepthwiseConv1dModule {
public:
    explicit DepthwiseConv1dModule(DepthwiseConv1dConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const DepthwiseConv1dWeights & weights) const;

private:
    DepthwiseConv1dConfig config_;
};

struct PointwiseConv1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    bool use_bias = true;
    bool quant = false;
};

using PointwiseConv1dWeights = Conv1dWeights;

class PointwiseConv1dModule {
public:
    explicit PointwiseConv1dModule(PointwiseConv1dConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PointwiseConv1dWeights & weights) const;

private:
    PointwiseConv1dConfig config_;
};

enum class StreamingPadMode {
    Constant,
    Replicate,
};

enum class StreamingConv1dPaddingMode {
    StreamingSame,
    StrictCausal,
    Explicit,
};

struct StreamingConv1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int dilation = 1;
    bool use_bias = true;
    StreamingPadMode pad_mode = StreamingPadMode::Constant;
    StreamingConv1dPaddingMode padding_mode = StreamingConv1dPaddingMode::StreamingSame;
    int64_t explicit_left = 0;
    int64_t explicit_right = 0;
};

using StreamingConv1dWeights = Conv1dWeights;

class StreamingConv1dModule {
public:
    explicit StreamingConv1dModule(StreamingConv1dConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const StreamingConv1dWeights & weights) const;

private:
    StreamingConv1dConfig config_;
};

using CausalConv1dPadMode = StreamingPadMode;
using CausalConv1dPaddingMode = StreamingConv1dPaddingMode;
using CausalConv1dConfig = StreamingConv1dConfig;
using CausalConv1dWeights = StreamingConv1dWeights;
using CausalConv1dModule = StreamingConv1dModule;

struct DepthwiseConvTranspose1dConfig {
    int64_t channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    bool use_bias = false;
};

struct DepthwiseConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class DepthwiseConvTranspose1dModule {
public:
    explicit DepthwiseConvTranspose1dModule(DepthwiseConvTranspose1dConfig config);
    // Accepts [channels, frames] or [1, channels, frames] input with a [channels, 1, 1, kernel] FIR weight.
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const DepthwiseConvTranspose1dWeights & weights) const;

private:
    DepthwiseConvTranspose1dConfig config_;
};

}  // namespace engine::modules

#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <vector>

namespace engine::codecs {

struct SEANetResidualWeights {
    engine::modules::StreamingConv1dWeights conv1;
    engine::modules::StreamingConv1dWeights conv2;
};

struct SEANetDecoderStageWeights {
    engine::modules::ConvTranspose1dWeights upsample;
    std::vector<SEANetResidualWeights> residual_blocks;
};

struct SEANetDecoderConfig {
    int64_t input_channels = 0;
    int64_t output_channels = 0;
    int64_t final_kernel_size = 0;
    int64_t residual_layers = 0;
    int64_t residual_hidden_divisor = 2;
    int64_t residual_dilation_base = 2;
    bool use_bias = true;
    engine::modules::StreamingPadMode pad_mode = engine::modules::StreamingPadMode::Constant;
    std::vector<int64_t> stage_channels;
    std::vector<int64_t> stage_strides;
};

struct SEANetDecoderWeights {
    engine::modules::StreamingConv1dWeights input_projection;
    std::vector<SEANetDecoderStageWeights> stages;
    engine::modules::StreamingConv1dWeights output_projection;
};

class SEANetDecoder {
public:
    explicit SEANetDecoder(SEANetDecoderConfig config);
    engine::core::TensorValue build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const SEANetDecoderWeights & weights) const;

private:
    SEANetDecoderConfig config_;
};

struct VocoderUpsampleBlockConfig {
    int64_t channels_in = 0;
    int64_t channels_out = 0;
    bool use_bias = true;
};

struct VocoderUpsampleBlockWeights {
    engine::modules::ConvTranspose1dWeights up;
    engine::modules::Conv1dWeights post;
};

class VocoderUpsampleBlock {
public:
    explicit VocoderUpsampleBlock(VocoderUpsampleBlockConfig config);
    engine::core::TensorValue build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const VocoderUpsampleBlockWeights & weights) const;

private:
    VocoderUpsampleBlockConfig config_;
};

struct WaveNetResidualBlockConfig {
    int64_t channels = 0;
    int dilation = 1;
    bool use_bias = true;
};

struct WaveNetResidualBlockWeights {
    engine::modules::Conv1dWeights filter;
    engine::modules::Conv1dWeights gate;
    engine::modules::Conv1dWeights residual;
};

class WaveNetResidualBlock {
public:
    explicit WaveNetResidualBlock(WaveNetResidualBlockConfig config);
    engine::core::TensorValue build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const WaveNetResidualBlockWeights & weights) const;

private:
    WaveNetResidualBlockConfig config_;
};

struct UNetResBlock1DConfig {
    int64_t channels = 0;
    int64_t groups = 8;
    float eps = 1e-5f;
    bool use_bias = true;
};

struct UNetResBlock1DWeights {
    engine::modules::NormWeights norm1;
    engine::modules::Conv1dWeights conv1;
    engine::modules::NormWeights norm2;
    engine::modules::Conv1dWeights conv2;
};

class UNetResBlock1D {
public:
    explicit UNetResBlock1D(UNetResBlock1DConfig config);
    engine::core::TensorValue build(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const UNetResBlock1DWeights & weights) const;

private:
    UNetResBlock1DConfig config_;
};

}  // namespace engine::codecs

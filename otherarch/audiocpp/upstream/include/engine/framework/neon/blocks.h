#pragma once

#include "engine/framework/neon/execution.h"

#include <optional>

namespace engine::neon {

struct ConvTranspose1dConfig {
    int input_channels = 0;
    int output_channels = 0;
    int kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int output_padding = 0;
};

// Weights are contiguous [input_channels, output_channels, kernel_size].
// Polyphase convolution avoids materializing a zero-inserted input.
ExecutionPlan::Value build_conv_transpose1d(ExecutionPlan &plan, ExecutionPlan::Value input,
    ConvTranspose1dConfig config, const std::vector<float> &weight, const std::vector<float> &bias = {});

struct Conv1dWeights {
    std::vector<float> weight, bias;
};

struct HifiGanConfig {
    int input_channels = 0;
    int initial_channels = 0;
    int output_channels = 1;
    int conditioning_channels = 0;
    std::vector<int> upsample_rates, upsample_kernels, residual_kernels;
    std::vector<std::vector<int>> residual_dilations;
    bool paired_convolutions = true;
    float negative_slope = 0.1F;
    float output_negative_slope = 0.1F;
};

struct HifiGanWeights {
    struct Residual {
        std::vector<Conv1dWeights> first, second;
    };
    Conv1dWeights pre, post, conditioning;
    std::vector<Conv1dWeights> upsample;
    std::vector<Residual> residual;
};

// Complete HiFi-GAN waveform decoder; optional global conditioning is [C,1]
// or [C,input_frames]. Weight loading and file formats belong to the caller.
ExecutionPlan::Value build_hifigan(ExecutionPlan &plan, ExecutionPlan::Value input,
    std::optional<ExecutionPlan::Value> conditioning, const HifiGanConfig &config, const HifiGanWeights &weights);

struct ResidualConvWeights {
    std::vector<float> first_weight, first_bias, second_weight, second_bias, scale;
};

struct ConvNeXtWeights {
    std::vector<float> depthwise_weight, depthwise_bias, norm_weight, norm_bias;
    std::vector<float> expand_weight, expand_bias, project_weight, project_bias, scale;
};

// These blocks compose ordinary operators and share the plan's allocation and
// execution policies. They do not introduce separate model-shaped executors.
ExecutionPlan::Value build_residual_conv(ExecutionPlan &plan, ExecutionPlan::Value input, int channels,
                                         int hidden_channels, int kernel_size, ActivationKind activation,
                                         const ResidualConvWeights &weights);

ExecutionPlan::Value build_convnext(ExecutionPlan &plan, ExecutionPlan::Value input, int channels,
                                    int expanded_channels, int kernel_size, float epsilon,
                                    const ConvNeXtWeights &weights);

} // namespace engine::neon

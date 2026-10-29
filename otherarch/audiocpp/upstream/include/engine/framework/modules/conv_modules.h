#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>
#include <optional>

namespace engine::modules {

struct Conv1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
    // Preserve BF16 convolution and bias rounding, with F32 graph outputs.
    bool bf16_compute = false;
};

struct Conv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv1dModule {
public:
    explicit Conv1dModule(Conv1dConfig config);

    const Conv1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv1dConfig config_;
};

// Raw Metal fast paths on channel-fast activations (ne = [channels, frames], contiguous
// F32) for the audio codec decoder's chained regions. Unlike the module build() methods
// these take and return raw ggml tensors without the canonical [frames, channels]
// orientation, so consecutive convolutions chain without paying two transposes per conv.
// The caller owns layout conversion at region edges and causal padding.
//
// conv1d_pertap_channel_fast: requires padding=0, stride=1; per-tap GEMM decomposition;
// returns [out_channels, output_frames] with bias broadcast-added when use_bias.
ggml_tensor * conv1d_pertap_channel_fast(
    core::ModuleBuildContext & ctx,
    const Conv1dWeights & weights,
    ggml_tensor * input_cf,
    const Conv1dConfig & config);

struct Conv2dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
    bool cuda_fast_lowering = false;
    bool cuda_large_shape_lowering = false;
    bool cuda_tile_f16_accum_output_lowering = false;
    bool cuda_cudnn = false;
    bool cuda_f16_output = false;
    // Metal: FP32 per-tap GEMMs avoid materializing large im2col buffers.
    bool metal_per_tap = false;
};

struct Conv2dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv2dModule {
public:
    explicit Conv2dModule(Conv2dConfig config);
    Conv2dModule(Conv2dConfig config, bool cuda_cudnn, bool cuda_f16_output = false);
    Conv2dModule(Conv2dConfig config, bool cuda_cudnn, bool cuda_f16_output, bool metal_per_tap);

    const Conv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv2dConfig config_;
};

struct Conv3dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_depth = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_depth = 1;
    int stride_height = 1;
    int stride_width = 1;
    int padding_depth = 0;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_depth = 1;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
    bool cuda_fast_lowering = false;
    bool cuda_large_shape_lowering = false;
    bool cuda_tile_f16_accum_output_lowering = false;
};

struct Conv3dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv3dModule {
public:
    explicit Conv3dModule(Conv3dConfig config);

    const Conv3dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv3dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv3dConfig config_;
};

struct CausalConv2dConfig {
    Conv2dConfig conv;
    int64_t pad_left = 0;
    int64_t pad_right = 0;
    int64_t pad_top = 0;
    int64_t pad_bottom = 0;
};

class CausalConv2dModule {
public:
    explicit CausalConv2dModule(CausalConv2dConfig config);

    const CausalConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    CausalConv2dConfig config_;
};

struct SameWidthCausalConv2dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    bool use_bias = true;
};

class SameWidthCausalConv2dModule {
public:
    explicit SameWidthCausalConv2dModule(SameWidthCausalConv2dConfig config);

    const SameWidthCausalConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    SameWidthCausalConv2dConfig config_;
};

struct PixelNormCausalConv2dResBlockConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 3;
    int pixel_norm_axis = 1;
    float pixel_norm_eps = 1e-6f;
};

struct PixelNormCausalConv2dResBlockWeights {
    Conv2dWeights conv1;
    Conv2dWeights conv2;
    std::optional<Conv2dWeights> shortcut;
};

class PixelNormCausalConv2dResBlockModule {
public:
    explicit PixelNormCausalConv2dResBlockModule(PixelNormCausalConv2dResBlockConfig config);

    const PixelNormCausalConv2dResBlockConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PixelNormCausalConv2dResBlockWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    PixelNormCausalConv2dResBlockConfig config_;
};

struct CausalConv2dUpsampleConfig {
    int64_t channels = 0;
    int64_t kernel_size = 3;
    int64_t scale_height = 2;
    int64_t scale_width = 2;
    int64_t crop_top = 1;
    int64_t crop_bottom = 0;
};

class CausalConv2dUpsampleModule {
public:
    explicit CausalConv2dUpsampleModule(CausalConv2dUpsampleConfig config);

    const CausalConv2dUpsampleConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    CausalConv2dUpsampleConfig config_;
};

struct DepthwiseConv2dConfig {
    int64_t channels = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
};

class DepthwiseConv2dModule {
public:
    explicit DepthwiseConv2dModule(DepthwiseConv2dConfig config);

    const DepthwiseConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    DepthwiseConv2dConfig config_;
};

struct ConvTranspose1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
};

struct ConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

bool is_conv_transpose1d_col2im_fast_path_eligible(
    const core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config) noexcept;

// conv_transpose1d_col2im_channel_fast: same col2im math as ConvTranspose1dModule's fast
// path but consumes channel-fast input directly (skipping its internal transpose) and
// returns the raw time-fast [frames_out, out_channels] tensor with bias included; the
// caller owns causal trimming and layout conversion.
ggml_tensor * conv_transpose1d_col2im_channel_fast(
    core::ModuleBuildContext & ctx,
    const ConvTranspose1dWeights & weights,
    ggml_tensor * input_cf,
    const ConvTranspose1dConfig & config);

class ConvTranspose1dModule {
public:
    explicit ConvTranspose1dModule(ConvTranspose1dConfig config);

    const ConvTranspose1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConvTranspose1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    ConvTranspose1dConfig config_;
};

}  // namespace engine::modules

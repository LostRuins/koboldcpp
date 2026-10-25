#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine::codecs {

struct WanVideoVAEConfig {
  int64_t latent_channels = 16;
  std::vector<float> mean{-0.7571F, -0.7089F, -0.9113F, 0.1075F,
                          -0.1745F, 0.9653F,  -0.1517F, 1.5508F,
                          0.4134F,  -0.0715F, 0.5517F,  -0.3632F,
                          -0.1922F, -0.9497F, 0.2503F,  -0.2921F};
  std::vector<float> std{2.8184F, 1.4541F, 2.3275F, 2.6558F, 1.2196F, 1.7708F,
                         2.6052F, 2.0743F, 3.2687F, 2.1526F, 2.8652F, 1.5579F,
                         1.6382F, 1.1253F, 2.8251F, 1.9160F};
};

struct WanVideoVAEArchitectureConfig {
  int64_t dim = 96;
  int64_t latent_channels = 16;
  std::vector<int64_t> dim_mult{1, 2, 4, 4};
  int64_t num_res_blocks = 2;
  std::vector<float> attention_scales;
  std::vector<bool> temporal_downsample{false, true, true};
  float rms_norm_eps = 0.0F;
  bool scale_encoded_latents = true;
  bool scale_decoded_latents = true;
  bool cache_raw_single_frame_inputs = false;
  bool use_cuda_fast_lowerings = false;
  bool use_cuda_large_shape_lowerings = false;
  bool use_cuda_tile_f16_accum_output_lowering = false;
};

enum class WanVideoVAELatentTransform {
  Encode,
  Decode,
};

class WanVideoVAELatentScaleModule {
public:
  WanVideoVAELatentScaleModule(WanVideoVAEConfig config,
                               WanVideoVAELatentTransform transform);

  engine::core::TensorValue build(engine::core::ModuleBuildContext &ctx,
                                  const engine::core::TensorValue &input,
                                  const engine::core::TensorValue &mean,
                                  const engine::core::TensorValue &std) const;

  static WanVideoVAEConfig s2v_14b_config();

private:
  WanVideoVAEConfig config_;
  WanVideoVAELatentTransform transform_;
};

class WanCausalConv3dModule {
public:
  explicit WanCausalConv3dModule(engine::modules::Conv3dConfig config);

  engine::core::TensorValue
  build(engine::core::ModuleBuildContext &ctx,
        const engine::core::TensorValue &input,
        const engine::modules::Conv3dWeights &weights) const;
  engine::core::TensorValue build(engine::core::ModuleBuildContext &ctx,
                                  const engine::core::TensorValue &input,
                                  const engine::modules::Conv3dWeights &weights,
                                  const engine::core::TensorValue *cache) const;

private:
  engine::modules::Conv3dConfig config_;
};

struct WanVideoRMSNorm3dConfig {
  int64_t channels = 0;
  float eps = 0.0F;
  bool cuda_fast_lowering = false;
};

struct WanVideoRMSNorm3dWeights {
  engine::core::TensorValue gamma;
};

class WanVideoRMSNorm3dModule {
public:
  explicit WanVideoRMSNorm3dModule(WanVideoRMSNorm3dConfig config);

  engine::core::TensorValue
  build(engine::core::ModuleBuildContext &ctx,
        const engine::core::TensorValue &input,
        const WanVideoRMSNorm3dWeights &weights) const;

private:
  WanVideoRMSNorm3dConfig config_;
};

struct WanVideoVAEResidualBlockConfig {
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t kernel_size = 3;
  float rms_norm_eps = 1.0e-6F;
  bool cuda_fast_lowering = false;
  bool cuda_large_shape_lowering = false;
  bool cuda_tile_f16_accum_output_lowering = false;
};

struct WanVideoVAEResidualBlockWeights {
  WanVideoRMSNorm3dWeights norm1;
  engine::modules::Conv3dWeights conv1;
  WanVideoRMSNorm3dWeights norm2;
  engine::modules::Conv3dWeights conv2;
  std::optional<engine::modules::Conv3dWeights> shortcut;
};

struct WanVideoVAECacheBuildState;

class WanVideoVAEResidualBlockModule {
public:
  explicit WanVideoVAEResidualBlockModule(
      WanVideoVAEResidualBlockConfig config);

  engine::core::TensorValue
  build(engine::core::ModuleBuildContext &ctx,
        const engine::core::TensorValue &input,
        const WanVideoVAEResidualBlockWeights &weights) const;
  engine::core::TensorValue
  build_cached(engine::core::ModuleBuildContext &ctx,
               const engine::core::TensorValue &input,
               const WanVideoVAEResidualBlockWeights &weights,
               WanVideoVAECacheBuildState &cache) const;

private:
  WanVideoVAEResidualBlockConfig config_;
};

enum class WanVideoVAEResampleMode {
  None,
  Upsample2d,
  Upsample3d,
  Downsample2d,
  Downsample3d,
};

struct WanVideoVAEResampleConfig {
  int64_t channels = 0;
  WanVideoVAEResampleMode mode = WanVideoVAEResampleMode::None;
  bool cuda_fast_lowering = false;
  bool cuda_large_shape_lowering = false;
  bool cuda_tile_f16_accum_output_lowering = false;
};

struct WanVideoVAEResampleWeights {
  engine::modules::Conv2dWeights spatial;
  std::optional<engine::modules::Conv3dWeights> time;
};

struct WanVideoVAECacheBuildState {
  const std::vector<engine::core::TensorValue> *input_caches = nullptr;
  std::vector<engine::core::TensorValue> *output_caches = nullptr;
  size_t cursor = 0;
  bool raw_single_frame_inputs = false;
  bool addressable_output_caches = true;
};

class WanVideoVAEResampleModule {
public:
  explicit WanVideoVAEResampleModule(WanVideoVAEResampleConfig config);

  engine::core::TensorValue
  build_initial(engine::core::ModuleBuildContext &ctx,
                const engine::core::TensorValue &input,
                const WanVideoVAEResampleWeights &weights) const;
  engine::core::TensorValue
  build_cached(engine::core::ModuleBuildContext &ctx,
               const engine::core::TensorValue &input,
               const WanVideoVAEResampleWeights &weights,
               WanVideoVAECacheBuildState &cache) const;

private:
  WanVideoVAEResampleConfig config_;
};

struct WanVideoVAEAttentionBlockConfig {
  int64_t channels = 0;
  float rms_norm_eps = 1.0e-6F;
  bool cuda_fast_lowering = false;
  bool cuda_large_shape_lowering = false;
  bool cuda_tile_f16_accum_output_lowering = false;
};

struct WanVideoVAEAttentionBlockWeights {
  WanVideoRMSNorm3dWeights norm;
  engine::modules::Conv2dWeights qkv;
  engine::modules::Conv2dWeights proj;
};

class WanVideoVAEAttentionBlockModule {
public:
  explicit WanVideoVAEAttentionBlockModule(
      WanVideoVAEAttentionBlockConfig config);

  engine::core::TensorValue
  build(engine::core::ModuleBuildContext &ctx,
        const engine::core::TensorValue &input,
        const WanVideoVAEAttentionBlockWeights &weights) const;

private:
  WanVideoVAEAttentionBlockConfig config_;
};

struct WanVideoVAEStageBlockWeights {
  WanVideoVAEResidualBlockWeights residual;
  std::optional<WanVideoVAEAttentionBlockWeights> attention;
};

struct WanVideoVAEStageWeights {
  std::vector<WanVideoVAEStageBlockWeights> blocks;
  std::optional<WanVideoVAEResampleWeights> resample;
};

struct WanVideoVAEEncoderWeights {
  engine::modules::Conv3dWeights conv_in;
  std::vector<WanVideoVAEStageWeights> stages;
  WanVideoVAEResidualBlockWeights middle1;
  WanVideoVAEAttentionBlockWeights middle_attention;
  WanVideoVAEResidualBlockWeights middle2;
  WanVideoRMSNorm3dWeights head_norm;
  engine::modules::Conv3dWeights head;
  engine::modules::Conv3dWeights quant_conv;
  engine::core::TensorValue latent_mean;
  engine::core::TensorValue latent_std;
};

class WanVideoVAEEncoderModule {
public:
  explicit WanVideoVAEEncoderModule(WanVideoVAEArchitectureConfig config);

  engine::core::TensorValue
  build_initial(engine::core::ModuleBuildContext &ctx,
                const engine::core::TensorValue &input,
                const WanVideoVAEEncoderWeights &weights) const;
  engine::core::TensorValue
  build_cached(engine::core::ModuleBuildContext &ctx,
               const engine::core::TensorValue &input,
               const WanVideoVAEEncoderWeights &weights,
               WanVideoVAECacheBuildState &cache) const;

private:
  WanVideoVAEArchitectureConfig config_;
};

struct WanVideoVAEDecoderWeights {
  engine::modules::Conv3dWeights quant_conv;
  engine::modules::Conv3dWeights conv_in;
  WanVideoVAEResidualBlockWeights middle1;
  WanVideoVAEAttentionBlockWeights middle_attention;
  WanVideoVAEResidualBlockWeights middle2;
  std::vector<WanVideoVAEStageWeights> stages;
  WanVideoRMSNorm3dWeights head_norm;
  engine::modules::Conv3dWeights head;
  engine::core::TensorValue latent_mean;
  engine::core::TensorValue latent_std;
};

class WanVideoVAEDecoderModule {
public:
  explicit WanVideoVAEDecoderModule(WanVideoVAEArchitectureConfig config);

  engine::core::TensorValue
  build_initial(engine::core::ModuleBuildContext &ctx,
                const engine::core::TensorValue &input,
                const WanVideoVAEDecoderWeights &weights) const;
  engine::core::TensorValue
  build_cached(engine::core::ModuleBuildContext &ctx,
               const engine::core::TensorValue &input,
               const WanVideoVAEDecoderWeights &weights,
               WanVideoVAECacheBuildState &cache) const;

private:
  WanVideoVAEArchitectureConfig config_;
};

} // namespace engine::codecs

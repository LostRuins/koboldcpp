#include "engine/models/voxcpm2/audiovae.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/streaming_audio.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::voxcpm2 {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
namespace assets_ns = engine::assets;

using Clock = std::chrono::steady_clock;

constexpr int64_t kResidualKernel = 7;

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

struct VAEConv1dWeights {
  modules::Conv1dWeights regular;
  modules::DepthwiseConv1dWeights depthwise;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t kernel_size = 0;
  bool depthwise_layout = false;
};

struct VAEConvTranspose1dWeights {
  modules::ConvTranspose1dWeights conv;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t kernel_size = 0;
};

struct VAESnakeWeights {
  core::TensorValue alpha;
};

struct VAESampleRateConditionWeights {
  core::TensorValue scale;
  core::TensorValue bias;
};

struct VAEResidualUnitWeights {
  VAESnakeWeights snake1;
  VAEConv1dWeights conv1;
  VAESnakeWeights snake2;
  VAEConv1dWeights conv2;
};

struct VAEDecoderBlockWeights {
  VAESampleRateConditionWeights sr_cond;
  VAESnakeWeights snake;
  VAEConvTranspose1dWeights upsample;
  std::vector<VAEResidualUnitWeights> residual_units;
  int64_t input_channels = 0;
  int64_t output_channels = 0;
  int stride = 1;
};

struct VAEEncoderBlockWeights {
  std::vector<VAEResidualUnitWeights> residual_units;
  VAESnakeWeights snake;
  VAEConv1dWeights downsample;
  int64_t input_channels = 0;
  int64_t output_channels = 0;
  int stride = 1;
};

struct VAEWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  VAEConv1dWeights encoder_first;
  std::vector<VAEEncoderBlockWeights> encoder_blocks;
  VAEConv1dWeights encoder_fc_mu;
  VAEConv1dWeights decoder_first_depthwise;
  VAEConv1dWeights decoder_first_pointwise;
  std::vector<VAEDecoderBlockWeights> decoder_blocks;
  VAESnakeWeights decoder_final_snake;
  VAEConv1dWeights decoder_final_conv;
};

int sample_rate_bucket(const VoxCPM2AudioVAEConfig &config) {
  int bucket = 0;
  while (bucket < static_cast<int>(config.sample_rate_bin_boundaries.size()) &&
         config.output_sample_rate >
             config.sample_rate_bin_boundaries[static_cast<size_t>(bucket)]) {
    ++bucket;
  }
  return bucket;
}

std::vector<float> fold_weight_norm(const std::vector<float> &weight_v,
                                    const std::vector<float> &weight_g,
                                    int64_t dim0, int64_t dim1,
                                    int64_t kernel) {
  if (static_cast<int64_t>(weight_v.size()) != dim0 * dim1 * kernel ||
      static_cast<int64_t>(weight_g.size()) != dim0) {
    throw std::runtime_error("VoxCPM2 AudioVAE weight-norm shape mismatch");
  }
  std::vector<float> out(weight_v.size(), 0.0F);
  for (int64_t d0 = 0; d0 < dim0; ++d0) {
    const size_t base = static_cast<size_t>(d0 * dim1 * kernel);
    double norm_sq = 0.0;
    for (int64_t i = 0; i < dim1 * kernel; ++i) {
      const double value = weight_v[base + static_cast<size_t>(i)];
      norm_sq += value * value;
    }
    const float scale = weight_g[static_cast<size_t>(d0)] /
                        static_cast<float>(std::sqrt(norm_sq));
    for (int64_t i = 0; i < dim1 * kernel; ++i) {
      out[base + static_cast<size_t>(i)] =
          weight_v[base + static_cast<size_t>(i)] * scale;
    }
  }
  return out;
}

std::vector<float> squeeze_weight_g(const std::vector<float> &values,
                                    int64_t channels) {
  if (static_cast<int64_t>(values.size()) != channels) {
    throw std::runtime_error("VoxCPM2 AudioVAE weight_g shape mismatch");
  }
  return values;
}

VAEConv1dWeights load_wn_conv1d(core::BackendWeightStore &store,
                                const assets_ns::TensorSource &source,
                                const std::string &prefix, int64_t out_channels,
                                int64_t in_channels, int64_t kernel_size,
                                bool depthwise,
                                assets_ns::TensorStorageType storage_type) {
  const int64_t stored_in = depthwise ? 1 : in_channels;
  const auto weight_v = source.require_f32(
      prefix + ".weight_v", {out_channels, stored_in, kernel_size});
  const auto weight_g = squeeze_weight_g(
      source.require_f32(prefix + ".weight_g", {out_channels, 1, 1}),
      out_channels);
  const auto folded = fold_weight_norm(weight_v, weight_g, out_channels,
                                       stored_in, kernel_size);
  VAEConv1dWeights out;
  out.in_channels = in_channels;
  out.out_channels = out_channels;
  out.kernel_size = kernel_size;
  out.depthwise_layout = depthwise;
  if (depthwise) {
    out.depthwise.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, 1, kernel_size}),
        storage_type, folded);
    out.depthwise.bias =
        store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  } else {
    out.regular.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
        storage_type, folded);
    out.regular.bias =
        store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  }
  return out;
}

VAEConvTranspose1dWeights
load_wn_conv_transpose1d(core::BackendWeightStore &store,
                         const assets_ns::TensorSource &source,
                         const std::string &prefix, int64_t in_channels,
                         int64_t out_channels, int64_t kernel_size,
                         assets_ns::TensorStorageType storage_type) {
  const auto weight_v = source.require_f32(
      prefix + ".weight_v", {in_channels, out_channels, kernel_size});
  const auto weight_g = squeeze_weight_g(
      source.require_f32(prefix + ".weight_g", {in_channels, 1, 1}),
      in_channels);
  VAEConvTranspose1dWeights out;
  out.in_channels = in_channels;
  out.out_channels = out_channels;
  out.kernel_size = kernel_size;
  out.conv.weight = store.make_from_f32(
      core::TensorShape::from_dims({in_channels, out_channels, kernel_size}),
      storage_type,
      fold_weight_norm(weight_v, weight_g, in_channels, out_channels,
                       kernel_size));
  out.conv.bias =
      store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  return out;
}

VAESnakeWeights load_snake(core::BackendWeightStore &store,
                           const assets_ns::TensorSource &source,
                           const std::string &name, int64_t channels) {
  VAESnakeWeights out;
  out.alpha = store.make_from_f32(core::TensorShape::from_dims({channels}),
                                  assets_ns::TensorStorageType::F32,
                                  source.require_f32(name, {1, channels, 1}));
  return out;
}

VAESampleRateConditionWeights load_sr_condition(
    core::BackendWeightStore &store, const assets_ns::TensorSource &source,
    const std::string &prefix, int64_t channels, int bucket, int buckets) {
  const auto scale =
      source.require_f32(prefix + ".scale_embed.weight", {buckets, channels});
  const auto bias =
      source.require_f32(prefix + ".bias_embed.weight", {buckets, channels});
  const auto offset = static_cast<std::ptrdiff_t>(bucket * channels);
  VAESampleRateConditionWeights out;
  out.scale = store.make_from_f32(
      core::TensorShape::from_dims({channels}),
      assets_ns::TensorStorageType::F32,
      std::vector<float>(scale.begin() + offset,
                         scale.begin() + offset + channels));
  out.bias =
      store.make_from_f32(core::TensorShape::from_dims({channels}),
                          assets_ns::TensorStorageType::F32,
                          std::vector<float>(bias.begin() + offset,
                                             bias.begin() + offset + channels));
  return out;
}

VAEResidualUnitWeights load_residual_unit(core::BackendWeightStore &store,
                                          const assets_ns::TensorSource &source,
                                          const std::string &prefix,
                                          int64_t channels,
                                          assets_ns::TensorStorageType storage_type) {
  VAEResidualUnitWeights out;
  out.snake1 = load_snake(store, source, prefix + ".block.0.alpha", channels);
  out.conv1 = load_wn_conv1d(store, source, prefix + ".block.1", channels,
                             channels, kResidualKernel, true, storage_type);
  out.snake2 = load_snake(store, source, prefix + ".block.2.alpha", channels);
  out.conv2 = load_wn_conv1d(store, source, prefix + ".block.3", channels,
                             channels, 1, false, storage_type);
  return out;
}

VAEEncoderBlockWeights load_encoder_block(core::BackendWeightStore &store,
                                          const assets_ns::TensorSource &source,
                                          const std::string &prefix,
                                          int64_t input_channels,
                                          int64_t output_channels, int stride,
                                          assets_ns::TensorStorageType storage_type) {
  VAEEncoderBlockWeights block;
  block.input_channels = input_channels;
  block.output_channels = output_channels;
  block.stride = stride;
  block.residual_units.push_back(
      load_residual_unit(store, source, prefix + ".block.0", input_channels,
                         storage_type));
  block.residual_units.push_back(
      load_residual_unit(store, source, prefix + ".block.1", input_channels,
                         storage_type));
  block.residual_units.push_back(
      load_residual_unit(store, source, prefix + ".block.2", input_channels,
                         storage_type));
  block.snake =
      load_snake(store, source, prefix + ".block.3.alpha", input_channels);
  block.downsample =
      load_wn_conv1d(store, source, prefix + ".block.4", output_channels,
                     input_channels, 2 * stride, false, storage_type);
  return block;
}

int64_t product(const std::vector<int64_t> &values) {
  int64_t out = 1;
  for (const int64_t value : values) {
    if (value <= 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE rate must be positive");
    }
    out *= value;
  }
  return out;
}

VAEWeights load_vae_weights(const VoxCPM2Assets &assets,
                            core::ExecutionContext &execution_context,
                            size_t weight_context_bytes,
                            assets_ns::TensorStorageType storage_type) {
  const auto &config = assets.config.audio_vae;
  const auto &source = *assets.audiovae_weights;
  VAEWeights weights;
  weights.store = std::make_shared<core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "voxcpm2.audiovae.weights", weight_context_bytes);
  auto &store = *weights.store;
  weights.encoder_first = load_wn_conv1d(store, source, "encoder.block.0",
                                         config.encoder_dim, 1, 7, false,
                                         storage_type);
  int64_t encoder_in_channels = config.encoder_dim;
  weights.encoder_blocks.reserve(config.encoder_rates.size());
  for (size_t i = 0; i < config.encoder_rates.size(); ++i) {
    const int64_t encoder_out_channels = encoder_in_channels * 2;
    weights.encoder_blocks.push_back(load_encoder_block(
        store, source, "encoder.block." + std::to_string(i + 1),
        encoder_in_channels, encoder_out_channels,
        static_cast<int>(config.encoder_rates[i]), storage_type));
    encoder_in_channels = encoder_out_channels;
  }
  weights.encoder_fc_mu =
      load_wn_conv1d(store, source, "encoder.fc_mu", config.latent_dim,
                     encoder_in_channels, 3, false, storage_type);

  weights.decoder_first_depthwise =
      load_wn_conv1d(store, source, "decoder.model.0", config.latent_dim,
                     config.latent_dim, 7, true, storage_type);
  weights.decoder_first_pointwise =
      load_wn_conv1d(store, source, "decoder.model.1", config.decoder_dim,
                     config.latent_dim, 1, false, storage_type);

  const int bucket = sample_rate_bucket(config);
  const int buckets =
      static_cast<int>(config.sample_rate_bin_boundaries.size()) + 1;
  weights.decoder_blocks.reserve(config.decoder_rates.size());
  for (size_t i = 0; i < config.decoder_rates.size(); ++i) {
    const int64_t input_channels =
        config.decoder_dim / (int64_t{1} << static_cast<int>(i));
    const int64_t output_channels =
        config.decoder_dim / (int64_t{1} << static_cast<int>(i + 1));
    const int model_index = static_cast<int>(i) + 2;
    const std::string prefix = "decoder.model." + std::to_string(model_index);
    VAEDecoderBlockWeights block;
    block.input_channels = input_channels;
    block.output_channels = output_channels;
    block.stride = static_cast<int>(config.decoder_rates[i]);
    block.sr_cond = load_sr_condition(
        store, source, "decoder.sr_cond_model." + std::to_string(model_index),
        input_channels, bucket, buckets);
    block.snake =
        load_snake(store, source, prefix + ".block.0.alpha", input_channels);
    block.upsample = load_wn_conv_transpose1d(
        store, source, prefix + ".block.1", input_channels, output_channels,
        2 * block.stride, storage_type);
    block.residual_units.push_back(load_residual_unit(
        store, source, prefix + ".block.2", output_channels, storage_type));
    block.residual_units.push_back(load_residual_unit(
        store, source, prefix + ".block.3", output_channels, storage_type));
    block.residual_units.push_back(load_residual_unit(
        store, source, prefix + ".block.4", output_channels, storage_type));
    weights.decoder_blocks.push_back(std::move(block));
  }

  const int64_t decoder_final_channels =
      config.decoder_dim /
      (int64_t{1} << static_cast<int>(config.decoder_rates.size()));
  weights.decoder_final_snake =
      load_snake(store, source,
                 "decoder.model." +
                     std::to_string(config.decoder_rates.size() + 2) + ".alpha",
                 decoder_final_channels);
  weights.decoder_final_conv = load_wn_conv1d(
      store, source,
      "decoder.model." + std::to_string(config.decoder_rates.size() + 3), 1,
      decoder_final_channels, 7, false, storage_type);
  store.upload();
  return weights;
}


std::shared_ptr<const VoxCPM2Assets>
require_assets(std::shared_ptr<const VoxCPM2Assets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM2 AudioVAE decoder requires assets");
  }
  return assets;
}

core::TensorValue zeros_like_prefix(core::ModuleBuildContext &ctx,
                                    const core::TensorValue &input,
                                    int64_t frames) {
  if (frames <= 0) {
    return {};
  }
  auto prefix =
      modules::RepeatModule(
          {core::TensorShape::from_dims(
              {input.shape.dims[0], input.shape.dims[1], frames})})
          .build(ctx, modules::SliceModule({2, 0, 1}).build(ctx, input));
  auto contiguous = core::ensure_backend_addressable_layout(ctx, prefix);
  return core::wrap_tensor(ggml_scale(ctx.ggml, contiguous.tensor, 0.0F),
                           prefix.shape, GGML_TYPE_F32);
}

core::TensorValue causal_pad_left(core::ModuleBuildContext &ctx,
                                  const core::TensorValue &input,
                                  int64_t frames) {
  if (frames <= 0) {
    return input;
  }
  return modules::ConcatModule({2}).build(
      ctx, zeros_like_prefix(ctx, input, frames), input);
}

// Snake: x + sin(alpha * x)^2 / (alpha + 1e-9), alpha per channel. alpha and its
// denominator are [1, C, 1] and broadcast inside ggml_mul / ggml_div instead of
// being repeated to the activation's full [1, C, T] size first: the same
// per-element operations in the same order (bit-identical output), without two
// full-size broadcast tensors and a full-size scale_bias per activation.
// The same Snake as one pass over the activation (CPU only): per row the same
// float operations in the same order as the ggml graph below - x * a, sinf,
// s * s, / (a * 1 + 1e-9), x + - so the output is bit-identical. Rows are split
// across the graph's threads.
void voxcpm2_snake_cpu(ggml_tensor *dst, const ggml_tensor *x,
                       const ggml_tensor *alpha, int ith, int nth, void *) {
  const int64_t channels = x->ne[1];
  const int64_t rows = channels * x->ne[2];
  for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
    const int64_t c = row % channels;
    const float a = static_cast<const float *>(alpha->data)[c];
    const float denom = a * 1.0F + 1.0e-9F;
    const auto *src = reinterpret_cast<const float *>(
        static_cast<const char *>(x->data) + c * x->nb[1] +
        (row / channels) * x->nb[2]);
    auto *out = reinterpret_cast<float *>(
        static_cast<char *>(dst->data) + c * dst->nb[1] +
        (row / channels) * dst->nb[2]);
    for (int64_t t = 0; t < x->ne[0]; ++t) {
      const float ax = src[t] * a;
      const float s = std::sin(ax);
      const float square = s * s;
      out[t] = src[t] + square / denom;
    }
  }
}

core::TensorValue snake_exact(core::ModuleBuildContext &ctx,
                              const core::TensorValue &input,
                              const VAESnakeWeights &weights,
                              int64_t channels) {
  const auto input_f32 = core::ensure_backend_addressable_layout(ctx, input);
  if (ctx.backend_type == core::BackendType::Cpu &&
      input_f32.tensor->type == GGML_TYPE_F32 &&
      weights.alpha.tensor->type == GGML_TYPE_F32 &&
      ggml_is_contiguous(input_f32.tensor) &&
      ggml_is_contiguous(weights.alpha.tensor) &&
      ggml_nelements(weights.alpha.tensor) == channels) {
    return core::wrap_tensor(
        ggml_map_custom2(ctx.ggml, input_f32.tensor, weights.alpha.tensor,
                         voxcpm2_snake_cpu, GGML_N_TASKS_MAX, nullptr),
        input.shape, GGML_TYPE_F32);
  }
  auto alpha = core::reshape_tensor(
      ctx, weights.alpha, core::TensorShape::from_dims({1, channels, 1}));
  auto ax =
      core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, alpha.tensor),
                        input.shape, GGML_TYPE_F32);
  auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input.shape,
                             GGML_TYPE_F32);
  auto s2 = core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor),
                              input.shape, GGML_TYPE_F32);
  auto denom =
      core::wrap_tensor(ggml_scale_bias(ctx.ggml, alpha.tensor, 1.0F, 1.0e-9F),
                        alpha.shape, GGML_TYPE_F32);
  auto frac = core::wrap_tensor(ggml_div(ctx.ggml, s2.tensor, denom.tensor),
                                input.shape, GGML_TYPE_F32);
  return core::wrap_tensor(ggml_add(ctx.ggml, input_f32.tensor, frac.tensor),
                           input.shape, GGML_TYPE_F32);
}

core::TensorValue apply_sr_condition(
    core::ModuleBuildContext &ctx, const core::TensorValue &input,
    const VAESampleRateConditionWeights &weights, int64_t channels) {
  const auto input_f32 = core::ensure_backend_addressable_layout(ctx, input);
  auto scale = core::reshape_tensor(
      ctx, weights.scale, core::TensorShape::from_dims({1, channels, 1}));
  auto bias = core::reshape_tensor(
      ctx, weights.bias, core::TensorShape::from_dims({1, channels, 1}));
  auto scaled =
      core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, scale.tensor),
                        input.shape, GGML_TYPE_F32);
  return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bias.tensor),
                           input.shape, GGML_TYPE_F32);
}

core::TensorValue causal_conv1d(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const VAEConv1dWeights &weights, int stride,
                                int padding, int dilation,
                                int output_padding = 0) {
  const int left_pad = 2 * padding - output_padding;
  if (left_pad < 0) {
    throw std::runtime_error(
        "VoxCPM2 AudioVAE causal convolution padding is invalid");
  }
  auto padded = causal_pad_left(ctx, input, left_pad);
  if (weights.depthwise_layout) {
    return modules::DepthwiseConv1dModule(
               {weights.out_channels, weights.kernel_size, stride, 0, dilation,
                weights.depthwise.bias.has_value()})
        .build(ctx, padded, weights.depthwise);
  }
  return modules::Conv1dModule({weights.in_channels, weights.out_channels,
                                weights.kernel_size, stride, 0, dilation,
                                weights.regular.bias.has_value()})
      .build(ctx, padded, weights.regular);
}

core::TensorValue
causal_conv_transpose1d(core::ModuleBuildContext &ctx,
                        const core::TensorValue &input,
                        const VAEConvTranspose1dWeights &weights, int stride) {
  auto full =
      modules::ConvTranspose1dModule({weights.in_channels, weights.out_channels,
                                      weights.kernel_size, stride, 0, 1,
                                      weights.conv.bias.has_value()})
          .build(ctx, input, weights.conv);
  const int64_t frames = input.shape.dims[2] * stride;
  auto view = ggml_view_3d(ctx.ggml, full.tensor, frames, weights.out_channels,
                           1, full.tensor->nb[1], full.tensor->nb[2], 0);
  return core::wrap_tensor(
      ggml_cont(ctx.ggml, view),
      core::TensorShape::from_dims({1, weights.out_channels, frames}),
      GGML_TYPE_F32);
}

core::TensorValue residual_unit(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const VAEResidualUnitWeights &weights,
                                int dilation) {
  const int padding = static_cast<int>(((kResidualKernel - 1) * dilation) / 2);
  auto hidden = snake_exact(ctx, input, weights.snake1, input.shape.dims[1]);
  hidden = causal_conv1d(ctx, hidden, weights.conv1, 1, padding, dilation);
  hidden = snake_exact(ctx, hidden, weights.snake2, input.shape.dims[1]);
  hidden = causal_conv1d(ctx, hidden, weights.conv2, 1, 0, 1);
  return modules::AddModule{}.build(ctx, input, hidden);
}

core::TensorValue encoder_block(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const VAEEncoderBlockWeights &weights) {
  auto hidden = residual_unit(ctx, input, weights.residual_units[0], 1);
  hidden = residual_unit(ctx, hidden, weights.residual_units[1], 3);
  hidden = residual_unit(ctx, hidden, weights.residual_units[2], 9);
  hidden = snake_exact(ctx, hidden, weights.snake, weights.input_channels);
  const int padding = static_cast<int>((weights.stride + 1) / 2);
  const int output_padding = weights.stride % 2;
  return causal_conv1d(ctx, hidden, weights.downsample, weights.stride, padding,
                       1, output_padding);
}

core::TensorValue decoder_block(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const VAEDecoderBlockWeights &weights) {
  auto hidden =
      apply_sr_condition(ctx, input, weights.sr_cond, weights.input_channels);
  hidden = snake_exact(ctx, hidden, weights.snake, weights.input_channels);
  hidden =
      causal_conv_transpose1d(ctx, hidden, weights.upsample, weights.stride);
  hidden = residual_unit(ctx, hidden, weights.residual_units[0], 1);
  hidden = residual_unit(ctx, hidden, weights.residual_units[1], 3);
  hidden = residual_unit(ctx, hidden, weights.residual_units[2], 9);
  return hidden;
}

} // namespace

class VoxCPM2AudioVAEDecoderRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM2Assets> assets,
       core::ExecutionContext &execution_context,
       VoxCPM2AudioVAEDecoderConfig config)
      : assets_(require_assets(std::move(assets))),
        execution_context_(execution_context), config_(config),
        weights_(load_vae_weights(*assets_, execution_context_,
                                  config_.weight_context_bytes,
                                  config_.weight_storage_type)) {
    if (config_.latent_frame_capacity < 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE latent frame capacity must be non-negative");
    }
    if (config_.encoder_sample_capacity <= 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE encoder sample capacity must be positive");
    }
    if (config_.tile_frames < 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE tile frames must be non-negative");
    }
  }

  ~Impl() {
    release_decoder_graph();
    release_encoder_graph();
  }

  runtime::AudioBuffer decode_features(const std::vector<float> &features,
                                       int64_t patches) {
    const auto &vae = assets_->config.audio_vae;
    if (patches < 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE patch count is negative");
    }
    const int64_t latent_frames = patches * assets_->config.patch_size;
    const int64_t expected = latent_frames * vae.latent_dim;
    if (static_cast<int64_t>(features.size()) != expected) {
      throw std::runtime_error("VoxCPM2 AudioVAE feature size mismatch");
    }
    if (config_.tile_frames > 0 && latent_frames > config_.tile_frames) {
      // Back-propagate the causal receptive field through the actual strides.
      // Walking backward: residual convolutions, then inverse upsampling.
      int64_t history = weights_.decoder_final_conv.kernel_size - 1;
      for (auto it = weights_.decoder_blocks.rbegin();
           it != weights_.decoder_blocks.rend(); ++it) {
        history += (kResidualKernel - 1) * (1 + 3 + 9);
        history = (history + it->upsample.kernel_size - 1 + it->stride - 1) /
                  it->stride;
      }
      history += weights_.decoder_first_depthwise.kernel_size - 1;
      runtime::StreamingAudioConfig tiling;
      tiling.policy = runtime::StreamingAudioChunkPolicy::Fixed;
      tiling.frames_per_chunk = static_cast<size_t>(config_.tile_frames);
      tiling.frame_width = static_cast<size_t>(vae.latent_dim);
      tiling.left_context_frames = static_cast<size_t>(history);
      tiled_audio_.begin(tiling, [this, history](const auto &window) {
        auto audio = decode_window(window.data, static_cast<int64_t>(window.frames()),
                                   config_.tile_frames + history);
        audio.samples.erase(audio.samples.begin(),
                            audio.samples.begin() + window.left_frames * decoder_stride_);
        return audio;
      });
      tiled_audio_.push(features);
      return tiled_audio_.finish();
    }
    return decode_window(features.data(), latent_frames, latent_frames);
  }

private:
  runtime::AudioBuffer decode_window(const float *features, int64_t latent_frames,
                                    int64_t capacity) {
    const auto &vae = assets_->config.audio_vae;
    ensure_decoder_graph(capacity);
    std::vector<float> input(
        static_cast<size_t>(vae.latent_dim * decoder_latent_frame_capacity_),
        0.0F);
    for (int64_t t = 0; t < latent_frames; ++t) {
      for (int64_t c = 0; c < vae.latent_dim; ++c) {
        input[static_cast<size_t>(c * decoder_latent_frame_capacity_ + t)] =
            features[static_cast<size_t>(t * vae.latent_dim + c)];
      }
    }
    ggml_backend_tensor_set(input_, input.data(), 0,
                            input.size() * sizeof(float));
    core::set_backend_threads(execution_context_.backend(),
                              std::max(1, execution_context_.config().threads));
    const ggml_status status =
        core::compute_backend_graph(execution_context_.backend(), graph_);
    ggml_backend_synchronize(execution_context_.backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM2 AudioVAE decoder graph compute failed");
    }
    const int64_t sample_count = latent_frames * decoder_stride_;
    std::vector<float> full(static_cast<size_t>(output_frames_), 0.0F);
    ggml_backend_tensor_get(output_, full.data(), 0,
                            full.size() * sizeof(float));
    runtime::AudioBuffer audio;
    audio.sample_rate = vae.output_sample_rate;
    audio.channels = 1;
    audio.samples.assign(
        full.begin(), full.begin() + static_cast<std::ptrdiff_t>(sample_count));
    return audio;
  }

public:
  VoxCPM2EncodedPrompt encode_prompt_audio(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio) {
    VoxCPM2EncodedPrompt out;
    if (prompt_audio.has_value()) {
      if (prompt_text.empty()) {
        throw std::runtime_error(
            "VoxCPM2 prompt audio requires prompt_text or reference_text");
      }
      out.prompt_text = prompt_text;
      auto encoded = encode_audio(*prompt_audio, true);
      out.prompt_features = std::move(encoded.features);
      out.prompt_patches = encoded.patches;
    }
    if (reference_audio.has_value()) {
      auto encoded = encode_audio(*reference_audio, false);
      out.reference_features = std::move(encoded.features);
      out.reference_patches = encoded.patches;
    }
    return out;
  }

  void release_runtime_memory() {
    release_decoder_graph();
    release_encoder_graph();
  }

private:
  struct EncodedFeatures {
    std::vector<float> features;
    int64_t patches = 0;
  };

  EncodedFeatures encode_audio(const runtime::AudioBuffer &audio,
                               bool left_pad) {
    ensure_encoder_graph();
    const auto &vae = assets_->config.audio_vae;
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(
        audio.samples, audio.channels);
    if (audio.sample_rate != vae.sample_rate) {
      engine::audio::SoxrResampleOptions options;
      options.profile =
          engine::audio::SoxrResampleProfile::ExplicitFloat32Runtime;
      options.output_length_policy =
          engine::audio::SoxrOutputLengthPolicy::ExactExpected;
      options.output_padding = 256;
      options.require_full_input = true;
      options.reject_empty_output = true;
      options.warning_context = "VoxCPM2 AudioVAE encoder";
      options.fallback_description = "linear resampling";
      mono = engine::audio::resample_mono_soxr_or_linear(
          mono, audio.sample_rate, vae.sample_rate, options);
    }
    const int64_t patch_samples = assets_->config.patch_size * encoder_stride_;
    if (patch_samples <= 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE patch sample size is invalid");
    }
    const int64_t sample_count = static_cast<int64_t>(mono.size());
    const int64_t padded_samples =
        ((sample_count + patch_samples - 1) / patch_samples) * patch_samples;
    if (padded_samples > config_.encoder_sample_capacity) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE encoder sample capacity exceeded");
    }
    std::vector<float> input(
        static_cast<size_t>(config_.encoder_sample_capacity), 0.0F);
    const int64_t offset = left_pad ? padded_samples - sample_count : 0;
    std::copy(mono.begin(), mono.end(),
              input.begin() + static_cast<std::ptrdiff_t>(offset));
    ggml_backend_tensor_set(encoder_input_, input.data(), 0,
                            input.size() * sizeof(float));
    core::set_backend_threads(execution_context_.backend(),
                              std::max(1, execution_context_.config().threads));
    const ggml_status status = core::compute_backend_graph(
        execution_context_.backend(), encoder_graph_);
    ggml_backend_synchronize(execution_context_.backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM2 AudioVAE encoder graph compute failed");
    }

    const int64_t latent_frames = padded_samples / encoder_stride_;
    const int64_t expected_capacity_frames =
        config_.encoder_sample_capacity / encoder_stride_;
    std::vector<float> full(
        static_cast<size_t>(vae.latent_dim * expected_capacity_frames), 0.0F);
    ggml_backend_tensor_get(encoder_output_, full.data(), 0,
                            full.size() * sizeof(float));
    if (latent_frames % assets_->config.patch_size != 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE encoded frames are not divisible by patch size");
    }
    EncodedFeatures encoded;
    encoded.patches = latent_frames / assets_->config.patch_size;
    encoded.features.resize(static_cast<size_t>(latent_frames * vae.latent_dim),
                            0.0F);
    for (int64_t t = 0; t < latent_frames; ++t) {
      for (int64_t c = 0; c < vae.latent_dim; ++c) {
        encoded.features[static_cast<size_t>(t * vae.latent_dim + c)] =
            full[static_cast<size_t>(c * expected_capacity_frames + t)];
      }
    }
    return encoded;
  }

  void ensure_encoder_graph() {
    if (encoder_graph_ != nullptr) {
      return;
    }
    build_encoder();
  }

  // With voxcpm2.audiovae_latent_capacity set, the decoder graph is a
  // reservation: it starts at that size, doubles until the request fits and is
  // reused by every shorter request. Without it, the graph is built at the
  // request's own frame count. The decoder is causal (causal_conv1d and
  // causal_conv_transpose1d only), so the zero frames a larger graph pads with
  // never change the kept samples; they only cost compute, and with the
  // power-of-two growth a request decodes up to twice its length (140 frames
  // ran a 256-frame graph). Rebuilding costs a few milliseconds and the
  // allocator below keeps its buffer, so exact sizing is the default.
  bool exact_decoder_capacity() const {
    return config_.latent_frame_capacity <= 0;
  }

  int64_t decoder_capacity_for(int64_t latent_frames) const {
    if (exact_decoder_capacity()) {
      const int64_t patch = std::max<int64_t>(1, assets_->config.patch_size);
      return std::max<int64_t>(patch, ((latent_frames + patch - 1) / patch) * patch);
    }
    const int64_t min_capacity =
        config_.latent_frame_capacity > 0 ? config_.latent_frame_capacity
                                          : assets_->config.patch_size;
    int64_t capacity = std::max<int64_t>(min_capacity, assets_->config.patch_size);
    while (capacity < latent_frames) {
      capacity *= 2;
    }
    return capacity;
  }

  void ensure_decoder_graph(int64_t latent_frames) {
    const bool reusable =
        exact_decoder_capacity()
            ? decoder_latent_frame_capacity_ == decoder_capacity_for(latent_frames)
            : latent_frames <= decoder_latent_frame_capacity_;
    if (graph_ != nullptr && reusable) {
      engine::debug::timing_log_scalar(
          "voxcpm2.audiovae.decoder.graph.rebuilt", false);
      engine::debug::timing_log_scalar(
          "voxcpm2.audiovae.decoder.graph.reused", true);
      engine::debug::timing_log_scalar(
          "voxcpm2.audiovae.decoder.graph.build_ms", 0.0);
      engine::debug::timing_log_scalar(
          "voxcpm2.audiovae.decoder.latent_capacity",
          decoder_latent_frame_capacity_);
      return;
    }
    const auto build_start = Clock::now();
    build_decoder(decoder_capacity_for(latent_frames));
    engine::debug::timing_log_scalar(
        "voxcpm2.audiovae.decoder.graph.rebuilt", true);
    engine::debug::timing_log_scalar(
        "voxcpm2.audiovae.decoder.graph.reused", false);
    engine::debug::timing_log_scalar(
        "voxcpm2.audiovae.decoder.graph.build_ms",
        engine::debug::elapsed_ms(build_start));
    engine::debug::timing_log_scalar(
        "voxcpm2.audiovae.decoder.latent_capacity",
        decoder_latent_frame_capacity_);
  }

  void release_decoder_graph(bool keep_allocator = false) {
    if (graph_ != nullptr) {
      core::release_backend_graph_resources(execution_context_.backend(), graph_);
    }
    if (gallocr_ != nullptr && !keep_allocator) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    input_ = nullptr;
    output_ = nullptr;
    ctx_.reset();
    output_frames_ = 0;
    decoder_latent_frame_capacity_ = 0;
  }

  void build_decoder(int64_t latent_frame_capacity) {
    const auto &vae = assets_->config.audio_vae;
    if (latent_frame_capacity <= 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE decoder graph capacity must be positive");
    }
    // Keep the allocator across size changes: ggml_gallocr_reserve only
    // reallocates its buffer when the new graph needs more than it holds.
    release_decoder_graph(/*keep_allocator=*/true);
    if (config_.graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE graph context bytes must be non-zero");
    }
    decoder_stride_ = product(vae.decoder_rates);
    output_frames_ = latent_frame_capacity * decoder_stride_;
    ggml_init_params params{config_.graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM2 AudioVAE decoder graph context");
    }
    core::ModuleBuildContext ctx{ctx_.get(), "voxcpm2.audiovae.decoder",
                                 execution_context_.backend_type()};
    auto hidden = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims(
            {1, vae.latent_dim, latent_frame_capacity}));
    input_ = hidden.tensor;
    ggml_set_input(input_);
    hidden =
        causal_conv1d(ctx, hidden, weights_.decoder_first_depthwise, 1, 3, 1);
    hidden =
        causal_conv1d(ctx, hidden, weights_.decoder_first_pointwise, 1, 0, 1);
    for (const auto &block : weights_.decoder_blocks) {
      hidden = decoder_block(ctx, hidden, block);
    }
    hidden = snake_exact(ctx, hidden, weights_.decoder_final_snake,
                         hidden.shape.dims[1]);
    hidden = causal_conv1d(ctx, hidden, weights_.decoder_final_conv, 1, 3, 1);
    hidden = modules::TanhModule{}.build(ctx, hidden);
    output_ = hidden.tensor;
    ggml_set_output(output_);
    graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
    ggml_build_forward_expand(graph_, output_);
    if (gallocr_ == nullptr) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(execution_context_.backend()));
    }
    if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
        !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
      release_decoder_graph();
      throw std::runtime_error("failed to allocate VoxCPM2 AudioVAE graph");
    }
    decoder_latent_frame_capacity_ = latent_frame_capacity;
    debug::timing_log_scalar("voxcpm2.audiovae.decoder.workspace_mb",
                            ggml_gallocr_get_buffer_size(gallocr_, 0) /
                                (1024.0 * 1024.0));
  }

  void release_encoder_graph() {
    if (encoder_graph_ != nullptr) {
      core::release_backend_graph_resources(execution_context_.backend(),
                                            encoder_graph_);
    }
    if (encoder_gallocr_ != nullptr) {
      ggml_gallocr_free(encoder_gallocr_);
      encoder_gallocr_ = nullptr;
    }
    encoder_graph_ = nullptr;
    encoder_input_ = nullptr;
    encoder_output_ = nullptr;
    encoder_ctx_.reset();
  }

  void build_encoder() {
    const auto &vae = assets_->config.audio_vae;
    release_encoder_graph();
    if (config_.encoder_graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM2 AudioVAE encoder graph context bytes must be non-zero");
    }
    encoder_stride_ = product(vae.encoder_rates);
    if (config_.encoder_sample_capacity % encoder_stride_ != 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE encoder sample capacity must "
                               "be divisible by encoder stride");
    }
    ggml_init_params params{config_.encoder_graph_context_bytes, nullptr, true};
    encoder_ctx_.reset(ggml_init(params));
    if (encoder_ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM2 AudioVAE encoder graph context");
    }
    core::ModuleBuildContext ctx{encoder_ctx_.get(), "voxcpm2.audiovae.encoder",
                                 execution_context_.backend_type()};
    auto hidden = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, config_.encoder_sample_capacity}));
    encoder_input_ = hidden.tensor;
    ggml_set_input(encoder_input_);
    hidden = causal_conv1d(ctx, hidden, weights_.encoder_first, 1, 3, 1);
    for (const auto &block : weights_.encoder_blocks) {
      hidden = encoder_block(ctx, hidden, block);
    }
    hidden = causal_conv1d(ctx, hidden, weights_.encoder_fc_mu, 1, 1, 1);
    encoder_output_ = hidden.tensor;
    ggml_set_output(encoder_output_);
    encoder_graph_ = ggml_new_graph_custom(encoder_ctx_.get(), 65536, false);
    ggml_build_forward_expand(encoder_graph_, encoder_output_);
    encoder_gallocr_ = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_.backend()));
    if (encoder_gallocr_ == nullptr ||
        !ggml_gallocr_reserve(encoder_gallocr_, encoder_graph_) ||
        !ggml_gallocr_alloc_graph(encoder_gallocr_, encoder_graph_)) {
      release_encoder_graph();
      throw std::runtime_error(
          "failed to allocate VoxCPM2 AudioVAE encoder graph");
    }
  }

  std::shared_ptr<const VoxCPM2Assets> assets_;
  core::ExecutionContext &execution_context_;
  VoxCPM2AudioVAEDecoderConfig config_;
  VAEWeights weights_;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  std::unique_ptr<ggml_context, GgmlContextDeleter> encoder_ctx_;
  ggml_tensor *input_ = nullptr;
  ggml_tensor *output_ = nullptr;
  ggml_tensor *encoder_input_ = nullptr;
  ggml_tensor *encoder_output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_cgraph *encoder_graph_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
  ggml_gallocr_t encoder_gallocr_ = nullptr;
  int64_t decoder_stride_ = 0;
  int64_t encoder_stride_ = 0;
  int64_t output_frames_ = 0;
  int64_t decoder_latent_frame_capacity_ = 0;
  runtime::StreamingAudioController<float> tiled_audio_;
};

VoxCPM2AudioVAEDecoderRuntime::VoxCPM2AudioVAEDecoderRuntime(
    std::shared_ptr<const VoxCPM2Assets> assets,
    core::ExecutionContext &execution_context,
    VoxCPM2AudioVAEDecoderConfig config)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   std::move(config))) {}

VoxCPM2AudioVAEDecoderRuntime::~VoxCPM2AudioVAEDecoderRuntime() = default;

runtime::AudioBuffer VoxCPM2AudioVAEDecoderRuntime::decode_features(
    const std::vector<float> &features, int64_t patches) {
  return impl_->decode_features(features, patches);
}

VoxCPM2EncodedPrompt VoxCPM2AudioVAEDecoderRuntime::encode_prompt_audio(
    const std::optional<runtime::AudioBuffer> &prompt_audio,
    const std::string &prompt_text,
    const std::optional<runtime::AudioBuffer> &reference_audio) {
  return impl_->encode_prompt_audio(prompt_audio, prompt_text, reference_audio);
}

void VoxCPM2AudioVAEDecoderRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

} // namespace engine::models::voxcpm2


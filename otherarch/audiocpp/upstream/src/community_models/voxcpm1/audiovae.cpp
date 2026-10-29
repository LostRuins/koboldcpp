#include "engine/community_models/voxcpm1/audiovae.h"

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

namespace engine::community_models::voxcpm1 {
namespace core = engine::core;
namespace modules = engine::modules;
namespace assets_ns = engine::assets;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kResidualKernel = 7;

enum class PaddingMode { Left, Right };

// ============================================================================
// AudioVAEStreamingDecodeState Implementation
// ============================================================================

} // namespace

AudioVAEStreamingDecodeState::AudioVAEStreamingDecodeState(AudioVAEStreamingDecodeState&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      buffer_(other.buffer_),
      slots_(std::move(other.slots_)),
      pending_updates_(std::move(other.pending_updates_)),
      cursor_(other.cursor_) {
  other.buffer_ = nullptr;
  other.cursor_ = 0;
}

AudioVAEStreamingDecodeState& AudioVAEStreamingDecodeState::operator=(AudioVAEStreamingDecodeState&& other) noexcept {
  if (this != &other) {
    reset();
    ctx_ = std::move(other.ctx_);
    buffer_ = other.buffer_;
    slots_ = std::move(other.slots_);
    pending_updates_ = std::move(other.pending_updates_);
    cursor_ = other.cursor_;

    other.buffer_ = nullptr;
    other.cursor_ = 0;
  }
  return *this;
}

AudioVAEStreamingDecodeState::~AudioVAEStreamingDecodeState() {
  reset();
}

void AudioVAEStreamingDecodeState::reset() {
  pending_updates_.clear();
  slots_.clear();
  cursor_ = 0;
  if (buffer_) {
    ggml_backend_buffer_free(buffer_);
    buffer_ = nullptr;
  }
  if (ctx_) {
    ctx_.reset();
  }
}

void AudioVAEStreamingDecodeState::clear() {
  pending_updates_.clear();
  cursor_ = 0;
  if (buffer_) {
    ggml_backend_buffer_clear(buffer_, 0);
  }
}

bool AudioVAEStreamingDecodeState::initialize(const std::vector<SlotSpec>& specs, core::ExecutionContext& execution_context) {
  reset();
  if (specs.empty()) {
    return false;
  }

  // Create ggml context with no_alloc for state tensors
  ggml_init_params params = {};
  params.mem_size = 1024 * 1024;  // 1MB metadata buffer
  params.mem_buffer = nullptr;
  params.no_alloc = true;
  ctx_.reset(ggml_init(params));
  if (!ctx_) {
    return false;
  }

  slots_.reserve(specs.size());

  for (const SlotSpec& spec : specs) {
    if (spec.frames <= 0 || spec.channels <= 0) {
      reset();
      return false;
    }

    ggml_tensor* tensor = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, spec.frames, spec.channels, 1);
    if (!tensor) {
      reset();
      return false;
    }
    const std::string tensor_name = "audio_vae.streaming_state." + spec.name;
    ggml_set_name(tensor, tensor_name.c_str());
    slots_.push_back(Slot{spec.frames, spec.channels, tensor, spec.name});
  }

  buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_context.backend());
  if (!buffer_) {
    reset();
    return false;
  }
  ggml_backend_buffer_clear(buffer_, 0);
  return true;
}

void AudioVAEStreamingDecodeState::begin_graph() {
  pending_updates_.clear();
  cursor_ = 0;
}

ggml_tensor* AudioVAEStreamingDecodeState::take_slot(int64_t frames, int64_t channels, const std::string& name) {
  if (cursor_ >= slots_.size()) {
    return nullptr;
  }
  Slot& slot = slots_[cursor_];
  if (slot.frames != frames || slot.channels != channels || slot.name != name) {
    return nullptr;
  }
  ++cursor_;
  return slot.tensor;
}

void AudioVAEStreamingDecodeState::queue_update(ggml_tensor* tensor) {
  if (!tensor || cursor_ == 0) {
    return;
  }
  pending_updates_.push_back(PendingUpdate{cursor_ - 1, tensor});
}

void AudioVAEStreamingDecodeState::build_update_graph(ggml_cgraph* graph) const {
  if (!graph) {
    return;
  }
  for (const PendingUpdate& update : pending_updates_) {
    if (!update.tensor) continue;
    ggml_set_output(update.tensor);
    ggml_build_forward_expand(graph, update.tensor);
  }
}

void AudioVAEStreamingDecodeState::publish_updates(core::ExecutionContext& execution_context) {
  (void)execution_context;
  for (const PendingUpdate& update : pending_updates_) {
    if (update.slot_index >= slots_.size()) continue;
    ggml_tensor* dst = slots_[update.slot_index].tensor;
    if (!dst || !update.tensor) continue;
    if (ggml_nbytes(update.tensor) != ggml_nbytes(dst)) continue;
    ggml_backend_tensor_copy(update.tensor, dst);
  }
}

namespace {

std::vector<float> trim_audio_silence_vad(const std::vector<float>& input,
                                          int sample_rate,
                                          float max_silence_ms = 100.0f,
                                          float top_db = 30.0f) {
    if (input.empty() || sample_rate <= 0) {
        return input;
    }

    constexpr int kFrameLength = 2048;
    constexpr int kHopLength = 512;
    const float ref = *std::max_element(input.begin(), input.end(), [](float a, float b) {
        return std::fabs(a) < std::fabs(b);
    });
    if (std::fabs(ref) <= 0.0f) {
        return input;
    }

    const float threshold = std::fabs(ref) * std::pow(10.0f, -top_db / 20.0f);
    const size_t n = input.size();
    int first_voice_frame = -1;
    int last_voice_frame = -1;

    for (size_t idx = 0, frame = 0; idx < n; idx += kHopLength, ++frame) {
        const size_t frame_end = std::min(idx + static_cast<size_t>(kFrameLength), n);
        const size_t frame_size = frame_end - idx;
        if (frame_size == 0) {
            break;
        }
        double energy = 0.0;
        for (size_t i = idx; i < frame_end; ++i) {
            energy += static_cast<double>(input[i]) * static_cast<double>(input[i]);
        }
        const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(frame_size)));
        if (rms >= threshold) {
            if (first_voice_frame < 0) {
                first_voice_frame = static_cast<int>(frame);
            }
            last_voice_frame = static_cast<int>(frame);
        }
        if (frame_end == n) {
            break;
        }
    }

    if (first_voice_frame < 0 || last_voice_frame < 0) {
        return input;
    }

    const int max_silence_samples = std::max(0, static_cast<int>(std::lround(max_silence_ms * sample_rate / 1000.0f)));
    const int start = std::max(0, first_voice_frame * kHopLength - max_silence_samples);
    const int end = std::min(static_cast<int>(n),
                             (last_voice_frame + 1) * kHopLength + (kFrameLength - kHopLength) + max_silence_samples);
    if (start >= end) {
        return input;
    }
    return std::vector<float>(input.begin() + start, input.begin() + end);
}

void pad_audio_for_patch_alignment(std::vector<float>& audio, size_t patch_len, PaddingMode mode) {
    if (patch_len == 0 || audio.empty() || (audio.size() % patch_len) == 0) {
        return;
    }
    const size_t padding = patch_len - (audio.size() % patch_len);
    if (mode == PaddingMode::Left) {
        audio.insert(audio.begin(), padding, 0.0f);
    } else {
        audio.insert(audio.end(), padding, 0.0f);
    }
}

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

int sample_rate_bucket(const VoxCPM1AudioVAEConfig &config) {
  int bucket = 0;
  while (bucket < static_cast<int>(config.sample_rate_bin_boundaries.size()) &&
         config.output_sample_rate >
             config.sample_rate_bin_boundaries[static_cast<size_t>(bucket)]) {
    ++bucket;
  }
  return bucket;
}

VAEConv1dWeights load_wn_conv1d(core::BackendWeightStore &store,
                                const assets_ns::TensorSource &source,
                                const std::string &prefix, int64_t out_channels,
                                int64_t in_channels, int64_t kernel_size,
                                bool depthwise,
                                assets_ns::TensorStorageType storage_type) {
  const int64_t stored_in = depthwise ? 1 : in_channels;
  const auto folded = source.require_f32(
      prefix + ".weight", {out_channels, stored_in, kernel_size});
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
  const auto folded = source.require_f32(
      prefix + ".weight", {in_channels, out_channels, kernel_size});
  VAEConvTranspose1dWeights out;
  out.in_channels = in_channels;
  out.out_channels = out_channels;
  out.kernel_size = kernel_size;
  out.conv.weight = store.make_from_f32(
      core::TensorShape::from_dims({in_channels, out_channels, kernel_size}),
      storage_type, folded);
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
      throw std::runtime_error("VoxCPM1 AudioVAE rate must be positive");
    }
    out *= value;
  }
  return out;
}

VAEWeights load_vae_weights(const VoxCPM1Assets &assets,
                            core::ExecutionContext &execution_context,
                            size_t weight_context_bytes,
                            assets_ns::TensorStorageType storage_type) {
  const auto &config = assets.config.audio_vae;
  const auto &source = *assets.audiovae_weights;
  VAEWeights weights;
  weights.store = std::make_shared<core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "voxcpm1.audiovae.weights", weight_context_bytes);
  auto &store = *weights.store;
  weights.encoder_first = load_wn_conv1d(store, source, "audio_vae.encoder.block.0",
                                         config.encoder_dim, 1, 7, false,
                                         storage_type);
  int64_t encoder_in_channels = config.encoder_dim;
  weights.encoder_blocks.reserve(config.encoder_rates.size());
  for (size_t i = 0; i < config.encoder_rates.size(); ++i) {
    const int64_t encoder_out_channels = encoder_in_channels * 2;
    weights.encoder_blocks.push_back(load_encoder_block(
        store, source, "audio_vae.encoder.block." + std::to_string(i + 1),
        encoder_in_channels, encoder_out_channels,
        static_cast<int>(config.encoder_rates[i]), storage_type));
    encoder_in_channels = encoder_out_channels;
  }
  weights.encoder_fc_mu =
      load_wn_conv1d(store, source, "audio_vae.encoder.fc_mu", config.latent_dim,
                     encoder_in_channels, 3, false, storage_type);

  weights.decoder_first_depthwise =
      load_wn_conv1d(store, source, "audio_vae.decoder.model.0", config.latent_dim,
                     config.latent_dim, 7, true, storage_type);
  weights.decoder_first_pointwise =
      load_wn_conv1d(store, source, "audio_vae.decoder.model.1", config.decoder_dim,
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
    const std::string prefix = "audio_vae.decoder.model." + std::to_string(model_index);
    VAEDecoderBlockWeights block;
    block.input_channels = input_channels;
    block.output_channels = output_channels;
    block.stride = static_cast<int>(config.decoder_rates[i]);
    block.sr_cond = load_sr_condition(
        store, source, "audio_vae.decoder.sr_cond_model." + std::to_string(model_index),
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
                 "audio_vae.decoder.model." +
                     std::to_string(config.decoder_rates.size() + 2) + ".alpha",
                 decoder_final_channels);
  weights.decoder_final_conv = load_wn_conv1d(
      store, source,
      "audio_vae.decoder.model." + std::to_string(config.decoder_rates.size() + 3), 1,
      decoder_final_channels, 7, false, storage_type);
  store.upload();
  return weights;
}


std::shared_ptr<const VoxCPM1Assets>
require_assets(std::shared_ptr<const VoxCPM1Assets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM1 AudioVAE decoder requires assets");
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

core::TensorValue snake_exact(core::ModuleBuildContext &ctx,
                              const core::TensorValue &input,
                              const VAESnakeWeights &weights,
                              int64_t channels) {
  const auto input_f32 = core::ensure_backend_addressable_layout(ctx, input);
  auto alpha = core::reshape_tensor(
      ctx, weights.alpha, core::TensorShape::from_dims({1, channels, 1}));
  alpha =
      core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, input_f32.tensor),
                        input.shape, GGML_TYPE_F32);
  auto ax =
      core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, alpha.tensor),
                        input.shape, GGML_TYPE_F32);
  auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input.shape,
                             GGML_TYPE_F32);
  auto s2 = core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor),
                              input.shape, GGML_TYPE_F32);
  auto denom =
      core::wrap_tensor(ggml_scale_bias(ctx.ggml, alpha.tensor, 1.0F, 1.0e-9F),
                        input.shape, GGML_TYPE_F32);
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
  scale =
      core::wrap_tensor(ggml_repeat(ctx.ggml, scale.tensor, input_f32.tensor),
                        input.shape, GGML_TYPE_F32);
  auto bias = core::reshape_tensor(
      ctx, weights.bias, core::TensorShape::from_dims({1, channels, 1}));
  bias = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, input_f32.tensor),
                           input.shape, GGML_TYPE_F32);
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
        "VoxCPM1 AudioVAE causal convolution padding is invalid");
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
  return causal_conv1d(ctx, hidden, weights.downsample, weights.stride, padding,
                       1);
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

// ============================================================================
// Stateful Convolution Methods for Streaming Decode
// ============================================================================

core::TensorValue causal_conv1d_stateful(core::ModuleBuildContext &ctx,
                                         const core::TensorValue &input,
                                         const VAEConv1dWeights &weights,
                                         int stride, int padding, int dilation,
                                         int output_padding,
                                         AudioVAEStreamingDecodeState &state,
                                         const std::string &state_name) {
  const int left_pad = 2 * padding - output_padding;
  if (left_pad < 0) {
    throw std::runtime_error(
        "VoxCPM1 AudioVAE causal convolution padding is invalid");
  }
  const int state_frames = left_pad;
  if (state_frames <= 0) {
    return causal_conv1d(ctx, input, weights, stride, padding, dilation,
                         output_padding);
  }

  ggml_tensor* prev = state.take_slot(state_frames, input.shape.dims[1],
                                      state_name);
  if (!prev) {
    throw std::runtime_error(
        "VoxCPM1 AudioVAE streaming state slot not found: " + state_name);
  }
  // State has [state_frames, channels, 1], need to wrap with correct shape [1, channels, state_frames]
  auto prev_val = core::wrap_tensor(
      prev, core::TensorShape::from_dims({1, input.shape.dims[1], state_frames}),
      GGML_TYPE_F32);
  auto padded = modules::ConcatModule({2}).build(ctx, prev_val, input);

  core::TensorValue result;
  if (weights.depthwise_layout) {
    result = modules::DepthwiseConv1dModule(
                 {weights.out_channels, weights.kernel_size, stride, 0, dilation,
                  weights.depthwise.bias.has_value()})
             .build(ctx, padded, weights.depthwise);
  } else {
    result = modules::Conv1dModule(
                 {weights.in_channels, weights.out_channels, weights.kernel_size,
                  stride, 0, dilation, weights.regular.bias.has_value()})
             .build(ctx, padded, weights.regular);
  }

  // Extract next state from the end of padded input
  const int64_t padded_frames = padded.shape.dims[2];
  const int64_t state_offset = (padded_frames - state_frames) * padded.tensor->nb[0];
  
  // View: extract last state_frames frames from padded input
  // Physical layout of padded.tensor is column-major [frames, channels, batch]
  // Use source tensor's strides for correct channel dimension
  const int64_t view_ne0 = state_frames;
  const int64_t view_ne1 = padded.tensor->ne[1];  // channels
  const int64_t view_ne2 = padded.tensor->ne[2];  // batch
  const size_t view_nb1 = padded.tensor->nb[1];
  const size_t view_nb2 = padded.tensor->nb[2];
  
  ggml_tensor* next_state = ggml_view_3d(
      ctx.ggml, padded.tensor, view_ne0, view_ne1, view_ne2,
      view_nb1, view_nb2, state_offset);
  next_state = ggml_cont(ctx.ggml, next_state);
  state.queue_update(next_state);

  return result;
}

core::TensorValue causal_conv1d_dw_stateful(core::ModuleBuildContext &ctx,
                                            const core::TensorValue &input,
                                            const VAEConv1dWeights &weights,
                                            int stride, int padding,
                                            int dilation,
                                            AudioVAEStreamingDecodeState &state,
                                            const std::string &state_name) {
  const int left_pad = 2 * padding;
  if (left_pad < 0) {
    throw std::runtime_error(
        "VoxCPM1 AudioVAE causal depthwise convolution padding is invalid");
  }
  const int state_frames = left_pad;
  if (state_frames <= 0) {
    return causal_conv1d(ctx, input, weights, stride, padding, dilation);
  }

  ggml_tensor* prev = state.take_slot(state_frames, input.shape.dims[1],
                                      state_name);
  if (!prev) {
    throw std::runtime_error(
        "VoxCPM1 AudioVAE streaming state slot not found: " + state_name);
  }
  // State has [state_frames, channels, 1], need to wrap with correct shape [1, channels, state_frames]
  auto prev_val = core::wrap_tensor(
      prev, core::TensorShape::from_dims({1, input.shape.dims[1], state_frames}),
      GGML_TYPE_F32);
  auto padded = modules::ConcatModule({2}).build(ctx, prev_val, input);

auto result = modules::DepthwiseConv1dModule(
                    {weights.out_channels, weights.kernel_size, stride, 0,
                     dilation, weights.depthwise.bias.has_value()})
              .build(ctx, padded, weights.depthwise);

  // Extract next state from the end of padded input (like causal_conv1d_stateful)
  const int64_t padded_frames = padded.shape.dims[2];
  const int64_t state_offset = (padded_frames - state_frames) * padded.tensor->nb[0];
  
  // View: extract last state_frames frames from padded input
  // Physical layout of padded.tensor is column-major [frames, channels, batch]
  // Use source tensor's strides for correct channel dimension
  const int64_t view_ne0 = state_frames;
  const int64_t view_ne1 = padded.tensor->ne[1];  // channels
  const int64_t view_ne2 = padded.tensor->ne[2];  // batch
  const size_t view_nb1 = padded.tensor->nb[1];
  const size_t view_nb2 = padded.tensor->nb[2];
  
  ggml_tensor* next_state = ggml_view_3d(
      ctx.ggml, padded.tensor, view_ne0, view_ne1, view_ne2,
      view_nb1, view_nb2, state_offset);
  next_state = ggml_cont(ctx.ggml, next_state);
  state.queue_update(next_state);

  return result;
}

core::TensorValue causal_conv_transpose1d_stateful(
    core::ModuleBuildContext &ctx, const core::TensorValue &input,
    const VAEConvTranspose1dWeights &weights, int stride, int padding,
    int output_padding, AudioVAEStreamingDecodeState &state,
    const std::string &state_name) {
  // For transpose conv, context frames = (kernel_size - 1) / stride
  const int64_t ctx_frames = (weights.kernel_size - 1) / stride;
  if (ctx_frames <= 0) {
    return causal_conv_transpose1d(ctx, input, weights, stride);
  }

ggml_tensor* prev = state.take_slot(ctx_frames, input.shape.dims[1],
                                       state_name);
  if (!prev) {
    throw std::runtime_error(
        "VoxCPM1 AudioVAE streaming state slot not found: " + state_name);
  }
  // State has [ctx_frames, channels, 1], need to wrap with correct shape [1, channels, ctx_frames]
  auto prev_val = core::wrap_tensor(
      prev, core::TensorShape::from_dims({1, input.shape.dims[1], ctx_frames}),
      GGML_TYPE_F32);
  auto x_full = modules::ConcatModule({2}).build(ctx, prev_val, input);

  auto full = modules::ConvTranspose1dModule(
                  {weights.in_channels, weights.out_channels, weights.kernel_size,
                   stride, 0, 1, weights.conv.bias.has_value()})
              .build(ctx, x_full, weights.conv);

  // Use correct conv_transpose1d output frames formula
  // (input_frames - 1) * stride + dilation * (kernel_size - 1) + 1
  // Here dilation=1, padding=0 (in the module config)
  const int64_t full_frames = (x_full.shape.dims[2] - 1) * stride + (weights.kernel_size - 1) + 1;
  auto view = ggml_view_3d(ctx.ggml, full.tensor, full_frames,
                           weights.out_channels, 1, full.tensor->nb[1],
                           full.tensor->nb[2], 0);
  auto result = core::wrap_tensor(
      ggml_cont(ctx.ggml, view),
      core::TensorShape::from_dims({1, weights.out_channels, full_frames}),
      GGML_TYPE_F32);

  // Crop: left = ctx_frames * stride, right = padding * 2 - output_padding
  const int64_t crop_left = ctx_frames * stride;
  const int crop_right = padding * 2 - output_padding;
  if (crop_left > 0 || crop_right > 0) {
    result = modules::SliceModule(
                 {2, static_cast<int64_t>(crop_left),
                  full_frames - crop_left - crop_right})
             .build(ctx, result);
  }

  const int64_t state_offset = (x_full.shape.dims[2] - ctx_frames) * x_full.tensor->nb[0];
  // View strides: use source tensor's strides for correct channel dimension
  const int64_t view_ne0 = ctx_frames;
  const int64_t view_ne1 = x_full.tensor->ne[1];
  const int64_t view_ne2 = x_full.tensor->ne[2];
  const size_t view_nb1 = x_full.tensor->nb[1];
  const size_t view_nb2 = x_full.tensor->nb[2];
  ggml_tensor* next_state = ggml_view_3d(
      ctx.ggml, x_full.tensor, view_ne0, view_ne1, view_ne2,
      view_nb1, view_nb2, state_offset);
  next_state = ggml_cont(ctx.ggml, next_state);
  state.queue_update(next_state);

  return result;
}

core::TensorValue residual_unit_stateful(core::ModuleBuildContext &ctx,
                                         const core::TensorValue &input,
                                         const VAEResidualUnitWeights &weights,
                                         int dilation,
                                         AudioVAEStreamingDecodeState &state,
                                         const std::string &state_prefix) {
  const int padding = static_cast<int>(((kResidualKernel - 1) * dilation) / 2);
  auto hidden = snake_exact(ctx, input, weights.snake1, input.shape.dims[1]);
  hidden = causal_conv1d_dw_stateful(ctx, hidden, weights.conv1, 1, padding,
                                     dilation, state,
                                     state_prefix + ".conv1");
  hidden = snake_exact(ctx, hidden, weights.snake2, input.shape.dims[1]);
  hidden = causal_conv1d(ctx, hidden, weights.conv2, 1, 0, 1);
  return modules::AddModule{}.build(ctx, input, hidden);
}

core::TensorValue decoder_block_stateful(core::ModuleBuildContext &ctx,
                                         const core::TensorValue &input,
                                         const VAEDecoderBlockWeights &weights,
                                         AudioVAEStreamingDecodeState &state,
                                         const std::string &state_prefix) {
  auto hidden =
      apply_sr_condition(ctx, input, weights.sr_cond, weights.input_channels);
  hidden = snake_exact(ctx, hidden, weights.snake, weights.input_channels);
  hidden = causal_conv_transpose1d_stateful(
      ctx, hidden, weights.upsample, weights.stride,
      static_cast<int>(std::ceil(weights.stride / 2.0f)), weights.stride % 2,
      state, state_prefix + ".transpose");
  hidden = residual_unit_stateful(ctx, hidden, weights.residual_units[0], 1,
                                  state, state_prefix + ".res0");
  hidden = residual_unit_stateful(ctx, hidden, weights.residual_units[1], 3,
                                  state, state_prefix + ".res1");
  hidden = residual_unit_stateful(ctx, hidden, weights.residual_units[2], 9,
                                  state, state_prefix + ".res2");
  return hidden;
}

} // namespace

class VoxCPM1AudioVAEDecoderRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1Assets> assets,
       core::ExecutionContext &execution_context,
       VoxCPM1AudioVAEDecoderConfig config)
      : assets_(require_assets(std::move(assets))),
        execution_context_(execution_context), config_(config),
        weights_(load_vae_weights(*assets_, execution_context_,
                                  config_.weight_context_bytes,
                                  config_.weight_storage_type)),
        decoder_stride_(product(assets_->config.audio_vae.decoder_rates)) {
    if (config_.latent_frame_capacity < 0) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE latent frame capacity must be non-negative");
    }
    if (config_.encoder_sample_capacity <= 0) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE encoder sample capacity must be positive");
    }
  }

  ~Impl() {
    release_streaming_decoder_graph();
    release_decoder_graph();
    release_encoder_graph();
  }

  runtime::AudioBuffer decode_features(const std::vector<float> &features,
                                       int64_t patches) {
    const auto &vae = assets_->config.audio_vae;
    if (patches < 0) {
      throw std::runtime_error("VoxCPM1 AudioVAE patch count is negative");
    }
    const int64_t latent_frames = patches * assets_->config.patch_size;
    const int64_t expected = latent_frames * vae.latent_dim;
    if (static_cast<int64_t>(features.size()) != expected) {
      throw std::runtime_error("VoxCPM1 AudioVAE feature size mismatch");
    }
    ensure_decoder_graph(latent_frames);
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
      throw std::runtime_error("VoxCPM1 AudioVAE decoder graph compute failed");
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

  VoxCPM1EncodedPrompt encode_prompt_audio(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio) {
    VoxCPM1EncodedPrompt out;
    if (prompt_audio.has_value()) {
      if (prompt_text.empty()) {
        throw std::runtime_error(
            "VoxCPM1 prompt audio requires prompt_text or reference_text");
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
    // The streaming graph references the request's AudioVAEStreamingDecodeState;
    // the session calls this before that state is destroyed.
    release_streaming_decoder_graph();
    release_decoder_graph();
    release_encoder_graph_impl();
  }

  void release_encoder_graph() { release_encoder_graph_impl(); }

  // Streaming decode support
  bool supports_streaming_decode() const {
    // The stateful graph runs the same decoder ops as the offline graph; its
    // state handling adds only concat, view/cont and backend tensor copies.
    // Metal is untested and stays on the per-patch decode.
    const core::BackendType t = execution_context_.backend_type();
    return t == core::BackendType::Cpu || t == core::BackendType::Cuda ||
           t == core::BackendType::Hip || t == core::BackendType::Vulkan;
  }

  bool initialize_streaming_decode_state(AudioVAEStreamingDecodeState& state) {
    if (!supports_streaming_decode()) {
      return false;
    }

    std::vector<AudioVAEStreamingDecodeState::SlotSpec> specs;
    specs.reserve(2 + weights_.decoder_blocks.size() * 4);

    // decoder.model.0 depthwise (padding=3, so state_frames=6)
    specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
        6,
        weights_.decoder_first_depthwise.out_channels,
        "decoder.model0.depthwise"});

    for (size_t i = 0; i < weights_.decoder_blocks.size(); ++i) {
      const VAEDecoderBlockWeights& block = weights_.decoder_blocks[i];
      const int stride = block.stride;
      const std::string prefix = "decoder.block" + std::to_string(i);
      const int64_t transpose_ctx = (block.upsample.kernel_size - 1) / stride;
      if (transpose_ctx > 0) {
        specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
            transpose_ctx,
            block.upsample.in_channels,
            prefix + ".transpose"});
      }

      // Residual units: conv1 is depthwise with different dilations
      specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
          6,
          block.residual_units[0].conv1.out_channels,
          prefix + ".res0.conv1"});
      specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
          18,
          block.residual_units[1].conv1.out_channels,
          prefix + ".res1.conv1"});
      specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
          54,
          block.residual_units[2].conv1.out_channels,
          prefix + ".res2.conv1"});
    }

    // decoder.final.conv (padding=3, so state_frames=6)
    specs.push_back(AudioVAEStreamingDecodeState::SlotSpec{
        6,
        weights_.decoder_final_conv.in_channels,
        "decoder.final.conv"});

    // Drop any previously built graph so it gets rebuilt against the new state
    release_streaming_decoder_graph();
    return state.initialize(specs, execution_context_);
  }

  runtime::AudioBuffer decode_streaming_step(
      const std::vector<float>& patch_features,
      AudioVAEStreamingDecodeState& state) {
    const auto &vae = assets_->config.audio_vae;
    const int64_t latent_frames = assets_->config.patch_size;
    const int64_t expected = latent_frames * vae.latent_dim;
    if (static_cast<int64_t>(patch_features.size()) != expected) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE streaming patch feature size mismatch");
    }

    // Build streaming decode graph (once) against the runtime state
    if (!streaming_graph_) {
      state.begin_graph();
      build_streaming_decoder_graph(state);
    }

    // Prepare input (single patch, latent_dim channels, patch_size frames)
    std::vector<float> input(static_cast<size_t>(vae.latent_dim * latent_frames), 0.0F);
    for (int64_t t = 0; t < latent_frames; ++t) {
      for (int64_t c = 0; c < vae.latent_dim; ++c) {
        input[static_cast<size_t>(c * latent_frames + t)] =
            patch_features[static_cast<size_t>(t * vae.latent_dim + c)];
      }
    }
    ggml_backend_tensor_set(streaming_input_, input.data(), 0,
                            input.size() * sizeof(float));

    core::set_backend_threads(execution_context_.backend(),
                              std::max(1, execution_context_.config().threads));
    const ggml_status status =
        core::compute_backend_graph(execution_context_.backend(), streaming_graph_);
    ggml_backend_synchronize(execution_context_.backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE streaming decoder graph compute failed");
    }

    // Publish state updates
    state.publish_updates(execution_context_);

    // Get output (decoder_stride_ samples per latent frame)
    const int64_t sample_count = latent_frames * decoder_stride_;
    std::vector<float> full(static_cast<size_t>(streaming_output_frames_), 0.0F);
    ggml_backend_tensor_get(streaming_output_, full.data(), 0,
                            full.size() * sizeof(float));

    runtime::AudioBuffer audio;
    audio.sample_rate = vae.output_sample_rate;
    audio.channels = 1;
    audio.samples.assign(full.begin(),
                         full.begin() + static_cast<std::ptrdiff_t>(sample_count));
    return audio;
  }

private:
  // Add streaming decode graph members
  std::unique_ptr<ggml_context, GgmlContextDeleter> streaming_ctx_;
  ggml_cgraph* streaming_graph_ = nullptr;
  ggml_gallocr_t streaming_gallocr_ = nullptr;
  ggml_tensor* streaming_input_ = nullptr;
  ggml_tensor* streaming_output_ = nullptr;
  int64_t streaming_output_frames_ = 0;

  void build_streaming_decoder_graph(AudioVAEStreamingDecodeState& state) {
    const auto &vae = assets_->config.audio_vae;
    const int64_t latent_frames = assets_->config.patch_size;
    if (latent_frames <= 0) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE patch size must be positive for streaming");
    }
    if (config_.graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE graph context bytes must be non-zero");
    }
    streaming_output_frames_ = latent_frames * decoder_stride_;

    ggml_init_params params{config_.graph_context_bytes, nullptr, true};
    streaming_ctx_.reset(ggml_init(params));
    if (!streaming_ctx_) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 AudioVAE streaming decoder graph context");
    }

    core::ModuleBuildContext ctx{streaming_ctx_.get(),
                                 "voxcpm1.audiovae.streaming_decoder",
                                 execution_context_.backend_type()};

    auto hidden = core::make_tensor(
        ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, vae.latent_dim, latent_frames}));
    streaming_input_ = hidden.tensor;
    ggml_set_input(streaming_input_);

    // decoder.model.0 depthwise (stateful)
    hidden = causal_conv1d_dw_stateful(
        ctx, hidden, weights_.decoder_first_depthwise, 1, 3, 1,
        state, "decoder.model0.depthwise");
    // decoder.model.1 pointwise (no state needed)
    hidden = causal_conv1d(ctx, hidden, weights_.decoder_first_pointwise, 1, 0, 1);

    // Decoder blocks (stateful)
    for (size_t i = 0; i < weights_.decoder_blocks.size(); ++i) {
      const auto& block = weights_.decoder_blocks[i];
      const std::string prefix = "decoder.block" + std::to_string(i);
      hidden = decoder_block_stateful(
          ctx, hidden, block, state, prefix);
    }

    // Final layers
    hidden = snake_exact(ctx, hidden, weights_.decoder_final_snake,
                         hidden.shape.dims[1]);
    hidden = causal_conv1d_stateful(
        ctx, hidden, weights_.decoder_final_conv, 1, 3, 1, 0,
        state, "decoder.final.conv");
    hidden = modules::TanhModule{}.build(ctx, hidden);

    streaming_output_ = hidden.tensor;
    ggml_set_output(streaming_output_);

    streaming_graph_ = ggml_new_graph_custom(streaming_ctx_.get(), 65536, false);
    ggml_build_forward_expand(streaming_graph_, streaming_output_);

    state.build_update_graph(streaming_graph_);

    streaming_gallocr_ = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_.backend()));
    if (!streaming_gallocr_ || !ggml_gallocr_reserve(streaming_gallocr_, streaming_graph_) ||
        !ggml_gallocr_alloc_graph(streaming_gallocr_, streaming_graph_)) {
      release_streaming_decoder_graph();
      throw std::runtime_error(
          "failed to allocate VoxCPM1 AudioVAE streaming decoder graph");
    }
  }

  void release_streaming_decoder_graph() {
    if (streaming_graph_) {
      core::release_backend_graph_resources(execution_context_.backend(), streaming_graph_);
    }
    if (streaming_gallocr_) {
      ggml_gallocr_free(streaming_gallocr_);
      streaming_gallocr_ = nullptr;
    }
    streaming_graph_ = nullptr;
    streaming_input_ = nullptr;
    streaming_output_ = nullptr;
    streaming_ctx_.reset();
    streaming_output_frames_ = 0;
  }



  // The rest of the members...
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
      options.warning_context = "VoxCPM1 AudioVAE encoder";
      options.fallback_description = "linear resampling";
      mono = engine::audio::resample_mono_soxr_or_linear(
          mono, audio.sample_rate, vae.sample_rate, options);
    }
    // VAD trim silence (match VoxCPM.cpp server_common.cpp:842/878)
    mono = trim_audio_silence_vad(mono, vae.sample_rate);
    // Patch-aligned padding (Left for prompt, Right for reference)
    const int64_t patch_samples = assets_->config.patch_size * encoder_stride_;
    pad_audio_for_patch_alignment(mono, static_cast<size_t>(patch_samples),
                                  left_pad ? PaddingMode::Left : PaddingMode::Right);
    // Final padding to encoder_sample_capacity
    const int64_t sample_count = static_cast<int64_t>(mono.size());
    const int64_t padded_samples =
        ((sample_count + patch_samples - 1) / patch_samples) * patch_samples;
    if (patch_samples <= 0) {
      throw std::runtime_error("VoxCPM1 AudioVAE patch sample size is invalid");
    }
    if (padded_samples > config_.encoder_sample_capacity) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE encoder sample capacity exceeded");
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
      throw std::runtime_error("VoxCPM1 AudioVAE encoder graph compute failed");
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
          "VoxCPM1 AudioVAE encoded frames are not divisible by patch size");
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

  int64_t decoder_capacity_for(int64_t latent_frames) const {
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
    if (graph_ != nullptr && latent_frames <= decoder_latent_frame_capacity_) {
      engine::debug::timing_log_scalar(
          "voxcpm1.audiovae.decoder.graph.rebuilt", false);
      engine::debug::timing_log_scalar(
          "voxcpm1.audiovae.decoder.graph.reused", true);
      engine::debug::timing_log_scalar(
          "voxcpm1.audiovae.decoder.graph.build_ms", 0.0);
      engine::debug::timing_log_scalar(
          "voxcpm1.audiovae.decoder.latent_capacity",
          decoder_latent_frame_capacity_);
      return;
    }
    const auto build_start = Clock::now();
    build_decoder(decoder_capacity_for(latent_frames));
    engine::debug::timing_log_scalar(
        "voxcpm1.audiovae.decoder.graph.rebuilt", true);
    engine::debug::timing_log_scalar(
        "voxcpm1.audiovae.decoder.graph.reused", false);
    engine::debug::timing_log_scalar(
        "voxcpm1.audiovae.decoder.graph.build_ms",
        engine::debug::elapsed_ms(build_start));
    engine::debug::timing_log_scalar(
        "voxcpm1.audiovae.decoder.latent_capacity",
        decoder_latent_frame_capacity_);
  }

  void release_decoder_graph() {
    if (graph_ != nullptr) {
      core::release_backend_graph_resources(execution_context_.backend(), graph_);
    }
    if (gallocr_ != nullptr) {
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
          "VoxCPM1 AudioVAE decoder graph capacity must be positive");
    }
    release_decoder_graph();
    if (config_.graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 AudioVAE graph context bytes must be non-zero");
    }
    output_frames_ = latent_frame_capacity * decoder_stride_;
    ggml_init_params params{config_.graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 AudioVAE decoder graph context");
    }
    core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.audiovae.decoder",
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
    gallocr_ = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_.backend()));
    if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
        !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
      release_decoder_graph();
      throw std::runtime_error("failed to allocate VoxCPM1 AudioVAE graph");
    }
    decoder_latent_frame_capacity_ = latent_frame_capacity;
  }

  void release_encoder_graph_impl() {
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
          "VoxCPM1 AudioVAE encoder graph context bytes must be non-zero");
    }
    encoder_stride_ = product(vae.encoder_rates);
    if (config_.encoder_sample_capacity % encoder_stride_ != 0) {
      throw std::runtime_error("VoxCPM1 AudioVAE encoder sample capacity must "
                               "be divisible by encoder stride");
    }
    ggml_init_params params{config_.encoder_graph_context_bytes, nullptr, true};
    encoder_ctx_.reset(ggml_init(params));
    if (encoder_ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 AudioVAE encoder graph context");
    }
    core::ModuleBuildContext ctx{encoder_ctx_.get(), "voxcpm1.audiovae.encoder",
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
          "failed to allocate VoxCPM1 AudioVAE encoder graph");
    }
  }

  std::shared_ptr<const VoxCPM1Assets> assets_;
  core::ExecutionContext &execution_context_;
  VoxCPM1AudioVAEDecoderConfig config_;
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
};

VoxCPM1AudioVAEDecoderRuntime::VoxCPM1AudioVAEDecoderRuntime(
    std::shared_ptr<const VoxCPM1Assets> assets,
    core::ExecutionContext &execution_context,
    VoxCPM1AudioVAEDecoderConfig config)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   std::move(config))) {}

VoxCPM1AudioVAEDecoderRuntime::~VoxCPM1AudioVAEDecoderRuntime() = default;

runtime::AudioBuffer VoxCPM1AudioVAEDecoderRuntime::decode_features(
    const std::vector<float> &features, int64_t patches) {
  return impl_->decode_features(features, patches);
}

bool VoxCPM1AudioVAEDecoderRuntime::supports_streaming_decode() const {
  return impl_->supports_streaming_decode();
}

bool VoxCPM1AudioVAEDecoderRuntime::initialize_streaming_decode_state(
    AudioVAEStreamingDecodeState &state) {
  return impl_->initialize_streaming_decode_state(state);
}

runtime::AudioBuffer VoxCPM1AudioVAEDecoderRuntime::decode_streaming_step(
    const std::vector<float> &patch_features,
    AudioVAEStreamingDecodeState &state) {
  return impl_->decode_streaming_step(patch_features, state);
}

VoxCPM1EncodedPrompt VoxCPM1AudioVAEDecoderRuntime::encode_prompt_audio(
    const std::optional<runtime::AudioBuffer> &prompt_audio,
    const std::string &prompt_text,
    const std::optional<runtime::AudioBuffer> &reference_audio) {
  return impl_->encode_prompt_audio(prompt_audio, prompt_text, reference_audio);
}

void VoxCPM1AudioVAEDecoderRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

void VoxCPM1AudioVAEDecoderRuntime::release_encoder_graph() {
  impl_->release_encoder_graph();
}

} // namespace engine::community_models::voxcpm1


#include "engine/framework/codecs/snac_decoder_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace engine::codecs {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
  void operator()(ggml_context *context) const noexcept {
    if (context != nullptr) {
      ggml_free(context);
    }
  }
};

struct GallocrDeleter {
  void operator()(ggml_gallocr_t allocator) const noexcept {
    if (allocator != nullptr) {
      ggml_gallocr_free(allocator);
    }
  }
};

struct ResidualWeights {
  modules::Snake1dWeights snake1;
  modules::DepthwiseConv1dWeights conv1;
  modules::Snake1dWeights snake2;
  modules::Conv1dWeights conv2;
};

struct DecoderBlockWeights {
  modules::Snake1dWeights snake;
  modules::ConvTranspose1dWeights up;
  modules::Conv1dWeights noise;
  std::vector<ResidualWeights> residuals;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int stride = 1;
};

struct HostQuantizerWeights {
  std::vector<float> codebook;
  std::vector<float> out_weight;
  std::vector<float> out_bias;
  int stride = 1;
};

struct SnacWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  std::vector<HostQuantizerWeights> quantizers;
  modules::DepthwiseConv1dWeights depthwise_input;
  modules::Conv1dWeights pointwise_input;
  std::vector<DecoderBlockWeights> blocks;
  modules::Snake1dWeights final_snake;
  modules::Conv1dWeights final_conv;
};

modules::Snake1dWeights load_snake(core::BackendWeightStore &store,
                                   const assets::TensorSource &source,
                                   const std::string &name, int64_t channels) {
  return {store.make_from_f32(core::TensorShape::from_dims({channels}),
                              assets::TensorStorageType::F32,
                              source.require_f32(name, {1, channels, 1}))};
}

ResidualWeights load_residual(core::BackendWeightStore &store,
                              const assets::TensorSource &source,
                              const std::string &prefix, int64_t channels,
                              int kernel_size,
                              assets::TensorStorageType storage_type) {
  ResidualWeights out;
  out.snake1 = load_snake(store, source, prefix + ".block.0.alpha", channels);
  out.conv1 = modules::binding::depthwise_conv1d_from_source(
      store, source, prefix + ".block.1", storage_type, channels, kernel_size,
      true);
  out.snake2 = load_snake(store, source, prefix + ".block.2.alpha", channels);
  out.conv2 = modules::binding::conv1d_from_source(
      store, source, prefix + ".block.3", storage_type, channels, channels, 1,
      true);
  return out;
}

HostQuantizerWeights load_quantizer(const assets::TensorSource &source,
                                    const SnacDecoderConfig &config,
                                    const SnacDecoderWeightBinding &binding,
                                    size_t index) {
  const std::string prefix = binding.quantizer_prefix + std::to_string(index);
  HostQuantizerWeights out;
  out.codebook = source.require_f32(
      prefix + ".codebook.weight", {config.codebook_size, config.codebook_dim});
  out.out_weight = source.require_f32(
      prefix + ".out_proj.weight", {config.latent_dim, config.codebook_dim, 1});
  out.out_bias =
      source.require_f32(prefix + ".out_proj.bias", {config.latent_dim});
  out.stride = config.quantizer_strides[index];
  return out;
}

SnacWeights load_weights(const assets::TensorSource &source,
                         const SnacDecoderConfig &config,
                         const SnacDecoderWeightBinding &binding,
                         core::ExecutionContext &execution,
                         const SnacDecoderRuntimeOptions &options) {
  SnacWeights out;
  out.store = std::make_shared<core::BackendWeightStore>(
      execution.backend(), execution.backend_type(),
      config.trace_name + ".weights", options.weight_context_bytes);
  out.quantizers.reserve(config.quantizer_strides.size());
  for (size_t index = 0; index < config.quantizer_strides.size(); ++index) {
    out.quantizers.push_back(load_quantizer(source, config, binding, index));
  }
  const auto storage_type = options.weight_storage_type;
  out.depthwise_input = modules::binding::depthwise_conv1d_from_source(
      *out.store, source, binding.decoder_prefix + "0", storage_type,
      config.latent_dim, config.input_kernel_size, true);
  out.pointwise_input = modules::binding::conv1d_from_source(
      *out.store, source, binding.decoder_prefix + "1", storage_type,
      config.decoder_channels.front(), config.latent_dim, 1, true);
  out.blocks.resize(config.upsample_strides.size());
  for (size_t stage = 0; stage < config.upsample_strides.size(); ++stage) {
    const int64_t in_channels = config.decoder_channels[stage];
    const int64_t out_channels = config.decoder_channels[stage + 1];
    const int stride = config.upsample_strides[stage];
    const std::string prefix =
        binding.decoder_prefix + std::to_string(stage + 2);
    auto &block = out.blocks[stage];
    block.in_channels = in_channels;
    block.out_channels = out_channels;
    block.stride = stride;
    block.snake =
        load_snake(*out.store, source, prefix + ".block.0.alpha", in_channels);
    block.up = modules::binding::conv_transpose1d_from_source(
        *out.store, source, prefix + ".block.1", storage_type, in_channels,
        out_channels, 2 * stride, true);
    block.noise = modules::binding::conv1d_from_source(
        *out.store, source, prefix + ".block.2.linear", storage_type,
        out_channels, out_channels, 1, false);
    block.residuals.reserve(config.residual_dilations.size());
    for (size_t residual = 0; residual < config.residual_dilations.size();
         ++residual) {
      block.residuals.push_back(load_residual(
          *out.store, source, prefix + ".block." + std::to_string(residual + 3),
          out_channels, config.residual_kernel_size, storage_type));
    }
  }
  const auto final_index = config.upsample_strides.size() + 2;
  const auto final_channels = config.decoder_channels.back();
  out.final_snake = load_snake(*out.store, source,
                               binding.decoder_prefix +
                                   std::to_string(final_index) + ".alpha",
                               final_channels);
  out.final_conv = modules::binding::conv1d_from_source(
      *out.store, source,
      binding.decoder_prefix + std::to_string(final_index + 1), storage_type, 1,
      final_channels, config.output_kernel_size, true);
  out.store->upload();
  return out;
}

core::TensorValue residual(core::ModuleBuildContext &context,
                           const core::TensorValue &input,
                           const ResidualWeights &weights, int kernel_size,
                           int dilation) {
  auto hidden = modules::Snake1dModule({input.shape.dims[1]})
                    .build(context, input, weights.snake1);
  hidden =
      modules::DepthwiseConv1dModule({
                                         input.shape.dims[1],
                                         kernel_size,
                                         1,
                                         ((kernel_size - 1) / 2) * dilation,
                                         dilation,
                                         true,
                                     })
          .build(context, hidden, weights.conv1);
  hidden = modules::Snake1dModule({input.shape.dims[1]})
               .build(context, hidden, weights.snake2);
  hidden = modules::Conv1dModule(
               {input.shape.dims[1], input.shape.dims[1], 1, 1, 0, 1, true})
               .build(context, hidden, weights.conv2);
  return modules::ResidualAddModule{}.build(context, input, hidden);
}

std::vector<float>
build_quantized_latents(const SnacCodes &codes,
                        const std::vector<HostQuantizerWeights> &weights,
                        const SnacDecoderConfig &config) {
  if (codes.codebooks.size() != weights.size()) {
    throw std::runtime_error(
        "SNAC codebook count does not match configuration");
  }
  const size_t output_frames = codes.codebooks.back().size();
  std::vector<float> out(static_cast<size_t>(config.latent_dim) * output_frames,
                         0.0F);
  for (size_t level = 0; level < weights.size(); ++level) {
    const auto &quantizer = weights[level];
    const auto &level_codes = codes.codebooks[level];
    if (level_codes.size() * static_cast<size_t>(quantizer.stride) !=
        output_frames) {
      throw std::runtime_error("SNAC codebook frame counts are inconsistent");
    }
    for (size_t frame = 0; frame < level_codes.size(); ++frame) {
      const int32_t code = level_codes[frame];
      if (code < 0 || code >= config.codebook_size) {
        throw std::runtime_error("SNAC code is out of range");
      }
      const size_t code_offset =
          static_cast<size_t>(code * config.codebook_dim);
      for (int64_t channel = 0; channel < config.latent_dim; ++channel) {
        float value = quantizer.out_bias[channel];
        const size_t weight_offset =
            static_cast<size_t>(channel * config.codebook_dim);
        for (int64_t dimension = 0; dimension < config.codebook_dim;
             ++dimension) {
          value += quantizer.out_weight[weight_offset + dimension] *
                   quantizer.codebook[code_offset + dimension];
        }
        for (int64_t repeat = 0; repeat < quantizer.stride; ++repeat) {
          const size_t output_frame =
              frame * static_cast<size_t>(quantizer.stride) +
              static_cast<size_t>(repeat);
          out[static_cast<size_t>(channel) * output_frames + output_frame] +=
              value;
        }
      }
    }
  }
  return out;
}

SnacDecoderConfig validate_config(SnacDecoderConfig config) {
  if (config.sample_rate <= 0 || config.codebook_size <= 0 ||
      config.codebook_dim <= 0 || config.latent_dim <= 0 ||
      config.input_kernel_size <= 0 || config.output_kernel_size <= 0 ||
      config.residual_kernel_size <= 0 ||
      config.residual_kernel_size % 2 == 0 || config.trace_name.empty()) {
    throw std::runtime_error("SNAC decoder configuration is invalid");
  }
  if (config.quantizer_strides.empty() || config.decoder_channels.size() < 2 ||
      config.upsample_strides.size() + 1 != config.decoder_channels.size() ||
      config.residual_dilations.empty()) {
    throw std::runtime_error("SNAC decoder stage configuration is invalid");
  }
  for (const auto value : config.quantizer_strides) {
    if (value <= 0) {
      throw std::runtime_error("SNAC quantizer stride must be positive");
    }
  }
  for (const auto value : config.decoder_channels) {
    if (value <= 0) {
      throw std::runtime_error("SNAC decoder channel count must be positive");
    }
  }
  for (const auto value : config.upsample_strides) {
    if (value <= 0) {
      throw std::runtime_error("SNAC upsample stride must be positive");
    }
  }
  for (const auto value : config.residual_dilations) {
    if (value <= 0) {
      throw std::runtime_error("SNAC residual dilation must be positive");
    }
  }
  return config;
}

std::shared_ptr<const assets::TensorSource>
folded_source(std::shared_ptr<const assets::TensorSource> source,
              const SnacDecoderWeightBinding &binding) {
  if (source == nullptr) {
    throw std::runtime_error("SNAC decoder requires a tensor source");
  }
  return assets::make_weight_norm_folded_tensor_source(
      std::move(source),
      {binding.quantizer_prefix + "*", binding.decoder_prefix + "*"});
}

} // namespace

struct SnacDecoderRuntime::Impl {
  struct Graph {
    Graph(core::ExecutionContext &execution, const SnacDecoderConfig &config,
          const SnacWeights &weights, size_t context_bytes,
          int64_t latent_frames)
        : frames(latent_frames) {
      ggml_init_params params{context_bytes, nullptr, true};
      context.reset(ggml_init(params));
      if (context == nullptr) {
        throw std::runtime_error("failed to create SNAC decoder graph context");
      }
      core::ModuleBuildContext build{context.get(), config.trace_name.c_str()};
      latent = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, frames,
                                  config.latent_dim, 1);
      auto hidden = core::wrap_tensor(
          latent, core::TensorShape::from_dims({1, config.latent_dim, frames}),
          GGML_TYPE_F32);
      hidden = modules::DepthwiseConv1dModule(
                   {config.latent_dim, config.input_kernel_size, 1,
                    (config.input_kernel_size - 1) / 2, 1, true})
                   .build(build, hidden, weights.depthwise_input);
      hidden = modules::Conv1dModule({config.latent_dim,
                                      config.decoder_channels.front(), 1, 1, 0,
                                      1, true})
                   .build(build, hidden, weights.pointwise_input);
      noise.resize(weights.blocks.size());
      int64_t current_frames = frames;
      for (size_t stage = 0; stage < weights.blocks.size(); ++stage) {
        const auto &block = weights.blocks[stage];
        hidden = modules::Snake1dModule({hidden.shape.dims[1]})
                     .build(build, hidden, block.snake);
        auto upsampled = modules::ConvTranspose1dModule({
                                                            block.in_channels,
                                                            block.out_channels,
                                                            2 * block.stride,
                                                            block.stride,
                                                            0,
                                                            1,
                                                            true,
                                                        })
                             .build(build, hidden, block.up);
        const int padding = (block.stride + 1) / 2;
        current_frames *= block.stride;
        hidden = modules::SliceModule({2, padding, current_frames})
                     .build(build, upsampled);

        noise[stage] = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32,
                                          current_frames, 1, 1);
        auto noise_value = core::wrap_tensor(
            noise[stage], core::TensorShape::from_dims({1, 1, current_frames}),
            GGML_TYPE_F32);
        noise_value =
            modules::RepeatModule({hidden.shape}).build(build, noise_value);
        auto amplitude =
            modules::Conv1dModule(
                {block.out_channels, block.out_channels, 1, 1, 0, 1, false})
                .build(build, hidden, block.noise);
        hidden = modules::AddModule{}.build(
            build, hidden,
            modules::MulModule{}.build(build, noise_value, amplitude));
        for (size_t residual_index = 0; residual_index < block.residuals.size();
             ++residual_index) {
          hidden = residual(build, hidden, block.residuals[residual_index],
                            config.residual_kernel_size,
                            config.residual_dilations[residual_index]);
        }
      }
      const auto final_channels = config.decoder_channels.back();
      hidden = modules::Snake1dModule({final_channels})
                   .build(build, hidden, weights.final_snake);
      hidden = modules::Conv1dModule(
                   {final_channels, 1, config.output_kernel_size, 1,
                    (config.output_kernel_size - 1) / 2, 1, true})
                   .build(build, hidden, weights.final_conv);
      output = modules::TanhModule{}.build(build, hidden).tensor;
      ggml_set_output(output);
      graph = ggml_new_graph_custom(context.get(), 32768, false);
      ggml_build_forward_expand(graph, output);
      allocator.reset(ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(execution.backend())));
      if (allocator == nullptr ||
          !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        throw std::runtime_error("failed to allocate SNAC decoder graph");
      }
    }

    int64_t frames = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> context;
    ggml_tensor *latent = nullptr;
    std::vector<ggml_tensor *> noise;
    ggml_tensor *output = nullptr;
    ggml_cgraph *graph = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GallocrDeleter>
        allocator;
  };

  Impl(SnacDecoderConfig config_in,
       std::shared_ptr<const assets::TensorSource> source,
       core::ExecutionContext &execution_in, SnacDecoderRuntimeOptions options,
       SnacDecoderWeightBinding binding)
      : config(validate_config(std::move(config_in))), execution(execution_in),
        graph_context_bytes(options.graph_context_bytes),
        weights(load_weights(*folded_source(std::move(source), binding), config,
                             binding, execution_in, options)),
        sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
            execution_in.backend_type(), execution_in.config().device,
            config.trace_name + ".noise", config.trace_name,
            sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
    if (options.weight_context_bytes == 0 || options.graph_context_bytes == 0) {
      throw std::runtime_error("SNAC decoder context sizes must be non-zero");
    }
  }

  runtime::AudioBuffer decode(const SnacCodes &codes, uint64_t seed) {
    if (codes.codebooks.empty() || codes.codebooks.back().empty()) {
      throw std::runtime_error("SNAC decoder requires at least one code frame");
    }
    auto latents = build_quantized_latents(codes, weights.quantizers, config);
    const int64_t latent_frames =
        static_cast<int64_t>(codes.codebooks.back().size());
    const bool rebuilt = graph == nullptr || graph->frames != latent_frames;
    if (rebuilt) {
      graph = std::make_unique<Graph>(execution, config, weights,
                                      graph_context_bytes, latent_frames);
    }
    debug::trace_log_scalar(config.trace_name + ".graph_rebuilt", rebuilt);

    ggml_backend_tensor_set(graph->latent, latents.data(), 0,
                            latents.size() * sizeof(float));
    uint64_t offset_blocks = 0;
    int64_t stage_frames = latent_frames;
    for (size_t stage = 0; stage < graph->noise.size(); ++stage) {
      stage_frames *= weights.blocks[stage].stride;
      auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(
          static_cast<size_t>(stage_frames), seed, offset_blocks,
          sampling_policy);
      offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
          static_cast<uint64_t>(stage_frames), sampling_policy);
      ggml_backend_tensor_set(graph->noise[stage], noise.data(), 0,
                              noise.size() * sizeof(float));
    }
    const auto started = Clock::now();
    core::compute_backend_graph(execution.backend(), graph->graph);
    ggml_backend_synchronize(execution.backend());
    debug::timing_log_scalar(config.trace_name + ".decode_ms",
                             debug::elapsed_ms(started, Clock::now()));

    runtime::AudioBuffer audio;
    audio.sample_rate = config.sample_rate;
    audio.channels = 1;
    audio.samples.resize(static_cast<size_t>(graph->output->ne[0]));
    ggml_backend_tensor_get(graph->output, audio.samples.data(), 0,
                            audio.samples.size() * sizeof(float));
    return audio;
  }

  SnacDecoderConfig config;
  core::ExecutionContext &execution;
  size_t graph_context_bytes;
  SnacWeights weights;
  sampling::TorchCudaSamplingPolicy sampling_policy;
  std::unique_ptr<Graph> graph;
};

SnacDecoderRuntime::SnacDecoderRuntime(
    SnacDecoderConfig config,
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext &execution, SnacDecoderRuntimeOptions options,
    SnacDecoderWeightBinding weight_binding)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(source),
                                   execution, options,
                                   std::move(weight_binding))) {}

SnacDecoderRuntime::~SnacDecoderRuntime() = default;
SnacDecoderRuntime::SnacDecoderRuntime(SnacDecoderRuntime &&) noexcept =
    default;
SnacDecoderRuntime &
SnacDecoderRuntime::operator=(SnacDecoderRuntime &&) noexcept = default;

runtime::AudioBuffer SnacDecoderRuntime::decode(const SnacCodes &codes,
                                                uint64_t seed) {
  return impl_->decode(codes, seed);
}

void SnacDecoderRuntime::release_runtime_graphs() { impl_->graph.reset(); }

} // namespace engine::codecs

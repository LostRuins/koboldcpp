#include "engine/community_models/sense_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/sanm.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sense_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 32768;
constexpr size_t kWeightContextBytes = 64 * 1024 * 1024;
constexpr float kLayerNormEpsilon = 1.0e-5F;

struct SanmBlockWeights {
  engine::modules::NormWeights norm1;
  engine::modules::LinearWeights linear_q_k_v;
  engine::modules::LinearWeights linear_out;
  engine::core::TensorValue fsmn_weight;
  engine::modules::LinearWeights w_1;
  engine::modules::LinearWeights w_2;
  engine::modules::NormWeights norm2;
};

engine::modules::LinearWeights
load_linear(engine::core::BackendWeightStore &store,
            const engine::assets::TensorSource &source,
            const std::string &prefix, int64_t input_size,
            int64_t output_size,
            engine::assets::TensorStorageType storage_type) {
  return {
      store.load_tensor(source, prefix + ".weight", storage_type,
                        {output_size, input_size}),
      store.load_f32_tensor(source, prefix + ".bias", {output_size}),
  };
}

engine::modules::NormWeights load_norm(engine::core::BackendWeightStore &store,
                                       const engine::assets::TensorSource &source,
                                       const std::string &prefix,
                                       int64_t size) {
  return {
      store.load_f32_tensor(source, prefix + ".weight", {size}),
      store.load_f32_tensor(source, prefix + ".bias", {size}),
  };
}

engine::core::TensorValue
load_fsmn_weight(engine::core::BackendWeightStore &store,
                 const engine::assets::TensorSource &source,
                 const std::string &name, const SenseAsrEncoderConfig &config) {
  const auto values =
      source.require_f32(name, {config.kernel_size, config.d_model});
  std::vector<float> packed(
      static_cast<size_t>(config.d_model * config.kernel_size), 0.0F);
  for (int64_t channel = 0; channel < config.d_model; ++channel) {
    for (int64_t index = 0; index < config.kernel_size; ++index) {
      packed[static_cast<size_t>(channel * config.kernel_size + index)] =
          values[static_cast<size_t>(index * config.d_model + channel)];
    }
  }
  return store.make_from_f32(
      engine::core::TensorShape::from_dims(
          {config.d_model, 1, config.kernel_size}),
      engine::assets::TensorStorageType::F32, std::move(packed));
}

SanmBlockWeights load_sanm_block(engine::core::BackendWeightStore &store,
                                 const engine::assets::TensorSource &source,
                                 const std::string &prefix, int64_t input_size,
                                 const SenseAsrEncoderConfig &config,
                                 engine::assets::TensorStorageType storage_type) {
  SanmBlockWeights weights;
  weights.norm1 = load_norm(store, source, prefix + ".norm1", input_size);
  weights.linear_q_k_v =
      load_linear(store, source, prefix + ".self_attn.linear_q_k_v",
                  input_size, 3 * config.d_model, storage_type);
  weights.linear_out =
      load_linear(store, source, prefix + ".self_attn.linear_out",
                  config.d_model, config.d_model, storage_type);
  weights.fsmn_weight = load_fsmn_weight(
      store, source, prefix + ".self_attn.fsmn_block.weight", config);
  weights.w_1 = load_linear(store, source, prefix + ".feed_forward.w_1",
                            config.d_model, config.ffn_dim, storage_type);
  weights.w_2 = load_linear(store, source, prefix + ".feed_forward.w_2",
                            config.ffn_dim, config.d_model, storage_type);
  weights.norm2 = load_norm(store, source, prefix + ".norm2", config.d_model);
  return weights;
}

struct EncoderWeights {
  std::unique_ptr<engine::core::BackendWeightStore> store;
  SanmBlockWeights stem;
  std::vector<SanmBlockWeights> main_layers;
  engine::modules::NormWeights after_norm;
  std::vector<SanmBlockWeights> timestamp_layers;
  engine::modules::NormWeights tp_norm;
  engine::modules::LinearWeights ctc_lo;
  std::vector<float> embed_rows;
};

std::unique_ptr<EncoderWeights>
load_encoder_weights(const SenseAsrAssets &assets,
                     engine::core::ExecutionContext &execution_context,
                     engine::assets::TensorStorageType storage_type) {
  auto weights = std::make_unique<EncoderWeights>();
  weights->store = std::make_unique<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "SenseVoice-Small encoder weights", kWeightContextBytes);
  const auto &source = *assets.model_weights;
  const auto &config = assets.config.encoder;
  const std::string stem_root = "encoder.encoders0.0";
  const std::string main_root = "encoder.encoders.";
  const std::string tp_root = "encoder.tp_encoders.";

  weights->stem =
      load_sanm_block(*weights->store, source, stem_root, config.input_size,
                      config, storage_type);
  weights->main_layers.reserve(static_cast<size_t>(config.num_blocks - 1));
  for (int64_t index = 0; index < config.num_blocks - 1; ++index) {
    weights->main_layers.push_back(load_sanm_block(
        *weights->store, source, main_root + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->after_norm = load_norm(*weights->store, source,
                                  "encoder.after_norm", config.d_model);
  weights->timestamp_layers.reserve(
      static_cast<size_t>(config.timestamp_prediction_layers));
  for (int64_t index = 0; index < config.timestamp_prediction_layers; ++index) {
    weights->timestamp_layers.push_back(load_sanm_block(
        *weights->store, source, tp_root + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->tp_norm =
      load_norm(*weights->store, source, "encoder.tp_norm", config.d_model);
  weights->ctc_lo =
      load_linear(*weights->store, source, "ctc.ctc_lo", config.d_model,
                  config.vocab_size, storage_type);

  const auto embed =
      source.require_f32("embed.weight", {16, config.input_size});
  if (static_cast<int64_t>(embed.size()) != 16 * config.input_size) {
    throw std::runtime_error(
        "SenseVoice-Small embed.weight has an unexpected size");
  }
  weights->embed_rows = embed;

  weights->store->upload();
  return weights;
}

std::vector<float> make_posenc_input(int64_t frames, int64_t channels) {
  if (channels <= 2 || channels % 2 != 0) {
    throw std::runtime_error(
        "SenseVoice positional encoding channel shape is invalid");
  }
  const int64_t half = channels / 2;
  const double increment = std::log(10000.0) / (static_cast<double>(half) - 1.0);
  std::vector<float> values(static_cast<size_t>(frames * channels), 0.0F);
  for (int64_t frame = 0; frame < frames; ++frame) {
    const double position = static_cast<double>(frame + 1);
    for (int64_t index = 0; index < half; ++index) {
      const double inverse_timescale = std::exp(static_cast<double>(index) * -increment);
      const double phase = position * inverse_timescale;
      const size_t base = static_cast<size_t>(frame * channels + index);
      values[base] = static_cast<float>(std::sin(phase));
      values[base + static_cast<size_t>(half)] = static_cast<float>(std::cos(phase));
    }
  }
  return values;
}

engine::core::TensorValue view_linear_rows(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &weight,
    int64_t row_offset,
    int64_t rows,
    int64_t cols,
    const char *label) {
  if (weight.shape.rank != 2 || weight.shape.dims[0] < row_offset + rows ||
      weight.shape.dims[1] != cols) {
    throw std::runtime_error(std::string("SenseVoice ") + label +
                             " weight view is invalid");
  }
  const size_t row_stride = weight.tensor->nb[1];
  const size_t byte_offset = static_cast<size_t>(row_offset) * row_stride;
  return engine::core::wrap_tensor(
      ggml_view_2d(ctx.ggml, weight.tensor, cols, rows, row_stride,
                   byte_offset),
      engine::core::TensorShape::from_dims({rows, cols}), weight.type);
}

engine::core::TensorValue view_linear_bias(
    engine::core::ModuleBuildContext &ctx,
    const std::optional<engine::core::TensorValue> &bias,
    int64_t offset,
    int64_t size,
    const char *label) {
  if (!bias.has_value() || bias->shape.rank != 1 ||
      bias->shape.dims[0] < offset + size) {
    throw std::runtime_error(std::string("SenseVoice ") + label +
                             " bias view is invalid");
  }
  return engine::core::wrap_tensor(
      ggml_view_1d(ctx.ggml, bias->tensor, size,
                   static_cast<size_t>(offset) * ggml_type_size(bias->type)),
      engine::core::TensorShape::from_dims({size}), bias->type);
}

engine::modules::SanmBlockWeightsView
framework_sanm_weights(engine::core::ModuleBuildContext &ctx,
                       const SanmBlockWeights &weights,
                       const SenseAsrEncoderConfig &config,
                       int64_t input_size) {
  return {
      weights.norm1,
      {
          view_linear_rows(ctx, weights.linear_q_k_v.weight, 0,
                           config.d_model, input_size, "query"),
          view_linear_bias(ctx, weights.linear_q_k_v.bias, 0, config.d_model,
                           "query"),
      },
      {
          view_linear_rows(ctx, weights.linear_q_k_v.weight, config.d_model,
                           config.d_model, input_size, "key"),
          view_linear_bias(ctx, weights.linear_q_k_v.bias, config.d_model,
                           config.d_model, "key"),
      },
      {
          view_linear_rows(ctx, weights.linear_q_k_v.weight, 2 * config.d_model,
                           config.d_model, input_size, "value"),
          view_linear_bias(ctx, weights.linear_q_k_v.bias, 2 * config.d_model,
                           config.d_model, "value"),
      },
      weights.linear_out,
      weights.fsmn_weight,
      weights.norm2,
      weights.w_1,
      weights.w_2,
  };
}

engine::modules::SanmBlockConfig block_config(
    const SenseAsrEncoderConfig &config, int64_t input_size) {
  engine::modules::SanmBlockConfig result;
  result.input_size = input_size;
  result.model_size = config.d_model;
  result.num_heads = config.attention_heads;
  result.ffn_size = config.ffn_dim;
  result.fsmn_kernel_size = config.kernel_size;
  result.layer_norm_eps = kLayerNormEpsilon;
  result.attention_lowering =
      engine::modules::ScaledDotProductAttentionLowering::Explicit;
  return result;
}

} // namespace

struct SenseAsrEncoderRuntime::Impl {
  struct Graph {
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context *ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph *graph = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue input;
    engine::core::TensorValue positions;
    ggml_tensor *output = nullptr;

    ~Graph() {
      host_plan.reset();
      if (backend != nullptr && graph != nullptr) {
        engine::core::release_backend_graph_resources(backend, graph);
      }
      if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
      }
      if (ggml != nullptr) {
        ggml_free(ggml);
      }
    }
  };

  Impl(std::shared_ptr<const SenseAsrAssets> assets_value,
       engine::core::ExecutionContext &execution_context_value,
       size_t graph_arena_bytes_value,
       engine::assets::TensorStorageType weight_storage)
      : assets(std::move(assets_value)),
        execution_context(&execution_context_value),
        graph_arena_bytes(graph_arena_bytes_value) {
    if (assets == nullptr || assets->model_weights == nullptr) {
      throw std::runtime_error(
          "SenseVoice encoder requires model assets and weights");
    }
    if (graph_arena_bytes == 0) {
      throw std::runtime_error(
          "SenseVoice encoder graph arena must be non-zero");
    }
    query_tokens = assets->config.encoder.query_tokens;
    weights =
        load_encoder_weights(*assets, *execution_context, weight_storage);
  }

  void ensure_graph(int64_t frames) {
    const auto &config = assets->config.encoder;
    const int64_t nq = static_cast<int64_t>(query_tokens.size());
    const int64_t total = nq + frames;
    if (total > config.max_frames) {
      throw std::runtime_error(
          "SenseVoice encoder input exceeds positional capacity");
    }
    if (cached_graph != nullptr && cached_graph->frames == frames &&
        cached_graph->backend == execution_context->backend()) {
      return;
    }

    const auto build_start = Clock::now();
    auto next = std::make_unique<Graph>();
    next->frames = frames;
    next->backend = execution_context->backend();
    ggml_init_params params{};
    params.mem_size = graph_arena_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    next->ggml = ggml_init(params);
    if (next->ggml == nullptr) {
      throw std::runtime_error(
          "failed to initialize SenseVoice encoder graph context");
    }

    engine::core::ModuleBuildContext context{
        next->ggml, "sense_asr.encoder", execution_context->backend_type()};
    const auto input_shape =
        engine::core::TensorShape::from_dims({1, total, config.input_size});
    auto input = engine::core::make_tensor(context, GGML_TYPE_F32, input_shape);
    ggml_set_input(input.tensor);
    next->input = input;

    auto hidden = engine::core::wrap_tensor(
        ggml_scale(context.ggml, input.tensor,
                   std::sqrt(static_cast<float>(config.d_model))),
        input_shape, GGML_TYPE_F32);
    next->positions =
        engine::core::make_tensor(context, GGML_TYPE_F32, input_shape);
    ggml_set_input(next->positions.tensor);
    hidden = engine::modules::AddModule{}.build(context, hidden, next->positions);

    hidden = engine::modules::sanm_projection_block(
        context, hidden,
        framework_sanm_weights(context, weights->stem, config,
                               config.input_size),
        block_config(config, config.input_size));
    const auto residual_config = block_config(config, config.d_model);
    for (size_t index = 0; index < weights->main_layers.size(); ++index) {
      hidden = engine::modules::sanm_residual_block(
          context, hidden,
          framework_sanm_weights(context, weights->main_layers[index], config,
                                 config.d_model),
          residual_config);
    }
    hidden = engine::modules::sanm_layer_norm(
        context, hidden, weights->after_norm, kLayerNormEpsilon);
    for (size_t index = 0; index < weights->timestamp_layers.size(); ++index) {
      hidden = engine::modules::sanm_residual_block(
          context, hidden,
          framework_sanm_weights(context, weights->timestamp_layers[index],
                                 config, config.d_model),
          residual_config);
    }
    hidden = engine::modules::sanm_layer_norm(
        context, hidden, weights->tp_norm, kLayerNormEpsilon);
    const auto logits = engine::modules::LinearModule(
                            {config.d_model, config.vocab_size, true})
                            .build(context, hidden, weights->ctc_lo);
    next->output = logits.tensor;
    ggml_set_output(next->output);

    next->graph =
        ggml_new_graph_custom(next->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(next->graph, next->output);
    engine::core::validate_backend_graph_supported(next->backend, next->graph,
                                                   "SenseVoice encoder");
    next->allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
    if (next->allocator == nullptr ||
        !ggml_gallocr_reserve(next->allocator, next->graph) ||
        !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
      throw std::runtime_error(
          "failed to allocate SenseVoice encoder graph tensors");
    }
    engine::core::prepare_host_graph_plan(*execution_context, next->graph,
                                          next->host_plan);

    cached_graph = std::move(next);
    engine::debug::timing_log_scalar(
        "sense_asr.encoder.graph_build_ms",
        engine::debug::elapsed_ms(build_start, Clock::now()));
  }

  SenseAsrEncoderOutput encode(const SenseAsrAudioFeatures &features) {
    const auto &config = assets->config.encoder;
    if (features.frames < 0 || features.feature_dim != config.input_size) {
      throw std::runtime_error("SenseVoice encoder input shape is invalid");
    }
    if (static_cast<int64_t>(features.values.size()) !=
        features.frames * features.feature_dim) {
      throw std::runtime_error(
          "SenseVoice encoder input value count mismatch");
    }
    if (features.frames <= 0) {
      return SenseAsrEncoderOutput{};
    }

    const auto encode_start = Clock::now();
    const int64_t nq = static_cast<int64_t>(query_tokens.size());
    const int64_t total = nq + features.frames;
    ensure_graph(features.frames);

    std::vector<float> input(static_cast<size_t>(total) * config.input_size, 0.0F);
    for (int64_t i = 0; i < nq; ++i) {
      const int32_t token = query_tokens[static_cast<size_t>(i)];
      if (token < 0 || token >= 16) {
        throw std::runtime_error(
            "SenseVoice query token is outside the embedding table");
      }
      const float *row =
          weights->embed_rows.data() + static_cast<size_t>(token) * static_cast<size_t>(config.input_size);
      std::copy(row, row + config.input_size,
                input.data() + static_cast<size_t>(i) * static_cast<size_t>(config.input_size));
    }
    std::copy(features.values.begin(), features.values.end(),
              input.begin() + static_cast<ptrdiff_t>(nq) * config.input_size);

    engine::core::write_tensor_f32(cached_graph->input, input);
    engine::core::write_tensor_f32(cached_graph->positions,
                                   make_posenc_input(total, config.input_size));
    engine::core::set_backend_threads(execution_context->backend(),
                                      execution_context->config().threads);
    // Bypass host graph plan to match reference behavior exactly
    const auto status = ggml_backend_graph_compute(execution_context->backend(),
                                                   cached_graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("SenseVoice encoder graph execution failed");
    }

    SenseAsrEncoderOutput output;
    output.logits = engine::core::read_tensor_f32(cached_graph->output);
    output.frames = total;
    output.vocab_size = config.vocab_size;
    engine::debug::timing_log_scalar(
        "sense_asr.encoder_ms",
        engine::debug::elapsed_ms(encode_start, Clock::now()));
    return output;
  }

  std::shared_ptr<const SenseAsrAssets> assets;
  engine::core::ExecutionContext *execution_context = nullptr;
  size_t graph_arena_bytes = 0;
  std::vector<int32_t> query_tokens;
  std::unique_ptr<EncoderWeights> weights;
  std::unique_ptr<Graph> cached_graph;
};

SenseAsrEncoderRuntime::SenseAsrEncoderRuntime(
    std::shared_ptr<const SenseAsrAssets> assets,
    engine::core::ExecutionContext &execution_context,
    size_t graph_arena_bytes,
    engine::assets::TensorStorageType weight_storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_storage)) {}

SenseAsrEncoderRuntime::~SenseAsrEncoderRuntime() = default;
SenseAsrEncoderRuntime::SenseAsrEncoderRuntime(
    SenseAsrEncoderRuntime &&) noexcept = default;
SenseAsrEncoderRuntime &SenseAsrEncoderRuntime::operator=(
    SenseAsrEncoderRuntime &&) noexcept = default;

void SenseAsrEncoderRuntime::prepare_capacity(int64_t frames) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  impl_->ensure_graph(frames);
}

void SenseAsrEncoderRuntime::set_query_tokens(
    std::vector<int32_t> query_tokens_value) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  if (query_tokens_value.empty()) {
    throw std::runtime_error("SenseVoice query tokens must not be empty");
  }
  for (int32_t token : query_tokens_value) {
    if (token < 0 || token >= 16) {
      throw std::runtime_error(
          "SenseVoice query token is outside the embedding table");
    }
  }
  impl_->query_tokens = std::move(query_tokens_value);
}

SenseAsrEncoderOutput
SenseAsrEncoderRuntime::encode(const SenseAsrAudioFeatures &features) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  return impl_->encode(features);
}

} // namespace engine::community_models::sense_asr

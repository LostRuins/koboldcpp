#include "engine/models/maya1/generator.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::maya1 {
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

constexpr int32_t kCodeEnd = 128258;
constexpr int32_t kSnacMin = 128266;
constexpr int32_t kSnacMax = 156937;

struct Maya1LlamaWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  std::shared_ptr<ggml_context> head_context;
  core::TensorValue embedding;
  modules::DecoderStackWeights stack;
  modules::NormWeights final_norm;
  modules::LinearWeights lm_head;
  int64_t lm_head_row_offset = 0;
};

std::vector<float> llama3_rope_factors(const Maya1Config &config) {
  constexpr double pi = 3.14159265358979323846;
  const double low_wavelength =
      static_cast<double>(config.rope.original_max_position_embeddings) /
      static_cast<double>(config.rope.low_freq_factor);
  const double high_wavelength =
      static_cast<double>(config.rope.original_max_position_embeddings) /
      static_cast<double>(config.rope.high_freq_factor);
  std::vector<float> factors(static_cast<size_t>(config.head_dim / 2), 1.0F);
  for (int64_t index = 0; index < config.head_dim / 2; ++index) {
    const double inverse_frequency =
        1.0 / std::pow(static_cast<double>(config.rope_theta),
                       static_cast<double>(2 * index) /
                           static_cast<double>(config.head_dim));
    const double wavelength = 2.0 * pi / inverse_frequency;
    double scaled = inverse_frequency;
    if (wavelength > low_wavelength) {
      scaled = inverse_frequency / static_cast<double>(config.rope.factor);
    } else if (wavelength >= high_wavelength) {
      const double smooth =
          (static_cast<double>(config.rope.original_max_position_embeddings) /
               wavelength -
           static_cast<double>(config.rope.low_freq_factor)) /
          (static_cast<double>(config.rope.high_freq_factor) -
           static_cast<double>(config.rope.low_freq_factor));
      scaled = (1.0 - smooth) * inverse_frequency /
                   static_cast<double>(config.rope.factor) +
               smooth * inverse_frequency;
    }
    factors[static_cast<size_t>(index)] =
        static_cast<float>(inverse_frequency / scaled);
  }
  return factors;
}

modules::DecoderActivationCastPolicy
activation_cast_policy(core::BackendType backend_type) {
  modules::DecoderActivationCastPolicy policy;
  if (backend_type != core::BackendType::Cuda) {
    return policy;
  }
  policy.enabled = true;
  policy.type = GGML_TYPE_BF16;
  policy.fused_round = true;
  policy.after_input_norm = true;
  policy.after_qkv_projection = true;
  policy.after_rope = true;
  policy.after_static_cache_update = true;
  policy.after_attention = true;
  policy.after_attention_output = true;
  policy.after_residual = true;
  policy.after_ffn_norm = true;
  policy.after_mlp_projection = true;
  policy.after_mlp_silu = true;
  policy.after_mlp_mul = true;
  policy.after_output = true;
  return policy;
}

Maya1LlamaWeights load_weights(const Maya1Assets &assets, ggml_backend_t backend,
                          core::BackendType backend_type, size_t context_bytes,
                          assets::TensorStorageType storage_type) {
  const auto &config = assets.config;
  const auto &source = *assets.model_weights;
  Maya1LlamaWeights out;
  out.store = std::make_shared<core::BackendWeightStore>(
      backend, backend_type, "maya1.ar.weights", context_bytes);
  out.embedding =
      out.store->load_tensor(source, "model.embed_tokens.weight", storage_type,
                             {config.vocab_size, config.hidden_size});
  const auto rope_factors = out.store->make_from_f32(
      core::TensorShape::from_dims({config.head_dim / 2}),
      assets::TensorStorageType::F32, llama3_rope_factors(config));
  out.stack.layers.reserve(static_cast<size_t>(config.layers));
  for (int64_t layer_index = 0; layer_index < config.layers; ++layer_index) {
    const std::string prefix = "model.layers." + std::to_string(layer_index);
    modules::DecoderLayerWeights layer;
    layer.input_norm = binding::norm_weight_from_source(
        *out.store, source, prefix + ".input_layernorm", config.hidden_size);
    layer.self_attention.q_weight = out.store->load_tensor(
        source, prefix + ".self_attn.q_proj.weight", storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    layer.self_attention.k_weight = out.store->load_tensor(
        source, prefix + ".self_attn.k_proj.weight", storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    layer.self_attention.v_weight = out.store->load_tensor(
        source, prefix + ".self_attn.v_proj.weight", storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    layer.self_attention.out_weight = out.store->load_tensor(
        source, prefix + ".self_attn.o_proj.weight", storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    layer.post_norm = binding::norm_weight_from_source(
        *out.store, source, prefix + ".post_attention_layernorm",
        config.hidden_size);
    layer.mlp.gate_proj = binding::linear_from_source(
        *out.store, source, prefix + ".mlp.gate_proj", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer.mlp.up_proj = binding::linear_from_source(
        *out.store, source, prefix + ".mlp.up_proj", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer.mlp.down_proj = binding::linear_from_source(
        *out.store, source, prefix + ".mlp.down_proj", storage_type,
        config.hidden_size, config.intermediate_size, false);
    layer.rope_frequency_factors = rope_factors;
    out.stack.layers.push_back(std::move(layer));
  }
  out.final_norm = binding::norm_weight_from_source(
      *out.store, source, "model.norm", config.hidden_size);
  out.lm_head = {out.embedding, std::nullopt};
  out.store->upload();
  if (backend_type == core::BackendType::Cuda) {
    const int64_t rows = kSnacMax - kCodeEnd + 1;
    out.head_context = std::shared_ptr<ggml_context>(
        ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
    if (!out.head_context) {
      throw std::runtime_error(
          "failed to initialize Maya1 sparse head context");
    }
    auto *base = out.embedding.tensor;
    auto *view =
        ggml_view_2d(out.head_context.get(), base, base->ne[0], rows,
                     base->nb[1], static_cast<size_t>(kCodeEnd) * base->nb[1]);
    if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("failed to initialize Maya1 sparse head view");
    }
    out.lm_head = {core::wrap_tensor(
                       view,
                       core::TensorShape::from_dims({rows, config.hidden_size}),
                       out.embedding.type),
                   std::nullopt};
    out.lm_head_row_offset = kCodeEnd;
  }
  return out;
}

modules::CausalDecoderConfig
llama_decoder_config(const Maya1Config &config, core::BackendType backend_type) {
  modules::CausalDecoderConfig out;
  out.stack.hidden_size = config.hidden_size;
  out.stack.intermediate_size = config.intermediate_size;
  out.stack.num_attention_heads = config.attention_heads;
  out.stack.num_key_value_heads = config.kv_heads;
  out.stack.head_dim = config.head_dim;
  out.stack.layers = config.layers;
  out.stack.rms_norm_eps = config.rms_norm_eps;
  out.stack.rope_theta = config.rope_theta;
  out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
  out.stack.use_qk_norm = false;
  out.stack.activation_cast = activation_cast_policy(backend_type);
  out.stack.runtime.attention.prefill_mode =
      modules::DecoderAttentionMode::FlashGroupedViewKV;
  out.stack.runtime.attention.static_mode =
      modules::DecoderAttentionMode::FlashGroupedViewKV;
  out.stack.runtime.static_cache.update_mode =
      modules::DecoderStaticCacheUpdateMode::DirectSetRows;
  if (backend_type != core::BackendType::Cpu) {
    out.stack.runtime.static_cache.set_rows_mode =
        modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.static_cache_type = GGML_TYPE_F16;
  }
  out.logits_size = config.vocab_size;
  out.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
  if (backend_type == core::BackendType::Cuda) {
    out.lm_head_input_type = GGML_TYPE_BF16;
  } else if (backend_type == core::BackendType::Vulkan ||
             backend_type == core::BackendType::Metal) {
    out.lm_head_input_type = GGML_TYPE_F16;
  }
  return out;
}

modules::CausalDecoderRuntimeConfig
llama_runtime_config(const Maya1Config &config, core::BackendType backend_type,
               const Maya1LlamaWeights &weights, size_t prefill_bytes,
               size_t decode_bytes) {
  modules::CausalDecoderRuntimeConfig out;
  out.trace_name = "maya1.ar";
  out.decoder = llama_decoder_config(config, backend_type);
  out.decoder.logits_size = weights.lm_head.weight.shape.dims[0];
  out.prefill_graph_arena_bytes = prefill_bytes;
  out.decode_graph_arena_bytes = decode_bytes;
  return out;
}

modules::CausalDecoderRuntimeWeights
llama_runtime_weights(const Maya1LlamaWeights &weights) {
  modules::CausalDecoderRuntimeWeights out;
  out.token_embedding = weights.embedding;
  out.stack = weights.stack;
  out.final_norm = weights.final_norm;
  out.lm_head = weights.lm_head;
  return out;
}

void apply_min_token_limit(std::vector<float> &logits, int64_t generated_tokens,
                           int64_t min_tokens, int64_t token_offset) {
  if (generated_tokens < min_tokens) {
    logits[static_cast<size_t>(kCodeEnd - token_offset)] =
        -std::numeric_limits<float>::infinity();
  }
}

} // namespace

struct Maya1LlamaGenerator::Impl {
  Impl(std::shared_ptr<const Maya1Assets> assets_in,
       core::ExecutionContext &execution, size_t prefill_bytes,
       size_t decode_bytes, size_t weight_bytes,
       assets::TensorStorageType storage_type)
      : assets(std::move(assets_in)), backend_type(execution.backend_type()),
        device(execution.config().device),
        weights(load_weights(*assets, execution.backend(), backend_type,
                             weight_bytes, storage_type)),
        llama_runtime(std::make_unique<modules::CausalDecoderRuntime>(
            execution,
            llama_runtime_config(assets->config, backend_type, weights, prefill_bytes,
                           decode_bytes),
            llama_runtime_weights(weights))),
        sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
            backend_type, device, "maya1.ar.sampling", "Maya1 AR",
            sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {}

  Maya1GenerationResult generate(const std::vector<int32_t> &prompt_ids,
                                 const Maya1GenerationOptions &options) {
    if (prompt_ids.empty()) {
      throw std::runtime_error("Maya1 AR prompt is empty");
    }
    if (options.min_tokens < 0 || options.max_tokens < options.min_tokens) {
      throw std::runtime_error("Maya1 token limits are invalid");
    }
    const int64_t required =
        static_cast<int64_t>(prompt_ids.size()) + options.max_tokens;
    if (required > assets->config.max_position_embeddings) {
      throw std::runtime_error("Maya1 prompt and output exceed model context");
    }

    const auto prefill_start = Clock::now();
    auto prefill = llama_runtime->prefill_tokens(prompt_ids);
    llama_runtime->start_decode_tokens(prefill.state, required);
    debug::timing_log_scalar("maya1.ar.prefill_ms",
                             debug::elapsed_ms(prefill_start, Clock::now()));

    sampling::HfSamplingOptions sampling_options;
    sampling_options.do_sample = true;
    sampling_options.temperature = options.temperature;
    sampling_options.top_k = 0;
    sampling_options.top_p = options.top_p;
    sampling_options.min_tokens_to_keep = 1;
    sampling_options.repetition_penalty = options.repetition_penalty;
    sampling::HfSampler sampler;
    sampling::HfSamplerScratch scratch;
    scratch.reserve_vocab(
        static_cast<size_t>(weights.lm_head.weight.shape.dims[0]));
    std::mt19937 fallback_rng(static_cast<uint32_t>(options.seed));
    const int64_t token_offset = weights.lm_head_row_offset;
    const int64_t token_end =
        token_offset + weights.lm_head.weight.shape.dims[0];
    std::vector<int32_t> history;
    history.reserve(prompt_ids.size() +
                    static_cast<size_t>(options.max_tokens));
    for (const int32_t token : prompt_ids) {
      if (token >= token_offset && token < token_end) {
        history.push_back(static_cast<int32_t>(token - token_offset));
      }
    }
    std::vector<float> logits = std::move(prefill.logits);
    uint64_t sample_call = 0;
    Maya1GenerationResult result;
    double sampling_ms = 0.0;
    double decode_graph_ms = 0.0;

    const auto decode_start = Clock::now();
    for (int64_t step = 0; step < options.max_tokens; ++step) {
      const float suppressed = -std::numeric_limits<float>::infinity();
      std::fill(logits.begin(), logits.begin() + (kCodeEnd - token_offset),
                suppressed);
      std::fill(logits.begin() + (kCodeEnd - token_offset) + 1,
                logits.begin() + (kSnacMin - token_offset), suppressed);
      std::fill(logits.begin() + (kSnacMax - token_offset) + 1, logits.end(),
                suppressed);
      apply_min_token_limit(logits, step, options.min_tokens, token_offset);
      const sampling::HfTorchSamplingState torch_state{
          &sampling_policy,
          options.seed,
          sample_call++,
          0,
          false,
          static_cast<uint64_t>(assets->config.vocab_size),
          static_cast<uint64_t>(token_offset)};
      const auto sampling_start = Clock::now();
      const int32_t local_token = sampler.sample(
          logits, history, sampling_options, scratch, fallback_rng,
          sampling_policy.cuda_fast_path ? &torch_state : nullptr, "Maya1 AR");
      sampling_ms += debug::elapsed_ms(sampling_start, Clock::now());
      const int32_t token = static_cast<int32_t>(local_token + token_offset);
      if (token == kCodeEnd && step >= options.min_tokens) {
        break;
      }
      history.push_back(local_token);
      if (token >= kSnacMin && token <= kSnacMax) {
        result.snac_tokens.push_back(token);
      }
      const auto decode_graph_start = Clock::now();
      logits = llama_runtime->decode_token(token).logits;
      decode_graph_ms += debug::elapsed_ms(decode_graph_start, Clock::now());
    }
    debug::timing_log_scalar("maya1.ar.decode_ms",
                             debug::elapsed_ms(decode_start, Clock::now()));
    debug::trace_log_scalar("maya1.ar.snac_tokens",
                            static_cast<int64_t>(result.snac_tokens.size()));
    debug::timing_log_scalar("maya1.ar.sampling_ms", sampling_ms);
    debug::timing_log_scalar("maya1.ar.decode_graph_ms", decode_graph_ms);
    return result;
  }

  std::shared_ptr<const Maya1Assets> assets;
  core::BackendType backend_type;
  int device;
  Maya1LlamaWeights weights;
  std::unique_ptr<modules::CausalDecoderRuntime> llama_runtime;
  sampling::TorchCudaSamplingPolicy sampling_policy;
};

Maya1LlamaGenerator::Maya1LlamaGenerator(std::shared_ptr<const Maya1Assets> assets,
                               core::ExecutionContext &execution,
                               size_t prefill_graph_arena_bytes,
                               size_t decode_graph_arena_bytes,
                               size_t weight_context_bytes,
                               assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution,
                                   prefill_graph_arena_bytes,
                                   decode_graph_arena_bytes,
                                   weight_context_bytes, weight_storage_type)) {
}

Maya1LlamaGenerator::~Maya1LlamaGenerator() = default;

Maya1GenerationResult
Maya1LlamaGenerator::generate(const std::vector<int32_t> &prompt_ids,
                         const Maya1GenerationOptions &options) {
  return impl_->generate(prompt_ids, options);
}

} // namespace engine::models::maya1

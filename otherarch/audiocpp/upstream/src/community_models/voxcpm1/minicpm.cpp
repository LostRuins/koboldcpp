#include "engine/community_models/voxcpm1/minicpm.h"

#include "minicpm_blocks.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::voxcpm1 {

namespace {

namespace weight_binding = engine::modules::binding;

VoxCPM1MiniCPMConfig residual_lm_config(const VoxCPM1Config &config) {
  VoxCPM1MiniCPMConfig out = config.lm;
  out.num_hidden_layers = config.residual_lm_num_layers;
  out.vocab_size = 0;
  out.no_rope = config.residual_lm_no_rope;
  return out;
}

VoxCPM1MiniCPMConfig
local_transformer_config(const VoxCPM1MiniCPMConfig &base,
                         const VoxCPM1LocalTransformerConfig &local) {
  VoxCPM1MiniCPMConfig out = base;
  out.hidden_size = local.hidden_dim;
  out.intermediate_size = local.ffn_dim;
  out.num_attention_heads = local.num_heads;
  out.num_hidden_layers = local.num_layers;
  out.num_key_value_heads = base.num_key_value_heads;
  out.kv_channels = local.kv_channels;
  out.vocab_size = 0;
  return out;
}

const std::vector<float> &
active_rope_factors(const VoxCPM1MiniCPMConfig &config) {
  if (config.max_position_embeddings >
      config.rope_scaling.original_max_position_embeddings) {
    return config.rope_scaling.long_factor;
  }
  return config.rope_scaling.short_factor;
}

float rope_attn_factor(const VoxCPM1MiniCPMConfig &config) {
  const auto original =
      static_cast<float>(config.rope_scaling.original_max_position_embeddings);
  if (original <= 1.0F ||
      config.max_position_embeddings <=
          config.rope_scaling.original_max_position_embeddings) {
    return 1.0F;
  }
  const float scale =
      static_cast<float>(config.max_position_embeddings) / original;
  return std::sqrt(1.0F + std::log(scale) / std::log(original));
}

engine::modules::LinearWeights
linear_weights(engine::core::BackendWeightStore &store,
               const engine::assets::TensorSource &source,
               const std::string &prefix,
               engine::assets::TensorStorageType storage_type,
               int64_t out_features, int64_t in_features, bool use_bias) {
  return weight_binding::linear_from_source(store, source, prefix, storage_type,
                                     out_features, in_features, use_bias);
}

VoxCPM1MiniCPMWeights load_minicpm_weights(
    engine::core::BackendWeightStore &store,
    const engine::assets::TensorSource &source, const std::string &embed_name,
    const std::string &layer_prefix, const std::string &norm_name,
    const VoxCPM1MiniCPMConfig &config,
    engine::assets::TensorStorageType storage_type, bool load_token_embedding) {
  const int64_t dim = head_dim(config);
  VoxCPM1MiniCPMWeights weights;
  weights.config = config;
  if (load_token_embedding && !embed_name.empty()) {
    weights.token_embedding =
        store.load_tensor(source, embed_name, storage_type,
                          {config.vocab_size, config.hidden_size});
  }
  if (!config.no_rope) {
    const auto &factors = active_rope_factors(config);
    if (static_cast<int64_t>(factors.size()) != dim / 2) {
      throw std::runtime_error("VoxCPM1 MiniCPM RoPE factor shape mismatch");
    }
    weights.rope_factors =
        store.make_from_f32(engine::core::TensorShape::from_dims({dim / 2}),
                            engine::assets::TensorStorageType::F32, factors);
    weights.rope_attn_factor = rope_attn_factor(config);
  }
  weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string lp =
        layer_prefix + "." + std::to_string(layer);
    VoxCPM1MiniCPMLayerWeights layer_weights;
    layer_weights.input_norm = weight_binding::norm_weight_from_source(
        store, source, lp + ".attn_norm", config.hidden_size);
    layer_weights.q_proj = linear_weights(
        store, source, lp + ".attn_q", storage_type,
        config.num_attention_heads * dim, config.hidden_size, false);
    layer_weights.k_proj = linear_weights(
        store, source, lp + ".attn_k", storage_type,
        config.num_key_value_heads * dim, config.hidden_size, false);
    layer_weights.v_proj = linear_weights(
        store, source, lp + ".attn_v", storage_type,
        config.num_key_value_heads * dim, config.hidden_size, false);
    layer_weights.o_proj = linear_weights(
        store, source, lp + ".attn_output", storage_type,
        config.hidden_size, config.num_attention_heads * dim, false);
    layer_weights.post_norm = weight_binding::norm_weight_from_source(
        store, source, lp + ".ffn_norm",
        config.hidden_size);
    layer_weights.gate_proj = linear_weights(
        store, source, lp + ".ffn_gate", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer_weights.up_proj = linear_weights(
        store, source, lp + ".ffn_up", storage_type,
        config.intermediate_size, config.hidden_size, false);
    layer_weights.down_proj = linear_weights(
        store, source, lp + ".ffn_down", storage_type,
        config.hidden_size, config.intermediate_size, false);
    weights.layers.push_back(std::move(layer_weights));
  }
  weights.norm = weight_binding::norm_weight_from_source(
      store, source, norm_name, config.hidden_size);
  return weights;
}

} // namespace

int64_t head_dim(const VoxCPM1MiniCPMConfig &config) {
  if (config.kv_channels <= 0 || config.num_attention_heads <= 0 ||
      config.num_key_value_heads <= 0) {
    throw std::runtime_error("VoxCPM1 MiniCPM attention config is invalid");
  }
  if (config.num_attention_heads % config.num_key_value_heads != 0) {
    throw std::runtime_error(
        "VoxCPM1 MiniCPM attention heads must be divisible by KV heads");
  }
  return config.kv_channels;
}

const VoxCPM1MiniCPMWeights &
select_minicpm_weights(const VoxCPM1ModelWeights &weights,
                       VoxCPM1MiniCPMKind kind) {
  switch (kind) {
  case VoxCPM1MiniCPMKind::BaseLM:
    return weights.base_lm;
  case VoxCPM1MiniCPMKind::ResidualLM:
    return weights.residual_lm;
  }
  throw std::runtime_error(
      "VoxCPM1 MiniCPM runtime received an unknown graph kind");
}

std::shared_ptr<const VoxCPM1ModelWeights>
load_model_weights(const VoxCPM1Assets &assets,
                   engine::core::ExecutionContext &execution_context,
                   size_t weight_context_bytes,
                   engine::assets::TensorStorageType storage_type) {
  auto weights = std::make_shared<VoxCPM1ModelWeights>();
  weights->store = std::make_shared<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "voxcpm1.model.weights", weight_context_bytes);
  auto &store = *weights->store;
  const auto &source = *assets.model_weights;
  weights->base_lm = load_minicpm_weights(store, source, "token_embd.weight",
                                          "blk", "output_norm",
                                          assets.config.lm, storage_type, true);
  weights->residual_lm = load_minicpm_weights(store, source, "",
                                              "residual_lm.blk",
                                              "residual_lm.output_norm",
                                              residual_lm_config(assets.config),
                                              storage_type, false);

  const auto encoder_config =
      local_transformer_config(assets.config.lm, assets.config.encoder);
  weights->feat_encoder.special_token =
      store.load_tensor(source, "locenc.special_token", storage_type,
                        {1, 1, 1, assets.config.encoder.hidden_dim});
  weights->feat_encoder.in_proj = linear_weights(
      store, source, "locenc.in_proj", storage_type,
      assets.config.encoder.hidden_dim, assets.config.feat_dim, true);
  weights->feat_encoder.encoder =
      load_minicpm_weights(store, source, "", "locenc.blk",
                           "locenc.output_norm",
                           encoder_config, storage_type, false);

  const auto dit_config =
      local_transformer_config(assets.config.lm, assets.config.dit);
  weights->dit.in_proj = linear_weights(
      store, source, "locdit.in_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.feat_dim, true);
  weights->dit.cond_proj = linear_weights(
      store, source, "locdit.cond_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.feat_dim, true);
  weights->dit.out_proj = linear_weights(
      store, source, "locdit.out_proj", storage_type,
      assets.config.feat_dim, assets.config.dit.hidden_dim, true);
  weights->dit.time_mlp_1 = linear_weights(
      store, source, "locdit.time_mlp.linear_1", storage_type,
      assets.config.dit.hidden_dim, assets.config.dit.hidden_dim, true);
  weights->dit.time_mlp_2 = linear_weights(
      store, source, "locdit.time_mlp.linear_2", storage_type,
      assets.config.dit.hidden_dim, assets.config.dit.hidden_dim, true);
  weights->dit.delta_time_mlp_1 = linear_weights(
      store, source, "locdit.delta_time_mlp.linear_1",
      storage_type, assets.config.dit.hidden_dim, assets.config.dit.hidden_dim,
      true);
  weights->dit.delta_time_mlp_2 = linear_weights(
      store, source, "locdit.delta_time_mlp.linear_2",
      storage_type, assets.config.dit.hidden_dim, assets.config.dit.hidden_dim,
      true);
  weights->dit.decoder =
      load_minicpm_weights(store, source, "", "locdit.blk",
                           "locdit.output_norm",
                           dit_config, storage_type, false);

  weights->projections.fsq_in_proj =
      linear_weights(store, source, "fsq.in_proj", storage_type,
                     assets.config.scalar_quantization_latent_dim,
                     assets.config.lm.hidden_size, true);
  weights->projections.fsq_out_proj =
      linear_weights(store, source, "fsq.out_proj", storage_type,
                     assets.config.lm.hidden_size,
                     assets.config.scalar_quantization_latent_dim, true);
  weights->projections.enc_to_lm_proj = linear_weights(
      store, source, "proj.enc_to_lm", storage_type,
      assets.config.lm.hidden_size, assets.config.encoder.hidden_dim, true);
  weights->projections.lm_to_dit_proj = linear_weights(
      store, source, "proj.lm_to_dit", storage_type,
      assets.config.dit.hidden_dim, assets.config.lm.hidden_size, true);
  weights->projections.res_to_dit_proj = linear_weights(
      store, source, "proj.res_to_dit", storage_type,
      assets.config.dit.hidden_dim, assets.config.lm.hidden_size, true);
  weights->projections.stop_proj = linear_weights(
      store, source, "stop.stop_proj", storage_type, assets.config.lm.hidden_size,
      assets.config.lm.hidden_size, true);
  weights->projections.stop_head =
      linear_weights(store, source, "stop.stop_head", storage_type, 2,
                     assets.config.lm.hidden_size, false);
  store.upload();
  return weights;
}



class VoxCPM1WeightsRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1Assets> assets,
       engine::core::ExecutionContext &execution_context,
       size_t weight_context_bytes,
       engine::assets::TensorStorageType weight_storage_type)
      : assets_(std::move(assets)), execution_context_(execution_context) {
    if (assets_ == nullptr) {
      throw std::runtime_error("VoxCPM1 weights runtime requires assets");
    }
    weights_ = load_model_weights(*assets_, execution_context_,
                                  weight_context_bytes, weight_storage_type);
  }

  const VoxCPM1Assets &assets() const noexcept { return *assets_; }
  const VoxCPM1ModelWeights &weights() const noexcept { return *weights_; }
  ggml_backend_t backend() const noexcept {
    return execution_context_.backend();
  }
  int threads() const noexcept {
    return std::max(1, execution_context_.config().threads);
  }
  bool weights_uploaded() const noexcept {
    return weights_ != nullptr && weights_->store != nullptr;
  }

private:
  std::shared_ptr<const VoxCPM1Assets> assets_;
  engine::core::ExecutionContext &execution_context_;
  std::shared_ptr<const VoxCPM1ModelWeights> weights_;
};

VoxCPM1WeightsRuntime::VoxCPM1WeightsRuntime(
    std::shared_ptr<const VoxCPM1Assets> assets,
    engine::core::ExecutionContext &execution_context,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   weight_context_bytes, weight_storage_type)) {
}

VoxCPM1WeightsRuntime::~VoxCPM1WeightsRuntime() = default;

const VoxCPM1Assets &VoxCPM1WeightsRuntime::assets() const noexcept {
  return impl_->assets();
}

const VoxCPM1ModelWeights &VoxCPM1WeightsRuntime::weights() const noexcept {
  return impl_->weights();
}

ggml_backend_t VoxCPM1WeightsRuntime::backend() const noexcept {
  return impl_->backend();
}

int VoxCPM1WeightsRuntime::threads() const noexcept { return impl_->threads(); }

bool VoxCPM1WeightsRuntime::weights_uploaded() const noexcept {
  return impl_->weights_uploaded();
}

class VoxCPM1TextEmbeddingRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error(
          "VoxCPM1 text embedding runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() {
    release_graph();
  }

  void release_runtime_memory() { release_graph(); }

  std::vector<float> embed_token(int32_t token_id) {
    const auto &config = weights_->assets().config.lm;
    if (token_id < 0 || token_id >= config.vocab_size) {
      throw std::runtime_error(
          "VoxCPM1 text embedding token id is out of range");
    }
    ggml_backend_tensor_set(token_id_, &token_id, 0, sizeof(token_id));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM1 text embedding graph compute failed");
    }
    std::vector<float> output(static_cast<size_t>(config.hidden_size), 0.0F);
    ggml_backend_tensor_get(output_, output.data(), 0,
                            output.size() * sizeof(float));
    return output;
  }

private:
  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    token_id_ = nullptr;
    output_ = nullptr;
    ctx_.reset();
  }

  void build(size_t graph_context_bytes) {
    const auto &config = weights_->assets().config.lm;
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 text embedding graph context bytes must be non-zero");
    }
    if (!weights_->weights().base_lm.token_embedding.has_value()) {
      throw std::runtime_error("VoxCPM1 text embedding weight is missing");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 text embedding graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.text_embedding"};
    token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
    if (mem_saver_) {
      ggml_set_input(token_id_);
    }
    auto token = engine::core::wrap_tensor(
        token_id_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
    auto embedding =
        engine::modules::EmbeddingModule(
            {config.vocab_size, config.hidden_size})
            .build(ctx, token, *weights_->weights().base_lm.token_embedding);
    const float scale =
        config.use_mup ? static_cast<float>(config.scale_emb) : 1.0F;
    embedding = scale_tensor(ctx, embedding, scale);
    embedding = engine::core::reshape_tensor(
        ctx, ensure_contiguous(ctx, embedding),
        engine::core::TensorShape::from_dims({config.hidden_size}));
    output_ = embedding.tensor;
    ggml_set_output(output_);
    if (mem_saver_ && output_->view_src != nullptr) {
      ggml_set_output(output_->view_src);
    }
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, output_);
    if (mem_saver_) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(weights_->backend()));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        if (gallocr_ != nullptr) {
          ggml_gallocr_free(gallocr_);
          gallocr_ = nullptr;
        }
        throw std::runtime_error(
            "failed to allocate VoxCPM1 text embedding graph");
      }
      return;
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    if (buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM1 text embedding graph");
    }
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  bool mem_saver_ = false;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *token_id_ = nullptr;
  ggml_tensor *output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM1TextEmbeddingRuntime::VoxCPM1TextEmbeddingRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1TextEmbeddingRuntime::~VoxCPM1TextEmbeddingRuntime() = default;

std::vector<float> VoxCPM1TextEmbeddingRuntime::embed_token(int32_t token_id) {
  return impl_->embed_token(token_id);
}

void VoxCPM1TextEmbeddingRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

struct MiniCPMLayerWithCacheOutput {
  engine::core::TensorValue output;
  engine::core::TensorValue key;
  engine::core::TensorValue value;
};

engine::core::TensorValue
mask_sequence(engine::core::ModuleBuildContext &ctx,
              const engine::core::TensorValue &input,
              const engine::core::TensorValue &mask) {
  auto repeated = engine::core::wrap_tensor(
      ggml_repeat(ctx.ggml, mask.tensor, input.tensor), input.shape,
      GGML_TYPE_F32);
  return engine::core::wrap_tensor(
      ggml_mul(ctx.ggml, input.tensor, repeated.tensor), input.shape,
      GGML_TYPE_F32);
}

MiniCPMLayerWithCacheOutput
minicpm_prefill_layer(engine::core::ModuleBuildContext &ctx,
                      const engine::core::TensorValue &input,
                      const engine::core::TensorValue &positions,
                      const VoxCPM1MiniCPMLayerWeights &layer,
                      const VoxCPM1MiniCPMWeights &weights) {
  const auto &config = weights.config;
  const int64_t dim = head_dim(config);
  const int64_t kv_repeats =
      config.num_attention_heads / config.num_key_value_heads;
  const engine::modules::AddModule add;
  auto hidden = engine::modules::RMSNormModule(
                    {config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, layer.input_norm);
  auto q = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_attention_heads * dim, false))
               .build(ctx, hidden, layer.q_proj);
  auto k = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.k_proj);
  auto v = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.v_proj);
  q = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, q, config.num_attention_heads, dim),
                         positions, weights);
  k = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, k, config.num_key_value_heads, dim),
                         positions, weights);
  v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
  auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank})
                     .build(ctx, q);
  auto k_heads = repeat_kv_heads(
      ctx,
      engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank})
          .build(ctx, k),
      kv_repeats);
  auto v_heads = repeat_kv_heads(
      ctx,
      engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank})
          .build(ctx, v),
      kv_repeats);
  auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, true);
  context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank})
                .build(ctx, context);
  context = engine::core::reshape_tensor(
      ctx, ensure_contiguous(ctx, context),
      engine::core::TensorShape::from_dims({input.shape.dims[0],
                                            input.shape.dims[1],
                                            config.num_attention_heads * dim}));
  auto attn = engine::modules::LinearModule(
                  binding::linear_config(config.num_attention_heads * dim,
                                         config.hidden_size, false))
                  .build(ctx, context, layer.o_proj);
  auto x =
      add.build(ctx, input,
                scale_tensor(ctx, attn,
                             config.use_mup ? config.scale_depth /
                                                  std::sqrt(static_cast<float>(
                                                      config.num_hidden_layers))
                                            : 1.0F));

  hidden = engine::modules::RMSNormModule(
               {config.hidden_size, config.rms_norm_eps, true, false})
               .build(ctx, x, layer.post_norm);
  auto gate = engine::modules::LinearModule(
                  binding::linear_config(config.hidden_size,
                                         config.intermediate_size, false))
                  .build(ctx, hidden, layer.gate_proj);
  gate = engine::modules::SiluModule{}.build(ctx, gate);
  auto up = engine::modules::LinearModule(
                binding::linear_config(config.hidden_size,
                                       config.intermediate_size, false))
                .build(ctx, hidden, layer.up_proj);
  auto gated = engine::modules::MulModule{}.build(ctx, gate, up);
  auto ff = engine::modules::LinearModule(
                binding::linear_config(config.intermediate_size,
                                       config.hidden_size, false))
                .build(ctx, gated, layer.down_proj);
  auto output = add.build(
      ctx, x,
      scale_tensor(ctx, ff,
                   config.use_mup
                       ? config.scale_depth / std::sqrt(static_cast<float>(
                                                  config.num_hidden_layers))
                       : 1.0F));
  return {output, k, v};
}


class VoxCPM1PromptPrefillRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), graph_context_bytes_(graph_context_bytes),
        mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error("VoxCPM1 prompt prefill runtime requires weights");
    }
    if (graph_context_bytes_ == 0) {
      throw std::runtime_error(
          "VoxCPM1 prompt prefill graph context bytes must be non-zero");
    }
  }

  ~Impl() { release_graph(); }

  void release_runtime_memory() { release_graph(); }

  VoxCPM1PromptPrefillOutput run(const VoxCPM1PromptPrefillInput &input) {
    const auto &config = weights_->assets().config;
    const int64_t hidden_size = config.lm.hidden_size;
    if (input.steps <= 0) {
      throw std::runtime_error("VoxCPM1 prompt prefill requires positive steps");
    }
    if (static_cast<int64_t>(input.input_embeddings.size()) !=
        input.steps * hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 prompt prefill input embedding size mismatch");
    }
    if (static_cast<int64_t>(input.current_embeddings.size()) !=
        input.steps * hidden_size) {
      throw std::runtime_error(
          "VoxCPM1 prompt prefill current embedding size mismatch");
    }
    if (static_cast<int64_t>(input.text_mask.size()) != input.steps ||
        static_cast<int64_t>(input.audio_mask.size()) != input.steps) {
      throw std::runtime_error("VoxCPM1 prompt prefill mask size mismatch");
    }
    if (sequence_steps_ != input.steps) {
      build(input.steps);
    }
    ggml_backend_tensor_set(input_embeddings_, input.input_embeddings.data(), 0,
                            input.input_embeddings.size() * sizeof(float));
    ggml_backend_tensor_set(current_embeddings_,
                            input.current_embeddings.data(), 0,
                            input.current_embeddings.size() * sizeof(float));
    ggml_backend_tensor_set(text_mask_, input.text_mask.data(), 0,
                            input.text_mask.size() * sizeof(float));
    ggml_backend_tensor_set(audio_mask_, input.audio_mask.data(), 0,
                            input.audio_mask.size() * sizeof(float));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM1 prompt prefill graph compute failed");
    }

    VoxCPM1PromptPrefillOutput output;
    output.lm_hidden.resize(static_cast<size_t>(hidden_size), 0.0F);
    output.residual_hidden.resize(static_cast<size_t>(hidden_size), 0.0F);
    ggml_backend_tensor_get(lm_hidden_output_, output.lm_hidden.data(), 0,
                            output.lm_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(residual_hidden_output_,
                            output.residual_hidden.data(), 0,
                            output.residual_hidden.size() * sizeof(float));
    output.base_state = read_state(base_keys_, base_values_,
                                   config.lm.num_key_value_heads *
                                       head_dim(config.lm));
    output.residual_state = read_state(residual_keys_, residual_values_,
                                       config.residual_lm_num_layers > 0
                                           ? config.lm.num_key_value_heads *
                                                 head_dim(config.lm)
                                           : 0);
    return output;
  }

private:
  engine::runtime::TransformerKVState
  read_state(const std::vector<ggml_tensor *> &keys,
             const std::vector<ggml_tensor *> &values, int64_t step_elems) {
    if (keys.size() != values.size() || step_elems <= 0) {
      throw std::runtime_error("VoxCPM1 prompt prefill KV state is invalid");
    }
    engine::runtime::TransformerKVState state;
    state.current_end = sequence_steps_;
    state.layers.resize(keys.size());
    const size_t layer_values =
        static_cast<size_t>(sequence_steps_ * step_elems);
    for (size_t layer = 0; layer < keys.size(); ++layer) {
      auto &layer_state = state.layers[layer];
      layer_state.valid_steps = sequence_steps_;
      layer_state.key.resize(layer_values);
      layer_state.value.resize(layer_values);
      ggml_backend_tensor_get(keys[layer], layer_state.key.data(), 0,
                              layer_state.key.size() * sizeof(float));
      ggml_backend_tensor_get(values[layer], layer_state.value.data(), 0,
                              layer_state.value.size() * sizeof(float));
    }
    return state;
  }

  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
      gallocr_ = nullptr;
    }
    graph_ = nullptr;
    input_embeddings_ = nullptr;
    current_embeddings_ = nullptr;
    text_mask_ = nullptr;
    audio_mask_ = nullptr;
    positions_ = nullptr;
    lm_hidden_output_ = nullptr;
    residual_hidden_output_ = nullptr;
    base_keys_.clear();
    base_values_.clear();
    residual_keys_.clear();
    residual_values_.clear();
    ctx_.reset();
    sequence_steps_ = 0;
  }

  void build(int64_t steps) {
    const auto &config = weights_->assets().config;
    const auto &model_weights = weights_->weights();
    release_graph();
    ggml_init_params params{graph_context_bytes_, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 prompt prefill graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm1.prompt_prefill"};
    auto input_embeddings = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims(
            {1, steps, config.lm.hidden_size}));
    input_embeddings_ = input_embeddings.tensor;
    if (mem_saver_) {
      ggml_set_input(input_embeddings_);
    }
    auto current_embeddings = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims(
            {1, steps, config.lm.hidden_size}));
    current_embeddings_ = current_embeddings.tensor;
    if (mem_saver_) {
      ggml_set_input(current_embeddings_);
    }
    text_mask_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, steps, 1);
    audio_mask_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, steps, 1);
    if (mem_saver_) {
      ggml_set_input(text_mask_);
      ggml_set_input(audio_mask_);
    }
    auto text_mask = engine::core::wrap_tensor(
        text_mask_, engine::core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    auto audio_mask = engine::core::wrap_tensor(
        audio_mask_, engine::core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps);
    if (mem_saver_) {
      ggml_set_input(positions_);
    }
    auto positions = engine::core::wrap_tensor(
        positions_, engine::core::TensorShape::from_dims({steps}),
        GGML_TYPE_I32);
    auto base_hidden = input_embeddings;
    for (const auto &layer : model_weights.base_lm.layers) {
      auto layer_out = minicpm_prefill_layer(
          ctx, base_hidden, positions, layer, model_weights.base_lm);
      base_hidden = layer_out.output;
      base_keys_.push_back(layer_out.key.tensor);
      base_values_.push_back(layer_out.value.tensor);
      if (mem_saver_) {
        ggml_set_output(base_keys_.back());
        if (base_keys_.back()->view_src != nullptr) {
          ggml_set_output(base_keys_.back()->view_src);
        }
        ggml_set_output(base_values_.back());
        if (base_values_.back()->view_src != nullptr) {
          ggml_set_output(base_values_.back()->view_src);
        }
      }
    }
    base_hidden = engine::modules::RMSNormModule(
                      {config.lm.hidden_size, config.lm.rms_norm_eps, true,
                       false})
                      .build(ctx, base_hidden, model_weights.base_lm.norm);

    auto fsq = engine::modules::LinearModule(
                   binding::linear_config(config.lm.hidden_size,
                                          config.scalar_quantization_latent_dim,
                                          true))
                   .build(ctx, base_hidden,
                          model_weights.projections.fsq_in_proj);
    fsq = engine::core::wrap_tensor(ggml_tanh(ctx.ggml, fsq.tensor), fsq.shape,
                                    GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, fsq.tensor,
                   static_cast<float>(config.scalar_quantization_scale)),
        fsq.shape, GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(ggml_round(ctx.ggml, fsq.tensor), fsq.shape,
                                    GGML_TYPE_F32);
    fsq = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, fsq.tensor,
                   1.0F / static_cast<float>(
                              config.scalar_quantization_scale)),
        fsq.shape, GGML_TYPE_F32);
    fsq = engine::modules::LinearModule(
              binding::linear_config(config.scalar_quantization_latent_dim,
                                     config.lm.hidden_size, true))
              .build(ctx, fsq, model_weights.projections.fsq_out_proj);
    auto masked_base = mask_sequence(ctx, base_hidden, text_mask);
    auto masked_fsq = mask_sequence(ctx, fsq, audio_mask);
    auto lm_hidden =
        engine::modules::AddModule{}.build(ctx, masked_base, masked_fsq);

    auto masked_current = mask_sequence(ctx, current_embeddings, audio_mask);
    auto residual_input =
        engine::modules::AddModule{}.build(ctx, lm_hidden, masked_current);

    auto residual_hidden = residual_input;
    for (const auto &layer : model_weights.residual_lm.layers) {
      auto layer_out = minicpm_prefill_layer(
          ctx, residual_hidden, positions, layer, model_weights.residual_lm);
      residual_hidden = layer_out.output;
      residual_keys_.push_back(layer_out.key.tensor);
      residual_values_.push_back(layer_out.value.tensor);
      if (mem_saver_) {
        ggml_set_output(residual_keys_.back());
        if (residual_keys_.back()->view_src != nullptr) {
          ggml_set_output(residual_keys_.back()->view_src);
        }
        ggml_set_output(residual_values_.back());
        if (residual_values_.back()->view_src != nullptr) {
          ggml_set_output(residual_values_.back()->view_src);
        }
      }
    }
    residual_hidden = engine::modules::RMSNormModule(
                          {config.lm.hidden_size, config.lm.rms_norm_eps, true,
                           false})
                          .build(ctx, residual_hidden,
                                 model_weights.residual_lm.norm);

    auto last_lm = engine::modules::SliceModule({1, steps - 1, 1})
                       .build(ctx, lm_hidden);
    last_lm = engine::core::reshape_tensor(
        ctx, ensure_contiguous(ctx, last_lm),
        engine::core::TensorShape::from_dims({config.lm.hidden_size}));
    lm_hidden_output_ = last_lm.tensor;
    auto last_residual = engine::modules::SliceModule({1, steps - 1, 1})
                             .build(ctx, residual_hidden);
    last_residual = engine::core::reshape_tensor(
        ctx, ensure_contiguous(ctx, last_residual),
        engine::core::TensorShape::from_dims({config.lm.hidden_size}));
    residual_hidden_output_ = last_residual.tensor;
    ggml_set_output(lm_hidden_output_);
    if (mem_saver_ && lm_hidden_output_->view_src != nullptr) {
      ggml_set_output(lm_hidden_output_->view_src);
    }
    ggml_set_output(residual_hidden_output_);
    if (mem_saver_ && residual_hidden_output_->view_src != nullptr) {
      ggml_set_output(residual_hidden_output_->view_src);
    }
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, lm_hidden_output_);
    ggml_build_forward_expand(graph_, residual_hidden_output_);
    if (mem_saver_) {
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(weights_->backend()));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        if (gallocr_ != nullptr) {
          ggml_gallocr_free(gallocr_);
          gallocr_ = nullptr;
        }
        release_graph();
        throw std::runtime_error(
            "failed to allocate VoxCPM1 prompt prefill graph");
      }
    } else {
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    }
    if (!mem_saver_ && buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM1 prompt prefill graph");
    }
    std::vector<int32_t> position_ids(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
      position_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions_, position_ids.data(), 0,
                            position_ids.size() * sizeof(int32_t));
    sequence_steps_ = steps;
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  size_t graph_context_bytes_ = 0;
  bool mem_saver_ = false;
  int64_t sequence_steps_ = 0;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *input_embeddings_ = nullptr;
  ggml_tensor *current_embeddings_ = nullptr;
  ggml_tensor *text_mask_ = nullptr;
  ggml_tensor *audio_mask_ = nullptr;
  ggml_tensor *positions_ = nullptr;
  ggml_tensor *lm_hidden_output_ = nullptr;
  ggml_tensor *residual_hidden_output_ = nullptr;
  std::vector<ggml_tensor *> base_keys_;
  std::vector<ggml_tensor *> base_values_;
  std::vector<ggml_tensor *> residual_keys_;
  std::vector<ggml_tensor *> residual_values_;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM1PromptPrefillRuntime::VoxCPM1PromptPrefillRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM1PromptPrefillRuntime::~VoxCPM1PromptPrefillRuntime() = default;

VoxCPM1PromptPrefillOutput
VoxCPM1PromptPrefillRuntime::run(const VoxCPM1PromptPrefillInput &input) {
  return impl_->run(input);
}

void VoxCPM1PromptPrefillRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

engine::core::TensorValue
minicpm_layer_with_static_cache(engine::core::ModuleBuildContext &ctx,
                                const engine::core::TensorValue &input,
                                const engine::core::TensorValue &positions,
                                const engine::core::TensorValue &cache_slot,
                                const engine::core::TensorValue &attention_mask,
                                const engine::core::TensorValue &cache_key,
                                const engine::core::TensorValue &cache_value,
                                const VoxCPM1MiniCPMLayerWeights &layer,
                                const VoxCPM1MiniCPMWeights &weights) {
  const auto &config = weights.config;
  const int64_t dim = head_dim(config);
  const engine::modules::AddModule add;
  auto hidden = engine::modules::RMSNormModule(
                    {config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, layer.input_norm);
  auto q = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_attention_heads * dim, false))
               .build(ctx, hidden, layer.q_proj);
  auto k = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.k_proj);
  auto v = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.num_key_value_heads * dim, false))
               .build(ctx, hidden, layer.v_proj);
  q = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, q, config.num_attention_heads, dim),
                         positions, weights);
  k = apply_minicpm_rope(ctx,
                         reshape_heads(ctx, k, config.num_key_value_heads, dim),
                         positions, weights);
  v = reshape_heads(ctx, v, config.num_key_value_heads, dim);

  const engine::modules::FastKVSetRowsModule set_rows;
  auto updated_key = set_rows.build(ctx, cache_key, k, cache_slot);
  auto updated_value = set_rows.build(ctx, cache_value, v, cache_slot);

  auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank})
                     .build(ctx, q);
  auto k_heads =
      engine::modules::TransposeModule({{0, 2, 1, 3}, updated_key.shape.rank})
          .build(ctx, updated_key);
  auto v_heads =
      engine::modules::TransposeModule({{0, 2, 1, 3}, updated_value.shape.rank})
          .build(ctx, updated_value);
  auto context = flash_attention_from_grouped_heads(
      ctx, q_heads, k_heads, v_heads, dim, attention_mask);
  context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank})
                .build(ctx, context);
  context = engine::core::reshape_tensor(
      ctx, ensure_contiguous(ctx, context),
      engine::core::TensorShape::from_dims({input.shape.dims[0],
                                            input.shape.dims[1],
                                            config.num_attention_heads * dim}));
  auto attn = engine::modules::LinearModule(
                  binding::linear_config(config.num_attention_heads * dim,
                                         config.hidden_size, false))
                  .build(ctx, context, layer.o_proj);
  auto x =
      add.build(ctx, input,
                scale_tensor(ctx, attn,
                             config.use_mup ? config.scale_depth /
                                                  std::sqrt(static_cast<float>(
                                                      config.num_hidden_layers))
                                            : 1.0F));

  hidden = engine::modules::RMSNormModule(
               {config.hidden_size, config.rms_norm_eps, true, false})
               .build(ctx, x, layer.post_norm);
  auto gate = engine::modules::LinearModule(
                  binding::linear_config(config.hidden_size,
                                         config.intermediate_size, false))
                  .build(ctx, hidden, layer.gate_proj);
  gate = engine::modules::SiluModule{}.build(ctx, gate);
  auto up = engine::modules::LinearModule(
                binding::linear_config(config.hidden_size,
                                       config.intermediate_size, false))
                .build(ctx, hidden, layer.up_proj);
  auto gated = engine::modules::MulModule{}.build(ctx, gate, up);
  auto ff = engine::modules::LinearModule(
                binding::linear_config(config.intermediate_size,
                                       config.hidden_size, false))
                .build(ctx, gated, layer.down_proj);
  return add.build(
      ctx, x,
      scale_tensor(ctx, ff,
                   config.use_mup
                       ? config.scale_depth / std::sqrt(static_cast<float>(
                                                  config.num_hidden_layers))
                       : 1.0F));
}

const char *minicpm_kind_name(VoxCPM1MiniCPMKind kind) {
  switch (kind) {
  case VoxCPM1MiniCPMKind::BaseLM:
    return "base_lm";
  case VoxCPM1MiniCPMKind::ResidualLM:
    return "residual_lm";
  }
  throw std::runtime_error(
      "VoxCPM1 MiniCPM runtime received an unknown graph kind");
}


class VoxCPM1MiniCPMStepRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
       VoxCPM1MiniCPMKind kind, int64_t cache_steps, size_t graph_context_bytes)
      : weights_(std::move(weights)), kind_(kind), cache_steps_(cache_steps),
        graph_context_bytes_(graph_context_bytes) {
    if (weights_ == nullptr) {
      throw std::runtime_error("VoxCPM1 MiniCPM step runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() {
    release_runtime_memory();
  }

  void reset() {
    ensure_graph();
    engine::runtime::TransformerKVState state;
    state.current_end = 0;
    state.layers.resize(
        select_minicpm_weights(weights_->weights(), kind_).layers.size());
    step_cache_.import_state(state);
  }

  void import_state(const engine::runtime::TransformerKVState &state) {
    ensure_graph();
    step_cache_.import_state(state);
  }

  engine::runtime::TransformerKVState export_state() const {
    return step_cache_.export_state();
  }

  VoxCPM1MiniCPMStepOutput run_step(const std::vector<float> &embedding) {
    ensure_graph();
    const auto &config =
        select_minicpm_weights(weights_->weights(), kind_).config;
    if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
      throw std::runtime_error("VoxCPM1 MiniCPM step embedding size mismatch");
    }
    if (step_cache_.valid_steps() >= cache_steps_) {
      throw std::runtime_error("VoxCPM1 MiniCPM step exceeds cache capacity");
    }
    ggml_backend_tensor_set(input_, embedding.data(), 0,
                            embedding.size() * sizeof(float));
    const int32_t position = static_cast<int32_t>(step_cache_.current_end());
    ggml_backend_tensor_set(position_, &position, 0, sizeof(position));
    const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
    ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));
    std::fill(attention_mask_buffer_.begin(), attention_mask_buffer_.end(),
              ggml_fp32_to_fp16(-INFINITY));
    for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
      attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
    }
    attention_mask_buffer_[static_cast<size_t>(cache_slot)] =
        ggml_fp32_to_fp16(0.0F);
    ggml_backend_tensor_set(attention_mask_, attention_mask_buffer_.data(), 0,
                            attention_mask_buffer_.size() *
                                sizeof(ggml_fp16_t));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("VoxCPM1 MiniCPM ") +
                               minicpm_kind_name(kind_) +
                               " step graph compute failed");
    }
    VoxCPM1MiniCPMStepOutput output;
    output.position = step_cache_.current_end();
    output.hidden.resize(static_cast<size_t>(config.hidden_size), 0.0F);
    ggml_backend_tensor_get(hidden_output_, output.hidden.data(), 0,
                            output.hidden.size() * sizeof(float));
    step_cache_.advance_after_direct_append(1);
    return output;
  }

  void release_runtime_memory() { release_graph(); }

private:
  void ensure_graph() {
    if (graph_ == nullptr) {
      build(graph_context_bytes_);
    }
  }

  void release_graph() {
    if (graph_ != nullptr) {
      engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    }
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
      buffer_ = nullptr;
    }
    graph_ = nullptr;
    input_ = nullptr;
    position_ = nullptr;
    cache_slot_ = nullptr;
    attention_mask_ = nullptr;
    hidden_output_ = nullptr;
    attention_mask_buffer_.clear();
    step_cache_ = engine::runtime::TransformerKVCache();
    ctx_.reset();
  }

  void build(size_t graph_context_bytes) {
    if (cache_steps_ <= 0) {
      throw std::runtime_error(
          "VoxCPM1 MiniCPM step graph requires positive cache capacity");
    }
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM1 MiniCPM step graph context bytes must be non-zero");
    }
    release_graph();
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM1 MiniCPM step graph context");
    }
    const auto &lm_weights = select_minicpm_weights(weights_->weights(), kind_);
    const auto &config = lm_weights.config;
    const int64_t dim = head_dim(config);
    const std::string graph_name =
        std::string("voxcpm1.") + minicpm_kind_name(kind_) + ".step";
    engine::core::ModuleBuildContext ctx{ctx_.get(), graph_name.c_str()};
    auto x = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 1, config.hidden_size}));
    input_ = x.tensor;
    position_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
    auto position = engine::core::wrap_tensor(
        position_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
    cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
    auto cache_slot = engine::core::wrap_tensor(
        cache_slot_, engine::core::TensorShape::from_dims({1}), GGML_TYPE_I32);
    attention_mask_ =
        ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
    auto attention_mask = engine::core::wrap_tensor(
        attention_mask_,
        engine::core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
        GGML_TYPE_F16);

    std::vector<engine::core::TensorValue> cache_keys;
    std::vector<engine::core::TensorValue> cache_values;
    cache_keys.reserve(static_cast<size_t>(config.num_hidden_layers));
    cache_values.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (const auto &layer : lm_weights.layers) {
      cache_keys.push_back(engine::core::make_tensor(
          ctx, GGML_TYPE_F32,
          engine::core::TensorShape::from_dims(
              {1, cache_steps_, config.num_key_value_heads, dim})));
      cache_values.push_back(engine::core::make_tensor(
          ctx, GGML_TYPE_F32,
          engine::core::TensorShape::from_dims(
              {1, cache_steps_, config.num_key_value_heads, dim})));
      x = minicpm_layer_with_static_cache(
          ctx, x, position, cache_slot, attention_mask, cache_keys.back(),
          cache_values.back(), layer, lm_weights);
    }
    step_cache_ = engine::runtime::TransformerKVCache(
        cache_steps_, config.num_key_value_heads * dim, std::move(cache_keys),
        std::move(cache_values));
    x = engine::modules::RMSNormModule(
            {config.hidden_size, config.rms_norm_eps, true, false})
            .build(ctx, x, lm_weights.norm);
    hidden_output_ = x.tensor;
    ggml_set_output(hidden_output_);
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, hidden_output_);
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    if (buffer_ == nullptr) {
      throw std::runtime_error("failed to allocate VoxCPM1 MiniCPM step graph");
    }
    attention_mask_buffer_.assign(static_cast<size_t>(cache_steps_),
                                  ggml_fp32_to_fp16(-INFINITY));
  }

  std::shared_ptr<const VoxCPM1WeightsRuntime> weights_;
  VoxCPM1MiniCPMKind kind_ = VoxCPM1MiniCPMKind::BaseLM;
  int64_t cache_steps_ = 0;
  size_t graph_context_bytes_ = 0;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *input_ = nullptr;
  ggml_tensor *position_ = nullptr;
  ggml_tensor *cache_slot_ = nullptr;
  ggml_tensor *attention_mask_ = nullptr;
  ggml_tensor *hidden_output_ = nullptr;
  std::vector<ggml_fp16_t> attention_mask_buffer_;
  engine::runtime::TransformerKVCache step_cache_;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
};

VoxCPM1MiniCPMStepRuntime::VoxCPM1MiniCPMStepRuntime(
    std::shared_ptr<const VoxCPM1WeightsRuntime> weights,
    VoxCPM1MiniCPMKind kind, int64_t cache_steps, size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), kind, cache_steps,
                                   graph_context_bytes)) {}

VoxCPM1MiniCPMStepRuntime::~VoxCPM1MiniCPMStepRuntime() = default;

void VoxCPM1MiniCPMStepRuntime::reset() { impl_->reset(); }

void VoxCPM1MiniCPMStepRuntime::import_state(
    const engine::runtime::TransformerKVState &state) {
  impl_->import_state(state);
}

engine::runtime::TransformerKVState
VoxCPM1MiniCPMStepRuntime::export_state() const {
  return impl_->export_state();
}

VoxCPM1MiniCPMStepOutput
VoxCPM1MiniCPMStepRuntime::run_step(const std::vector<float> &embedding) {
  return impl_->run_step(embedding);
}

void VoxCPM1MiniCPMStepRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

} // namespace engine::community_models::voxcpm1


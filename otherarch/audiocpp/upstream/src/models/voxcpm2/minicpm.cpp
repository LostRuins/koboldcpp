#include "engine/models/voxcpm2/minicpm.h"

#include "minicpm_blocks.h"

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
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

namespace engine::models::voxcpm2 {

namespace {

namespace weight_binding = engine::modules::binding;

VoxCPM2MiniCPMConfig residual_lm_config(const VoxCPM2Config &config) {
  VoxCPM2MiniCPMConfig out = config.lm;
  out.num_hidden_layers = config.residual_lm_num_layers;
  out.vocab_size = 0;
  out.no_rope = config.residual_lm_no_rope;
  return out;
}

VoxCPM2MiniCPMConfig
local_transformer_config(const VoxCPM2MiniCPMConfig &base,
                         const VoxCPM2LocalTransformerConfig &local) {
  VoxCPM2MiniCPMConfig out = base;
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
active_rope_factors(const VoxCPM2MiniCPMConfig &config) {
  if (config.max_position_embeddings >
      config.rope_scaling.original_max_position_embeddings) {
    return config.rope_scaling.long_factor;
  }
  return config.rope_scaling.short_factor;
}

float rope_attn_factor(const VoxCPM2MiniCPMConfig &config) {
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

VoxCPM2MiniCPMWeights
load_minicpm_weights(engine::core::BackendWeightStore &store,
                     const engine::assets::TensorSource &source,
                     const std::string &prefix,
                     const VoxCPM2MiniCPMConfig &config,
                     engine::assets::TensorStorageType storage_type,
                     bool load_token_embedding, bool optimized) {
  const int64_t dim = head_dim(config);
  VoxCPM2MiniCPMWeights weights;
  weights.config = config;
  if (load_token_embedding) {
    weights.token_embedding =
        store.load_tensor(source, prefix + ".embed_tokens.weight", storage_type,
                          {config.vocab_size, config.hidden_size});
  }
  if (!config.no_rope) {
    const auto &factors = active_rope_factors(config);
    if (static_cast<int64_t>(factors.size()) != dim / 2) {
      throw std::runtime_error("VoxCPM2 MiniCPM RoPE factor shape mismatch");
    }
    weights.rope_factors =
        store.make_from_f32(engine::core::TensorShape::from_dims({dim / 2}),
                            engine::assets::TensorStorageType::F32, factors);
    weights.rope_attn_factor = rope_attn_factor(config);
  }
  weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string layer_prefix =
        prefix + ".layers." + std::to_string(layer);
    VoxCPM2MiniCPMLayerWeights layer_weights;
    layer_weights.input_norm = weight_binding::norm_weight_from_source(
        store, source, layer_prefix + ".input_layernorm", config.hidden_size);
    if (optimized) {
      layer_weights.qkv_proj =
          engine::modules::PackedLinearWeightsBuilder(
              {config.hidden_size,
               {{layer_prefix + ".self_attn.q_proj.weight", std::nullopt,
                 config.num_attention_heads * dim},
                {layer_prefix + ".self_attn.k_proj.weight", std::nullopt,
                 config.num_key_value_heads * dim},
                {layer_prefix + ".self_attn.v_proj.weight", std::nullopt,
                 config.num_key_value_heads * dim}},
               false})
              .build(store, source, storage_type);
    } else {
      layer_weights.q_proj = linear_weights(
          store, source, layer_prefix + ".self_attn.q_proj", storage_type,
          config.num_attention_heads * dim, config.hidden_size, false);
      layer_weights.k_proj = linear_weights(
          store, source, layer_prefix + ".self_attn.k_proj", storage_type,
          config.num_key_value_heads * dim, config.hidden_size, false);
      layer_weights.v_proj = linear_weights(
          store, source, layer_prefix + ".self_attn.v_proj", storage_type,
          config.num_key_value_heads * dim, config.hidden_size, false);
    }
    layer_weights.o_proj = linear_weights(
        store, source, layer_prefix + ".self_attn.o_proj", storage_type,
        config.hidden_size, config.num_attention_heads * dim, false);
    layer_weights.post_norm = weight_binding::norm_weight_from_source(
        store, source, layer_prefix + ".post_attention_layernorm",
        config.hidden_size);
    if (optimized) {
      layer_weights.gate_up_proj =
          engine::modules::PackedLinearWeightsBuilder(
              {config.hidden_size,
               {{layer_prefix + ".mlp.gate_proj.weight", std::nullopt,
                 config.intermediate_size},
                {layer_prefix + ".mlp.up_proj.weight", std::nullopt,
                 config.intermediate_size}},
               false})
              .build(store, source, storage_type);
    } else {
      layer_weights.gate_proj = linear_weights(
          store, source, layer_prefix + ".mlp.gate_proj", storage_type,
          config.intermediate_size, config.hidden_size, false);
      layer_weights.up_proj = linear_weights(
          store, source, layer_prefix + ".mlp.up_proj", storage_type,
          config.intermediate_size, config.hidden_size, false);
    }
    layer_weights.down_proj = linear_weights(
        store, source, layer_prefix + ".mlp.down_proj", storage_type,
        config.hidden_size, config.intermediate_size, false);
    weights.layers.push_back(std::move(layer_weights));
  }
  weights.norm = weight_binding::norm_weight_from_source(
      store, source, prefix + ".norm", config.hidden_size);
  return weights;
}

} // namespace

int64_t head_dim(const VoxCPM2MiniCPMConfig &config) {
  if (config.kv_channels <= 0 || config.num_attention_heads <= 0 ||
      config.num_key_value_heads <= 0) {
    throw std::runtime_error("VoxCPM2 MiniCPM attention config is invalid");
  }
  if (config.num_attention_heads % config.num_key_value_heads != 0) {
    throw std::runtime_error(
        "VoxCPM2 MiniCPM attention heads must be divisible by KV heads");
  }
  return config.kv_channels;
}

const VoxCPM2MiniCPMWeights &
select_minicpm_weights(const VoxCPM2ModelWeights &weights,
                       VoxCPM2MiniCPMKind kind) {
  switch (kind) {
  case VoxCPM2MiniCPMKind::BaseLM:
    return weights.base_lm;
  case VoxCPM2MiniCPMKind::ResidualLM:
    return weights.residual_lm;
  }
  throw std::runtime_error(
      "VoxCPM2 MiniCPM runtime received an unknown graph kind");
}

std::shared_ptr<const VoxCPM2ModelWeights>
load_model_weights(const VoxCPM2Assets &assets,
                   engine::core::ExecutionContext &execution_context,
                   size_t weight_context_bytes,
                   engine::assets::TensorStorageType storage_type,
                   bool optimized) {
  auto weights = std::make_shared<VoxCPM2ModelWeights>();
  weights->store = std::make_shared<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "voxcpm2.model.weights", weight_context_bytes);
  auto &store = *weights->store;
  const auto &source = *assets.model_weights;
  weights->base_lm =
      load_minicpm_weights(store, source, "base_lm", assets.config.lm,
                           storage_type, true, optimized);
  weights->residual_lm = load_minicpm_weights(store, source, "residual_lm",
                                              residual_lm_config(assets.config),
                                              storage_type, false, optimized);

  const auto encoder_config =
      local_transformer_config(assets.config.lm, assets.config.encoder);
  weights->feat_encoder.special_token =
      store.load_tensor(source, "feat_encoder.special_token", storage_type,
                        {1, 1, 1, assets.config.encoder.hidden_dim});
  weights->feat_encoder.in_proj = linear_weights(
      store, source, "feat_encoder.in_proj", storage_type,
      assets.config.encoder.hidden_dim, assets.config.feat_dim, true);
  weights->feat_encoder.encoder =
      load_minicpm_weights(store, source, "feat_encoder.encoder",
                           encoder_config, storage_type, false, optimized);

  const auto dit_config =
      local_transformer_config(assets.config.lm, assets.config.dit);
  weights->dit.in_proj = linear_weights(
      store, source, "feat_decoder.estimator.in_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.feat_dim, true);
  weights->dit.cond_proj = linear_weights(
      store, source, "feat_decoder.estimator.cond_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.feat_dim, true);
  weights->dit.out_proj = linear_weights(
      store, source, "feat_decoder.estimator.out_proj", storage_type,
      assets.config.feat_dim, assets.config.dit.hidden_dim, true);
  weights->dit.time_mlp_1 = linear_weights(
      store, source, "feat_decoder.estimator.time_mlp.linear_1", storage_type,
      assets.config.dit.hidden_dim, assets.config.dit.hidden_dim, true);
  weights->dit.time_mlp_2 = linear_weights(
      store, source, "feat_decoder.estimator.time_mlp.linear_2", storage_type,
      assets.config.dit.hidden_dim, assets.config.dit.hidden_dim, true);
  weights->dit.delta_time_mlp_1 = linear_weights(
      store, source, "feat_decoder.estimator.delta_time_mlp.linear_1",
      storage_type, assets.config.dit.hidden_dim, assets.config.dit.hidden_dim,
      true);
  weights->dit.delta_time_mlp_2 = linear_weights(
      store, source, "feat_decoder.estimator.delta_time_mlp.linear_2",
      storage_type, assets.config.dit.hidden_dim, assets.config.dit.hidden_dim,
      true);
  weights->dit.decoder =
      load_minicpm_weights(store, source, "feat_decoder.estimator.decoder",
                           dit_config, storage_type, false, optimized);

  weights->projections.fsq_in_proj =
      linear_weights(store, source, "fsq_layer.in_proj", storage_type,
                     assets.config.scalar_quantization_latent_dim,
                     assets.config.lm.hidden_size, true);
  weights->projections.fsq_out_proj =
      linear_weights(store, source, "fsq_layer.out_proj", storage_type,
                     assets.config.lm.hidden_size,
                     assets.config.scalar_quantization_latent_dim, true);
  weights->projections.enc_to_lm_proj = linear_weights(
      store, source, "enc_to_lm_proj", storage_type,
      assets.config.lm.hidden_size, assets.config.encoder.hidden_dim, true);
  weights->projections.lm_to_dit_proj = linear_weights(
      store, source, "lm_to_dit_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.lm.hidden_size, true);
  weights->projections.res_to_dit_proj = linear_weights(
      store, source, "res_to_dit_proj", storage_type,
      assets.config.dit.hidden_dim, assets.config.lm.hidden_size, true);
  weights->projections.fusion_concat_proj = linear_weights(
      store, source, "fusion_concat_proj", storage_type,
      assets.config.lm.hidden_size, assets.config.lm.hidden_size * 2, true);
  weights->projections.stop_proj = linear_weights(
      store, source, "stop_proj", storage_type, assets.config.lm.hidden_size,
      assets.config.lm.hidden_size, true);
  weights->projections.stop_head =
      linear_weights(store, source, "stop_head", storage_type, 2,
                     assets.config.lm.hidden_size, false);
  store.upload();
  return weights;
}

class VoxCPM2WeightsRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM2Assets> assets,
       engine::core::ExecutionContext &execution_context,
       size_t weight_context_bytes,
       engine::assets::TensorStorageType weight_storage_type, bool optimized)
      : assets_(std::move(assets)), execution_context_(execution_context) {
    if (assets_ == nullptr) {
      throw std::runtime_error("VoxCPM2 weights runtime requires assets");
    }
    weights_ =
        load_model_weights(*assets_, execution_context_, weight_context_bytes,
                           weight_storage_type, optimized);
  }

  const VoxCPM2Assets &assets() const noexcept { return *assets_; }
  const VoxCPM2ModelWeights &weights() const noexcept { return *weights_; }
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
  std::shared_ptr<const VoxCPM2Assets> assets_;
  engine::core::ExecutionContext &execution_context_;
  std::shared_ptr<const VoxCPM2ModelWeights> weights_;
};

VoxCPM2WeightsRuntime::VoxCPM2WeightsRuntime(
    std::shared_ptr<const VoxCPM2Assets> assets,
    engine::core::ExecutionContext &execution_context,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type, bool optimized)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   weight_context_bytes, weight_storage_type,
                                   optimized)) {}

VoxCPM2WeightsRuntime::~VoxCPM2WeightsRuntime() = default;

const VoxCPM2Assets &VoxCPM2WeightsRuntime::assets() const noexcept {
  return impl_->assets();
}

const VoxCPM2ModelWeights &VoxCPM2WeightsRuntime::weights() const noexcept {
  return impl_->weights();
}

ggml_backend_t VoxCPM2WeightsRuntime::backend() const noexcept {
  return impl_->backend();
}

int VoxCPM2WeightsRuntime::threads() const noexcept { return impl_->threads(); }

bool VoxCPM2WeightsRuntime::weights_uploaded() const noexcept {
  return impl_->weights_uploaded();
}

class VoxCPM2TextEmbeddingRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
       size_t graph_context_bytes, bool mem_saver)
      : weights_(std::move(weights)), mem_saver_(mem_saver) {
    if (weights_ == nullptr) {
      throw std::runtime_error(
          "VoxCPM2 text embedding runtime requires weights");
    }
    build(graph_context_bytes);
  }

  ~Impl() {
    engine::core::release_backend_graph_resources(weights_->backend(), graph_);
    if (buffer_ != nullptr) {
      ggml_backend_buffer_free(buffer_);
    }
    if (gallocr_ != nullptr) {
      ggml_gallocr_free(gallocr_);
    }
  }

  std::vector<float> embed_token(int32_t token_id) {
    const auto &config = weights_->assets().config.lm;
    if (token_id < 0 || token_id >= config.vocab_size) {
      throw std::runtime_error(
          "VoxCPM2 text embedding token id is out of range");
    }
    ggml_backend_tensor_set(token_id_, &token_id, 0, sizeof(token_id));
    engine::core::set_backend_threads(weights_->backend(), weights_->threads());
    const ggml_status status =
        engine::core::compute_backend_graph(weights_->backend(), graph_);
    ggml_backend_synchronize(weights_->backend());
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("VoxCPM2 text embedding graph compute failed");
    }
    std::vector<float> output(static_cast<size_t>(config.hidden_size), 0.0F);
    ggml_backend_tensor_get(output_, output.data(), 0,
                            output.size() * sizeof(float));
    return output;
  }

private:
  void build(size_t graph_context_bytes) {
    const auto &config = weights_->assets().config.lm;
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM2 text embedding graph context bytes must be non-zero");
    }
    if (!weights_->weights().base_lm.token_embedding.has_value()) {
      throw std::runtime_error("VoxCPM2 text embedding weight is missing");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM2 text embedding graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm2.text_embedding"};
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
            "failed to allocate VoxCPM2 text embedding graph");
      }
      return;
    }
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_->backend());
    if (buffer_ == nullptr) {
      throw std::runtime_error(
          "failed to allocate VoxCPM2 text embedding graph");
    }
  }

  std::shared_ptr<const VoxCPM2WeightsRuntime> weights_;
  bool mem_saver_ = false;
  std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
  ggml_tensor *token_id_ = nullptr;
  ggml_tensor *output_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM2TextEmbeddingRuntime::VoxCPM2TextEmbeddingRuntime(
    std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
    size_t graph_context_bytes, bool mem_saver)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes,
                                   mem_saver)) {}

VoxCPM2TextEmbeddingRuntime::~VoxCPM2TextEmbeddingRuntime() = default;

std::vector<float> VoxCPM2TextEmbeddingRuntime::embed_token(int32_t token_id) {
  return impl_->embed_token(token_id);
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
                      const VoxCPM2MiniCPMLayerWeights &layer,
                      const VoxCPM2MiniCPMWeights &weights) {
  const auto &config = weights.config;
  const int64_t dim = head_dim(config);
  const int64_t kv_repeats =
      config.num_attention_heads / config.num_key_value_heads;
  const engine::modules::AddModule add;
  auto hidden = engine::modules::RMSNormModule(
                    {config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, layer.input_norm);
  engine::core::TensorValue q, k, v;
  if (layer.qkv_proj) {
    auto packed = engine::modules::LinearModule(
                      binding::linear_config(config.hidden_size,
                                             (config.num_attention_heads +
                                              2 * config.num_key_value_heads) *
                                                 dim,
                                             false))
                      .build(ctx, hidden, *layer.qkv_proj);
    const auto axis = static_cast<int>(hidden.shape.rank - 1);
    q = engine::modules::SliceModule(
            {axis, 0, config.num_attention_heads * dim})
            .build(ctx, packed);
    k = engine::modules::SliceModule({axis, config.num_attention_heads * dim,
                                      config.num_key_value_heads * dim})
            .build(ctx, packed);
    v = engine::modules::SliceModule(
            {axis,
             (config.num_attention_heads + config.num_key_value_heads) * dim,
             config.num_key_value_heads * dim})
            .build(ctx, packed);
  } else {
    q = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_attention_heads * dim, false))
            .build(ctx, hidden, layer.q_proj);
    k = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_key_value_heads * dim, false))
            .build(ctx, hidden, layer.k_proj);
    v = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_key_value_heads * dim, false))
            .build(ctx, hidden, layer.v_proj);
  }
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
  engine::core::TensorValue gate, up;
  if (layer.gate_up_proj) {
    auto packed =
        engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   2 * config.intermediate_size, false))
            .build(ctx, hidden, *layer.gate_up_proj);
    const auto axis = static_cast<int>(hidden.shape.rank - 1);
    gate = engine::modules::SliceModule({axis, 0, config.intermediate_size})
               .build(ctx, packed);
    up = engine::modules::SliceModule(
             {axis, config.intermediate_size, config.intermediate_size})
             .build(ctx, packed);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
  } else {
    gate = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.intermediate_size, false))
               .build(ctx, hidden, layer.gate_proj);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    up = engine::modules::LinearModule(
             binding::linear_config(config.hidden_size,
                                    config.intermediate_size, false))
             .build(ctx, hidden, layer.up_proj);
  }
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


class VoxCPM2PromptPrefillRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
       size_t graph_context_bytes)
      : weights_(std::move(weights)), graph_context_bytes_(graph_context_bytes) {
    if (weights_ == nullptr) {
      throw std::runtime_error("VoxCPM2 prompt prefill runtime requires weights");
    }
    if (graph_context_bytes_ == 0) {
      throw std::runtime_error(
          "VoxCPM2 prompt prefill graph context bytes must be non-zero");
    }
  }

  ~Impl() { release_graph(); }

  VoxCPM2PromptPrefillOutput run(const VoxCPM2PromptPrefillInput &input) {
    const auto &config = weights_->assets().config;
    const int64_t hidden_size = config.lm.hidden_size;
    if (input.steps <= 0) {
      throw std::runtime_error("VoxCPM2 prompt prefill requires positive steps");
    }
    if (static_cast<int64_t>(input.input_embeddings.size()) !=
        input.steps * hidden_size) {
      throw std::runtime_error(
          "VoxCPM2 prompt prefill input embedding size mismatch");
    }
    if (static_cast<int64_t>(input.current_embeddings.size()) !=
        input.steps * hidden_size) {
      throw std::runtime_error(
          "VoxCPM2 prompt prefill current embedding size mismatch");
    }
    if (static_cast<int64_t>(input.text_mask.size()) != input.steps ||
        static_cast<int64_t>(input.audio_mask.size()) != input.steps) {
      throw std::runtime_error("VoxCPM2 prompt prefill mask size mismatch");
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
      throw std::runtime_error("VoxCPM2 prompt prefill graph compute failed");
    }

    VoxCPM2PromptPrefillOutput output;
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

  void release_runtime_memory() { release_graph(); }

private:
  engine::runtime::TransformerKVState
  read_state(const std::vector<ggml_tensor *> &keys,
             const std::vector<ggml_tensor *> &values, int64_t step_elems) {
    if (keys.size() != values.size() || step_elems <= 0) {
      throw std::runtime_error("VoxCPM2 prompt prefill KV state is invalid");
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
          "failed to initialize VoxCPM2 prompt prefill graph context");
    }
    engine::core::ModuleBuildContext ctx{ctx_.get(), "voxcpm2.prompt_prefill"};
    auto input_embeddings = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims(
            {1, steps, config.lm.hidden_size}));
    input_embeddings_ = input_embeddings.tensor;
    auto current_embeddings = engine::core::make_tensor(
        ctx, GGML_TYPE_F32,
        engine::core::TensorShape::from_dims(
            {1, steps, config.lm.hidden_size}));
    current_embeddings_ = current_embeddings.tensor;
    text_mask_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, steps, 1);
    audio_mask_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, steps, 1);
    auto text_mask = engine::core::wrap_tensor(
        text_mask_, engine::core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    auto audio_mask = engine::core::wrap_tensor(
        audio_mask_, engine::core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, steps);
    auto positions = engine::core::wrap_tensor(
        positions_, engine::core::TensorShape::from_dims({steps}),
        GGML_TYPE_I32);
    auto base_hidden = input_embeddings;
    for (const auto &layer : model_weights.base_lm.layers) {
      auto layer_out = minicpm_prefill_layer(
          ctx, base_hidden, positions, layer, model_weights.base_lm);
      base_hidden = layer_out.output;
      auto *key = ggml_cpy(ctx_.get(), layer_out.key.tensor,
                           ggml_dup_tensor(ctx_.get(), layer_out.key.tensor));
      auto *value =
          ggml_cpy(ctx_.get(), layer_out.value.tensor,
                   ggml_dup_tensor(ctx_.get(), layer_out.value.tensor));
      ggml_set_output(key);
      ggml_set_output(value);
      base_keys_.push_back(key);
      base_values_.push_back(value);
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
        engine::modules::ConcatModule({2}).build(ctx, lm_hidden, masked_current);
    residual_input =
        engine::modules::LinearModule(
            binding::linear_config(config.lm.hidden_size * 2,
                                   config.lm.hidden_size, true))
            .build(ctx, residual_input,
                   model_weights.projections.fusion_concat_proj);

    auto residual_hidden = residual_input;
    for (const auto &layer : model_weights.residual_lm.layers) {
      auto layer_out = minicpm_prefill_layer(
          ctx, residual_hidden, positions, layer, model_weights.residual_lm);
      residual_hidden = layer_out.output;
      auto *key = ggml_cpy(ctx_.get(), layer_out.key.tensor,
                           ggml_dup_tensor(ctx_.get(), layer_out.key.tensor));
      auto *value =
          ggml_cpy(ctx_.get(), layer_out.value.tensor,
                   ggml_dup_tensor(ctx_.get(), layer_out.value.tensor));
      ggml_set_output(key);
      ggml_set_output(value);
      residual_keys_.push_back(key);
      residual_values_.push_back(value);
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
    ggml_set_output(residual_hidden_output_);
    graph_ = ggml_new_graph_custom(ctx_.get(), kDefaultGraphNodes, false);
    ggml_build_forward_expand(graph_, lm_hidden_output_);
    ggml_build_forward_expand(graph_, residual_hidden_output_);
    for (auto *key : base_keys_) {
      ggml_build_forward_expand(graph_, key);
    }
    for (auto *value : base_values_) {
      ggml_build_forward_expand(graph_, value);
    }
    for (auto *key : residual_keys_) {
      ggml_build_forward_expand(graph_, key);
    }
    for (auto *value : residual_values_) {
      ggml_build_forward_expand(graph_, value);
    }
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
          "failed to allocate VoxCPM2 prompt prefill graph");
    }
    std::vector<int32_t> position_ids(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
      position_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions_, position_ids.data(), 0,
                            position_ids.size() * sizeof(int32_t));
    sequence_steps_ = steps;
  }

  std::shared_ptr<const VoxCPM2WeightsRuntime> weights_;
  size_t graph_context_bytes_ = 0;
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
  ggml_gallocr_t gallocr_ = nullptr;
};

VoxCPM2PromptPrefillRuntime::VoxCPM2PromptPrefillRuntime(
    std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
    size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), graph_context_bytes)) {}

VoxCPM2PromptPrefillRuntime::~VoxCPM2PromptPrefillRuntime() = default;

VoxCPM2PromptPrefillOutput
VoxCPM2PromptPrefillRuntime::run(const VoxCPM2PromptPrefillInput &input) {
  return impl_->run(input);
}

void VoxCPM2PromptPrefillRuntime::release_runtime_memory() {
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
                                const VoxCPM2MiniCPMLayerWeights &layer,
                                const VoxCPM2MiniCPMWeights &weights) {
  const auto &config = weights.config;
  const int64_t dim = head_dim(config);
  const engine::modules::AddModule add;
  auto hidden = engine::modules::RMSNormModule(
                    {config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, input, layer.input_norm);
  engine::core::TensorValue q, k, v;
  if (layer.qkv_proj) {
    auto packed = engine::modules::LinearModule(
                      binding::linear_config(config.hidden_size,
                                             (config.num_attention_heads +
                                              2 * config.num_key_value_heads) *
                                                 dim,
                                             false))
                      .build(ctx, hidden, *layer.qkv_proj);
    const auto axis = static_cast<int>(hidden.shape.rank - 1);
    q = engine::modules::SliceModule(
            {axis, 0, config.num_attention_heads * dim})
            .build(ctx, packed);
    k = engine::modules::SliceModule({axis, config.num_attention_heads * dim,
                                      config.num_key_value_heads * dim})
            .build(ctx, packed);
    v = engine::modules::SliceModule(
            {axis,
             (config.num_attention_heads + config.num_key_value_heads) * dim,
             config.num_key_value_heads * dim})
            .build(ctx, packed);
  } else {
    q = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_attention_heads * dim, false))
            .build(ctx, hidden, layer.q_proj);
    k = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_key_value_heads * dim, false))
            .build(ctx, hidden, layer.k_proj);
    v = engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   config.num_key_value_heads * dim, false))
            .build(ctx, hidden, layer.v_proj);
  }
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
  engine::core::TensorValue gate, up;
  if (layer.gate_up_proj) {
    auto packed =
        engine::modules::LinearModule(
            binding::linear_config(config.hidden_size,
                                   2 * config.intermediate_size, false))
            .build(ctx, hidden, *layer.gate_up_proj);
    const auto axis = static_cast<int>(hidden.shape.rank - 1);
    gate = engine::modules::SliceModule({axis, 0, config.intermediate_size})
               .build(ctx, packed);
    up = engine::modules::SliceModule(
             {axis, config.intermediate_size, config.intermediate_size})
             .build(ctx, packed);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
  } else {
    gate = engine::modules::LinearModule(
               binding::linear_config(config.hidden_size,
                                      config.intermediate_size, false))
               .build(ctx, hidden, layer.gate_proj);
    gate = engine::modules::SiluModule{}.build(ctx, gate);
    up = engine::modules::LinearModule(
             binding::linear_config(config.hidden_size,
                                    config.intermediate_size, false))
             .build(ctx, hidden, layer.up_proj);
  }
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

const char *minicpm_kind_name(VoxCPM2MiniCPMKind kind) {
  switch (kind) {
  case VoxCPM2MiniCPMKind::BaseLM:
    return "base_lm";
  case VoxCPM2MiniCPMKind::ResidualLM:
    return "residual_lm";
  }
  throw std::runtime_error(
      "VoxCPM2 MiniCPM runtime received an unknown graph kind");
}


class VoxCPM2MiniCPMStepRuntime::Impl {
public:
  Impl(std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
       VoxCPM2MiniCPMKind kind, int64_t cache_steps, size_t graph_context_bytes)
      : weights_(std::move(weights)), kind_(kind), cache_steps_(cache_steps),
        graph_context_bytes_(graph_context_bytes) {
    if (weights_ == nullptr) {
      throw std::runtime_error("VoxCPM2 MiniCPM step runtime requires weights");
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

  int64_t cache_steps() const { return cache_steps_; }
  int64_t valid_steps() const { return step_cache_.valid_steps(); }

  void import_state(const engine::runtime::TransformerKVState &state) {
    ensure_graph();
    step_cache_.import_state(state);
  }

  engine::runtime::TransformerKVState export_state() const {
    return step_cache_.export_state();
  }

  VoxCPM2MiniCPMStepOutput run_step(const std::vector<float> &embedding) {
    ensure_graph();
    const auto &config =
        select_minicpm_weights(weights_->weights(), kind_).config;
    if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
      throw std::runtime_error("VoxCPM2 MiniCPM step embedding size mismatch");
    }
    if (step_cache_.valid_steps() >= cache_steps_) {
      throw std::runtime_error("VoxCPM2 MiniCPM step exceeds cache capacity");
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
      throw std::runtime_error(std::string("VoxCPM2 MiniCPM ") +
                               minicpm_kind_name(kind_) +
                               " step graph compute failed");
    }
    VoxCPM2MiniCPMStepOutput output;
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
          "VoxCPM2 MiniCPM step graph requires positive cache capacity");
    }
    if (graph_context_bytes == 0) {
      throw std::runtime_error(
          "VoxCPM2 MiniCPM step graph context bytes must be non-zero");
    }
    release_graph();
    ggml_init_params params{graph_context_bytes, nullptr, true};
    ctx_.reset(ggml_init(params));
    if (ctx_ == nullptr) {
      throw std::runtime_error(
          "failed to initialize VoxCPM2 MiniCPM step graph context");
    }
    const auto &lm_weights = select_minicpm_weights(weights_->weights(), kind_);
    const auto &config = lm_weights.config;
    const int64_t dim = head_dim(config);
    const std::string graph_name =
        std::string("voxcpm2.") + minicpm_kind_name(kind_) + ".step";
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
      throw std::runtime_error("failed to allocate VoxCPM2 MiniCPM step graph");
    }
    attention_mask_buffer_.assign(static_cast<size_t>(cache_steps_),
                                  ggml_fp32_to_fp16(-INFINITY));
  }

  std::shared_ptr<const VoxCPM2WeightsRuntime> weights_;
  VoxCPM2MiniCPMKind kind_ = VoxCPM2MiniCPMKind::BaseLM;
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

VoxCPM2MiniCPMStepRuntime::VoxCPM2MiniCPMStepRuntime(
    std::shared_ptr<const VoxCPM2WeightsRuntime> weights,
    VoxCPM2MiniCPMKind kind, int64_t cache_steps, size_t graph_context_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), kind, cache_steps,
                                   graph_context_bytes)) {}

VoxCPM2MiniCPMStepRuntime::~VoxCPM2MiniCPMStepRuntime() = default;

void VoxCPM2MiniCPMStepRuntime::reset() { impl_->reset(); }

void VoxCPM2MiniCPMStepRuntime::import_state(
    const engine::runtime::TransformerKVState &state) {
  impl_->import_state(state);
}

engine::runtime::TransformerKVState
VoxCPM2MiniCPMStepRuntime::export_state() const {
  return impl_->export_state();
}

int64_t VoxCPM2MiniCPMStepRuntime::cache_steps() const {
  return impl_->cache_steps();
}

int64_t VoxCPM2MiniCPMStepRuntime::valid_steps() const {
  return impl_->valid_steps();
}

VoxCPM2MiniCPMStepOutput
VoxCPM2MiniCPMStepRuntime::run_step(const std::vector<float> &embedding) {
  return impl_->run_step(embedding);
}

void VoxCPM2MiniCPMStepRuntime::release_runtime_memory() {
  impl_->release_runtime_memory();
}

} // namespace engine::models::voxcpm2


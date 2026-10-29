#include "engine/models/neutts/backbone.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/weight_binding.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace engine::models::neutts {
namespace {

namespace binding = engine::modules::binding;

modules::DecoderActivationCastPolicy neutts_activation_cast_policy(core::BackendType backend_type) {
    modules::DecoderActivationCastPolicy policy;
    if (backend_type == core::BackendType::Cpu || backend_type == core::BackendType::Vulkan ||
        backend_type == core::BackendType::Metal) {
        return policy;
    }
    policy.enabled = true;
    policy.type = GGML_TYPE_BF16;
    policy.after_input_norm = true;
    policy.after_qkv_projection = true;
    policy.after_qk_norm = true;
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

void validate_backbone_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "NeuTTS backbone weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

modules::DecoderLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const NeuTTSBackboneConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".input_layernorm",
        config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".self_attn.q_norm",
        config.head_dim);
    out.k_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".self_attn.k_norm",
        config.head_dim);
    out.post_norm = binding::norm_weight_from_source(
        store,
        source,
        prefix + ".post_attention_layernorm",
        config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

}  // namespace

modules::CausalDecoderConfig make_neutts_qwen3_config(
    const NeuTTSBackboneConfig & config,
    core::BackendType backend_type) {
    modules::CausalDecoderConfig out;
    out.stack.hidden_size = config.hidden_size;
    out.stack.num_attention_heads = config.attention_heads;
    out.stack.num_key_value_heads = config.kv_heads;
    out.stack.head_dim = config.head_dim;
    out.stack.intermediate_size = config.intermediate_size;
    out.stack.layers = config.layers;
    out.stack.rms_norm_eps = config.rms_norm_eps;
    out.stack.rope_theta = config.rope_theta;
    out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.stack.attention_precision = GGML_PREC_F32;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.activation_cast = neutts_activation_cast_policy(backend_type);
    out.stack.use_qk_norm = true;
    out.stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    out.lm_head_precision = GGML_PREC_DEFAULT;
    if (backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) {
        out.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu) {
        out.lm_head_input_type = GGML_TYPE_BF16;
    }
    return out;
}

NeuTTSQwen3Weights load_neutts_backbone_weights(
    const NeuTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    validate_backbone_storage_type(storage_type);
    const auto & config = assets.backbone;
    const auto & source = *assets.backbone_weights;
    NeuTTSQwen3Weights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "neutts.backbone.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.decoder.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.decoder.stack.layers.push_back(load_layer_weights(
            *weights.store,
            source,
            config,
            storage_type,
            layer));
    }
    weights.decoder.final_norm = binding::norm_weight_from_source(
        *weights.store,
        source,
        "model.norm",
        config.hidden_size);
    weights.decoder.lm_head = {weights.token_embedding, std::nullopt};
    weights.store->upload();
    return weights;
}

}  // namespace engine::models::neutts

#include "engine/community_models/soprano_tts/generator.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::soprano_tts {

struct SopranoQwen3Weights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue token_embedding;
    engine::modules::DecoderStackWeights stack;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights lm_head;
};

namespace {

namespace binding = engine::modules::binding;

std::shared_ptr<const SopranoTTSAssets> require_assets(
    std::shared_ptr<const SopranoTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Soprano LM generator requires assets");
    }
    return assets;
}

modules::DecoderLayerWeights load_layer_weights(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const SopranoTTSConfig & config,
    engine::assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(
        store, source, prefix + ".input_layernorm", config.hidden_size);
    // Fused QKV projection: concatenate Q|K|V rows so the decoder runs a
    // single GEMM per layer (DecoderQKVLayout::PackedQKV).
    const int64_t q_out = config.attention_heads * config.head_dim;
    const int64_t kv_out = config.kv_heads * config.head_dim;
    std::vector<float> qkv_rows = source.require_f32(
        prefix + ".self_attn.q_proj.weight", {q_out, config.hidden_size});
    const auto k_rows = source.require_f32(
        prefix + ".self_attn.k_proj.weight", {kv_out, config.hidden_size});
    const auto v_rows = source.require_f32(
        prefix + ".self_attn.v_proj.weight", {kv_out, config.hidden_size});
    qkv_rows.insert(qkv_rows.end(), k_rows.begin(), k_rows.end());
    qkv_rows.insert(qkv_rows.end(), v_rows.begin(), v_rows.end());
    out.self_attention.qkv_weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({q_out + kv_out * 2, config.hidden_size}),
        storage_type,
        std::move(qkv_rows));
    out.self_attention.out_weight = store.load_tensor(
        source, prefix + ".self_attn.o_proj.weight", storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(
        store, source, prefix + ".self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(
        store, source, prefix + ".self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(
        store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    // Fused gate/up projection: gate|up rows in one GEMM; the decoder's
    // PackedGateUp mode also uses the fused swiglu kernel.
    std::vector<float> gate_up_rows = source.require_f32(
        prefix + ".mlp.gate_proj.weight",
        {config.intermediate_size, config.hidden_size});
    const auto up_rows = source.require_f32(
        prefix + ".mlp.up_proj.weight",
        {config.intermediate_size, config.hidden_size});
    gate_up_rows.insert(gate_up_rows.end(), up_rows.begin(), up_rows.end());
    out.mlp.gate_up_proj = modules::LinearWeights{
        store.make_from_f32(
            engine::core::TensorShape::from_dims(
                {config.intermediate_size * 2, config.hidden_size}),
            storage_type,
            std::move(gate_up_rows)),
        std::nullopt};
    out.mlp.down_proj = binding::linear_from_source(
        store, source, prefix + ".mlp.down_proj", storage_type,
        config.hidden_size, config.intermediate_size, false);
    return out;
}
modules::DecoderActivationCastPolicy soprano_activation_cast_policy(
    core::BackendType backend_type) {
    // No activation cast for Soprano — keep everything in F32 for parity.
    (void)backend_type;
    return modules::DecoderActivationCastPolicy{};
}

modules::CausalDecoderConfig make_soprano_qwen3_config(
    const SopranoTTSConfig & config,
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
    out.stack.attention_precision = GGML_PREC_DEFAULT;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.activation_cast = soprano_activation_cast_policy(backend_type);
    out.stack.use_qk_norm = true;
    // Fused projections: single QKV GEMM + single gate/up GEMM with the
    // fused swiglu kernel (Soprano has no activation casts, so it qualifies).
    out.stack.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
    out.stack.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
    out.stack.runtime.attention.prefill_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = config.vocab_size;
    out.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
    out.use_lm_head_bias = false;
    out.lm_head_precision = GGML_PREC_DEFAULT;
    if (backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) {
        out.lm_head_input_type = GGML_TYPE_F16;
    } else if (backend_type != core::BackendType::Cpu) {
        out.lm_head_input_type = GGML_TYPE_BF16;
    }
    return out;
}

std::shared_ptr<const SopranoQwen3Weights> load_soprano_qwen3_weights(
    const SopranoTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<SopranoQwen3Weights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "soprano_tts.lm.weights", weight_context_bytes);
    const auto & config = assets.config;
    const auto & source = *assets.backbone_weights;
    weights->token_embedding = weights->store->load_tensor(
        source, "model.embed_tokens.weight", storage_type,
        {config.vocab_size, config.hidden_size});
    weights->stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights->stack.layers.push_back(load_layer_weights(
            *weights->store, source, config, storage_type, layer));
    }
weights->final_norm = binding::norm_weight_from_source(
        *weights->store, source, "model.norm", config.hidden_size);
    weights->lm_head = binding::linear_from_source(
        *weights->store, source, "lm_head", storage_type,
        config.vocab_size, config.hidden_size, false);
    weights->store->upload();
    return weights;
}

modules::CausalDecoderRuntimeConfig make_soprano_decode_runtime_config(
    const SopranoTTSConfig & config,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes) {
    modules::CausalDecoderRuntimeConfig out;
    out.trace_name = "soprano_tts.lm";
    out.decoder = make_soprano_qwen3_config(config, backend_type);
    out.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    out.decode_graph_arena_bytes = decode_graph_arena_bytes;
    // Both logits (sampling + EOS) and the 512-d hidden frame (audio) are needed.
    out.output_mode = modules::CausalDecoderOutputMode::Logits;
    out.return_hidden = true;
    return out;
}

modules::CausalDecoderRuntimeWeights make_soprano_decode_weights(
    const SopranoQwen3Weights & weights) {
    modules::CausalDecoderRuntimeWeights out;
    out.token_embedding = weights.token_embedding;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    out.lm_head = weights.lm_head;
    return out;
}

// Hidden-mode weights are intentionally not built: Hidden mode produces
// NaN/Inf prefill output (see docs/models/soprano_tts.md §6h), so the LM
// always runs single-pass Logits+return_hidden=true.

}  // namespace

class SopranoQwen3Generator::Impl {
public:
    Impl(
        std::shared_ptr<const SopranoTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : assets_(require_assets(std::move(assets))),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          weights_(std::make_shared<SopranoQwen3Weights>(std::move(*load_soprano_qwen3_weights(
              *assets_, execution.backend(), backend_type_, weight_context_bytes,
              weight_storage_type)))) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Soprano LM backend is not initialized");
        }
        qwen3_runtime = std::make_unique<modules::CausalDecoderRuntime>(
            execution,
            make_soprano_decode_runtime_config(
                assets_->config, backend_type_,
                prefill_graph_arena_bytes, decode_graph_arena_bytes),
            make_soprano_decode_weights(*weights_));
    }

    Result generate(const std::vector<int32_t> & prompt_ids,
                    const SopranoGenerationOptions & options) {
        if (prompt_ids.empty()) {
            throw std::runtime_error("Soprano LM requires a non-empty prompt");
        }
        const SopranoTTSConfig & config = assets_->config;
        std::vector<float> features;
        std::vector<int32_t> tokens;

        // Single-pass AR generation capturing both logits and hidden states.
        // With F32 weights + correct tokenizer, return_hidden=true now
        // produces correct results.
        auto prefill = qwen3_runtime->prefill_tokens(prompt_ids);
        // Honor the requested token limit (matches HF max_new_tokens), capped
        // so prompt + generated always fits the model context window.
        const int64_t max_new_tokens = std::max<int64_t>(
            1,
            std::min<int64_t>(
                options.max_new_tokens,
                config.max_position_embeddings -
                    static_cast<int64_t>(prompt_ids.size())));
        // Size the KV cache to the actual worst-case need (prompt + generated
        // frames) instead of the full 1024-token context. Smaller cache means
        // less KV memory for attention to walk on every decode step.
        qwen3_runtime->start_decode_tokens(prefill.state, max_new_tokens +
            static_cast<int64_t>(prompt_ids.size()));

        // First feature: last prompt token's post-norm hidden state.
        features.insert(features.end(), prefill.hidden.begin(), prefill.hidden.end());

        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = true;
        sampling_options.temperature = options.temperature;
        sampling_options.top_k = 0;
        sampling_options.top_p = options.top_p;
        sampling_options.min_tokens_to_keep = 1;
        sampling_options.repetition_penalty = options.repetition_penalty;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        scratch.reserve_vocab(static_cast<size_t>(config.vocab_size));
        std::mt19937 fallback_rng(static_cast<uint32_t>(options.seed));

        std::vector<int32_t> history(prompt_ids.begin(), prompt_ids.end());
        std::vector<float> logits = std::move(prefill.logits);
        const int32_t eos_id = config.eos_token_id;

        for (int64_t step = 0; step < max_new_tokens; ++step) {
            if (options.eos_bias != 0.0F) {
                logits[static_cast<size_t>(eos_id)] += options.eos_bias;
            }
            const int32_t token = sampler.sample(
                logits, history, sampling_options, scratch, fallback_rng, nullptr,
                "soprano_tts AR");
            if (token == eos_id) {
                history.push_back(token);
                break;
            }
            history.push_back(token);
            tokens.push_back(token);
            auto decode = qwen3_runtime->decode_token(token);
            features.insert(features.end(), decode.hidden.begin(), decode.hidden.end());
            logits = std::move(decode.logits);
        }

        if (tokens.empty()) {
            throw std::runtime_error("Soprano LM produced no audio frames");
        }

        Result out;
        out.tokens = std::move(tokens);
        const int64_t T_gen = static_cast<int64_t>(out.tokens.size()) + 1; // +1 for prefill
        out.frames = T_gen;
        out.features = std::move(features);
        return out;
    }

    void release_runtime_graphs() {
        qwen3_runtime->release_runtime_graphs();
    }

    std::shared_ptr<const SopranoTTSAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    std::shared_ptr<const SopranoQwen3Weights> weights_;
    std::unique_ptr<modules::CausalDecoderRuntime> qwen3_runtime;
};

SopranoQwen3Generator::SopranoQwen3Generator(
    const SopranoTTSAssets & assets,
    engine::core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::make_shared<SopranoTTSAssets>(assets), execution,
          prefill_graph_arena_bytes, decode_graph_arena_bytes,
          weight_context_bytes, weight_storage_type)) {}

SopranoQwen3Generator::~SopranoQwen3Generator() = default;

SopranoQwen3Generator::Result SopranoQwen3Generator::generate(
    const std::vector<int32_t> & prompt_ids,
    const SopranoGenerationOptions & options) {
    return impl_->generate(prompt_ids, options);
}

void SopranoQwen3Generator::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::community_models::soprano_tts

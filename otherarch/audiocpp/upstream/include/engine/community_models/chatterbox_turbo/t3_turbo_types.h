#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

struct T3TurboGraphWeight {
    std::vector<float> values;
    engine::core::TensorValue tensor;
};

// GPT2-style T3 backbone used by Chatterbox Turbo: LayerNorm (weight+bias),
// fused-QKV causal self attention, GELU(tanh) MLP with bias, absolute
// position embedding ("wpe") applied host-side before the transformer stack.
// No RoPE, no perceiver resampler, no emotion-exaggeration conditioning.
struct T3TurboInferenceWeights {
    int64_t hidden_size = 1024;
    int64_t speaker_embed_size = 256;
    int64_t num_heads = 16;
    int64_t mlp_intermediate_size = 4096;
    int64_t text_vocab = 0;
    int64_t speech_vocab = 0;
    int64_t max_positions = 0;

    struct TransformerLayer {
        std::vector<float> ln1_weight;
        std::vector<float> ln1_bias;
        engine::core::TensorValue ln1_weight_tensor;
        engine::core::TensorValue ln1_bias_tensor;
        T3TurboGraphWeight attn_qkv_weight;
        std::vector<float> attn_qkv_bias;
        engine::core::TensorValue attn_qkv_bias_tensor;
        T3TurboGraphWeight attn_output_weight;
        std::vector<float> attn_output_bias;
        engine::core::TensorValue attn_output_bias_tensor;
        std::vector<float> ln2_weight;
        std::vector<float> ln2_bias;
        engine::core::TensorValue ln2_weight_tensor;
        engine::core::TensorValue ln2_bias_tensor;
        T3TurboGraphWeight ffn_fc_weight;
        std::vector<float> ffn_fc_bias;
        engine::core::TensorValue ffn_fc_bias_tensor;
        T3TurboGraphWeight ffn_proj_weight;
        std::vector<float> ffn_proj_bias;
        engine::core::TensorValue ffn_proj_bias_tensor;
    };

    engine::assets::TensorData spkr_enc_weight;
    engine::assets::TensorData spkr_enc_bias;
    engine::assets::TensorData text_embedding_weight;
    engine::assets::TensorData speech_embedding_weight;
    engine::assets::TensorData wpe_weight;
    std::vector<float> ln_f_weight;
    std::vector<float> ln_f_bias;
    engine::core::TensorValue ln_f_weight_tensor;
    engine::core::TensorValue ln_f_bias_tensor;
    T3TurboGraphWeight text_head_weight;
    T3TurboGraphWeight speech_head_weight;
    std::vector<float> speech_head_bias;
    engine::core::TensorValue speech_head_bias_tensor;
    std::vector<TransformerLayer> layers;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct T3TurboGenerateRequest {
    std::vector<float> speaker_embedding;
    std::vector<int32_t> cond_prompt_speech_tokens;
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> initial_speech_tokens;
    int64_t max_new_tokens = 1000;
    bool stop_on_eos = true;
    bool do_sample = true;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int64_t top_k = 1000;
    float repetition_penalty = 1.2f;
    uint32_t seed = 0;
};

struct T3TurboGenerateOutputs {
    std::vector<int32_t> predicted_tokens;
    int64_t token_count = 0;
    bool hit_eos = false;
    double prefix_cache_build_ms = 0.0;
    double prefill_runner_ms = 0.0;
    double decode_runner_ms = 0.0;
};

}  // namespace engine::community_models::chatterbox_turbo

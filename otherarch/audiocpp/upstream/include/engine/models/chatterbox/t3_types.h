#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::chatterbox {

struct T3GraphWeight {
    std::vector<float> values;
    engine::core::TensorValue tensor;
};

struct T3InferenceWeights {
    int64_t hidden_size = 1024;
    int64_t speaker_embed_size = 256;
    int64_t num_heads = 16;
    int64_t perceiver_num_heads = 4;
    int64_t perceiver_query_tokens = 32;
    int64_t mlp_intermediate_size = 4096;
    int64_t text_vocab = 0;
    int64_t speech_vocab = 0;
    int64_t text_position_vocab = 0;
    int64_t speech_position_vocab = 0;

    struct TransformerLayer {
        std::vector<float> input_layernorm_weight;
        engine::core::TensorValue input_layernorm_tensor;
        T3GraphWeight q_proj_weight;
        T3GraphWeight k_proj_weight;
        T3GraphWeight v_proj_weight;
        T3GraphWeight o_proj_weight;
        std::vector<float> post_attention_layernorm_weight;
        engine::core::TensorValue post_attention_layernorm_tensor;
        T3GraphWeight gate_proj_weight;
        T3GraphWeight up_proj_weight;
        T3GraphWeight down_proj_weight;
    };

    engine::assets::TensorData spkr_enc_weight;
    engine::assets::TensorData spkr_enc_bias;
    engine::assets::TensorData emotion_adv_weight;
    engine::assets::TensorData perceiver_pre_attention_query;
    engine::assets::TensorData perceiver_norm_weight;
    engine::assets::TensorData perceiver_norm_bias;
    engine::assets::TensorData perceiver_to_q_weight;
    engine::assets::TensorData perceiver_to_q_bias;
    engine::assets::TensorData perceiver_to_k_weight;
    engine::assets::TensorData perceiver_to_k_bias;
    engine::assets::TensorData perceiver_to_v_weight;
    engine::assets::TensorData perceiver_to_v_bias;
    engine::assets::TensorData perceiver_proj_out_weight;
    engine::assets::TensorData perceiver_proj_out_bias;
    engine::assets::TensorData text_embedding_weight;
    engine::assets::TensorData speech_embedding_weight;
    engine::assets::TensorData text_position_weight;
    engine::assets::TensorData speech_position_weight;
    std::vector<float> tfmr_norm_weight;
    engine::core::TensorValue tfmr_norm_tensor;
    engine::core::TensorValue rope_factors_tensor;
    T3GraphWeight text_head_weight;
    T3GraphWeight speech_head_weight;
    std::vector<TransformerLayer> layers;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct T3GenerateRequest {
    std::vector<float> speaker_embedding;
    std::vector<int32_t> cond_prompt_speech_tokens;
    std::vector<float> emotion_adv;
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> initial_speech_tokens;
    int64_t max_new_tokens = 256;
    bool stop_on_eos = true;
    bool do_sample = true;
    float temperature = 0.8f;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repetition_penalty = 1.2f;
    float guidance_scale = 0.5f;
    uint32_t seed = 0;
};

struct T3GenerateOutputs {
    std::vector<int32_t> predicted_tokens;
    int64_t token_count = 0;
    bool hit_eos = false;
    double prefix_cache_build_ms = 0.0;
    double decoder_cache_clone_ms = 0.0;
    double prefill_runner_ms = 0.0;
    double decode_runner_ms = 0.0;
    double logits_ms = 0.0;
    double sampling_ms = 0.0;
    double next_embed_ms = 0.0;
};

}  // namespace engine::models::chatterbox

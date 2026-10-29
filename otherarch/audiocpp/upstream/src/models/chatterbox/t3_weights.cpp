#include "engine/models/chatterbox/t3_component.h"

#include "engine/framework/assets/tensor_source.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace engine::models::chatterbox {

namespace {

#if defined(INTPTR_MAX) && (INTPTR_MAX == INT32_MAX)
constexpr size_t kDefaultWeightStoreBytes = 1024ull * 1024ull * 1024ull;
#else
constexpr size_t kDefaultWeightStoreBytes = 4096ull * 1024ull * 1024ull;
#endif

T3GraphWeight load_t3_graph_weight(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type,
    const std::vector<int64_t> & shape,
    bool load_reference_f32_values) {
    T3GraphWeight weight;
    if (load_reference_f32_values) {
        weight.values = source.require_f32(name, shape);
    }
    weight.tensor = store.load_tensor(source, name, storage_type, shape);
    return weight;
}

engine::assets::TensorStorageType tensor_data_storage_for_shape(
    engine::assets::TensorStorageType requested,
    const std::vector<int64_t> & shape) {
    if (requested == engine::assets::TensorStorageType::Native) {
        return requested;
    }
    const ggml_type type = engine::assets::ggml_type_for_tensor_storage(requested);
    if (!ggml_is_quantized(type)) {
        return requested;
    }
    if (shape.size() < 2 || shape.back() % ggml_blck_size(type) != 0) {
        return engine::assets::TensorStorageType::F32;
    }
    return requested;
}

engine::assets::TensorData require_t3_tensor_data(
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType requested,
    const std::vector<int64_t> & shape) {
    return source.require_tensor(name, tensor_data_storage_for_shape(requested, shape), shape);
}

std::vector<float> llama3_rope_factors_for_t3(int64_t head_dim) {
    constexpr double kTheta = 500000.0;
    constexpr double kFactor = 8.0;
    constexpr double kLowFreqFactor = 1.0;
    constexpr double kHighFreqFactor = 4.0;
    constexpr double kOldContextLen = 8192.0;
    const double low_freq_wavelen = kOldContextLen / kLowFreqFactor;
    const double high_freq_wavelen = kOldContextLen / kHighFreqFactor;
    std::vector<float> factors(static_cast<size_t>(head_dim / 2), 1.0f);
    for (int64_t i = 0; i < head_dim / 2; ++i) {
        const double inv_freq = 1.0 / std::pow(kTheta, static_cast<double>(2 * i) / static_cast<double>(head_dim));
        const double wavelen = (2.0 * M_PI) / inv_freq;
        double coeff = 1.0;
        if (wavelen > low_freq_wavelen) {
            coeff = 1.0 / kFactor;
        } else if (wavelen >= high_freq_wavelen) {
            const double smooth_factor =
                (kOldContextLen / wavelen - kLowFreqFactor) / (kHighFreqFactor - kLowFreqFactor);
            coeff = (1.0 - smooth_factor) * (1.0 / kFactor) + smooth_factor;
        }
        factors[static_cast<size_t>(i)] = static_cast<float>(1.0 / coeff);
    }
    return factors;
}

}  // namespace

std::shared_ptr<const T3InferenceWeights> load_t3_inference_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType graph_weight_storage_type,
    bool load_reference_f32_graph_weights) {
    auto weights = std::make_shared<T3InferenceWeights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.t3.weights",
        kDefaultWeightStoreBytes);
    const auto text_emb_info = source.require_metadata("text_emb.weight");
    const auto speech_emb_info = source.require_metadata("speech_emb.weight");
    const auto text_pos_info = source.require_metadata("text_pos_emb.emb.weight");
    const auto speech_pos_info = source.require_metadata("speech_pos_emb.emb.weight");
    weights->hidden_size = text_emb_info.shape.at(1);
    weights->speaker_embed_size = source.require_metadata("cond_enc.spkr_enc.weight").shape.at(1);
    weights->num_heads = 16;
    weights->perceiver_num_heads = 4;
    weights->perceiver_query_tokens = source.require_metadata("cond_enc.perceiver.pre_attention_query").shape.at(1);
    weights->mlp_intermediate_size = source.require_metadata("tfmr.layers.0.mlp.gate_proj.weight").shape.at(0);
    weights->text_vocab = text_emb_info.shape.at(0);
    weights->speech_vocab = speech_emb_info.shape.at(0);
    weights->text_position_vocab = text_pos_info.shape.at(0);
    weights->speech_position_vocab = speech_pos_info.shape.at(0);
    weights->spkr_enc_weight = require_t3_tensor_data(source, "cond_enc.spkr_enc.weight", graph_weight_storage_type, {1024, 256});
    weights->spkr_enc_bias = source.require_tensor("cond_enc.spkr_enc.bias", engine::assets::TensorStorageType::F32, {1024});
    weights->emotion_adv_weight =
        require_t3_tensor_data(source, "cond_enc.emotion_adv_fc.weight", graph_weight_storage_type, {1024, 1});
    weights->perceiver_pre_attention_query = require_t3_tensor_data(
        source,
        "cond_enc.perceiver.pre_attention_query",
        graph_weight_storage_type,
        {1, 32, 1024});
    weights->perceiver_norm_weight = source.require_tensor(
        "cond_enc.perceiver.attn.norm.weight",
        engine::assets::TensorStorageType::F32,
        {1024});
    weights->perceiver_norm_bias = source.require_tensor(
        "cond_enc.perceiver.attn.norm.bias",
        engine::assets::TensorStorageType::F32,
        {1024});
    weights->perceiver_to_q_weight =
        require_t3_tensor_data(source, "cond_enc.perceiver.attn.to_q.weight", graph_weight_storage_type, {1024, 1024});
    weights->perceiver_to_q_bias = source.require_tensor("cond_enc.perceiver.attn.to_q.bias", engine::assets::TensorStorageType::F32, {1024});
    weights->perceiver_to_k_weight =
        require_t3_tensor_data(source, "cond_enc.perceiver.attn.to_k.weight", graph_weight_storage_type, {1024, 1024});
    weights->perceiver_to_k_bias = source.require_tensor("cond_enc.perceiver.attn.to_k.bias", engine::assets::TensorStorageType::F32, {1024});
    weights->perceiver_to_v_weight =
        require_t3_tensor_data(source, "cond_enc.perceiver.attn.to_v.weight", graph_weight_storage_type, {1024, 1024});
    weights->perceiver_to_v_bias = source.require_tensor("cond_enc.perceiver.attn.to_v.bias", engine::assets::TensorStorageType::F32, {1024});
    weights->perceiver_proj_out_weight =
        require_t3_tensor_data(source, "cond_enc.perceiver.attn.proj_out.weight", graph_weight_storage_type, {1024, 1024});
    weights->perceiver_proj_out_bias = source.require_tensor("cond_enc.perceiver.attn.proj_out.bias", engine::assets::TensorStorageType::F32, {1024});
    weights->text_embedding_weight = require_t3_tensor_data(source, "text_emb.weight", graph_weight_storage_type, text_emb_info.shape);
    weights->speech_embedding_weight = require_t3_tensor_data(source, "speech_emb.weight", graph_weight_storage_type, speech_emb_info.shape);
    weights->text_position_weight =
        require_t3_tensor_data(source, "text_pos_emb.emb.weight", graph_weight_storage_type, text_pos_info.shape);
    weights->speech_position_weight =
        require_t3_tensor_data(source, "speech_pos_emb.emb.weight", graph_weight_storage_type, speech_pos_info.shape);
    weights->tfmr_norm_weight = source.require_f32("tfmr.norm.weight", {1024});
    weights->tfmr_norm_tensor = weights->store->load_f32_tensor(source, "tfmr.norm.weight", {1024});
    weights->rope_factors_tensor = weights->store->make_f32(
        engine::core::TensorShape::from_dims({weights->hidden_size / weights->num_heads / 2}),
        llama3_rope_factors_for_t3(weights->hidden_size / weights->num_heads));
    weights->text_head_weight = load_t3_graph_weight(
        *weights->store,
        source,
        "text_head.weight",
        graph_weight_storage_type,
        source.require_metadata("text_head.weight").shape,
        load_reference_f32_graph_weights);
    weights->speech_head_weight = load_t3_graph_weight(
        *weights->store,
        source,
        "speech_head.weight",
        graph_weight_storage_type,
        source.require_metadata("speech_head.weight").shape,
        load_reference_f32_graph_weights);
    weights->layers.resize(30);
    for (int layer = 0; layer < 30; ++layer) {
        const std::string prefix = "tfmr.layers." + std::to_string(layer);
        auto & out = weights->layers[static_cast<size_t>(layer)];
        out.input_layernorm_weight = source.require_f32(prefix + ".input_layernorm.weight", {1024});
        out.input_layernorm_tensor = weights->store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {1024});
        out.q_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".self_attn.q_proj.weight",
            graph_weight_storage_type,
            {1024, 1024},
            load_reference_f32_graph_weights);
        out.k_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".self_attn.k_proj.weight",
            graph_weight_storage_type,
            {1024, 1024},
            load_reference_f32_graph_weights);
        out.v_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".self_attn.v_proj.weight",
            graph_weight_storage_type,
            {1024, 1024},
            load_reference_f32_graph_weights);
        out.o_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".self_attn.o_proj.weight",
            graph_weight_storage_type,
            {1024, 1024},
            load_reference_f32_graph_weights);
        out.post_attention_layernorm_weight = source.require_f32(prefix + ".post_attention_layernorm.weight", {1024});
        out.post_attention_layernorm_tensor = weights->store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {1024});
        out.gate_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".mlp.gate_proj.weight",
            graph_weight_storage_type,
            {4096, 1024},
            load_reference_f32_graph_weights);
        out.up_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".mlp.up_proj.weight",
            graph_weight_storage_type,
            {4096, 1024},
            load_reference_f32_graph_weights);
        out.down_proj_weight = load_t3_graph_weight(
            *weights->store,
            source,
            prefix + ".mlp.down_proj.weight",
            graph_weight_storage_type,
            {1024, 4096},
            load_reference_f32_graph_weights);
    }
    weights->store->upload();
    return weights;
}

}  // namespace engine::models::chatterbox

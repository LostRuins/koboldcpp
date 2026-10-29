#include "engine/community_models/chatterbox_turbo/t3_turbo_component.h"

#include <cstdint>
#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {

namespace {

#if defined(INTPTR_MAX) && (INTPTR_MAX == INT32_MAX)
constexpr size_t kDefaultWeightStoreBytes = 1024ull * 1024ull * 1024ull;
#else
constexpr size_t kDefaultWeightStoreBytes = 4096ull * 1024ull * 1024ull;
#endif

T3TurboGraphWeight load_graph_weight(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType storage_type,
    const std::vector<int64_t> & shape,
    bool load_reference_f32_values) {
    T3TurboGraphWeight weight;
    if (load_reference_f32_values) {
        weight.values = source.require_f32(name, shape);
    }
    weight.tensor = store.load_tensor(source, name, storage_type, shape);
    return weight;
}

engine::assets::TensorStorageType storage_for_shape(
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

engine::assets::TensorData require_tensor_data(
    const engine::assets::TensorSource & source,
    const std::string & name,
    engine::assets::TensorStorageType requested,
    const std::vector<int64_t> & shape) {
    return source.require_tensor(name, storage_for_shape(requested, shape), shape);
}

int64_t count_turbo_layers(const engine::assets::TensorSource & source) {
    int64_t count = 0;
    while (source.has_tensor("blk." + std::to_string(count) + ".attn_norm.weight")) {
        ++count;
    }
    if (count == 0) {
        throw std::runtime_error("Chatterbox Turbo T3 weights contain no transformer layers");
    }
    return count;
}

}  // namespace

std::shared_ptr<const T3TurboInferenceWeights> load_t3_turbo_inference_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType graph_weight_storage_type,
    bool load_reference_f32_graph_weights) {
    auto weights = std::make_shared<T3TurboInferenceWeights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox_turbo.t3.weights",
        kDefaultWeightStoreBytes);

    const auto text_emb_info = source.require_metadata("text_emb.weight");
    const auto speech_emb_info = source.require_metadata("speech_emb.weight");
    const auto wpe_info = source.require_metadata("wpe.weight");

    weights->hidden_size = text_emb_info.shape.at(1);
    weights->text_vocab = text_emb_info.shape.at(0);
    weights->speech_vocab = speech_emb_info.shape.at(0);
    weights->max_positions = wpe_info.shape.at(0);
    weights->speaker_embed_size = source.require_metadata("cond.spkr_enc.weight").shape.at(1);
    weights->num_heads = 16;
    weights->mlp_intermediate_size = source.require_metadata("blk.0.ffn_fc.weight").shape.at(0);

    const int64_t hidden = weights->hidden_size;

    weights->spkr_enc_weight = require_tensor_data(
        source, "cond.spkr_enc.weight", graph_weight_storage_type, {hidden, weights->speaker_embed_size});
    weights->spkr_enc_bias = source.require_tensor("cond.spkr_enc.bias", engine::assets::TensorStorageType::F32, {hidden});

    weights->text_embedding_weight =
        require_tensor_data(source, "text_emb.weight", graph_weight_storage_type, text_emb_info.shape);
    weights->speech_embedding_weight =
        require_tensor_data(source, "speech_emb.weight", graph_weight_storage_type, speech_emb_info.shape);
    weights->wpe_weight = require_tensor_data(source, "wpe.weight", graph_weight_storage_type, wpe_info.shape);

    weights->ln_f_weight = source.require_f32("output_norm.weight", {hidden});
    weights->ln_f_weight_tensor = weights->store->load_f32_tensor(source, "output_norm.weight", {hidden});
    weights->ln_f_bias = source.require_f32("output_norm.bias", {hidden});
    weights->ln_f_bias_tensor = weights->store->load_f32_tensor(source, "output_norm.bias", {hidden});

    weights->text_head_weight = load_graph_weight(
        *weights->store,
        source,
        "text_head.weight",
        graph_weight_storage_type,
        source.require_metadata("text_head.weight").shape,
        load_reference_f32_graph_weights);
    weights->speech_head_weight = load_graph_weight(
        *weights->store,
        source,
        "speech_head.weight",
        graph_weight_storage_type,
        source.require_metadata("speech_head.weight").shape,
        load_reference_f32_graph_weights);
    weights->speech_head_bias = source.require_f32("speech_head.bias", {weights->speech_vocab});
    weights->speech_head_bias_tensor = weights->store->load_f32_tensor(source, "speech_head.bias", {weights->speech_vocab});

    const int64_t num_layers = count_turbo_layers(source);
    const int64_t intermediate = weights->mlp_intermediate_size;
    weights->layers.resize(static_cast<size_t>(num_layers));
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer);
        auto & out = weights->layers[static_cast<size_t>(layer)];

        out.ln1_weight = source.require_f32(prefix + ".attn_norm.weight", {hidden});
        out.ln1_weight_tensor = weights->store->load_f32_tensor(source, prefix + ".attn_norm.weight", {hidden});
        out.ln1_bias = source.require_f32(prefix + ".attn_norm.bias", {hidden});
        out.ln1_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".attn_norm.bias", {hidden});

        out.attn_qkv_weight = load_graph_weight(
            *weights->store,
            source,
            prefix + ".attn_qkv.weight",
            graph_weight_storage_type,
            {3 * hidden, hidden},
            load_reference_f32_graph_weights);
        out.attn_qkv_bias = source.require_f32(prefix + ".attn_qkv.bias", {3 * hidden});
        out.attn_qkv_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".attn_qkv.bias", {3 * hidden});

        out.attn_output_weight = load_graph_weight(
            *weights->store,
            source,
            prefix + ".attn_output.weight",
            graph_weight_storage_type,
            {hidden, hidden},
            load_reference_f32_graph_weights);
        out.attn_output_bias = source.require_f32(prefix + ".attn_output.bias", {hidden});
        out.attn_output_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".attn_output.bias", {hidden});

        out.ln2_weight = source.require_f32(prefix + ".ffn_norm.weight", {hidden});
        out.ln2_weight_tensor = weights->store->load_f32_tensor(source, prefix + ".ffn_norm.weight", {hidden});
        out.ln2_bias = source.require_f32(prefix + ".ffn_norm.bias", {hidden});
        out.ln2_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".ffn_norm.bias", {hidden});

        out.ffn_fc_weight = load_graph_weight(
            *weights->store,
            source,
            prefix + ".ffn_fc.weight",
            graph_weight_storage_type,
            {intermediate, hidden},
            load_reference_f32_graph_weights);
        out.ffn_fc_bias = source.require_f32(prefix + ".ffn_fc.bias", {intermediate});
        out.ffn_fc_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".ffn_fc.bias", {intermediate});

        out.ffn_proj_weight = load_graph_weight(
            *weights->store,
            source,
            prefix + ".ffn_proj.weight",
            graph_weight_storage_type,
            {hidden, intermediate},
            load_reference_f32_graph_weights);
        out.ffn_proj_bias = source.require_f32(prefix + ".ffn_proj.bias", {hidden});
        out.ffn_proj_bias_tensor = weights->store->load_f32_tensor(source, prefix + ".ffn_proj.bias", {hidden});
    }

    weights->store->upload();
    return weights;
}

}  // namespace engine::community_models::chatterbox_turbo

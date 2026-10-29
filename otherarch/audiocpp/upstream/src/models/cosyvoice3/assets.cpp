#include "engine/models/cosyvoice3/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::cosyvoice3 {
namespace {

constexpr const char * kFamily = "cosyvoice3";

std::filesystem::path find_gguf_path(const engine::assets::ResourceBundle & resources) {
    for (const auto & file : resources.files()) {
        if (file.path.extension() == ".gguf") {
            return file.path;
        }
    }
    return {};
}

void validate_llm_shapes(const engine::assets::TensorSource & source, CosyVoice3Config & config) {
    const auto embedding = source.require_metadata("llm.model.model.embed_tokens.weight");
    if (embedding.shape.size() != 2) {
        throw std::runtime_error("CosyVoice3 LLM embedding must be rank 2");
    }
    config.text_vocab_size = embedding.shape[0];
    config.hidden_size = embedding.shape[1];
    engine::assets::require_tensor_shape(source, "speech_embedding.weight", {config.speech_token_size + config.speech_reserved_tokens, config.hidden_size});
    engine::assets::require_tensor_shape(source, "llm_decoder.weight", {config.speech_token_size + config.speech_reserved_tokens, config.hidden_size});
    engine::assets::require_tensor_shape(source, "llm.model.model.layers.0.self_attn.q_proj.weight", {config.heads * config.head_dim, config.hidden_size});
    engine::assets::require_tensor_shape(source, "llm.model.model.layers.0.self_attn.k_proj.weight", {config.kv_heads * config.head_dim, config.hidden_size});
    engine::assets::require_tensor_shape(source, "llm.model.model.layers.0.self_attn.v_proj.weight", {config.kv_heads * config.head_dim, config.hidden_size});
}

void validate_flow_shapes(const engine::assets::TensorSource & source, const CosyVoice3Config & config) {
    engine::assets::require_tensor_shape(source, "input_embedding.weight", {config.speech_token_size, config.flow_mel_channels});
    engine::assets::require_tensor_shape(source, "spk_embed_affine_layer.weight", {config.flow_mel_channels, config.speaker_dim});
    engine::assets::require_tensor_shape(source, "decoder.rand_noise", {1, config.flow_mel_channels, 50 * 300});
    engine::assets::require_tensor_shape(source, "decoder.estimator.input_embed.proj.weight", {config.flow_hidden_size, 320});
    engine::assets::require_tensor_shape(source, "decoder.estimator.proj_out.weight", {config.flow_mel_channels, config.flow_hidden_size});
}

void validate_hift_shapes(const engine::assets::TensorSource & source) {
    engine::assets::require_tensor_shape(source, "conv_pre.parametrizations.weight.original1", {512, 80, 5});
    engine::assets::require_tensor_shape(source, "conv_post.parametrizations.weight.original1", {18, 64, 7});
}

}  // namespace

std::shared_ptr<const CosyVoice3Assets> load_cosyvoice3_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<CosyVoice3Assets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->model_root = assets->resources.model_root();
    assets->gguf_path = find_gguf_path(assets->resources);
    assets->llm_weights = assets->resources.open_tensor_source("llm_weights");
    assets->flow_weights = assets->resources.open_tensor_source("flow_weights");
    assets->hift_weights = assets->resources.open_tensor_source("hift_weights");
    assets->campplus_weights = assets->resources.open_tensor_source("campplus_weights");
    assets->speech_tokenizer_weights = assets->resources.open_tensor_source("speech_tokenizer_weights");
    assets->blank_en_weights = assets->resources.open_tensor_source("blank_en_weights");

    validate_llm_shapes(*assets->llm_weights, assets->config);
    validate_flow_shapes(*assets->flow_weights, assets->config);
    validate_hift_shapes(*assets->hift_weights);
    return assets;
}

}  // namespace engine::models::cosyvoice3

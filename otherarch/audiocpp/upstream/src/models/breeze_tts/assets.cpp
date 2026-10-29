#include "engine/models/breeze_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>

namespace engine::models::breeze_tts {
namespace {

constexpr const char * kFamily = "breeze_tts";

int64_t nested_i64(const engine::io::json::Value & object, const std::string & key, int64_t fallback) {
    return engine::io::json::optional_i64(object, key, fallback);
}

float nested_f32(const engine::io::json::Value & object, const std::string & key, float fallback) {
    return engine::io::json::optional_f32(object, key, fallback);
}

float rope_theta_for_type(
    const engine::io::json::Value & rope_parameters,
    const std::string & layer_type,
    float fallback) {
    const auto * object = rope_parameters.find(layer_type);
    if (object == nullptr || !object->is_object()) {
        return fallback;
    }
    return nested_f32(*object, "rope_theta", fallback);
}

float rope_freq_scale_for_type(
    const engine::io::json::Value & rope_parameters,
    const std::string & layer_type,
    float fallback) {
    const auto * object = rope_parameters.find(layer_type);
    if (object == nullptr || !object->is_object()) {
        return fallback;
    }
    const std::string rope_type = engine::io::json::optional_string(*object, "rope_type", "default");
    if (rope_type != "linear") {
        return 1.0F;
    }
    const float factor = nested_f32(*object, "factor", 1.0F);
    if (!(factor > 0.0F)) {
        throw std::runtime_error("BreezeTTS text rope linear factor must be positive");
    }
    return 1.0F / factor;
}

void parse_config(const engine::io::json::Value & root, BreezeTTSConfig & config) {
    config.hidden_size = nested_i64(root, "hidden_size", config.hidden_size);
    config.intermediate_size = nested_i64(root, "intermediate_size", config.intermediate_size);
    config.layers = nested_i64(root, "num_hidden_layers", config.layers);
    config.heads = nested_i64(root, "num_attention_heads", config.heads);
    config.kv_heads = nested_i64(root, "num_key_value_heads", config.kv_heads);
    config.head_dim = nested_i64(root, "head_dim", config.head_dim);
    config.vocab_size = nested_i64(root, "vocab_size", config.vocab_size);
    config.lm_head_size = config.vocab_size + 1;
    config.text_vocab_size = nested_i64(root, "text_vocab_size", config.text_vocab_size);
    config.num_codebooks = nested_i64(root, "num_codebooks", config.num_codebooks);
    config.max_position_embeddings = nested_i64(root, "max_position_embeddings", config.max_position_embeddings);
    config.rms_norm_eps = nested_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = nested_f32(root, "rope_theta", config.rope_theta);
    if (const auto * rope = root.find("rope_scaling"); rope != nullptr && rope->is_object()) {
        config.rope_scaling_enabled = true;
        config.rope_scaling_factor = nested_f32(*rope, "factor", config.rope_scaling_factor);
        config.rope_low_freq_factor = nested_f32(*rope, "low_freq_factor", config.rope_low_freq_factor);
        config.rope_high_freq_factor = nested_f32(*rope, "high_freq_factor", config.rope_high_freq_factor);
        config.rope_original_max_position_embeddings = nested_i64(
            *rope,
            "original_max_position_embeddings",
            config.rope_original_max_position_embeddings);
    }
    if (const auto * backbone = root.find("backbone_config"); backbone != nullptr && backbone->is_object()) {
        config.hidden_size = nested_i64(*backbone, "hidden_size", config.hidden_size);
        config.intermediate_size = nested_i64(*backbone, "intermediate_size", config.intermediate_size);
        config.layers = nested_i64(*backbone, "num_hidden_layers", config.layers);
        config.heads = nested_i64(*backbone, "num_attention_heads", config.heads);
        config.kv_heads = nested_i64(*backbone, "num_key_value_heads", config.kv_heads);
        config.head_dim = nested_i64(*backbone, "head_dim", config.head_dim);
        config.rms_norm_eps = nested_f32(*backbone, "rms_norm_eps", config.rms_norm_eps);
        config.rope_theta = nested_f32(*backbone, "rope_theta", config.rope_theta);
        config.rope_scaling_enabled = false;
        if (const auto * rope = backbone->find("rope_scaling"); rope != nullptr && rope->is_object()) {
            config.rope_scaling_enabled = true;
            config.rope_scaling_factor = nested_f32(*rope, "factor", config.rope_scaling_factor);
            config.rope_low_freq_factor = nested_f32(*rope, "low_freq_factor", config.rope_low_freq_factor);
            config.rope_high_freq_factor = nested_f32(*rope, "high_freq_factor", config.rope_high_freq_factor);
            config.rope_original_max_position_embeddings = nested_i64(
                *rope,
                "original_max_position_embeddings",
                config.rope_original_max_position_embeddings);
        }
    }
    config.audio_token_id = nested_i64(root, "audio_token_id", config.audio_token_id);
    config.audio_eos_token_id = nested_i64(root, "audio_eos_token_id", config.audio_eos_token_id);
    config.codebook_pad_token_id = nested_i64(root, "codebook_pad_token_id", config.codebook_pad_token_id);
    config.codebook_eos_token_id = nested_i64(root, "codebook_eos_token_id", config.codebook_eos_token_id);

    if (const auto * text = root.find("text_encoder_config"); text != nullptr && text->is_object()) {
        config.text_hidden_size = nested_i64(*text, "hidden_size", config.text_hidden_size);
        config.text_intermediate_size = nested_i64(*text, "intermediate_size", config.text_intermediate_size);
        config.text_layers = nested_i64(*text, "num_hidden_layers", config.text_layers);
        config.text_heads = nested_i64(*text, "num_attention_heads", config.text_heads);
        config.text_kv_heads = nested_i64(*text, "num_key_value_heads", config.text_kv_heads);
        config.text_head_dim = nested_i64(*text, "head_dim", config.text_head_dim);
        config.text_max_position_embeddings = nested_i64(*text, "max_position_embeddings", config.text_max_position_embeddings);
        config.text_rms_norm_eps = nested_f32(*text, "rms_norm_eps", config.text_rms_norm_eps);
        config.text_query_pre_attn_scalar = nested_f32(*text, "query_pre_attn_scalar", config.text_query_pre_attn_scalar);
        if (const auto * rope = text->find("rope_parameters"); rope != nullptr && rope->is_object()) {
            if (const auto * full = rope->find("full_attention"); full != nullptr && full->is_object()) {
                config.text_rope_theta = nested_f32(*full, "rope_theta", config.text_rope_theta);
                config.text_rope_linear_factor = nested_f32(*full, "factor", config.text_rope_linear_factor);
            }
            const auto layer_types = engine::io::json::optional_string_array(*text, "layer_types");
            if (!layer_types.empty()) {
                if (static_cast<int64_t>(layer_types.size()) != config.text_layers) {
                    throw std::runtime_error("BreezeTTS text layer_types must match text layer count");
                }
                config.text_layer_rope_theta.clear();
                config.text_layer_rope_freq_scale.clear();
                config.text_layer_rope_theta.reserve(layer_types.size());
                config.text_layer_rope_freq_scale.reserve(layer_types.size());
                for (const auto & layer_type : layer_types) {
                    config.text_layer_rope_theta.push_back(rope_theta_for_type(*rope, layer_type, config.text_rope_theta));
                    config.text_layer_rope_freq_scale.push_back(
                        rope_freq_scale_for_type(*rope, layer_type, 1.0F / config.text_rope_linear_factor));
                }
            }
        }
    }
    if (const auto * depth = root.find("depth_decoder_config"); depth != nullptr && depth->is_object()) {
        config.depth_hidden_size = nested_i64(*depth, "hidden_size", config.depth_hidden_size);
        config.depth_intermediate_size = nested_i64(*depth, "intermediate_size", config.depth_intermediate_size);
        config.depth_layers = nested_i64(*depth, "num_hidden_layers", config.depth_layers);
        config.depth_heads = nested_i64(*depth, "num_attention_heads", config.depth_heads);
        config.depth_kv_heads = nested_i64(*depth, "num_key_value_heads", config.depth_kv_heads);
        config.depth_head_dim = nested_i64(*depth, "head_dim", config.depth_head_dim);
        config.depth_rms_norm_eps = nested_f32(*depth, "rms_norm_eps", config.depth_rms_norm_eps);
        config.depth_rope_theta = nested_f32(*depth, "rope_theta", config.depth_rope_theta);
        config.depth_rope_scaling_enabled = false;
        if (const auto * rope = depth->find("rope_scaling"); rope != nullptr && rope->is_object()) {
            config.depth_rope_scaling_enabled = true;
            config.depth_rope_scaling_factor = nested_f32(*rope, "factor", config.depth_rope_scaling_factor);
            config.depth_rope_low_freq_factor = nested_f32(*rope, "low_freq_factor", config.depth_rope_low_freq_factor);
            config.depth_rope_high_freq_factor = nested_f32(*rope, "high_freq_factor", config.depth_rope_high_freq_factor);
            config.depth_rope_original_max_position_embeddings = nested_i64(
                *rope,
                "original_max_position_embeddings",
                config.depth_rope_original_max_position_embeddings);
        }
    }
}

void validate_shapes(const engine::assets::TensorSource & source, const BreezeTTSConfig & config) {
    engine::assets::require_tensor_shape(source, "embed_text_tokens.weight", {config.text_vocab_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "text_encoder_proj.weight", {config.hidden_size, config.text_hidden_size});
    engine::assets::require_tensor_shape(source, "lm_head.weight", {config.lm_head_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "depth_decoder.model.embed_tokens.weight", {config.num_codebooks * config.vocab_size, config.hidden_size});
    engine::assets::require_tensor_shape(source, "depth_decoder.codebooks_head.weight", {config.num_codebooks - 1, config.depth_hidden_size, config.vocab_size});
}

}  // namespace

std::shared_ptr<const BreezeTTSAssets> load_breeze_tts_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<BreezeTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->model_root = assets->resources.model_root();
    assets->weights = assets->resources.open_tensor_source("model_weights");
    parse_config(assets->resources.parse_json("config_json"), assets->config);
    validate_shapes(*assets->weights, assets->config);
    return assets;
}

}  // namespace engine::models::breeze_tts

#include "engine/models/fish_audio/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::fish_audio {
namespace json = engine::io::json;
namespace {

FishAudioTextConfig parse_text_config(const json::Value & value) {
    if (json::optional_string(value, "model_type", "") != "fish_qwen3") {
        throw std::runtime_error("Fish Audio text_config.model_type mismatch");
    }
    FishAudioTextConfig config;
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.n_layer = json::require_i64(value, "n_layer");
    config.dim = json::require_i64(value, "dim");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.n_head = json::require_i64(value, "n_head");
    config.n_local_heads = json::optional_i64(value, "n_local_heads", config.n_head);
    config.head_dim = json::require_i64(value, "head_dim");
    config.max_seq_len = json::require_i64(value, "max_seq_len");
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    config.norm_eps = json::optional_f32(value, "norm_eps", config.norm_eps);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.attention_qk_norm = json::optional_bool(value, "attention_qk_norm", config.attention_qk_norm);
    engine::io::require_positive(config.vocab_size, "text vocab_size");
    engine::io::require_positive(config.n_layer, "text n_layer");
    engine::io::require_positive(config.dim, "text dim");
    engine::io::require_positive(config.intermediate_size, "text intermediate_size");
    engine::io::require_positive(config.n_head, "text n_head");
    engine::io::require_positive(config.n_local_heads, "text n_local_heads");
    engine::io::require_positive(config.head_dim, "text head_dim");
    engine::io::require_positive(config.max_seq_len, "text max_seq_len");
    engine::io::require_divisible(config.n_head, config.n_local_heads, "text n_head / n_local_heads");
    return config;
}

FishAudioFastConfig parse_fast_config(const json::Value & value) {
    if (json::optional_string(value, "model_type", "") != "fish_qwen3_audio_decoder") {
        throw std::runtime_error("Fish Audio audio_decoder_config.model_type mismatch");
    }
    FishAudioFastConfig config;
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.num_codebooks = json::require_i64(value, "num_codebooks");
    config.n_layer = json::require_i64(value, "n_layer");
    config.dim = json::require_i64(value, "dim");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.n_head = json::require_i64(value, "n_head");
    config.n_local_heads = json::optional_i64(value, "n_local_heads", config.n_head);
    config.head_dim = json::require_i64(value, "head_dim");
    config.max_seq_len = json::optional_i64(value, "max_seq_len", config.num_codebooks + 1);
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    config.norm_eps = json::optional_f32(value, "norm_eps", config.norm_eps);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.attention_qk_norm = json::optional_bool(value, "attention_qk_norm", config.attention_qk_norm);
    engine::io::require_positive(config.vocab_size, "fast vocab_size");
    engine::io::require_positive(config.num_codebooks, "fast num_codebooks");
    engine::io::require_positive(config.n_layer, "fast n_layer");
    engine::io::require_positive(config.dim, "fast dim");
    engine::io::require_positive(config.intermediate_size, "fast intermediate_size");
    engine::io::require_positive(config.n_head, "fast n_head");
    engine::io::require_positive(config.n_local_heads, "fast n_local_heads");
    engine::io::require_positive(config.head_dim, "fast head_dim");
    engine::io::require_divisible(config.n_head, config.n_local_heads, "fast n_head / n_local_heads");
    return config;
}

FishAudioConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    FishAudioConfig config;
    config.model_type = json::optional_string(root, "model_type", "");
    if (config.model_type != "fish_qwen3_omni") {
        throw std::runtime_error("Fish Audio model_type mismatch");
    }
    config.torch_dtype = json::optional_string(root, "torch_dtype", config.torch_dtype);
    config.semantic_start_token_id = json::require_i64(root, "semantic_start_token_id");
    config.semantic_end_token_id = json::require_i64(root, "semantic_end_token_id");
    config.im_end_token_id = json::require_i64(root, "eos_token_id");
    config.norm_fastlayer_input = json::optional_bool(root, "norm_fastlayer_input", config.model_type == "fish_qwen3_omni");
    config.text = parse_text_config(root.require("text_config"));
    config.fast = parse_fast_config(root.require("audio_decoder_config"));
    config.codec.total_codebooks = config.fast.num_codebooks;
    if (config.fast.dim != config.text.dim) {
        throw std::runtime_error("Fish Audio fast dim must match text dim for S2-Pro");
    }
    if (!config.text.tie_word_embeddings) {
        throw std::runtime_error("Fish Audio S2-Pro expects tied text embeddings");
    }
    if (config.semantic_start_token_id <= 0 || config.semantic_end_token_id < config.semantic_start_token_id) {
        throw std::runtime_error("Fish Audio semantic token range is invalid");
    }
    return config;
}

void validate_weight_anchors(const FishAudioAssets & assets) {
    assets.model_weights->require_metadata("embeddings.weight");
    assets.model_weights->require_metadata("codebook_embeddings.weight");
    assets.model_weights->require_metadata("layers.0.attention.q_proj.weight");
    assets.model_weights->require_metadata("layers.0.attention.k_proj.weight");
    assets.model_weights->require_metadata("layers.0.attention.v_proj.weight");
    assets.model_weights->require_metadata("fast_layers.0.attention.q_proj.weight");
    assets.model_weights->require_metadata("fast_embeddings.weight");
    assets.model_weights->require_metadata("fast_output.weight");
    assets.codec_weights->require_metadata("quantizer.semantic_quantizer.quantizers.0.codebook.weight");
    assets.codec_weights->require_metadata("decoder.model.0.conv.weight");
}

}  // namespace

std::shared_ptr<const FishAudioAssets> load_fish_audio_assets(const std::filesystem::path & model_path) {
    FishAudioAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("fish_audio"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.codec_weights = assets.resources.open_tensor_source("codec_weights");
    validate_weight_anchors(assets);
    return std::make_shared<FishAudioAssets>(std::move(assets));
}

}  // namespace engine::models::fish_audio

#include "engine/community_models/soprano_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::community_models::soprano_tts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "soprano_tts";

SopranoTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (json::require_string(root, "model_type") != "qwen3") {
        throw std::runtime_error("Soprano config must use model_type qwen3");
    }
    SopranoTTSConfig out;
    out.hidden_size = json::require_i64(root, "hidden_size");
    out.intermediate_size = json::require_i64(root, "intermediate_size");
    out.layers = json::require_i64(root, "num_hidden_layers");
    out.attention_heads = json::require_i64(root, "num_attention_heads");
    out.kv_heads = json::require_i64(root, "num_key_value_heads");
    if (const auto * hd = root.find("head_dim")) {
        out.head_dim = hd->as_i64();
    } else {
        out.head_dim = out.hidden_size / out.attention_heads;
    }
    out.vocab_size = json::require_i64(root, "vocab_size");
    out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
    if (const auto * rope = root.find("rope_parameters")) {
        out.rope_theta = json::optional_f32(*rope, "rope_theta", out.rope_theta);
    } else {
        out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
    }
    out.max_position_embeddings = json::optional_i64(root, "max_position_embeddings", out.max_position_embeddings);
    if (const auto * eos = root.find("eos_token_id")) {
        out.eos_token_id = static_cast<int32_t>(eos->as_i64());
    }
    if (const auto * bos = root.find("bos_token_id")) {
        out.bos_token_id = static_cast<int32_t>(bos->as_i64());
    }
    // Decoder dimensions are fixed by the SopranoDecoder architecture.
    out.decoder_input_channels = out.hidden_size;
    return out;
}

}  // namespace

std::shared_ptr<const SopranoTTSAssets> load_soprano_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SopranoTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path, engine::model_spec::default_spec_path(kFamily));
    assets->config = parse_config(assets->resources);
    assets->backbone_weights = assets->resources.open_tensor_source("backbone");
    assets->decoder_weights = assets->resources.open_tensor_source("decoder");
    return assets;
}

}  // namespace engine::community_models::soprano_tts
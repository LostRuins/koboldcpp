#include "engine/models/maya1/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::maya1 {
namespace {

namespace json = engine::io::json;

Maya1Config parse_config(const assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("config");
  if (json::require_string(root, "model_type") != "llama") {
    throw std::runtime_error("Maya1 config must use model_type llama");
  }
  Maya1Config out;
  out.hidden_size = json::require_i64(root, "hidden_size");
  out.intermediate_size = json::require_i64(root, "intermediate_size");
  out.layers = json::require_i64(root, "num_hidden_layers");
  out.attention_heads = json::require_i64(root, "num_attention_heads");
  out.kv_heads = json::require_i64(root, "num_key_value_heads");
  out.head_dim = json::optional_i64(root, "head_dim",
                                    out.hidden_size / out.attention_heads);
  out.vocab_size = json::require_i64(root, "vocab_size");
  out.max_position_embeddings =
      json::require_i64(root, "max_position_embeddings");
  out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
  out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
  const auto *rope = root.find("rope_scaling");
  if (rope == nullptr || !rope->is_object()) {
    throw std::runtime_error("Maya1 config requires rope_scaling");
  }
  out.rope.factor = json::optional_f32(*rope, "factor", out.rope.factor);
  out.rope.low_freq_factor =
      json::optional_f32(*rope, "low_freq_factor", out.rope.low_freq_factor);
  out.rope.high_freq_factor =
      json::optional_f32(*rope, "high_freq_factor", out.rope.high_freq_factor);
  out.rope.original_max_position_embeddings =
      json::optional_i64(*rope, "original_max_position_embeddings",
                         out.rope.original_max_position_embeddings);
  return out;
}

Maya1GenerationConfig
parse_generation(const assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("generation_config");
  Maya1GenerationConfig out;
  out.top_p = json::optional_f32(root, "top_p", out.top_p);
  return out;
}

} // namespace

std::shared_ptr<const Maya1Assets>
load_maya1_assets(const std::filesystem::path &model_path) {
  auto resources =
      model_spec::load_resource_bundle_for_family(model_path, "maya1");
  Maya1Assets out;
  out.config = parse_config(resources);
  out.generation = parse_generation(resources);
  out.model_weights = resources.open_tensor_source("model_weights");
  out.codec_weights = resources.open_tensor_source("codec_weights");
  out.resources = std::move(resources);
  return std::make_shared<Maya1Assets>(std::move(out));
}

} // namespace engine::models::maya1

#include "engine/models/supertonic/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine::models::supertonic {
namespace json = engine::io::json;
namespace {

SupertonicConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("tts_config");
    SupertonicConfig config;
    const auto & ae = root.require("ae");
    const auto & ttl = root.require("ttl");
    config.sample_rate = static_cast<int>(json::require_i64(ae, "sample_rate"));
    config.base_chunk_size = json::require_i64(ae, "base_chunk_size");
    config.chunk_compress_factor = json::require_i64(ttl, "chunk_compress_factor");
    config.latent_dim = json::require_i64(ttl, "latent_dim");
    engine::io::require_positive(config.sample_rate, "sample_rate");
    engine::io::require_positive(config.base_chunk_size, "base_chunk_size");
    engine::io::require_positive(config.chunk_compress_factor, "chunk_compress_factor");
    engine::io::require_positive(config.latent_dim, "latent_dim");
    return config;
}

std::unordered_map<uint32_t, int64_t> parse_unicode_indexer(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("unicode_indexer");
    std::unordered_map<uint32_t, int64_t> out;
    if (root.is_array()) {
        const auto & values = root.as_array();
        out.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const int64_t token = values[i].as_i64();
            if (token >= 0) {
                out.emplace(static_cast<uint32_t>(i), token);
            }
        }
    } else if (root.is_object()) {
        out.reserve(root.as_object().size());
        for (const auto & [key, value] : root.as_object()) {
            size_t parsed = 0;
            const auto codepoint = static_cast<uint32_t>(std::stoul(key, &parsed, 10));
            if (parsed != key.size()) {
                throw std::runtime_error("Supertonic unicode indexer contains a non-numeric codepoint key: " + key);
            }
            const int64_t token = value.as_i64();
            if (token >= 0) {
                out.emplace(codepoint, token);
            }
        }
    } else {
        throw std::runtime_error("Supertonic unicode indexer must be an array or object");
    }
    if (out.empty()) {
        throw std::runtime_error("Supertonic unicode indexer is empty");
    }
    return out;
}

}  // namespace

std::shared_ptr<const SupertonicAssets> load_supertonic_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SupertonicAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("supertonic"));
    assets->config = parse_config(assets->resources);
    assets->weights = assets->resources.open_tensor_source("weights");
    assets->unicode_indexer = parse_unicode_indexer(assets->resources);
    return assets;
}

}  // namespace engine::models::supertonic

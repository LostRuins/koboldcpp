#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::models::supertonic {

struct SupertonicConfig {
    int sample_rate = 44100;
    int64_t base_chunk_size = 512;
    int64_t chunk_compress_factor = 6;
    int64_t latent_dim = 24;
};

struct SupertonicAssets {
    assets::ResourceBundle resources;
    SupertonicConfig config;
    std::shared_ptr<const assets::TensorSource> weights;
    std::unordered_map<uint32_t, int64_t> unicode_indexer;
};

std::shared_ptr<const SupertonicAssets> load_supertonic_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::supertonic

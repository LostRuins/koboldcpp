#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::models::chatterbox {

struct ChatterboxAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> voice_encoder_weights;
    std::shared_ptr<const engine::assets::TensorSource> s3gen_weights;
    std::shared_ptr<const engine::assets::TensorSource> t3_english_weights;
    std::shared_ptr<const engine::assets::TensorSource> t3_multilingual_v2_weights;
    std::shared_ptr<const engine::assets::TensorSource> t3_multilingual_v3_weights;
};

std::shared_ptr<const ChatterboxAssets> load_chatterbox_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::chatterbox

#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::community_models::chatterbox_turbo {

struct ChatterboxTurboAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> t3_turbo_weights;
    std::shared_ptr<const engine::assets::TensorSource> builtin_conditionals_turbo;
    std::shared_ptr<const engine::assets::TensorSource> s3gen_weights;
};

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::chatterbox_turbo

#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/models/fish_audio/types.h"

#include <filesystem>
#include <memory>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::fish_audio {

struct FishAudioAssets {
    assets::ResourceBundle resources;
    FishAudioConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const FishAudioAssets> load_fish_audio_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::fish_audio

#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/ming_audio_vae_runtime.h"
#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"

namespace engine::models::ming3_tts {

struct MingConfig {
    modules::MingBackboneConfig backbone;
    int64_t vocabulary_size = 0;
    modules::MingFlowConfig flow;
    modules::MingAudioCodecConfig codec;
};

MingConfig parse_ming_config(const io::json::Value & config);

struct MingAssets {
    assets::ResourceBundle resources;
    MingConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> speaker_weights;
};

std::shared_ptr<const MingAssets> load_ming3_tts_assets(const std::filesystem::path & path);

}  // namespace engine::models::ming3_tts

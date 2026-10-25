#include "engine/models/chatterbox/assets.h"

#include "engine/framework/model_spec/package.h"

namespace engine::models::chatterbox {

std::shared_ptr<const ChatterboxAssets> load_chatterbox_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<ChatterboxAssets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("chatterbox"));
    out->voice_encoder_weights = out->resources.open_tensor_source("voice_encoder_weights");
    out->s3gen_weights = out->resources.open_tensor_source("s3gen_weights");
    out->t3_english_weights = out->resources.open_tensor_source("t3_english_weights");
    out->t3_multilingual_v2_weights = out->resources.open_tensor_source("t3_multilingual_v2_weights");
    out->t3_multilingual_v3_weights = out->resources.open_tensor_source("t3_multilingual_v3_weights");
    return out;
}

}  // namespace engine::models::chatterbox

#include "engine/community_models/chatterbox_turbo/assets.h"

#include "engine/framework/model_spec/package.h"

namespace engine::community_models::chatterbox_turbo {

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<ChatterboxTurboAssets>();
    out->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("chatterbox_turbo"));
    out->t3_turbo_weights = out->resources.open_tensor_source("t3_turbo_weights");
    out->builtin_conditionals_turbo = out->resources.open_tensor_source("builtin_conditionals_turbo");
    out->s3gen_weights = out->resources.open_tensor_source("s3gen_weights");
    return out;
}

}  // namespace engine::community_models::chatterbox_turbo

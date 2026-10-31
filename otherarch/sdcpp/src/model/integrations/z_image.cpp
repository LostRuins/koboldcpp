#include "model/diffusion/z_image.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_z_image(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                   tensor_storage_map,
                                                                   version,
                                                                   "",
                                                                   false,
                                                                   weight_manager,
                                                                   tokenizers);
    result.diffusion               = std::make_shared<ZImage::ZImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            version,
                                                                            weight_manager);
    return true;
}

const std::vector<ModelDefinition>& z_image_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_Z_IMAGE;
            definition.geometry                       = {1, 16, 3};
            definition.name                           = "Z-Image";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::FLUX;
            definition.sampling                       = {FLOW_PRED, 3.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_z_image;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

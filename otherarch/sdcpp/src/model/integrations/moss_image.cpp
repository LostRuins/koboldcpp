#include "model_definition.h"

#include "conditioning/conditioner.hpp"
#include "model/diffusion/moss_image.hpp"
#include "pipeline/model_builders.h"

static bool build_moss_image(const sd::model_builders::Context& ctx,
                             const TokenizerConfig& tokenizers,
                             sd::model_builders::CoreRunners& runners) {
    runners.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                        ctx.tensor_storage_map, ctx.version, "", false,
                                                        ctx.weight_manager, tokenizers);
    runners.diffusion   = std::make_shared<MOSSImage::MOSSImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                       ctx.tensor_storage_map, "model.diffusion_model", ctx.weight_manager);
    return true;
}

const std::vector<ModelDefinition>& moss_image_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        ModelDefinition definition;
        definition.version                        = VERSION_MOSS_IMAGE;
        definition.geometry                       = {1, 128, 3};
        definition.name                           = "MOSS-Image";
        definition.traits.is_dit                  = true;
        definition.traits.vae_kind                = ModelVAEKind::FLUX2;
        definition.sampling                       = {FLOW_PRED, 5.f};
        definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
        definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
        definition.build                          = build_moss_image;
        return std::vector<ModelDefinition>{std::move(definition)};
    }();
    return definitions;
}

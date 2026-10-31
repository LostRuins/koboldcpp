#include "conditioning/conditioner.hpp"
#include "model/diffusion/flux.hpp"
#include "model/integrations/flux.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_flux2(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    bool is_chroma                 = false;
    result.conditioner             = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                       tensor_storage_map,
                                                       version,
                                                       "",
                                                       false,
                                                       weight_manager,
                                                       tokenizers);
    result.diffusion               = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                          tensor_storage_map,
                                                          "model.diffusion_model",
                                                          version,
                                                          weight_manager,
                                                          sd_ctx_params->model_args);
    return true;
}

const std::vector<ModelDefinition>& flux2_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX2;
            definition.geometry                         = {1, 128, 3};
            definition.ref_image_preset                 = "flux2";
            definition.name                             = "Flux.2";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX2;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX2_SCHEDULER;
            definition.build                            = build_flux2;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX2_KLEIN;
            definition.geometry                         = {1, 128, 3};
            definition.ref_image_preset                 = "flux2";
            definition.name                             = "Flux.2 klein";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX2;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX2_SCHEDULER;
            definition.build                            = build_flux2;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_SEFI_IMAGE;
            definition.geometry                       = {1, 144, 3};
            definition.ref_image_preset               = "flux2";
            definition.name                           = "SeFi-Image";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::FLUX2;
            definition.sampling                       = {SEFI_FLOW_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_flux2;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

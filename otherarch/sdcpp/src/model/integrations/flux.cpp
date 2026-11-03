#include "model/diffusion/flux.hpp"
#include "conditioning/conditioner.hpp"
#include "model/integrations/flux.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_flux(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    bool is_chroma                 = false;
    for (auto pair : tensor_storage_map) {
        if (pair.first.find("distilled_guidance_layer.in_proj.weight") != std::string::npos) {
            is_chroma = true;
            break;
        }
    }
    if (is_chroma) {
        result.conditioner = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                              tensor_storage_map,
                                                              false,
                                                              1,
                                                              false,
                                                              weight_manager,
                                                              sd_ctx_params->model_args);
    } else if (version == VERSION_OVIS_IMAGE) {
        result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                           tensor_storage_map,
                                                           version,
                                                           "",
                                                           false,
                                                           weight_manager,
                                                           tokenizers);
    } else {
        result.conditioner = std::make_shared<FluxCLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                tensor_storage_map,
                                                                weight_manager,
                                                                tokenizers);
    }
    result.diffusion = std::make_shared<Flux::FluxRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                          tensor_storage_map,
                                                          "model.diffusion_model",
                                                          version,
                                                          weight_manager,
                                                          sd_ctx_params->model_args);
    return true;
}

const std::vector<ModelDefinition>& flux_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX;
            definition.geometry                         = {1, 16, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Flux";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX_FILL;
            definition.geometry                         = {1, 16, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Flux Fill";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_inpaint                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLUX_CONTROLS;
            definition.geometry                         = {1, 16, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Flux Control";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_control                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_FLEX_2;
            definition.geometry                         = {1, 16, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Flex.2";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.is_inpaint                = true;
            definition.traits.is_control                = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_CHROMA_RADIANCE;
            definition.geometry                         = {1, 3, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Chroma Radiance";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.traits.fake_vae                  = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_OVIS_IMAGE;
            definition.geometry                         = {1, 16, 3};
            definition.ref_image_preset                 = "flux_kontext";
            definition.name                             = "Ovis Image";
            definition.traits.is_dit                    = true;
            definition.traits.vae_kind                  = ModelVAEKind::FLUX;
            definition.sampling                         = {FLUX_FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = FLUX_SCHEDULER;
            definition.build                            = build_flux;
            definition.sampling.flow_shift_from_weights = flux_guidance_flow_shift;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

float flux_guidance_flow_shift(const String2TensorStorage& weights) {
    for (const auto& [name, tensor] : weights) {
        if (starts_with(name, "model.diffusion_model.guidance_in.in_layer.weight"))
            return 1.15f;
    }
    return 1.f;
}

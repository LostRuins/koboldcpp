#include "model/diffusion/qwen_image.hpp"
#include "conditioning/conditioner.hpp"
#include "model/diffusion/qwen_image_2_1.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_qwen_image(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    bool enable_vision             = version != VERSION_QWEN_IMAGE_LAYERED;
    result.conditioner             = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                   tensor_storage_map,
                                                                   version,
                                                                   "",
                                                                   enable_vision,
                                                                   weight_manager,
                                                                   tokenizers);
    if (version == VERSION_QWEN_IMAGE_2_1) {
        result.diffusion = std::make_shared<Qwen::QwenImage21Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                     tensor_storage_map,
                                                                     "model.diffusion_model",
                                                                     weight_manager,
                                                                     sd_ctx_params->model_args);
    } else {
        result.diffusion = std::make_shared<Qwen::QwenImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                   tensor_storage_map,
                                                                   "model.diffusion_model",
                                                                   version,
                                                                   weight_manager,
                                                                   sd_ctx_params->model_args);
    }
    return true;
}

const std::vector<ModelDefinition>& qwen_image_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_QWEN_IMAGE;
            definition.geometry                       = {1, 16, 3};
            definition.ref_image_preset               = "qwen";
            definition.name                           = "Qwen Image";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::WAN;
            definition.traits.video_tae               = true;
            definition.sampling                       = {FLOW_PRED, 3.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_qwen_image;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_QWEN_IMAGE_LAYERED;
            definition.geometry                       = {1, 16, 4};
            definition.ref_image_preset               = "qwen_layered";
            definition.name                           = "Qwen Image Layered";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::WAN;
            definition.traits.video_tae               = true;
            definition.sampling                       = {FLOW_PRED, 3.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_qwen_image;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                        = VERSION_QWEN_IMAGE_2_1;
            definition.geometry                       = {2, 64, 4};
            definition.ref_image_preset               = "qwen";
            definition.name                           = "Qwen Image 2.1";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::WAN;
            definition.sampling                       = {FLOW_PRED, 3.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = FLUX_SCHEDULER;
            definition.build                          = build_qwen_image;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

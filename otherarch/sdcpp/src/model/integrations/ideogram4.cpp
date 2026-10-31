#include "model/diffusion/ideogram4.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_ideogram4(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
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
    result.diffusion               = std::make_shared<Ideogram4::Ideogram4Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                                  tensor_storage_map,
                                                                                  "model.diffusion_model",
                                                                                  weight_manager);
    return true;
}

const std::vector<ModelDefinition>& ideogram4_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_IDEOGRAM4;
            definition.geometry                       = {1, 128, 3};
            definition.name                           = "Ideogram 4";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::FLUX2;
            definition.sampling                       = {FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = LOGIT_NORMAL_SCHEDULER;
            definition.build                          = build_ideogram4;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

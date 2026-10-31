#include "conditioning/conditioner.hpp"
#include "model/diffusion/ltxv.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_ltxav(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<LTXAVEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                     tensor_storage_map,
                                                                     "text_encoders.llm",
                                                                     "text_embedding_projection",
                                                                     weight_manager,
                                                                     tokenizers);
    result.diffusion               = std::make_shared<LTXV::LTXAVRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         weight_manager);
    return true;
}

const std::vector<ModelDefinition>& ltxav_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_LTXAV;
            definition.geometry                         = {1, 128, 3};
            definition.name                             = "LTXAV";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::LTXAV;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLUX_FLOW_PRED, 2.37f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = LTX2_SCHEDULER;
            definition.build                            = build_ltxav;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

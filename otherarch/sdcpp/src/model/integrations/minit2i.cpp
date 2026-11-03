#include "model/diffusion/minit2i.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_minit2i(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<MiniT2IConditioner>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                              tensor_storage_map,
                                                              weight_manager);
    result.diffusion               = std::make_shared<MiniT2I::MiniT2IRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                tensor_storage_map,
                                                                "model.diffusion_model.model.net",
                                                                weight_manager);
    return true;
}

const std::vector<ModelDefinition>& minit2i_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_MINIT2I;
            definition.geometry                       = {1, 3, 3};
            definition.name                           = "MiniT2I";
            definition.traits.is_dit                  = true;
            definition.traits.fake_vae                = true;
            definition.sampling                       = {MINIT2I_FLOW_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_minit2i;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

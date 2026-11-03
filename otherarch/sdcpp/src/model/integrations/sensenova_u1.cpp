#include "model/diffusion/sensenova_u1.h"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_sensenova_u1(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<SenseNovaU1Conditioner>();
    result.diffusion               = std::make_shared<SenseNovaU1::SenseNovaU1Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                        tensor_storage_map,
                                                                        "",
                                                                        weight_manager);
    return true;
}

const std::vector<ModelDefinition>& sensenova_u1_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_SENSENOVA_U1_5;
            definition.geometry                       = {32, 3, 3};
            definition.name                           = "SenseNova U1.5";
            definition.traits.is_dit                  = true;
            definition.traits.fake_vae                = true;
            definition.sampling                       = {SENSENOVA_U1_FLOW_PRED, 3.f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_sensenova_u1;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

#include "model/diffusion/lingbot_video.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_lingbot_video(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    bool enable_vision             = false;
    for (const auto& [name, _] : tensor_storage_map) {
        if (starts_with(name, "text_encoders.llm.visual.")) {
            enable_vision = true;
            break;
        }
    }
    result.conditioner = std::make_shared<LLMEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                       tensor_storage_map,
                                                       version,
                                                       "",
                                                       enable_vision,
                                                       weight_manager,
                                                       tokenizers);
    result.diffusion   = std::make_shared<LingBotVideo::LingBotVideoRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            weight_manager,
                                                                            sd_ctx_params->model_args);
    return true;
}

const std::vector<ModelDefinition>& lingbot_video_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_LINGBOT_VIDEO;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "LingBot Video";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 3.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_lingbot_video;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

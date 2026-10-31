#include "model/diffusion/pixart.hpp"
#include "conditioning/conditioner.hpp"
#include "model/vae/auto_encoder_kl.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static void configure_pixart_vae(const sd::model_builders::Context& ctx,
                                 const sd::model_builders::VAEOptions&,
                                 VAE& vae) {
    auto& model = static_cast<AutoEncoderKL&>(vae);
    // Alpha-512 and Sigma share tensor layouts; Alpha-512 needs an explicit scale override.
    if (ctx.tensor_storage_map.count("model.diffusion_model.csize_embedder.mlp.0.weight") != 0) {
        model.scale_factor = 0.18215f;
    }
    for (const auto& [key, value] : parse_key_value_args(ctx.params.model_args, "model arg")) {
        if (key == "pixart_vae_scale_factor") {
            float parsed = 0.f;
            if (parse_strict_float(value, parsed) && std::isfinite(parsed) && parsed > 0.f) {
                model.scale_factor = parsed;
            } else {
                LOG_WARN("ignoring invalid PixArt model arg '%s=%s'", key.c_str(), value.c_str());
            }
        }
    }
    LOG_VERBOSE("pixart: VAE scale factor = %.5f", model.scale_factor);
}

static bool build_pixart(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                      tensor_storage_map,
                                                                      true,
                                                                      0,
                                                                      false,
                                                                      weight_manager,
                                                                      sd_ctx_params->model_args);
    result.diffusion               = std::make_shared<PixArt::PixArtRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                            tensor_storage_map,
                                                                            "model.diffusion_model",
                                                                            weight_manager,
                                                                            sd_ctx_params->model_args);
    return true;
}

const std::vector<ModelDefinition>& pixart_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_PIXART;
            definition.geometry                       = {2, 4, 3};
            definition.name                           = "PixArt";
            definition.traits.is_dit                  = true;
            definition.sampling                       = {EPS_PRED, INFINITY};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = DISCRETE_SCHEDULER;
            definition.build                          = build_pixart;
            definition.configure_vae                  = configure_pixart_vae;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

#include "model/diffusion/llada_image.hpp"
#include "conditioning/conditioner.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_llada_image(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<LLaDAImageEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                              tensor_storage_map,
                                                              "text_encoders.llm",
                                                              "queryformer",
                                                              "text_projection",
                                                              "sigvq",
                                                              weight_manager,
                                                              tokenizers);
    result.diffusion               = std::make_shared<LLaDAImage::LLaDAImageRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model",
                                                                      weight_manager);
    return true;
}

const std::vector<ModelDefinition>& llada_image_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                        = VERSION_LLADA_IMAGE;
            definition.geometry                       = {1, 128, 3};
            definition.ref_image_preset               = "llada_image";
            definition.name                           = "LLaDA-Image";
            definition.traits.is_dit                  = true;
            definition.traits.vae_kind                = ModelVAEKind::FLUX2;
            definition.sampling                       = {FLOW_PRED, 1.0f};
            definition.sampling.default_sample_method = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler     = LLADA_IMAGE_SCHEDULER;
            definition.build                          = build_llada_image;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

// The attention projections keep their diffusers names (JointAttention's split_qkv mode), so
// only the patch-size-keyed dicts need flattening. Latents arrive already patchified from the
// Flux2 VAE, so the only patch key is 1-1.

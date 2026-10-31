#include "model/diffusion/wan.hpp"
#include "conditioning/conditioner.hpp"
#include "model/audio/wav2vec2.hpp"
#include "model_definition.h"
#include "pipeline/model_builders.h"

static bool build_wan(const sd::model_builders::Context& ctx, const TokenizerConfig& tokenizers, sd::model_builders::CoreRunners& result) {
    const auto* sd_ctx_params      = &ctx.params;
    const auto& tensor_storage_map = ctx.tensor_storage_map;
    const auto version             = ctx.version;
    const auto& weight_manager     = ctx.weight_manager;
    result.conditioner             = std::make_shared<T5CLIPEmbedder>(ctx.backends.runtime_backend(SDBackendModule::TE),
                                                                      tensor_storage_map,
                                                                      true,
                                                                      0,
                                                                      true,
                                                                      weight_manager);
    result.diffusion               = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                      tensor_storage_map,
                                                                      "model.diffusion_model",
                                                                      version,
                                                                      weight_manager);
    if (strlen(SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path)) > 0) {
        result.high_noise_diffusion = std::make_shared<WAN::WanRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                       tensor_storage_map,
                                                                       "model.high_noise_diffusion_model",
                                                                       version,
                                                                       weight_manager);
    }
    if (result.diffusion->get_desc() == "Wan2.1-I2V-14B" ||
        result.diffusion->get_desc() == "Wan2.1-FLF2V-14B" ||
        result.diffusion->get_desc() == "Wan2.1-I2V-1.3B") {
        if (!sd::model_builders::ensure_backend_pair(ctx.backends, SDBackendModule::CLIP_VISION)) {
            return false;
        }
        result.clip_vision = std::make_shared<FrozenCLIPVisionEmbedder>(ctx.backends.runtime_backend(SDBackendModule::CLIP_VISION),
                                                                        tensor_storage_map,
                                                                        weight_manager);
    }
    if (version == VERSION_WAN2_2_S2V &&
        tensor_storage_map.count("wav2vec2.encoder.layer_norm.bias") > 0) {
        if (!sd::model_builders::ensure_backend_pair(ctx.backends, SDBackendModule::AUDIO_ENCODER)) {
            return false;
        }
        result.audio_encoder = std::make_shared<Wav2Vec2::Wav2Vec2ModelRunner>(ctx.backends.runtime_backend(SDBackendModule::AUDIO_ENCODER),
                                                                               tensor_storage_map,
                                                                               "wav2vec2.",
                                                                               weight_manager);
    }
    return true;
}

const std::vector<ModelDefinition>& wan_definitions() {
    static const std::vector<ModelDefinition> definitions = [] {
        std::vector<ModelDefinition> result;
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.x";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_I2V;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.2 I2V";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_TI2V;
            definition.geometry                         = {2, 48, 3};
            definition.name                             = "Wan 2.2 TI2V";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        {
            ModelDefinition definition;
            definition.version                          = VERSION_WAN2_2_S2V;
            definition.geometry                         = {2, 16, 3};
            definition.name                             = "Wan 2.2 S2V";
            definition.traits.is_dit                    = true;
            definition.traits.supports_video_generation = true;
            definition.traits.vae_kind                  = ModelVAEKind::WAN;
            definition.traits.video_tae                 = true;
            definition.sampling                         = {FLOW_PRED, 5.f};
            definition.sampling.default_sample_method   = EULER_SAMPLE_METHOD;
            definition.sampling.default_scheduler       = DISCRETE_SCHEDULER;
            definition.build                            = build_wan;
            result.push_back(std::move(definition));
        }
        return result;
    }();
    return definitions;
}

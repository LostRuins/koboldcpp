#include "api/api.h"
#include "api/options.h"
#include "api/result.h"
#include "pipeline/diffusion_engine.h"
#include "pipeline/generation.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

static sd_status_t validate_generation(StableDiffusionGGML* engine, const sd::CacheConfig& cache, bool video) {
    if (video ? !sd_version_supports_video_generation(engine->version) &&
                    !(engine->config_->animatediff_loaded && sd_version_supports_animatediff(engine->version))
              : !sd_version_supports_image_generation(engine->version))
        return sd::api::error(SD_STATUS_UNSUPPORTED, nullptr, "model does not support this operation");
    if (cache.mode == SD_CACHE_UCACHE && !sd_version_is_unet(engine->version))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "cache.mode", "ucache requires a UNet model");
    if ((cache.mode == SD_CACHE_EASYCACHE || cache.mode == SD_CACHE_DBCACHE ||
         cache.mode == SD_CACHE_TAYLORSEER || cache.mode == SD_CACHE_CACHE_DIT) &&
        !sd_version_is_dit(engine->version))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "cache.mode", "selected cache requires a DiT model");
    if (cache.mode == SD_CACHE_SPECTRUM && !sd_version_is_unet(engine->version) && !sd_version_is_dit(engine->version))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "cache.mode", "spectrum requires a UNet or DiT model");
    return SD_STATUS_OK;
}

static sd_status_t generation_failure(StableDiffusionGGML* engine) {
    if (engine->get_cancel_flag() == SD_CANCEL_ALL)
        return sd::api::error(SD_STATUS_CANCELLED, nullptr, "generation cancelled");
    return sd::api::error(SD_STATUS_GENERATION_FAILED, nullptr, "generation failed; see log for details");
}

static sd_status_t run_image(sd_ctx_t* ctx, const sd::ImageConfig* params, sd::ImageGenerationResult** result_out, const sd_opt_t* keys = nullptr) {
    if (result_out)
        *result_out = nullptr;
    if (!result_out || !ctx || !ctx->sd || !params)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context, params and result_out are required");
    StableDiffusionGGML::ExecutionScope execution(*ctx->sd);
    if (!execution.ready)
        return execution.operation.acquired
                   ? sd::api::error(SD_STATUS_LOAD_FAILED, nullptr, "could not refresh model sources")
                   : sd::api::error(SD_STATUS_BUSY, nullptr, "context is already executing");
    auto status = validate_generation(ctx->sd, params->cache, false);
    if (status != SD_STATUS_OK)
        return status;
    if (sd::api::is_set(keys, "qwen_image_layers") && ctx->sd->version != VERSION_QWEN_IMAGE_LAYERED)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "qwen_image_layers", "requires Qwen Image Layered");
    const bool native_control = sd_version_is_control(ctx->sd->version) || ctx->sd->version == VERSION_FLEX_2 ||
                                (sd_version_is_z_image(ctx->sd->version) && ctx->sd->model_manager &&
                                 ctx->sd->model_manager->registered_params_size({ModelComponent::ControlNet}) > 0);
    if (params->control_image.data && !ctx->sd->control_net && !native_control)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "control_image", "model has no active control-image component");
    if (params->ip_adapter_image.data && !ctx->sd->ip_adapter)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "ip_adapter_image", "no IP-Adapter is loaded");
    if (params->pm_params.id_images_count > 0 && !ctx->sd->config_->photomaker_source_available)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "photomaker.id_images", "no PhotoMaker component is loaded");
    if (params->pulid_params.id_embedding_path && *params->pulid_params.id_embedding_path &&
        (ctx->sd->version != VERSION_FLUX || !ctx->sd->config_->params.pulid_weights_path ||
         !*ctx->sd->config_->params.pulid_weights_path))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "pulid.id_embedding_path", "requires a Flux model with PuLID weights");
    if (params->sample_params.guidance.img_cfg != INFINITY && params->sample_params.guidance.img_cfg != 1.f &&
        !sd_version_is_inpaint_or_unet_edit(ctx->sd->version) &&
        !(params->ref_images_count > 0 && sd_version_supports_ref_latent_img_cfg(ctx->sd->version)))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "sample.guidance.img_cfg", "model does not support image CFG for these inputs");
    sd::Image* images = nullptr;
    int count         = 0;
    std::vector<nlohmann::json> infos;
    try {
        if (!sd::pipeline::generate_image(ctx->sd, params, &images, &count, &infos)) {
            sd::free_images(images, count);
            return generation_failure(ctx->sd);
        }
    } catch (...) {
        sd::free_images(images, count);
        throw;
    }
    return sd::make_image_generation_result(images, count, infos, result_out) ? SD_STATUS_OK
                                                                              : sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate image result");
}

static sd_status_t run_video(sd_ctx_t* ctx, const sd::VideoConfig* params, sd::VideoGenerationResult** result_out, const sd_opt_t* keys = nullptr) {
    if (result_out)
        *result_out = nullptr;
    if (!result_out || !ctx || !ctx->sd || !params)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "context, params and result_out are required");
    StableDiffusionGGML::ExecutionScope execution(*ctx->sd);
    if (!execution.ready)
        return execution.operation.acquired
                   ? sd::api::error(SD_STATUS_LOAD_FAILED, nullptr, "could not refresh model sources")
                   : sd::api::error(SD_STATUS_BUSY, nullptr, "context is already executing");
    auto status = validate_generation(ctx->sd, params->cache, true);
    if (status != SD_STATUS_OK)
        return status;
    const bool vace = ctx->sd->diffusion_model &&
                      (ctx->sd->diffusion_model->get_desc() == "Wan2.1-VACE-1.3B" ||
                       ctx->sd->diffusion_model->get_desc() == "Wan2.x-VACE-14B");
    if (params->control_frames_size > 0 && !vace)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "control_frames", "requires a VACE model");
    if (params->ref_videos_count > 0 && !sd_version_is_minimax_h3(ctx->sd->version))
        return sd::api::error(SD_STATUS_UNSUPPORTED, "ref_videos", "requires a MiniMax-H3 model");
    if (params->ref_audios_count > 0 && !sd_version_is_minimax_h3(ctx->sd->version) && ctx->sd->version != VERSION_WAN2_2_S2V)
        return sd::api::error(SD_STATUS_UNSUPPORTED, "ref_audios", "requires MiniMax-H3 or Wan2.2 S2V");
    if (ctx->sd->version == VERSION_WAN2_2_S2V) {
        if (params->end_image.data)
            return sd::api::error(SD_STATUS_UNSUPPORTED, "end_image", "Wan2.2 S2V does not support an end image");
        if (params->ref_audios_count > 1 || (params->ref_audios_count && !ctx->sd->audio_encoder))
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "ref_audios", "Wan2.2 S2V requires one driving track and an audio encoder");
    }
    auto result = std::unique_ptr<sd::VideoGenerationResult, decltype(&sd::free_video_generation_result)>(
        static_cast<sd::VideoGenerationResult*>(calloc(1, sizeof(sd::VideoGenerationResult))), sd::free_video_generation_result);
    if (!result)
        return sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate video result");
    nlohmann::json info;
    if (!sd::pipeline::generate_video(ctx->sd, params, &result->frames, &result->frame_count,
                                      &result->audio, &result->fps, &info))
        return generation_failure(ctx->sd);
    result->generation_info = sd::copy_generation_info(info);
    if (!result->generation_info)
        return sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate video metadata");
    *result_out = result.release();
    return SD_STATUS_OK;
}

sd_status_t sd::generate_image(sd_ctx_t* ctx, const ImageConfig& config, ImageGenerationResult** result_out, const sd_opt_t* source) noexcept {
    if (result_out)
        *result_out = nullptr;
    return sd::api::call([&] { return run_image(ctx, &config, result_out, source); }, SD_STATUS_GENERATION_FAILED);
}

sd_status_t sd::generate_video(sd_ctx_t* ctx, const VideoConfig& config, VideoGenerationResult** result_out, const sd_opt_t* source) noexcept {
    if (result_out)
        *result_out = nullptr;
    return sd::api::call([&] { return run_video(ctx, &config, result_out, source); }, SD_STATUS_GENERATION_FAILED);
}

#include "kcpp_sd_extensions.h"
#include "model/diffusion/flux.hpp"

namespace kcpp_sd {

    sd_model_info sd_get_model_info(sd_ctx_t* ctx)
    {
        sd_model_info res = {};
        res.version = ctx->sd->version;
        res.vae_scale_factor = ctx->sd->get_vae_scale_factor();
        int diffusion_model_down_factor = ctx->sd->get_diffusion_model_down_factor();
        res.spatial_multiple = res.vae_scale_factor * diffusion_model_down_factor;
        return res;
    }

    void set_lora_cache(sd_ctx_t *ctx, bool enable) {
        ctx->sd->kcpp_lora_cache_populate = enable;
    }

    void apply_loras(sd_ctx_t *ctx, const std::vector<sd_lora_t>& lora_specs)
    {
        ctx->sd->apply_loras(lora_specs.data(), lora_specs.size());
    }

}

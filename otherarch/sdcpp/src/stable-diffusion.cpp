#include "stable-diffusion.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>

#include "core/ggml_extend.h"
#include "core/util.h"
#include "pipeline/diffusion_engine.h"
#include "pipeline/generation.h"
#include "pipeline/request.h"

#define NONE_STR "NONE"

static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
    float reuse_threshold = params.reuse_threshold;
    if (reuse_threshold == INFINITY) {
        if (params.mode == SD_CACHE_EASYCACHE) {
            reuse_threshold = 0.2f;
        } else if (params.mode == SD_CACHE_UCACHE) {
            reuse_threshold = 1.0f;
        }
    }
    return std::max(0.0f, reuse_threshold);
}

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
    "er_sde",
    "euler_cfg_pp",
    "euler_a_cfg_pp",
    "euler_ge",
    "dpm++2m_sde",
    "dpm++2m_sde_bt",
    "lms",
};

static_assert(SAMPLE_METHOD_COUNT == sizeof(sample_method_to_str) / sizeof(sample_method_to_str[0]),
              "\nnumber of elements in sample_method_to_str[] != SAMPLE_METHOD_COUNT");

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
    "ltx2",
    "logit_normal",
    "flux2",
    "flux",
    "beta",
};

static_assert(SCHEDULER_COUNT == sizeof(scheduler_to_str) / sizeof(scheduler_to_str[0]),
              "\nnumber of elements in scheduler_to_str[] != SCHEDULER_COUNT");

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    if (!strcmp(str, "normal")) {
        return DISCRETE_SCHEDULER;
    }
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "sefi_flow",
    "minit2i_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return NONE_STR;
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

const char* hires_upscaler_to_str[] = {
    "None",
    "Latent",
    "Latent (nearest)",
    "Latent (nearest-exact)",
    "Latent (antialiased)",
    "Latent (bicubic)",
    "Latent (bicubic antialiased)",
    "Lanczos",
    "Nearest",
    "Model",
};

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler) {
    if (upscaler >= SD_HIRES_UPSCALER_NONE && upscaler < SD_HIRES_UPSCALER_COUNT) {
        return hires_upscaler_to_str[upscaler];
    }
    return NONE_STR;
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

const char* sd_vae_format_name(enum sd_vae_format_t format) {
    switch (format) {
        case SD_VAE_FORMAT_AUTO:
            return "auto";
        case SD_VAE_FORMAT_FLUX:
            return "flux";
        case SD_VAE_FORMAT_SD3:
            return "sd3";
        case SD_VAE_FORMAT_FLUX2:
            return "flux2";
        case SD_VAE_FORMAT_WAN:
            return "wan";
        default:
            return NONE_STR;
    }
}

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = INFINITY;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
    cache_params->spectrum_w                  = 0.40f;
    cache_params->spectrum_m                  = 3;
    cache_params->spectrum_lam                = 1.0f;
    cache_params->spectrum_window_size        = 2;
    cache_params->spectrum_flex_window        = 0.50f;
    cache_params->spectrum_warmup_steps       = 4;
    cache_params->spectrum_stop_percent       = 0.9f;
}

void sd_hires_params_init(sd_hires_params_t* hires_params) {
    *hires_params                     = {};
    hires_params->enabled             = false;
    hires_params->upscaler            = SD_HIRES_UPSCALER_LATENT;
    hires_params->model_path          = nullptr;
    hires_params->scale               = 2.0f;
    hires_params->target_width        = 0;
    hires_params->target_height       = 0;
    hires_params->steps               = 0;
    hires_params->denoising_strength  = 0.7f;
    hires_params->upscale_tile_size   = 128;
    hires_params->custom_sigmas       = nullptr;
    hires_params->custom_sigmas_count = 0;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                           = {};
    sd_ctx_params->n_threads                 = sd_get_num_physical_cores();
    sd_ctx_params->wtype                     = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                  = CUDA_RNG;
    sd_ctx_params->sampler_rng_type          = RNG_TYPE_COUNT;
    sd_ctx_params->prediction                = PREDICTION_COUNT;
    sd_ctx_params->lora_apply_mode           = LORA_APPLY_AUTO;
    sd_ctx_params->max_vram                  = nullptr;
    sd_ctx_params->disable_prefetch          = false;
    sd_ctx_params->disable_segmented_compute = false;
    sd_ctx_params->eager_load                = false;
    sd_ctx_params->enable_mmap               = false;
    sd_ctx_params->diffusion_flash_attn      = false;
    sd_ctx_params->vae_format                = SD_VAE_FORMAT_AUTO;
    sd_ctx_params->backend                   = nullptr;
    sd_ctx_params->params_backend            = nullptr;
    sd_ctx_params->split_mode                = nullptr;
    sd_ctx_params->auto_fit                  = true;
    sd_ctx_params->rpc_servers               = nullptr;
    sd_ctx_params->model_args                = nullptr;
    sd_ctx_params->pulid_weights_path        = nullptr;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(8192);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 8192 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "uncond_diffusion_model_path: %s\n"
             "embeddings_connectors_path: %s\n"
             "vae_path: %s\n"
             "audio_vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "pulid_weights_path: %s\n"
             "tensor_type_rules: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "max_vram: %s\n"
             "disable_prefetch: %s\n"
             "disable_segmented_compute: %s\n"
             "eager_load: %s\n"
             "backend: %s\n"
             "params_backend: %s\n"
             "split_mode: %s\n"
             "model_args: %s\n"
             "auto_fit: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "vae_format: %s\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->uncond_diffusion_model_path),
             SAFE_STR(sd_ctx_params->embeddings_connectors_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->audio_vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->pulid_weights_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             SAFE_STR(sd_ctx_params->max_vram),
             BOOL_STR(sd_ctx_params->disable_prefetch),
             BOOL_STR(sd_ctx_params->disable_segmented_compute),
             BOOL_STR(sd_ctx_params->eager_load),
             SAFE_STR(sd_ctx_params->backend),
             SAFE_STR(sd_ctx_params->params_backend),
             SAFE_STR(sd_ctx_params->split_mode),
             SAFE_STR(sd_ctx_params->model_args),
             BOOL_STR(sd_ctx_params->auto_fit),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             sd_vae_format_name(sd_ctx_params->vae_format));

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->eta                         = INFINITY;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
    sample_params->flow_shift                  = INFINITY;
    sample_params->extra_sample_args           = nullptr;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d, "
             "flow_shift: %.2f, "
             "extra_sample_args: %s)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep,
             sample_params->flow_shift,
             SAFE_STR(sample_params->extra_sample_args));

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip           = -1;
    sd_img_gen_params->ref_images_count    = 0;
    sd_img_gen_params->ref_image_args      = "";
    sd_img_gen_params->width               = 512;
    sd_img_gen_params->height              = 512;
    sd_img_gen_params->strength            = 0.75f;
    sd_img_gen_params->seed                = -1;
    sd_img_gen_params->batch_count         = 1;
    sd_img_gen_params->control_strength    = 0.9f;
    sd_img_gen_params->ip_adapter_strength = 1.0f;
    sd_img_gen_params->qwen_image_layers   = 3;
    sd_img_gen_params->circular_x          = false;
    sd_img_gen_params->circular_y          = false;
    sd_img_gen_params->pm_params           = {nullptr, 0, nullptr, 20.f};
    sd_img_gen_params->pulid_params        = {nullptr, 1.0f};
    sd_img_gen_params->vae_tiling_params   = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_cache_params_init(&sd_img_gen_params->cache);
    sd_hires_params_init(&sd_img_gen_params->hires);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "qwen_image_layers: %d\n"
             "ref_images_count: %d\n"
             "ref_image_args: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s (temporal=%s, extra_tiling_args=%s)\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "hires: {enabled=%s, upscaler=%s, model_path=%s, scale=%.2f, target=%dx%d, steps=%d, denoising_strength=%.2f}\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->qwen_image_layers,
             sd_img_gen_params->ref_images_count,
             SAFE_STR(sd_img_gen_params->ref_image_args),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.temporal_tiling),
             SAFE_STR(sd_img_gen_params->vae_tiling_params.extra_tiling_args),
             BOOL_STR(sd_img_gen_params->circular_x),
             BOOL_STR(sd_img_gen_params->circular_y),
             BOOL_STR(sd_img_gen_params->hires.enabled),
             sd_hires_upscaler_name(sd_img_gen_params->hires.upscaler),
             SAFE_STR(sd_img_gen_params->hires.model_path),
             sd_img_gen_params->hires.scale,
             sd_img_gen_params->hires.target_width,
             sd_img_gen_params->hires.target_height,
             sd_img_gen_params->hires.steps,
             sd_img_gen_params->hires.denoising_strength);
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    } else if (sd_img_gen_params->cache.mode == SD_CACHE_UCACHE) {
        cache_mode_str = "ucache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             get_cache_reuse_threshold(sd_img_gen_params->cache),
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}

void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params) {
    *sd_vid_gen_params = {};
    sd_sample_params_init(&sd_vid_gen_params->sample_params);
    sd_sample_params_init(&sd_vid_gen_params->high_noise_sample_params);
    sd_vid_gen_params->high_noise_sample_params.sample_steps = -1;
    sd_vid_gen_params->width                                 = 512;
    sd_vid_gen_params->height                                = 512;
    sd_vid_gen_params->strength                              = 0.75f;
    sd_vid_gen_params->seed                                  = -1;
    sd_vid_gen_params->video_frames                          = 6;
    sd_vid_gen_params->fps                                   = 16;
    sd_vid_gen_params->input_audio                           = nullptr;
    sd_vid_gen_params->moe_boundary                          = 0.875f;
    sd_vid_gen_params->vace_strength                         = 1.f;
    sd_vid_gen_params->vae_tiling_params                     = {false, false, 0, 0, 0.5f, 0.0f, 0.0f, nullptr};
    sd_vid_gen_params->hires.enabled                         = false;
    sd_vid_gen_params->hires.upscaler                        = SD_HIRES_UPSCALER_LATENT;
    sd_vid_gen_params->hires.scale                           = 2.f;
    sd_vid_gen_params->hires.target_width                    = 0;
    sd_vid_gen_params->hires.target_height                   = 0;
    sd_vid_gen_params->hires.steps                           = 0;
    sd_vid_gen_params->hires.denoising_strength              = 0.7f;
    sd_vid_gen_params->hires.upscale_tile_size               = 128;
    sd_vid_gen_params->hires.custom_sigmas                   = nullptr;
    sd_vid_gen_params->hires.custom_sigmas_count             = 0;
    sd_vid_gen_params->circular_x                            = false;
    sd_vid_gen_params->circular_y                            = false;
    sd_cache_params_init(&sd_vid_gen_params->cache);
}

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

static bool sd_version_supports_video_generation(SDVersion version) {
    return version == VERSION_SVD || sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_lingbot_video(version) || sd_version_is_ltxav(version) || sd_version_is_minimax_h3(version);
}

static bool sd_version_supports_image_generation(SDVersion version) {
    return !sd_version_supports_video_generation(version);
}

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    if (sd_ctx && sd_ctx->sd) {
        if (mode < SD_CANCEL_ALL || mode > SD_CANCEL_RESET) {
            mode = SD_CANCEL_ALL;
        }
        sd_ctx->sd->set_cancel_flag(mode);
    }
}

void free_sd_audio(sd_audio_t* audio) {
    if (audio == nullptr) {
        return;
    }
    free(audio->data);
    audio->data = nullptr;
    free(audio);
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (sd_ctx->sd->config_->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        return true;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || path == nullptr) {
        return false;
    }
    return sd_ctx->sd->load_control_net_from_file(path);
}

SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->unload_control_net();
}

SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->control_net != nullptr;
}

const char* sd_get_model_version_name(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->version >= VERSION_COUNT) {
        return "Unknown";
    }
    return model_version_to_str[sd_ctx->sd->version];
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    return sd::pipeline::default_sample_method(sd_ctx != nullptr ? sd_ctx->sd : nullptr);
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    return sd::pipeline::default_scheduler(sd_ctx != nullptr ? sd_ctx->sd : nullptr, sample_method);
}

SD_API bool generate_image(sd_ctx_t* sd_ctx,
                           const sd_img_gen_params_t* params,
                           sd_image_t** images_out,
                           int* num_images_out) {
    if (images_out != nullptr)
        *images_out = nullptr;
    if (num_images_out != nullptr)
        *num_images_out = 0;
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || params == nullptr) {
        return false;
    }
    StableDiffusionGGML::ExecutionScope execution(*sd_ctx->sd);
    return execution.ready && sd::pipeline::generate_image(sd_ctx->sd, params, images_out, num_images_out);
}

SD_API bool generate_video(sd_ctx_t* sd_ctx,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           sd_image_t** frames_out,
                           int* num_frames_out,
                           sd_audio_t** audio_out) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_vid_gen_params == nullptr) {
        return false;
    }

    if (frames_out != nullptr) {
        *frames_out = nullptr;
    }
    if (audio_out != nullptr) {
        *audio_out = nullptr;
    }
    if (num_frames_out != nullptr) {
        *num_frames_out = 0;
    }

    StableDiffusionGGML::ExecutionScope execution(*sd_ctx->sd);
    if (!execution.ready) {
        return false;
    }

    return sd::pipeline::generate_video(sd_ctx->sd, sd_vid_gen_params, frames_out, num_frames_out, audio_out);
}

SD_API void free_sd_images(sd_image_t* result_images, int num_images) {
    if (result_images == nullptr) {
        return;
    }

    for (int i = 0; i < num_images; ++i) {
        if (result_images[i].data != nullptr) {
            free(result_images[i].data);
            result_images[i].data = nullptr;
        }
    }

    free(result_images);
}

#include "kcpp_sd_extensions.h"
#include "model/diffusion/flux.hpp"

namespace kcpp_sd {

    int get_loaded_sd_version(sd_ctx_t* ctx) {
        return ctx->sd->version;
    }

    bool loaded_model_is_chroma(sd_ctx_t* ctx) {
        if (ctx != nullptr && ctx->sd != nullptr) {
            auto maybe_flux = std::dynamic_pointer_cast<Flux::FluxRunner>(ctx->sd->diffusion_model);
            if (maybe_flux != nullptr) {
                return maybe_flux->config.is_chroma;
            }
        }
        return false;
    }

    int get_spatial_multiple(sd_ctx_t* ctx) {
        return ctx->sd->get_vae_scale_factor() * ctx->sd->get_diffusion_model_down_factor();
    }

    model_info get_model_info(sd_ctx_t* ctx)
    {
        model_info res = {};
        auto loadedsdver = get_loaded_sd_version(ctx);
        res.is_wan = (loadedsdver == SDVersion::VERSION_WAN2 || loadedsdver == SDVersion::VERSION_WAN2_2_I2V || loadedsdver == SDVersion::VERSION_WAN2_2_TI2V);
        res.is_qwenimg = (loadedsdver == SDVersion::VERSION_QWEN_IMAGE);
        res.is_chroma = loaded_model_is_chroma(ctx);
        res.is_kontext = (loadedsdver==SDVersion::VERSION_FLUX && !res.is_chroma);
        res.is_flux2 = (loadedsdver == SDVersion::VERSION_FLUX2 || loadedsdver == SDVersion::VERSION_FLUX2_KLEIN);
        res.is_flux1 = (loadedsdver == SDVersion::VERSION_FLUX);
        res.is_zimage = (loadedsdver == SDVersion::VERSION_Z_IMAGE);
        res.is_sdxs = (loadedsdver == SDVersion::VERSION_SDXS_512_DS || loadedsdver == SDVersion::VERSION_SDXS_09);
        res.is_sd1 = (loadedsdver == SDVersion::VERSION_SD1);
        res.is_sd2 = (loadedsdver == SDVersion::VERSION_SD2);
        res.is_sdxl = sd_version_is_sdxl((SDVersion)loadedsdver);
        res.is_ltx = sd_version_is_ltxav((SDVersion)loadedsdver);
        res.is_minimaxh3 = sd_version_is_minimax_h3((SDVersion)loadedsdver);
        res.is_boogu = sd_version_is_boogu_image((SDVersion)loadedsdver);
        res.supports_ref_image = sd_version_supports_ref_latent_img_cfg((SDVersion)loadedsdver);
        res.vae_scale_factor = ctx->sd->get_vae_scale_factor();
        res.spatial_multiple = get_spatial_multiple(ctx);
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

#include "request.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "diffusion_engine.h"
#include "runtime/denoiser.hpp"

namespace sd::pipeline {

    const char* sampling_methods_str[] = {
        "Euler",
        "Euler A",
        "Heun",
        "DPM2",
        "DPM++ (2s)",
        "DPM++ (2M)",
        "modified DPM++ (2M)",
        "iPNDM",
        "iPNDM_v",
        "LCM",
        "DDIM \"trailing\"",
        "TCD",
        "Res Multistep",
        "Res 2s",
        "ER-SDE",
        "Euler CFG++",
        "Euler A CFG++",
        "Euler GE",
        "DPM++ (2M) SDE",
        "DPM++ (2M) SDE BT",
        "LMS",
    };

    static_assert(SAMPLE_METHOD_COUNT == sizeof(sampling_methods_str) / sizeof(sampling_methods_str[0]),
                  "\nnumber of elements in sampling_methods_str[] != SAMPLE_METHOD_COUNT");

    static bool sd_version_supports_img_cfg(SDVersion version, bool has_ref_images) {
        return sd_version_is_inpaint_or_unet_edit(version) ||
               (has_ref_images && sd_version_supports_ref_latent_img_cfg(version));
    }

    enum sample_method_t default_sample_method(const StableDiffusionGGML* sd) {
        if (sd != nullptr) {
            if (sd_version_is_pid(sd->version)) {
                return LCM_SAMPLE_METHOD;
            }
            if (sd_version_is_dit(sd->version)) {
                return EULER_SAMPLE_METHOD;
            }
        }
        return EULER_A_SAMPLE_METHOD;
    }

    enum scheduler_t default_scheduler(const StableDiffusionGGML* sd, enum sample_method_t sample_method) {
        if (sd != nullptr) {
            auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd->denoiser);
            if (edm_v_denoiser) {
                return EXPONENTIAL_SCHEDULER;
            }
        }
        if (sample_method == LCM_SAMPLE_METHOD || sample_method == TCD_SAMPLE_METHOD) {
            return LCM_SCHEDULER;
        } else if (sample_method == DDIM_TRAILING_SAMPLE_METHOD) {
            return SIMPLE_SCHEDULER;
        } else if (sd != nullptr && sd_version_is_flux(sd->version)) {
            return FLUX_SCHEDULER;
        } else if (sd != nullptr && sd_version_is_flux2(sd->version)) {
            return FLUX2_SCHEDULER;
        } else if (sd != nullptr && sd_version_is_ltxav(sd->version)) {
            return LTX2_SCHEDULER;
        } else if (sd != nullptr && sd_version_is_ideogram4(sd->version)) {
            return LOGIT_NORMAL_SCHEDULER;
        }
        return DISCRETE_SCHEDULER;
    }

    static int64_t resolve_seed(int64_t seed) {
        if (seed >= 0) {
            return seed;
        }
        srand((int)time(nullptr));
        return rand();
    }

    static enum sample_method_t resolve_sample_method(StableDiffusionGGML* sd, enum sample_method_t sample_method) {
        if (sample_method == SAMPLE_METHOD_COUNT) {
            return default_sample_method(sd);
        }
        return sample_method;
    }

    static scheduler_t resolve_scheduler(StableDiffusionGGML* sd,
                                         scheduler_t scheduler,
                                         enum sample_method_t sample_method) {
        if (scheduler == SCHEDULER_COUNT) {
            return default_scheduler(sd, sample_method);
        }
        return scheduler;
    }

    float resolve_eta(StableDiffusionGGML* sd,
                      float eta,
                      enum sample_method_t sample_method) {
        if (eta == INFINITY) {
            if (sd->version == VERSION_HIDREAM_O1) {
                return 8.f;
            }
            switch (sample_method) {
                case DDIM_TRAILING_SAMPLE_METHOD:
                case TCD_SAMPLE_METHOD:
                case RES_MULTISTEP_SAMPLE_METHOD:
                case RES_2S_SAMPLE_METHOD:
                    return 0.0f;
                case EULER_A_SAMPLE_METHOD:
                case DPMPP2S_A_SAMPLE_METHOD:
                case ER_SDE_SAMPLE_METHOD:
                case EULER_A_CFG_PP_SAMPLE_METHOD:
                case DPMPP2M_SDE_SAMPLE_METHOD:
                case DPMPP2M_SDE_BT_SAMPLE_METHOD:
                    return 1.0f;
                default:;
            }
            return 0.0f;
        }
        return eta;
    }

    GenerationRequest::GenerationRequest(StableDiffusionGGML* sd, const sd_img_gen_params_t* sd_img_gen_params) {
        prompt                      = SAFE_STR(sd_img_gen_params->prompt);
        negative_prompt             = SAFE_STR(sd_img_gen_params->negative_prompt);
        width                       = sd_img_gen_params->width;
        height                      = sd_img_gen_params->height;
        vae_scale_factor            = sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd->get_diffusion_model_down_factor();
        seed                        = sd_img_gen_params->seed;
        batch_count                 = sd_img_gen_params->batch_count;
        qwen_image_layers           = std::max(0, sd_img_gen_params->qwen_image_layers);
        clip_skip                   = sd_img_gen_params->clip_skip;
        shifted_timestep            = sd_img_gen_params->sample_params.shifted_timestep;
        strength                    = sd_img_gen_params->strength;
        control_strength            = sd_img_gen_params->control_strength;
        eta                         = sd_img_gen_params->sample_params.eta;
        has_ref_images              = sd_img_gen_params->ref_images_count > 0;
        guidance                    = sd_img_gen_params->sample_params.guidance;
        pm_params                   = sd_img_gen_params->pm_params;
        pulid_params                = sd_img_gen_params->pulid_params;
        hires                       = sd_img_gen_params->hires;
        cache_params                = &sd_img_gen_params->cache;
        resolve(sd);
    }

    GenerationRequest::GenerationRequest(StableDiffusionGGML* sd, const sd_vid_gen_params_t* sd_vid_gen_params) {
        prompt           = SAFE_STR(sd_vid_gen_params->prompt);
        negative_prompt  = SAFE_STR(sd_vid_gen_params->negative_prompt);
        width            = sd_vid_gen_params->width;
        height           = sd_vid_gen_params->height;
        requested_frames = std::max(1, sd_vid_gen_params->video_frames);
        frames           = sd->align_video_frames(requested_frames);
        clip_skip        = sd_vid_gen_params->clip_skip;
        fps              = std::max(1, sd_vid_gen_params->fps);
        if (sd_version_is_minimax_h3(sd->version) && fps != 24) {
            LOG_WARN("MiniMax-H3 uses 24 fps; overriding requested fps %d", fps);
            fps = 24;
        }
        vae_scale_factor            = sd->get_vae_scale_factor();
        diffusion_model_down_factor = sd->get_diffusion_model_down_factor();
        seed                        = sd_vid_gen_params->seed;
        strength                    = sd_vid_gen_params->strength;
        cache_params                = &sd_vid_gen_params->cache;
        vace_strength               = sd_vid_gen_params->vace_strength;
        guidance                    = sd_vid_gen_params->sample_params.guidance;
        high_noise_guidance         = sd_vid_gen_params->high_noise_sample_params.guidance;
        hires                       = sd_vid_gen_params->hires;
        resolve(sd);
        if (frames != requested_frames) {
            LOG_WARN("align video frames from %d to %d for %s",
                     requested_frames,
                     frames,
                     model_version_to_str[sd->version]);
        }
    }

    void GenerationRequest::align_generation_request_size() {
        align_image_size(&width, &height, "generation request");
    }

    void GenerationRequest::align_image_size(int* target_width, int* target_height, const char* label) {
        int spatial_multiple = vae_scale_factor * diffusion_model_down_factor;
        int width_offset     = align_up_offset(*target_width, spatial_multiple);
        int height_offset    = align_up_offset(*target_height, spatial_multiple);
        if (width_offset <= 0 && height_offset <= 0) {
            return;
        }

        int original_width  = *target_width;
        int original_height = *target_height;

        *target_width += width_offset;
        *target_height += height_offset;
        LOG_WARN("align %s up %dx%d to %dx%d (multiple=%d)",
                 label,
                 original_width,
                 original_height,
                 *target_width,
                 *target_height,
                 spatial_multiple);
    }

    void GenerationRequest::resolve_hires() {
        if (!hires.enabled) {
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_NONE) {
            hires.enabled = false;
            return;
        }
        if (hires.upscaler < SD_HIRES_UPSCALER_NONE || hires.upscaler >= SD_HIRES_UPSCALER_COUNT) {
            LOG_WARN("hires upscaler '%d' is invalid, disabling hires", hires.upscaler);
            hires.enabled = false;
            return;
        }
        if (hires.upscaler == SD_HIRES_UPSCALER_MODEL && strlen(SAFE_STR(hires.model_path)) == 0) {
            LOG_WARN("hires model upscaler requires a model path, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.scale <= 0.f && hires.target_width <= 0 && hires.target_height <= 0) {
            LOG_WARN("hires scale must be positive when no target size is set, disabling hires");
            hires.enabled = false;
            return;
        }
        if (hires.custom_sigmas_count < 0) {
            LOG_WARN("hires custom sigmas count is negative, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count > 0 && hires.custom_sigmas == nullptr) {
            LOG_WARN("hires custom sigmas count is positive but custom sigmas are null, ignoring custom sigmas");
            hires.custom_sigmas_count = 0;
        }
        if (hires.custom_sigmas_count == 1) {
            LOG_WARN("hires custom sigmas requires at least two values, ignoring custom sigmas");
            hires.custom_sigmas       = nullptr;
            hires.custom_sigmas_count = 0;
        }
        hires.denoising_strength = std::clamp(hires.denoising_strength, 0.0001f, 1.f);
        hires.steps              = std::max(0, hires.steps);

        if (hires.target_width > 0 && hires.target_height > 0) {
            // pass
        } else if (hires.target_width > 0) {
            hires.target_height = hires.target_width;
        } else if (hires.target_height > 0) {
            hires.target_width = hires.target_height;
        } else {
            hires.target_width  = static_cast<int>(std::round(width * hires.scale));
            hires.target_height = static_cast<int>(std::round(height * hires.scale));
        }

        if (hires.target_width <= 0 || hires.target_height <= 0) {
            LOG_WARN("hires target size is not positive, disabling hires");
            hires.enabled = false;
            return;
        }
        align_image_size(&hires.target_width, &hires.target_height, "hires target");
    }

    void GenerationRequest::resolve_guidance(StableDiffusionGGML* sd,
                                             sd_guidance_params_t* guidance,
                                             bool* use_uncond,
                                             bool* use_img_uncond,
                                             bool has_ref_images,
                                             const char* stage_name) {
        GGML_ASSERT(guidance != nullptr);
        GGML_ASSERT(use_uncond != nullptr);
        GGML_ASSERT(use_img_uncond != nullptr);
        // out_img_uncond + text_cfg_scale * (out_cond - out_uncond) + image_cfg_scale * (out_uncond - out_img_uncond)
        // -> text_cfg_scale * out_cond + (image_cfg_scale - text_cfg_scale) * out_uncond + (1 - image_cfg_scale) * out_img_uncond
        // out_cond       : prompt, image latent
        // out_uncond     : negative prompt, image latent
        // out_img_uncond : negative prompt, zero image latent
        // image_cfg_scale == 1 reduces 3-cond CFG to 2-cond CFG.
        bool img_cfg_was_set = std::isfinite(guidance->img_cfg);
        if (!img_cfg_was_set) {
            guidance->img_cfg = 1.f;
        }

        if (!sd_version_supports_img_cfg(sd->version, has_ref_images)) {
            if (img_cfg_was_set && guidance->img_cfg != 1.f) {
                LOG_WARN("3-conditioning CFG is not supported with this model, disabling it for better performance");
            }
            guidance->img_cfg = 1.f;
        }

        if (guidance->img_cfg != guidance->txt_cfg) {
            *use_uncond = true;
        }

        if (guidance->img_cfg != 1.f) {
            *use_img_uncond = true;
        }

        if (guidance->txt_cfg < 1.f) {
            const char* prefix = stage_name == nullptr ? "" : stage_name;
            if (guidance->txt_cfg == 0.f) {
                LOG_WARN("%sunconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)",
                         prefix);
            } else {
                LOG_WARN("%scfg value out of expected range may produce unexpected results", prefix);
            }
        }
    }

    void GenerationRequest::resolve(StableDiffusionGGML* sd) {
        align_generation_request_size();
        resolve_hires();
        seed = resolve_seed(seed);

        resolve_guidance(sd, &guidance, &use_uncond, &use_img_uncond, has_ref_images);
        if (sd->high_noise_diffusion_model) {
            resolve_guidance(sd,
                             &high_noise_guidance,
                             &use_high_noise_uncond,
                             &use_high_noise_img_uncond,
                             has_ref_images,
                             "high noise: ");
        }

        if (shifted_timestep > 0 && !sd_version_is_sdxl(sd->version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
    }

    SamplePlan::SamplePlan(StableDiffusionGGML* sd,
                           const sd_img_gen_params_t* sd_img_gen_params,
                           const GenerationRequest& request) {
        sample_method     = sd_img_gen_params->sample_params.sample_method;
        extra_sample_args = sd_img_gen_params->sample_params.extra_sample_args;
        eta               = sd_img_gen_params->sample_params.eta;
        sample_steps      = sd_img_gen_params->sample_params.sample_steps;
        resolve(sd, &request, &sd_img_gen_params->sample_params);
    }

    SamplePlan::SamplePlan(StableDiffusionGGML* sd,
                           const sd_vid_gen_params_t* sd_vid_gen_params,
                           const GenerationRequest& request) {
        sample_method     = sd_vid_gen_params->sample_params.sample_method;
        extra_sample_args = sd_vid_gen_params->sample_params.extra_sample_args;
        eta               = sd_vid_gen_params->sample_params.eta;
        sample_steps      = sd_vid_gen_params->sample_params.sample_steps;
        if (sd->high_noise_diffusion_model) {
            high_noise_sample_steps      = sd_vid_gen_params->high_noise_sample_params.sample_steps;
            high_noise_sample_method     = sd_vid_gen_params->high_noise_sample_params.sample_method;
            high_noise_extra_sample_args = sd_vid_gen_params->high_noise_sample_params.extra_sample_args;
            high_noise_eta               = sd_vid_gen_params->high_noise_sample_params.eta;
        }
        moe_boundary = sd_vid_gen_params->moe_boundary;
        resolve(sd, &request, &sd_vid_gen_params->sample_params);
    }

    void SamplePlan::resolve(StableDiffusionGGML* sd,
                             const GenerationRequest* request,
                             const sd_sample_params_t* sample_params) {
        sample_method = resolve_sample_method(sd, sample_method);

        total_steps = sample_steps + std::max(0, high_noise_sample_steps);

        if (sample_params->custom_sigmas_count > 0) {
            sigmas      = std::vector<float>(sample_params->custom_sigmas,
                                        sample_params->custom_sigmas + sample_params->custom_sigmas_count);
            total_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("total_steps != custom_sigmas_count - 1, set total_steps to %d", total_steps);
            if (sample_steps >= total_steps) {
                sample_steps = total_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
            }
            if (high_noise_sample_steps > 0) {
                high_noise_sample_steps = total_steps - sample_steps;
                LOG_WARN("total_steps != custom_sigmas_count - 1, set high_noise_sample_steps to %d", high_noise_sample_steps);
            }
        } else {
            scheduler_t scheduler = resolve_scheduler(sd,
                                                      sample_params->scheduler,
                                                      sample_method);
            int sample_seq_len    = sd->get_image_seq_len(request->height, request->width);
            if (sd_version_is_ltxav(sd->version) && request->frames > 0) {
                int latent_frames = ((request->frames - 1) / 8) + 1;
                sample_seq_len *= latent_frames;
            } else if (sd_version_is_minimax_h3(sd->version) && request->frames > 0) {
                sample_seq_len *= sd->video_frames_to_latent_frames(request->frames);
            }
            sigmas = sd->denoiser->get_sigmas(total_steps,
                                              sample_seq_len,
                                              scheduler,
                                              sd->version,
                                              sample_params->extra_sample_args);
        }

        eta = resolve_eta(sd, eta, sample_method);

        if (high_noise_sample_steps < 0) {
            for (size_t i = 0; i < sigmas.size(); ++i) {
                if (sigmas[i] < moe_boundary) {
                    high_noise_sample_steps = static_cast<int>(i);
                    break;
                }
            }
            LOG_VERBOSE("switching from high noise model at step %d", high_noise_sample_steps);
        }

        LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
        if (high_noise_sample_steps > 0) {
            high_noise_sample_method = resolve_sample_method(sd,
                                                             high_noise_sample_method);
            high_noise_eta           = resolve_eta(sd, high_noise_eta, high_noise_sample_method);
            LOG_INFO("sampling(high noise) using %s method", sampling_methods_str[high_noise_sample_method]);
        }
    }

    std::vector<float> make_hires_sigma_schedule(StableDiffusionGGML* sd,
                                                 const sd_hires_params_t& hires,
                                                 const sd_sample_params_t& sample_params,
                                                 sample_method_t sample_method,
                                                 int default_steps,
                                                 int sample_seq_len,
                                                 int* scheduler_steps_out) {
        if (scheduler_steps_out != nullptr) {
            *scheduler_steps_out = 0;
        }

        if (hires.custom_sigmas_count > 0 && hires.custom_sigmas != nullptr) {
            std::vector<float> custom_sigmas(hires.custom_sigmas,
                                             hires.custom_sigmas + hires.custom_sigmas_count);
            if (scheduler_steps_out != nullptr) {
                *scheduler_steps_out = static_cast<int>(custom_sigmas.size()) - 1;
            }
            return custom_sigmas;
        }

        int effective_steps = hires.steps > 0 ? hires.steps : default_steps;
        effective_steps     = std::max(1, effective_steps);

        // sd-webui behavior: scale up total steps so trimming by denoising_strength yields exactly hires_steps effective steps,
        // unlike img2img which trims from a fixed step count.
        int scheduler_steps = static_cast<int>(effective_steps / hires.denoising_strength);
        scheduler_steps     = std::max(1, scheduler_steps);

        scheduler_t scheduler     = resolve_scheduler(sd,
                                                      sample_params.scheduler,
                                                      sample_method);
        std::vector<float> sigmas = sd->denoiser->get_sigmas(scheduler_steps,
                                                             sample_seq_len,
                                                             scheduler,
                                                             sd->version,
                                                             sample_params.extra_sample_args);
        size_t t_enc              = static_cast<size_t>(scheduler_steps * hires.denoising_strength);
        if (t_enc >= static_cast<size_t>(scheduler_steps)) {
            t_enc = static_cast<size_t>(scheduler_steps) - 1;
        }
        if (scheduler_steps_out != nullptr) {
            *scheduler_steps_out = scheduler_steps;
        }
        return std::vector<float>(sigmas.begin() + scheduler_steps - static_cast<int>(t_enc) - 1,
                                  sigmas.end());
    }

}  // namespace sd::pipeline

#include "generation_info.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "pipeline/request.h"
#include "runtime/guidance.h"

namespace sd {
    using nlohmann::json;

    static const char* text(const char* value) {
        return value ? value : "";
    }

    static json sampling_info(const sd_sample_params_t& p, bool high_noise = false) {
        const auto& g = p.guidance;
        json info     = {{"sample_steps", p.sample_steps}, {"guidance", {{"txt_cfg", g.txt_cfg}, {"distilled_guidance", g.distilled_guidance}}}};
        if (!high_noise && p.shifted_timestep > 0)
            info["shifted_timestep"] = p.shifted_timestep;
        if (std::isfinite(g.img_cfg) && g.img_cfg != 1.f)
            info["guidance"]["img_cfg"] = g.img_cfg;
        if (p.sample_method != SAMPLE_METHOD_COUNT)
            info["sample_method"] = sd_sample_method_name(p.sample_method);
        if (!high_noise && p.scheduler != SCHEDULER_COUNT)
            info["scheduler"] = sd_scheduler_name(p.scheduler);
        if (std::isfinite(p.eta))
            info["eta"] = p.eta;
        if (!high_noise && std::isfinite(p.flow_shift))
            info["flow_shift"] = p.flow_shift;
        if (!high_noise && p.custom_sigmas && p.custom_sigmas_count > 0)
            info["custom_sigmas"] = std::vector<float>(p.custom_sigmas, p.custom_sigmas + p.custom_sigmas_count);
        if (p.extra_sample_args && *p.extra_sample_args)
            info["extra_sample_args"] = p.extra_sample_args;
        if (g.slg.layers && g.slg.layer_count > 0 &&
            (g.slg.scale != 0.f || guidance::parse_skip_layer_guidance_uncond_arg(p.extra_sample_args)))
            info["guidance"]["slg"] = {{"scale", g.slg.scale}, {"layer_start", g.slg.layer_start}, {"layer_end", g.slg.layer_end}, {"layers", std::vector<int>(g.slg.layers, g.slg.layers + g.slg.layer_count)}};
        return info;
    }

    static json hires_info(const sd_hires_params_t& p) {
        json info = {{"enabled", p.enabled}};
        if (!p.enabled)
            return info;
        info.update({{"upscaler", sd_hires_upscaler_name(p.upscaler)}, {"scale", p.scale}, {"target_width", p.target_width}, {"target_height", p.target_height}, {"steps", p.steps}, {"denoising_strength", p.denoising_strength}});
        if (p.upscaler == SD_HIRES_UPSCALER_MODEL) {
            info["model_path"]        = text(p.model_path);
            info["upscale_tile_size"] = p.upscale_tile_size;
        }
        if (p.custom_sigmas && p.custom_sigmas_count > 0)
            info["custom_sigmas"] = std::vector<float>(p.custom_sigmas, p.custom_sigmas + p.custom_sigmas_count);
        return info;
    }

    static json cache_info(const sd_cache_params_t& p) {
        static const char* modes[] = {"disabled", "easycache", "ucache", "dbcache", "taylorseer", "cache_dit", "spectrum"};
        if (p.mode < SD_CACHE_DISABLED || p.mode > SD_CACHE_SPECTRUM)
            return {{"mode", static_cast<int>(p.mode)}};
        json info = {{"mode", modes[p.mode]}};
        if (p.mode == SD_CACHE_DISABLED)
            return info;
        if (p.mode == SD_CACHE_EASYCACHE || p.mode == SD_CACHE_UCACHE) {
            if (std::isfinite(p.reuse_threshold))
                info["reuse_threshold"] = p.reuse_threshold;
            info.update({{"start_percent", p.start_percent}, {"end_percent", p.end_percent}});
            if (p.mode == SD_CACHE_UCACHE)
                info.update({{"error_decay_rate", p.error_decay_rate}, {"use_relative_threshold", p.use_relative_threshold}, {"reset_error_on_compute", p.reset_error_on_compute}});
        } else if (p.mode == SD_CACHE_DBCACHE || p.mode == SD_CACHE_TAYLORSEER || p.mode == SD_CACHE_CACHE_DIT) {
            // TaylorSeer also uses the shared DBCache thresholds and limits.
            info.update({{"Fn_compute_blocks", p.Fn_compute_blocks},
                         {"Bn_compute_blocks", p.Bn_compute_blocks},
                         {"residual_diff_threshold", p.residual_diff_threshold},
                         {"max_warmup_steps", p.max_warmup_steps},
                         {"max_cached_steps", p.max_cached_steps},
                         {"max_continuous_cached_steps", p.max_continuous_cached_steps}});
            if (p.mode == SD_CACHE_TAYLORSEER || p.mode == SD_CACHE_CACHE_DIT)
                info.update({{"taylorseer_n_derivatives", p.taylorseer_n_derivatives}, {"taylorseer_skip_interval", p.taylorseer_skip_interval}});
            if (p.scm_mask && *p.scm_mask)
                info.update({{"scm_mask", p.scm_mask}, {"scm_policy_dynamic", p.scm_policy_dynamic}});
        } else if (p.mode == SD_CACHE_SPECTRUM) {
            info.update({{"spectrum_w", p.spectrum_w},
                         {"spectrum_m", p.spectrum_m},
                         {"spectrum_lam", p.spectrum_lam},
                         {"spectrum_window_size", p.spectrum_window_size},
                         {"spectrum_flex_window", p.spectrum_flex_window},
                         {"spectrum_warmup_steps", p.spectrum_warmup_steps},
                         {"spectrum_stop_percent", p.spectrum_stop_percent}});
        }
        return info;
    }

    template <typename Params>
    static json common_info(const Params& p, const char* kind) {
        json info = {{"schema", "sdcpp.generation/v1"}, {"kind", kind}, {"prompt", text(p.prompt)}, {"negative_prompt", text(p.negative_prompt)}, {"width", p.width}, {"height", p.height}, {"clip_skip", p.clip_skip}, {"seed", std::to_string(p.seed)}, {"sample_params", sampling_info(p.sample_params)}};
        if (p.hires.enabled)
            info["hires"] = hires_info(p.hires);
        if (p.cache.mode != SD_CACHE_DISABLED)
            info["cache"] = cache_info(p.cache);
        if (p.circular_x)
            info["circular_x"] = true;
        if (p.circular_y)
            info["circular_y"] = true;
        if (p.ref_image_args && *p.ref_image_args)
            info["ref_image_args"] = p.ref_image_args;
        if (p.image_preprocess.rules && *p.image_preprocess.rules)
            info["image_preprocess"] = {{"rules", p.image_preprocess.rules}};
        const auto& t = p.vae_tiling_params;
        if (t.enabled || t.temporal_tiling) {
            auto& tiling = info["vae_tiling_params"];
            if (t.enabled)
                tiling.update({{"enabled", true}, {"tile_size_w", t.tile_size_w}, {"tile_size_h", t.tile_size_h}, {"target_overlap", t.target_overlap}, {"rel_size_w", t.rel_size_w}, {"rel_size_h", t.rel_size_h}});
            if (t.temporal_tiling)
                tiling["temporal_tiling"] = true;
            if (t.extra_tiling_args && *t.extra_tiling_args)
                tiling["extra_tiling_args"] = t.extra_tiling_args;
        }
        if (p.lora_count > 0) {
            info["loras"] = json::array();
            for (uint32_t i = 0; i < p.lora_count; ++i)
                info["loras"].push_back({{"path", text(p.loras[i].path)}, {"multiplier", p.loras[i].multiplier}, {"is_high_noise", p.loras[i].is_high_noise}});
        }
        return info;
    }

    static json image_info(const sd_img_gen_params_t& p) {
        auto info = common_info(p, "image");
        if (p.init_image.data)
            info["strength"] = p.strength;
        if (p.control_image.data)
            info["control_strength"] = p.control_strength;
        if (p.ip_adapter_image.data)
            info["ip_adapter_strength"] = p.ip_adapter_strength;
        if (p.pm_params.id_images && p.pm_params.id_images_count > 0) {
            info["pm_params"] = {{"style_strength", p.pm_params.style_strength}};
            if (p.pm_params.id_embed_path && *p.pm_params.id_embed_path)
                info["pm_params"]["id_embed_path"] = p.pm_params.id_embed_path;
        }
        if (p.pulid_params.id_embedding_path && *p.pulid_params.id_embedding_path)
            info["pulid_params"] = {{"id_embedding_path", p.pulid_params.id_embedding_path}, {"id_weight", p.pulid_params.id_weight}};
        return info;
    }

    template <typename Params>
    static Params resolved_params(const pipeline::GenerationRequest& request, Params p) {
        p.width             = request.width;
        p.height            = request.height;
        p.seed              = request.seed;
        p.clip_skip         = request.clip_skip;
        p.sample_params     = request.sample_params;
        p.hires             = request.hires;
        p.vae_tiling_params = request.vae_tiling_params;
        return p;
    }

    json generation_info(const pipeline::GenerationRequest& request, const sd_img_gen_params_t& params) {
        auto p              = resolved_params(request, params);
        p.qwen_image_layers = request.qwen_image_layers;
        auto info           = image_info(p);
        if (!request.models.empty())
            info["models"] = request.models;
        if (request.is_layered_image)
            info["qwen_image_layers"] = p.qwen_image_layers;
        return info;
    }

    json generation_info(const pipeline::GenerationRequest& request, const sd_vid_gen_params_t& params) {
        const auto p = resolved_params(request, params);
        auto info    = common_info(p, "video");
        if (!request.models.empty())
            info["models"] = request.models;
        info.update({{"video_frames", request.frames}, {"fps", request.fps}});
        if (p.init_image.data || p.end_image.data)
            info["strength"] = p.strength;
        if (request.use_vace)
            info["vace_strength"] = p.vace_strength;
        if (request.use_high_noise)
            info.update({{"moe_boundary", p.moe_boundary}, {"high_noise_sample_params", sampling_info(request.high_noise_sample_params, true)}});
        if (p.ref_videos_count > 0) {
            info["ref_videos"] = json::array();
            for (int i = 0; i < p.ref_videos_count; ++i)
                info["ref_videos"].push_back({{"fps", p.ref_videos[i].fps}});
        }
        return info;
    }

    json adetailer_generation_info(const sd_adetailer_params_t& params, const sd_img_gen_params_t& inpaint_params) {
        auto inpaint        = image_info(inpaint_params);
        inpaint["strength"] = inpaint_params.strength;
        // ADetailer clears these inputs before generating each crop.
        for (const auto* key : {"control_strength", "pm_params", "pulid_params", "hires", "image_preprocess"})
            inpaint.erase(key);
        sd_img_gen_params_t defaults;
        sd_img_gen_params_init(&defaults);
        if (inpaint_params.qwen_image_layers != defaults.qwen_image_layers)
            inpaint["qwen_image_layers"] = inpaint_params.qwen_image_layers;
        json adetailer = json::object();
        if (params.prompt && *params.prompt)
            adetailer["prompt"] = params.prompt;
        if (params.negative_prompt && *params.negative_prompt)
            adetailer["negative_prompt"] = params.negative_prompt;
        if (params.extra_ad_args && *params.extra_ad_args)
            adetailer["extra_ad_args"] = params.extra_ad_args;
        return {{"schema", "sdcpp.generation/v1"}, {"kind", "image"}, {"adetailer", adetailer}, {"inpaint_params", inpaint}};
    }

    json upscale_generation_info(uint32_t upscale_factor) {
        return {{"schema", "sdcpp.generation/v1"}, {"kind", "image"}, {"upscale", {{"upscale_factor", upscale_factor}}}};
    }

    char* copy_generation_info(const json& info) {
        const std::string text = info.dump(-1, ' ', false, json::error_handler_t::replace);
        char* result           = static_cast<char*>(malloc(text.size() + 1));
        if (result)
            memcpy(result, text.c_str(), text.size() + 1);
        return result;
    }

    bool make_image_generation_result(sd_image_t* images, int count, const std::vector<json>& infos, sd_image_generation_result_t** result_out) {
        if (!result_out || !images || count <= 0 || infos.size() != static_cast<size_t>(count)) {
            free_sd_images(images, count);
            return false;
        }
        auto* result = static_cast<sd_image_generation_result_t*>(calloc(1, sizeof(sd_image_generation_result_t)));
        if (!result) {
            free_sd_images(images, count);
            return false;
        }
        result->items = static_cast<sd_image_result_t*>(calloc(count, sizeof(sd_image_result_t)));
        if (!result->items) {
            free(result);
            free_sd_images(images, count);
            return false;
        }
        result->count = count;
        try {
            for (int i = 0; i < count; ++i) {
                result->items[i].image           = images[i];
                images[i]                        = {};
                result->items[i].generation_info = copy_generation_info(infos[i]);
                if (!result->items[i].generation_info) {
                    free_sd_images(images, count);
                    free_sd_image_generation_result(result);
                    return false;
                }
            }
        } catch (...) {
            free_sd_images(images, count);
            free_sd_image_generation_result(result);
            return false;
        }
        free(images);
        *result_out = result;
        return true;
    }
}

void free_sd_image_generation_result(sd_image_generation_result_t* result) {
    if (!result)
        return;
    for (int i = 0; i < result->count; ++i) {
        free(result->items[i].image.data);
        free(result->items[i].generation_info);
    }
    free(result->items);
    free(result);
}

void free_sd_video_generation_result(sd_video_generation_result_t* result) {
    if (!result)
        return;
    free_sd_images(result->frames, result->frame_count);
    free_sd_audio(result->audio);
    free(result->generation_info);
    free(result);
}

#include "option_types.h"

#include <cstring>

static constexpr const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return "NONE";
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

static constexpr const char* sample_method_to_str[] = {
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
    return "NONE";
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

static constexpr const char* scheduler_to_str[] = {
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
    "llada_image",
};

static_assert(SCHEDULER_COUNT == sizeof(scheduler_to_str) / sizeof(scheduler_to_str[0]),
              "\nnumber of elements in scheduler_to_str[] != SCHEDULER_COUNT");

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return "NONE";
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

static constexpr const char* prediction_to_str[] = {
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
    "sefi_flow",
    "minit2i_flow",
    "sensenova_u1_flow",
    "flux3_flow",
    "h3_av_flow",
    "ming_image_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction >= EPS_PRED && prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return "NONE";
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

static constexpr const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return "NONE";
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}

static constexpr const char* lora_apply_mode_to_str[] = {
    "auto",
    "immediately",
    "at_runtime",
};

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode) {
    if (mode < LORA_APPLY_MODE_COUNT) {
        return lora_apply_mode_to_str[mode];
    }
    return "NONE";
}

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str) {
    for (int i = 0; i < LORA_APPLY_MODE_COUNT; i++) {
        if (!strcmp(str, lora_apply_mode_to_str[i])) {
            return (enum lora_apply_mode_t)i;
        }
    }
    return LORA_APPLY_MODE_COUNT;
}

static constexpr const char* hires_upscaler_to_str[] = {
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
    return "NONE";
}

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str) {
    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; i++) {
        if (!strcmp(str, hires_upscaler_to_str[i])) {
            return (enum sd_hires_upscaler_t)i;
        }
    }
    return SD_HIRES_UPSCALER_COUNT;
}

static constexpr const char* control_type_to_str[] = {
    "none",
    "openpose",
    "depth",
    "scribble",
    "canny",
    "normal",
    "segment",
    "tile",
    "repaint",
};

static_assert(SD_CONTROL_TYPE_COUNT == sizeof(control_type_to_str) / sizeof(control_type_to_str[0]),
              "\nnumber of elements in control_type_to_str[] != SD_CONTROL_TYPE_COUNT");

const char* sd_control_type_name(enum sd_control_type_t control_type) {
    if (control_type >= SD_CONTROL_TYPE_NONE && control_type < SD_CONTROL_TYPE_COUNT) {
        return control_type_to_str[control_type];
    }
    return "NONE";
}

enum sd_control_type_t str_to_sd_control_type(const char* str) {
    for (int i = 0; i < SD_CONTROL_TYPE_COUNT; i++) {
        if (!strcmp(str, control_type_to_str[i])) {
            return (enum sd_control_type_t)i;
        }
    }
    static const struct {
        const char* alias;
        enum sd_control_type_t type;
    } aliases[] = {
        {"hed", SD_CONTROL_TYPE_SCRIBBLE},
        {"softedge", SD_CONTROL_TYPE_SCRIBBLE},
        {"pidi", SD_CONTROL_TYPE_SCRIBBLE},
        {"ted", SD_CONTROL_TYPE_SCRIBBLE},
        {"lineart", SD_CONTROL_TYPE_CANNY},
        {"anime_lineart", SD_CONTROL_TYPE_CANNY},
        {"mlsd", SD_CONTROL_TYPE_CANNY},
        {"seg", SD_CONTROL_TYPE_SEGMENT},
        {"inpaint", SD_CONTROL_TYPE_REPAINT},
        {"outpaint", SD_CONTROL_TYPE_REPAINT},
    };
    for (const auto& entry : aliases) {
        if (!strcmp(str, entry.alias)) {
            return entry.type;
        }
    }
    return SD_CONTROL_TYPE_COUNT;
}

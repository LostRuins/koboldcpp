#ifndef __SD_CORE_OPTION_TYPES_H__
#define __SD_CORE_OPTION_TYPES_H__

#include "stable-diffusion.h"

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    DPM2_SAMPLE_METHOD,
    DPMPP2S_A_SAMPLE_METHOD,
    DPMPP2M_SAMPLE_METHOD,
    DPMPP2Mv2_SAMPLE_METHOD,
    IPNDM_SAMPLE_METHOD,
    IPNDM_V_SAMPLE_METHOD,
    LCM_SAMPLE_METHOD,
    DDIM_TRAILING_SAMPLE_METHOD,
    TCD_SAMPLE_METHOD,
    RES_MULTISTEP_SAMPLE_METHOD,
    RES_2S_SAMPLE_METHOD,
    ER_SDE_SAMPLE_METHOD,
    EULER_CFG_PP_SAMPLE_METHOD,
    EULER_A_CFG_PP_SAMPLE_METHOD,
    EULER_GE_SAMPLE_METHOD,
    DPMPP2M_SDE_SAMPLE_METHOD,
    DPMPP2M_SDE_BT_SAMPLE_METHOD,
    LMS_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    LTX2_SCHEDULER,
    LOGIT_NORMAL_SCHEDULER,
    FLUX2_SCHEDULER,
    FLUX_SCHEDULER,
    BETA_SCHEDULER,
    LLADA_IMAGE_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED,
    V_PRED,
    EDM_V_PRED,
    FLOW_PRED,
    FLUX_FLOW_PRED,
    SEFI_FLOW_PRED,
    MINIT2I_FLOW_PRED,
    SENSENOVA_U1_FLOW_PRED,
    FLUX3_FLOW_PRED,
    H3_AV_FLOW_PRED,
    MING_IMAGE_FLOW_PRED,
    PREDICTION_COUNT
};

enum lora_apply_mode_t {
    LORA_APPLY_AUTO,
    LORA_APPLY_IMMEDIATELY,
    LORA_APPLY_AT_RUNTIME,
    LORA_APPLY_MODE_COUNT,
};

enum sd_vae_format_t {
    SD_VAE_FORMAT_AUTO = -1,
    SD_VAE_FORMAT_FLUX,
    SD_VAE_FORMAT_SD3,
    SD_VAE_FORMAT_FLUX2,
    SD_VAE_FORMAT_WAN,
    SD_VAE_FORMAT_COUNT,
};

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
    SD_CACHE_SPECTRUM,
};

enum sd_hires_upscaler_t {
    SD_HIRES_UPSCALER_NONE,
    SD_HIRES_UPSCALER_LATENT,
    SD_HIRES_UPSCALER_LATENT_NEAREST,
    SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT,
    SD_HIRES_UPSCALER_LATENT_ANTIALIASED,
    SD_HIRES_UPSCALER_LATENT_BICUBIC,
    SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED,
    SD_HIRES_UPSCALER_LANCZOS,
    SD_HIRES_UPSCALER_NEAREST,
    SD_HIRES_UPSCALER_MODEL,
    SD_HIRES_UPSCALER_COUNT,
};

enum sd_control_type_t {
    SD_CONTROL_TYPE_NONE,
    SD_CONTROL_TYPE_OPENPOSE,
    SD_CONTROL_TYPE_DEPTH,
    SD_CONTROL_TYPE_SCRIBBLE,  // thick lines: scribble, hed, softedge, ted
    SD_CONTROL_TYPE_CANNY,     // thin lines: canny, lineart, anime lineart, mlsd
    SD_CONTROL_TYPE_NORMAL,
    SD_CONTROL_TYPE_SEGMENT,
    SD_CONTROL_TYPE_TILE,
    SD_CONTROL_TYPE_REPAINT,  // inpaint, outpaint
    SD_CONTROL_TYPE_COUNT,
};

const char* sd_rng_type_name(enum rng_type_t rng_type);

enum rng_type_t str_to_rng_type(const char* str);

const char* sd_sample_method_name(enum sample_method_t sample_method);

enum sample_method_t str_to_sample_method(const char* str);

const char* sd_scheduler_name(enum scheduler_t scheduler);

enum scheduler_t str_to_scheduler(const char* str);

const char* sd_prediction_name(enum prediction_t prediction);

enum prediction_t str_to_prediction(const char* str);

const char* sd_lora_apply_mode_name(enum lora_apply_mode_t mode);

enum lora_apply_mode_t str_to_lora_apply_mode(const char* str);

const char* sd_hires_upscaler_name(enum sd_hires_upscaler_t upscaler);

enum sd_hires_upscaler_t str_to_sd_hires_upscaler(const char* str);

const char* sd_control_type_name(enum sd_control_type_t control_type);

enum sd_control_type_t str_to_sd_control_type(const char* str);

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};
const char* sd_preview_name(enum preview_t preview);
enum preview_t str_to_preview(const char* str);

#endif  // __SD_CORE_OPTION_TYPES_H__

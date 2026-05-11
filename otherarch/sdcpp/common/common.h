#ifndef __EXAMPLES_COMMON_COMMON_H__
#define __EXAMPLES_COMMON_COMMON_H__

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "log.h"
#include "resource_owners.hpp"
#include "stable-diffusion.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

extern const char* const modes_str[];
#define SD_ALL_MODES_STR "img_gen, vid_gen, convert, upscale, metadata"

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    UPSCALE,
    METADATA,
    MODE_COUNT
};

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::string* target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int* target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    float* target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    bool* target;
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::function<int(int argc, const char** argv, int index)> cb;
};

struct ArgOptions {
    std::vector<StringOption> string_options;
    std::vector<IntOption> int_options;
    std::vector<FloatOption> float_options;
    std::vector<BoolOption> bool_options;
    std::vector<ManualOption> manual_options;

    static std::string wrap_text(const std::string& text, size_t width, size_t indent);
    void print() const;
};

bool parse_options(int argc, const char** argv, const std::vector<ArgOptions>& options_list);
bool decode_base64_image(const std::string& encoded_input,
                         int target_channels,
                         int expected_width,
                         int expected_height,
                         SDImageOwner& out_image);

struct SDContextParams {
    int n_threads = -1;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    std::string photo_maker_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir = ".";
    std::string hires_upscalers_dir;

    std::map<std::string, std::string> embedding_map;
    std::vector<sd_embedding_t> embedding_vec;

    rng_type_t rng_type         = CUDA_RNG;
    rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
    bool offload_params_to_cpu  = false;
    bool enable_mmap            = false;
    bool control_net_cpu        = false;
    bool clip_on_cpu            = false;
    bool vae_on_cpu             = false;
    bool flash_attn             = false;
    bool diffusion_flash_attn   = false;
    bool diffusion_conv_direct  = false;
    bool vae_conv_direct        = false;

    bool circular   = false;
    bool circular_x = false;
    bool circular_y = false;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;

    bool qwen_image_zero_cond_t = false;

    prediction_t prediction           = PREDICTION_COUNT;
    lora_apply_mode_t lora_apply_mode = LORA_APPLY_AUTO;

    bool force_sdxl_vae_conv_scale = false;

    float flow_shift = INFINITY;
    ArgOptions get_options();
    void build_embedding_map();
    bool resolve(SDMode mode);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode);
    std::string to_string() const;
    sd_ctx_params_t to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately, bool taesd_preview);
};

struct SDGenerationParams {
    // User-facing input fields.
    std::string prompt;
    std::string negative_prompt;
    int clip_skip              = -1;  // <= 0 represents unspecified
    int width                  = -1;
    int height                 = -1;
    int batch_count            = 1;
    int64_t seed               = 42;
    float strength             = 0.75f;
    float control_strength     = 0.9f;
    bool auto_resize_ref_image = true;
    bool increase_ref_index    = false;
    bool embed_image_metadata  = true;

    std::string init_image_path;
    std::string end_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    std::string control_video_path;

    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    std::vector<int> skip_layers            = {7, 8, 9};
    std::vector<int> high_noise_skip_layers = {7, 8, 9};

    std::vector<float> custom_sigmas;

    std::string cache_mode;
    std::string cache_option;
    std::string scm_mask;
    bool scm_policy_dynamic = true;
    sd_cache_params_t cache_params{};

    float moe_boundary                   = 0.875f;
    int video_frames                     = 1;
    int fps                              = 16;
    float vace_strength                  = 1.f;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};

    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    float pm_style_strength = 20.f;

    int upscale_repeats   = 1;
    int upscale_tile_size = 128;

    bool hires_enabled         = false;
    std::string hires_upscaler = "Latent";
    std::string hires_upscaler_model_path;
    float hires_scale              = 2.f;
    int hires_width                = 0;
    int hires_height               = 0;
    int hires_steps                = 0;
    float hires_denoising_strength = 0.7f;
    int hires_upscale_tile_size    = 128;

    std::map<std::string, float> lora_map;
    std::map<std::string, float> high_noise_lora_map;

    // Derived and normalized fields.
    std::string prompt_with_lora;  // for metadata record only
    std::vector<sd_lora_t> lora_vec;
    sd_hires_upscaler_t resolved_hires_upscaler;

    // Owned execution payload.
    SDImageOwner init_image;
    SDImageOwner end_image;
    std::vector<SDImageOwner> ref_images;
    SDImageOwner mask_image;
    SDImageOwner control_image;
    std::vector<SDImageOwner> pm_id_images;
    std::vector<SDImageOwner> control_frames;

    // Backing storage for sd_img_gen_params_t view fields.
    std::vector<sd_image_t> ref_image_views;
    std::vector<sd_image_t> pm_id_image_views;
    std::vector<sd_image_t> control_frame_views;

    SDGenerationParams();
    SDGenerationParams(const SDGenerationParams& other)                = default;
    SDGenerationParams& operator=(const SDGenerationParams& other)     = default;
    SDGenerationParams(SDGenerationParams&& other) noexcept            = default;
    SDGenerationParams& operator=(SDGenerationParams&& other) noexcept = default;
    ArgOptions get_options();
    bool from_json_str(const std::string& json_str,
                       const std::function<std::string(const std::string&)>& lora_path_resolver = {});
    bool initialize_cache_params();
    void extract_and_remove_lora(const std::string& lora_model_dir);
    bool width_and_height_are_set() const;
    void set_width_and_height_if_unset(int w, int h);
    int get_resolved_width() const;
    int get_resolved_height() const;
    bool resolve(const std::string& lora_model_dir, const std::string& hires_upscalers_dir, bool strict = false);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode,
                              const std::string& lora_model_dir,
                              const std::string& hires_upscalers_dir,
                              bool strict = false);
    sd_img_gen_params_t to_sd_img_gen_params_t();
    sd_vid_gen_params_t to_sd_vid_gen_params_t();
    std::string to_string() const;
};

std::string version_string();
std::string build_sdcpp_image_metadata_json(const SDContextParams& ctx_params,
                                            const SDGenerationParams& gen_params,
                                            int64_t seed,
                                            SDMode mode = IMG_GEN);
std::string get_image_params(const SDContextParams& ctx_params,
                             const SDGenerationParams& gen_params,
                             int64_t seed,
                             SDMode mode = IMG_GEN);

#endif  // __EXAMPLES_COMMON_COMMON_H__

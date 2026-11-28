#ifndef __EXAMPLES_COMMON_COMMON_H__
#define __EXAMPLES_COMMON_COMMON_H__

#include "generation_metadata.h"
#include "options.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "log.h"
#include "resource_owners.hpp"
#include "stable-diffusion.h"

// The command-line argument at `index` as UTF-8 (on Windows, read again from
// the UTF-16 command line: argv itself is in the ANSI code page).
std::string argv_to_utf8(int index, const char** argv);

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

extern const char* const modes_str[];
#define SD_ALL_MODES_STR "img_gen, adetailer, vid_gen, convert, upscale, metadata, vid_upscale"

enum SDMode {
    IMG_GEN,
    ADETAILER,
    VID_GEN,
    CONVERT,
    UPSCALE,
    METADATA,
    VID_UPSCALE,
    MODE_COUNT
};

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int concat;
    OptionTarget<std::string> target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    OptionTarget<int> target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    OptionTarget<float> target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    OptionTarget<bool> target;
};

struct ManualFunction {
    std::function<int(int, const char**, int, bool&)> _func;

    ManualFunction() = default;

    ManualFunction(std::function<int(int argc, const char** argv, int index, bool& valid)> func)
        : _func(std::move(func)) {
    }

    template <typename F>
    ManualFunction(F func)
        : _func(make_function(func)) {
    }

    int operator()(int argc, const char** argv, int index, bool& valid) const {
        return _func(argc, argv, index, valid);
    }

private:
    template <typename F>
    static std::function<int(int, const char**, int, bool&)> make_function(F func) {
        if constexpr (std::is_invocable_v<F, int, const char**, int, bool&>) {
            return func;
        } else {
            return [func](int argc, const char** argv, int index, bool&) {
                return func(argc, argv, index);
            };
        }
    }
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    ManualFunction cb;
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

void add_log_options(ArgOptions& options, sd_log_level_t& level);
bool parse_options(int argc, const char** argv, const std::vector<ArgOptions>& options_list);
bool decode_base64_image(const std::string& encoded_input,
                         int target_channels,
                         int expected_width,
                         int expected_height,
                         SDImageOwner& out_image);

struct SDContextParams {
    std::optional<int> n_threads;
    std::optional<int> conditioning_cache_size;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string clip_vision_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string tokenizer;
    std::string diffusion_model_path;
    std::string high_noise_diffusion_model_path;
    std::string uncond_diffusion_model_path;
    std::string embeddings_connectors_path;
    std::string vae_path;
    std::optional<std::string> vae_format;
    std::string audio_vae_path;
    std::string audio_encoder_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string ip_adapter_path;
    std::string motion_module_path;
    std::string embedding_dir;
    std::string photo_maker_path;
    std::string pulid_weights_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir = ".";
    std::string hires_upscalers_dir;

    std::map<std::string, std::string> embedding_map;
    std::vector<sd_embedding_t> embedding_vec;

    std::optional<std::string> rng_type;
    std::optional<std::string> sampler_rng_type;
    bool offload_params_to_cpu = false;
    std::optional<std::string> max_vram;
    std::optional<bool> disable_prefetch;
    std::optional<bool> disable_segmented_compute;
    std::optional<bool> eager_load;
    std::optional<bool> batched_cfg;
    std::optional<std::string> backend;
    std::optional<std::string> params_backend;
    std::optional<std::string> split_mode;
    std::optional<std::string> model_args;
    std::optional<bool> auto_fit;
    std::optional<std::string> rpc_servers;
    bool list_devices = false;
    std::string effective_backend;
    std::string effective_params_backend;
    std::optional<bool> enable_mmap;
    bool control_net_cpu = false;
    bool clip_on_cpu     = false;
    bool vae_on_cpu      = false;
    std::optional<std::string> flash_attn_mode;
    std::optional<std::string> conv_direct_mode;
    std::optional<bool> sage_attn;
    std::optional<bool> diffusion_conv_direct;
    std::optional<bool> vae_conv_direct;
    std::optional<bool> taesd_preview_only;

    std::optional<std::string> prediction;
    std::optional<std::string> lora_apply_mode;

    std::optional<bool> force_sdxl_vae_conv_scale;
    std::optional<float> linear_scale;
    std::optional<float> attn_scale;

    ArgOptions get_options();
    bool print_devices() const;
    void build_embedding_map();
    void prepare_backend_assignments();
    bool resolve(SDMode mode);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode);
    std::string to_string() const;
    SDOptPtr to_context_opt();
};

struct SDGenerationParams {
    // User-facing input fields.
    std::string prompt;
    std::string negative_prompt;
    std::string ad_model_path;
    std::string ad_prompt;
    std::string ad_negative_prompt;
    std::string extra_ad_args;
    std::optional<int> clip_skip;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<int> batch_count;
    std::optional<int> qwen_image_layers;
    std::optional<int64_t> seed;
    std::optional<float> strength;
    std::optional<float> control_strength;
    std::optional<float> ip_adapter_strength;
    bool increase_ref_index   = false;
    bool embed_image_metadata = true;

    std::string init_image_path;
    std::string end_image_path;
    std::string mask_image_path;
    std::string control_image_path;
    std::optional<std::string> control_type;
    std::string ip_adapter_image_path;
    std::vector<std::string> ref_image_paths;
    std::vector<std::string> ref_video_paths;
    std::vector<std::string> ref_video_audio_paths;
    std::vector<std::string> ref_audio_paths;
    std::string control_video_path;

    SDSampleParams sample_params;
    SDSampleParams high_noise_sample_params;
    std::string extra_sample_args;
    std::string high_noise_extra_sample_args;
    std::optional<std::vector<int>> skip_layers;
    std::optional<std::vector<int>> high_noise_skip_layers;

    std::optional<std::vector<float>> custom_sigmas;

    std::string cache_mode;
    std::string cache_option;
    std::string scm_mask;
    std::optional<bool> scm_policy_dynamic;
    SDCacheParams cache_params{};

    std::optional<float> moe_boundary;
    std::optional<int> video_frames;
    std::optional<int> fps;
    std::optional<float> vace_strength;
    SDTilingParams vae_tiling_params;
    std::string extra_tiling_args;

    std::string ref_image_args;
    std::string image_preprocess;

    std::string pm_id_images_dir;
    std::string pm_id_embed_path;
    std::optional<float> pm_style_strength;

    std::string pulid_id_embedding_path;
    std::optional<float> pulid_id_weight;

    int upscale_repeats = 1;
    std::optional<int> upscale_tile_size;

    bool circular = false;
    std::optional<bool> circular_x;
    std::optional<bool> circular_y;

    std::optional<bool> hires_enabled;
    std::optional<std::string> hires_upscaler;
    std::string hires_upscaler_model_path;
    std::optional<float> hires_scale;
    std::optional<int> hires_width;
    std::optional<int> hires_height;
    std::optional<int> hires_steps;
    std::optional<float> hires_denoising_strength;
    std::optional<int> hires_upscale_tile_size;
    std::optional<std::vector<float>> hires_custom_sigmas;

    std::map<std::string, float> lora_map;
    std::map<std::string, float> high_noise_lora_map;

    // Derived and normalized fields.
    std::string prompt_with_lora;  // for metadata record only
    std::vector<sd_lora_t> lora_vec;
    std::string resolved_hires_upscaler;

    // Owned execution payload.
    SDImageOwner init_image;
    SDImageOwner end_image;
    std::vector<SDImageOwner> ref_images;
    std::map<std::string, SDImageOwner> prompt_images;
    std::vector<std::vector<SDImageOwner>> ref_videos;
    std::vector<SDAudioOwner> ref_video_audios;
    std::vector<SDAudioOwner> ref_audios;
    SDImageOwner mask_image;
    SDImageOwner control_image;
    SDImageOwner ip_adapter_image;
    std::vector<SDImageOwner> pm_id_images;
    std::vector<SDImageOwner> control_frames;

    // Media views used while populating the opaque options.
    std::vector<SDImage> ref_image_views;
    std::vector<std::vector<SDImage>> ref_video_frame_views;
    std::vector<SDReferenceVideo> ref_video_views;
    std::vector<SDAudio> ref_audio_views;
    std::vector<SDImage> pm_id_image_views;
    std::vector<SDImage> control_frame_views;

    SDGenerationParams();
    SDGenerationParams(const SDGenerationParams& other)                = default;
    SDGenerationParams& operator=(const SDGenerationParams& other)     = default;
    SDGenerationParams(SDGenerationParams&& other) noexcept            = default;
    SDGenerationParams& operator=(SDGenerationParams&& other) noexcept = default;
    ArgOptions get_options();
    bool from_json_str(const std::string& json_str,
                       const std::function<std::string(const std::string&)>& lora_path_resolver = {});
    bool parse_image_preprocess_json(const std::string& json_str);
    bool parse_prompt_images_json(const std::string& json_str);
    bool initialize_cache_params();
    bool extract_and_remove_lora(const std::string& lora_model_dir);
    bool width_and_height_are_set() const;
    void set_width_and_height_if_unset(int w, int h);
    int get_requested_width() const;
    int get_requested_height() const;
    bool resolve(const std::string& lora_model_dir, const std::string& hires_upscalers_dir, bool strict = false, bool validate_missing_loras = false);
    bool validate(SDMode mode);
    bool resolve_and_validate(SDMode mode,
                              const std::string& lora_model_dir,
                              const std::string& hires_upscalers_dir,
                              bool strict = false);
    SDOptPtr to_image_opt(bool adetail = false);
    SDOptPtr to_video_opt();
    std::string to_string() const;
};

std::string version_string();
#endif  // __EXAMPLES_COMMON_COMMON_H__

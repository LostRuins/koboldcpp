#ifndef SD_CONFIG_H
#define SD_CONFIG_H

#include "core/media.h"

#include "core/option_types.h"
#include "stable-diffusion.h"

namespace sd {
    struct TilingConfig {
        bool enabled;
        bool temporal_tiling;
        // Spatial tile dimensions in image pixels for both encode and decode; 0 uses 256.
        int tile_size_w;
        int tile_size_h;
        float target_overlap;
        // Positive values override tile_size: <= 1 is a dimension fraction, > 1 a target tile count.
        float rel_size_w;
        float rel_size_h;
        const char* extra_tiling_args;
    };

    struct ImagePreprocessConfig {
        // Semicolon-separated target=...,key=value rules. NULL preserves defaults.
        const char* rules;
    };

    struct SLGConfig {
        int* layers;
        size_t layer_count;
        float layer_start;
        float layer_end;
        float scale;
    };

    struct GuidanceConfig {
        float txt_cfg;
        float img_cfg;
        float distilled_guidance;
        sd::SLGConfig slg;
    };

    struct SampleConfig {
        sd::GuidanceConfig guidance;
        enum scheduler_t scheduler;
        enum sample_method_t sample_method;
        int sample_steps;
        float eta;
        int shifted_timestep;
        float* custom_sigmas;
        int custom_sigmas_count;
        float flow_shift;
        const char* extra_sample_args;

        SampleConfig();
    };

    struct PhotoMakerConfig {
        sd::Image* id_images;
        int id_images_count;
        const char* id_embed_path;
        float style_strength;
    };

    struct PuLIDConfig {
        const char* id_embedding_path;
        float id_weight;
    };

    struct CacheConfig {
        enum sd_cache_mode_t mode;
        float reuse_threshold;
        float start_percent;
        float end_percent;
        float error_decay_rate;
        bool use_relative_threshold;
        bool reset_error_on_compute;
        int Fn_compute_blocks;
        int Bn_compute_blocks;
        float residual_diff_threshold;
        int max_warmup_steps;
        int max_cached_steps;
        int max_continuous_cached_steps;
        int taylorseer_n_derivatives;
        int taylorseer_skip_interval;
        const char* scm_mask;
        bool scm_policy_dynamic;
        float spectrum_w;
        int spectrum_m;
        float spectrum_lam;
        int spectrum_window_size;
        float spectrum_flex_window;
        int spectrum_warmup_steps;
        float spectrum_stop_percent;

        CacheConfig();
    };

    struct HiresConfig {
        bool enabled;
        enum sd_hires_upscaler_t upscaler;
        const char* model_path;
        float scale;
        int target_width;
        int target_height;
        int steps;
        float denoising_strength;
        int upscale_tile_size;
        float* custom_sigmas;
        int custom_sigmas_count;

        HiresConfig();
    };

    struct ContextConfig {
        const char* model_path{};
        const char* clip_l_path{};
        const char* clip_g_path{};
        const char* clip_vision_path{};
        const char* t5xxl_path{};
        const char* llm_path{};
        const char* llm_vision_path{};
        const char* diffusion_model_path{};
        const char* high_noise_diffusion_model_path{};
        const char* uncond_diffusion_model_path{};
        const char* embeddings_connectors_path{};
        const char* vae_path{};
        const char* audio_vae_path{};
        const char* audio_encoder_path{};
        const char* taesd_path{};
        const char* control_net_path{};
        const char* ip_adapter_path{};
        const char* motion_module_path{};
        const sd_embedding_t* embeddings{};
        uint32_t embedding_count{};
        const char* photo_maker_path{};
        const char* pulid_weights_path{};
        const char* tensor_type_rules{};
        int n_threads{};
        enum sd_type_t wtype {};
        enum rng_type_t rng_type {};
        enum rng_type_t sampler_rng_type {};
        enum prediction_t prediction {};
        enum lora_apply_mode_t lora_apply_mode {};
        bool enable_mmap{};
        bool flash_attn{};
        bool diffusion_flash_attn{};
        bool tae_preview_only{};
        bool diffusion_conv_direct{};
        bool vae_conv_direct{};
        bool force_sdxl_vae_conv_scale{};
        enum sd_vae_format_t vae_format {};
        const char* max_vram{};   // Optional per-device GiB budget for managed weights and runner buffers; 0 uses live free VRAM without an explicit budget
        bool disable_prefetch{};  // Disable asynchronous next-segment weight prefetch
        bool eager_load{};        // Load all params into the params backend at model-load time instead of lazily on first use
        const char* backend{};
        const char* params_backend{};
        const char* split_mode{};  // weight distribution for multi-device modules: layer (default) or row, or per-module assignments e.g. "diffusion=row"
        bool auto_fit{};
        const char* rpc_servers{};
        const char* model_args{};
        bool disable_segmented_compute{};  // Force monolithic graph execution even when automatic graph cutting would fit memory better
        bool batched_cfg{};                // Run the conditional and unconditional CFG branches in one batched UNet forward when supported
        float linear_scale{};              // Override linear input scaling; 0 keeps the model default
        float attn_scale{};                // Override flash-attention K/V scaling; 0 keeps the model default
        const char* tokenizer{};           // tokenizer.json path or main=FILE,clip-l=FILE,clip-g=FILE assignments; required for PiD and Lens
        bool sage_attn{};
        int conditioning_cache_size{};   // Maximum cached conditioning entries per context; 0 disables caching (default: 4)
        const char* flash_attn_mode{};   // on/off/auto or module assignments (e.g. "auto,te=off"); NULL/empty honors legacy flags.
        const char* conv_direct_mode{};  // on/off/auto or diffusion/vae/controlnet assignments; NULL/empty honors legacy flags.

        ContextConfig();
    };

    struct ImageConfig {
        const sd_lora_t* loras{};
        uint32_t lora_count{};
        const char* prompt{};
        const char* negative_prompt{};
        int clip_skip{};
        sd::Image init_image{};
        sd::Image* ref_images{};
        int ref_images_count{};
        const char* ref_image_args{};
        sd::Image mask_image{};
        int width{};
        int height{};
        sd::SampleConfig sample_params{};
        float strength{};
        int64_t seed{};
        int batch_count{};
        sd::Image control_image{};
        float control_strength{};
        sd::Image ip_adapter_image{};
        float ip_adapter_strength{};
        sd::PhotoMakerConfig pm_params{};
        sd::PuLIDConfig pulid_params{};
        sd::TilingConfig vae_tiling_params{};
        sd::CacheConfig cache{};
        sd::HiresConfig hires{};
        int qwen_image_layers{};
        bool circular_x{};
        bool circular_y{};
        sd::ImagePreprocessConfig image_preprocess{};
        enum sd_control_type_t control_type {};

        ImageConfig();
    };

    struct VideoConfig {
        const sd_lora_t* loras{};
        uint32_t lora_count{};
        const char* prompt{};
        const char* negative_prompt{};
        int clip_skip{};
        sd::Image init_image{};
        sd::Image end_image{};
        sd::Image* ref_images{};
        int ref_images_count{};
        sd::ReferenceVideo* ref_videos{};
        int ref_videos_count{};
        sd::Audio* ref_audios{};
        int ref_audios_count{};
        sd::Image* control_frames{};
        int control_frames_size{};
        int width{};
        int height{};
        sd::SampleConfig sample_params{};
        sd::SampleConfig high_noise_sample_params{};
        float moe_boundary{};
        float strength{};
        int64_t seed{};
        int video_frames{};
        int fps{};
        float vace_strength{};
        sd::TilingConfig vae_tiling_params{};
        sd::CacheConfig cache{};
        sd::HiresConfig hires{};
        bool circular_x{};
        bool circular_y{};
        sd::ImagePreprocessConfig image_preprocess{};
        const char* ref_image_args{};

        VideoConfig();
    };

    struct VideoUpscalerConfig {
        const char* diffusion_model_path{};
        const char* vae_path{};
        // Explicit output scale; RelayVSR requires 2 or 4 matching both checkpoints.
        int upscale_factor{};
        int n_threads{};
        const char* backend{};
        const char* params_backend{};
        // auto/on/off or diffusion/vae assignments; null selects auto.
        const char* flash_attn_mode{};
        const char* conv_direct_mode{};
    };

    struct VideoUpscaleConfig {
        const sd::Image* frames{};
        int frame_count{};
        int fps      = 24;
        int64_t seed = 42;
        // RelayVSR reference window, including both endpoints; defaults to 16.
        int window_size = 16;
        // SeedVR2 independent temporal chunks in input frames (1 or 4n+1); defaults to 5.
        int temporal_chunk_size = 5;
    };

    struct ADetailConfig {
        const char* prompt{};
        const char* negative_prompt{};
        const char* extra_ad_args{};
    };

    struct PostprocessConfig {
        sd::Image image{};
        bool conv_direct = false;
        int tile_size    = 128;
        VideoUpscalerConfig video_upscaler;
        VideoUpscaleConfig video_upscale;
        ADetailConfig adetail;
    };

    struct OptConfig {
        ContextConfig context;
        ImageConfig image;
        VideoConfig video;
        PostprocessConfig post;
    };

}  // namespace sd

#endif  // SD_CONFIG_H

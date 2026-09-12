#ifndef __SD_PIPELINE_REQUEST_H__
#define __SD_PIPELINE_REQUEST_H__

#include <string>
#include <vector>

#include "stable-diffusion.h"

class StableDiffusionGGML;

namespace sd::pipeline {

    extern const char* sampling_methods_str[];

    enum sample_method_t default_sample_method(const StableDiffusionGGML* sd);

    enum scheduler_t default_scheduler(const StableDiffusionGGML* sd, enum sample_method_t sample_method);

    float resolve_eta(StableDiffusionGGML* sd,
                      float eta,
                      enum sample_method_t sample_method);

    struct GenerationRequest {
        std::string prompt;
        std::string negative_prompt;
        int width                                = -1;
        int height                               = -1;
        int clip_skip                            = -1;
        int vae_scale_factor                     = -1;
        int diffusion_model_down_factor          = -1;
        int64_t seed                             = -1;
        bool use_uncond                          = false;
        bool use_img_uncond                      = false;
        bool use_high_noise_uncond               = false;
        bool use_high_noise_img_uncond           = false;
        bool has_ref_images                      = false;
        const sd_cache_params_t* cache_params    = nullptr;
        int batch_count                          = 1;
        int qwen_image_layers                    = 3;
        int shifted_timestep                     = 0;
        float strength                           = 1.f;
        float control_strength                   = 0.f;
        float eta                                = 0.f;
        sd_guidance_params_t guidance            = {};
        sd_guidance_params_t high_noise_guidance = {};
        sd_pm_params_t pm_params                 = {};
        sd_pulid_params_t pulid_params           = {};
        sd_hires_params_t hires                  = {};
        int frames                               = -1;
        int requested_frames                     = -1;
        int fps                                  = 16;
        float vace_strength                      = 1.f;

        GenerationRequest(StableDiffusionGGML* sd, const sd_img_gen_params_t* sd_img_gen_params);

        GenerationRequest(StableDiffusionGGML* sd, const sd_vid_gen_params_t* sd_vid_gen_params);

        void align_generation_request_size();

        void align_image_size(int* target_width, int* target_height, const char* label);

        void resolve_hires();

        static void resolve_guidance(StableDiffusionGGML* sd,
                                     sd_guidance_params_t* guidance,
                                     bool* use_uncond,
                                     bool* use_img_uncond,
                                     bool has_ref_images,
                                     const char* stage_name = nullptr);

        void resolve(StableDiffusionGGML* sd);
    };

    struct SamplePlan {
        enum sample_method_t sample_method            = SAMPLE_METHOD_COUNT;
        enum sample_method_t high_noise_sample_method = SAMPLE_METHOD_COUNT;
        const char* extra_sample_args                 = nullptr;
        const char* high_noise_extra_sample_args      = nullptr;
        float eta                                     = 0.f;
        float high_noise_eta                          = 0.f;
        int sample_steps                              = 0;
        int high_noise_sample_steps                   = 0;
        int total_steps                               = 0;
        float moe_boundary                            = 0.f;
        std::vector<float> sigmas;

        SamplePlan(StableDiffusionGGML* sd,
                   const sd_img_gen_params_t* sd_img_gen_params,
                   const GenerationRequest& request);

        SamplePlan(StableDiffusionGGML* sd,
                   const sd_vid_gen_params_t* sd_vid_gen_params,
                   const GenerationRequest& request);

        void resolve(StableDiffusionGGML* sd,
                     const GenerationRequest* request,
                     const sd_sample_params_t* sample_params);
    };

    std::vector<float> make_hires_sigma_schedule(StableDiffusionGGML* sd,
                                                 const sd_hires_params_t& hires,
                                                 const sd_sample_params_t& sample_params,
                                                 sample_method_t sample_method,
                                                 int default_steps,
                                                 int sample_seq_len,
                                                 int* scheduler_steps_out);

}  // namespace sd::pipeline

#endif  // __SD_PIPELINE_REQUEST_H__

#ifndef __SD_PIPELINE_GENERATION_H__
#define __SD_PIPELINE_GENERATION_H__

#include "core/media.h"

#include "conditioning/conditioner.hpp"
#include "config.h"
#include "json.hpp"
#include "stable-diffusion.h"

class StableDiffusionGGML;

namespace sd {
    struct ImageGenerationResult;
    struct VideoGenerationResult;
    sd_status_t generate_image(sd_ctx_t* ctx, const ImageConfig& config, ImageGenerationResult** result_out, const sd_opt_t* source = nullptr) noexcept;
    sd_status_t generate_video(sd_ctx_t* ctx, const VideoConfig& config, VideoGenerationResult** result_out, const sd_opt_t* source = nullptr) noexcept;
}

static inline bool sd_version_supports_animatediff(SDVersion version) {
    return version == VERSION_SD1 || version == VERSION_SD1_INPAINT || version == VERSION_SD1_PIX2PIX;
}

namespace sd::pipeline {

    struct ImageGenerationLatents {
        sd::Tensor<float> init_latent;
        sd::Tensor<float> concat_latent;
        sd::Tensor<float> img_uncond_concat_latent;
        sd::Tensor<float> audio_latent;
        sd::Tensor<float> video_positions;
        sd::Tensor<float> control_image;
        std::vector<sd::Tensor<float>> ref_images;
        std::vector<sd::Tensor<float>> ref_latents;
        std::vector<sd::Tensor<float>> reference_audio_latents;
        std::vector<MiniMaxH3ReferenceBlock> minimax_reference_blocks;
        std::vector<MiniMaxH3PresentationItem> minimax_presentation_refs;
        std::vector<int32_t> keyframe_indices;
        sd::Tensor<float> denoise_mask;
        sd::Tensor<float> clip_vision_output;
        sd::Tensor<float> vace_context;
        sd::Tensor<float> s2v_audio_embed;
        int64_t ref_image_num                  = 0;
        int64_t video_conditioning_frame_count = 0;
        int64_t video_target_frame_count       = 0;
        int audio_length                       = 0;
    };

    struct ImageGenerationEmbeds {
        SDCondition cond;
        SDCondition uncond;
        SDCondition img_uncond;
    };

    struct ConditionerRunnerEndOnExit {
        Conditioner* conditioner = nullptr;
        ~ConditionerRunnerEndOnExit() {
            if (conditioner != nullptr) {
                conditioner->runner_end();
            }
        }
    };

    // Callers hold ExecutionScope; AnimateDiff reuses the image path within the same scope.
    bool generate_image(StableDiffusionGGML* sd,
                        const sd::ImageConfig* sd_img_gen_params,
                        sd::Image** images_out,
                        int* num_images_out,
                        std::vector<nlohmann::json>* infos_out = nullptr);

    bool generate_video(StableDiffusionGGML* sd,
                        const sd::VideoConfig* sd_vid_gen_params,
                        sd::Image** frames_out,
                        int* num_frames_out,
                        sd::Audio** audio_out,
                        int* fps_out,
                        nlohmann::json* info_out = nullptr);

    sd::Tensor<float> upscale_ltx_spatial_video_latent(StableDiffusionGGML* sd,
                                                       const char* model_path,
                                                       const sd::Tensor<float>& packed_latent,
                                                       int audio_length);

    sd::Tensor<float> ensure_image_tensor_channels(sd::Tensor<float> image, int channels);

}  // namespace sd::pipeline

#endif  // __SD_PIPELINE_GENERATION_H__

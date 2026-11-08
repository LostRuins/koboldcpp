#ifndef __SD_PIPELINE_VIDEO_UPSCALE_H__
#define __SD_PIPELINE_VIDEO_UPSCALE_H__

#include <memory>

#include "core/ggml_extend_backend.h"
#include "core/tensor.hpp"
#include "model_manager.h"
#include "stable-diffusion.h"

namespace sd {
    namespace pipeline {

        struct VideoUpscaleRuntime {
            SDBackendManager backends;
            std::shared_ptr<ModelManager> manager;
            int n_threads = 1;
        };

        class VideoUpscaleModel {
        public:
            virtual ~VideoUpscaleModel()                                                                             = default;
            virtual bool init(VideoUpscaleRuntime& runtime, const String2TensorStorage& weights, int upscale_factor) = 0;
            virtual void end()                                                                                       = 0;
            virtual bool upscale(const sd_video_upscale_params_t& params, sd_video_generation_result_t* result)      = 0;
        };

        bool is_relay_vsr_model(const String2TensorStorage& weights);
        std::unique_ptr<VideoUpscaleModel> create_relay_vsr_upscaler();

        class VideoUpscaler {
            VideoUpscaleRuntime runtime_;
            int upscale_factor_ = 0;
            std::unique_ptr<sd::ParallelExecutor> executor_;
            std::unique_ptr<VideoUpscaleModel> model_;

        public:
            ~VideoUpscaler();
            bool init(const sd_video_upscaler_ctx_params_t& params);
            bool upscale(const sd_video_upscale_params_t& params, sd_video_generation_result_t** result_out);
        };

    }  // namespace pipeline
}  // namespace sd

#endif  // __SD_PIPELINE_VIDEO_UPSCALE_H__

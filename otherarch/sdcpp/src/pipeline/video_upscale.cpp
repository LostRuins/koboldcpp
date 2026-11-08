#include "pipeline/video_upscale.h"

#include <climits>

#include "core/util.h"

namespace sd {
    namespace pipeline {

        VideoUpscaler::~VideoUpscaler() {
            if (model_)
                model_->end();
            // ModelManager retains parameter pointers owned by the model runners.
            runtime_.manager.reset();
            model_.reset();
        }

        bool VideoUpscaler::init(const sd_video_upscaler_ctx_params_t& params) {
            if (!params.diffusion_model_path || !*params.diffusion_model_path ||
                !params.vae_path || !*params.vae_path || params.upscale_factor <= 0) {
                LOG_ERROR("video upscale requires diffusion and decoder model paths and a positive upscale factor");
                return false;
            }
            upscale_factor_    = params.upscale_factor;
            runtime_.n_threads = params.n_threads > 0 ? params.n_threads : sd_get_num_physical_cores();
            executor_          = std::make_unique<sd::ParallelExecutor>(runtime_.n_threads);
            sd::ParallelScope scope(executor_.get());
            std::string error;
            if (!runtime_.backends.init(params.backend, params.params_backend, nullptr, &error)) {
                LOG_ERROR("video upscaler backend: %s", error.c_str());
                return false;
            }
            for (auto module : {SDBackendModule::DIFFUSION, SDBackendModule::VAE}) {
                if (!runtime_.backends.runtime_backend(module) || !runtime_.backends.params_backend(module))
                    return false;
            }
            ModelLoader loader;
            if (!loader.init_from_file(params.diffusion_model_path, "diffusion.") ||
                !loader.init_from_file(params.vae_path, "vae."))
                return false;
            const auto& weights = loader.get_tensor_storage_map();
            if (is_relay_vsr_model(weights))
                model_ = create_relay_vsr_upscaler();
            else {
                LOG_ERROR("unsupported video upscale model architecture");
                return false;
            }
            runtime_.manager = std::make_shared<ModelManager>();
            runtime_.manager->set_n_threads(runtime_.n_threads);
            if (!runtime_.manager->set_loader(loader))
                return false;
            return model_->init(runtime_, weights, upscale_factor_);
        }

        bool VideoUpscaler::upscale(const sd_video_upscale_params_t& params, sd_video_generation_result_t** result_out) {
            if (!result_out)
                return false;
            *result_out = nullptr;
            if (!model_ || !params.frames || params.frame_count <= 0 || params.fps <= 0) {
                LOG_ERROR("video upscale requires nonempty RGB frames and positive FPS");
                return false;
            }
            const auto& first = params.frames[0];
            if (!first.width || !first.height || first.width > uint32_t(INT_MAX / upscale_factor_) ||
                first.height > uint32_t(INT_MAX / upscale_factor_))
                return false;
            for (int i = 0; i < params.frame_count; ++i) {
                const auto& frame = params.frames[i];
                if (!frame.data || frame.channel != 3 || frame.width != first.width || frame.height != first.height) {
                    LOG_ERROR("video upscale input frames must have equal dimensions and three RGB channels");
                    return false;
                }
            }
            sd::ParallelScope scope(executor_.get());
            auto result = std::unique_ptr<sd_video_generation_result_t, decltype(&free_sd_video_generation_result)>(
                static_cast<sd_video_generation_result_t*>(calloc(1, sizeof(sd_video_generation_result_t))), free_sd_video_generation_result);
            if (!result)
                return false;
            result->frames = static_cast<sd_image_t*>(calloc(params.frame_count, sizeof(sd_image_t)));
            if (!result->frames)
                return false;
            result->frame_count = params.frame_count;
            result->fps         = params.fps;
            if (!model_->upscale(params, result.get()))
                return false;
            *result_out = result.release();
            return true;
        }

    }  // namespace pipeline
}  // namespace sd

struct video_upscaler_ctx_t {
    sd::pipeline::VideoUpscaler upscaler;
};

void sd_video_upscaler_ctx_params_init(sd_video_upscaler_ctx_params_t* params) {
    if (!params)
        return;
    *params           = {};
    params->n_threads = sd_get_num_physical_cores();
}

void sd_video_upscale_params_init(sd_video_upscale_params_t* params) {
    if (!params)
        return;
    *params             = {};
    params->fps         = 24;
    params->seed        = 42;
    params->window_size = 16;
}

video_upscaler_ctx_t* new_video_upscaler_ctx(const sd_video_upscaler_ctx_params_t* params) {
    if (!params)
        return nullptr;
    try {
        auto ctx = std::make_unique<video_upscaler_ctx_t>();
        if (!ctx->upscaler.init(*params))
            return nullptr;
        return ctx.release();
    } catch (const std::exception& error) {
        LOG_ERROR("video upscaler initialization failed: %s", error.what());
        return nullptr;
    }
}

bool upscale_video(video_upscaler_ctx_t* ctx, const sd_video_upscale_params_t* params, sd_video_generation_result_t** result_out) {
    if (!result_out)
        return false;
    *result_out = nullptr;
    if (!ctx || !params)
        return false;
    try {
        return ctx->upscaler.upscale(*params, result_out);
    } catch (const std::exception& error) {
        LOG_ERROR("video upscale failed: %s", error.what());
        return false;
    }
}

void free_video_upscaler_ctx(video_upscaler_ctx_t* ctx) {
    delete ctx;
}

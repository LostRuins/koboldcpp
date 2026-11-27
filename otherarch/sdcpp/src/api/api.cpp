#include "api.h"
#include "options.h"
#include "result.h"
#include "tensor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "core/ggml_extend.h"
#include "core/option_types.h"
#include "core/util.h"
#include "pipeline/diffusion_engine.h"
#include "pipeline/generation.h"

static thread_local char last_error[512] = {};

void sd::api::clear_error() {
    last_error[0] = '\0';
}

sd_status_t sd::api::error(sd_status_t status, const char* key, const char* reason) {
    snprintf(last_error, sizeof(last_error), "%s%s%s", key ? key : "", key ? ": " : "", reason);
    return status;
}

const char* sd_get_last_error(void) {
    return last_error;
}

const char* sd_type_name(enum sd_type_t type) {
    if (type == SD_TYPE_F8_E4M3) {
        return "f8_e4m3";
    }
    if (type == SD_TYPE_F8_E5M2) {
        return "f8_e5m2";
    }
    const auto ggml_type = sd_type_to_ggml_type(type);
    if (ggml_type != GGML_TYPE_COUNT) {
        return ggml_type_name(ggml_type);
    }
    return "NONE";
}

enum sd_type_t str_to_sd_type(const char* str) {
    if (!strcmp(str, "f8_e4m3")) {
        return SD_TYPE_F8_E4M3;
    }
    if (!strcmp(str, "f8_e5m2")) {
        return SD_TYPE_F8_E5M2;
    }
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

sd_status_t sd_set_preview_callback(sd_preview_cb_t cb, const char* mode, int interval, bool denoised, bool noisy, void* data) {
    sd::api::clear_error();
    if (!mode)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "preview.mode", "mode is required");
    const auto parsed = str_to_preview(mode);
    if (parsed == PREVIEW_COUNT)
        return sd::api::error(SD_STATUS_INVALID_ARGUMENT, "preview.mode", "unknown preview mode");
    sd_set_preview_callback_impl(cb, parsed, interval, denoised, noisy, data);
    return SD_STATUS_OK;
}

void sd_set_prompt_image_callback(sd_ctx_t* ctx, sd_prompt_image_cb_t cb, void* data) {
    if (ctx == nullptr || ctx->sd == nullptr) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(ctx->sd->execution_mutex);
    if (!ctx->sd->executing_) {
        ctx->sd->prompt_image_callback      = cb;
        ctx->sd->prompt_image_callback_data = data;
    }
}

void sd_set_generation_progress_callback(sd_ctx_t* ctx, sd_generation_progress_cb_t cb, void* data) {
    if (ctx == nullptr || ctx->sd == nullptr) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(ctx->sd->execution_mutex);
    if (!ctx->sd->executing_) {
        ctx->sd->generation_progress.set_callback(cb, data);
    }
}

static sd_ctx_t* create_context(const sd::ContextConfig& params) {
    auto engine = std::make_unique<StableDiffusionGGML>();
    if (!engine->init(&params))
        return nullptr;
    auto context = std::make_unique<sd_ctx_t>();
    context->sd  = engine.release();
    return context.release();
}

sd_status_t sd_ctx_create(const sd_ctx_opt_t* params, sd_ctx_t** ctx_out) {
    if (ctx_out)
        *ctx_out = nullptr;
    return sd::api::call([&] {
        if (!ctx_out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "ctx_out is required");
        sd::OptConfig config;
        auto status         = sd::api::resolve(params, sd::api::Context, config);
        const auto& context = config.context;
        if (status != SD_STATUS_OK)
            return status;
        *ctx_out = create_context(context);
        return *ctx_out ? SD_STATUS_OK : sd::api::error(SD_STATUS_LOAD_FAILED, nullptr, "context creation failed; see log for details");
    },
                         SD_STATUS_LOAD_FAILED);
}

void sd_ctx_free(sd_ctx_t* sd_ctx) {
    if (!sd_ctx)
        return;
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    delete sd_ctx;
}

SD_API void sd_cancel_generation(sd_ctx_t* sd_ctx, enum sd_cancel_mode_t mode) {
    if (sd_ctx && sd_ctx->sd) {
        if (mode < SD_CANCEL_ALL || mode > SD_CANCEL_RESET) {
            mode = SD_CANCEL_ALL;
        }
        sd_ctx->sd->set_cancel_flag(mode);
    }
}

SD_API bool sd_ctx_supports_image_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_version_supports_image_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_supports_video_generation(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (sd_ctx->sd->config_->animatediff_loaded && sd_version_supports_animatediff(sd_ctx->sd->version)) {
        return true;
    }
    return sd_version_supports_video_generation(sd_ctx->sd->version);
}

SD_API bool sd_ctx_load_control_net(sd_ctx_t* sd_ctx, const char* path) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || path == nullptr) {
        return false;
    }
    return sd_ctx->sd->load_control_net_from_file(path);
}

SD_API bool sd_ctx_unload_control_net(sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    return sd_ctx->sd->unload_control_net();
}

SD_API bool sd_ctx_has_control_net(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return false;
    }
    if (sd_ctx->sd->control_net != nullptr) {
        return true;
    }
    // z-image keeps control net weights inside the diffusion runner, registered
    // under ModelComponent::ControlNet instead of a separate control_net runner.
    return sd_ctx->sd->model_manager != nullptr &&
           sd_ctx->sd->model_manager->registered_params_size({ModelComponent::ControlNet}) > 0;
}

const char* sd_get_model_version_name(const sd_ctx_t* sd_ctx) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr || sd_ctx->sd->version >= VERSION_COUNT) {
        return "Unknown";
    }
    return sd_version_name(sd_ctx->sd->version);
}

sd_status_t sd_generate_image(sd_ctx_t* ctx, const sd_img_gen_opt_t* params, sd_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    return sd::api::call([&] {
        if (!result_out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result_out is required");
        sd::OptConfig config;
        auto status       = sd::api::resolve(params, sd::api::Image, config);
        const auto& image = config.image;
        if (status != SD_STATUS_OK)
            return status;
        sd::ImageGenerationResult* images = nullptr;
        status                            = sd::generate_image(ctx, image, &images, params);
        if (status != SD_STATUS_OK)
            return status;
        return sd::make_image_result(images, result_out) ? SD_STATUS_OK
                                                         : sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate result");
    },
                         SD_STATUS_GENERATION_FAILED);
}

sd_status_t sd_generate_video(sd_ctx_t* ctx, const sd_vid_gen_opt_t* params, sd_result_t** result_out) {
    if (result_out)
        *result_out = nullptr;
    return sd::api::call([&] {
        if (!result_out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "result_out is required");
        sd::OptConfig config;
        auto status       = sd::api::resolve(params, sd::api::Video, config);
        const auto& video = config.video;
        if (status != SD_STATUS_OK)
            return status;
        sd::VideoGenerationResult* frames = nullptr;
        status                            = sd::generate_video(ctx, video, &frames, params);
        if (status != SD_STATUS_OK)
            return status;
        return sd::make_video_result(frames, result_out) ? SD_STATUS_OK
                                                         : sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate result");
    },
                         SD_STATUS_GENERATION_FAILED);
}

sd_status_t sd_preprocess_canny(const sd_tensor_t* image, float high_threshold, float low_threshold, float weak, float strong, bool inverse, sd_tensor_t** image_out) {
    if (image_out)
        *image_out = nullptr;
    return sd::api::call([&] {
        if (!image || !image_out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "image and image_out are required");
        sd::api::image_view(*image);
        auto output = sd::api::allocate_tensor(image->type, image->shape);
        memcpy(output.get()->mutable_data(), image->data(), image->bytes);
        if (!sd::preprocess_canny(sd::api::image_view(*output.get()), high_threshold, low_threshold, weak, strong, inverse))
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "Canny preprocessing failed");
        *image_out = sd_tensor_retain(output.get());
        return SD_STATUS_OK;
    });
}

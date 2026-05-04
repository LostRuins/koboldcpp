#include "upscaler.h"
#include "ggml_extend.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"

UpscalerGGML::UpscalerGGML(int n_threads,
                           bool direct,
                           int tile_size)
    : n_threads(n_threads),
      direct(direct),
      tile_size(tile_size) {
}

bool UpscalerGGML::load_from_file(const std::string& esrgan_path,
                                  bool offload_params_to_cpu,
                                  int n_threads) {
    ggml_log_set(ggml_log_callback_default, nullptr);
#ifdef SD_USE_CUDA
    LOG_DEBUG("Using CUDA backend");
    backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
    LOG_DEBUG("Using Metal backend");
    backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
    LOG_DEBUG("Using Vulkan backend");
    backend = ggml_backend_vk_init(0);
#endif
#ifdef SD_USE_OPENCL
    LOG_DEBUG("Using OpenCL backend");
    backend = ggml_backend_opencl_init();
#endif
#ifdef SD_USE_SYCL
    LOG_DEBUG("Using SYCL backend");
    backend = ggml_backend_sycl_init(0);
#endif
    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(esrgan_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
    }
    model_loader.set_wtype_override(model_data_type);
    if (!backend) {
        LOG_DEBUG("Using CPU backend");
        backend = ggml_backend_cpu_init();
    }
    LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
    esrgan_upscaler = std::make_shared<ESRGAN>(backend, offload_params_to_cpu, tile_size, model_loader.get_tensor_storage_map());
    if (direct) {
        esrgan_upscaler->set_conv2d_direct_enabled(true);
    }
    if (!esrgan_upscaler->load_from_file(esrgan_path, n_threads)) {
        return false;
    }
    return true;
}

sd::Tensor<float> UpscalerGGML::upscale_tensor(const sd::Tensor<float>& input_tensor) {
    sd::Tensor<float> upscaled;
    if (tile_size <= 0 || (input_tensor.shape()[0] <= tile_size && input_tensor.shape()[1] <= tile_size)) {
        upscaled = esrgan_upscaler->compute(n_threads, input_tensor);
    } else {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) -> sd::Tensor<float> {
            auto output_tile = esrgan_upscaler->compute(n_threads, input_tile);
            if (output_tile.empty()) {
                LOG_ERROR("esrgan compute failed while processing a tile");
                return {};
            }
            return output_tile;
        };

        upscaled = process_tiles_2d(input_tensor,
                                    static_cast<int>(input_tensor.shape()[0] * esrgan_upscaler->scale),
                                    static_cast<int>(input_tensor.shape()[1] * esrgan_upscaler->scale),
                                    esrgan_upscaler->scale,
                                    tile_size,
                                    tile_size,
                                    0.25f,
                                    false,
                                    false,
                                    on_processing);
    }
    esrgan_upscaler->free_compute_buffer();
    if (upscaled.empty()) {
        LOG_ERROR("esrgan compute failed");
        return {};
    }
    return upscaled;
}

sd_image_t UpscalerGGML::upscale(sd_image_t input_image, uint32_t upscale_factor) {
    // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
    sd_image_t upscaled_image = {0, 0, 0, nullptr};
    int output_width          = (int)input_image.width * esrgan_upscaler->scale;
    int output_height         = (int)input_image.height * esrgan_upscaler->scale;
    LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
             input_image.width, input_image.height, output_width, output_height);

    sd::Tensor<float> input_tensor = sd_image_to_tensor(input_image);
    sd::Tensor<float> upscaled;
    int64_t t0 = ggml_time_ms();
    upscaled   = upscale_tensor(input_tensor);
    if (upscaled.empty()) {
        return upscaled_image;
    }
    sd_image_t upscaled_data = tensor_to_sd_image(upscaled);
    int64_t t3               = ggml_time_ms();
    LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
    upscaled_image = upscaled_data;
    return upscaled_image;
}

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = nullptr;
};

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 bool offload_params_to_cpu,
                                 bool direct,
                                 int n_threads,
                                 int tile_size) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size);
    if (upscaler_ctx->upscaler == nullptr) {
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, offload_params_to_cpu, n_threads)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
        free(upscaler_ctx);
        return nullptr;
    }
    return upscaler_ctx;
}

sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
    return upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}

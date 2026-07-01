#include "upscaler.h"
#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "model_loader.h"
#include "stable-diffusion.h"

#include <cstdlib>
#include <utility>

UpscalerGGML::UpscalerGGML(int n_threads,
                           bool direct,
                           int tile_size,
                           std::string backend_spec,
                           std::string params_backend_spec)
    : n_threads(n_threads),
      direct(direct),
      tile_size(tile_size),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {
}

UpscalerGGML::~UpscalerGGML() {
    // ModelManager holds raw ggml tensor pointers owned by the runner context.
    model_manager.reset();
    esrgan_upscaler.reset();
}

void UpscalerGGML::set_max_graph_vram_bytes(size_t max_vram_bytes) {
    max_graph_vram_bytes = max_vram_bytes;
    if (esrgan_upscaler) {
        esrgan_upscaler->set_max_graph_vram_bytes(max_vram_bytes);
    }
}

void UpscalerGGML::set_stream_layers_enabled(bool enabled) {
    stream_layers_enabled = enabled;
    if (esrgan_upscaler) {
        esrgan_upscaler->set_stream_layers_enabled(enabled);
    }
}

bool UpscalerGGML::load_from_file(const std::string& esrgan_path,
                                  int n_threads) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    std::string error;
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              &error)) {
        LOG_ERROR("upscaler backend config failed: %s", error.c_str());
        return false;
    }
    auto backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto params_backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto ensure_backend_pair = [&](SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    };
    if (!ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return false;
    }

    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);

    ModelLoader& model_loader = model_manager->loader();
    if (!model_loader.init_from_file_and_convert_name(esrgan_path, "", VERSION_ESRGAN)) {
        LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        return false;
    }
    model_loader.set_wtype_override(model_data_type);
    LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
    esrgan_upscaler = std::make_shared<ESRGAN>(backend_for(SDBackendModule::UPSCALER),
                                               model_loader.get_tensor_storage_map(),
                                               model_manager);
    if (esrgan_upscaler == nullptr || esrgan_upscaler->rrdb_net == nullptr) {
        LOG_ERROR("init esrgan model from metadata failed: '%s'", esrgan_path.c_str());
        return false;
    }
    esrgan_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    esrgan_upscaler->set_stream_layers_enabled(stream_layers_enabled);
    if (direct) {
        esrgan_upscaler->set_conv2d_direct_enabled(true);
    }

    std::map<std::string, ggml_tensor*> tensors;
    esrgan_upscaler->get_param_tensors(tensors);
    if (!model_manager->register_param_tensors("ESRGAN",
                                               std::move(tensors),
                                               backend_manager.params_backend_is_disk(SDBackendModule::UPSCALER) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
                                               backend_for(SDBackendModule::UPSCALER),
                                               params_backend_for(SDBackendModule::UPSCALER)) ||
        !model_manager->validate_registered_tensors()) {
        LOG_ERROR("register esrgan tensors with model manager failed");
        return false;
    }
    return true;
}

sd::Tensor<float> UpscalerGGML::upscale_tensor(const sd::Tensor<float>& input_tensor) {
    sd::Tensor<float> upscaled;
    const int scale = esrgan_upscaler->config.scale;
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
                                    static_cast<int>(input_tensor.shape()[0] * scale),
                                    static_cast<int>(input_tensor.shape()[1] * scale),
                                    scale,
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
    const int scale           = esrgan_upscaler->config.scale;
    int output_width          = (int)input_image.width * scale;
    int output_height         = (int)input_image.height * scale;
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
                                 bool direct,
                                 int n_threads,
                                 int tile_size,
                                 const char* backend,
                                 const char* params_backend) {
    upscaler_ctx_t* upscaler_ctx = (upscaler_ctx_t*)malloc(sizeof(upscaler_ctx_t));
    if (upscaler_ctx == nullptr) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    upscaler_ctx->upscaler = new UpscalerGGML(n_threads, direct, tile_size, SAFE_STR(backend), SAFE_STR(params_backend));
    if (upscaler_ctx->upscaler == nullptr) {
        return nullptr;
    }

    if (!upscaler_ctx->upscaler->load_from_file(esrgan_path, n_threads)) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
        free(upscaler_ctx);
        return nullptr;
    }
    return upscaler_ctx;
}

bool upscale(upscaler_ctx_t* upscaler_ctx,
             sd_image_t input_image,
             uint32_t upscale_factor,
             sd_image_t** images_out,
             int* num_images_out) {
    if (images_out != nullptr) {
        *images_out = nullptr;
    }
    if (num_images_out != nullptr) {
        *num_images_out = 0;
    }
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return false;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return false;
    }

    result_images[0] = upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
    if (result_images[0].data == nullptr) {
        free(result_images);
        return false;
    }

    if (num_images_out != nullptr) {
        *num_images_out = 1;
    }
    if (images_out != nullptr) {
        *images_out = result_images;
    } else {
        free_sd_images(result_images, 1);
    }
    return true;
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->config.scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}

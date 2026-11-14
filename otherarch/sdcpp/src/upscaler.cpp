#include "upscaler.h"
#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "generation_info.h"
#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/mosr.hpp"
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
      tensor_executor(n_threads > 0 ? n_threads : sd_get_num_physical_cores()),
      direct(direct),
      tile_size(tile_size),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {}

UpscalerGGML::~UpscalerGGML() = default;

void UpscalerGGML::set_max_graph_vram_bytes(size_t max_vram_bytes) {
    max_graph_vram_bytes = max_vram_bytes;
    if (model != nullptr) {
        model->runner()->set_max_graph_vram_bytes(max_vram_bytes);
    }
}

int UpscalerGGML::get_scale() {
    return model != nullptr ? model->scale() : 1;
}

bool UpscalerGGML::load_from_file(const std::string& path, int n_threads) {
    sd::ParallelScope tensor_scope(&tensor_executor);
    kcpp_sd_ggml_log_set();
    model.reset();
    std::string error;
    if (!backend_manager.init(backend_spec.c_str(), params_backend_spec.c_str(), nullptr, &error)) {
        LOG_ERROR("upscaler backend config failed: %s", error.c_str());
        return false;
    }
    model_manager = std::make_shared<ModelManager>();
    model_manager->set_n_threads(n_threads);
    model_manager->set_enable_mmap(false);
    model = sd::pipeline::load_image_upscale_model(model_manager, backend_manager, path,
                                                   max_graph_vram_bytes, direct, model_data_type);
    return model != nullptr;
}

sd::Tensor<float> UpscalerGGML::upscale_tensor(const sd::Tensor<float>& input_tensor) {
    sd::ParallelScope tensor_scope(&tensor_executor);
    if (model == nullptr) {
        return {};
    }
    return sd::pipeline::upscale_image_tensor(*model, input_tensor, n_threads, tile_size);
}

sd_image_t UpscalerGGML::upscale(sd_image_t input_image, uint32_t upscale_factor) {
    sd::ParallelScope tensor_scope(&tensor_executor);
    // upscale_factor, unused for RealESRGAN_x4plus_anime_6B.pth
    sd_image_t upscaled_image = {0, 0, 0, nullptr};
    const int scale           = get_scale();
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
             sd_image_generation_result_t** result_out) try {
    if (result_out)
        *result_out = nullptr;
    if (!result_out)
        return false;
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return false;
    }

    sd_image_t* result_images = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (result_images == nullptr) {
        return false;
    }

    auto release_images = [](sd_image_t* images) { free_sd_images(images, 1); };
    std::unique_ptr<sd_image_t, decltype(release_images)> image_owner(result_images, release_images);
    const std::vector<nlohmann::json> infos{sd::upscale_generation_info(upscale_factor)};
    result_images[0] = upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
    if (!result_images[0].data)
        return false;

    return sd::make_image_generation_result(image_owner.release(), 1, infos, result_out);
} catch (const std::exception& e) {
    LOG_ERROR("upscale failed: %s", e.what());
    return false;
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->get_scale();
}

int get_upscaler_model_scale(const char* model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return 0;
    }
    try {
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(model_path, "", VERSION_ESRGAN)) {
            return 0;
        }
        const auto& tensors = loader.get_tensor_storage_map();
        if (tensors.count("gblocks.0.weight") > 0) {
            MoSRConfig config = MoSRConfig::detect_from_weights(tensors);
            return config.is_valid() ? config.scale : 0;
        }
        auto first = tensors.find("conv_first.weight");
        auto last  = tensors.find("conv_last.weight");
        if (first == tensors.end() || last == tensors.end() ||
            tensors.count("body.0.rdb1.conv1.weight") == 0 ||
            first->second.n_dims != 4 || last->second.n_dims != 4 ||
            first->second.ne[2] != 3 || last->second.ne[3] != 3) {
            return 0;
        }
        return ESRGANConfig::detect_from_weights(tensors).scale;
    } catch (const std::exception&) {
        return 0;
    }
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (!upscaler_ctx)
        return;
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    free(upscaler_ctx);
}

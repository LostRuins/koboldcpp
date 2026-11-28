#include "upscaler.h"
#include "api/options.h"
#include "api/result.h"
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

sd::Image UpscalerGGML::upscale(sd::Image input_image) {
    sd::ParallelScope tensor_scope(&tensor_executor);
    sd::Image upscaled_image = {0, 0, 0, nullptr};
    const int scale          = get_scale();
    int output_width         = (int)input_image.width * scale;
    int output_height        = (int)input_image.height * scale;
    LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
             input_image.width, input_image.height, output_width, output_height);

    sd::Tensor<float> input_tensor = sd_image_to_tensor(input_image);
    sd::Tensor<float> upscaled;
    int64_t t0 = ggml_time_ms();
    upscaled   = upscale_tensor(input_tensor);
    if (upscaled.empty()) {
        return upscaled_image;
    }
    sd::Image upscaled_data = tensor_to_sd_image(upscaled);
    int64_t t3              = ggml_time_ms();
    LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
    upscaled_image = upscaled_data;
    return upscaled_image;
}

struct sd_upscaler {
    std::unique_ptr<UpscalerGGML> upscaler;
};

sd_status_t sd_upscaler_create(const sd_upscaler_opt_t* opt, sd_upscaler_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "out is required");
        sd::OptConfig config;
        auto status = sd::api::resolve(opt, sd::api::Upscaler, config);
        if (status != SD_STATUS_OK)
            return status;
        auto result      = std::make_unique<sd_upscaler_t>();
        const auto& ctx  = config.context;
        result->upscaler = std::make_unique<UpscalerGGML>(ctx.n_threads, config.post.conv_direct, config.post.tile_size,
                                                          SAFE_STR(ctx.backend), SAFE_STR(ctx.params_backend));
        if (!result->upscaler->load_from_file(ctx.model_path, ctx.n_threads))
            return sd::api::error(SD_STATUS_LOAD_FAILED, "model_path", "could not load upscaler");
        *out = result.release();
        return SD_STATUS_OK;
    },
                         SD_STATUS_LOAD_FAILED);
}

void sd_upscaler_free(sd_upscaler_t* upscaler) {
    delete upscaler;
}

sd_status_t sd_upscale(sd_upscaler_t* upscaler, const sd_upscale_opt_t* opt, sd_result_t** out) {
    if (out)
        *out = nullptr;
    return sd::api::call([&] {
        if (!out)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "out is required");
        sd::OptConfig config;
        auto status = sd::api::resolve(opt, sd::api::Upscale, config);
        if (status != SD_STATUS_OK)
            return status;
        if (!upscaler || !upscaler->upscaler)
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "upscaler is required");
        const auto& image = config.post.image;
        const auto scale  = upscaler->upscaler->get_scale();
        if (scale <= 0 || image.width > uint32_t(INT32_MAX / scale) || image.height > uint32_t(INT32_MAX / scale))
            return sd::api::error(SD_STATUS_OUT_OF_RANGE, "image", "upscaled dimensions exceed the supported range");
        auto release = [](sd::Image* images) { sd::free_images(images, 1); };
        std::unique_ptr<sd::Image, decltype(release)> images(static_cast<sd::Image*>(calloc(1, sizeof(sd::Image))), release);
        if (!images)
            return sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate result");
        images.get()[0] = upscaler->upscaler->upscale(image);
        if (!images->data)
            return sd::api::error(SD_STATUS_GENERATION_FAILED, nullptr, "upscale failed");
        const std::vector<nlohmann::json> infos{sd::upscale_generation_info(scale)};
        return sd::make_image_result(images.release(), 1, infos, out) ? SD_STATUS_OK
                                                                      : sd::api::error(SD_STATUS_OUT_OF_MEMORY, nullptr, "could not allocate result metadata");
    },
                         SD_STATUS_GENERATION_FAILED);
}

int sd_upscaler_get_scale(const sd_upscaler_t* upscaler) {
    sd::api::clear_error();
    if (!upscaler || !upscaler->upscaler) {
        sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "upscaler is required");
        return 0;
    }
    return upscaler->upscaler->get_scale();
}

int sd_upscaler_probe_scale(const char* model_path) {
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

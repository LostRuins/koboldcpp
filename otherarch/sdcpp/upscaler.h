#ifndef __SD_UPSCALER_H__
#define __SD_UPSCALER_H__

#include "esrgan.hpp"
#include "stable-diffusion.h"
#include "tensor.hpp"

#include <memory>
#include <string>

struct UpscalerGGML {
    ggml_backend_t backend    = nullptr;  // general backend
    ggml_type model_data_type = GGML_TYPE_F16;
    std::shared_ptr<ESRGAN> esrgan_upscaler;
    std::string esrgan_path;
    int n_threads;
    bool direct                 = false;
    int tile_size               = 128;
    size_t max_graph_vram_bytes = 0;

    UpscalerGGML(int n_threads,
                 bool direct   = false,
                 int tile_size = 128);

    bool load_from_file(const std::string& esrgan_path,
                        bool offload_params_to_cpu,
                        int n_threads);
    void set_max_graph_vram_bytes(size_t max_vram_bytes);
    sd::Tensor<float> upscale_tensor(const sd::Tensor<float>& input_tensor);
    sd_image_t upscale(sd_image_t input_image, uint32_t upscale_factor);
};

#endif  // __SD_UPSCALER_H__

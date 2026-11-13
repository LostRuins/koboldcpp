#ifndef __SD_PIPELINE_IMAGE_UPSCALE_H__
#define __SD_PIPELINE_IMAGE_UPSCALE_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "ggml.h"

class ModelManager;
class SDBackendManager;
struct GGMLRunner;
struct ESRGAN;
struct MoSR;

namespace sd::pipeline {
    class ImageUpscaleModel {
        friend std::unique_ptr<ImageUpscaleModel> load_image_upscale_model(
            const std::shared_ptr<ModelManager>&,
            SDBackendManager&,
            const std::string&,
            size_t,
            bool,
            ggml_type);
        friend Tensor<float> upscale_image_tensor(ImageUpscaleModel&, const Tensor<float>&, int, int);

        std::weak_ptr<ModelManager> manager_;
        std::unique_ptr<ESRGAN> esrgan_;
        std::unique_ptr<MoSR> mosr_;
        std::vector<ggml_tensor*> weights_;
        uint64_t source_file_ = 0;

        explicit ImageUpscaleModel(const std::shared_ptr<ModelManager>& manager);

    public:
        ~ImageUpscaleModel();
        GGMLRunner* runner() const;
        int scale() const;
    };

    std::unique_ptr<ImageUpscaleModel> load_image_upscale_model(
        const std::shared_ptr<ModelManager>& manager,
        SDBackendManager& backends,
        const std::string& path,
        size_t max_vram_bytes = 0,
        bool conv_direct      = false,
        ggml_type weight_type = GGML_TYPE_F16);

    Tensor<float> upscale_image_tensor(ImageUpscaleModel& model,
                                       const Tensor<float>& input,
                                       int n_threads,
                                       int tile_size);
}

#endif  // __SD_PIPELINE_IMAGE_UPSCALE_H__

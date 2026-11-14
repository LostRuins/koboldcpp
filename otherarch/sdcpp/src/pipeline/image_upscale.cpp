#include "pipeline/image_upscale.h"

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/upscaler/esrgan.hpp"
#include "model/upscaler/mosr.hpp"
#include "model_manager.h"
#include "runtime/tiling.h"

namespace sd::pipeline {
    ImageUpscaleModel::ImageUpscaleModel(const std::shared_ptr<ModelManager>& manager)
        : manager_(manager) {}

    ImageUpscaleModel::~ImageUpscaleModel() {
        if (auto* value = runner()) {
            value->runner_end();
        }
        if (auto manager = manager_.lock()) {
            GGML_ASSERT(manager->unregister_param_tensors(weights_));
            if (source_file_ != 0) {
                GGML_ASSERT(manager->unregister_model_source(source_file_));
            }
        }
    }

    GGMLRunner* ImageUpscaleModel::runner() const {
        return mosr_ != nullptr ? static_cast<GGMLRunner*>(mosr_.get()) : esrgan_.get();
    }

    int ImageUpscaleModel::scale() const {
        if (mosr_ != nullptr) {
            return mosr_->config.scale;
        }
        return esrgan_ != nullptr ? esrgan_->config.scale : 1;
    }

    std::unique_ptr<ImageUpscaleModel> load_image_upscale_model(
        const std::shared_ptr<ModelManager>& manager,
        SDBackendManager& backends,
        const std::string& path,
        size_t max_vram_bytes,
        bool conv_direct,
        ggml_type weight_type) {
        if (manager == nullptr) {
            return nullptr;
        }
        const auto backend        = backends.runtime_backend(SDBackendModule::UPSCALER);
        const auto params_backend = backends.params_backend(SDBackendModule::UPSCALER);
        if (backend == nullptr || params_backend == nullptr) {
            LOG_ERROR("failed to initialize upscaler backends");
            return nullptr;
        }
        ModelLoader loader;
        if (!loader.init_from_file_and_convert_name(path, "", VERSION_ESRGAN)) {
            LOG_ERROR("init upscaler model loader failed: '%s'", path.c_str());
            return nullptr;
        }
        loader.set_wtype_override(weight_type);
        const auto tensors  = loader.get_tensor_storage_map();
        auto model          = std::unique_ptr<ImageUpscaleModel>(new ImageUpscaleModel(manager));
        model->source_file_ = manager->register_model_source(std::move(loader));
        if (model->source_file_ == 0) {
            return nullptr;
        }
        std::map<std::string, ggml_tensor*> weights;
        if (tensors.count("gblocks.0.weight") != 0) {
            model->mosr_ = std::make_unique<MoSR>(backend, tensors, manager);
            if (model->mosr_->mosr_net == nullptr) {
                LOG_ERROR("unsupported MoSR weights: '%s'", path.c_str());
                return nullptr;
            }
            model->mosr_->get_param_tensors(weights);
        } else {
            model->esrgan_ = std::make_unique<ESRGAN>(backend, tensors, manager);
            if (model->esrgan_->rrdb_net == nullptr) {
                LOG_ERROR("unsupported ESRGAN weights: '%s'", path.c_str());
                return nullptr;
            }
            model->esrgan_->get_param_tensors(weights);
        }
        for (const auto& entry : weights) {
            model->weights_.push_back(entry.second);
        }
        const auto mode = backends.params_backend_is_disk(SDBackendModule::UPSCALER)
                              ? ModelManager::ResidencyMode::Disk
                              : ModelManager::ResidencyMode::ParamBackend;
        if (!manager->register_param_tensors(ModelComponent::Upscaler, std::move(weights), mode,
                                             backend, params_backend, nullptr, false, false, nullptr,
                                             model->source_file_, VERSION_ESRGAN) ||
            !manager->validate_param_tensors(model->weights_)) {
            LOG_ERROR("register upscaler weights failed: '%s'", path.c_str());
            return nullptr;
        }
        model->runner()->set_max_graph_vram_bytes(max_vram_bytes);
        model->runner()->set_conv2d_direct_enabled(conv_direct);
        LOG_INFO("Upscaler architecture: %s, weight type: %s", model->runner()->get_desc().c_str(), ggml_type_name(weight_type));
        return model;
    }

    Tensor<float> upscale_image_tensor(ImageUpscaleModel& model, const Tensor<float>& input, int n_threads, int tile_size) {
        if (input.empty() || input.dim() != 4 || (input.shape()[2] != 3 && input.shape()[2] != 4)) {
            LOG_ERROR("upscaler expects a 4D RGB or RGBA image tensor");
            return {};
        }
        struct RunnerEnd {
            GGMLRunner* runner;
            ~RunnerEnd() { runner->runner_end(); }
        } cleanup{model.runner()};
        const bool has_alpha = input.shape()[2] == 4;
        auto rgb             = has_alpha ? sd::ops::slice(input, 2, 0, 3) : input;
        const auto compute   = [&](const Tensor<float>& tensor) {
            return model.mosr_ != nullptr ? model.mosr_->compute(n_threads, tensor) : model.esrgan_->compute(n_threads, tensor);
        };
        const int scale = model.scale();
        Tensor<float> output;
        if (tile_size <= 0 || (input.shape()[0] <= tile_size && input.shape()[1] <= tile_size)) {
            output = compute(rgb);
        } else {
            output = process_tiles_2d(rgb,
                                      static_cast<int>(input.shape()[0] * scale),
                                      static_cast<int>(input.shape()[1] * scale),
                                      scale, tile_size, tile_size, 0.25f, false, false, compute);
        }
        if (output.empty()) {
            LOG_ERROR("upscaler compute failed");
            return {};
        }
        if (has_alpha) {
            auto alpha = sd::ops::slice(input, 2, 3, 4);
            auto shape = alpha.shape();
            shape[0]   = output.shape()[0];
            shape[1]   = output.shape()[1];
            alpha      = sd::ops::interpolate(alpha, shape, sd::ops::InterpolateMode::Bilinear);
            output     = sd::ops::concat(output, alpha, 2);
        }
        return output;
    }
}

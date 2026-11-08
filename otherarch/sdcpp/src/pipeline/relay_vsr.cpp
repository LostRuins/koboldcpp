#include "pipeline/video_upscale.h"

#include <climits>

#include "core/rng_philox.hpp"
#include "generation_info.h"
#include "model/diffusion/relay_vsr.hpp"
#include "model/upscaler/flash_vsr.hpp"
#include "name_conversion.h"

namespace sd {
    namespace pipeline {

        static sd::Tensor<float> prepare_vsr_frame(const sd_image_t& frame, int width, int height) {
            auto result = sd_image_to_tensor(frame, -1, -1, false);
            result /= 127.5f;
            result -= 1.f;
            if (width > int(frame.width)) {
                auto edge    = sd::ops::slice(result, 0, frame.width - 1, frame.width);
                auto padding = sd::zeros<float>({width - frame.width, frame.height, 3, 1}) + edge;
                result       = sd::ops::concat(result, padding, 0);
            }
            if (height > int(frame.height)) {
                auto edge    = sd::ops::slice(result, 1, frame.height - 1, frame.height);
                auto padding = sd::zeros<float>({width, height - frame.height, 3, 1}) + edge;
                result       = sd::ops::concat(result, padding, 1);
            }
            return result;
        }

        static sd::Tensor<float> prepare_vsr_lq(const sd::Tensor<float>& frame, int scale) {
            auto input = frame;
            if (scale == 4)
                input = sd::ops::interpolate(frame, {frame.shape()[0] * 4, frame.shape()[1] * 4, 3, 1}, sd::ops::InterpolateMode::Bilinear);
            const int patch = 4 * scale;
            const int width = int(input.shape()[0]), height = int(input.shape()[1]);
            input.reshape_({patch, width / patch, patch, height / patch, 3, 1});
            input.permute_({1, 3, 0, 2, 4, 5});
            input.reshape_({width / patch, height / patch, 3 * patch * patch, 1});
            return input;
        }

        class RelayVSRUpscaler : public VideoUpscaleModel {
            std::unique_ptr<RelayVSR::WanVSRRunner> wan_;
            std::unique_ptr<RelayVSR::FlashVSRRunner> flash_;
            int scale_     = 0;
            int n_threads_ = 1;

        public:
            bool init(VideoUpscaleRuntime& runtime, const String2TensorStorage& source_weights, int upscale_factor) override {
                if (upscale_factor != 2 && upscale_factor != 4) {
                    LOG_ERROR("RelayVSR requires an explicit upscale factor of 2 or 4 matching both checkpoints");
                    return false;
                }
                scale_             = upscale_factor;
                n_threads_         = runtime.n_threads;
                auto wan_backend   = runtime.backends.runtime_backend(SDBackendModule::DIFFUSION);
                auto flash_backend = runtime.backends.runtime_backend(SDBackendModule::VAE);
                auto wan_params    = runtime.backends.params_backend(SDBackendModule::DIFFUSION);
                auto flash_params  = runtime.backends.params_backend(SDBackendModule::VAE);
                String2TensorStorage weights;
                std::map<std::string, std::string> source_names;
                for (const auto& [name, storage] : source_weights) {
                    auto canonical = name;
                    if (starts_with(canonical, "diffusion."))
                        canonical = "wan." + canonical.substr(10);
                    else if (starts_with(canonical, "vae."))
                        canonical = "flash." + canonical.substr(4);
                    canonical = convert_relayvsr_tensor_name(canonical);
                    if (!source_names.emplace(canonical, name).second)
                        throw std::runtime_error("Duplicate RelayVSR weight name: " + canonical);
                    weights[canonical]      = storage;
                    weights[canonical].name = canonical;
                }
                wan_                = std::make_unique<RelayVSR::WanVSRRunner>(wan_backend, weights, upscale_factor, runtime.manager);
                flash_              = std::make_unique<RelayVSR::FlashVSRRunner>(flash_backend, weights, upscale_factor, runtime.manager);
                auto register_model = [&](auto& runner, ModelComponent component, SDBackendModule module, ggml_backend_t compute, ggml_backend_t params) {
                    std::map<std::string, ggml_tensor*> tensors, sources;
                    runner->get_param_tensors(tensors);
                    for (const auto& [name, tensor] : tensors) {
                        auto source = source_names.find(name);
                        if (source == source_names.end())
                            throw std::runtime_error("Missing RelayVSR tensor: " + name);
                        sources.emplace(source->second, tensor);
                        ggml_set_name(tensor, source->second.c_str());
                    }
                    runtime.configure_runner(*runner, module, true, module == SDBackendModule::DIFFUSION);
                    return runtime.manager->register_param_tensors(component, std::move(sources),
                                                                   runtime.backends.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
                                                                   compute, params);
                };
                if (!register_model(wan_, ModelComponent::Diffusion, SDBackendModule::DIFFUSION, wan_backend, wan_params) ||
                    !register_model(flash_, ModelComponent::VAE, SDBackendModule::VAE, flash_backend, flash_params) ||
                    !runtime.manager->validate_registered_tensors())
                    return false;
                return true;
            }

            void end() override {
                if (wan_)
                    wan_->runner_end();
                if (flash_)
                    flash_->runner_end();
            }

            bool upscale(const sd_video_upscale_params_t& params, sd_video_generation_result_t* result) override {
                const auto* frames    = params.frames;
                const int frame_count = params.frame_count, fps = params.fps, window_size = params.window_size;
                const int64_t seed = params.seed;
                if (window_size < 2) {
                    LOG_ERROR("RelayVSR requires a reference window of at least two frames");
                    return false;
                }
                if (frames[0].width > uint32_t(INT_MAX / scale_ - 32) || frames[0].height > uint32_t(INT_MAX / scale_ - 32))
                    return false;
                struct Reset {
                    RelayVSRUpscaler* model;
                    ~Reset() {
                        model->wan_->reset();
                        model->flash_->reset();
                    }
                } reset{this};
                wan_->reset();
                flash_->reset();
                const int stride    = 16 / scale_;
                const int alignment = 2 * stride;
                const int width     = (int(frames[0].width) + alignment - 1) / alignment * alignment;
                const int height    = (int(frames[0].height) + alignment - 1) / alignment * alignment;
                PhiloxRNG rng;
                rng.manual_seed(uint64_t(seed));
                auto keyframe = [&](int index, int slot) {
                    LOG_INFO("RelayVSR reference frame %d/%d", index + 1, frame_count);
                    auto input = prepare_vsr_frame(frames[index], width, height);
                    sd::Tensor<float> noise({width / stride, height / stride, 1, 48});
                    auto values = rng.randn(uint32_t(noise.numel()));
                    std::copy(values.begin(), values.end(), noise.data());
                    auto latent = wan_->compute(n_threads_, noise, prepare_vsr_lq(input, scale_));
                    if (latent.empty())
                        return false;
                    latent.reshape_({width / stride, height / stride, 48, 1});
                    return !flash_->compute(n_threads_, latent, slot).empty();
                };
                int start_slot = 0;
                if (!keyframe(0, start_slot))
                    return false;
                for (int first = 0; first < frame_count;) {
                    const int last = first + std::min(window_size - 1, frame_count - 1 - first);
                    int end_slot   = last == first ? start_slot : 1 - start_slot;
                    if (last != first && !keyframe(last, end_slot))
                        return false;
                    flash_->history = false;
                    for (int i = first; i <= last; ++i) {
                        auto input    = prepare_vsr_frame(frames[i], width, height);
                        auto residual = flash_->compute(n_threads_, input, -1, start_slot, end_slot);
                        if (residual.empty())
                            return false;
                        if (first != 0 && i == first)
                            continue;
                        auto output   = sd::ops::interpolate(input, {width * scale_, height * scale_, 3, 1}, sd::ops::InterpolateMode::Bicubic);
                        auto& image   = result->frames[i];
                        image.width   = frames[i].width * scale_;
                        image.height  = frames[i].height * scale_;
                        image.channel = 3;
                        output += residual;
                        output += 1.f;
                        output *= 127.5f;
                        output = sd::ops::slice(sd::ops::slice(output, 0, 0, image.width), 1, 0, image.height);
                        if (!std::all_of(output.values().begin(), output.values().end(), [](float value) { return std::isfinite(value); }))
                            throw std::runtime_error("RelayVSR produced non-finite pixels");
                        output = sd::ops::clamp(output, 0.f, 255.f);
                        output.permute_({2, 0, 1, 3});
                        image.data = static_cast<uint8_t*>(malloc(size_t(image.width) * image.height * 3));
                        if (!image.data)
                            throw std::bad_alloc();
                        std::transform(output.values().begin(), output.values().end(), image.data, [](float value) { return uint8_t(value); });
                        LOG_INFO("RelayVSR output frame %d/%d", i + 1, frame_count);
                        if (auto progress = sd_get_progress_callback())
                            progress(i + 1, frame_count, 0.f, sd_get_progress_callback_data());
                    }
                    if (last == frame_count - 1)
                        break;
                    first      = last;
                    start_slot = end_slot;
                }
                auto info                      = sd::upscale_generation_info(scale_);
                info["kind"]                   = "video";
                info["upscale"]["model"]       = "relayvsr";
                info["upscale"]["window_size"] = window_size;
                info["seed"]                   = seed;
                info["fps"]                    = fps;
                info["video_frames"]           = frame_count;
                result->generation_info        = sd::copy_generation_info(info);
                if (!result->generation_info)
                    throw std::bad_alloc();
                return true;
            }
        };

        bool is_relay_vsr_model(const String2TensorStorage& weights) {
            return (weights.count("diffusion.positive_conditioning") || weights.count("diffusion.prompt_embeds")) &&
                   (weights.count("diffusion.lq_proj_in.conv1.weight") || weights.count("diffusion.lq_proj.conv1.weight"));
        }

        std::unique_ptr<VideoUpscaleModel> create_relay_vsr_upscaler() {
            return std::make_unique<RelayVSRUpscaler>();
        }

    }  // namespace pipeline
}  // namespace sd

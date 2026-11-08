#include "pipeline/video_upscale.h"

#include <climits>

#include "core/rng_philox.hpp"
#include "generation_info.h"
#include "model/diffusion/seedvr2.hpp"
#include "model/vae/seedvr2_vae.hpp"

namespace sd {
    namespace pipeline {

        class SeedVR2Upscaler : public VideoUpscaleModel {
            std::unique_ptr<SeedVR2::SeedVR2Runner> dit_;
            std::unique_ptr<SeedVR2::SeedVR2VAERunner> vae_;
            int scale_     = 0;
            int n_threads_ = 1;

        public:
            bool init(VideoUpscaleRuntime& runtime, const String2TensorStorage& weights, int upscale_factor) override {
                scale_     = upscale_factor;
                n_threads_ = runtime.n_threads;
                dit_       = std::make_unique<SeedVR2::SeedVR2Runner>(runtime.backends.runtime_backend(SDBackendModule::DIFFUSION), weights, runtime.manager);
                vae_       = std::make_unique<SeedVR2::SeedVR2VAERunner>(runtime.backends.runtime_backend(SDBackendModule::VAE), weights, runtime.manager);
                std::set<std::string> used;
                auto register_model = [&](auto& runner, ModelComponent component, SDBackendModule module) {
                    std::map<std::string, ggml_tensor*> tensors;
                    runner->get_param_tensors(tensors);
                    for (const auto& [name, tensor] : tensors) {
                        if (!weights.count(name))
                            throw std::runtime_error("Missing SeedVR2 tensor: " + name);
                        used.insert(name);
                    }
                    runtime.configure_runner(*runner, module, module == SDBackendModule::VAE, module == SDBackendModule::VAE);
                    return runtime.manager->register_param_tensors(component, std::move(tensors),
                                                                   runtime.backends.params_backend_is_disk(module) ? ModelManager::ResidencyMode::Disk : ModelManager::ResidencyMode::ParamBackend,
                                                                   runtime.backends.runtime_backend(module), runtime.backends.params_backend(module));
                };
                if (!register_model(dit_, ModelComponent::Diffusion, SDBackendModule::DIFFUSION) ||
                    !register_model(vae_, ModelComponent::VAE, SDBackendModule::VAE) || !runtime.manager->validate_registered_tensors())
                    return false;
                for (const auto& [name, storage] : weights) {
                    if (!used.count(name) && name != "diffusion.negative_conditioning" && !ends_with(name, ".comfy_quant"))
                        throw std::runtime_error("Unused SeedVR2 tensor: " + name);
                }
                LOG_INFO("SeedVR2 validated %zu parameter tensors; negative conditioning is unused at guidance 1", used.size());
                return true;
            }

            void end() override {
                if (dit_)
                    dit_->runner_end();
                if (vae_)
                    vae_->runner_end();
            }

            bool upscale(const sd_video_upscale_params_t& params, sd_video_generation_result_t* result) override {
                const int chunk = params.temporal_chunk_size;
                if (chunk < 1 || chunk > INT_MAX - 3 || (chunk - 1) % 4) {
                    LOG_ERROR("SeedVR2 temporal chunk size must be 1 or 4n+1 input frames");
                    return false;
                }
                struct Reset {
                    SeedVR2Upscaler* model;
                    ~Reset() { model->end(); }
                } reset{this};
                end();
                const int width = int(params.frames[0].width) * scale_, height = int(params.frames[0].height) * scale_;
                if (width > INT_MAX - 15 || height > INT_MAX - 15)
                    return false;
                const int pw = (width + 15) / 16 * 16, ph = (height + 15) / 16 * 16;
                PhiloxRNG rng;
                rng.manual_seed(uint64_t(params.seed));
                for (int start = 0; start < params.frame_count;) {
                    const int count        = std::min(chunk, params.frame_count - start);
                    const int padded_count = count == 1 ? 1 : ((count - 1 + 3) / 4) * 4 + 1;
                    LOG_INFO("SeedVR2 chunk %d..%d/%d", start + 1, start + count, params.frame_count);
                    const int64_t begin = ggml_time_us();
                    std::vector<sd::Tensor<float>> images;
                    for (int i = 0; i < count; ++i) {
                        auto image = sd_image_to_tensor(params.frames[start + i], -1, -1, false);
                        image /= 255.f;
                        image = sd::ops::interpolate(image, {width, height, 3, 1}, sd::ops::InterpolateMode::Bicubic);
                        image = sd::ops::clamp(image, 0.f, 1.f);
                        if (pw != width)
                            image = sd::ops::concat(image, sd::zeros<float>({pw - width, height, 3, 1}), 0);
                        if (ph != height)
                            image = sd::ops::concat(image, sd::zeros<float>({pw, ph - height, 3, 1}), 1);
                        image *= 2.f;
                        image -= 1.f;
                        images.push_back(image);
                    }
                    auto pixels = images.front();
                    for (int i = 1; i < padded_count; ++i)
                        pixels = sd::ops::concat(pixels, images[std::min(i, count - 1)], 3);
                    pixels.permute_({0, 1, 3, 2});
                    const int64_t prepared = ggml_time_us();
                    auto latent            = vae_->compute(n_threads_, pixels, false);
                    if (latent.empty())
                        return false;
                    latent *= 0.9152f;
                    const int64_t encoded = ggml_time_us();
                    const int lw = pw / 8, lh = ph / 8, lt = (padded_count - 1) / 4 + 1;
                    sd::Tensor<float> noise({lw, lh, lt, 16});
                    auto values = rng.randn(uint32_t(noise.numel()));
                    std::copy(values.begin(), values.end(), noise.data());
                    auto mask = sd::zeros<float>({lw, lh, lt, 1});
                    mask += 1.f;
                    auto input = sd::ops::concat(sd::ops::concat(noise, latent, 3), mask, 3);
                    input.reshape_({2, lw / 2, 2, lh / 2, lt, 33});
                    input.permute_({5, 0, 2, 1, 3, 4});
                    input.reshape_({132, (lw / 2) * (lh / 2) * lt});
                    auto prediction = dit_->compute(n_threads_, input, lw / 2, lh / 2, lt);
                    if (prediction.empty())
                        return false;
                    prediction.reshape_({16, 2, 2, lw / 2, lh / 2, lt});
                    prediction.permute_({1, 3, 2, 4, 5, 0});
                    prediction.reshape_({lw, lh, lt, 16});
                    latent                 = (noise - prediction) / 0.9152f;
                    const int64_t denoised = ggml_time_us();
                    auto decoded           = vae_->compute(n_threads_, latent, true);
                    if (decoded.empty())
                        return false;
                    const int64_t decoded_at = ggml_time_us();
                    if (!std::all_of(decoded.values().begin(), decoded.values().end(), [](float x) { return std::isfinite(x); }))
                        throw std::runtime_error("SeedVR2 produced non-finite pixels");
                    decoded = sd::ops::clamp((decoded + 1.f) * 127.5f, 0.f, 255.f);
                    for (int i = 0; i < count; ++i) {
                        auto output = sd::ops::slice(sd::ops::slice(sd::ops::slice(decoded, 2, i, i + 1), 0, 0, width), 1, 0, height);
                        output.permute_({3, 0, 1, 2});
                        auto& image   = result->frames[start + i];
                        image.width   = width;
                        image.height  = height;
                        image.channel = 3;
                        image.data    = static_cast<uint8_t*>(malloc(size_t(width) * height * 3));
                        if (!image.data)
                            throw std::bad_alloc();
                        std::transform(output.values().begin(), output.values().end(), image.data, [](float x) { return uint8_t(x); });
                    }
                    if (auto progress = sd_get_progress_callback())
                        progress(start + count, params.frame_count, 0.f, sd_get_progress_callback_data());
                    LOG_INFO("SeedVR2 seconds: preprocess %.3f, VAE encode %.3f, DiT and latent transforms %.3f, VAE decode %.3f, postprocess %.3f",
                             (prepared - begin) / 1e6, (encoded - prepared) / 1e6, (denoised - encoded) / 1e6, (decoded_at - denoised) / 1e6, (ggml_time_us() - decoded_at) / 1e6);
                    start += count;
                }
                auto info                              = sd::upscale_generation_info(scale_);
                info["kind"]                           = "video";
                info["upscale"]["model"]               = "seedvr2";
                info["upscale"]["temporal_chunk_size"] = chunk;
                info["seed"]                           = params.seed;
                info["fps"]                            = params.fps;
                info["video_frames"]                   = params.frame_count;
                result->generation_info                = sd::copy_generation_info(info);
                return result->generation_info != nullptr;
            }
        };

        bool is_seedvr2_model(const String2TensorStorage& weights) {
            return weights.count("diffusion.vid_in.proj.weight") && weights.count("diffusion.blocks.0.attn.rope.rope.freqs") &&
                   weights.count("diffusion.positive_conditioning");
        }

        std::unique_ptr<VideoUpscaleModel> create_seedvr2_upscaler() {
            return std::make_unique<SeedVR2Upscaler>();
        }
    }
}

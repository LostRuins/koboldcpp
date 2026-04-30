#ifndef __VAE_HPP__
#define __VAE_HPP__

#include "common_block.hpp"
#include "tensor_ggml.hpp"

struct VAE : public GGMLRunner {
protected:
    SDVersion version;
    bool scale_input                                      = true;
    virtual sd::Tensor<float> _compute(const int n_threads,
                                       const sd::Tensor<float>& z,
                                       bool decode_graph) = 0;

    static inline void scale_tensor_to_minus1_1(sd::Tensor<float>* tensor) {
        GGML_ASSERT(tensor != nullptr);
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            (*tensor)[i] = (*tensor)[i] * 2.0f - 1.0f;
        }
    }

    static inline void scale_tensor_to_0_1(sd::Tensor<float>* tensor) {
        GGML_ASSERT(tensor != nullptr);
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            float value  = ((*tensor)[i] + 1.0f) * 0.5f;
            (*tensor)[i] = std::max(0.0f, std::min(1.0f, value));
        }
    }

    sd::Tensor<float> tiled_compute(const sd::Tensor<float>& input,
                                    int n_threads,
                                    int output_width,
                                    int output_height,
                                    int scale,
                                    int p_tile_size_x,
                                    int p_tile_size_y,
                                    float tile_overlap_factor,
                                    bool circular_x,
                                    bool circular_y,
                                    bool decode_graph,
                                    const char* error_message,
                                    bool silent = false) {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) {
            auto output_tile = _compute(n_threads, input_tile, decode_graph);
            if (output_tile.empty()) {
                LOG_ERROR("%s", error_message);
                return sd::Tensor<float>();
            }
            return output_tile;
        };
        return ::process_tiles_2d(input,
                                  output_width,
                                  output_height,
                                  scale,
                                  p_tile_size_x,
                                  p_tile_size_y,
                                  tile_overlap_factor,
                                  circular_x,
                                  circular_y,
                                  on_processing,
                                  silent);
    }

public:
    VAE(SDVersion version, ggml_backend_t backend, bool offload_params_to_cpu)
        : version(version), GGMLRunner(backend, offload_params_to_cpu) {}

    int get_scale_factor() {
        int scale_factor = 8;
        if (version == VERSION_WAN2_2_TI2V) {
            scale_factor = 16;
        } else if (sd_version_uses_flux2_vae(version)) {
            scale_factor = 16;
        } else if (version == VERSION_CHROMA_RADIANCE) {
            scale_factor = 1;
        }
        return scale_factor;
    }

    virtual int get_encoder_output_channels(int input_channels) = 0;

    void get_tile_sizes(int& tile_size_x,
                        int& tile_size_y,
                        float& tile_overlap,
                        const sd_tiling_params_t& params,
                        int64_t latent_x,
                        int64_t latent_y,
                        float encoding_factor = 1.0f) {
        tile_overlap       = std::max(std::min(params.target_overlap, 0.5f), 0.0f);
        auto get_tile_size = [&](int requested_size, float factor, int64_t latent_size) {
            const int default_tile_size  = 32;
            const int min_tile_dimension = 4;
            int tile_size                = default_tile_size;
            // factor <= 1 means simple fraction of the latent dimension
            // factor > 1 means number of tiles across that dimension
            if (factor > 0.f) {
                if (factor > 1.0)
                    factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                tile_size = static_cast<int>(std::round(latent_size * factor));
            } else if (requested_size >= min_tile_dimension) {
                tile_size = requested_size;
            }
            tile_size = static_cast<int>(tile_size * encoding_factor);
            return std::max(std::min(tile_size, static_cast<int>(latent_size)), min_tile_dimension);
        };

        tile_size_x = get_tile_size(params.tile_size_x, params.rel_size_x, latent_x);
        tile_size_y = get_tile_size(params.tile_size_y, params.rel_size_y, latent_y);
    }

    sd::Tensor<float> encode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling_params,
                             bool circular_x = false,
                             bool circular_y = false) {
        int64_t t0              = ggml_time_ms();
        sd::Tensor<float> input = x;
        sd::Tensor<float> output;
        if (scale_input) {
            scale_tensor_to_minus1_1(&input);
        }

        if (tiling_params.enabled) {
            const int scale_factor = get_scale_factor();
            int64_t W              = input.shape()[0] / scale_factor;
            int64_t H              = input.shape()[1] / scale_factor;
            float tile_overlap;
            int tile_size_x, tile_size_y;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, W, H, 1.30539f);
            LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
            output = tiled_compute(input,
                                   n_threads,
                                   static_cast<int>(W),
                                   static_cast<int>(H),
                                   scale_factor,
                                   tile_size_x,
                                   tile_size_y,
                                   tile_overlap,
                                   circular_x,
                                   circular_y,
                                   false,
                                   "vae encode compute failed while processing a tile");
        } else {
            output = _compute(n_threads, input, false);
        }

        free_compute_buffer();

        if (output.empty()) {
            LOG_ERROR("vae encode compute failed");
            return {};
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae encode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    sd::Tensor<float> decode(int n_threads,
                             const sd::Tensor<float>& x,
                             sd_tiling_params_t tiling_params,
                             bool decode_video = false,
                             bool circular_x   = false,
                             bool circular_y   = false,
                             bool silent       = false) {
        int64_t t0              = ggml_time_ms();
        sd::Tensor<float> input = x;
        sd::Tensor<float> output;

        if (tiling_params.enabled) {
            const int scale_factor = get_scale_factor();
            int64_t W              = input.shape()[0] * scale_factor;
            int64_t H              = input.shape()[1] * scale_factor;
            float tile_overlap;
            int tile_size_x, tile_size_y;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, input.shape()[0], input.shape()[1]);
            if (!silent) {
                LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
            }
            output = tiled_compute(
                input,
                n_threads,
                static_cast<int>(W),
                static_cast<int>(H),
                scale_factor,
                tile_size_x,
                tile_size_y,
                tile_overlap,
                circular_x,
                circular_y,
                true,
                "vae decode compute failed while processing a tile",
                silent);
        } else {
            output = _compute(n_threads, input, true);
        }

        free_compute_buffer();

        if (output.empty()) {
            LOG_ERROR("vae decode compute failed");
            return {};
        }
        if (scale_input) {
            scale_tensor_to_0_1(&output);
        }
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae decode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    virtual sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) = 0;
    virtual sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix)         = 0;
    virtual void set_conv2d_scale(float scale) { SD_UNUSED(scale); };
};

struct FakeVAE : public VAE {
    FakeVAE(SDVersion version, ggml_backend_t backend, bool offload_params_to_cpu)
        : VAE(version, backend, offload_params_to_cpu) {}

    int get_encoder_output_channels(int input_channels) {
        return input_channels;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        SD_UNUSED(n_threads);
        SD_UNUSED(decode_graph);
        return z;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {}

    std::string get_desc() override {
        return "fake_vae";
    }
};

#endif  // __VAE_HPP__

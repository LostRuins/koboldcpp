#ifndef __SD_MODEL_VAE_VAE_HPP__
#define __SD_MODEL_VAE_VAE_HPP__

#include <cmath>
#include <limits>

#include "core/tensor_ggml.hpp"
#include "model/common/block.hpp"
#include "model/vae/vae_tiling.hpp"
#include "model_manager.h"
#include "runtime/tiling.h"

struct VAE : public GGMLRunner {
protected:
    SDVersion version;
    std::string weight_prefix;
    bool scale_input                                      = true;
    virtual sd::Tensor<float> _compute(const int n_threads,
                                       const sd::Tensor<float>& z,
                                       bool decode_graph) = 0;

    virtual bool supports_temporal_tiling(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return false;
    }

    virtual int get_default_temporal_tile_frames(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 4;
    }

    virtual int get_default_temporal_tile_overlap(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 1;
    }

    virtual int get_temporal_tile_output_scale(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 1;
    }

    virtual sd::Tensor<float> _compute_temporal_tiled(const int n_threads,
                                                      const sd::Tensor<float>& input,
                                                      VAETemporalDirection direction,
                                                      const VAETemporalTilingConfig& config) {
        if (direction != VAETemporalDirection::DECODE) {
            return _compute(n_threads, input, false);
        }

        VAETemporalTilingConfig resolved_config = config;
        const int output_scale                  = get_temporal_tile_output_scale(direction);
        if (output_scale > 1 &&
            resolved_config.overlap == 0 &&
            input.shape()[2] > resolved_config.tile_frames) {
            LOG_WARN("%s temporal decode requires at least one overlapping latent frame; using overlap=1",
                     get_desc().c_str());
            resolved_config.overlap = 1;
        }

        auto plan = make_vae_temporal_tile_plan(input.shape()[2], resolved_config);
        LOG_VERBOSE("%s temporal tiling: tile_frames=%d, overlap=%d, total_frames=%lld, tiles=%d",
                    get_desc().c_str(),
                    plan.tile_frames,
                    plan.overlap,
                    (long long)input.shape()[2],
                    (int)plan.tiles.size());
        return process_vae_temporal_tiles_blended(
            input,
            plan,
            output_scale,
            [&](const sd::Tensor<float>& input_tile, const VAETemporalTile& tile) {
                LOG_VERBOSE("%s temporal tile %d/%d: input frames [%lld, %lld)",
                            get_desc().c_str(),
                            tile.index + 1,
                            (int)plan.tiles.size(),
                            (long long)tile.start,
                            (long long)tile.end);
                return _compute(n_threads, input_tile, true);
            });
    }

    sd::Tensor<float> compute_with_temporal_tiling(const int n_threads,
                                                   const sd::Tensor<float>& input,
                                                   VAETemporalDirection direction,
                                                   const sd_tiling_params_t& tiling_params) {
        if (!tiling_params.temporal_tiling || input.dim() != 5 || input.shape()[2] <= 1) {
            return _compute(n_threads, input, direction == VAETemporalDirection::DECODE);
        }
        if (!supports_temporal_tiling(direction)) {
            LOG_WARN("%s does not support temporal tiling for %s; processing the full temporal dimension",
                     get_desc().c_str(),
                     direction == VAETemporalDirection::DECODE ? "decode" : "encode");
            return _compute(n_threads, input, direction == VAETemporalDirection::DECODE);
        }

        auto config = resolve_vae_temporal_tiling_config(
            tiling_params,
            get_default_temporal_tile_frames(direction),
            get_default_temporal_tile_overlap(direction));
        return _compute_temporal_tiled(n_threads, input, direction, config);
    }

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
                                    int p_tile_size_w,
                                    int p_tile_size_h,
                                    float tile_overlap_factor,
                                    bool circular_x,
                                    bool circular_y,
                                    bool decode_graph,
                                    const sd_tiling_params_t& tiling_params,
                                    const char* error_message,
                                    bool silent = false) {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) {
            auto output_tile = compute_with_temporal_tiling(
                n_threads,
                input_tile,
                decode_graph ? VAETemporalDirection::DECODE : VAETemporalDirection::ENCODE,
                tiling_params);
            if (output_tile.empty()) {
                LOG_ERROR("%s", error_message);
                return sd::Tensor<float>();
            }
            return output_tile;
        };
        const bool original_circular_x = circular_x_enabled;
        const bool original_circular_y = circular_y_enabled;
        const int64_t latent_width     = decode_graph ? input.shape()[0] : output_width;
        const int64_t latent_height    = decode_graph ? input.shape()[1] : output_height;
        circular_x                     = circular_x || original_circular_x;
        circular_y                     = circular_y || original_circular_y;
        // Full-width axes wrap in convolutions; split axes wrap between tiles.
        set_circular_axes(circular_x && p_tile_size_w >= latent_width,
                          circular_y && p_tile_size_h >= latent_height);
        auto output = ::process_tiles_2d(input,
                                         output_width,
                                         output_height,
                                         scale,
                                         p_tile_size_w,
                                         p_tile_size_h,
                                         tile_overlap_factor,
                                         circular_x && p_tile_size_w < latent_width,
                                         circular_y && p_tile_size_h < latent_height,
                                         on_processing,
                                         silent);
        set_circular_axes(original_circular_x, original_circular_y);
        return output;
    }

public:
    VAE(SDVersion version,
        ggml_backend_t backend,
        const std::string& weight_prefix                    = "",
        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : version(version), weight_prefix(weight_prefix), GGMLRunner(backend, weight_manager) {}

    int get_scale_factor() {
        int scale_factor = 8;
        if (version == VERSION_LTXAV) {
            scale_factor = 32;
        } else if (version == VERSION_WAN2_2_TI2V || version == VERSION_QWEN_IMAGE_2_1 || sd_version_is_hunyuan_video(version) || sd_version_is_mage_flow(version) || sd_version_is_minimax_h3(version)) {
            scale_factor = 16;
        } else if (sd_version_uses_flux2_vae(version)) {
            scale_factor = 16;
        } else if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 || sd_version_is_minit2i(version) || sd_version_is_sensenova_u1(version)) {
            scale_factor = 1;
        }
        return scale_factor;
    }

    virtual int get_encoder_output_channels(int input_channels) = 0;

    bool can_temporal_tile_decode() const {
        return supports_temporal_tiling(VAETemporalDirection::DECODE);
    }

    virtual sd_tiling_params_t resolve_tiling_params(sd_tiling_params_t params) const {
        return params;
    }

    bool get_tile_sizes(int& tile_size_w,
                        int& tile_size_h,
                        float& tile_overlap,
                        const sd_tiling_params_t& params,
                        int64_t latent_w,
                        int64_t latent_h) {
        const auto tiling = resolve_tiling_params(params);
        if (latent_w <= 0 || latent_h <= 0 ||
            latent_w > std::numeric_limits<int>::max() || latent_h > std::numeric_limits<int>::max() ||
            !std::isfinite(tiling.target_overlap)) {
            LOG_ERROR("invalid VAE tiling dimensions or overlap");
            return false;
        }
        const int scale_factor = get_scale_factor();
        tile_overlap           = std::max(std::min(tiling.target_overlap, 0.5f), 0.0f);
        auto get_tile_size     = [&](int requested_size, double factor, int64_t latent_size, int& tile_size) {
            if (requested_size < 0 || !std::isfinite(factor) || factor < 0.0) {
                LOG_ERROR("VAE tile sizes and relative sizes must be finite and non-negative");
                return false;
            }
            const int min_tile_dimension = std::min(4, static_cast<int>(latent_size));
            double size                  = (requested_size > 0 ? requested_size : 256) / scale_factor;
            if (factor > 0.0) {
                if (factor > 1.0) {
                    factor = 1.0 / (factor * (1.0 - tile_overlap) + tile_overlap);
                }
                size = std::floor(static_cast<double>(latent_size) * factor);
            }
            if (size < min_tile_dimension && (requested_size > 0 || factor > 0.0)) {
                LOG_ERROR("VAE tile size must be at least %d image pixels on this axis", min_tile_dimension * scale_factor);
                return false;
            }
            tile_size = static_cast<int>(std::min(static_cast<double>(latent_size), std::max<double>(min_tile_dimension, size)));
            return true;
        };

        return get_tile_size(tiling.tile_size_w, tiling.rel_size_w, latent_w, tile_size_w) &&
               get_tile_size(tiling.tile_size_h, tiling.rel_size_h, latent_h, tile_size_h);
    }

    virtual sd::Tensor<float> encode(int n_threads,
                                     const sd::Tensor<float>& x,
                                     sd_tiling_params_t tiling_params,
                                     bool circular_x = false,
                                     bool circular_y = false) {
        int64_t t0              = ggml_time_ms();
        tiling_params           = resolve_tiling_params(tiling_params);
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
            int tile_size_w, tile_size_h;
            if (!get_tile_sizes(tile_size_w, tile_size_h, tile_overlap, tiling_params, W, H)) {
                return {};
            }
            LOG_VERBOSE("VAE encode tile size: %dx%d pixels (%dx%d latent)",
                        tile_size_w * scale_factor, tile_size_h * scale_factor, tile_size_w, tile_size_h);
            output = tiled_compute(input,
                                   n_threads,
                                   static_cast<int>(W),
                                   static_cast<int>(H),
                                   scale_factor,
                                   tile_size_w,
                                   tile_size_h,
                                   tile_overlap,
                                   circular_x,
                                   circular_y,
                                   false,
                                   tiling_params,
                                   "vae encode compute failed while processing a tile");
        } else {
            output = compute_with_temporal_tiling(n_threads,
                                                  input,
                                                  VAETemporalDirection::ENCODE,
                                                  tiling_params);
        }

        runner_end();

        if (output.empty()) {
            LOG_ERROR("vae encode compute failed");
            return {};
        }
        int64_t t1 = ggml_time_ms();
        LOG_VERBOSE("computing vae encode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    virtual sd::Tensor<float> decode(int n_threads,
                                     const sd::Tensor<float>& x,
                                     sd_tiling_params_t tiling_params,
                                     bool decode_video = false,
                                     bool circular_x   = false,
                                     bool circular_y   = false,
                                     bool silent       = false) {
        int64_t t0              = ggml_time_ms();
        tiling_params           = resolve_tiling_params(tiling_params);
        sd::Tensor<float> input = x;
        sd::Tensor<float> output;

        if (tiling_params.enabled) {
            const int scale_factor = get_scale_factor();
            int64_t W              = input.shape()[0] * scale_factor;
            int64_t H              = input.shape()[1] * scale_factor;
            float tile_overlap;
            int tile_size_w, tile_size_h;
            if (!get_tile_sizes(tile_size_w, tile_size_h, tile_overlap, tiling_params, input.shape()[0], input.shape()[1])) {
                return {};
            }
            if (!silent) {
                LOG_VERBOSE("VAE decode tile size: %dx%d pixels (%dx%d latent)",
                            tile_size_w * scale_factor, tile_size_h * scale_factor, tile_size_w, tile_size_h);
            }
            output = tiled_compute(
                input,
                n_threads,
                static_cast<int>(W),
                static_cast<int>(H),
                scale_factor,
                tile_size_w,
                tile_size_h,
                tile_overlap,
                circular_x,
                circular_y,
                true,
                tiling_params,
                "vae decode compute failed while processing a tile",
                silent);
        } else {
            output = compute_with_temporal_tiling(n_threads,
                                                  input,
                                                  VAETemporalDirection::DECODE,
                                                  tiling_params);
        }

        runner_end();

        if (output.empty()) {
            LOG_ERROR("vae decode compute failed");
            return {};
        }
        if (scale_input) {
            scale_tensor_to_0_1(&output);
        }
        int64_t t1 = ggml_time_ms();
        LOG_VERBOSE("computing vae decode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    virtual sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) = 0;
    virtual sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors)                                   = 0;
    virtual void set_conv2d_scale(float scale) { SD_UNUSED(scale); };
};

struct FakeVAE : public VAE {
    FakeVAE(SDVersion version,
            ggml_backend_t backend,
            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(version, backend, "", weight_manager) {}

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

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {}

    std::string get_desc() override {
        return "fake_vae";
    }
};

#endif  // __SD_MODEL_VAE_VAE_HPP__

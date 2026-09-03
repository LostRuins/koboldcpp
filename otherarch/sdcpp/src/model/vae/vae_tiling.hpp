#ifndef __SD_MODEL_VAE_VAE_TILING_HPP__
#define __SD_MODEL_VAE_VAE_TILING_HPP__

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/tensor.hpp"
#include "core/util.h"

enum class VAETemporalDirection {
    ENCODE,
    DECODE,
};

struct VAETemporalTilingConfig {
    int tile_frames = 1;
    int overlap     = 0;
};

struct VAETemporalTile {
    int index     = 0;
    int64_t start = 0;
    int64_t end   = 0;
    int overlap   = 0;
    bool first    = false;
    bool last     = false;
};

struct VAETemporalTilePlan {
    int tile_frames = 1;
    int overlap     = 0;
    int stride      = 1;
    std::vector<VAETemporalTile> tiles;
};

inline VAETemporalTilingConfig resolve_vae_temporal_tiling_config(const sd_tiling_params_t& params,
                                                                  int default_tile_frames,
                                                                  int default_overlap) {
    VAETemporalTilingConfig config;
    config.tile_frames = std::max(1, default_tile_frames);
    config.overlap     = std::max(0, default_overlap);

    for (const auto& [key, value] : parse_key_value_args(params.extra_tiling_args, "VAE extra tiling arg")) {
        if (key != "temporal_tile_frames" && key != "temporal_tile_size" && key != "temporal_tile_overlap") {
            continue;
        }

        int parsed = 0;
        if (!parse_strict_int(value, parsed)) {
            LOG_WARN("ignoring invalid VAE extra tiling arg '%s=%s'", key.c_str(), value.c_str());
        } else if (key == "temporal_tile_overlap") {
            config.overlap = std::max(0, parsed);
        } else {
            config.tile_frames = std::max(1, parsed);
        }
    }
    return config;
}

inline VAETemporalTilePlan make_vae_temporal_tile_plan(int64_t total_frames,
                                                       const VAETemporalTilingConfig& config) {
    VAETemporalTilePlan plan;
    plan.tile_frames = std::max(1, config.tile_frames);
    plan.overlap     = std::max(0, config.overlap);
    if (total_frames <= 1) {
        plan.overlap = 0;
    }

    if (plan.overlap >= plan.tile_frames) {
        LOG_WARN("temporal_tile_overlap (%d) is greater than or equal to temporal_tile_frames (%d), adjusting values to avoid empty decode windows",
                 plan.overlap,
                 plan.tile_frames);
        plan.overlap = plan.tile_frames - 1;
    }
    if (total_frames > 1 && plan.overlap >= total_frames) {
        LOG_WARN("temporal_tile_overlap (%d) is greater than or equal to total frames (%lld), adjusting values to process at least one tile",
                 plan.overlap,
                 (long long)total_frames);
        plan.overlap = static_cast<int>(total_frames - 1);
    }

    plan.stride = std::max(1, plan.tile_frames - plan.overlap);
    for (int64_t start = 0; start < total_frames - plan.overlap; start += plan.stride) {
        VAETemporalTile tile;
        tile.index   = static_cast<int>(plan.tiles.size());
        tile.start   = start;
        tile.end     = std::min<int64_t>(total_frames, start + plan.tile_frames);
        tile.overlap = tile.end < total_frames ? plan.overlap : 0;
        tile.first   = start == 0;
        tile.last    = tile.end == total_frames;
        plan.tiles.push_back(tile);
    }
    return plan;
}

template <typename Fn>
inline sd::Tensor<float> process_vae_temporal_tiles(const sd::Tensor<float>& input,
                                                    const VAETemporalTilePlan& plan,
                                                    Fn&& on_processing) {
    sd::Tensor<float> output;
    for (const auto& tile : plan.tiles) {
        auto input_tile  = sd::ops::slice(input, 2, tile.start, tile.end);
        auto output_tile = on_processing(input_tile, tile);
        if (output_tile.empty()) {
            return {};
        }
        output = output.empty() ? std::move(output_tile)
                                : sd::ops::concat(output, output_tile, 2);
    }
    return output;
}

template <typename Fn>
inline sd::Tensor<float> process_vae_temporal_tiles_blended(const sd::Tensor<float>& input,
                                                            const VAETemporalTilePlan& plan,
                                                            int output_scale,
                                                            Fn&& on_processing) {
    GGML_ASSERT(output_scale >= 1);
    const int64_t output_frames = 1 + (input.shape()[2] - 1) * output_scale;
    const int overlap_frames    = plan.overlap > 0 ? 1 + (plan.overlap - 1) * output_scale : 0;
    std::vector<float> weights(static_cast<size_t>(output_frames), 0.f);
    sd::Tensor<float> output;

    auto smootherstep = [](float value) {
        return value * value * value * (value * (value * 6.f - 15.f) + 10.f);
    };

    for (const auto& tile : plan.tiles) {
        auto input_tile  = sd::ops::slice(input, 2, tile.start, tile.end);
        auto output_tile = on_processing(input_tile, tile);
        if (output_tile.empty()) {
            return {};
        }

        const int64_t expected_tile_frames = 1 + (input_tile.shape()[2] - 1) * output_scale;
        if (output_tile.dim() < 3 || output_tile.shape()[2] != expected_tile_frames) {
            LOG_ERROR("unexpected temporal tile output shape: expected %lld frames, got %lld",
                      (long long)expected_tile_frames,
                      output_tile.dim() < 3 ? -1LL : (long long)output_tile.shape()[2]);
            return {};
        }

        if (output.empty()) {
            auto output_shape = output_tile.shape();
            output_shape[2]   = output_frames;
            output            = sd::Tensor<float>::zeros(std::move(output_shape));
        } else {
            if (output.dim() != output_tile.dim()) {
                LOG_ERROR("temporal tile output rank mismatch: expected %lld, got %lld",
                          (long long)output.dim(),
                          (long long)output_tile.dim());
                return {};
            }
            for (size_t dim = 0; dim < static_cast<size_t>(output.dim()); ++dim) {
                if (dim != 2 && output.shape()[dim] != output_tile.shape()[dim]) {
                    LOG_ERROR("temporal tile output shape mismatch at dimension %zu", dim);
                    return {};
                }
            }
        }

        const int64_t output_start = tile.start * output_scale;
        const int64_t inner        = output.shape()[0] * output.shape()[1];
        const int64_t outer        = output.numel() / (inner * output.shape()[2]);
        const int64_t tile_frames  = output_tile.shape()[2];
        for (int64_t frame = 0; frame < tile_frames; ++frame) {
            float weight = 1.f;
            if (!tile.first && overlap_frames > 0 && frame < overlap_frames) {
                weight *= smootherstep(static_cast<float>(frame + 1) /
                                       static_cast<float>(overlap_frames + 1));
            }
            if (!tile.last && overlap_frames > 0 && frame >= tile_frames - overlap_frames) {
                weight *= smootherstep(static_cast<float>(tile_frames - frame) /
                                       static_cast<float>(overlap_frames + 1));
            }

            const int64_t output_frame = output_start + frame;
            GGML_ASSERT(output_frame >= 0 && output_frame < output_frames);
            weights[static_cast<size_t>(output_frame)] += weight;
            for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
                const int64_t src_offset = (outer_index * tile_frames + frame) * inner;
                const int64_t dst_offset = (outer_index * output_frames + output_frame) * inner;
                for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
                    output[dst_offset + inner_index] += output_tile[src_offset + inner_index] * weight;
                }
            }
        }
    }

    if (output.empty()) {
        return {};
    }
    const int64_t inner = output.shape()[0] * output.shape()[1];
    const int64_t outer = output.numel() / (inner * output.shape()[2]);
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        const float weight = weights[static_cast<size_t>(frame)];
        if (weight <= 0.f) {
            LOG_ERROR("temporal tiling left output frame %lld uncovered", (long long)frame);
            return {};
        }
        for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
            const int64_t offset = (outer_index * output_frames + frame) * inner;
            for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
                output[offset + inner_index] /= weight;
            }
        }
    }
    return output;
}

#endif  // __SD_MODEL_VAE_VAE_TILING_HPP__

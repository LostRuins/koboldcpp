#ifndef __SD_MODEL_COMMON_VSR_HPP__
#define __SD_MODEL_COMMON_VSR_HPP__

#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include "model/common/ggml_block.hpp"
#include "model/common/rope.hpp"

namespace VSR {

    struct GraphInputs {
        std::map<std::vector<int>, sd::Tensor<float>> rotations;
        mutable std::unordered_map<const void*, ggml_tensor*> bindings;

        void begin_graph() {
            bindings.clear();
        }

        void clear() {
            bindings.clear();
            rotations.clear();
        }

        template <typename T>
        ggml_tensor* input(GGMLRunnerContext* ctx, const sd::Tensor<T>& value) const {
            auto existing = bindings.find(value.data());
            if (existing != bindings.end())
                return existing->second;
            auto result = sd::make_ggml_tensor(ctx->ggml_ctx, value, false);
            ctx->set_backend_tensor_data(result, value.data());
            bindings.emplace(value.data(), result);
            return result;
        }

        ggml_tensor* rope(GGMLRunnerContext* ctx, int width, int height, const std::vector<int>& times, const std::vector<int>& axes) {
            std::vector<int> key = {width, height, axes[0], axes[1], axes[2]};
            key.insert(key.end(), times.begin(), times.end());
            auto existing = rotations.find(key);
            if (existing != rotations.end())
                return input(ctx, existing->second);
            const int dim = axes[0] + axes[1] + axes[2];
            auto& pe      = rotations.emplace(std::move(key), sd::Tensor<float>({2, 2, dim / 2, int64_t(width) * height * int64_t(times.size())})).first->second;
            float* out    = pe.data();
            for (int t : times) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const int pos[] = {t, y, x};
                        for (int a = 0; a < 3; ++a) {
                            for (int i = 0; i < axes[a]; i += 2) {
                                const double angle = pos[a] * std::pow(10000.0, -double(i) / axes[a]);
                                *out++             = float(std::cos(angle));
                                *out++             = float(-std::sin(angle));
                                *out++             = float(std::sin(angle));
                                *out++             = float(std::cos(angle));
                            }
                        }
                    }
                }
            }
            return input(ctx, pe);
        }
    };

    static ggml_tensor* tokens(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2] * x->ne[3]);
        return ggml_cont(ctx, ggml_transpose(ctx, x));
    }

    static ggml_tensor* image(ggml_context* ctx, ggml_tensor* x, int width, int height) {
        const auto channels = x->ne[0];
        x                   = ggml_cont(ctx, ggml_transpose(ctx, x));
        return ggml_reshape_4d(ctx, x, width, height, channels, 1);
    }

    static ggml_tensor* pixel_shuffle(ggml_context* ctx, ggml_tensor* x, int width, int height, int scale, bool channel_first = true) {
        const int64_t channels = x->ne[0] / (scale * scale);
        if (!channel_first) {
            x = ggml_reshape_4d(ctx, x, channels, scale, scale, width * height);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
        }
        x = ggml_reshape_4d(ctx, x, scale, scale, channels, width * height);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 3, 1));
        x = ggml_reshape_4d(ctx, x, scale * width, height, scale, channels);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_4d(ctx, x, scale * width, scale * height, channels, 1);
    }

    struct AttentionTile {
        int width;
        int height;
        sd::Tensor<int32_t> queries;
        sd::Tensor<int32_t> keys;
        sd::Tensor<float> mask;
    };

    struct LocalAttention {
        int width;
        int height;
        std::vector<std::pair<int, int>> windows;
        bool preserve_boundary;
        mutable std::vector<AttentionTile> tiles;
        GraphInputs inputs;

        void begin_graph() {
            inputs.begin_graph();
        }

        LocalAttention(int width, int height, const std::vector<std::pair<int, int>>& windows, bool preserve_boundary)
            : width(width), height(height), windows(windows), preserve_boundary(preserve_boundary) {}

        void prepare_tiles() const {
            if (!tiles.empty())
                return;
            constexpr int tile_size = 16;
            for (int y = 0; y < height; y += tile_size) {
                for (int x = 0; x < width; x += tile_size) {
                    AttentionTile tile;
                    tile.width   = std::min(tile_size, width - x);
                    tile.height  = std::min(tile_size, height - y);
                    tile.queries = sd::Tensor<int32_t>({tile.width * tile.height});
                    std::vector<int32_t> keys;
                    auto bounds = [&](int qy, int qx, int plane) {
                        auto [wh, ww] = windows[plane];
                        if (preserve_boundary) {
                            wh             = std::min(wh, height);
                            ww             = std::min(ww, width);
                            const int top  = std::clamp(qy - wh / 2, 0, height - wh);
                            const int left = std::clamp(qx - ww / 2, 0, width - ww);
                            return std::array<int, 4>{top, left, top + wh, left + ww};
                        }
                        return std::array<int, 4>{std::max(0, qy - wh / 2), std::max(0, qx - ww / 2),
                                                  std::min(height, qy + wh / 2), std::min(width, qx + ww / 2)};
                    };
                    for (int p = 0; p < int(windows.size()); ++p) {
                        auto begin = bounds(y, x, p);
                        auto end   = bounds(y + tile.height - 1, x + tile.width - 1, p);
                        for (int ky = begin[0]; ky < end[2]; ++ky)
                            for (int kx = begin[1]; kx < end[3]; ++kx)
                                keys.push_back(p * height * width + ky * width + kx);
                    }
                    tile.keys = sd::Tensor<int32_t>({int64_t(keys.size())});
                    std::copy(keys.begin(), keys.end(), tile.keys.data());
                    tile.mask = sd::Tensor<float>({int64_t(keys.size()), tile.width * tile.height});
                    for (int q = 0; q < tile.width * tile.height; ++q) {
                        int qy                 = y + q / tile.width;
                        int qx                 = x + q % tile.width;
                        tile.queries.data()[q] = qy * width + qx;
                        for (size_t k = 0; k < keys.size(); ++k) {
                            int p                                 = keys[k] / (width * height);
                            int ky                                = (keys[k] % (width * height)) / width;
                            int kx                                = keys[k] % width;
                            auto b                                = bounds(qy, qx, p);
                            tile.mask.data()[q * keys.size() + k] = ky >= b[0] && ky < b[2] && kx >= b[1] && kx < b[3]
                                                                        ? 0.f
                                                                        : -std::numeric_limits<float>::infinity();
                        }
                    }
                    tiles.push_back(std::move(tile));
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, int heads) const {
            auto c = ctx->ggml_ctx;
#ifndef SD_USE_UPSTREAM_GGML
            if (ctx->flash_attn_enabled && windows.size() <= 4) {
                SDWindowAttentionParams params = {};
                params.width                   = width;
                params.height                  = height;
                params.heads                   = heads;
                params.planes                  = int(windows.size());
                params.preserve_boundary       = preserve_boundary;
                for (size_t i = 0; i < windows.size(); ++i) {
                    params.window_height[i] = windows[i].first;
                    params.window_width[i]  = windows[i].second;
                }
                if (auto out = ggml_ext_window_attn(ctx, q, k, v, params))
                    return out;
            }
#endif
            prepare_tiles();
            ggml_tensor* output = nullptr;
            ggml_tensor* row    = nullptr;
            int row_width       = 0;
            for (const auto& tile : tiles) {
                auto qi   = inputs.input(ctx, tile.queries);
                auto ki   = inputs.input(ctx, tile.keys);
                auto tq   = ggml_get_rows(c, q, qi);
                auto tk   = ggml_get_rows(c, k, ki);
                auto tv   = ggml_get_rows(c, v, ki);
                auto mask = inputs.input(ctx, tile.mask);
                auto out  = ggml_ext_attention_ext(ctx, tq, tk, tv, heads, mask, false, ctx->flash_attn_enabled);
                out       = ggml_reshape_3d(c, out, out->ne[0], tile.width, tile.height);
                row       = row ? ggml_concat(c, row, out, 1) : out;
                row_width += tile.width;
                if (row_width == width) {
                    output    = output ? ggml_concat(c, output, row, 2) : row;
                    row       = nullptr;
                    row_width = 0;
                }
            }
            return ggml_reshape_2d(c, output, output->ne[0], width * height);
        }
    };

    static ggml_tensor* apply_rope(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, int heads) {
        auto c              = ctx->ggml_ctx;
        const int64_t dim   = x->ne[0];
        const int64_t count = x->ne[1];
        x                   = ggml_reshape_4d(c, x, dim / heads, heads, count, 1);
        x                   = Rope::apply_rope(ctx, x, pe);
        x                   = ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3));
        return ggml_reshape_2d(c, x, dim, count);
    }

}  // namespace VSR

#endif  // __SD_MODEL_COMMON_VSR_HPP__

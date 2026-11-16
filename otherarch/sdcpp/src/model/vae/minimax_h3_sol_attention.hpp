#ifndef __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__

#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <vector>

#include "core/ggml_extend.h"
#include "core/tensor.hpp"

namespace MiniMaxH3VAE {

    constexpr int64_t sol_window_size    = 16;
    constexpr int64_t sol_block_size     = 5;
    constexpr int64_t sol_global_rows    = 5;
    constexpr int64_t sol_spatial_shift  = sol_window_size / 2;
    constexpr int64_t sol_temporal_shift = 2;

    struct SolAttentionPlan {
        template <typename Tensor>
        struct Batch {
            Tensor output_rows;
            Tensor queries;
            Tensor keys;
            Tensor previous_queries;
            Tensor previous_keys;
        };

        template <typename Tensor>
        struct Block {
            Tensor rows;
            std::vector<Batch<Tensor>> batches;
        };

        std::vector<Block<sd::Tensor<int32_t>>> blocks;

        SolAttentionPlan(int64_t width, int64_t height, int64_t frames, bool shifted) {
            const int64_t frame_size = width * height;
            const int64_t pixels     = frames * frame_size;
            const int64_t block_rows = sol_block_size * frame_size;
            const int64_t count      = (frames + sol_block_size - 1) / sol_block_size;
            const int64_t shift      = shifted ? sol_spatial_shift : 0;
            const int64_t time_shift = shifted ? sol_temporal_shift : 0;

            auto pack = [](std::vector<int32_t> rows) {
                const auto size = static_cast<int64_t>(rows.size());
                return sd::Tensor<int32_t>({size}, std::move(rows));
            };
            for (int64_t block = 0; block < count; ++block) {
                std::vector<int32_t> rows;
                const int64_t start = block * block_rows;
                const int64_t end   = std::min(pixels, start + block_rows);
                for (int64_t i = start; i < end; ++i) {
                    rows.push_back(static_cast<int32_t>(i));
                }
                std::vector<int32_t> globals;
                std::vector<int32_t> queries;
                for (int64_t i = 0; i < sol_global_rows; ++i) {
                    globals.push_back(static_cast<int32_t>(pixels + block * sol_global_rows + i));
                    queries.push_back(static_cast<int32_t>(rows.size()));
                    rows.push_back(globals.back());
                }
                blocks.push_back({pack(std::move(rows)), {}});
                blocks.back().batches.push_back({pack(std::move(globals)), pack(std::move(queries)), {}, {}, {}});
            }

            struct Window {
                std::vector<int32_t> output_rows;
                std::vector<int32_t> queries;
                std::vector<int32_t> keys;
                std::vector<int32_t> previous_queries;
                std::vector<int32_t> previous_keys;
            };
            for (int64_t t = -time_shift; t < frames; t += sol_block_size) {
                const int64_t first       = std::max<int64_t>(0, t);
                const int64_t end         = std::min(frames, t + sol_block_size);
                const int64_t first_block = first / sol_block_size;
                const int64_t last_block  = (end - 1) / sol_block_size;
                std::map<int64_t, std::vector<Window>> groups;
                for (int64_t y = -shift; y < height; y += sol_window_size) {
                    for (int64_t x = -shift; x < width; x += sol_window_size) {
                        Window window;
                        for (int64_t i = first; i < end; ++i) {
                            for (int64_t j = std::max<int64_t>(0, y); j < std::min(height, y + sol_window_size); ++j) {
                                for (int64_t k = std::max<int64_t>(0, x); k < std::min(width, x + sol_window_size); ++k) {
                                    const int64_t row = (i * height + j) * width + k;
                                    window.output_rows.push_back(static_cast<int32_t>(row));
                                    const int64_t block = i / sol_block_size;
                                    auto& queries       = block == last_block ? window.queries : window.previous_queries;
                                    queries.push_back(static_cast<int32_t>(row - block * block_rows));
                                }
                            }
                        }
                        window.keys          = window.queries;
                        window.previous_keys = window.previous_queries;
                        for (int64_t block = first_block; block <= last_block; ++block) {
                            const int64_t start = std::min(block_rows, pixels - block * block_rows);
                            auto& keys          = block == last_block ? window.keys : window.previous_keys;
                            for (int64_t i = 0; i < sol_global_rows; ++i) {
                                keys.push_back(static_cast<int32_t>(start + i));
                            }
                        }
                        groups[window.output_rows.size()].push_back(std::move(window));
                    }
                }
                // Bound the workspace of backends that fall back to unfused attention.
                constexpr size_t max_windows = 4;
                for (const auto& group : groups) {
                    const auto& windows = group.second;
                    for (size_t first_window = 0; first_window < windows.size(); first_window += max_windows) {
                        const auto n      = static_cast<int64_t>(std::min(max_windows, windows.size() - first_window));
                        auto pack_windows = [&](auto member) -> sd::Tensor<int32_t> {
                            const auto rows = static_cast<int64_t>((windows[first_window].*member).size());
                            if (rows == 0) {
                                return {};
                            }
                            sd::Tensor<int32_t> indices({rows, n});
                            for (int64_t i = 0; i < n; ++i) {
                                const auto& source = windows[first_window + i].*member;
                                std::copy(source.begin(), source.end(), indices.data() + i * rows);
                            }
                            return indices;
                        };
                        blocks[last_block].batches.push_back({pack_windows(&Window::output_rows),
                                                              pack_windows(&Window::queries),
                                                              pack_windows(&Window::keys),
                                                              pack_windows(&Window::previous_queries),
                                                              pack_windows(&Window::previous_keys)});
                    }
                }
            }
        }

        struct Inputs {
            std::vector<Block<ggml_tensor*>> blocks;
            ggml_tensor* row_ids = nullptr;

            template <typename Project>
            ggml_tensor* forward(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* pe,
                                 int64_t heads,
                                 Project project) const {
                auto ctx          = runner_ctx->ggml_ctx;
                const int64_t dim = x->ne[0];
                auto output       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, x->ne[1]);
                auto positions    = ggml_reshape_2d(ctx, pe, pe->ne[0] * pe->ne[1] * pe->ne[2], pe->ne[3]);
                auto gather       = [&](ggml_tensor* rows, ggml_tensor* indices) {
                    auto flat = ggml_reshape_1d(ctx, indices, ggml_nelements(indices));
                    return ggml_reshape_3d(ctx, ggml_get_rows(ctx, rows, flat), dim, indices->ne[0], indices->ne[1]);
                };
                std::array<ggml_tensor*, 3> previous{};
                for (const auto& block : blocks) {
                    auto local_pe = ggml_reshape_4d(ctx, ggml_get_rows(ctx, positions, block.rows),
                                                    pe->ne[0], pe->ne[1], pe->ne[2], block.rows->ne[0]);
                    auto current  = project(ggml_get_rows(ctx, x, block.rows), local_pe);
                    for (const auto& batch : block.batches) {
                        // Gather first so a crossing window never copies either full block.
                        auto window = [&](size_t i, ggml_tensor* now, ggml_tensor* before) {
                            // Global queries read the whole native block in its existing order.
                            auto rows = now == nullptr ? current[i] : gather(current[i], now);
                            if (before != nullptr) {
                                rows = ggml_concat(ctx, gather(previous[i], before), rows, 1);
                            }
                            return rows;
                        };
                        auto attended = ggml_ext_attention_ext(runner_ctx,
                                                               window(0, batch.queries, batch.previous_queries),
                                                               window(1, batch.keys, batch.previous_keys),
                                                               window(2, batch.keys, batch.previous_keys),
                                                               heads, nullptr, false, runner_ctx->flash_attn_enabled);
                        auto indices  = ggml_reshape_1d(ctx, batch.output_rows, ggml_nelements(batch.output_rows));
                        output        = ggml_set_rows(ctx, output, ggml_reshape_2d(ctx, attended, dim, indices->ne[0]), indices);
                        runner_ctx->expand_graph(output);
                    }
                    // Later windows can only read this block and its immediate successor.
                    previous = current;
                }
                return output;
            }
        };

        template <typename BindInput>
        Inputs bind(BindInput bind_input) const {
            Inputs inputs;
            for (const auto& block : blocks) {
                Block<ggml_tensor*> bound{bind_input(block.rows), {}};
                for (const auto& batch : block.batches) {
                    auto bind_optional = [&](const auto& indices) {
                        return indices.empty() ? nullptr : bind_input(indices);
                    };
                    bound.batches.push_back({bind_input(batch.output_rows), bind_input(batch.queries),
                                             bind_optional(batch.keys), bind_optional(batch.previous_queries),
                                             bind_optional(batch.previous_keys)});
                }
                inputs.blocks.push_back(std::move(bound));
            }
            return inputs;
        }
    };

}  // namespace MiniMaxH3VAE

#endif  // __SD_MODEL_VAE_MINIMAX_H3_SOL_ATTENTION_HPP__

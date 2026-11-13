#ifndef __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__
#define __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/rng_mt19937.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/vae/minimax_h3_sol_attention.hpp"
#include "model/vae/vae.hpp"

namespace MiniMaxH3VAE {

    constexpr int H3_VIDEO_VAE_GRAPH_SIZE = 262144;

    static int64_t sol_output_frames(int64_t latent_frames, int patch_size_t) {
        const int64_t blocks = (latent_frames + sol_block_size - 1) / sol_block_size;
        return latent_frames * patch_size_t - blocks * (patch_size_t - 1);
    }

    struct MiniMaxH3VAEConfig {
        int num_layers      = 36;
        int patch_size_t    = 4;
        int hidden_size     = 2048;
        bool sol            = false;
        bool vit_decoder_2d = false;

        static MiniMaxH3VAEConfig detect_from_weights(const String2TensorStorage& tensors, const std::string& prefix) {
            MiniMaxH3VAEConfig config;
            config.sol = tensors.count(prefix + ".decoder.blocks.0.scale1") != 0;
            auto proj  = tensors.find(prefix + ".decoder.proj_out.weight");
            if (const auto embedder = tensors.find(prefix + ".decoder.x_embedder.weight"); embedder != tensors.end()) {
                config.hidden_size = static_cast<int>(embedder->second.ne[1]);
            }
            if (proj != tensors.end()) {
                config.patch_size_t = static_cast<int>(proj->second.ne[1] / (3 * 16 * 16));
                if (config.patch_size_t != 1 && config.patch_size_t != 4) {
                    throw std::runtime_error("unsupported MiniMax-H3 VAE temporal patch size");
                }
                int layers              = 0;
                const auto layer_prefix = prefix + (config.sol ? ".decoder.blocks." : ".decoder.transformer_blocks.");
                for (const auto& [name, _] : tensors) {
                    if (starts_with(name, layer_prefix)) {
                        layers = std::max(layers, std::atoi(name.c_str() + layer_prefix.size()) + 1);
                    }
                }
                if (layers > 0) {
                    config.num_layers = layers;
                }
            }
            config.vit_decoder_2d = config.patch_size_t == 1 && config.hidden_size == 1024;
            if (config.sol || proj != tensors.end()) {
                LOG_VERBOSE("minimax_h3_vae: decoder=%s, layers=%d, patch_size_t=%d, hidden_size=%d",
                            config.sol ? "sol" : "original",
                            config.num_layers,
                            config.patch_size_t,
                            config.hidden_size);
            }
            return config;
        }
    };

    struct CausalConv3d : public Conv3d {
        std::tuple<int, int, int> temporal_padding;

        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride  = {1, 1, 1},
                     std::tuple<int, int, int> padding = {0, 0, 0})
            : Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     {0, 0, 0}),
              temporal_padding(padding) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto reflect_pad = [&](ggml_tensor* value, int dim, int amount) {
                for (int i = 0; i < amount; ++i) {
                    GGML_ASSERT(value->ne[dim] > 1);
                    auto left  = ggml_ext_slice(ctx, value, dim, 1, 2);
                    auto right = ggml_ext_slice(ctx,
                                                value,
                                                dim,
                                                value->ne[dim] - 2,
                                                value->ne[dim] - 1);
                    value      = ggml_concat(ctx->ggml_ctx, left, value, dim);
                    value      = ggml_concat(ctx->ggml_ctx, value, right, dim);
                }
                return value;
            };

            x                = reflect_pad(x, 0, std::get<2>(temporal_padding));
            x                = reflect_pad(x, 1, std::get<1>(temporal_padding));
            int temporal_pad = std::get<0>(temporal_padding) * 2;
            if (temporal_pad > 0) {
                x = ggml_ext_pad_ext(ctx,
                                     x,
                                     0,
                                     0,
                                     0,
                                     0,
                                     temporal_pad,
                                     0,
                                     0,
                                     0);
            }
            return Conv3d::forward(ctx, x);
        }
    };

    struct TemporalGroupNorm : public GroupNorm {
        explicit TemporalGroupNorm(int64_t channels)
            : GroupNorm(32, channels, 1e-6f, true) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* result = nullptr;
            for (int64_t t = 0; t < x->ne[2]; ++t) {
                auto frame = ggml_ext_slice(ctx, x, 2, t, t + 1);
                GGML_ASSERT(frame->ne[3] % num_channels == 0);
                int64_t batch_size = frame->ne[3] / num_channels;
                frame              = ggml_cont(ctx->ggml_ctx, frame);
                frame              = ggml_reshape_4d(ctx->ggml_ctx,
                                                     frame,
                                                     frame->ne[0],
                                                     frame->ne[1],
                                                     num_channels,
                                                     batch_size);
                frame              = GroupNorm::forward(ctx, frame);
                frame              = ggml_reshape_4d(ctx->ggml_ctx,
                                                     frame,
                                                     frame->ne[0],
                                                     frame->ne[1],
                                                     1,
                                                     num_channels * batch_size);
                result             = result == nullptr ? frame : ggml_concat(ctx->ggml_ctx, result, frame, 2);
            }
            return result;
        }
    };

    struct Downsample3D : public GGMLBlock {
        int spatial_stride;

        Downsample3D(int64_t in_channels,
                     int64_t out_channels,
                     int temporal_stride,
                     int spatial_stride)
            : spatial_stride(spatial_stride) {
            blocks["conv"] = std::make_shared<CausalConv3d>(in_channels,
                                                            out_channels,
                                                            std::tuple{3, 3, 3},
                                                            std::tuple{temporal_stride, spatial_stride, spatial_stride},
                                                            std::tuple{1, 0, 0});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (spatial_stride == 2) {
                GGML_ASSERT(x->ne[0] > 1 && x->ne[1] > 1);
                auto right  = ggml_ext_slice(ctx, x, 0, x->ne[0] - 2, x->ne[0] - 1);
                x           = ggml_concat(ctx->ggml_ctx, x, right, 0);
                auto bottom = ggml_ext_slice(ctx, x, 1, x->ne[1] - 2, x->ne[1] - 1);
                x           = ggml_concat(ctx->ggml_ctx, x, bottom, 1);
            }
            return std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"])->forward(ctx, x);
        }
    };

    struct ResnetBlock3D : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        ResnetBlock3D(int64_t in_channels,
                      int64_t out_channels)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<TemporalGroupNorm>(in_channels);
            blocks["norm2"] = std::make_shared<TemporalGroupNorm>(out_channels);
            blocks["conv1"] = std::make_shared<CausalConv3d>(in_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             std::tuple{1, 1, 1});
            blocks["conv2"] = std::make_shared<CausalConv3d>(out_channels,
                                                             out_channels,
                                                             std::tuple{3, 3, 3},
                                                             std::tuple{1, 1, 1},
                                                             std::tuple{1, 1, 1});
            if (in_channels != out_channels) {
                blocks["nin_shortcut"] = std::make_shared<CausalConv3d>(in_channels,
                                                                        out_channels,
                                                                        std::tuple{1, 1, 1});
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm2"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);
            auto h     = conv1->forward(ctx, ggml_silu(ctx->ggml_ctx, norm1->forward(ctx, x)));
            h          = conv2->forward(ctx, ggml_silu(ctx->ggml_ctx, norm2->forward(ctx, h)));
            if (in_channels != out_channels) {
                x = std::dynamic_pointer_cast<CausalConv3d>(blocks["nin_shortcut"])->forward(ctx, x);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct Encoder : public GGMLBlock {
        static constexpr int levels                            = 6;
        static constexpr std::array<int, levels> multipliers   = {1, 2, 2, 4, 4, 8};
        static constexpr std::array<int, levels> spatial_down  = {2, 2, 2, 2, 1, 1};
        static constexpr std::array<int, levels> temporal_down = {1, 2, 2, 1, 1, 1};

        Encoder() {
            constexpr int ch  = 128;
            blocks["conv_in"] = std::make_shared<CausalConv3d>(3,
                                                               ch,
                                                               std::tuple{3, 3, 3},
                                                               std::tuple{1, 1, 1},
                                                               std::tuple{1, 1, 1});
            int64_t previous  = ch;
            for (int level = 0; level < levels; ++level) {
                int64_t current = ch * multipliers[level];
                for (int block = 0; block < 2; ++block) {
                    blocks["down." + std::to_string(level) + ".block." + std::to_string(block)] =
                        std::make_shared<ResnetBlock3D>(block == 0 ? previous : current,
                                                        current);
                }
                if (spatial_down[level] * temporal_down[level] > 1) {
                    blocks["down." + std::to_string(level) + ".downsample"] =
                        std::make_shared<Downsample3D>(current,
                                                       current,
                                                       temporal_down[level],
                                                       spatial_down[level]);
                }
                previous = current;
            }
            blocks["norm_out"] = std::make_shared<TemporalGroupNorm>(previous);
            blocks["conv_out"] = std::make_shared<CausalConv3d>(previous,
                                                                48,
                                                                std::tuple{3, 3, 3},
                                                                std::tuple{1, 1, 1},
                                                                std::tuple{1, 1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"])->forward(ctx, x);
            for (int level = 0; level < levels; ++level) {
                for (int block = 0; block < 2; ++block) {
                    x = std::dynamic_pointer_cast<ResnetBlock3D>(
                            blocks["down." + std::to_string(level) + ".block." + std::to_string(block)])
                            ->forward(ctx, x);
                }
                auto downsample = blocks.find("down." + std::to_string(level) + ".downsample");
                if (downsample != blocks.end()) {
                    x = std::dynamic_pointer_cast<Downsample3D>(downsample->second)->forward(ctx, x);
                }
            }
            auto norm = std::dynamic_pointer_cast<TemporalGroupNorm>(blocks["norm_out"]);
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);
            return conv->forward(ctx, ggml_silu(ctx->ggml_ctx, norm->forward(ctx, x)));
        }
    };

    static ggml_tensor* attention_layout(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
    }

    static ggml_tensor* apply_partial_rope(GGMLRunnerContext* runner_ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* pe) {
        auto ctx        = runner_ctx->ggml_ctx;
        int64_t rot_dim = pe->ne[2] * 2;
        auto rotated    = Rope::apply_rope(runner_ctx,
                                           ggml_ext_slice(runner_ctx, x, 0, 0, rot_dim),
                                           pe,
                                           false);
        if (rot_dim == x->ne[0]) {
            return rotated;
        }
        auto tail = attention_layout(ctx,
                                     ggml_ext_slice(runner_ctx, x, 0, rot_dim, x->ne[0]));
        return ggml_concat(ctx, rotated, tail, 0);
    }

    struct DecoderAttention : public GGMLBlock {
        static constexpr int head_dim = 64;
        int dim;
        int num_head;

        explicit DecoderAttention(int dim = 2048)
            : dim(dim), num_head(dim / head_dim) {
            blocks["to_qkv"] = std::make_shared<Linear>(dim, dim * 3, true);
            blocks["to_out"] = std::make_shared<Linear>(dim, dim, true);
        }

        std::array<ggml_tensor*, 3> project(GGMLRunnerContext* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* pe) {
            auto to_qkv         = std::dynamic_pointer_cast<Linear>(blocks["to_qkv"]);
            auto qkv_projection = to_qkv->forward(ctx, x);
            int64_t sequence    = x->ne[1];
            int64_t batch_size  = x->ne[2] * x->ne[3];
            qkv_projection      = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv_projection,
                                                  3 * head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            auto qkv            = ggml_ext_chunk(ctx, qkv_projection, 3, 0);
            auto q              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[0],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            auto k              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[1],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            auto v              = ggml_reshape_4d(ctx->ggml_ctx,
                                                  qkv[2],
                                                  head_dim,
                                                  num_head,
                                                  sequence,
                                                  batch_size);
            q                   = ggml_rms_norm(ctx->ggml_ctx, q, 1e-5f);
            k                   = ggml_rms_norm(ctx->ggml_ctx, k, 1e-5f);
            q                   = apply_partial_rope(ctx, q, pe);
            k                   = apply_partial_rope(ctx, k, pe);
            return {q, k, v};
        }

        std::array<ggml_tensor*, 3> project_sol(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe) {
            auto to_qkv         = std::dynamic_pointer_cast<Linear>(blocks["to_qkv"]);
            auto qkv_projection = to_qkv->forward(ctx, x);
            qkv_projection      = ggml_reshape_4d(ctx->ggml_ctx, qkv_projection, 3 * head_dim, num_head, x->ne[1], x->ne[2] * x->ne[3]);
            auto qkv            = ggml_ext_chunk(ctx, qkv_projection, 3, 0, false);
            auto q              = ggml_rms_norm(ctx->ggml_ctx, qkv[0], 1e-5f);
            auto k              = ggml_rms_norm(ctx->ggml_ctx, qkv[1], 1e-5f);
            return {Rope::apply_rope(ctx, q, pe, false),
                    Rope::apply_rope(ctx, k, pe, false),
                    ggml_ext_cont(ctx, qkv[2])};
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             RMSNorm& norm,
                             const SolAttentionPlan::Inputs* sol) {
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);
            if (sol != nullptr) {
                auto project_rows = [&](ggml_tensor* rows, ggml_tensor* local_pe) {
                    auto [q, k, v] = project_sol(ctx, norm.forward(ctx, rows), local_pe);
                    auto as_rows   = [&](ggml_tensor* tensor) {
                        return ggml_reshape_2d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, tensor), dim, rows->ne[1]);
                    };
                    return std::array<ggml_tensor*, 3>{
                        as_rows(ggml_permute(ctx->ggml_ctx, q, 0, 2, 1, 3)),
                        as_rows(ggml_permute(ctx->ggml_ctx, k, 0, 2, 1, 3)),
                        as_rows(v)};
                };
                return to_out->forward(ctx, sol->forward(ctx, x, pe, num_head, project_rows));
            }
            auto [q, k, v] = project(ctx, norm.forward(ctx, x), pe);
            auto out       = ggml_ext_attention_ext(ctx,
                                                    q,
                                                    k,
                                                    v,
                                                    num_head,
                                                    nullptr,
                                                    true,
                                                    ctx->flash_attn_enabled);
            return to_out->forward(ctx, out);
        }
    };

    struct DecoderFeedForward : public GGMLBlock {
        explicit DecoderFeedForward(int dim = 2048) {
            blocks["w1"] = std::make_shared<Linear>(dim, dim * 8, true);
            blocks["w2"] = std::make_shared<Linear>(dim * 4, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1   = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2   = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto gate = ggml_swiglu(ctx->ggml_ctx, w1->forward(ctx, x));
            return w2->forward(ctx, gate);
        }
    };

    struct DecoderBlock : public GGMLBlock {
        int dim;

        explicit DecoderBlock(int dim = 2048)
            : dim(dim) {
            blocks["norm1"] = std::make_shared<RMSNorm>(dim, 1e-5f);
            blocks["attn"]  = std::make_shared<DecoderAttention>(dim);
            blocks["norm2"] = std::make_shared<RMSNorm>(dim, 1e-5f);
            blocks["ff"]    = std::make_shared<DecoderFeedForward>(dim);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["scale1"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["scale2"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             const SolAttentionPlan::Inputs* sol,
                             int64_t rows_per_chunk) {
            auto norm1  = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn   = std::dynamic_pointer_cast<DecoderAttention>(blocks["attn"]);
            auto norm2  = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto ff     = std::dynamic_pointer_cast<DecoderFeedForward>(blocks["ff"]);
            x           = ggml_add(ctx->ggml_ctx,
                                   x,
                                   ggml_mul(ctx->ggml_ctx,
                                            attn->forward(ctx, x, pe, *norm1, sol),
                                            params["scale1"]));
            auto finish = [&](ggml_tensor* rows) {
                auto update = ff->forward(ctx, norm2->forward(ctx, rows));
                return ggml_add(ctx->ggml_ctx, rows, ggml_mul(ctx->ggml_ctx, update, params["scale2"]));
            };
            if (rows_per_chunk == 0 || x->ne[1] <= rows_per_chunk) {
                return finish(x);
            }
            // Normalize and update each independent chunk before writing it to the layer output.
            auto output = ggml_new_tensor_2d(ctx->ggml_ctx, x->type, x->ne[0], x->ne[1]);
            for (int64_t row = 0; row < x->ne[1]; row += rows_per_chunk) {
                auto input   = ggml_ext_slice(ctx, x, 1, row, std::min(row + rows_per_chunk, x->ne[1]), false);
                auto indices = ggml_ext_slice(ctx, sol->row_ids, 0, row, std::min(row + rows_per_chunk, x->ne[1]), false);
                output       = ggml_set_rows(ctx->ggml_ctx, output, finish(input), indices);
                ctx->expand_graph(output);
            }
            return output;
        }
    };

    struct Decoder : public GGMLBlock {
        int dim;
        int num_layers;
        static constexpr int num_register_tokens = 4;
        static constexpr int patch_size          = 16;
        int patch_size_t;
        bool sol;

        explicit Decoder(const MiniMaxH3VAEConfig& config)
            : dim(config.hidden_size), num_layers(config.num_layers), patch_size_t(config.patch_size_t), sol(config.sol) {
            blocks["x_embedder"] = std::make_shared<Linear>(24, dim, true);
            for (int i = 0; i < num_layers; ++i) {
                blocks[(sol ? "blocks." : "transformer_blocks.") + std::to_string(i)] =
                    std::make_shared<DecoderBlock>(dim);
            }
            blocks["norm_out"] = std::make_shared<LayerNorm>(dim, 1e-5f, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(dim,
                                                          3 * patch_size_t * patch_size * patch_size,
                                                          true,
                                                          true);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["register_tokens"] = ggml_new_tensor_2d(ctx,
                                                           GGML_TYPE_F32,
                                                           dim,
                                                           num_register_tokens);
            if (patch_size_t != 1 && !sol) {
                params["mask_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* z,
                             ggml_tensor* pe,
                             ggml_tensor* shifted_pe,
                             const std::array<SolAttentionPlan::Inputs, 2>& sol_inputs,
                             ggml_tensor* pixel_mean,
                             ggml_tensor* pixel_std) {
            int64_t width      = z->ne[0];
            int64_t height     = z->ne[1];
            int64_t num_frames = z->ne[2];
            int64_t batch_size = z->ne[3] / 24;
            GGML_ASSERT(batch_size == 1);

            z                   = ggml_cont(ctx->ggml_ctx,
                                            ggml_ext_torch_permute(ctx, z, 3, 0, 1, 2));
            z                   = ggml_reshape_3d(ctx->ggml_ctx,
                                                  z,
                                                  24,
                                                  width * height * num_frames,
                                                  batch_size);
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto h              = x_embedder->forward(ctx, z);
            int64_t num_patches = h->ne[1];
            auto zero           = ggml_ext_scale(ctx,
                                                 ggml_ext_slice(ctx, h, 1, 0, 1),
                                                 0.f);
            auto registers      = ggml_concat(ctx->ggml_ctx, params["register_tokens"], zero, 1);
            if (sol) {
                const int64_t num_blocks = (num_frames + sol_block_size - 1) / sol_block_size;
                registers                = ggml_repeat(ctx->ggml_ctx, registers,
                                                       ggml_new_tensor_2d(ctx->ggml_ctx, registers->type, dim, num_blocks * sol_global_rows));
            }
            h = ggml_concat(ctx->ggml_ctx, h, registers, 1);

            const int64_t rows_per_chunk = sol ? width * height + sol_global_rows : 0;
            for (int i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<DecoderBlock>(
                    blocks[(sol ? "blocks." : "transformer_blocks.") + std::to_string(i)]);
                h = block->forward(ctx, h, sol && i % 2 != 0 ? shifted_pe : pe, sol ? &sol_inputs[i % 2] : nullptr, rows_per_chunk);
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "minimax_h3_vae.decoder.blocks." + std::to_string(i),
                                                   "hidden_states");
            }

            auto norm_out              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto proj_out              = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto denormalize_and_clamp = [&](ggml_tensor* pixels) {
                auto scaled = ggml_mul(ctx->ggml_ctx, pixels, pixel_std);
                return ggml_clamp(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, scaled, pixel_mean), 0.f, 1.f);
            };
            if (sol) {
                const int64_t frame_rows    = width * height;
                const int64_t output_frames = sol_output_frames(num_frames, patch_size_t);
                const int64_t frame_size    = width * height * patch_size * patch_size;
                const int64_t channels      = pixel_mean->ne[3];
                auto output                 = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32, frame_size, channels * output_frames);
                int64_t frame_offset        = 0;
                for (int64_t t = 0; t < num_frames; t += sol_block_size) {
                    const int64_t end = std::min(t + sol_block_size, num_frames);
                    auto rows         = ggml_ext_slice(ctx, h, 1, t * frame_rows, end * frame_rows, false);
                    auto patches      = proj_out->forward(ctx, norm_out->forward(ctx, rows));
                    auto pixels       = DiT::unpatchify_3d(ctx, patches, end - t, height, width,
                                                           patch_size_t, patch_size, patch_size, true);
                    // Keep only the last frame of the first latent patch in each native block.
                    pixels              = ggml_ext_slice(ctx, pixels, 2, patch_size_t - 1, pixels->ne[2], false);
                    pixels              = denormalize_and_clamp(pixels);
                    const int64_t count = pixels->ne[2];
                    for (int64_t c = 0; c < channels; ++c) {
                        auto plane          = ggml_ext_slice(ctx, pixels, 3, c, c + 1, false);
                        auto frames         = ggml_reshape_2d(ctx->ggml_ctx, ggml_ext_cont(ctx, plane), frame_size, count);
                        const int64_t start = c * output_frames + frame_offset;
                        auto indices        = ggml_ext_slice(ctx, sol_inputs[0].row_ids, 0, start, start + count, false);
                        output              = ggml_set_rows(ctx->ggml_ctx, output, frames, indices);
                    }
                    // Write each block before expanding the next block's pixel graph.
                    ctx->expand_graph(output);
                    frame_offset += count;
                }
                return ggml_reshape_4d(ctx->ggml_ctx, output, width * patch_size, height * patch_size, output_frames, channels);
            }
            h           = proj_out->forward(ctx, norm_out->forward(ctx, h));
            h           = ggml_ext_slice(ctx, h, 1, 0, num_patches);
            auto pixels = DiT::unpatchify_3d(ctx, h, num_frames, height, width,
                                             patch_size_t, patch_size, patch_size, true);
            return denormalize_and_clamp(pixels);
        }
    };

    struct MiniMaxH3VideoVAE : public GGMLBlock {
        bool single_frame;
        bool vit_decoder_2d;

        explicit MiniMaxH3VideoVAE(const MiniMaxH3VAEConfig& config)
            : single_frame(config.patch_size_t == 1), vit_decoder_2d(config.vit_decoder_2d) {
            blocks["encoder"]         = std::make_shared<Encoder>();
            blocks["quant_conv"]      = std::make_shared<Conv3d>(48, 48, std::tuple{1, 1, 1});
            blocks["post_quant_conv"] = std::make_shared<Conv3d>(24,
                                                                 24,
                                                                 std::tuple{1, 1, 1});
            if (vit_decoder_2d) {
                blocks["post_quant_conv"] = std::make_shared<Conv2d>(24, 24, std::pair{1, 1});
            }
            blocks["decoder"] = std::make_shared<Decoder>(config);
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* pixels,
                            ggml_tensor* pixel_mean,
                            ggml_tensor* pixel_std) {
            pixels       = ggml_div(ctx->ggml_ctx,
                                    ggml_sub(ctx->ggml_ctx, pixels, pixel_mean),
                                    pixel_std);
            auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
            auto quant   = std::dynamic_pointer_cast<Conv3d>(blocks["quant_conv"]);
            auto moments = quant->forward(ctx, encoder->forward(ctx, pixels));
            return single_frame ? moments : ggml_ext_slice(ctx, moments, 3, 0, 24);
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* latent,
                            ggml_tensor* pe,
                            ggml_tensor* shifted_pe,
                            const std::array<SolAttentionPlan::Inputs, 2>& sol_inputs,
                            ggml_tensor* pixel_mean,
                            ggml_tensor* pixel_std) {
            auto post_quant = std::dynamic_pointer_cast<UnaryBlock>(blocks["post_quant_conv"]);
            auto decoder    = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);
            if (vit_decoder_2d) {
                GGML_ASSERT(latent->ne[2] == 1);
                latent = ggml_reshape_4d(ctx->ggml_ctx, latent, latent->ne[0], latent->ne[1], 24, latent->ne[3] / 24);
            }
            latent = post_quant->forward(ctx, latent);
            if (vit_decoder_2d) {
                latent = ggml_reshape_4d(ctx->ggml_ctx, latent, latent->ne[0], latent->ne[1], 1, latent->ne[2] * latent->ne[3]);
            }
            return decoder->forward(ctx, latent, pe, shifted_pe, sol_inputs, pixel_mean, pixel_std);
        }
    };

    struct MiniMaxH3VideoVAERunner : public VAE {
        MiniMaxH3VAEConfig config;
        MiniMaxH3VideoVAE model;
        sd::Tensor<float> pixel_mean;
        sd::Tensor<float> pixel_std;
        sd::Tensor<float> latents_mean;
        sd::Tensor<float> latents_std;
        sd::Tensor<float> rope_cache;

        MiniMaxH3VideoVAERunner(ggml_backend_t backend,
                                const String2TensorStorage& tensor_storage_map,
                                const std::string& prefix                           = "first_stage_model",
                                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : VAE(VERSION_MINIMAX_H3, backend, prefix, weight_manager),
              config(MiniMaxH3VAEConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config),
              pixel_mean({1, 1, 1, 3}, {0.485f, 0.456f, 0.406f}),
              pixel_std({1, 1, 1, 3}, {0.229f, 0.224f, 0.225f}),
              latents_mean({1, 1, 1, 24},
                           {0.858090341091156f, -0.960659146308899f, 1.066164016723633f, -0.509032547473907f,
                            -0.272758185863495f, -1.367541432380676f, -0.255325496196747f, -0.269075542688370f,
                            -0.537684082984924f, -0.046409729868174f, 0.665737032890320f, 0.196901276707649f,
                            -0.546060800552368f, -0.403534203767776f, -0.236830249428749f, 0.259284526109695f,
                            -0.301339447498322f, 0.211341992020607f, -1.120684862136841f, 0.358193337917328f,
                            -0.042251437902451f, 0.260482996702194f, 0.228640928864479f, 0.705603182315826f}),
              latents_std({1, 1, 1, 24},
                          {1.222377419471741f, 1.276726365089417f, 1.683177471160889f, 1.754945516586304f,
                           1.563621640205383f, 2.194143533706665f, 0.965313792228699f, 1.056988596916199f,
                           0.841948926448822f, 0.772995293140411f, 1.895593762397766f, 0.946841835975647f,
                           0.799680948257446f, 0.449889004230499f, 0.719739973545075f, 0.693629324436188f,
                           2.961095094680786f, 2.769419908523560f, 3.049618482589722f, 2.108805418014527f,
                           3.276226282119751f, 3.162735700607300f, 2.281681299209595f, 2.612784385681153f}) {
            scale_input = false;
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            if (config.sol) {
                return "sol_h3_vae";
            }
            if (config.vit_decoder_2d) {
                return "minimax_h3_image_vae";
            }
            return config.patch_size_t == 1 ? "p_h3_image_vae" : "minimax_h3_video_vae";
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return config.patch_size_t == 1 ? 48 : 24;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            model.get_param_tensors(tensors, weight_prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            if (config.patch_size_t == 1) {
                auto mean   = sd::ops::slice(vae_output, 3, 0, 24);
                auto logvar = sd::ops::slice(vae_output, 3, 24, 48);
                auto noise  = MT19937RNG(42).randn(static_cast<uint32_t>(mean.numel()));
                for (int64_t i = 0; i < mean.numel(); ++i) {
                    mean[i] += std::exp(0.5f * std::clamp(logvar[i], -30.f, 20.f)) * noise[i];
                }
                return mean;
            }
            return vae_output;
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents * latents_std + latents_mean;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return (latents - latents_mean) / latents_std;
        }

        static sd::Tensor<float> ensure_video_shape(const sd::Tensor<float>& tensor) {
            if (tensor.dim() == 5) {
                return tensor;
            }
            GGML_ASSERT(tensor.dim() == 4);
            return tensor.reshape({tensor.shape()[0],
                                   tensor.shape()[1],
                                   1,
                                   tensor.shape()[2],
                                   tensor.shape()[3]});
        }

        sd_tiling_params_t resolve_tiling_params(sd_tiling_params_t params) const override {
            return config.sol ? params : resolve_original_tiling_params(params);
        }

        static sd_tiling_params_t resolve_original_tiling_params(sd_tiling_params_t params) {
            if (!params.enabled) {
                params.target_overlap = 0.25f;
            }
            if (params.tile_size_w == 0 && params.rel_size_w == 0.f) {
                params.tile_size_w = 256;
            }
            if (params.tile_size_h == 0 && params.rel_size_h == 0.f) {
                params.tile_size_h = 256;
            }
            params.enabled         = true;
            params.temporal_tiling = false;
            return params;
        }

        static sd::Tensor<float> repeat_last_frame(const sd::Tensor<float>& input,
                                                   int64_t count) {
            auto result = input;
            auto last   = sd::ops::slice(input, 2, input.shape()[2] - 1, input.shape()[2]);
            for (int64_t i = 0; i < count; ++i) {
                result = sd::ops::concat(result, last, 2);
            }
            return result;
        }

        static sd::Tensor<float> blend_temporal(const sd::Tensor<float>& previous,
                                                const sd::Tensor<float>& current,
                                                int64_t extent) {
            auto output            = current;
            extent                 = std::min({extent, previous.shape()[2], current.shape()[2]});
            int64_t previous_start = previous.shape()[2] - extent;
            for (int64_t b = 0; b < current.shape()[4]; ++b) {
                for (int64_t c = 0; c < current.shape()[3]; ++c) {
                    for (int64_t t = 0; t < extent; ++t) {
                        float wb = static_cast<float>(t) / extent;
                        float wa = 1.f - wb;
                        for (int64_t h = 0; h < current.shape()[1]; ++h) {
                            for (int64_t w = 0; w < current.shape()[0]; ++w) {
                                output.index(w, h, t, c, b) =
                                    previous.index(w, h, previous_start + t, c, b) * wa +
                                    current.index(w, h, t, c, b) * wb;
                            }
                        }
                    }
                }
            }
            return output;
        }

        sd::Tensor<float> encode(int n_threads,
                                 const sd::Tensor<float>& x,
                                 sd_tiling_params_t tiling_params,
                                 bool circular_x = false,
                                 bool circular_y = false) override {
            auto input  = ensure_video_shape(x);
            auto tiling = resolve_original_tiling_params(tiling_params);
            if (config.patch_size_t == 1 && input.shape()[2] != 1) {
                LOG_ERROR("H3 image VAE encodes single-frame images only");
                return {};
            }
            if (input.shape()[2] == 1) {
                auto encoded = VAE::encode(n_threads, input, tiling, circular_x, circular_y);
                if (!encoded.empty() && encoded.shape()[2] > 1) {
                    encoded = sd::ops::slice(encoded,
                                             2,
                                             encoded.shape()[2] - 1,
                                             encoded.shape()[2]);
                }
                return encoded;
            }

            int64_t pad = (-input.shape()[2]) % 17;
            if (pad < 0) {
                pad += 17;
            }
            if (pad > 0) {
                input = repeat_last_frame(input, pad);
            }
            auto plan   = make_vae_temporal_tile_plan(input.shape()[2], {17, 0});
            auto result = process_vae_temporal_tiles(input, plan, [&](const sd::Tensor<float>& chunk, const VAETemporalTile& tile) {
                SD_UNUSED(tile);
                // keep the runner alive across chunks; ending it here would
                // evict and reload the encoder weights every chunk
                return VAE::encode(n_threads, chunk, tiling, circular_x, circular_y, false);
            });
            runner_end();
            if (result.empty()) {
                return {};
            }
            if (result.shape()[2] > 3) {
                result = sd::ops::slice(result, 2, 0, result.shape()[2] - 3);
            }
            return result;
        }

        bool supports_temporal_tiling(VAETemporalDirection direction) const override {
            return config.sol && direction == VAETemporalDirection::DECODE;
        }

        int get_default_temporal_tile_frames(VAETemporalDirection direction) const override {
            SD_UNUSED(direction);
            return 2 * sol_block_size;
        }

        int get_default_temporal_tile_overlap(VAETemporalDirection direction) const override {
            SD_UNUSED(direction);
            return sol_block_size;
        }

        sd::Tensor<float> _compute_temporal_tiled(int n_threads,
                                                  const sd::Tensor<float>& input,
                                                  VAETemporalDirection direction,
                                                  const VAETemporalTilingConfig& tiling) override {
            GGML_ASSERT(supports_temporal_tiling(direction));
            if (tiling.tile_frames % sol_block_size || tiling.overlap % sol_block_size ||
                tiling.overlap >= tiling.tile_frames) {
                LOG_ERROR("Sol-H3-VAE temporal tile size and overlap must be multiples of %d latent frames, with overlap smaller than the tile",
                          static_cast<int>(sol_block_size));
                return {};
            }
            if (input.shape()[2] <= tiling.tile_frames) {
                return _compute(n_threads, input, true);
            }
            auto plan = make_vae_temporal_tile_plan(input.shape()[2], tiling);
            // With block-aligned starts, the same mapping gives tile lengths and output offsets.
            auto frame_count = [&](int64_t frames) { return sol_output_frames(frames, config.patch_size_t); };
            return process_vae_temporal_tiles_blended(input, plan, frame_count, frame_count,
                                                      [&](const sd::Tensor<float>& tile, const VAETemporalTile& window) {
                                                          LOG_VERBOSE("Sol-H3-VAE temporal tile %d/%d: latent frames [%lld, %lld)",
                                                                      window.index + 1, static_cast<int>(plan.tiles.size()),
                                                                      static_cast<long long>(window.start), static_cast<long long>(window.end));
                                                          return _compute(n_threads, tile, true);
                                                      });
        }

        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& x,
                                 sd_tiling_params_t tiling_params,
                                 bool decode_video = false,
                                 bool circular_x   = false,
                                 bool circular_y   = false,
                                 bool silent       = false) override {
            auto input = ensure_video_shape(x);
            if (config.sol) {
                circular_x = circular_x || circular_x_enabled;
                circular_y = circular_y || circular_y_enabled;
                if (circular_x || circular_y) {
                    int tile_w = static_cast<int>(input.shape()[0]);
                    int tile_h = static_cast<int>(input.shape()[1]);
                    float overlap;
                    if (tiling_params.enabled &&
                        !get_tile_sizes(tile_w, tile_h, overlap, tiling_params, input.shape()[0], input.shape()[1])) {
                        return {};
                    }
                    if ((circular_x && tile_w >= input.shape()[0]) || (circular_y && tile_h >= input.shape()[1])) {
                        LOG_ERROR("Sol-H3-VAE circular decoding requires spatial tiles smaller than the input on every circular axis");
                        return {};
                    }
                }
                return VAE::decode(n_threads, input, tiling_params, decode_video, circular_x, circular_y, silent);
            }
            auto tiling = resolve_tiling_params(tiling_params);
            if (config.patch_size_t == 1 && input.shape()[2] != 1) {
                LOG_ERROR("H3 image VAE decodes single-frame latents only");
                return {};
            }
            if (input.shape()[2] == 1) {
                auto decoded = VAE::decode(n_threads,
                                           input,
                                           tiling,
                                           decode_video,
                                           circular_x,
                                           circular_y,
                                           silent);
                if (!decoded.empty() && decoded.shape()[2] > 1) {
                    decoded = sd::ops::slice(decoded,
                                             2,
                                             decoded.shape()[2] - 1,
                                             decoded.shape()[2]);
                }
                return decoded;
            }

            constexpr int64_t tokens_per_chunk  = 5;
            constexpr int64_t token_drop        = 3;
            constexpr int64_t token_overlap     = 2;
            constexpr int64_t frames_per_chunk  = 20;
            constexpr int64_t frame_pre_padding = 3;
            constexpr int64_t frame_overlap     = 5;

            int64_t pseudo_tokens = input.shape()[2] + token_drop;
            int64_t pad_tokens    = (tokens_per_chunk - pseudo_tokens % tokens_per_chunk) % tokens_per_chunk;
            pseudo_tokens += pad_tokens;
            int64_t num_chunks = pseudo_tokens / tokens_per_chunk - 1;
            if (num_chunks < 1) {
                pad_tokens += tokens_per_chunk;
                num_chunks += 1;
            }
            if (pad_tokens > 0) {
                input = repeat_last_frame(input, pad_tokens);
            }

            sd::Tensor<float> overlap;
            auto plan = make_vae_temporal_tile_plan(
                input.shape()[2],
                {static_cast<int>(tokens_per_chunk + token_overlap), static_cast<int>(token_overlap)});
            GGML_ASSERT(plan.tiles.size() == static_cast<size_t>(num_chunks));
            auto result = process_vae_temporal_tiles(input, plan, [&](const sd::Tensor<float>& chunk, const VAETemporalTile& tile) {
                // keep the runner alive across chunks; ending it here would
                // evict and reload the decoder weights every chunk
                auto decoded = VAE::decode(n_threads,
                                           chunk,
                                           tiling,
                                           true,
                                           circular_x,
                                           circular_y,
                                           silent,
                                           false);
                if (decoded.empty()) {
                    return sd::Tensor<float>();
                }

                int64_t first_end = std::min<int64_t>(frames_per_chunk, decoded.shape()[2]);
                auto first        = sd::ops::slice(decoded,
                                                   2,
                                                   std::min<int64_t>(frame_pre_padding, first_end),
                                                   first_end);
                if (!overlap.empty()) {
                    first   = blend_temporal(overlap, first, frame_overlap);
                    overlap = {};
                }

                if (decoded.shape()[2] > frames_per_chunk + frame_pre_padding) {
                    overlap = sd::ops::slice(decoded,
                                             2,
                                             frames_per_chunk + frame_pre_padding,
                                             decoded.shape()[2]);
                }
                if (tile.last && !overlap.empty()) {
                    first   = sd::ops::concat(first, overlap, 2);
                    overlap = {};
                }
                return first;
            });
            runner_end();
            if (result.empty()) {
                return {};
            }

            int64_t expected_frames = input.shape()[2] <= 1 ? 1 : ((x.shape()[2] - 2) / 5) * 17 + 5;
            expected_frames         = std::max<int64_t>(1, expected_frames);
            if (result.shape()[2] > expected_frames) {
                result = sd::ops::slice(result, 2, 0, expected_frames);
            }
            return result;
        }

        sd::Tensor<float> build_rope(int64_t width,
                                     int64_t height,
                                     int64_t num_frames,
                                     bool shifted = false) {
            std::vector<std::vector<float>> ids;
            const int64_t global_rows = config.sol ? ((num_frames + sol_block_size - 1) / sol_block_size) * sol_global_rows : 5;
            ids.reserve(static_cast<size_t>(width * height * num_frames + global_rows));
            constexpr float two_pi      = 6.28318530717958647692f;
            const int64_t spatial_shift = shifted ? sol_spatial_shift : 0;
            const int64_t time_shift    = shifted ? sol_temporal_shift : 0;
            for (int64_t t = 0; t < num_frames; ++t) {
                const float time = config.sol ? ((t + time_shift) % sol_block_size + 0.5f) / sol_block_size
                                              : (config.patch_size_t == 1 ? 0.5f / 7.f : (t + 0.5f) / num_frames);
                float pt         = (2.f * time - 1.f) * two_pi;
                for (int64_t h = 0; h < height; ++h) {
                    const float y = config.sol ? ((h + spatial_shift) % sol_window_size + 0.5f) / sol_window_size : (h + 0.5f) / height;
                    float ph      = (2.f * y - 1.f) * two_pi;
                    for (int64_t w = 0; w < width; ++w) {
                        const float x = config.sol ? ((w + spatial_shift) % sol_window_size + 0.5f) / sol_window_size : (w + 0.5f) / width;
                        float pw      = (2.f * x - 1.f) * two_pi;
                        ids.push_back(config.vit_decoder_2d ? std::vector<float>{ph, pw} : std::vector<float>{pt, ph, pw});
                    }
                }
            }
            for (int64_t i = 0; i < global_rows; ++i) {
                ids.push_back(std::vector<float>(config.vit_decoder_2d ? 2 : 3, 0.f));
            }
            auto values = Rope::embed_nd(ids,
                                         1,
                                         100.f,
                                         config.vit_decoder_2d ? std::vector<int>{16, 16} : std::vector<int>{16, 16, 16});
            return sd::Tensor<float>({2,
                                      2,
                                      config.vit_decoder_2d ? 16 : 24,
                                      static_cast<int64_t>(ids.size())},
                                     std::move(values));
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            auto input = ensure_video_shape(z);
            std::unique_ptr<SolAttentionPlan> sol_plan;
            std::unique_ptr<SolAttentionPlan> shifted_plan;
            sd::Tensor<float> shifted_rope;
            sd::Tensor<int32_t> sol_row_ids;
            if (decode_graph) {
                rope_cache = build_rope(input.shape()[0],
                                        input.shape()[1],
                                        input.shape()[2]);
                if (config.sol) {
                    const auto& shape  = input.shape();
                    sol_plan           = std::make_unique<SolAttentionPlan>(shape[0], shape[1], shape[2], false);
                    shifted_plan       = std::make_unique<SolAttentionPlan>(shape[0], shape[1], shape[2], true);
                    shifted_rope       = build_rope(shape[0], shape[1], shape[2], true);
                    const int64_t rows = std::max(shifted_rope.shape()[3], pixel_mean.shape()[3] *
                                                                               sol_output_frames(shape[2], config.patch_size_t));
                    sol_row_ids        = sd::Tensor<int32_t>({rows});
                    for (int64_t i = 0; i < rows; ++i) {
                        sol_row_ids.data()[i] = static_cast<int32_t>(i);
                    }
                }
            }
            auto get_graph = [&]() -> ggml_cgraph* {
                auto value       = make_input(input);
                auto mean        = make_input(pixel_mean);
                auto std         = make_input(pixel_std);
                auto graph       = new_graph_custom(H3_VIDEO_VAE_GRAPH_SIZE);
                auto runner_ctx  = get_context(graph);
                ggml_tensor* out = nullptr;
                if (decode_graph) {
                    auto pe         = make_input(rope_cache);
                    auto shifted_pe = config.sol ? make_input(shifted_rope) : nullptr;
                    std::array<SolAttentionPlan::Inputs, 2> sol_inputs;
                    if (sol_plan) {
                        auto bind_input       = [&](const auto& tensor) { return make_input(tensor); };
                        sol_inputs            = {sol_plan->bind(bind_input), shifted_plan->bind(bind_input)};
                        auto row_ids          = make_input(sol_row_ids);
                        sol_inputs[0].row_ids = row_ids;
                        sol_inputs[1].row_ids = row_ids;
                    }
                    out = model.decode(&runner_ctx, value, pe, shifted_pe, sol_inputs, mean, std);
                } else {
                    out = model.encode(&runner_ctx, value, mean, std);
                }
                ggml_build_forward_expand(graph, out);
                return graph;
            };
            return restore_trailing_singleton_dims(
                GGMLRunner::compute(get_graph,
                                    n_threads,
                                    false),
                5);
        }
    };

}  // namespace MiniMaxH3VAE

#endif  // __SD_MODEL_VAE_MINIMAX_H3_VAE_HPP__

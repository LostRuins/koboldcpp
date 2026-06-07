#ifndef __SD_MODEL_DIFFUSION_PID_HPP__
#define __SD_MODEL_DIFFUSION_PID_HPP__

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/mmdit.hpp"

namespace Pid {
    constexpr int PID_GRAPH_SIZE = 196608;
    constexpr float PID_PI       = 3.14159265358979323846f;

    struct PixelDiTConfig {
        int64_t in_channels            = 3;
        int64_t hidden_size            = 1536;
        int64_t num_groups             = 24;
        int64_t patch_mlp_hidden_dim   = 4096;
        int64_t pixel_hidden_size      = 16;
        int64_t pixel_attn_hidden_size = 1152;
        int64_t pixel_num_groups       = 16;
        int64_t patch_depth            = 14;
        int64_t pixel_depth            = 2;
        int64_t patch_size             = 16;
        int64_t txt_embed_dim          = 2304;
        int64_t txt_max_length         = 300;
        float text_rope_theta          = 10000.f;
        int64_t lq_latent_channels     = 16;
        int64_t lq_hidden_dim          = 512;
        int64_t lq_num_res_blocks      = 4;
        int64_t lq_interval            = 2;
        int64_t lq_sr_scale            = 4;
        int64_t lq_latent_down_factor  = 8;
        int64_t rope_ref_grid_h        = 64;
        int64_t rope_ref_grid_w        = 64;

        static PixelDiTConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            PixelDiTConfig config;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t pos = name.find("patch_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index    = atoi(items[1].c_str());
                        config.patch_depth = std::max<int64_t>(config.patch_depth, block_index + 1);
                    }
                }
                pos = name.find("pixel_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index    = atoi(items[1].c_str());
                        config.pixel_depth = std::max<int64_t>(config.pixel_depth, block_index + 1);
                    }
                }
                if (name.find("lq_proj.latent_proj.0.weight") != std::string::npos) {
                    config.lq_latent_channels    = tensor_storage.ne[2];
                    config.lq_latent_down_factor = config.lq_latent_channels >= 64 ? 16 : 8;
                }
                if (name.find("patch_blocks.0.mlp_x.w1.weight") != std::string::npos) {
                    config.patch_mlp_hidden_dim = tensor_storage.ne[1];
                }
            }
            LOG_DEBUG("pid: patch_depth = %" PRId64 ", pixel_depth = %" PRId64 ", patch_mlp_hidden_dim = %" PRId64 ", lq_latent_channels = %" PRId64 ", lq_latent_down_factor = %" PRId64,
                      config.patch_depth,
                      config.pixel_depth,
                      config.patch_mlp_hidden_dim,
                      config.lq_latent_channels,
                      config.lq_latent_down_factor);
            return config;
        }
    };

    inline std::vector<float> make_rope_1d(int length,
                                           int dim,
                                           float theta) {
        GGML_ASSERT(dim % 2 == 0);
        return Rope::flatten(Rope::rope(Rope::linspace(0.f, static_cast<float>(length - 1), length), dim, theta));
    }

    inline std::vector<float> make_rope_2d(int height,
                                           int width,
                                           int dim,
                                           float theta    = 10000.f,
                                           float scale    = 16.f,
                                           int ref_grid_h = 0,
                                           int ref_grid_w = 0) {
        GGML_ASSERT(dim % 4 == 0);
        return Rope::embed_2d_interleaved(height, width, dim, theta, scale, ref_grid_h, ref_grid_w);
    }

    inline std::vector<float> make_pixel_abs_pos(int height,
                                                 int width,
                                                 int dim) {
        GGML_ASSERT(dim % 4 == 0);
        int half_dim = dim / 2;
        std::vector<float> x_pos;
        std::vector<float> y_pos;
        x_pos.reserve(static_cast<size_t>(height) * width);
        y_pos.reserve(static_cast<size_t>(height) * width);
        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                x_pos.push_back(static_cast<float>(ix));
                y_pos.push_back(static_cast<float>(iy));
            }
        }

        auto x_emb = timestep_embedding(x_pos, half_dim, 10000, false);
        auto y_emb = timestep_embedding(y_pos, half_dim, 10000, false);

        std::vector<float> out(static_cast<size_t>(dim) * height * width);
        for (int pos = 0; pos < height * width; ++pos) {
            size_t out_base = static_cast<size_t>(pos) * dim;
            size_t emb_base = static_cast<size_t>(pos) * half_dim;
            for (int i = 0; i < half_dim; ++i) {
                out[out_base + i]            = x_emb[emb_base + i];
                out[out_base + half_dim + i] = y_emb[emb_base + i];
            }
        }
        return out;
    }

    inline ggml_tensor* apply_adaln(ggml_context* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* shift,
                                    ggml_tensor* scale) {
        return ggml_add(ctx, ggml_add(ctx, x, ggml_mul(ctx, x, scale)), shift);
    }

    struct PatchTokenEmbedder : public GGMLBlock {
        bool use_rms_norm;

        PatchTokenEmbedder(int64_t in_chans,
                           int64_t embed_dim,
                           bool use_rms_norm = false,
                           bool bias         = true)
            : use_rms_norm(use_rms_norm) {
            blocks["proj"] = std::make_shared<Linear>(in_chans, embed_dim, bias);
            if (use_rms_norm) {
                blocks["norm"] = std::make_shared<RMSNorm>(embed_dim, 1e-6f);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            x         = proj->forward(ctx, x);
            if (use_rms_norm) {
                auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
                x         = norm->forward(ctx, x);
            }
            return x;
        }
    };

    struct PixelDiTTimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size;

        PixelDiTTimestepEmbedder(int64_t hidden_size,
                                 int frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size, 10);
            t_emb      = mlp_0->forward(ctx, t_emb);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            return mlp_2->forward(ctx, t_emb);
        }
    };

    struct FeedForward : public GGMLBlock {
        FeedForward(int64_t dim, int64_t hidden_dim) {
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);
            auto h  = ggml_silu_inplace(ctx->ggml_ctx, w1->forward(ctx, x));
            h       = ggml_mul_inplace(ctx->ggml_ctx, h, w3->forward(ctx, x));
            return w2->forward(ctx, h);
        }
    };

    struct FinalLayer : public GGMLBlock {
        FinalLayer(int64_t hidden_size, int64_t out_channels) {
            blocks["norm"]   = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, out_channels, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            return linear->forward(ctx, norm->forward(ctx, x));
        }
    };

    struct RotaryAttention : public GGMLBlock {
        int64_t dim;
        int64_t num_heads;

        RotaryAttention(int64_t dim, int64_t num_heads)
            : dim(dim), num_heads(num_heads) {
            int64_t head_dim = dim / num_heads;
            blocks["qkv"]    = std::make_shared<Linear>(dim, dim * 3, false);
            blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["proj"]   = std::make_shared<Linear>(dim, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pos) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto proj     = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            auto qkv         = qkv_proj->forward(ctx, x);
            auto qkv_vec     = split_qkv(ctx->ggml_ctx, qkv);
            int64_t L        = x->ne[1];
            int64_t N        = x->ne[2];
            int64_t head_dim = dim / num_heads;
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, L, N);
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, L, N);
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, L, N);
            q                = q_norm->forward(ctx, q);
            k                = k_norm->forward(ctx, k);
            x                = Rope::attention(ctx, q, k, v, pos, nullptr, 1.0f / 128.f, true);
            return proj->forward(ctx, x);
        }
    };

    struct MMDiTJointAttention : public GGMLBlock {
        int64_t dim;
        int64_t num_heads;

        MMDiTJointAttention(int64_t dim, int64_t num_heads)
            : dim(dim), num_heads(num_heads) {
            int64_t head_dim   = dim / num_heads;
            blocks["qkv_x"]    = std::make_shared<Linear>(dim, dim * 3, false);
            blocks["qkv_y"]    = std::make_shared<Linear>(dim, dim * 3, false);
            blocks["q_norm_x"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm_x"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["q_norm_y"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm_y"] = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["proj_x"]   = std::make_shared<Linear>(dim, dim, true);
            blocks["proj_y"]   = std::make_shared<Linear>(dim, dim, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* y,
                                                      ggml_tensor* pos_img,
                                                      ggml_tensor* pos_txt) {
            auto qkv_x_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_x"]);
            auto qkv_y_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_y"]);
            auto q_norm_x   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm_x"]);
            auto k_norm_x   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm_x"]);
            auto q_norm_y   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm_y"]);
            auto k_norm_y   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm_y"]);
            auto proj_x     = std::dynamic_pointer_cast<Linear>(blocks["proj_x"]);
            auto proj_y     = std::dynamic_pointer_cast<Linear>(blocks["proj_y"]);

            int64_t Nx       = x->ne[1];
            int64_t Ny       = y->ne[1];
            int64_t N        = x->ne[2];
            int64_t head_dim = dim / num_heads;

            auto qkv_x = split_qkv(ctx->ggml_ctx, qkv_x_proj->forward(ctx, x));
            auto qx    = ggml_reshape_4d(ctx->ggml_ctx, qkv_x[0], head_dim, num_heads, Nx, N);
            auto kx    = ggml_reshape_4d(ctx->ggml_ctx, qkv_x[1], head_dim, num_heads, Nx, N);
            auto vx    = ggml_reshape_4d(ctx->ggml_ctx, qkv_x[2], head_dim, num_heads, Nx, N);
            qx         = q_norm_x->forward(ctx, qx);
            kx         = k_norm_x->forward(ctx, kx);

            auto qkv_y = split_qkv(ctx->ggml_ctx, qkv_y_proj->forward(ctx, y));
            auto qy    = ggml_reshape_4d(ctx->ggml_ctx, qkv_y[0], head_dim, num_heads, Ny, N);
            auto ky    = ggml_reshape_4d(ctx->ggml_ctx, qkv_y[1], head_dim, num_heads, Ny, N);
            auto vy    = ggml_reshape_4d(ctx->ggml_ctx, qkv_y[2], head_dim, num_heads, Ny, N);
            qy         = q_norm_y->forward(ctx, qy);
            ky         = k_norm_y->forward(ctx, ky);

            auto q_joint   = ggml_concat(ctx->ggml_ctx, qy, qx, 2);
            auto k_joint   = ggml_concat(ctx->ggml_ctx, ky, kx, 2);
            auto v_joint   = ggml_concat(ctx->ggml_ctx, vy, vx, 2);
            auto pos_joint = ggml_concat(ctx->ggml_ctx, pos_txt, pos_img, 3);
            auto out       = Rope::attention(ctx, q_joint, k_joint, v_joint, pos_joint, nullptr, 1.0f, true);

            auto out_y = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, Ny);
            auto out_x = ggml_ext_slice(ctx->ggml_ctx, out, 1, Ny, Ny + Nx);
            return {proj_x->forward(ctx, out_x), proj_y->forward(ctx, out_y)};
        }
    };

    struct MMDiTBlockT2I : public GGMLBlock {
        int64_t hidden_size;

        MMDiTBlockT2I(int64_t hidden_size, int64_t groups, int64_t mlp_hidden_dim)
            : hidden_size(hidden_size) {
            blocks["norm_x1"]                = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["norm_y1"]                = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["attn"]                   = std::make_shared<MMDiTJointAttention>(hidden_size, groups);
            blocks["norm_x2"]                = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["norm_y2"]                = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["mlp_x"]                  = std::make_shared<FeedForward>(hidden_size, mlp_hidden_dim);
            blocks["mlp_y"]                  = std::make_shared<FeedForward>(hidden_size, mlp_hidden_dim);
            blocks["adaLN_modulation_img.0"] = std::make_shared<Linear>(hidden_size, 6 * hidden_size, true);
            blocks["adaLN_modulation_txt.0"] = std::make_shared<Linear>(hidden_size, 6 * hidden_size, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* y,
                                                      ggml_tensor* c,
                                                      ggml_tensor* pos_img,
                                                      ggml_tensor* pos_txt) {
            auto norm_x1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_x1"]);
            auto norm_y1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_y1"]);
            auto attn    = std::dynamic_pointer_cast<MMDiTJointAttention>(blocks["attn"]);
            auto norm_x2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_x2"]);
            auto norm_y2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_y2"]);
            auto mlp_x   = std::dynamic_pointer_cast<FeedForward>(blocks["mlp_x"]);
            auto mlp_y   = std::dynamic_pointer_cast<FeedForward>(blocks["mlp_y"]);
            auto ada_img = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation_img.0"]);
            auto ada_txt = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation_txt.0"]);

            auto mx = ggml_ext_chunk(ctx->ggml_ctx, ada_img->forward(ctx, c), 6, 0);
            auto my = ggml_ext_chunk(ctx->ggml_ctx, ada_txt->forward(ctx, c), 6, 0);

            auto x_norm   = apply_adaln(ctx->ggml_ctx, norm_x1->forward(ctx, x), mx[0], mx[1]);
            auto y_norm   = apply_adaln(ctx->ggml_ctx, norm_y1->forward(ctx, y), my[0], my[1]);
            auto attn_out = attn->forward(ctx, x_norm, y_norm, pos_img, pos_txt);

            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out.first, mx[2]));
            y = ggml_add(ctx->ggml_ctx, y, ggml_mul(ctx->ggml_ctx, attn_out.second, my[2]));

            auto x_mlp = mlp_x->forward(ctx, apply_adaln(ctx->ggml_ctx, norm_x2->forward(ctx, x), mx[3], mx[4]));
            auto y_mlp = mlp_y->forward(ctx, apply_adaln(ctx->ggml_ctx, norm_y2->forward(ctx, y), my[3], my[4]));
            x          = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, x_mlp, mx[5]));
            y          = ggml_add(ctx->ggml_ctx, y, ggml_mul(ctx->ggml_ctx, y_mlp, my[5]));
            return {x, y};
        }
    };

    struct PixelTokenEmbedder : public GGMLBlock {
        int64_t in_channels;
        int64_t hidden_size_output;

        PixelTokenEmbedder(int64_t in_channels, int64_t hidden_size_output)
            : in_channels(in_channels), hidden_size_output(hidden_size_output) {
            blocks["proj"] = std::make_shared<Linear>(in_channels, hidden_size_output, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* inputs,
                             int64_t patch_size,
                             ggml_tensor* pos_full) {
            auto proj  = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            int64_t W  = inputs->ne[0];
            int64_t H  = inputs->ne[1];
            int64_t B  = inputs->ne[3];
            int64_t L  = (W / patch_size) * (H / patch_size);
            int64_t P2 = patch_size * patch_size;

            auto x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, inputs, 2, 0, 1, 3));
            x      = ggml_reshape_3d(ctx->ggml_ctx, x, in_channels, W * H, B);
            x      = proj->forward(ctx, x);
            x      = ggml_add(ctx->ggml_ctx, x, pos_full);
            x      = ggml_reshape_4d(ctx->ggml_ctx, x, hidden_size_output, W, H, B);
            x      = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
            x      = DiT::patchify(ctx->ggml_ctx, x, static_cast<int>(patch_size), static_cast<int>(patch_size), false);
            x      = ggml_reshape_3d(ctx->ggml_ctx, x, hidden_size_output, P2, L * B);
            return x;
        }
    };

    struct PiTBlock : public GGMLBlock {
        int64_t pixel_dim;
        int64_t context_dim;
        int64_t attn_dim;
        int64_t num_heads;
        int64_t patch_size;

        PiTBlock(int64_t pixel_dim,
                 int64_t context_dim,
                 int64_t patch_size,
                 int64_t attn_dim,
                 int64_t num_heads)
            : pixel_dim(pixel_dim),
              context_dim(context_dim),
              attn_dim(attn_dim),
              num_heads(num_heads),
              patch_size(patch_size) {
            int64_t p2                   = patch_size * patch_size;
            blocks["compress_to_attn"]   = std::make_shared<Linear>(p2 * pixel_dim, attn_dim, true);
            blocks["expand_from_attn"]   = std::make_shared<Linear>(attn_dim, p2 * pixel_dim, true);
            blocks["norm1"]              = std::make_shared<RMSNorm>(pixel_dim, 1e-6f);
            blocks["attn"]               = std::make_shared<RotaryAttention>(attn_dim, num_heads);
            blocks["norm2"]              = std::make_shared<RMSNorm>(pixel_dim, 1e-6f);
            blocks["mlp"]                = std::make_shared<Mlp>(pixel_dim, pixel_dim * 4);
            blocks["adaLN_modulation.0"] = std::make_shared<Linear>(context_dim, 6 * pixel_dim * p2, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* s_cond,
                             int64_t image_height,
                             int64_t image_width,
                             ggml_tensor* pos_comp) {
            auto compress = std::dynamic_pointer_cast<Linear>(blocks["compress_to_attn"]);
            auto expand   = std::dynamic_pointer_cast<Linear>(blocks["expand_from_attn"]);
            auto norm1    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn     = std::dynamic_pointer_cast<RotaryAttention>(blocks["attn"]);
            auto norm2    = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto mlp      = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);
            auto ada      = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);

            int64_t Hs = image_height / patch_size;
            int64_t Ws = image_width / patch_size;
            int64_t L  = Hs * Ws;
            int64_t BL = x->ne[2];
            int64_t B  = BL / L;
            int64_t P2 = patch_size * patch_size;

            auto ada_params = ada->forward(ctx, s_cond);
            ada_params      = ggml_reshape_3d(ctx->ggml_ctx, ada_params, 6 * pixel_dim, P2, BL);
            auto mod        = ggml_ext_chunk(ctx->ggml_ctx, ada_params, 6, 0);

            auto x_norm    = apply_adaln(ctx->ggml_ctx, norm1->forward(ctx, x), mod[0], mod[1]);
            auto x_flat    = ggml_reshape_2d(ctx->ggml_ctx, x_norm, P2 * pixel_dim, BL);
            auto x_comp    = compress->forward(ctx, x_flat);
            x_comp         = ggml_reshape_3d(ctx->ggml_ctx, x_comp, attn_dim, L, B);
            auto attn_out  = attn->forward(ctx, x_comp, pos_comp);
            auto attn_flat = expand->forward(ctx, ggml_reshape_2d(ctx->ggml_ctx, attn_out, attn_dim, BL));
            auto attn_exp  = ggml_reshape_3d(ctx->ggml_ctx, attn_flat, pixel_dim, P2, BL);
            x              = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_exp, mod[2]));

            auto mlp_out = mlp->forward(ctx, apply_adaln(ctx->ggml_ctx, norm2->forward(ctx, x), mod[3], mod[4]));
            return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, mlp_out, mod[5]));
        }
    };

    struct SigmaAwareGate : public GGMLBlock {
        int64_t dim;

        SigmaAwareGate(int64_t dim)
            : dim(dim) {
            blocks["content_proj"] = std::make_shared<Linear>(dim * 2, dim, true);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         std::string prefix                             = "") override {
            params["log_alpha"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* lq,
                             ggml_tensor* sigma) {
            auto content_proj = std::dynamic_pointer_cast<Linear>(blocks["content_proj"]);

            auto content_logit = content_proj->forward(ctx, ggml_concat(ctx->ggml_ctx, x, lq, 0));
            sigma              = ggml_reshape_3d(ctx->ggml_ctx, sigma, 1, 1, sigma->ne[0]);
            auto alpha         = ggml_exp(ctx->ggml_ctx, params["log_alpha"]);
            auto offset        = ggml_neg(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, alpha, sigma));
            auto gate          = ggml_sigmoid(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, content_logit, offset));
            return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, gate, lq));
        }
    };

    struct PiDResBlock : public GGMLBlock {
        PiDResBlock(int64_t channels) {
            blocks["block.0"] = std::make_shared<GroupNorm>(4, channels, 1e-5f);
            blocks["block.2"] = std::make_shared<Conv2d>(channels, channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
            blocks["block.3"] = std::make_shared<GroupNorm>(4, channels, 1e-5f);
            blocks["block.5"] = std::make_shared<Conv2d>(channels, channels, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<GroupNorm>(blocks["block.0"]);
            auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["block.2"]);
            auto norm2 = std::dynamic_pointer_cast<GroupNorm>(blocks["block.3"]);
            auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["block.5"]);
            auto h     = ggml_silu_inplace(ctx->ggml_ctx, norm1->forward(ctx, x));
            h          = conv1->forward(ctx, h);
            h          = ggml_silu_inplace(ctx->ggml_ctx, norm2->forward(ctx, h));
            h          = conv2->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct LQProjection2D : public GGMLBlock {
        PixelDiTConfig config;

        LQProjection2D(const PixelDiTConfig& config)
            : config(config) {
            blocks["latent_proj.0"] = std::make_shared<Conv2d>(config.lq_latent_channels, config.lq_hidden_dim, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
            blocks["latent_proj.2"] = std::make_shared<Conv2d>(config.lq_hidden_dim, config.lq_hidden_dim, std::pair<int, int>{3, 3}, std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
            for (int i = 0; i < config.lq_num_res_blocks; ++i) {
                blocks["latent_proj." + std::to_string(3 + i)] = std::make_shared<PiDResBlock>(config.lq_hidden_dim);
            }

            int num_outputs = static_cast<int>((config.patch_depth + config.lq_interval - 1) / config.lq_interval);
            for (int i = 0; i < num_outputs; ++i) {
                blocks["output_heads." + std::to_string(i)] = std::make_shared<Linear>(config.lq_hidden_dim, config.hidden_size, true);
                blocks["gate_modules." + std::to_string(i)] = std::make_shared<SigmaAwareGate>(config.hidden_size);
            }
        }

        bool is_gate_active(int block_idx) const {
            return block_idx % config.lq_interval == 0;
        }

        int get_output_index(int block_idx) const {
            return block_idx / static_cast<int>(config.lq_interval);
        }

        ggml_tensor* gate(GGMLRunnerContext* ctx,
                          ggml_tensor* x,
                          ggml_tensor* lq,
                          ggml_tensor* sigma,
                          int out_idx) {
            auto gate_module = std::dynamic_pointer_cast<SigmaAwareGate>(blocks["gate_modules." + std::to_string(out_idx)]);
            return gate_module->forward(ctx, x, lq, sigma);
        }

        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                          ggml_tensor* lq_latent,
                                          int64_t target_pH,
                                          int64_t target_pW) {
            auto conv0             = std::dynamic_pointer_cast<Conv2d>(blocks["latent_proj.0"]);
            auto conv2             = std::dynamic_pointer_cast<Conv2d>(blocks["latent_proj.2"]);
            float z_to_patch_ratio = static_cast<float>(config.lq_sr_scale * config.lq_latent_down_factor) /
                                     static_cast<float>(config.patch_size);
            GGML_ASSERT(z_to_patch_ratio >= 1.0f);
            if (lq_latent->ne[0] != target_pW || lq_latent->ne[1] != target_pH) {
                lq_latent = ggml_interpolate(ctx->ggml_ctx,
                                             lq_latent,
                                             target_pW,
                                             target_pH,
                                             lq_latent->ne[2],
                                             lq_latent->ne[3],
                                             GGML_SCALE_MODE_NEAREST);
            }

            auto feat = conv0->forward(ctx, lq_latent);
            feat      = ggml_silu_inplace(ctx->ggml_ctx, feat);
            feat      = conv2->forward(ctx, feat);
            for (int i = 0; i < config.lq_num_res_blocks; ++i) {
                auto block = std::dynamic_pointer_cast<PiDResBlock>(blocks["latent_proj." + std::to_string(3 + i)]);
                feat       = block->forward(ctx, feat);
            }

            int64_t B   = feat->ne[3];
            int64_t C   = feat->ne[2];
            int64_t L   = target_pH * target_pW;
            auto tokens = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, feat, 2, 0, 1, 3));
            tokens      = ggml_reshape_3d(ctx->ggml_ctx, tokens, C, L, B);

            int num_outputs = static_cast<int>((config.patch_depth + config.lq_interval - 1) / config.lq_interval);
            std::vector<ggml_tensor*> outputs;
            outputs.reserve(num_outputs);
            for (int i = 0; i < num_outputs; ++i) {
                auto head = std::dynamic_pointer_cast<Linear>(blocks["output_heads." + std::to_string(i)]);
                outputs.push_back(head->forward(ctx, tokens));
            }
            return outputs;
        }
    };

    struct PixelDiT : public GGMLBlock {
        PixelDiTConfig config;

        PixelDiT() = default;

        PixelDiT(const PixelDiTConfig& config)
            : config(config) {
            blocks["pixel_embedder"] = std::make_shared<PixelTokenEmbedder>(config.in_channels, config.pixel_hidden_size);
            blocks["s_embedder"]     = std::make_shared<PatchTokenEmbedder>(config.in_channels * config.patch_size * config.patch_size, config.hidden_size, false, true);
            blocks["t_embedder"]     = std::make_shared<PixelDiTTimestepEmbedder>(config.hidden_size);
            blocks["y_embedder"]     = std::make_shared<PatchTokenEmbedder>(config.txt_embed_dim, config.hidden_size, true, true);
            for (int i = 0; i < config.patch_depth; ++i) {
                blocks["patch_blocks." + std::to_string(i)] = std::make_shared<MMDiTBlockT2I>(config.hidden_size, config.num_groups, config.patch_mlp_hidden_dim);
            }
            for (int i = 0; i < config.pixel_depth; ++i) {
                blocks["pixel_blocks." + std::to_string(i)] = std::make_shared<PiTBlock>(config.pixel_hidden_size,
                                                                                         config.hidden_size,
                                                                                         config.patch_size,
                                                                                         config.pixel_attn_hidden_size,
                                                                                         config.pixel_num_groups);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(config.pixel_hidden_size, config.in_channels);
            blocks["lq_proj"]     = std::make_shared<LQProjection2D>(config);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         std::string prefix                             = "") override {
            params["y_pos_embedding"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.hidden_size, config.txt_max_length, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timesteps,
                             ggml_tensor* context,
                             ggml_tensor* lq_latent,
                             ggml_tensor* degrade_sigma,
                             ggml_tensor* pos_img,
                             ggml_tensor* pos_txt,
                             ggml_tensor* pixel_pos_full,
                             ggml_tensor* pixel_pos_comp) {
            auto pixel_embedder = std::dynamic_pointer_cast<PixelTokenEmbedder>(blocks["pixel_embedder"]);
            auto s_embedder     = std::dynamic_pointer_cast<PatchTokenEmbedder>(blocks["s_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<PixelDiTTimestepEmbedder>(blocks["t_embedder"]);
            auto y_embedder     = std::dynamic_pointer_cast<PatchTokenEmbedder>(blocks["y_embedder"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            auto lq_proj        = std::dynamic_pointer_cast<LQProjection2D>(blocks["lq_proj"]);

            int64_t W_orig = x->ne[0];
            int64_t H_orig = x->ne[1];
            x              = DiT::pad_to_patch_size(ctx, x, static_cast<int>(config.patch_size), static_cast<int>(config.patch_size));
            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t B      = x->ne[3];
            int64_t Hs     = H / config.patch_size;
            int64_t Ws     = W / config.patch_size;
            int64_t L      = Hs * Ws;
            int64_t P2     = config.patch_size * config.patch_size;

            auto x_patches = DiT::patchify(ctx->ggml_ctx, x, static_cast<int>(config.patch_size), static_cast<int>(config.patch_size), true);
            auto t_emb     = t_embedder->forward(ctx, timesteps);
            auto condition = ggml_silu(ctx->ggml_ctx, t_emb);

            GGML_ASSERT(context != nullptr);
            int64_t Ltxt = std::min<int64_t>(context->ne[1], config.txt_max_length);
            auto y       = ggml_ext_slice(ctx->ggml_ctx, context, 1, 0, Ltxt);
            auto y_emb   = y_embedder->forward(ctx, y);
            auto y_pos   = ggml_ext_slice(ctx->ggml_ctx, params["y_pos_embedding"], 1, 0, Ltxt);
            y_emb        = ggml_add(ctx->ggml_ctx, y_emb, y_pos);

            std::vector<ggml_tensor*> lq_features = lq_proj->forward(ctx, lq_latent, Hs, Ws);

            auto s = s_embedder->forward(ctx, x_patches);

            for (int i = 0; i < config.patch_depth; ++i) {
                if (lq_proj->is_gate_active(i)) {
                    int out_idx = lq_proj->get_output_index(i);
                    if (out_idx < static_cast<int>(lq_features.size())) {
                        s = lq_proj->gate(ctx, s, lq_features[out_idx], degrade_sigma, out_idx);
                    }
                }
                auto block = std::dynamic_pointer_cast<MMDiTBlockT2I>(blocks["patch_blocks." + std::to_string(i)]);
                auto out   = block->forward(ctx,
                                            s,
                                            y_emb,
                                            condition,
                                            pos_img,
                                            pos_txt);
                s          = out.first;
                y_emb      = out.second;
                sd::ggml_graph_cut::mark_graph_cut(s, "pid.patch_blocks." + std::to_string(i), "s");
                sd::ggml_graph_cut::mark_graph_cut(y_emb, "pid.patch_blocks." + std::to_string(i), "y");
            }
            s = ggml_silu(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, s, t_emb));

            auto s_cond = ggml_reshape_2d(ctx->ggml_ctx, s, config.hidden_size, L * B);
            auto pixels = pixel_embedder->forward(ctx, x, config.patch_size, pixel_pos_full);
            for (int i = 0; i < config.pixel_depth; ++i) {
                auto block = std::dynamic_pointer_cast<PiTBlock>(blocks["pixel_blocks." + std::to_string(i)]);
                pixels     = block->forward(ctx, pixels, s_cond, H, W, pixel_pos_comp);
                sd::ggml_graph_cut::mark_graph_cut(pixels, "pid.pixel_blocks." + std::to_string(i), "pixels");
            }

            pixels   = final_layer->forward(ctx, pixels);
            pixels   = ggml_reshape_3d(ctx->ggml_ctx, pixels, config.in_channels * P2, L, B);
            auto out = DiT::unpatchify(ctx->ggml_ctx,
                                       pixels,
                                       Hs,
                                       Ws,
                                       static_cast<int>(config.patch_size),
                                       static_cast<int>(config.patch_size),
                                       false);
            out      = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H_orig);
            out      = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W_orig);
            return out;
        }
    };

    struct PiDRunner : public DiffusionModelRunner {
        PixelDiTConfig config;
        PixelDiT model;
        std::vector<float> pos_img_vec;
        std::vector<float> pos_txt_vec;
        std::vector<float> pixel_pos_vec;
        std::vector<float> pixel_pos_comp_vec;

        PiDRunner(ggml_backend_t backend,
                  ggml_backend_t params_backend,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix = "model.diffusion_model")
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(PixelDiTConfig::detect_from_weights(tensor_storage_map, prefix)) {
            model = PixelDiT(config);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "PiD";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& lq_latent_tensor,
                                 const sd::Tensor<float>& degrade_sigma_tensor) {
            ggml_cgraph* gf            = new_graph_custom(PID_GRAPH_SIZE);
            ggml_tensor* x             = make_input(x_tensor);
            ggml_tensor* timesteps     = make_input(timesteps_tensor);
            ggml_tensor* context       = make_input(context_tensor);
            ggml_tensor* lq_latent     = make_input(lq_latent_tensor);
            ggml_tensor* degrade_sigma = make_input(degrade_sigma_tensor);

            int64_t W  = x->ne[0];
            int64_t H  = x->ne[1];
            int64_t B  = x->ne[3];
            int64_t Wp = align_up(static_cast<int>(W), static_cast<int>(config.patch_size));
            int64_t Hp = align_up(static_cast<int>(H), static_cast<int>(config.patch_size));
            int64_t Hs = Hp / config.patch_size;
            int64_t Ws = Wp / config.patch_size;

            pos_img_vec  = make_rope_2d(static_cast<int>(Hs),
                                        static_cast<int>(Ws),
                                        static_cast<int>(config.hidden_size / config.num_groups),
                                        10000.f,
                                        16.f,
                                        static_cast<int>(config.rope_ref_grid_h),
                                        static_cast<int>(config.rope_ref_grid_w));
            auto pos_img = ggml_new_tensor_4d(compute_ctx,
                                              GGML_TYPE_F32,
                                              2,
                                              2,
                                              config.hidden_size / config.num_groups / 2,
                                              Hs * Ws);
            set_backend_tensor_data(pos_img, pos_img_vec.data());

            int64_t Ltxt = std::min<int64_t>(context->ne[1], config.txt_max_length);
            pos_txt_vec  = make_rope_1d(static_cast<int>(Ltxt),
                                        static_cast<int>(config.hidden_size / config.num_groups),
                                        config.text_rope_theta);
            auto pos_txt = ggml_new_tensor_4d(compute_ctx,
                                              GGML_TYPE_F32,
                                              2,
                                              2,
                                              config.hidden_size / config.num_groups / 2,
                                              Ltxt);
            set_backend_tensor_data(pos_txt, pos_txt_vec.data());

            pixel_pos_vec  = make_pixel_abs_pos(static_cast<int>(Hp),
                                                static_cast<int>(Wp),
                                                static_cast<int>(config.pixel_hidden_size));
            auto pixel_pos = ggml_new_tensor_3d(compute_ctx,
                                                GGML_TYPE_F32,
                                                config.pixel_hidden_size,
                                                Wp * Hp,
                                                1);
            set_backend_tensor_data(pixel_pos, pixel_pos_vec.data());

            pixel_pos_comp_vec  = make_rope_2d(static_cast<int>(Hs),
                                               static_cast<int>(Ws),
                                               static_cast<int>(config.pixel_attn_hidden_size / config.pixel_num_groups),
                                               10000.f,
                                               16.f,
                                               static_cast<int>(config.rope_ref_grid_h),
                                               static_cast<int>(config.rope_ref_grid_w));
            auto pixel_pos_comp = ggml_new_tensor_4d(compute_ctx,
                                                     GGML_TYPE_F32,
                                                     2,
                                                     2,
                                                     config.pixel_attn_hidden_size / config.pixel_num_groups / 2,
                                                     Hs * Ws);
            set_backend_tensor_data(pixel_pos_comp, pixel_pos_comp_vec.data());

            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx,
                                            x,
                                            timesteps,
                                            context,
                                            lq_latent,
                                            degrade_sigma,
                                            pos_img,
                                            pos_txt,
                                            pixel_pos,
                                            pixel_pos_comp);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& lq_latent,
                                  const sd::Tensor<float>& degrade_sigma) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, lq_latent, degrade_sigma);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);
            GGML_ASSERT(diffusion_params.ref_latents != nullptr);
            GGML_ASSERT(!diffusion_params.ref_latents->empty());
            auto degrade_sigma = sd::Tensor<float>::from_vector({0.0f});
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           *diffusion_params.context,
                           diffusion_params.ref_latents->front(),
                           degrade_sigma);
        }
    };
}  // namespace Pid

#endif  // __SD_MODEL_DIFFUSION_PID_HPP__

#ifndef __SD_MODEL_DIFFUSION_MINIT2I_HPP__
#define __SD_MODEL_DIFFUSION_MINIT2I_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

namespace MiniT2I {
    constexpr int MINIT2I_GRAPH_SIZE = 196608;

    struct MiniT2IConfig {
        int64_t image_size         = 512;
        int64_t patch_size         = 16;
        int64_t in_channels        = 3;
        int64_t txt_input_size     = 1024;
        int64_t hidden_size        = 768;
        int64_t txt_hidden_size    = 768;
        int64_t cond_vec_size      = 768;
        int64_t depth_double       = 17;
        int64_t txt_preamble_depth = 2;
        int64_t num_heads          = 12;
        int64_t head_dim           = 64;
        float mlp_ratio            = 2.6667f;
        int64_t pca_channels       = 128;
        int64_t prompt_length      = 256;
        int64_t n_T                = 100;
        float cfg_interval_start   = 0.0f;
        float cfg_interval_end     = 1.0f;

        static MiniT2IConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            MiniT2IConfig config;
            config.depth_double       = 0;
            config.txt_preamble_depth = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "img_embedder.proj1.weight") && tensor_storage.n_dims == 4) {
                    config.patch_size   = tensor_storage.ne[0];
                    config.in_channels  = tensor_storage.ne[2];
                    config.pca_channels = tensor_storage.ne[3];
                } else if (ends_with(name, "img_embedder.proj2.weight") && tensor_storage.n_dims == 4) {
                    config.pca_channels = tensor_storage.ne[2];
                    config.hidden_size  = tensor_storage.ne[3];
                } else if (ends_with(name, "txt_embedder.weight") && tensor_storage.n_dims == 2) {
                    config.txt_input_size  = tensor_storage.ne[0];
                    config.txt_hidden_size = tensor_storage.ne[1];
                } else if (ends_with(name, "pooled_embedder.weight") && tensor_storage.n_dims == 2) {
                    config.cond_vec_size = tensor_storage.ne[1];
                } else if (ends_with(name, "double_blocks.0.img_qkv.weight") && tensor_storage.n_dims == 2) {
                    int64_t inner3     = tensor_storage.ne[1];
                    int64_t inner      = inner3 / 3;
                    config.hidden_size = tensor_storage.ne[0];
                    if (config.hidden_size == 768) {
                        config.num_heads = 12;
                        config.head_dim  = 64;
                    } else if (config.hidden_size == 1248) {
                        config.num_heads = 24;
                        config.head_dim  = 52;
                    } else if (inner > 0) {
                        config.head_dim  = 64;
                        config.num_heads = std::max<int64_t>(1, inner / config.head_dim);
                    }
                } else if (ends_with(name, "final_layer.linear.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area = config.patch_size * config.patch_size;
                    config.hidden_size = tensor_storage.ne[0];
                    config.in_channels = patch_area > 0 ? tensor_storage.ne[1] / patch_area : config.in_channels;
                } else if (ends_with(name, "mask_token") && tensor_storage.n_dims >= 2) {
                    config.prompt_length = tensor_storage.ne[1];
                }

                size_t pos = name.find("double_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int64_t idx         = atoi(items[1].c_str());
                        config.depth_double = std::max<int64_t>(config.depth_double, idx + 1);
                    }
                }
                pos = name.find("txt_preamble_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int64_t idx               = atoi(items[1].c_str());
                        config.txt_preamble_depth = std::max<int64_t>(config.txt_preamble_depth, idx + 1);
                    }
                }
            }

            if (config.depth_double <= 0) {
                config.depth_double = config.hidden_size == 1248 ? 23 : 17;
            }
            if (config.txt_preamble_depth <= 0) {
                config.txt_preamble_depth = 2;
            }
            if (config.head_dim <= 0 || config.num_heads <= 0) {
                config.head_dim  = config.hidden_size == 1248 ? 52 : 64;
                config.num_heads = config.hidden_size / config.head_dim;
            }
            LOG_DEBUG("minit2i: hidden_size=%" PRId64 ", txt_hidden_size=%" PRId64 ", heads=%" PRId64 ", head_dim=%" PRId64 ", double_blocks=%" PRId64 ", txt_blocks=%" PRId64 ", patch=%" PRId64 ", in_channels=%" PRId64,
                      config.hidden_size,
                      config.txt_hidden_size,
                      config.num_heads,
                      config.head_dim,
                      config.depth_double,
                      config.txt_preamble_depth,
                      config.patch_size,
                      config.in_channels);
            return config;
        }
    };

    inline std::vector<float> make_2d_sincos_pos_embed(int grid_size, int dim) {
        GGML_ASSERT(dim % 4 == 0);
        int half_dim = dim / 2;
        int quarter  = half_dim / 2;
        std::vector<float> out(static_cast<size_t>(grid_size) * grid_size * dim);
        std::vector<float> omega(quarter);
        for (int i = 0; i < quarter; ++i) {
            omega[i] = 1.0f / std::pow(10000.0f, static_cast<float>(i) / static_cast<float>(quarter));
        }
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                size_t base = static_cast<size_t>(y * grid_size + x) * dim;
                for (int i = 0; i < quarter; ++i) {
                    float ay                           = y * omega[i];
                    float ax                           = x * omega[i];
                    out[base + i]                      = std::sin(ax);
                    out[base + quarter + i]            = std::cos(ax);
                    out[base + half_dim + i]           = std::sin(ay);
                    out[base + half_dim + quarter + i] = std::cos(ay);
                }
            }
        }
        return out;
    }

    inline std::vector<float> make_text_rope(int length, int head_dim) {
        return Rope::flatten(Rope::rope(Rope::linspace(0.f, static_cast<float>(length - 1), length), head_dim, 10000.f));
    }

    inline std::vector<float> make_vision_rope(int side, int head_dim) {
        GGML_ASSERT(head_dim % 4 == 0);
        int dim     = head_dim / 2;
        int quarter = dim / 2;
        int length  = side * side;
        std::vector<float> out(static_cast<size_t>(length) * (head_dim / 2) * 4);
        std::vector<float> freqs(quarter);
        for (int i = 0; i < quarter; ++i) {
            freqs[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(dim));
        }
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                int pos     = y * side + x;
                size_t base = static_cast<size_t>(pos) * (head_dim / 2) * 4;
                for (int i = 0; i < quarter; ++i) {
                    float ay        = y * freqs[i];
                    float ax        = x * freqs[i];
                    float angles[2] = {ay, ax};
                    for (int axis = 0; axis < 2; ++axis) {
                        int j                 = axis * quarter + i;
                        out[base + 4 * j]     = std::cos(angles[axis]);
                        out[base + 4 * j + 1] = -std::sin(angles[axis]);
                        out[base + 4 * j + 2] = std::sin(angles[axis]);
                        out[base + 4 * j + 3] = std::cos(angles[axis]);
                    }
                }
            }
        }
        return out;
    }

    struct SwiGLUMlp : public GGMLBlock {
        SwiGLUMlp(int64_t in_features, int64_t hidden_features) {
            int64_t hidden_dim = ((hidden_features + 7) / 8) * 8;
            blocks["w1"]       = std::make_shared<Linear>(in_features, hidden_dim, false);
            blocks["w3"]       = std::make_shared<Linear>(in_features, hidden_dim, false);
            blocks["w2"]       = std::make_shared<Linear>(hidden_dim, in_features, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1   = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w3   = std::dynamic_pointer_cast<Linear>(blocks["w3"]);
            auto w2   = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto gate = ggml_silu(ctx->ggml_ctx, w1->forward(ctx, x));
            auto up   = w3->forward(ctx, x);
            return w2->forward(ctx, ggml_mul(ctx->ggml_ctx, gate, up));
        }
    };

    struct BottleneckPatchEmbed : public GGMLBlock {
        int64_t patch_size;

        BottleneckPatchEmbed(int64_t patch_size, int64_t in_channels, int64_t pca_channels, int64_t hidden_size)
            : patch_size(patch_size) {
            blocks["proj1"] = std::make_shared<Conv2d>(in_channels,
                                                       pca_channels,
                                                       std::pair<int, int>{static_cast<int>(patch_size), static_cast<int>(patch_size)},
                                                       std::pair<int, int>{static_cast<int>(patch_size), static_cast<int>(patch_size)},
                                                       std::pair<int, int>{0, 0},
                                                       std::pair<int, int>{1, 1},
                                                       false);
            blocks["proj2"] = std::make_shared<Conv2d>(pca_channels,
                                                       hidden_size,
                                                       std::pair<int, int>{1, 1},
                                                       std::pair<int, int>{1, 1},
                                                       std::pair<int, int>{0, 0},
                                                       std::pair<int, int>{1, 1},
                                                       true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj1"]);
            auto proj2 = std::dynamic_pointer_cast<Conv2d>(blocks["proj2"]);
            x          = proj1->forward(ctx, x);
            x          = proj2->forward(ctx, x);
            x          = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x          = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            return x;
        }
    };

    struct TimestepEmbedder : public GGMLBlock {
        int frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size, int frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto t_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, t, frequency_embedding_size, 10000, 1.0f);
            t_emb      = mlp_0->forward(ctx, t_emb);
            t_emb      = ggml_silu_inplace(ctx->ggml_ctx, t_emb);
            return mlp_2->forward(ctx, t_emb);
        }
    };

    inline std::vector<ggml_tensor*> split_qkv(ggml_context* ctx, ggml_tensor* qkv, int64_t num_heads, int64_t head_dim) {
        int64_t N = qkv->ne[2];
        int64_t L = qkv->ne[1];
        auto q    = ggml_view_4d(ctx, qkv, head_dim, num_heads, L, N,
                                 qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], 0);
        auto k    = ggml_view_4d(ctx, qkv, head_dim, num_heads, L, N,
                                 qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], qkv->nb[0] * head_dim * num_heads);
        auto v    = ggml_view_4d(ctx, qkv, head_dim, num_heads, L, N,
                                 qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2], qkv->nb[0] * head_dim * num_heads * 2);
        return {q, k, v};
    }

    struct PlainTextTransformerBlock : public GGMLBlock {
        int64_t num_heads;
        int64_t head_dim;

        PlainTextTransformerBlock(int64_t hidden_size, int64_t num_heads, int64_t head_dim, float mlp_ratio)
            : num_heads(num_heads), head_dim(head_dim) {
            int64_t inner_dim   = num_heads * head_dim;
            blocks["norm1"]     = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["norm2"]     = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["qkv"]       = std::make_shared<Linear>(hidden_size, inner_dim * 3, true);
            blocks["attn_proj"] = std::make_shared<Linear>(inner_dim, hidden_size, true);
            blocks["mlp"]       = std::make_shared<SwiGLUMlp>(hidden_size, static_cast<int64_t>(hidden_size * mlp_ratio));
            blocks["q_norm"]    = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm"]    = std::make_shared<RMSNorm>(head_dim, 1e-6f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* txt, ggml_tensor* pe) {
            auto norm1     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto qkv_proj  = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto attn_proj = std::dynamic_pointer_cast<Linear>(blocks["attn_proj"]);
            auto mlp       = std::dynamic_pointer_cast<SwiGLUMlp>(blocks["mlp"]);
            auto q_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

            auto qkv = split_qkv(ctx->ggml_ctx, qkv_proj->forward(ctx, norm1->forward(ctx, txt)), num_heads, head_dim);
            auto q   = q_norm->forward(ctx, qkv[0]);
            auto k   = k_norm->forward(ctx, qkv[1]);
            auto v   = qkv[2];
            auto out = Rope::attention(ctx, q, k, v, pe, nullptr, 1.0f, false);
            txt      = ggml_add(ctx->ggml_ctx, txt, attn_proj->forward(ctx, out));
            txt      = ggml_add(ctx->ggml_ctx, txt, mlp->forward(ctx, norm2->forward(ctx, txt)));
            return txt;
        }
    };

    struct DoubleStreamDiTBlock : public GGMLBlock {
        int64_t num_heads;
        int64_t head_dim;

        DoubleStreamDiTBlock(int64_t hidden_size, int64_t txt_hidden_size, int64_t num_heads, int64_t head_dim, float mlp_ratio)
            : num_heads(num_heads), head_dim(head_dim) {
            int64_t inner_dim       = num_heads * head_dim;
            blocks["img_norm1"]     = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["img_norm2"]     = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["txt_norm1"]     = std::make_shared<RMSNorm>(txt_hidden_size, 1e-6f);
            blocks["txt_norm2"]     = std::make_shared<RMSNorm>(txt_hidden_size, 1e-6f);
            blocks["img_qkv"]       = std::make_shared<Linear>(hidden_size, inner_dim * 3, true);
            blocks["txt_qkv"]       = std::make_shared<Linear>(txt_hidden_size, inner_dim * 3, true);
            blocks["q_norm"]        = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm"]        = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["img_attn_proj"] = std::make_shared<Linear>(inner_dim, hidden_size, true);
            blocks["txt_attn_proj"] = std::make_shared<Linear>(inner_dim, txt_hidden_size, true);
            blocks["img_mlp"]       = std::make_shared<SwiGLUMlp>(hidden_size, static_cast<int64_t>(hidden_size * mlp_ratio));
            blocks["txt_mlp"]       = std::make_shared<SwiGLUMlp>(txt_hidden_size, static_cast<int64_t>(txt_hidden_size * mlp_ratio));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* pe) {
            auto img_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["img_norm2"]);
            auto txt_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm2"]);
            auto img_qkv_p = std::dynamic_pointer_cast<Linear>(blocks["img_qkv"]);
            auto txt_qkv_p = std::dynamic_pointer_cast<Linear>(blocks["txt_qkv"]);
            auto q_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto img_proj  = std::dynamic_pointer_cast<Linear>(blocks["img_attn_proj"]);
            auto txt_proj  = std::dynamic_pointer_cast<Linear>(blocks["txt_attn_proj"]);
            auto img_mlp   = std::dynamic_pointer_cast<SwiGLUMlp>(blocks["img_mlp"]);
            auto txt_mlp   = std::dynamic_pointer_cast<SwiGLUMlp>(blocks["txt_mlp"]);

            int64_t li = img->ne[1];
            int64_t lt = txt->ne[1];

            auto img_qkv = split_qkv(ctx->ggml_ctx, img_qkv_p->forward(ctx, img_norm1->forward(ctx, img)), num_heads, head_dim);
            auto txt_qkv = split_qkv(ctx->ggml_ctx, txt_qkv_p->forward(ctx, txt_norm1->forward(ctx, txt)), num_heads, head_dim);

            auto q = ggml_concat(ctx->ggml_ctx, q_norm->forward(ctx, txt_qkv[0]), q_norm->forward(ctx, img_qkv[0]), 2);
            auto k = ggml_concat(ctx->ggml_ctx, k_norm->forward(ctx, txt_qkv[1]), k_norm->forward(ctx, img_qkv[1]), 2);
            auto v = ggml_concat(ctx->ggml_ctx, txt_qkv[2], img_qkv[2], 2);

            auto out     = Rope::attention(ctx, q, k, v, pe, nullptr, 1.0f, false);
            auto out_txt = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, lt);
            auto out_img = ggml_ext_slice(ctx->ggml_ctx, out, 1, lt, lt + li);

            img = ggml_add(ctx->ggml_ctx, img, img_proj->forward(ctx, out_img));
            txt = ggml_add(ctx->ggml_ctx, txt, txt_proj->forward(ctx, out_txt));
            img = ggml_add(ctx->ggml_ctx, img, img_mlp->forward(ctx, img_norm2->forward(ctx, img)));
            txt = ggml_add(ctx->ggml_ctx, txt, txt_mlp->forward(ctx, txt_norm2->forward(ctx, txt)));
            return {img, txt};
        }
    };

    struct FinalLayer : public GGMLBlock {
        FinalLayer(int64_t hidden_size, int64_t patch_size, int64_t out_channels) {
            blocks["norm_final"] = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            blocks["linear"]     = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm_final = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_final"]);
            auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            return linear->forward(ctx, norm_final->forward(ctx, x));
        }
    };

    struct MMJiT : public GGMLBlock {
        MiniT2IConfig config;

        MMJiT(const MiniT2IConfig& config)
            : config(config) {
            blocks["img_embedder"]    = std::make_shared<BottleneckPatchEmbed>(config.patch_size, config.in_channels, config.pca_channels, config.hidden_size);
            blocks["txt_embedder"]    = std::make_shared<Linear>(config.txt_input_size, config.txt_hidden_size, false);
            blocks["t_embedder"]      = std::make_shared<TimestepEmbedder>(config.cond_vec_size);
            blocks["pooled_embedder"] = std::make_shared<Linear>(config.txt_input_size, config.cond_vec_size, false);
            for (int64_t i = 0; i < config.txt_preamble_depth; ++i) {
                blocks["txt_preamble_blocks." + std::to_string(i)] = std::make_shared<PlainTextTransformerBlock>(config.txt_hidden_size, config.num_heads, config.head_dim, config.mlp_ratio);
            }
            for (int64_t i = 0; i < config.depth_double; ++i) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamDiTBlock>(config.hidden_size, config.txt_hidden_size, config.num_heads, config.head_dim, config.mlp_ratio);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(config.hidden_size, config.patch_size, config.in_channels);
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            enum ggml_type wtype = get_type(prefix + "mask_token", tensor_storage_map, GGML_TYPE_F32);
            params["mask_token"] = ggml_new_tensor_3d(ctx, wtype, config.txt_input_size, 1, 1);
        }

        ggml_tensor* apply_text_mask(GGMLRunnerContext* ctx, ggml_tensor* context, ggml_tensor* mask) {
            if (mask == nullptr) {
                return context;
            }
            mask            = ggml_reshape_3d(ctx->ggml_ctx, mask, 1, mask->ne[0], mask->ne[1]);
            mask            = ggml_repeat(ctx->ggml_ctx, mask, context);
            auto keep       = ggml_mul(ctx->ggml_ctx, context, mask);
            auto inv        = ggml_sub(ctx->ggml_ctx, ggml_ext_ones_like(ctx->ggml_ctx, mask), mask);
            auto mask_token = ggml_repeat(ctx->ggml_ctx, params["mask_token"], context);
            return ggml_add(ctx->ggml_ctx, keep, ggml_mul(ctx->ggml_ctx, mask_token, inv));
        }

        ggml_tensor* pool_context(GGMLRunnerContext* ctx, ggml_tensor* context) {
            int64_t dim = context->ne[0];
            int64_t len = context->ne[1];
            int64_t N   = context->ne[2];
            auto x      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, context, 1, 0, 2, 3));
            x           = ggml_reshape_3d(ctx->ggml_ctx, x, len, dim, N);
            x           = ggml_mean(ctx->ggml_ctx, x);
            x           = ggml_reshape_2d(ctx->ggml_ctx, x, dim, N);
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* img,
                             ggml_tensor* context,
                             ggml_tensor* mask,
                             ggml_tensor* pos_embed,
                             ggml_tensor* txt_pe,
                             ggml_tensor* joint_pe) {
            auto img_embedder = std::dynamic_pointer_cast<BottleneckPatchEmbed>(blocks["img_embedder"]);
            auto txt_embedder = std::dynamic_pointer_cast<Linear>(blocks["txt_embedder"]);
            auto final_layer  = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            int64_t W  = img->ne[0];
            int64_t H  = img->ne[1];
            int64_t hp = H / config.patch_size;
            int64_t wp = W / config.patch_size;

            context = apply_text_mask(ctx, context, mask);
            auto x  = img_embedder->forward(ctx, img);
            x       = ggml_add(ctx->ggml_ctx, x, pos_embed);

            auto txt = txt_embedder->forward(ctx, context);
            for (int64_t i = 0; i < config.txt_preamble_depth; ++i) {
                auto block = std::dynamic_pointer_cast<PlainTextTransformerBlock>(blocks["txt_preamble_blocks." + std::to_string(i)]);
                txt        = block->forward(ctx, txt, txt_pe);
                sd::ggml_graph_cut::mark_graph_cut(txt, "minit2i.txt_preamble_blocks." + std::to_string(i), "txt");
            }
            for (int64_t i = 0; i < config.depth_double; ++i) {
                auto block = std::dynamic_pointer_cast<DoubleStreamDiTBlock>(blocks["double_blocks." + std::to_string(i)]);
                auto out   = block->forward(ctx, x, txt, joint_pe);
                x          = out.first;
                txt        = out.second;
                sd::ggml_graph_cut::mark_graph_cut(x, "minit2i.double_blocks." + std::to_string(i), "x");
                sd::ggml_graph_cut::mark_graph_cut(txt, "minit2i.double_blocks." + std::to_string(i), "txt");
            }
            auto combined = ggml_concat(ctx->ggml_ctx, txt, x, 1);
            auto out      = final_layer->forward(ctx, combined);
            auto img_out  = ggml_ext_slice(ctx->ggml_ctx, out, 1, txt->ne[1], txt->ne[1] + x->ne[1]);
            return DiT::unpatchify(ctx->ggml_ctx, img_out, hp, wp, static_cast<int>(config.patch_size), static_cast<int>(config.patch_size), false);
        }
    };

    struct MiniT2IRunner : public DiffusionModelRunner {
        MiniT2IConfig config;
        MMJiT model;
        ggml_context* position_cache_ctx            = nullptr;
        ggml_backend_buffer_t position_cache_buffer = nullptr;
        ggml_tensor* cached_pos_embed               = nullptr;
        ggml_tensor* cached_txt_pe                  = nullptr;
        ggml_tensor* cached_joint_pe                = nullptr;
        int64_t cached_img_side                     = -1;
        int64_t cached_txt_len                      = -1;
        int64_t cached_hidden_size                  = -1;
        int64_t cached_head_dim                     = -1;

        MiniT2IRunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map      = {},
                      const std::string prefix                            = "",
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(MiniT2IConfig::detect_from_weights(tensor_storage_map, this->prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, this->prefix);
        }

        ~MiniT2IRunner() override {
            free_position_cache();
        }

        std::string get_desc() override {
            return "MiniT2I";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        void free_position_cache() {
            if (position_cache_buffer != nullptr) {
                ggml_backend_buffer_free(position_cache_buffer);
                position_cache_buffer = nullptr;
            }
            if (position_cache_ctx != nullptr) {
                ggml_free(position_cache_ctx);
                position_cache_ctx = nullptr;
            }
            cached_pos_embed   = nullptr;
            cached_txt_pe      = nullptr;
            cached_joint_pe    = nullptr;
            cached_img_side    = -1;
            cached_txt_len     = -1;
            cached_hidden_size = -1;
            cached_head_dim    = -1;
        }

        void ensure_position_cache(int64_t img_side, int64_t txt_len) {
            if (cached_img_side == img_side &&
                cached_txt_len == txt_len &&
                cached_hidden_size == config.hidden_size &&
                cached_head_dim == config.head_dim &&
                cached_pos_embed != nullptr &&
                cached_txt_pe != nullptr &&
                cached_joint_pe != nullptr) {
                return;
            }

            free_position_cache();

            auto pos_embed_vec = make_2d_sincos_pos_embed(static_cast<int>(img_side), static_cast<int>(config.hidden_size));
            auto txt_pe_vec    = make_text_rope(static_cast<int>(txt_len), static_cast<int>(config.head_dim));
            auto img_pe_vec    = make_vision_rope(static_cast<int>(img_side), static_cast<int>(config.head_dim));
            auto joint_pe_vec  = txt_pe_vec;
            joint_pe_vec.insert(joint_pe_vec.end(), img_pe_vec.begin(), img_pe_vec.end());

            ggml_init_params params;
            params.mem_size    = static_cast<size_t>(3 * ggml_tensor_overhead());
            params.mem_buffer  = nullptr;
            params.no_alloc    = true;
            position_cache_ctx = ggml_init(params);
            GGML_ASSERT(position_cache_ctx != nullptr);

            cached_pos_embed = ggml_new_tensor_3d(position_cache_ctx, GGML_TYPE_F32, config.hidden_size, img_side * img_side, 1);
            ggml_set_name(cached_pos_embed, "minit2i.pos_embed");
            cached_txt_pe = ggml_new_tensor_4d(position_cache_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, txt_len);
            ggml_set_name(cached_txt_pe, "minit2i.txt_pe");
            cached_joint_pe = ggml_new_tensor_4d(position_cache_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, txt_len + img_side * img_side);
            ggml_set_name(cached_joint_pe, "minit2i.joint_pe");

            position_cache_buffer = ggml_backend_alloc_ctx_tensors(position_cache_ctx, runtime_backend);
            GGML_ASSERT(position_cache_buffer != nullptr);
            ggml_backend_buffer_set_usage(position_cache_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(cached_pos_embed, pos_embed_vec.data(), 0, ggml_nbytes(cached_pos_embed));
            ggml_backend_tensor_set(cached_txt_pe, txt_pe_vec.data(), 0, ggml_nbytes(cached_txt_pe));
            ggml_backend_tensor_set(cached_joint_pe, joint_pe_vec.data(), 0, ggml_nbytes(cached_joint_pe));
            ggml_backend_synchronize(runtime_backend);

            cached_img_side    = img_side;
            cached_txt_len     = txt_len;
            cached_hidden_size = config.hidden_size;
            cached_head_dim    = config.head_dim;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& mask_tensor) {
            ggml_cgraph* gf      = new_graph_custom(MINIT2I_GRAPH_SIZE);
            ggml_tensor* x       = make_input(x_tensor);
            ggml_tensor* context = make_input(context_tensor);
            ggml_tensor* mask    = make_input(mask_tensor);
            SD_UNUSED(timesteps_tensor);

            int64_t W        = x->ne[0];
            int64_t H        = x->ne[1];
            int64_t img_side = H / config.patch_size;
            int64_t txt_len  = context->ne[1];
            ensure_position_cache(img_side, txt_len);

            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx, x, context, mask, cached_pos_embed, cached_txt_pe, cached_joint_pe);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& mask) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, mask);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);
            const auto* extra = diffusion_extra_as<MiniT2IDiffusionExtra>(diffusion_params);
            GGML_ASSERT(extra->mask != nullptr);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           *diffusion_params.context,
                           *extra->mask);
        }
    };
}  // namespace MiniT2I

#endif  // __SD_MODEL_DIFFUSION_MINIT2I_HPP__

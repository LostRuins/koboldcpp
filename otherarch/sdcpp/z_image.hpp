#ifndef __Z_IMAGE_HPP__
#define __Z_IMAGE_HPP__

#include <algorithm>

#include "flux.hpp"
#include "ggml_extend.hpp"
#include "mmdit.hpp"

// Ref: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
// Ref: https://github.com/huggingface/diffusers/pull/12703

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace ZImage {
    constexpr int Z_IMAGE_GRAPH_SIZE = 20480;
    constexpr int ADALN_EMBED_DIM    = 256;
    constexpr int SEQ_MULTI_OF       = 32;

    struct JointAttention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        JointAttention(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
            : head_dim(head_dim), num_heads(num_heads), num_kv_heads(num_kv_heads), qk_norm(qk_norm) {
            blocks["qkv"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads * 2) * head_dim, false);
            float scale   = 1.f;
#if GGML_USE_HIP
            // Prevent NaN issues with certain ROCm setups
            scale = 1.f / 16.f;
#endif
            blocks["out"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false, false, false, scale);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim);
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto qkv_proj   = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            auto qkv = qkv_proj->forward(ctx, x);                                                                            // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
            qkv      = ggml_reshape_4d(ctx->ggml_ctx, qkv, head_dim, num_heads + num_kv_heads * 2, qkv->ne[1], qkv->ne[2]);  // [N, n_token, num_heads + num_kv_heads*2, head_dim]
            qkv      = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, qkv, 0, 2, 3, 1));                     // [num_heads + num_kv_heads*2, N, n_token, head_dim]

            auto q = ggml_view_4d(ctx->ggml_ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], num_heads, qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);                                           // [num_heads, N, n_token, head_dim]
            auto k = ggml_view_4d(ctx->ggml_ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], num_kv_heads, qkv->nb[1], qkv->nb[2], qkv->nb[3], qkv->nb[3] * num_heads);                   // [num_kv_heads, N, n_token, head_dim]
            auto v = ggml_view_4d(ctx->ggml_ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], num_kv_heads, qkv->nb[1], qkv->nb[2], qkv->nb[3], qkv->nb[3] * (num_heads + num_kv_heads));  // [num_kv_heads, N, n_token, head_dim]

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 3, 1, 2));  // [N, n_token, num_heads, head_dim]
            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 3, 1, 2));  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 0, 3, 1, 2));  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f / 128.f);  // [N, n_token, num_heads * head_dim]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t hidden_dim,
                    int64_t multiple_of,
                    float ffn_dim_multiplier = 0.f) {
            if (ffn_dim_multiplier > 0.f) {
                hidden_dim = static_cast<int64_t>(ffn_dim_multiplier * hidden_dim);
            }
            hidden_dim   = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);

            bool force_prec_f32 = false;
            float scale         = 1.f / 128.f;
#ifdef SD_USE_VULKAN
            force_prec_f32 = true;
#endif
            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example, when using CUDA but the weights are k-quants.
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false, false, force_prec_f32, scale);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            auto x1 = w1->forward(ctx, x);
            auto x3 = w3->forward(ctx, x);
            x       = ggml_mul(ctx->ggml_ctx, ggml_silu(ctx->ggml_ctx, x1), x3);
            x       = w2->forward(ctx, x);

            return x;
        }
    };

    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        return x;
    }

    struct JointTransformerBlock : public GGMLBlock {
    protected:
        bool modulation;

    public:
        JointTransformerBlock(int layer_id,
                              int64_t hidden_size,
                              int64_t head_dim,
                              int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t multiple_of,
                              float ffn_dim_multiplier,
                              float norm_eps,
                              bool qk_norm,
                              bool modulation = true)
            : modulation(modulation) {
            blocks["attention"]       = std::make_shared<JointAttention>(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm);
            blocks["feed_forward"]    = std::make_shared<FeedForward>(hidden_size, hidden_size, multiple_of, ffn_dim_multiplier);
            blocks["attention_norm1"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm1"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["attention_norm2"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm2"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            if (modulation) {
                blocks["adaLN_modulation.0"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), 4 * hidden_size);
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask        = nullptr,
                                    struct ggml_tensor* adaln_input = nullptr) {
            auto attention       = std::dynamic_pointer_cast<JointAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                GGML_ASSERT(adaln_input != nullptr);
                auto adaLN_modulation_0 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);

                auto m         = adaLN_modulation_0->forward(ctx, adaln_input);  // [N, 4 * hidden_size]
                auto mods      = ggml_ext_chunk(ctx->ggml_ctx, m, 4, 0);
                auto scale_msa = mods[0];
                auto gate_msa  = mods[1];
                auto scale_mlp = mods[2];
                auto gate_mlp  = mods[3];

                auto residual = x;
                x             = modulate(ctx->ggml_ctx, attention_norm1->forward(ctx, x), scale_msa);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_msa));
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, x), scale_mlp);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_mlp));
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            } else {
                GGML_ASSERT(adaln_input == nullptr);

                auto residual = x;
                x             = attention_norm1->forward(ctx, x);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = ffn_norm1->forward(ctx, x);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            }

            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    public:
        FinalLayer(int64_t hidden_size,
                   int64_t patch_size,
                   int64_t out_channels) {
            blocks["norm_final"]         = std::make_shared<LayerNorm>(hidden_size, 1e-06f, false);
            blocks["linear"]             = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true, true);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), hidden_size);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, hidden_size]
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);

            return x;
        }
    };

    struct ZImageParams {
        int patch_size             = 2;
        int64_t hidden_size        = 3840;
        int64_t in_channels        = 16;
        int64_t out_channels       = 16;
        int64_t num_layers         = 30;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 128;
        int64_t num_heads          = 30;
        int64_t num_kv_heads       = 30;
        int64_t multiple_of        = 256;
        float ffn_dim_multiplier   = 8.0f / 3.0f;
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2560;
        int theta                  = 256;
        std::vector<int> axes_dim  = {32, 48, 48};
        int64_t axes_dim_sum       = 128;
    };

    class ZImageModel : public GGMLBlock {
    protected:
        ZImageParams z_image_params;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
            params["x_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
        }

    public:
        ZImageModel() = default;
        ZImageModel(ZImageParams z_image_params)
            : z_image_params(z_image_params) {
            blocks["x_embedder"]     = std::make_shared<Linear>(z_image_params.patch_size * z_image_params.patch_size * z_image_params.in_channels, z_image_params.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(MIN(z_image_params.hidden_size, 1024), 256, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(z_image_params.cap_feat_dim, z_image_params.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(z_image_params.cap_feat_dim, z_image_params.hidden_size);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["noise_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     false);

                blocks["context_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["layers." + std::to_string(i)] = block;
            }

            blocks["final_layer"] = std::make_shared<FinalLayer>(z_image_params.hidden_size, z_image_params.patch_size, z_image_params.out_channels);
        }

        struct ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                              struct ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int pad_h = (z_image_params.patch_size - H % z_image_params.patch_size) % z_image_params.patch_size;
            int pad_w = (z_image_params.patch_size - W % z_image_params.patch_size) % z_image_params.patch_size;
            x         = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x) {
            // x: [N, C, H, W]
            // return: [N, h*w, patch_size*patch_size*C]
            int64_t N = x->ne[3];
            int64_t C = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];
            int64_t p = z_image_params.patch_size;
            int64_t h = H / z_image_params.patch_size;
            int64_t w = W / z_image_params.patch_size;

            GGML_ASSERT(h * p == H && w * p == W);

            x = ggml_reshape_4d(ctx, x, p, w, p, h * C * N);                 // [N*C*h, p, w, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));            // [N*C*h, w, p, p]
            x = ggml_reshape_4d(ctx, x, p * p, w * h, C, N);                 // [N, C, h*w, p*p]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, h*w, C, p*p]
            x = ggml_reshape_3d(ctx, x, C * p * p, w * h, N);                // [N, h*w, p*p*C]
            return x;
        }

        struct ggml_tensor* process_img(GGMLRunnerContext* ctx,
                                        struct ggml_tensor* x) {
            x = pad_to_patch_size(ctx, x);
            x = patchify(ctx->ggml_ctx, x);
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t h,
                                       int64_t w) {
            // x: [N, h*w, patch_size*patch_size*C]
            // return: [N, C, H, W]
            int64_t N = x->ne[2];
            int64_t C = x->ne[0] / z_image_params.patch_size / z_image_params.patch_size;
            int64_t H = h * z_image_params.patch_size;
            int64_t W = w * z_image_params.patch_size;
            int64_t p = z_image_params.patch_size;

            GGML_ASSERT(C * p * p == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, p * p, w * h, N);                 // [N, h*w, p*p, C]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [N, C, h*w, p*p]
            x = ggml_reshape_4d(ctx, x, p, p, w, h * C * N);                 // [N*C*h, w, p, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));            // [N*C*h, p, w, p]
            x = ggml_reshape_4d(ctx, x, W, H, C, N);                         // [N, C, h*p, w*p]

            return x;
        }

        struct ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* timestep,
                                         struct ggml_tensor* context,
                                         struct ggml_tensor* pe) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto norm_final     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_final"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);  // [N, n_txt_token + n_txt_pad_token, hidden_size]
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);  // [N, n_img_token + n_img_pad_token, hidden_size]
            }

            GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]);

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt->ne[1], pe->ne[3]);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);

                txt = block->forward(ctx, txt, txt_pe, nullptr, nullptr);
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);

                img = block->forward(ctx, img, img_pe, nullptr, t_emb);
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, hidden_size]

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, pe, nullptr, t_emb);
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, ph*pw*C]

            img = ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, n_txt_token + n_txt_pad_token, n_txt_token + n_txt_pad_token + n_img_token);  // [N, n_img_token, ph*pw*C]

            return img;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe,
                                    std::vector<ggml_tensor*> ref_latents = {}) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            auto img             = process_img(ctx, x);
            uint64_t n_img_token = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = process_img(ctx, ref);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            int64_t h_len = ((H + (z_image_params.patch_size / 2)) / z_image_params.patch_size);
            int64_t w_len = ((W + (z_image_params.patch_size / 2)) / z_image_params.patch_size);

            auto out = forward_core(ctx, img, timestep, context, pe);

            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, n_img_token);  // [N, n_img_token, ph*pw*C]
            out = unpatchify(ctx->ggml_ctx, out, h_len, w_len);           // [N, C, H + pad_h, W + pad_w]

            // slice
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N, C, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N, C, H, W]

            out = ggml_scale(ctx->ggml_ctx, out, -1.f);

            return out;
        }
    };

    struct ZImageRunner : public GGMLRunner {
    public:
        ZImageParams z_image_params;
        ZImageModel z_image;
        std::vector<float> pe_vec;
        std::vector<float> timestep_vec;
        SDVersion version;

        ZImageRunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     SDVersion version                              = VERSION_Z_IMAGE)
            : GGMLRunner(backend, offload_params_to_cpu) {
            z_image = ZImageModel(z_image_params);
            z_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "z_image";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            z_image.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE);

            x         = to_backend(x);
            context   = to_backend(context);
            timesteps = to_backend(timesteps);

            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            pe_vec      = Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                               static_cast<int>(x->ne[0]),
                                               z_image_params.patch_size,
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context->ne[1]),
                                               SEQ_MULTI_OF,
                                               ref_latents,
                                               increase_ref_index,
                                               z_image_params.theta,
                                               circular_y_enabled,
                                               circular_x_enabled,
                                               z_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            struct ggml_tensor* out = z_image.forward(&runner_ctx,
                                                      x,
                                                      timesteps,
                                                      context,
                                                      pe,
                                                      ref_latents);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "./z_image_x.bin");
                print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 0.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 2560, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "./z_image_context.bin");
                print_ggml_tensor(context);

                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, x, timesteps, context, {}, false, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("z_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            std::shared_ptr<ZImageRunner> z_image = std::make_shared<ZImageRunner>(backend,
                                                                                   false,
                                                                                   tensor_storage_map,
                                                                                   "model.diffusion_model",
                                                                                   VERSION_QWEN_IMAGE);

            z_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            z_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("z_image model loaded");
            z_image->test();
        }
    };

}  // namespace ZImage

#endif  // __Z_IMAGE_HPP__

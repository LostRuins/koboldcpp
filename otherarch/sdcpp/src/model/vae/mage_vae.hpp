#ifndef __SD_MODEL_VAE_MAGE_VAE_HPP__
#define __SD_MODEL_VAE_MAGE_VAE_HPP__

#include "model/diffusion/dit.hpp"
#include "model/vae/vae.hpp"

namespace MageVAE {
    constexpr int MAGE_VAE_GRAPH_SIZE = 327680;
    constexpr int HIDDEN_SIZE         = 384;
    constexpr int LATENT_CHANNELS     = 128;
    constexpr int PATCH_SIZE          = 16;

    struct LayerNorm2d : public UnaryBlock {
        int64_t channels;
        bool affine;
        std::string prefix;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            this->prefix = prefix;
            if (affine) {
                params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
                params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            }
        }

        LayerNorm2d(int64_t channels, bool affine = true)
            : channels(channels), affine(affine) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            ggml_tensor* weight = affine ? params["weight"] : nullptr;
            ggml_tensor* bias   = affine ? params["bias"] : nullptr;
            if (affine && ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, weight, prefix + "weight");
                bias   = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, bias, prefix + "bias");
            }
            // [N, C, H, W] -> [N, H, W, C] so layer norm reduces over channels.
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
            x = ggml_ext_layer_norm(ctx->ggml_ctx, x, weight, bias, 1e-6f);
            return ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
        }
    };

    inline ggml_tensor* modulate_2d(ggml_context* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* shift,
                                    ggml_tensor* scale) {
        shift = ggml_reshape_4d(ctx, shift, 1, 1, shift->ne[0], shift->ne[1]);
        scale = ggml_reshape_4d(ctx, scale, 1, 1, scale->ne[0], scale->ne[1]);
        return ggml_add(ctx, ggml_mul(ctx, x, ggml_add(ctx, scale, ggml_ext_ones(ctx, 1, 1, 1, 1))), shift);
    }

    inline ggml_tensor* channel_attention(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          Conv2d* projection) {
        auto pooled = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
        pooled      = ggml_mean(ctx->ggml_ctx, pooled);
        pooled      = ggml_reshape_4d(ctx->ggml_ctx, pooled, 1, 1, x->ne[2], x->ne[3]);
        pooled      = ggml_sigmoid(ctx->ggml_ctx, projection->forward(ctx, pooled));
        return ggml_mul(ctx->ggml_ctx, x, pooled);
    }

    struct TimestepEmbedder : public GGMLBlock {
        TimestepEmbedder() {
            blocks["mlp.0"] = std::make_shared<Linear>(256, HIDDEN_SIZE);
            blocks["mlp.2"] = std::make_shared<Linear>(HIDDEN_SIZE, HIDDEN_SIZE);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto linear_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto x        = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, 256, 10000, 1.f);
            x             = linear_0->forward(ctx, x);
            x             = ggml_silu_inplace(ctx->ggml_ctx, x);
            return linear_2->forward(ctx, x);
        }
    };

    struct EncoderDiCoBlock : public UnaryBlock {
        explicit EncoderDiCoBlock(int64_t channels) {
            blocks["conv1"] = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["conv2"] = std::make_shared<Conv2d_grouped>(channels, channels, static_cast<int>(channels), std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
            blocks["conv3"] = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["ca.1"]  = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["conv4"] = std::make_shared<Conv2d>(channels, channels * 4, std::pair{1, 1});
            blocks["conv5"] = std::make_shared<Conv2d>(channels * 4, channels, std::pair{1, 1});
            blocks["norm1"] = std::make_shared<LayerNorm2d>(channels);
            blocks["norm2"] = std::make_shared<LayerNorm2d>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input) override {
            auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv2"]);
            auto conv3 = std::dynamic_pointer_cast<Conv2d>(blocks["conv3"]);
            auto ca    = std::dynamic_pointer_cast<Conv2d>(blocks["ca.1"]);
            auto conv4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv4"]);
            auto conv5 = std::dynamic_pointer_cast<Conv2d>(blocks["conv5"]);
            auto norm1 = std::dynamic_pointer_cast<LayerNorm2d>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm2d>(blocks["norm2"]);

            auto x = norm1->forward(ctx, input);
            x      = conv1->forward(ctx, x);
            x      = conv2->forward(ctx, x);
            x      = ggml_gelu(ctx->ggml_ctx, x);
            x      = channel_attention(ctx, x, ca.get());
            x      = conv3->forward(ctx, x);
            x      = ggml_add(ctx->ggml_ctx, input, x);
            auto h = norm2->forward(ctx, x);
            h      = conv4->forward(ctx, h);
            h      = ggml_gelu(ctx->ggml_ctx, h);
            h      = conv5->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct DiCoBlock : public GGMLBlock {
        explicit DiCoBlock(int64_t channels) {
            blocks["conv1"]              = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["conv2"]              = std::make_shared<Conv2d_grouped>(channels, channels, static_cast<int>(channels), std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
            blocks["conv3"]              = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["ca.1"]               = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["conv4"]              = std::make_shared<Conv2d>(channels, channels * 4, std::pair{1, 1});
            blocks["conv5"]              = std::make_shared<Conv2d>(channels * 4, channels, std::pair{1, 1});
            blocks["norm1"]              = std::make_shared<LayerNorm2d>(channels, false);
            blocks["norm2"]              = std::make_shared<LayerNorm2d>(channels, false);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(channels, channels * 6);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input, ggml_tensor* condition) {
            auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv2"]);
            auto conv3 = std::dynamic_pointer_cast<Conv2d>(blocks["conv3"]);
            auto ca    = std::dynamic_pointer_cast<Conv2d>(blocks["ca.1"]);
            auto conv4 = std::dynamic_pointer_cast<Conv2d>(blocks["conv4"]);
            auto conv5 = std::dynamic_pointer_cast<Conv2d>(blocks["conv5"]);
            auto norm1 = std::dynamic_pointer_cast<LayerNorm2d>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm2d>(blocks["norm2"]);
            auto ada   = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto params = ada->forward(ctx, ggml_silu(ctx->ggml_ctx, condition));
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, params, 6, 0);
            auto x      = norm1->forward(ctx, input);
            x           = modulate_2d(ctx->ggml_ctx, x, chunks[0], chunks[1]);
            x           = conv1->forward(ctx, x);
            x           = conv2->forward(ctx, x);
            x           = ggml_gelu(ctx->ggml_ctx, x);
            x           = channel_attention(ctx, x, ca.get());
            x           = conv3->forward(ctx, x);
            auto gate_1 = ggml_reshape_4d(ctx->ggml_ctx, chunks[2], 1, 1, chunks[2]->ne[0], chunks[2]->ne[1]);
            x           = ggml_add(ctx->ggml_ctx, input, ggml_mul(ctx->ggml_ctx, x, gate_1));

            auto h      = norm2->forward(ctx, x);
            h           = modulate_2d(ctx->ggml_ctx, h, chunks[3], chunks[4]);
            h           = conv4->forward(ctx, h);
            h           = ggml_gelu(ctx->ggml_ctx, h);
            h           = conv5->forward(ctx, h);
            auto gate_2 = ggml_reshape_4d(ctx->ggml_ctx, chunks[5], 1, 1, chunks[5]->ne[0], chunks[5]->ne[1]);
            return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, h, gate_2));
        }
    };

    struct MageResnetBlock : public UnaryBlock {
        explicit MageResnetBlock(int64_t channels) {
            blocks["norm1"] = std::make_shared<GroupNorm32>(channels);
            blocks["conv1"] = std::make_shared<Conv2d>(channels, channels, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
            blocks["norm2"] = std::make_shared<GroupNorm32>(channels);
            blocks["conv2"] = std::make_shared<Conv2d>(channels, channels, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input) override {
            auto norm1 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
            auto x     = conv1->forward(ctx, ggml_silu(ctx->ggml_ctx, norm1->forward(ctx, input)));
            x          = conv2->forward(ctx, ggml_silu(ctx->ggml_ctx, norm2->forward(ctx, x)));
            return ggml_add(ctx->ggml_ctx, input, x);
        }
    };

    inline ggml_tensor* replicate_pad_right_bottom(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   int pad_w,
                                                   int pad_h) {
        if (pad_w > 0) {
            auto edge = ggml_ext_slice(ctx, x, 0, x->ne[0] - 1, x->ne[0]);
            edge      = ggml_repeat_4d(ctx, edge, pad_w, x->ne[1], x->ne[2], x->ne[3]);
            x         = ggml_concat(ctx, x, edge, 0);
        }
        if (pad_h > 0) {
            auto edge = ggml_ext_slice(ctx, x, 1, x->ne[1] - 1, x->ne[1]);
            edge      = ggml_repeat_4d(ctx, edge, x->ne[0], pad_h, x->ne[2], x->ne[3]);
            x         = ggml_concat(ctx, x, edge, 1);
        }
        return x;
    }

    struct MageAttnBlock : public UnaryBlock {
        int64_t channels;
        int patch_size;

        MageAttnBlock(int64_t channels, int patch_size = 32)
            : channels(channels), patch_size(patch_size) {
            blocks["norm"]     = std::make_shared<GroupNorm32>(channels);
            blocks["q"]        = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["k"]        = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["v"]        = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
            blocks["proj_out"] = std::make_shared<Conv2d>(channels, channels, std::pair{1, 1});
        }

        ggml_tensor* to_patches(ggml_context* ctx, ggml_tensor* x) {
            x = DiT::patchify(ctx, x, patch_size, patch_size);
            x = ggml_reshape_4d(ctx, x, patch_size * patch_size, channels, x->ne[1], x->ne[2]);
            // [N, np, C, P] -> [N, np, P, C] for attention over P pixels.
            x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            return ggml_reshape_3d(ctx, x, channels, patch_size * patch_size, x->ne[2] * x->ne[3]);
        }

        ggml_tensor* from_patches(ggml_context* ctx,
                                  ggml_tensor* x,
                                  int64_t patch_count,
                                  int64_t batch_size,
                                  int64_t h_patches,
                                  int64_t w_patches) {
            x = ggml_reshape_4d(ctx, x, channels, patch_size * patch_size, patch_count, batch_size);
            // [N, np, P, C] -> [N, np, C, P] before spatial unpatchify.
            x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            x = ggml_reshape_3d(ctx, x, patch_size * patch_size * channels, patch_count, batch_size);
            return DiT::unpatchify(ctx, x, h_patches, w_patches, patch_size, patch_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* input) override {
            auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
            auto q_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["q"]);
            auto k_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["k"]);
            auto v_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["v"]);
            auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);

            int64_t width  = input->ne[0];
            int64_t height = input->ne[1];
            int64_t batch  = input->ne[3];
            int pad_w      = (patch_size - static_cast<int>(width % patch_size)) % patch_size;
            int pad_h      = (patch_size - static_cast<int>(height % patch_size)) % patch_size;
            int64_t wp     = (width + pad_w) / patch_size;
            int64_t hp     = (height + pad_h) / patch_size;
            int64_t np     = wp * hp;

            auto h = norm->forward(ctx, input);
            auto q = replicate_pad_right_bottom(ctx->ggml_ctx, q_proj->forward(ctx, h), pad_w, pad_h);
            auto k = replicate_pad_right_bottom(ctx->ggml_ctx, k_proj->forward(ctx, h), pad_w, pad_h);
            auto v = replicate_pad_right_bottom(ctx->ggml_ctx, v_proj->forward(ctx, h), pad_w, pad_h);
            q      = to_patches(ctx->ggml_ctx, q);
            k      = to_patches(ctx->ggml_ctx, k);
            v      = to_patches(ctx->ggml_ctx, v);
            h      = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);
            h      = from_patches(ctx->ggml_ctx, h, np, batch, hp, wp);
            if (pad_h > 0) {
                h = ggml_ext_slice(ctx->ggml_ctx, h, 1, 0, height);
            }
            if (pad_w > 0) {
                h = ggml_ext_slice(ctx->ggml_ctx, h, 0, 0, width);
            }
            return ggml_add(ctx->ggml_ctx, input, proj_out->forward(ctx, h));
        }
    };

    struct Decoder : public UnaryBlock {
        Decoder() {
            blocks["conv_in"]  = std::make_shared<Conv2d>(LATENT_CHANNELS, HIDDEN_SIZE, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
            blocks["block.0"]  = std::make_shared<MageResnetBlock>(HIDDEN_SIZE);
            blocks["block.1"]  = std::make_shared<MageAttnBlock>(HIDDEN_SIZE);
            blocks["block.2"]  = std::make_shared<MageResnetBlock>(HIDDEN_SIZE);
            blocks["block.3"]  = std::make_shared<MageAttnBlock>(HIDDEN_SIZE);
            blocks["block.4"]  = std::make_shared<MageResnetBlock>(HIDDEN_SIZE);
            blocks["norm_out"] = std::make_shared<GroupNorm32>(HIDDEN_SIZE);
            blocks["conv_out"] = std::make_shared<Conv2d>(HIDDEN_SIZE, HIDDEN_SIZE, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            x = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"])->forward(ctx, x);
            for (int i = 0; i < 5; ++i) {
                x = std::dynamic_pointer_cast<UnaryBlock>(blocks["block." + std::to_string(i)])->forward(ctx, x);
            }
            x = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"])->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            return std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"])->forward(ctx, x);
        }
    };

    struct DConvEncoder : public UnaryBlock {
        DConvEncoder() {
            blocks["patch_cond_embed"] = std::make_shared<Conv2d>(3, 768, std::pair{PATCH_SIZE, PATCH_SIZE}, std::pair{PATCH_SIZE, PATCH_SIZE});
            for (int i = 0; i < 2; ++i) {
                blocks["head_blocks." + std::to_string(i)] = std::make_shared<EncoderDiCoBlock>(768);
            }
            blocks["proj_down"]  = std::make_shared<Conv2d>(768, HIDDEN_SIZE, std::pair{1, 1});
            blocks["z_proj"]     = std::make_shared<Conv2d>(LATENT_CHANNELS, HIDDEN_SIZE, std::pair{1, 1});
            blocks["fuse_proj"]  = std::make_shared<Conv2d>(HIDDEN_SIZE * 2, HIDDEN_SIZE, std::pair{1, 1});
            blocks["t_embedder"] = std::make_shared<TimestepEmbedder>();
            for (int i = 0; i < 21; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<DiCoBlock>(HIDDEN_SIZE);
            }
            blocks["norm_out"] = std::make_shared<LayerNorm2d>(HIDDEN_SIZE);
            blocks["proj_out"] = std::make_shared<Conv2d>(HIDDEN_SIZE, LATENT_CHANNELS * 2, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image) override {
            auto cond = std::dynamic_pointer_cast<Conv2d>(blocks["patch_cond_embed"])->forward(ctx, image);
            for (int i = 0; i < 2; ++i) {
                cond = std::dynamic_pointer_cast<EncoderDiCoBlock>(blocks["head_blocks." + std::to_string(i)])->forward(ctx, cond);
            }
            cond   = std::dynamic_pointer_cast<Conv2d>(blocks["proj_down"])->forward(ctx, cond);
            auto z = ggml_ext_zeros(ctx->ggml_ctx, cond->ne[0], cond->ne[1], LATENT_CHANNELS, cond->ne[3]);
            z      = std::dynamic_pointer_cast<Conv2d>(blocks["z_proj"])->forward(ctx, z);
            z      = ggml_concat(ctx->ggml_ctx, cond, z, 2);
            z      = std::dynamic_pointer_cast<Conv2d>(blocks["fuse_proj"])->forward(ctx, z);
            auto t = ggml_ext_zeros(ctx->ggml_ctx, image->ne[3], 1, 1, 1);
            t      = ggml_reshape_1d(ctx->ggml_ctx, t, image->ne[3]);
            auto c = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"])->forward(ctx, t);
            for (int i = 0; i < 21; ++i) {
                z = std::dynamic_pointer_cast<DiCoBlock>(blocks["blocks." + std::to_string(i)])->forward(ctx, z, c);
            }
            z = std::dynamic_pointer_cast<LayerNorm2d>(blocks["norm_out"])->forward(ctx, z);
            return std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"])->forward(ctx, z);
        }
    };

    struct MLPResBlock : public GGMLBlock {
        MLPResBlock() {
            blocks["in_ln"]              = std::make_shared<LayerNorm>(32, 1e-6f);
            blocks["mlp.0"]              = std::make_shared<Linear>(32, 32);
            blocks["mlp.2"]              = std::make_shared<Linear>(32, 32);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(32, 96);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* condition) {
            auto params = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"])->forward(ctx, ggml_silu(ctx->ggml_ctx, condition));
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, params, 3, 0);
            auto h      = std::dynamic_pointer_cast<LayerNorm>(blocks["in_ln"])->forward(ctx, x);
            h           = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, h, ggml_add(ctx->ggml_ctx, chunks[1], ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1))), chunks[0]);
            h           = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"])->forward(ctx, h);
            h           = ggml_silu(ctx->ggml_ctx, h);
            h           = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, chunks[2], h));
        }
    };

    struct DConvDenoiser : public GGMLBlock {
        DConvDenoiser() {
            blocks["t_embedder"]            = std::make_shared<TimestepEmbedder>();
            blocks["y_embedder_x"]          = std::make_shared<Conv2d>(HIDDEN_SIZE, 32 * PATCH_SIZE * PATCH_SIZE, std::pair{1, 1});
            blocks["x_embedder.embedder.0"] = std::make_shared<Linear>(3 + 32 + 64, 32);
            blocks["s_embedder.proj1"]      = std::make_shared<Conv2d>(3, LATENT_CHANNELS, std::pair{PATCH_SIZE, PATCH_SIZE}, std::pair{PATCH_SIZE, PATCH_SIZE}, std::pair{0, 0}, std::pair{1, 1}, false);
            blocks["s_embedder.proj2"]      = std::make_shared<Conv2d>(LATENT_CHANNELS + HIDDEN_SIZE, HIDDEN_SIZE, std::pair{1, 1});
            for (int i = 0; i < 21; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<DiCoBlock>(HIDDEN_SIZE);
            }
            blocks["dec_net.cond_embed"] = std::make_shared<Linear>(HIDDEN_SIZE, PATCH_SIZE * PATCH_SIZE * 32);
            blocks["dec_net.input_proj"] = std::make_shared<Linear>(32, 32);
            for (int i = 0; i < 3; ++i) {
                blocks["dec_net.res_blocks." + std::to_string(i)] = std::make_shared<MLPResBlock>();
            }
            blocks["final_layer.norm"]   = std::make_shared<RMSNorm>(32);
            blocks["final_layer.linear"] = std::make_shared<Linear>(32, 3);
            blocks["y_embedder.decoder"] = std::make_shared<Decoder>();
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* dct) {
            auto cond      = std::dynamic_pointer_cast<Decoder>(blocks["y_embedder.decoder"])->forward(ctx, latent);
            int64_t w      = cond->ne[0];
            int64_t h      = cond->ne[1];
            int64_t n      = cond->ne[3];
            int64_t length = w * h;

            auto image = ggml_ext_zeros(ctx->ggml_ctx, w * PATCH_SIZE, h * PATCH_SIZE, 3, n);
            auto t     = ggml_ext_zeros(ctx->ggml_ctx, n, 1, 1, 1);
            t          = ggml_reshape_1d(ctx->ggml_ctx, t, n);
            auto c     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"])->forward(ctx, t);

            auto s0 = std::dynamic_pointer_cast<Conv2d>(blocks["s_embedder.proj1"])->forward(ctx, image);
            s0      = ggml_concat(ctx->ggml_ctx, s0, cond, 2);
            auto s  = std::dynamic_pointer_cast<Conv2d>(blocks["s_embedder.proj2"])->forward(ctx, s0);
            for (int i = 0; i < 21; ++i) {
                s = std::dynamic_pointer_cast<DiCoBlock>(blocks["blocks." + std::to_string(i)])->forward(ctx, s, c);
            }
            // [N, C, H, W] -> [N*H*W, C].
            s = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, s, 1, 2, 0, 3));
            s = ggml_reshape_2d(ctx->ggml_ctx, s, HIDDEN_SIZE, length * n);

            auto y = std::dynamic_pointer_cast<Conv2d>(blocks["y_embedder_x"])->forward(ctx, cond);
            // Split 32*P channels as [32, P], then produce [N*L, P, 32].
            y          = ggml_reshape_4d(ctx->ggml_ctx, y, length, PATCH_SIZE * PATCH_SIZE, 32, n);
            y          = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, y, 2, 1, 0, 3));
            y          = ggml_reshape_3d(ctx->ggml_ctx, y, 32, PATCH_SIZE * PATCH_SIZE, length * n);
            auto zeros = ggml_ext_zeros(ctx->ggml_ctx, 3, PATCH_SIZE * PATCH_SIZE, length * n, 1);
            dct        = ggml_repeat_4d(ctx->ggml_ctx, dct, 64, PATCH_SIZE * PATCH_SIZE, length * n, 1);
            auto x     = ggml_concat(ctx->ggml_ctx, zeros, y, 0);
            x          = ggml_concat(ctx->ggml_ctx, x, dct, 0);
            x          = std::dynamic_pointer_cast<Linear>(blocks["x_embedder.embedder.0"])->forward(ctx, x);
            x          = std::dynamic_pointer_cast<Linear>(blocks["dec_net.input_proj"])->forward(ctx, x);

            auto dec_cond = std::dynamic_pointer_cast<Linear>(blocks["dec_net.cond_embed"])->forward(ctx, s);
            dec_cond      = ggml_reshape_3d(ctx->ggml_ctx, dec_cond, 32, PATCH_SIZE * PATCH_SIZE, length * n);
            for (int i = 0; i < 3; ++i) {
                x = std::dynamic_pointer_cast<MLPResBlock>(blocks["dec_net.res_blocks." + std::to_string(i)])->forward(ctx, x, dec_cond);
            }
            x = std::dynamic_pointer_cast<RMSNorm>(blocks["final_layer.norm"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Linear>(blocks["final_layer.linear"])->forward(ctx, x);
            // [N*L, P, 3] -> [N, L, 3*P] for fold/unpatchify.
            x = ggml_reshape_4d(ctx->ggml_ctx, x, 3, PATCH_SIZE * PATCH_SIZE, length, n);
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x = ggml_reshape_3d(ctx->ggml_ctx, x, 3 * PATCH_SIZE * PATCH_SIZE, length, n);
            return DiT::unpatchify(ctx->ggml_ctx, x, h, w, PATCH_SIZE, PATCH_SIZE);
        }
    };

    struct MageVAEModel : public GGMLBlock {
        MageVAEModel() {
            blocks["student.dconv_encoder"] = std::make_shared<DConvEncoder>();
            blocks["pipeline"]              = std::make_shared<DConvDenoiser>();
        }

        ggml_tensor* encode(GGMLRunnerContext* ctx, ggml_tensor* image) {
            return std::dynamic_pointer_cast<DConvEncoder>(blocks["student.dconv_encoder"])->forward(ctx, image);
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent, ggml_tensor* dct) {
            return std::dynamic_pointer_cast<DConvDenoiser>(blocks["pipeline"])->forward(ctx, latent, dct);
        }
    };

    struct MageVAERunner : public VAE {
        MageVAEModel model;
        std::vector<float> dct_vec;

        MageVAERunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map,
                      const std::string& prefix,
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : VAE(VERSION_MAGE_FLOW, backend, prefix, weight_manager) {
            model = MageVAEModel();
            model.init(params_ctx, tensor_storage_map, prefix);
            dct_vec.resize(64 * PATCH_SIZE * PATCH_SIZE);
            constexpr float pi = 3.14159265358979323846f;
            for (int py = 0; py < PATCH_SIZE; ++py) {
                float y = static_cast<float>(py) / static_cast<float>(PATCH_SIZE - 1);
                for (int px = 0; px < PATCH_SIZE; ++px) {
                    float x = static_cast<float>(px) / static_cast<float>(PATCH_SIZE - 1);
                    int pos = py * PATCH_SIZE + px;
                    for (int fy = 0; fy < 8; ++fy) {
                        for (int fx = 0; fx < 8; ++fx) {
                            int freq                 = fx * 8 + fy;
                            float freq_x             = static_cast<float>(fx) * 8.f / 7.f;
                            float freq_y             = static_cast<float>(fy) * 8.f / 7.f;
                            float coeff              = 1.f / (1.f + freq_x * freq_y);
                            dct_vec[freq + 64 * pos] = std::cos(x * freq_x * pi) *
                                                       std::cos(y * freq_y * pi) * coeff;
                        }
                    }
                }
            }
        }

        std::string get_desc() override {
            return "mage_vae";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            model.get_param_tensors(tensors, weight_prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& input_tensor, bool decode_graph) {
            ggml_cgraph* gf  = new_graph_custom(MAGE_VAE_GRAPH_SIZE);
            auto input       = make_input(input_tensor);
            auto runner_ctx  = get_context();
            ggml_tensor* dct = nullptr;
            if (decode_graph) {
                dct = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, 64, PATCH_SIZE * PATCH_SIZE, 1);
                set_backend_tensor_data(dct, dct_vec.data());
            }
            auto out = decode_graph ? model.decode(&runner_ctx, input, dct) : model.encode(&runner_ctx, input);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& input,
                                   bool decode_graph) override {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(input, decode_graph);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), input.dim());
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return LATENT_CHANNELS * 2;
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
            const auto chunks         = sd::ops::chunk(vae_output, 2, 2);
            const auto& mean          = chunks[0];
            const auto& logvar        = chunks[1];
            sd::Tensor<float> stddev  = sd::ops::exp(0.5f * sd::ops::clamp(logvar, -20.0f, 10.0f));
            sd::Tensor<float> noise   = sd::Tensor<float>::randn_like(mean, rng);
            sd::Tensor<float> latents = mean + stddev * noise;
            return latents;
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return latents;
        }
    };
}  // namespace MageVAE

#endif  // __SD_MODEL_VAE_MAGE_VAE_HPP__

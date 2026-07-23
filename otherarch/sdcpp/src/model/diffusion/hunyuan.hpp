#ifndef __SD_MODEL_DIFFUSION_HUNYUAN_HPP__
#define __SD_MODEL_DIFFUSION_HUNYUAN_HPP__

#include <memory>

#include "model/common/block.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/mmdit.hpp"
#include "model/diffusion/wan.hpp"
#include "model_manager.h"

namespace Hunyuan {
    constexpr int HUNYUAN_VIDEO_GRAPH_SIZE = 65536;

    // Ref: https://github.com/huggingface/diffusers/pull/12696
    struct IndividualTokenRefinerBlock : public GGMLBlock {
    protected:
        int64_t num_heads;

    public:
        IndividualTokenRefinerBlock(int64_t num_heads,
                                    int64_t head_dim,
                                    int64_t mlp_ratio = 4,
                                    bool attn_bias    = true)
            : num_heads(num_heads) {
            int64_t hidden_size      = num_heads * head_dim;
            blocks["self_attn.qkv"]  = std::make_shared<Linear>(hidden_size, hidden_size * 3, attn_bias);
            blocks["self_attn.proj"] = std::make_shared<Linear>(hidden_size, hidden_size, attn_bias);

            blocks["norm1"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f, true);
            blocks["norm2"] = std::make_shared<LayerNorm>(hidden_size, 1e-6f, true);

            blocks["mlp.0"] = std::make_shared<Linear>(hidden_size, hidden_size * mlp_ratio);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size * mlp_ratio, hidden_size);

            // adaLN_modulation.0 is nn.SiLU()
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(hidden_size, hidden_size * 2);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* txt, ggml_tensor* t_emb, ggml_tensor* mask) {
            auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto norm2              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto self_attn_qkv      = std::dynamic_pointer_cast<Linear>(blocks["self_attn.qkv"]);
            auto self_attn_proj     = std::dynamic_pointer_cast<Linear>(blocks["self_attn.proj"]);
            auto mlp_fc1            = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_fc2            = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            // self attn
            auto qkv     = self_attn_qkv->forward(ctx, norm1->forward(ctx, txt));
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            auto q       = qkv_vec[0];
            auto k       = qkv_vec[1];
            auto v       = qkv_vec[2];

            auto attn_out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, mask, false, ctx->flash_attn_enabled);
            attn_out      = self_attn_proj->forward(ctx, attn_out);

            // adaLN_modulation
            auto emb  = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, t_emb));
            auto mods = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);

            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, attn_out, mods[0]));

            // mlp
            auto mlp_out = mlp_fc1->forward(ctx, norm2->forward(ctx, txt));
            mlp_out      = ggml_silu_inplace(ctx->ggml_ctx, mlp_out);
            mlp_out      = mlp_fc2->forward(ctx, mlp_out);
            txt          = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, mlp_out, mods[1]));

            return txt;
        }
    };

    struct IndividualTokenRefiner : public GGMLBlock {
    protected:
        int num_layers;

    public:
        IndividualTokenRefiner(int64_t num_heads,
                               int64_t head_dim,
                               int num_layers,
                               int64_t mlp_ratio = 4,
                               bool attn_bias    = true)
            : num_layers(num_layers) {
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<IndividualTokenRefinerBlock>(num_heads, head_dim, mlp_ratio, attn_bias);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* txt, ggml_tensor* t_emb, ggml_tensor* mask) {
            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<IndividualTokenRefinerBlock>(blocks["blocks." + std::to_string(i)]);

                txt = block->forward(ctx, txt, t_emb, mask);
            }

            return txt;
        }
    };

    struct TokenRefiner : public GGMLBlock {
    public:
        TokenRefiner(int64_t in_channels,
                     int64_t num_heads,
                     int64_t head_dim,
                     int num_layers,
                     int64_t mlp_ratio = 4,
                     bool attn_bias    = true) {
            int64_t hidden_size                = num_heads * head_dim;
            blocks["input_embedder"]           = std::make_shared<Linear>(in_channels, hidden_size);
            blocks["t_embedder"]               = std::make_shared<Flux::MLPEmbedder>(256, hidden_size);
            blocks["c_embedder"]               = std::make_shared<Flux::MLPEmbedder>(in_channels, hidden_size);
            blocks["individual_token_refiner"] = std::make_shared<IndividualTokenRefiner>(num_heads, head_dim, num_layers, mlp_ratio, attn_bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* txt, ggml_tensor* timestep, ggml_tensor* mask) {
            auto input_embedder           = std::dynamic_pointer_cast<Linear>(blocks["input_embedder"]);
            auto t_embedder               = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["t_embedder"]);
            auto c_embedder               = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["c_embedder"]);
            auto individual_token_refiner = std::dynamic_pointer_cast<IndividualTokenRefiner>(blocks["individual_token_refiner"]);

            auto t_emb = t_embedder->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, 256, 10000, 1.f));

            auto h                  = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, txt, 1, 0, 2, 3));
            auto pooled_projections = ggml_scale(ctx->ggml_ctx, ggml_sum_rows(ctx->ggml_ctx, h), 1.f / txt->ne[1]);
            pooled_projections      = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, pooled_projections, 1, 0, 2, 3));
            auto c_emb              = c_embedder->forward(ctx, pooled_projections);

            t_emb = ggml_add(ctx->ggml_ctx, t_emb, c_emb);
            txt   = input_embedder->forward(ctx, txt);
            txt   = individual_token_refiner->forward(ctx, txt, t_emb, mask);
            return txt;
        }
    };

    struct ByT5Mapper : public UnaryBlock {
        ByT5Mapper(int64_t in_dim, int64_t hidden_size) {
            blocks["layernorm"] = std::make_shared<LayerNorm>(in_dim);
            blocks["fc1"]       = std::make_shared<Linear>(in_dim, 2048);
            blocks["fc2"]       = std::make_shared<Linear>(2048, 2048);
            blocks["fc3"]       = std::make_shared<Linear>(2048, hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto layernorm = std::dynamic_pointer_cast<LayerNorm>(blocks["layernorm"]);
            auto fc1       = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2       = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
            auto fc3       = std::dynamic_pointer_cast<Linear>(blocks["fc3"]);

            x = fc1->forward(ctx, layernorm->forward(ctx, x));
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = fc2->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            return fc3->forward(ctx, x);
        }
    };

    struct HunyuanVideoConfig {
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        int64_t in_channels                  = 65;
        int64_t out_channels                 = 32;
        int64_t hidden_size                  = 2048;
        int64_t vec_in_dim                   = 0;
        int64_t context_in_dim               = 3584;
        int64_t vision_in_dim                = 0;
        float mlp_ratio                      = 4.0f;
        int num_heads                        = 16;
        int depth                            = 54;
        int depth_single_blocks              = 0;
        bool qkv_bias                        = true;
        bool guidance_embed                  = false;
        bool use_byt5                        = false;
        bool use_cond_type_embedding         = false;
        bool use_meanflow                    = false;
        bool use_meanflow_sum                = false;
        float theta                          = 256;
        std::vector<int> axes_dim            = {16, 56, 56};
        int axes_dim_sum                     = 128;

        int64_t patch_volume() const {
            return static_cast<int64_t>(std::get<0>(patch_size)) * std::get<1>(patch_size) * std::get<2>(patch_size);
        }

        static HunyuanVideoConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                      const std::string& prefix) {
            HunyuanVideoConfig config;
            config.depth               = 0;
            config.depth_single_blocks = 0;
            bool inferred              = false;

            int64_t img_embed_dim = 0;
            for (const auto& [name, storage] : tensor_storage_map) {
                if (starts_with(name, prefix) && ends_with(name, "img_in.proj.bias")) {
                    img_embed_dim = storage.ne[0];
                    break;
                }
            }

            for (const auto& entry : tensor_storage_map) {
                const auto& name    = entry.first;
                const auto& storage = entry.second;
                if (!starts_with(name, prefix)) {
                    continue;
                }

                auto update_depth = [&](const char* block_prefix, int* depth) {
                    size_t pos = name.find(block_prefix);
                    if (pos == std::string::npos) {
                        return;
                    }
                    pos += strlen(block_prefix);
                    size_t end = name.find('.', pos);
                    if (end != std::string::npos) {
                        *depth = std::max(*depth, atoi(name.substr(pos, end - pos).c_str()) + 1);
                    }
                };
                update_depth("double_blocks.", &config.depth);
                update_depth("single_blocks.", &config.depth_single_blocks);

                if (ends_with(name, "img_in.proj.weight") && storage.n_dims == 5) {
                    config.patch_size  = {static_cast<int>(storage.ne[2]),
                                          static_cast<int>(storage.ne[1]),
                                          static_cast<int>(storage.ne[0])};
                    config.in_channels = storage.ne[3];
                    config.hidden_size = storage.ne[4];
                    inferred           = true;
                } else if (ends_with(name, "img_in.proj.weight") && storage.n_dims == 4) {
                    config.patch_size = {static_cast<int>(storage.ne[2]),
                                         static_cast<int>(storage.ne[1]),
                                         static_cast<int>(storage.ne[0])};
                    if (img_embed_dim > 0 && storage.ne[3] % img_embed_dim == 0) {
                        config.hidden_size = img_embed_dim;
                        config.in_channels = storage.ne[3] / img_embed_dim;
                    }
                    inferred = true;
                } else if (ends_with(name, "txt_in.input_embedder.weight")) {
                    config.context_in_dim = storage.ne[0];
                    inferred              = true;
                } else if (ends_with(name, "vector_in.in_layer.weight")) {
                    config.vec_in_dim = storage.ne[0];
                } else if (ends_with(name, "vision_in.proj.0.weight")) {
                    config.vision_in_dim = storage.ne[0];
                } else if (ends_with(name, "double_blocks.0.img_attn.norm.key_norm.scale") ||
                           ends_with(name, "double_blocks.0.img_attn.norm.key_norm.weight")) {
                    config.num_heads = static_cast<int>(config.hidden_size / storage.ne[0]);
                } else if (ends_with(name, "double_blocks.0.img_mlp.0.weight")) {
                    config.mlp_ratio = static_cast<float>(storage.ne[1]) / static_cast<float>(storage.ne[0]);
                }

                config.guidance_embed = config.guidance_embed || name.find("guidance_in.") != std::string::npos;
                config.use_byt5       = config.use_byt5 || name.find("byt5_in.") != std::string::npos;
                config.use_meanflow   = config.use_meanflow || name.find("time_r_in.") != std::string::npos;
            }

            config.use_cond_type_embedding = tensor_storage_map.find(prefix + ".cond_type_embedding.weight") != tensor_storage_map.end();
            config.use_meanflow_sum        = config.vision_in_dim > 0;

            auto final_iter = tensor_storage_map.find(prefix + ".final_layer.linear.weight");
            if (final_iter != tensor_storage_map.end()) {
                config.out_channels = final_iter->second.ne[1] / config.patch_volume();
            }
            config.qkv_bias = tensor_storage_map.find(prefix + ".double_blocks.0.img_attn.qkv.bias") != tensor_storage_map.end();

            GGML_ASSERT(config.hidden_size % config.num_heads == 0);
            GGML_ASSERT(config.hidden_size / config.num_heads == config.axes_dim_sum);

            if (inferred) {
                LOG_DEBUG("hunyuan video: depth = %d, single depth = %d, in_channels = %" PRId64 ", out_channels = %" PRId64 ", hidden_size = %" PRId64 ", context_in_dim = %" PRId64 ", patch_size = %dx%dx%d",
                          config.depth,
                          config.depth_single_blocks,
                          config.in_channels,
                          config.out_channels,
                          config.hidden_size,
                          config.context_in_dim,
                          std::get<0>(config.patch_size),
                          std::get<1>(config.patch_size),
                          std::get<2>(config.patch_size));
            }
            return config;
        }
    };

    class HunyuanVideoModel : public GGMLBlock {
    protected:
        HunyuanVideoConfig config;

        void init_params(struct ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (config.use_cond_type_embedding) {
                ggml_type type                                  = get_type(prefix + "cond_type_embedding.weight", tensor_storage_map, GGML_TYPE_F16);
                GGMLBlock::params["cond_type_embedding.weight"] = ggml_new_tensor_2d(ctx, type, config.hidden_size, 3);
            }
        }

    public:
        HunyuanVideoModel() {}
        explicit HunyuanVideoModel(HunyuanVideoConfig config)
            : config(std::move(config)) {
            int64_t head_dim  = this->config.hidden_size / this->config.num_heads;
            blocks["txt_in"]  = std::make_shared<TokenRefiner>(this->config.context_in_dim, this->config.num_heads, head_dim, 2);
            blocks["img_in"]  = std::make_shared<PatchEmbed>(static_cast<int64_t>(224) /*Not used*/,
                                                            this->config.patch_size,
                                                            this->config.in_channels,
                                                            this->config.hidden_size);
            blocks["time_in"] = std::make_shared<Flux::MLPEmbedder>(256, this->config.hidden_size);
            if (this->config.vec_in_dim > 0) {
                blocks["vector_in"] = std::make_shared<Flux::MLPEmbedder>(this->config.vec_in_dim, this->config.hidden_size);
            }
            if (this->config.vision_in_dim > 0) {
                blocks["vision_in"] = std::make_shared<WAN::MLPProj>(this->config.vision_in_dim, this->config.hidden_size);
            }
            if (this->config.guidance_embed) {
                blocks["guidance_in"] = std::make_shared<Flux::MLPEmbedder>(256, this->config.hidden_size);
            }
            if (this->config.use_byt5) {
                blocks["byt5_in"] = std::make_shared<ByT5Mapper>(1472, this->config.hidden_size);
            }
            if (this->config.use_meanflow) {
                blocks["time_r_in"] = std::make_shared<Flux::MLPEmbedder>(256, this->config.hidden_size);
            }

            for (int i = 0; i < this->config.depth; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<Flux::DoubleStreamBlock>(this->config.hidden_size,
                                                                                                         this->config.num_heads,
                                                                                                         this->config.mlp_ratio,
                                                                                                         i,
                                                                                                         this->config.qkv_bias);
            }

            for (int i = 0; i < this->config.depth_single_blocks; i++) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<Flux::SingleStreamBlock>(this->config.hidden_size,
                                                                                                         this->config.num_heads,
                                                                                                         this->config.mlp_ratio,
                                                                                                         i,
                                                                                                         0.f);
            }

            blocks["final_layer"] = std::make_shared<Flux::LastLayer>(this->config.hidden_size,
                                                                      std::get<2>(this->config.patch_size),
                                                                      this->config.out_channels,
                                                                      false,
                                                                      true,
                                                                      this->config.patch_volume());
        }

        ggml_tensor* pad_to_patch_size(struct ggml_context* ctx,
                                       ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];

            int pt    = std::get<0>(config.patch_size);
            int ph    = std::get<1>(config.patch_size);
            int pw    = std::get<2>(config.patch_size);
            int pad_t = (pt - static_cast<int>(T % pt)) % pt;
            int pad_h = (ph - static_cast<int>(H % ph)) % ph;
            int pad_w = (pw - static_cast<int>(W % pw)) % pw;
            x         = ggml_pad(ctx, x, pad_w, pad_h, pad_t, 0);  // [N*C, T + pad_t, H + pad_h, W + pad_w]

            return x;
        }

        ggml_tensor* unpatchify(struct ggml_context* ctx,
                                ggml_tensor* x,
                                int64_t t_len,
                                int64_t h_len,
                                int64_t w_len) {
            // x: [N, t_len*h_len*w_len, C*pt*ph*pw]
            // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(config.patch_size);
            int64_t ph = std::get<1>(config.patch_size);
            int64_t pw = std::get<2>(config.patch_size);
            int64_t C  = x->ne[0] / pt / ph / pw;

            GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
            x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
            return x;
        }

        ggml_tensor* add_condition_type(GGMLRunnerContext* ctx, ggml_tensor* x, int type) {
            if (!config.use_cond_type_embedding) {
                return x;
            }
            auto weight = GGMLBlock::params["cond_type_embedding.weight"];
            auto row    = ggml_view_1d(ctx->ggml_ctx,
                                       weight,
                                       weight->ne[0],
                                       static_cast<size_t>(type) * weight->nb[1]);
            auto target = ggml_new_tensor_3d(ctx->ggml_ctx, row->type, config.hidden_size, x->ne[1], x->ne[2]);
            auto embed  = ggml_repeat(ctx->ggml_ctx, row, target);
            embed       = ggml_cast(ctx->ggml_ctx, embed, x->type);
            return ggml_add(ctx->ggml_ctx, x, embed);
        }

        ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                  ggml_tensor* img,
                                  ggml_tensor* txt,
                                  ggml_tensor* timestep,
                                  ggml_tensor* pe,
                                  ggml_tensor* guidance   = nullptr,
                                  ggml_tensor* y          = nullptr,
                                  ggml_tensor* txt_byt5   = nullptr,
                                  ggml_tensor* clip_fea   = nullptr,
                                  ggml_tensor* timestep_r = nullptr,
                                  int64_t N               = 1) {
            // img: [N*C, T, H, W], C => in_dim
            // txt: [N, L, text_dim]
            // timestep: [N,] or [T]
            // return: [N, t_len*h_len*w_len, out_dim*pt*ph*pw]

            GGML_ASSERT(N == 1);

            auto img_in      = std::dynamic_pointer_cast<PatchEmbed>(blocks["img_in"]);
            auto txt_in      = std::dynamic_pointer_cast<TokenRefiner>(blocks["txt_in"]);
            auto time_in     = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["time_in"]);
            auto final_layer = std::dynamic_pointer_cast<Flux::LastLayer>(blocks["final_layer"]);

            img      = img_in->forward(ctx, img);                     // [N*C, t_len*h_len*w_len, hidden_size]
            txt      = txt_in->forward(ctx, txt, timestep, nullptr);  // [N, n_txt_token, hidden_size]
            auto vec = time_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, 256, 10000, 1.f));
            if (config.use_meanflow && timestep_r != nullptr) {
                auto time_r_in = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["time_r_in"]);
                auto vec_r     = time_r_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep_r, 256, 10000, 1000.f));
                vec            = ggml_add(ctx->ggml_ctx, vec, vec_r);
                if (!config.use_meanflow_sum) {
                    vec = ggml_scale(ctx->ggml_ctx, vec, 0.5f);
                }
            }
            if (config.vec_in_dim > 0 && y != nullptr) {
                auto vector_in = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["vector_in"]);
                vec            = ggml_add(ctx->ggml_ctx, vec, vector_in->forward(ctx, y));
            }
            if (config.guidance_embed && guidance != nullptr) {
                auto guidance_in  = std::dynamic_pointer_cast<Flux::MLPEmbedder>(blocks["guidance_in"]);
                auto guidance_emb = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 256, 10000, 1.f);
                vec               = ggml_add(ctx->ggml_ctx, vec, guidance_in->forward(ctx, guidance_emb));
            }

            txt = add_condition_type(ctx, txt, 0);
            if (config.use_byt5 && txt_byt5 != nullptr) {
                auto byt5_in = std::dynamic_pointer_cast<ByT5Mapper>(blocks["byt5_in"]);
                txt_byt5     = add_condition_type(ctx, byt5_in->forward(ctx, txt_byt5), 1);
                txt          = config.use_cond_type_embedding ? ggml_concat(ctx->ggml_ctx, txt_byt5, txt, 1)
                                                              : ggml_concat(ctx->ggml_ctx, txt, txt_byt5, 1);
            }
            if (config.vision_in_dim > 0 && clip_fea != nullptr) {
                auto vision_in = std::dynamic_pointer_cast<WAN::MLPProj>(blocks["vision_in"]);
                clip_fea       = add_condition_type(ctx, vision_in->forward(ctx, clip_fea), 2);
                txt            = ggml_concat(ctx->ggml_ctx, clip_fea, txt, 1);
            }

            for (int i = 0; i < config.depth; i++) {
                auto block = std::dynamic_pointer_cast<Flux::DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);

                auto img_txt = block->forward(ctx, img, txt, vec, pe, nullptr);
                img          = img_txt.first;   // [N, n_img_token, hidden_size]
                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]
            }

            if (config.depth_single_blocks > 0) {
                auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
                for (int i = 0; i < config.depth_single_blocks; i++) {
                    auto block = std::dynamic_pointer_cast<Flux::SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);
                    txt_img    = block->forward(ctx, txt_img, vec, pe, nullptr);
                }

                txt_img = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, txt_img, 0, 2, 1, 3));
                img     = ggml_view_3d(ctx->ggml_ctx,
                                       txt_img,
                                       txt_img->ne[0],
                                       txt_img->ne[1],
                                       img->ne[1],
                                       txt_img->nb[1],
                                       txt_img->nb[2],
                                       txt_img->nb[2] * txt->ne[1]);
                img     = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, img, 0, 2, 1, 3));
            }

            img = final_layer->forward(ctx, img, vec);  // (N, t_len*h_len*w_len, out_channels * patch_size ** 3)

            return img;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* guidance   = nullptr,
                             ggml_tensor* y          = nullptr,
                             ggml_tensor* txt_byt5   = nullptr,
                             ggml_tensor* clip_fea   = nullptr,
                             ggml_tensor* timestep_r = nullptr,
                             int64_t N               = 1) {
            // Forward pass of DiT.
            // x: [N*C, T, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N*C, T, H, W]

            GGML_ASSERT(N == 1);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            x         = pad_to_patch_size(ctx->ggml_ctx, x);

            int64_t pt    = std::get<0>(config.patch_size);
            int64_t ph    = std::get<1>(config.patch_size);
            int64_t pw    = std::get<2>(config.patch_size);
            int64_t t_len = (T + pt - 1) / pt;
            int64_t h_len = (H + ph - 1) / ph;
            int64_t w_len = (W + pw - 1) / pw;

            auto out = forward_orig(ctx, x, context, timestep, pe, guidance, y, txt_byt5, clip_fea, timestep_r, N);

            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]

            // slice
            out = ggml_ext_slice(ctx->ggml_ctx, out, 2, 0, T);  // [N*C, T, H + pad_h, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N*C, T, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N*C, T, H, W]

            return out;
        }
    };

    struct HunyuanVideoRunner : public DiffusionModelRunner {
    public:
        HunyuanVideoConfig config;
        HunyuanVideoModel hunyuan_video;
        std::vector<float> pe_vec;
        SDVersion version;

        HunyuanVideoRunner(ggml_backend_t backend,
                           const String2TensorStorage& tensor_storage_map      = {},
                           const std::string prefix                            = "",
                           SDVersion version                                   = VERSION_HUNYUAN_VIDEO,
                           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(HunyuanVideoConfig::detect_from_weights(tensor_storage_map, prefix)),
              version(version) {
            LOG_INFO("HunyuanVideo blocks: %d double, %d single", config.depth, config.depth_single_blocks);

            hunyuan_video = HunyuanVideoModel(config);
            hunyuan_video.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "hunyuan_video";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            hunyuan_video.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& c_concat_tensor   = {},
                                 const sd::Tensor<float>& y_tensor          = {},
                                 const sd::Tensor<float>& guidance_tensor   = {},
                                 const sd::Tensor<float>& byt5_tensor       = {},
                                 const sd::Tensor<float>& vision_tensor     = {},
                                 const sd::Tensor<float>& timestep_r_tensor = {}) {
            ggml_cgraph* gf = new_graph_custom(HUNYUAN_VIDEO_GRAPH_SIZE);

            ggml_tensor* x          = make_input(x_tensor);
            ggml_tensor* timesteps  = make_input(timesteps_tensor);
            ggml_tensor* context    = make_input(context_tensor);
            ggml_tensor* c_concat   = make_optional_input(c_concat_tensor);
            ggml_tensor* y          = make_optional_input(y_tensor);
            ggml_tensor* guidance   = make_optional_input(guidance_tensor);
            ggml_tensor* byt5       = make_optional_input(byt5_tensor);
            ggml_tensor* vision     = make_optional_input(vision_tensor);
            ggml_tensor* timestep_r = make_optional_input(timestep_r_tensor);

            GGML_ASSERT(x->ne[3] == config.out_channels);
            if (c_concat != nullptr) {
                x = ggml_concat(compute_ctx, x, c_concat, 3);
            }
            GGML_ASSERT(x->ne[3] <= config.in_channels);
            if (x->ne[3] < config.in_channels) {
                x = ggml_pad(compute_ctx, x, 0, 0, 0, static_cast<int>(config.in_channels - x->ne[3]));
            }

            int text_len = static_cast<int>(context->ne[1]);
            if (byt5 != nullptr) {
                text_len += static_cast<int>(byt5->ne[1]);
            }
            if (vision != nullptr) {
                text_len += static_cast<int>(vision->ne[1]);
            }
            pe_vec          = Rope::gen_hunyuan_video_pe(static_cast<int>(x->ne[2]),
                                                         static_cast<int>(x->ne[1]),
                                                         static_cast<int>(x->ne[0]),
                                                         std::get<0>(config.patch_size),
                                                         std::get<1>(config.patch_size),
                                                         std::get<2>(config.patch_size),
                                                         1,
                                                         text_len,
                                                         config.theta,
                                                         config.axes_dim);
            int64_t pos_len = static_cast<int64_t>(pe_vec.size() / config.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx = get_context();

            ggml_tensor* out = hunyuan_video.forward(&runner_ctx,
                                                     x,
                                                     timesteps,
                                                     context,
                                                     pe,
                                                     guidance,
                                                     y,
                                                     byt5,
                                                     vision,
                                                     timestep_r);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& c_concat   = {},
                                  const sd::Tensor<float>& y          = {},
                                  const sd::Tensor<float>& guidance   = {},
                                  const sd::Tensor<float>& byt5       = {},
                                  const sd::Tensor<float>& vision     = {},
                                  const sd::Tensor<float>& timestep_r = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, byt5, vision, timestep_r);
            };

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);
            const auto* extra = diffusion_extra_as<HunyuanVideoDiffusionExtra>(diffusion_params);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           *diffusion_params.context,
                           tensor_or_empty(diffusion_params.c_concat),
                           tensor_or_empty(diffusion_params.y),
                           tensor_or_empty(extra->guidance),
                           tensor_or_empty(extra->byt5),
                           tensor_or_empty(extra->vision),
                           tensor_or_empty(extra->timestep_r));
        }
    };

}  // namespace Hunyuan

#endif  // __SD_MODEL_DIFFUSION_HUNYUAN_HPP__

#ifndef __ANIMA_HPP__
#define __ANIMA_HPP__

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "common.hpp"
#include "flux.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"

namespace Anima {
    constexpr int ANIMA_GRAPH_SIZE = 65536;

    __STATIC_INLINE__ struct ggml_tensor* patchify_2d(struct ggml_context* ctx,
                                                      struct ggml_tensor* x,
                                                      int64_t patch_size) {
        // x: [W*r, H*q, T, C]
        // return: [W, H, T, C*q*r]
        if (patch_size == 1) {
            return x;
        }
        GGML_ASSERT(x->ne[2] == 1);

        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t T = x->ne[2];
        int64_t C = x->ne[3];
        int64_t p = patch_size;
        int64_t h = H / p;
        int64_t w = W / p;

        GGML_ASSERT(T == 1);
        GGML_ASSERT(h * p == H && w * p == W);

        // Reuse Flux patchify layout on a [W, H, C, N] view.
        x = ggml_reshape_4d(ctx, x, W, H, C, T);  // [W, H, C, N]

        // Flux patchify: [N, C, H, W] -> [N, h*w, C*p*p]
        x = ggml_reshape_4d(ctx, x, p, w, p, h * C * T);           // [p, w, p, h*C*N]
        x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [p, p, w, h*C*N]
        x = ggml_reshape_4d(ctx, x, p * p, w * h, C, T);           // [p*p, h*w, C, N]
        x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [p*p, C, h*w, N]
        x = ggml_reshape_3d(ctx, x, p * p * C, w * h, T);          // [C*p*p, h*w, N]

        // Return [w, h, T, C*p*p]
        x = ggml_reshape_4d(ctx, x, p * p * C, w, h, T);           // [C*p*p, w, h, N]
        x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));  // [w, h, N, C*p*p]
        return x;
    }

    __STATIC_INLINE__ struct ggml_tensor* unpatchify_2d(struct ggml_context* ctx,
                                                        struct ggml_tensor* x,
                                                        int64_t patch_size) {
        // x: [W, H, T, C*q*r]
        // return: [W*r, H*q, T, C]
        if (patch_size == 1) {
            return x;
        }
        GGML_ASSERT(x->ne[2] == 1);

        int64_t w  = x->ne[0];
        int64_t h  = x->ne[1];
        int64_t T  = x->ne[2];
        int64_t p  = patch_size;
        int64_t nm = p * p;
        int64_t Cp = x->ne[3];
        int64_t C  = Cp / nm;
        int64_t W  = w * p;
        int64_t H  = h * p;

        GGML_ASSERT(T == 1);
        GGML_ASSERT(C * nm == Cp);

        // [w, h, 1, C*p*p] -> [W, H, 1, C]
        x = ggml_reshape_4d(ctx, x, w, h * C, p, p);                         // [w, h*C, p2, p1]
        x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 3, 1));  // [p2, w, p1, h*C]
        x = ggml_reshape_4d(ctx, x, W, H, T, C);                             // [W, H, 1, C]
        return x;
    }

    __STATIC_INLINE__ struct ggml_tensor* apply_gate(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     struct ggml_tensor* gate) {
        gate = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);  // [N, 1, C]
        return ggml_mul(ctx, x, gate);
    }

    struct XEmbedder : public GGMLBlock {
    public:
        XEmbedder(int64_t in_dim, int64_t out_dim) {
            blocks["proj.1"] = std::make_shared<Linear>(in_dim, out_dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj.1"]);
            return proj->forward(ctx, x);
        }
    };

    struct TimestepEmbedder : public GGMLBlock {
    public:
        TimestepEmbedder(int64_t in_dim, int64_t out_dim) {
            blocks["1.linear_1"] = std::make_shared<Linear>(in_dim, in_dim, false);
            blocks["1.linear_2"] = std::make_shared<Linear>(in_dim, out_dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1.linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["1.linear_2"]);

            x = linear_1->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = linear_2->forward(ctx, x);
            return x;
        }
    };

    struct AdaLayerNormZero : public GGMLBlock {
    protected:
        int64_t in_features;

    public:
        AdaLayerNormZero(int64_t in_features, int64_t hidden_features = 256)
            : in_features(in_features) {
            blocks["norm"] = std::make_shared<LayerNorm>(in_features, 1e-6f, false, false);
            blocks["1"]    = std::make_shared<Linear>(in_features, hidden_features, false);
            blocks["2"]    = std::make_shared<Linear>(hidden_features, 3 * in_features, false);
        }

        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                                    struct ggml_tensor* hidden_states,
                                                                    struct ggml_tensor* embedded_timestep,
                                                                    struct ggml_tensor* temb = nullptr) {
            auto norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            auto emb = ggml_silu(ctx->ggml_ctx, embedded_timestep);
            emb      = linear_1->forward(ctx, emb);
            emb      = linear_2->forward(ctx, emb);  // [N, 3*C]

            if (temb != nullptr) {
                emb = ggml_add(ctx->ggml_ctx, emb, temb);
            }

            auto emb_chunks = ggml_ext_chunk(ctx->ggml_ctx, emb, 3, 0);
            auto shift      = emb_chunks[0];
            auto scale      = emb_chunks[1];
            auto gate       = emb_chunks[2];

            auto x = norm->forward(ctx, hidden_states);
            x      = Flux::modulate(ctx->ggml_ctx, x, shift, scale);

            return {x, gate};
        }
    };

    struct AdaLayerNorm : public GGMLBlock {
    protected:
        int64_t embedding_dim;

    public:
        AdaLayerNorm(int64_t in_features, int64_t hidden_features = 256)
            : embedding_dim(in_features) {
            blocks["norm"] = std::make_shared<LayerNorm>(in_features, 1e-6f, false, false);
            blocks["1"]    = std::make_shared<Linear>(in_features, hidden_features, false);
            blocks["2"]    = std::make_shared<Linear>(hidden_features, 2 * in_features, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* hidden_states,
                                    struct ggml_tensor* embedded_timestep,
                                    struct ggml_tensor* temb = nullptr) {
            auto norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            auto emb = ggml_silu(ctx->ggml_ctx, embedded_timestep);
            emb      = linear_1->forward(ctx, emb);
            emb      = linear_2->forward(ctx, emb);  // [N, 2*C]

            if (temb != nullptr) {
                auto temb_2c = ggml_view_2d(ctx->ggml_ctx, temb, 2 * embedding_dim, temb->ne[1], temb->nb[1], 0);
                emb          = ggml_add(ctx->ggml_ctx, emb, temb_2c);
            }

            auto emb_chunks = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);
            auto shift      = emb_chunks[0];
            auto scale      = emb_chunks[1];

            auto x = norm->forward(ctx, hidden_states);
            x      = Flux::modulate(ctx->ggml_ctx, x, shift, scale);
            return x;
        }
    };

    struct AnimaAttention : public GGMLBlock {
    protected:
        int64_t num_heads;
        int64_t head_dim;
        std::string out_proj_name;

    public:
        AnimaAttention(int64_t query_dim,
                       int64_t context_dim,
                       int64_t num_heads,
                       int64_t head_dim,
                       const std::string& out_proj_name = "output_proj")
            : num_heads(num_heads), head_dim(head_dim), out_proj_name(out_proj_name) {
            int64_t inner_dim = num_heads * head_dim;

            blocks["q_proj"]            = std::make_shared<Linear>(query_dim, inner_dim, false);
            blocks["k_proj"]            = std::make_shared<Linear>(context_dim, inner_dim, false);
            blocks["v_proj"]            = std::make_shared<Linear>(context_dim, inner_dim, false);
            blocks["q_norm"]            = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks["k_norm"]            = std::make_shared<RMSNorm>(head_dim, 1e-6f);
            blocks[this->out_proj_name] = std::make_shared<Linear>(inner_dim, query_dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* hidden_states,
                                    struct ggml_tensor* encoder_hidden_states = nullptr,
                                    struct ggml_tensor* pe_q                  = nullptr,
                                    struct ggml_tensor* pe_k                  = nullptr) {
            if (encoder_hidden_states == nullptr) {
                encoder_hidden_states = hidden_states;
            }

            auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

            auto q = q_proj->forward(ctx, hidden_states);
            auto k = k_proj->forward(ctx, encoder_hidden_states);
            auto v = v_proj->forward(ctx, encoder_hidden_states);

            int64_t N   = q->ne[2];
            int64_t L_q = q->ne[1];
            int64_t L_k = k->ne[1];

            auto q4 = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, L_q, N);  // [N, L_q, H, D]
            auto k4 = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, L_k, N);  // [N, L_k, H, D]
            auto v4 = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, L_k, N);  // [N, L_k, H, D]

            q4 = q_norm->forward(ctx, q4);
            k4 = k_norm->forward(ctx, k4);

            struct ggml_tensor* attn_out = nullptr;
            if (pe_q != nullptr || pe_k != nullptr) {
                if (pe_q == nullptr) {
                    pe_q = pe_k;
                }
                if (pe_k == nullptr) {
                    pe_k = pe_q;
                }
                auto q_rope = Rope::apply_rope(ctx->ggml_ctx, q4, pe_q, false);
                auto k_rope = Rope::apply_rope(ctx->ggml_ctx, k4, pe_k, false);
                attn_out    = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                     ctx->backend,
                                                     q_rope,
                                                     k_rope,
                                                     v4,
                                                     num_heads,
                                                     nullptr,
                                                     true,
                                                     ctx->flash_attn_enabled);
            } else {
                auto q_flat = ggml_reshape_3d(ctx->ggml_ctx, q4, head_dim * num_heads, L_q, N);
                auto k_flat = ggml_reshape_3d(ctx->ggml_ctx, k4, head_dim * num_heads, L_k, N);
                attn_out    = ggml_ext_attention_ext(ctx->ggml_ctx,
                                                     ctx->backend,
                                                     q_flat,
                                                     k_flat,
                                                     v,
                                                     num_heads,
                                                     nullptr,
                                                     false,
                                                     ctx->flash_attn_enabled);
            }

            return out_proj->forward(ctx, attn_out);
        }
    };

    struct AnimaMLP : public GGMLBlock {
    public:
        AnimaMLP(int64_t dim, int64_t hidden_dim) {
            blocks["layer1"] = std::make_shared<Linear>(dim, hidden_dim, false);
            blocks["layer2"] = std::make_shared<Linear>(hidden_dim, dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto layer1 = std::dynamic_pointer_cast<Linear>(blocks["layer1"]);
            auto layer2 = std::dynamic_pointer_cast<Linear>(blocks["layer2"]);

            x = layer1->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x = layer2->forward(ctx, x);
            return x;
        }
    };

    struct AdapterMLP : public GGMLBlock {
    public:
        AdapterMLP(int64_t dim, int64_t hidden_dim) {
            blocks["0"] = std::make_shared<Linear>(dim, hidden_dim, true);
            blocks["2"] = std::make_shared<Linear>(hidden_dim, dim, true);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto layer0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto layer2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            x = layer0->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x, true);
            x = layer2->forward(ctx, x);
            return x;
        }
    };

    struct LLMAdapterBlock : public GGMLBlock {
    public:
        LLMAdapterBlock(int64_t model_dim = 1024, int64_t source_dim = 1024, int64_t num_heads = 16, int64_t head_dim = 64) {
            blocks["norm_self_attn"]  = std::make_shared<RMSNorm>(model_dim, 1e-6f);
            blocks["self_attn"]       = std::make_shared<AnimaAttention>(model_dim, model_dim, num_heads, head_dim, "o_proj");
            blocks["norm_cross_attn"] = std::make_shared<RMSNorm>(model_dim, 1e-6f);
            blocks["cross_attn"]      = std::make_shared<AnimaAttention>(model_dim, source_dim, num_heads, head_dim, "o_proj");
            blocks["norm_mlp"]        = std::make_shared<RMSNorm>(model_dim, 1e-6f);
            blocks["mlp"]             = std::make_shared<AdapterMLP>(model_dim, model_dim * 4);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* target_pe,
                                    struct ggml_tensor* context_pe) {
            auto norm_self_attn  = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_self_attn"]);
            auto self_attn       = std::dynamic_pointer_cast<AnimaAttention>(blocks["self_attn"]);
            auto norm_cross_attn = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_cross_attn"]);
            auto cross_attn      = std::dynamic_pointer_cast<AnimaAttention>(blocks["cross_attn"]);
            auto norm_mlp        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_mlp"]);
            auto mlp             = std::dynamic_pointer_cast<AdapterMLP>(blocks["mlp"]);

            auto h = norm_self_attn->forward(ctx, x);
            h      = self_attn->forward(ctx, h, nullptr, target_pe, target_pe);
            x      = ggml_add(ctx->ggml_ctx, x, h);

            h = norm_cross_attn->forward(ctx, x);
            h = cross_attn->forward(ctx, h, context, target_pe, context_pe);
            x = ggml_add(ctx->ggml_ctx, x, h);

            h = norm_mlp->forward(ctx, x);
            h = mlp->forward(ctx, h);
            x = ggml_add(ctx->ggml_ctx, x, h);

            return x;
        }
    };

    struct LLMAdapter : public GGMLBlock {
    protected:
        int num_layers;

    public:
        LLMAdapter(int64_t source_dim = 1024,
                   int64_t target_dim = 1024,
                   int64_t model_dim  = 1024,
                   int num_layers     = 6,
                   int num_heads      = 16)
            : num_layers(num_layers) {
            int64_t head_dim = model_dim / num_heads;

            blocks["embed"] = std::make_shared<Embedding>(32128, target_dim);
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] =
                    std::make_shared<LLMAdapterBlock>(model_dim, source_dim, num_heads, head_dim);
            }
            blocks["out_proj"] = std::make_shared<Linear>(model_dim, target_dim, true);
            blocks["norm"]     = std::make_shared<RMSNorm>(target_dim, 1e-6f);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* source_hidden_states,
                                    struct ggml_tensor* target_input_ids,
                                    struct ggml_tensor* target_pe,
                                    struct ggml_tensor* source_pe) {
            GGML_ASSERT(target_input_ids != nullptr);
            if (ggml_n_dims(target_input_ids) == 1) {
                target_input_ids = ggml_reshape_2d(ctx->ggml_ctx, target_input_ids, target_input_ids->ne[0], 1);
            }

            auto embed    = std::dynamic_pointer_cast<Embedding>(blocks["embed"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);
            auto norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed->forward(ctx, target_input_ids);  // [N, target_len, target_dim]

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<LLMAdapterBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, source_hidden_states, target_pe, source_pe);
            }

            x = out_proj->forward(ctx, x);
            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    public:
        TransformerBlock(int64_t hidden_size,
                         int64_t text_embed_dim,
                         int64_t num_heads,
                         int64_t head_dim,
                         int64_t mlp_ratio      = 4,
                         int64_t adaln_lora_dim = 256) {
            blocks["adaln_modulation_self_attn"]  = std::make_shared<AdaLayerNormZero>(hidden_size, adaln_lora_dim);
            blocks["self_attn"]                   = std::make_shared<AnimaAttention>(hidden_size, hidden_size, num_heads, head_dim);
            blocks["adaln_modulation_cross_attn"] = std::make_shared<AdaLayerNormZero>(hidden_size, adaln_lora_dim);
            blocks["cross_attn"]                  = std::make_shared<AnimaAttention>(hidden_size, text_embed_dim, num_heads, head_dim);
            blocks["adaln_modulation_mlp"]        = std::make_shared<AdaLayerNormZero>(hidden_size, adaln_lora_dim);
            blocks["mlp"]                         = std::make_shared<AnimaMLP>(hidden_size, hidden_size * mlp_ratio);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* hidden_states,
                                    struct ggml_tensor* encoder_hidden_states,
                                    struct ggml_tensor* embedded_timestep,
                                    struct ggml_tensor* temb,
                                    struct ggml_tensor* image_pe) {
            auto norm1 = std::dynamic_pointer_cast<AdaLayerNormZero>(blocks["adaln_modulation_self_attn"]);
            auto attn1 = std::dynamic_pointer_cast<AnimaAttention>(blocks["self_attn"]);
            auto norm2 = std::dynamic_pointer_cast<AdaLayerNormZero>(blocks["adaln_modulation_cross_attn"]);
            auto attn2 = std::dynamic_pointer_cast<AnimaAttention>(blocks["cross_attn"]);
            auto norm3 = std::dynamic_pointer_cast<AdaLayerNormZero>(blocks["adaln_modulation_mlp"]);
            auto mlp   = std::dynamic_pointer_cast<AnimaMLP>(blocks["mlp"]);

            auto [normed1, gate1] = norm1->forward(ctx, hidden_states, embedded_timestep, temb);
            auto h                = attn1->forward(ctx, normed1, nullptr, image_pe, image_pe);
            hidden_states         = ggml_add(ctx->ggml_ctx, hidden_states, apply_gate(ctx->ggml_ctx, h, gate1));

            auto [normed2, gate2] = norm2->forward(ctx, hidden_states, embedded_timestep, temb);
            h                     = attn2->forward(ctx, normed2, encoder_hidden_states, nullptr, nullptr);
            hidden_states         = ggml_add(ctx->ggml_ctx, hidden_states, apply_gate(ctx->ggml_ctx, h, gate2));

            auto [normed3, gate3] = norm3->forward(ctx, hidden_states, embedded_timestep, temb);
            h                     = mlp->forward(ctx, normed3);
            hidden_states         = ggml_add(ctx->ggml_ctx, hidden_states, apply_gate(ctx->ggml_ctx, h, gate3));

            return hidden_states;
        }
    };

    struct FinalLayer : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t patch_size;
        int64_t out_channels;

    public:
        FinalLayer(int64_t hidden_size, int64_t patch_size, int64_t out_channels)
            : hidden_size(hidden_size), patch_size(patch_size), out_channels(out_channels) {
            blocks["adaln_modulation"] = std::make_shared<AdaLayerNorm>(hidden_size, 256);
            blocks["linear"]           = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* hidden_states,
                                    struct ggml_tensor* embedded_timestep,
                                    struct ggml_tensor* temb) {
            auto adaln  = std::dynamic_pointer_cast<AdaLayerNorm>(blocks["adaln_modulation"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            hidden_states = adaln->forward(ctx, hidden_states, embedded_timestep, temb);
            hidden_states = linear->forward(ctx, hidden_states);
            return hidden_states;
        }
    };

    struct AnimaNet : public GGMLBlock {
    public:
        int64_t in_channels       = 16;
        int64_t out_channels      = 16;
        int64_t hidden_size       = 2048;
        int64_t text_embed_dim    = 1024;
        int64_t num_heads         = 16;
        int64_t head_dim          = 128;
        int64_t patch_size        = 2;
        int64_t num_layers        = 28;
        std::vector<int> axes_dim = {44, 42, 42};
        int theta                 = 10000;

    public:
        AnimaNet() = default;
        explicit AnimaNet(int64_t num_layers)
            : num_layers(num_layers) {
            blocks["x_embedder"]       = std::make_shared<XEmbedder>((in_channels + 1) * patch_size * patch_size, hidden_size);
            blocks["t_embedder"]       = std::make_shared<TimestepEmbedder>(hidden_size, hidden_size * 3);
            blocks["t_embedding_norm"] = std::make_shared<RMSNorm>(hidden_size, 1e-6f);
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(hidden_size,
                                                                                           text_embed_dim,
                                                                                           num_heads,
                                                                                           head_dim);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(hidden_size, patch_size, out_channels);
            blocks["llm_adapter"] = std::make_shared<LLMAdapter>(1024, 1024, 1024, 6, 16);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* encoder_hidden_states,
                                    struct ggml_tensor* image_pe,
                                    struct ggml_tensor* t5_ids       = nullptr,
                                    struct ggml_tensor* t5_weights   = nullptr,
                                    struct ggml_tensor* adapter_q_pe = nullptr,
                                    struct ggml_tensor* adapter_k_pe = nullptr) {
            GGML_ASSERT(x->ne[3] == 1);

            auto x_embedder       = std::dynamic_pointer_cast<XEmbedder>(blocks["x_embedder"]);
            auto t_embedder       = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto t_embedding_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["t_embedding_norm"]);
            auto final_layer      = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            auto llm_adapter      = std::dynamic_pointer_cast<LLMAdapter>(blocks["llm_adapter"]);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], 1, x->ne[2] * x->ne[3]);  // [N*C, T, H, W] style

            int64_t pad_h = (patch_size - H % patch_size) % patch_size;
            int64_t pad_w = (patch_size - W % patch_size) % patch_size;
            if (pad_h > 0 || pad_w > 0) {
                x = ggml_ext_pad(ctx->ggml_ctx, x, static_cast<int>(pad_w), static_cast<int>(pad_h), 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            }

            auto padding_mask = ggml_ext_zeros(ctx->ggml_ctx, x->ne[0], x->ne[1], x->ne[2], 1);
            x                 = ggml_concat(ctx->ggml_ctx, x, padding_mask, 3);  // concat mask channel

            x = patchify_2d(ctx->ggml_ctx, x, patch_size);  // [C*4, T, H/2, W/2]

            int64_t w_len = x->ne[0];
            int64_t h_len = x->ne[1];
            int64_t t_len = x->ne[2];
            x             = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3], 1);
            x             = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, n_token, C]

            x = x_embedder->forward(ctx, x);

            auto timestep_proj     = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, static_cast<int>(hidden_size));
            auto temb              = t_embedder->forward(ctx, timestep_proj);
            auto embedded_timestep = t_embedding_norm->forward(ctx, timestep_proj);

            if (t5_ids != nullptr) {
                auto adapted_context = llm_adapter->forward(ctx, encoder_hidden_states, t5_ids, adapter_q_pe, adapter_k_pe);
                if (t5_weights != nullptr) {
                    auto w = t5_weights;
                    if (ggml_n_dims(w) == 1) {
                        w = ggml_reshape_3d(ctx->ggml_ctx, w, 1, w->ne[0], 1);
                    }
                    w               = ggml_repeat_4d(ctx->ggml_ctx, w, adapted_context->ne[0], adapted_context->ne[1], adapted_context->ne[2], 1);
                    adapted_context = ggml_mul(ctx->ggml_ctx, adapted_context, w);
                }
                if (adapted_context->ne[1] < 512) {
                    auto pad_ctx    = ggml_ext_zeros(ctx->ggml_ctx,
                                                     adapted_context->ne[0],
                                                     512 - adapted_context->ne[1],
                                                     adapted_context->ne[2],
                                                     1);
                    adapted_context = ggml_concat(ctx->ggml_ctx, adapted_context, pad_ctx, 1);
                } else if (adapted_context->ne[1] > 512) {
                    adapted_context = ggml_ext_slice(ctx->ggml_ctx, adapted_context, 1, 0, 512);
                }
                encoder_hidden_states = adapted_context;
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, encoder_hidden_states, embedded_timestep, temb, image_pe);
            }

            x = final_layer->forward(ctx, x, embedded_timestep, temb);  // [N, n_token, C*4]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [n_token, C*4, N]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w_len, h_len, t_len, x->ne[1]);                    // [C*4, T, H/2, W/2]
            x = unpatchify_2d(ctx->ggml_ctx, x, patch_size);                                         // [C, T, H, W]

            x = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, H);                                  // [C, T, H, W + pad]
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, W);                                  // [C, T, H, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], x->ne[1], x->ne[3], x->ne[2]);  // [N, C, H, W]

            return x;
        }
    };

    struct AnimaRunner : public GGMLRunner {
    public:
        std::vector<float> image_pe_vec;
        std::vector<float> adapter_q_pe_vec;
        std::vector<float> adapter_k_pe_vec;
        AnimaNet net;

        AnimaRunner(ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "model.diffusion_model")
            : GGMLRunner(backend, offload_params_to_cpu) {
            int64_t num_layers    = 0;
            std::string layer_tag = prefix + ".net.blocks.";
            for (const auto& kv : tensor_storage_map) {
                const std::string& tensor_name = kv.first;
                size_t pos                     = tensor_name.find(layer_tag);
                if (pos == std::string::npos) {
                    continue;
                }
                size_t start = pos + layer_tag.size();
                size_t end   = tensor_name.find('.', start);
                if (end == std::string::npos) {
                    continue;
                }
                int64_t layer_id = atoll(tensor_name.substr(start, end - start).c_str());
                num_layers       = std::max(num_layers, layer_id + 1);
            }
            if (num_layers <= 0) {
                num_layers = 28;
            }
            LOG_INFO("anima net layers: %" PRId64, num_layers);

            net = AnimaNet(num_layers);
            net.init(params_ctx, tensor_storage_map, prefix + ".net");
        }

        std::string get_desc() override {
            return "anima";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            net.get_param_tensors(tensors, prefix + ".net");
        }

        static std::vector<float> gen_1d_rope_pe_vec(int64_t seq_len, int dim, float theta = 10000.f) {
            std::vector<float> pos(seq_len);
            for (int64_t i = 0; i < seq_len; i++) {
                pos[i] = static_cast<float>(i);
            }
            auto rope_emb = Rope::rope(pos, dim, theta);
            return Rope::flatten(rope_emb);
        }

        static float calc_ntk_factor(float extrapolation_ratio, int axis_dim) {
            if (extrapolation_ratio == 1.0f || axis_dim <= 2) {
                return 1.0f;
            }
            return std::pow(extrapolation_ratio, static_cast<float>(axis_dim) / static_cast<float>(axis_dim - 2));
        }

        static std::vector<float> gen_anima_image_pe_vec(int bs,
                                                         int h,
                                                         int w,
                                                         int patch_size,
                                                         int theta,
                                                         const std::vector<int>& axes_dim,
                                                         float h_extrapolation_ratio,
                                                         float w_extrapolation_ratio,
                                                         float t_extrapolation_ratio) {
            static const std::vector<ggml_tensor*> empty_ref_latents;
            auto ids = Rope::gen_flux_ids(h,
                                          w,
                                          patch_size,
                                          bs,
                                          static_cast<int>(axes_dim.size()),
                                          0,
                                          {},
                                          empty_ref_latents,
                                          false,
                                          1.0f);

            std::vector<float> axis_thetas = {
                static_cast<float>(theta) * calc_ntk_factor(t_extrapolation_ratio, axes_dim[0]),
                static_cast<float>(theta) * calc_ntk_factor(h_extrapolation_ratio, axes_dim[1]),
                static_cast<float>(theta) * calc_ntk_factor(w_extrapolation_ratio, axes_dim[2]),
            };
            return Rope::embed_nd(ids, bs, axis_thetas, axes_dim);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        struct ggml_tensor* t5_ids     = nullptr,
                                        struct ggml_tensor* t5_weights = nullptr) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(ANIMA_GRAPH_SIZE);

            x          = to_backend(x);
            timesteps  = to_backend(timesteps);
            context    = to_backend(context);
            t5_ids     = to_backend(t5_ids);
            t5_weights = to_backend(t5_weights);

            int64_t pad_h = (net.patch_size - x->ne[1] % net.patch_size) % net.patch_size;
            int64_t pad_w = (net.patch_size - x->ne[0] % net.patch_size) % net.patch_size;
            int64_t h_pad = x->ne[1] + pad_h;
            int64_t w_pad = x->ne[0] + pad_w;

            image_pe_vec          = gen_anima_image_pe_vec(1,
                                                           static_cast<int>(h_pad),
                                                           static_cast<int>(w_pad),
                                                           static_cast<int>(net.patch_size),
                                                           net.theta,
                                                           net.axes_dim,
                                                           4.0f,
                                                           4.0f,
                                                           1.0f);
            int64_t image_pos_len = static_cast<int64_t>(image_pe_vec.size()) / (2 * 2 * (net.head_dim / 2));
            auto image_pe         = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, net.head_dim / 2, image_pos_len);
            set_backend_tensor_data(image_pe, image_pe_vec.data());

            ggml_tensor* adapter_q_pe = nullptr;
            ggml_tensor* adapter_k_pe = nullptr;
            if (t5_ids != nullptr) {
                int64_t target_len = t5_ids->ne[0];
                int64_t source_len = context->ne[1];

                adapter_q_pe_vec = gen_1d_rope_pe_vec(target_len, 64, 10000.f);
                adapter_k_pe_vec = gen_1d_rope_pe_vec(source_len, 64, 10000.f);

                int64_t target_pos_len = static_cast<int64_t>(adapter_q_pe_vec.size()) / (2 * 2 * 32);
                int64_t source_pos_len = static_cast<int64_t>(adapter_k_pe_vec.size()) / (2 * 2 * 32);

                adapter_q_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, 32, target_pos_len);
                adapter_k_pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, 32, source_pos_len);
                set_backend_tensor_data(adapter_q_pe, adapter_q_pe_vec.data());
                set_backend_tensor_data(adapter_k_pe, adapter_k_pe_vec.data());
            }

            auto runner_ctx = get_context();
            auto out        = net.forward(&runner_ctx,
                                          x,
                                          timesteps,
                                          context,
                                          image_pe,
                                          t5_ids,
                                          t5_weights,
                                          adapter_q_pe,
                                          adapter_k_pe);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* t5_ids      = nullptr,
                     struct ggml_tensor* t5_weights  = nullptr,
                     struct ggml_tensor** output     = nullptr,
                     struct ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, t5_ids, t5_weights);
            };
            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };
}  // namespace Anima

#endif  // __ANIMA_HPP__

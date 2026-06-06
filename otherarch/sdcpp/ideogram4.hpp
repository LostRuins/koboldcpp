#ifndef __IDEOGRAM4_HPP__
#define __IDEOGRAM4_HPP__

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "diffusion_model.hpp"
#include "ggml_extend.hpp"
#include "ggml_graph_cut.h"
#include "rope.hpp"

namespace Ideogram4 {
    constexpr int IDEOGRAM4_GRAPH_SIZE    = 65536;
    constexpr int OUTPUT_IMAGE_INDICATOR  = 2;
    constexpr int IMAGE_POSITION_OFFSET   = 65536;
    constexpr int DEFAULT_MROPE_SECTION_T = 24;
    constexpr int DEFAULT_MROPE_SECTION_H = 20;
    constexpr int DEFAULT_MROPE_SECTION_W = 20;
    constexpr int TIMESTEP_MAX_PERIOD     = 10000;
    constexpr int LLM_HIDDEN_STATE_LAYERS = 13;

    struct Ideogram4Config {
        int64_t emb_dim                = 4608;
        int64_t num_layers             = 34;
        int64_t num_heads              = 18;
        int64_t intermediate_size      = 12288;
        int64_t adanln_dim             = 512;
        int64_t in_channels            = 128;
        int64_t llm_features_dim       = 53248;
        int64_t rope_theta             = 5000000;
        float norm_eps                 = 1e-5f;
        int patch_size                 = 2;
        int ae_channels                = 32;
        std::vector<int> mrope_section = {DEFAULT_MROPE_SECTION_T,
                                          DEFAULT_MROPE_SECTION_H,
                                          DEFAULT_MROPE_SECTION_W};

        static Ideogram4Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                   const std::string& prefix) {
            Ideogram4Config config;
            int64_t detected_layers  = 0;
            std::string layer_prefix = prefix.empty() ? "layers." : prefix + ".layers.";
            for (const auto& [name, _] : tensor_storage_map) {
                if (name.find(layer_prefix) != 0) {
                    continue;
                }
                std::string tail = name.substr(layer_prefix.size());
                size_t dot       = tail.find('.');
                if (dot == std::string::npos) {
                    continue;
                }
                int layer_idx   = std::atoi(tail.substr(0, dot).c_str());
                detected_layers = std::max<int64_t>(detected_layers, layer_idx + 1);
            }
            if (detected_layers > 0) {
                config.num_layers = detected_layers;
                LOG_DEBUG("ideogram4: num_layers = %" PRId64 ", emb_dim = %" PRId64 ", num_heads = %" PRId64 ", intermediate_size = %" PRId64,
                          config.num_layers,
                          config.emb_dim,
                          config.num_heads,
                          config.intermediate_size);
            }
            return config;
        }
    };

    __STATIC_INLINE__ ggml_tensor* timestep_embedding_sin_cos(ggml_context* ctx,
                                                              ggml_tensor* timesteps,
                                                              int dim) {
        GGML_ASSERT(dim % 2 == 0);
        auto embedding = ggml_ext_timestep_embedding(ctx, timesteps, dim, TIMESTEP_MAX_PERIOD, 10.f);
        auto chunks    = ggml_ext_chunk(ctx, embedding, 2, 0);
        return ggml_concat(ctx, chunks[1], chunks[0], 0);
    }

    __STATIC_INLINE__ ggml_tensor* to_token_modulation(ggml_context* ctx, ggml_tensor* x) {
        // [N, C] -> [N, 1, C] in PyTorch layout.
        if (ggml_n_dims(x) < 3 || x->ne[1] != 1) {
            x = ggml_reshape_3d(ctx, x, x->ne[0], 1, x->ne[1]);
        }
        return x;
    }

    __STATIC_INLINE__ ggml_tensor* interleave_hidden_state_layers(ggml_context* ctx, ggml_tensor* x) {
        // Match upstream stack(...).permute(1, 2, 3, 0).reshape(...):
        // [layers * hidden, tokens, batch] -> [hidden * layers, tokens, batch].
        GGML_ASSERT(x->ne[0] % LLM_HIDDEN_STATE_LAYERS == 0);
        const int64_t hidden_size = x->ne[0] / LLM_HIDDEN_STATE_LAYERS;
        const int64_t token_count = x->ne[1];
        const int64_t batch_count = x->ne[2];

        x = ggml_reshape_4d(ctx, x, hidden_size, LLM_HIDDEN_STATE_LAYERS, token_count, batch_count);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        return ggml_reshape_3d(ctx, x, hidden_size * LLM_HIDDEN_STATE_LAYERS, token_count, batch_count);
    }

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale) {
        scale = to_token_modulation(ctx, scale);
        return ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    }

    __STATIC_INLINE__ ggml_tensor* patchify(ggml_context* ctx, ggml_tensor* x, const Ideogram4Config& config) {
        // x: [N, 128, H, W] with channel order [ae, ph, pw].
        // return: [N, H*W, 128] with token channel order [ph, pw, ae].
        const int64_t W = x->ne[0];
        const int64_t H = x->ne[1];
        const int64_t C = x->ne[2];
        const int64_t N = x->ne[3];

        GGML_ASSERT(N == 1);
        GGML_ASSERT(C == config.ae_channels * config.patch_size * config.patch_size);

        x = ggml_cont(ctx, x);
        x = ggml_reshape_4d(ctx, x, W * H, config.patch_size, config.patch_size, config.ae_channels);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 1, 2, 0));
        x = ggml_reshape_3d(ctx, x, C, W * H, N);
        return x;
    }

    __STATIC_INLINE__ ggml_tensor* unpatchify(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int64_t H,
                                              int64_t W,
                                              const Ideogram4Config& config) {
        const int64_t C = x->ne[0];
        const int64_t N = x->ne[2];

        GGML_ASSERT(N == 1);
        GGML_ASSERT(C == config.ae_channels * config.patch_size * config.patch_size);
        GGML_ASSERT(x->ne[1] == H * W);

        x = ggml_reshape_4d(ctx, x, config.ae_channels, config.patch_size, config.patch_size, H * W);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 1, 2, 0));
        x = ggml_reshape_4d(ctx, x, W, H, C, N);
        return x;
    }

    __STATIC_INLINE__ std::shared_ptr<Linear> make_linear(int64_t in_features,
                                                          int64_t out_features,
                                                          bool bias = true) {
        return std::make_shared<Linear>(in_features, out_features, bias, false, false, 1.f, true);
    }

    __STATIC_INLINE__ std::vector<float> gen_ideogram4_pe(int grid_h,
                                                          int grid_w,
                                                          int bs,
                                                          int context_len,
                                                          int head_dim,
                                                          int rope_theta,
                                                          const std::vector<int>& mrope_section) {
        GGML_ASSERT(bs == 1);
        std::vector<std::vector<float>> ids(static_cast<size_t>(bs) * (context_len + grid_h * grid_w),
                                            std::vector<float>(3, 0.f));

        for (int i = 0; i < context_len; ++i) {
            ids[i] = {static_cast<float>(i), static_cast<float>(i), static_cast<float>(i)};
        }

        int cursor = context_len;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                ids[cursor++] = {static_cast<float>(IMAGE_POSITION_OFFSET),
                                 static_cast<float>(IMAGE_POSITION_OFFSET + y),
                                 static_cast<float>(IMAGE_POSITION_OFFSET + x)};
            }
        }

        return Rope::embed_interleaved_mrope(ids, bs, static_cast<float>(rope_theta), head_dim, mrope_section);
    }

    class Ideogram4Attention : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t num_heads;
        int64_t head_dim;

    public:
        Ideogram4Attention(int64_t hidden_size, int64_t num_heads, float eps)
            : hidden_size(hidden_size), num_heads(num_heads), head_dim(hidden_size / num_heads) {
            GGML_ASSERT(hidden_size % num_heads == 0);
            blocks["qkv"]    = make_linear(hidden_size, hidden_size * 3, false);
            blocks["norm_q"] = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["norm_k"] = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["o"]      = make_linear(hidden_size, hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];

            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm_q   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

            auto qkv     = qkv_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            auto q       = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, n_token, N);
            auto k       = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, n_token, N);
            auto v       = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, n_token, N);

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f / 128.f, false);
            x = out_proj->forward(ctx, x);
            return x;
        }
    };

    class Ideogram4MLP : public GGMLBlock {
    public:
        Ideogram4MLP(int64_t dim, int64_t hidden_dim) {
            blocks["w1"] = make_linear(dim, hidden_dim, false);
            blocks["w2"] = make_linear(hidden_dim, dim, false);
            blocks["w3"] = make_linear(dim, hidden_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            auto x1 = ggml_silu(ctx->ggml_ctx, w1->forward(ctx, x));
            auto x3 = w3->forward(ctx, x);
            x       = ggml_mul(ctx->ggml_ctx, x1, x3);
            x       = w2->forward(ctx, x);
            return x;
        }
    };

    class Ideogram4TransformerBlock : public GGMLBlock {
    public:
        Ideogram4TransformerBlock(const Ideogram4Config& config) {
            blocks["attention"]        = std::make_shared<Ideogram4Attention>(config.emb_dim, config.num_heads, config.norm_eps);
            blocks["feed_forward"]     = std::make_shared<Ideogram4MLP>(config.emb_dim, config.intermediate_size);
            blocks["attention_norm1"]  = std::make_shared<RMSNorm>(config.emb_dim, config.norm_eps);
            blocks["ffn_norm1"]        = std::make_shared<RMSNorm>(config.emb_dim, config.norm_eps);
            blocks["attention_norm2"]  = std::make_shared<RMSNorm>(config.emb_dim, config.norm_eps);
            blocks["ffn_norm2"]        = std::make_shared<RMSNorm>(config.emb_dim, config.norm_eps);
            blocks["adaln_modulation"] = make_linear(config.adanln_dim, 4 * config.emb_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* adaln_input,
                             ggml_tensor* mask = nullptr) {
            auto attention        = std::dynamic_pointer_cast<Ideogram4Attention>(blocks["attention"]);
            auto feed_forward     = std::dynamic_pointer_cast<Ideogram4MLP>(blocks["feed_forward"]);
            auto attention_norm1  = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1        = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2  = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2        = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);
            auto adaln_modulation = std::dynamic_pointer_cast<Linear>(blocks["adaln_modulation"]);

            auto mod       = adaln_modulation->forward(ctx, adaln_input);
            auto mods      = ggml_ext_chunk(ctx->ggml_ctx, mod, 4, 0);
            auto scale_msa = mods[0];
            auto gate_msa  = to_token_modulation(ctx->ggml_ctx, ggml_tanh(ctx->ggml_ctx, mods[1]));
            auto scale_mlp = mods[2];
            auto gate_mlp  = to_token_modulation(ctx->ggml_ctx, ggml_tanh(ctx->ggml_ctx, mods[3]));

            auto attn_out = attention_norm1->forward(ctx, x);
            attn_out      = modulate(ctx->ggml_ctx, attn_out, scale_msa);
            attn_out      = attention->forward(ctx, attn_out, pe, mask);
            attn_out      = attention_norm2->forward(ctx, attn_out);
            x             = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));

            auto ffn_out = ffn_norm1->forward(ctx, x);
            ffn_out      = modulate(ctx->ggml_ctx, ffn_out, scale_mlp);
            ffn_out      = feed_forward->forward(ctx, ffn_out);
            ffn_out      = ffn_norm2->forward(ctx, ffn_out);
            x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, ffn_out, gate_mlp));

            return x;
        }
    };

    class Ideogram4EmbedScalar : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        Ideogram4EmbedScalar(int64_t dim)
            : dim(dim) {
            blocks["mlp_in"]  = make_linear(dim, dim, true);
            blocks["mlp_out"] = make_linear(dim, dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto mlp_in  = std::dynamic_pointer_cast<Linear>(blocks["mlp_in"]);
            auto mlp_out = std::dynamic_pointer_cast<Linear>(blocks["mlp_out"]);

            x = timestep_embedding_sin_cos(ctx->ggml_ctx, x, static_cast<int>(dim));
            x = ggml_silu(ctx->ggml_ctx, mlp_in->forward(ctx, x));
            x = mlp_out->forward(ctx, x);
            return x;
        }
    };

    class Ideogram4FinalLayer : public GGMLBlock {
    public:
        Ideogram4FinalLayer(const Ideogram4Config& config) {
            blocks["norm_final"]       = std::make_shared<LayerNorm>(config.emb_dim, 1e-6f, false);
            blocks["linear"]           = make_linear(config.emb_dim, config.in_channels, true);
            blocks["adaln_modulation"] = make_linear(config.adanln_dim, config.emb_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* c) {
            auto norm_final       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear           = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaln_modulation = std::dynamic_pointer_cast<Linear>(blocks["adaln_modulation"]);

            auto scale = adaln_modulation->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);
            return x;
        }
    };

    class Ideogram4Transformer : public GGMLBlock {
    protected:
        Ideogram4Config config;

    public:
        Ideogram4Transformer() = default;
        explicit Ideogram4Transformer(Ideogram4Config config)
            : config(std::move(config)) {
            blocks["input_proj"]            = make_linear(this->config.in_channels, this->config.emb_dim, true);
            blocks["llm_cond_norm"]         = std::make_shared<RMSNorm>(this->config.llm_features_dim, 1e-6f);
            blocks["llm_cond_proj"]         = make_linear(this->config.llm_features_dim, this->config.emb_dim, true);
            blocks["t_embedding"]           = std::make_shared<Ideogram4EmbedScalar>(this->config.emb_dim);
            blocks["adaln_proj"]            = make_linear(this->config.emb_dim, this->config.adanln_dim, true);
            blocks["embed_image_indicator"] = std::make_shared<Embedding>(2, this->config.emb_dim);

            for (int i = 0; i < this->config.num_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<Ideogram4TransformerBlock>(this->config);
            }
            blocks["final_layer"] = std::make_shared<Ideogram4FinalLayer>(this->config);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             ggml_tensor* image_indicator_ids) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t N = x->ne[3];
            GGML_ASSERT(N == 1);

            auto input_proj            = std::dynamic_pointer_cast<Linear>(blocks["input_proj"]);
            auto llm_cond_norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["llm_cond_norm"]);
            auto llm_cond_proj         = std::dynamic_pointer_cast<Linear>(blocks["llm_cond_proj"]);
            auto t_embedding           = std::dynamic_pointer_cast<Ideogram4EmbedScalar>(blocks["t_embedding"]);
            auto adaln_proj            = std::dynamic_pointer_cast<Linear>(blocks["adaln_proj"]);
            auto embed_image_indicator = std::dynamic_pointer_cast<Embedding>(blocks["embed_image_indicator"]);
            auto final_layer           = std::dynamic_pointer_cast<Ideogram4FinalLayer>(blocks["final_layer"]);

            auto img = patchify(ctx->ggml_ctx, x, config);
            img      = input_proj->forward(ctx, img);

            ggml_tensor* h      = img;
            int64_t context_len = 0;
            if (context != nullptr) {
                if (ggml_n_dims(context) < 3) {
                    context = ggml_reshape_3d(ctx->ggml_ctx, context, context->ne[0], context->ne[1], 1);
                }
                context     = interleave_hidden_state_layers(ctx->ggml_ctx, context);
                context_len = context->ne[1];
                auto txt    = llm_cond_norm->forward(ctx, context);
                txt         = llm_cond_proj->forward(ctx, txt);
                h           = ggml_concat(ctx->ggml_ctx, txt, img, 1);
            }

            auto indicator_embedding = embed_image_indicator->forward(ctx, image_indicator_ids);
            h                        = ggml_add(ctx->ggml_ctx, h, indicator_embedding);

            auto t_cond      = t_embedding->forward(ctx, timestep);
            auto adaln_input = ggml_silu(ctx->ggml_ctx, adaln_proj->forward(ctx, t_cond));

            for (int i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<Ideogram4TransformerBlock>(blocks["layers." + std::to_string(i)]);
                h          = block->forward(ctx, h, pe, adaln_input, nullptr);
                sd::ggml_graph_cut::mark_graph_cut(h, "ideogram4.layers." + std::to_string(i), "hidden");
            }

            h = final_layer->forward(ctx, h, adaln_input);
            if (context_len > 0) {
                h = ggml_ext_slice(ctx->ggml_ctx, h, 1, context_len, h->ne[1]);
            }

            h = unpatchify(ctx->ggml_ctx, h, H, W, config);
            h = ggml_ext_scale(ctx->ggml_ctx, h, -1.f);
            return h;
        }
    };

    class Ideogram4Runner : public DiffusionModelRunner {
    protected:
        bool should_use_uncond_model(const DiffusionParams& diffusion_params) const {
            return has_uncond_model &&
                   diffusion_params.context == nullptr &&
                   diffusion_params.y != nullptr &&
                   !diffusion_params.y->empty();
        }

    public:
        Ideogram4Config config;
        Ideogram4Transformer model;
        Ideogram4Transformer uncond_model;
        bool has_uncond_model = false;
        std::string uncond_prefix;
        std::vector<float> pe_vec;
        std::vector<int32_t> image_indicator_vec;

        Ideogram4Runner(ggml_backend_t backend,
                        ggml_backend_t params_backend,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "")
            : DiffusionModelRunner(backend, params_backend, prefix),
              config(Ideogram4Config::detect_from_weights(tensor_storage_map, prefix)),
              uncond_prefix(prefix + ".uncond") {
            model = Ideogram4Transformer(config);
            model.init(params_ctx, tensor_storage_map, prefix);
            for (const auto& pair : tensor_storage_map) {
                const std::string& name = pair.first;
                if (starts_with(name, uncond_prefix)) {
                    has_uncond_model = true;
                    break;
                }
            }
            if (has_uncond_model) {
                LOG_DEBUG("using uncond model");
                uncond_model = Ideogram4Transformer(config);
                uncond_model.init(params_ctx, tensor_storage_map, uncond_prefix);
            }
        }

        std::string get_desc() override {
            return "ideogram4";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
            if (has_uncond_model) {
                uncond_model.get_param_tensors(tensors, this->uncond_prefix);
            }
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 bool use_uncond_model = false) {
            ggml_cgraph* gf        = new_graph_custom(IDEOGRAM4_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            Ideogram4Transformer& active_model = use_uncond_model ? uncond_model : model;

            ggml_tensor* context = nullptr;
            int64_t context_len  = 0;
            if (!context_tensor.empty()) {
                context     = make_input(context_tensor);
                context_len = context->ne[1];
            }

            int64_t grid_w   = x->ne[0];
            int64_t grid_h   = x->ne[1];
            int64_t pos_len  = context_len + grid_h * grid_w;
            int64_t head_dim = config.emb_dim / config.num_heads;

            pe_vec  = gen_ideogram4_pe(static_cast<int>(grid_h),
                                       static_cast<int>(grid_w),
                                       static_cast<int>(x->ne[3]),
                                       static_cast<int>(context_len),
                                       static_cast<int>(head_dim),
                                       static_cast<int>(config.rope_theta),
                                       config.mrope_section);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            image_indicator_vec.assign(static_cast<size_t>(pos_len), 1);
            for (int64_t i = 0; i < context_len; ++i) {
                image_indicator_vec[static_cast<size_t>(i)] = 0;
            }
            auto indicator = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_I32, pos_len, x->ne[3]);
            set_backend_tensor_data(indicator, image_indicator_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = active_model.forward(&runner_ctx, x, timesteps, context, pe, indicator);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  bool use_uncond_model = false) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, use_uncond_model);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            bool use_uncond_model = should_use_uncond_model(diffusion_params);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           use_uncond_model);
        }
    };
}  // namespace Ideogram4

#endif  // __IDEOGRAM4_HPP__

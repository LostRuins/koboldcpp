#ifndef __SD_ERNIE_IMAGE_HPP__
#define __SD_ERNIE_IMAGE_HPP__

#include <memory>
#include <vector>

#include "common_dit.hpp"
#include "flux.hpp"
#include "qwen_image.hpp"
#include "rope.hpp"

namespace ErnieImage {
    constexpr int ERNIE_IMAGE_GRAPH_SIZE = 40960;

    __STATIC_INLINE__ ggml_tensor* timestep_embedding_sin_cos(ggml_context* ctx,
                                                              ggml_tensor* timesteps,
                                                              int dim,
                                                              int max_period = 10000) {
        auto emb       = ggml_ext_timestep_embedding(ctx, timesteps, dim, max_period, 1.0f);
        int64_t half   = dim / 2;
        auto cos_part  = ggml_view_2d(ctx, emb, half, emb->ne[1], emb->nb[1], 0);
        auto sin_part  = ggml_view_2d(ctx, emb, half, emb->ne[1], emb->nb[1], half * emb->nb[0]);
        auto sin_first = ggml_concat(ctx, sin_part, cos_part, 0);
        return sin_first;
    }

    __STATIC_INLINE__ ggml_tensor* apply_rotary_emb(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pe) {
        // x: [N, S, heads, head_dim]
        // pe: [2, S, 1, head_dim], stored as ggml [head_dim, 1, S, 2].
        int64_t head_dim = x->ne[0];
        int64_t heads    = x->ne[1];
        int64_t S        = x->ne[2];
        int64_t N        = x->ne[3];
        int64_t rot_dim  = pe->ne[0];
        GGML_ASSERT(rot_dim <= head_dim);
        GGML_ASSERT(rot_dim % 2 == 0);
        GGML_ASSERT(pe->ne[1] == 1 && pe->ne[2] == S && pe->ne[3] == 2);

        x           = ggml_cont(ctx, x);
        auto x_rot  = ggml_ext_slice(ctx, x, 0, 0, rot_dim, false);
        auto x_pass = rot_dim < head_dim ? ggml_ext_slice(ctx, x, 0, rot_dim, head_dim, false) : nullptr;

        int64_t half = rot_dim / 2;
        auto x1      = ggml_view_4d(ctx, x_rot, half, heads, S, N, x_rot->nb[1], x_rot->nb[2], x_rot->nb[3], 0);
        auto x2      = ggml_view_4d(ctx, x_rot, half, heads, S, N, x_rot->nb[1], x_rot->nb[2], x_rot->nb[3], half * x_rot->nb[0]);
        x1           = ggml_cont(ctx, x1);
        x2           = ggml_cont(ctx, x2);
        auto rotated = ggml_concat(ctx, ggml_neg(ctx, x2), x1, 0);

        auto cos_emb = ggml_ext_slice(ctx, pe, 3, 0, 1, false);
        auto sin_emb = ggml_ext_slice(ctx, pe, 3, 1, 2, false);

        auto out = ggml_add(ctx, ggml_mul(ctx, x_rot, cos_emb), ggml_mul(ctx, rotated, sin_emb));
        if (x_pass != nullptr) {
            out = ggml_concat(ctx, out, x_pass, 0);
        }
        return out;
    }

    struct ErnieImageAttention : public GGMLBlock {
        int64_t num_heads;
        int64_t head_dim;

        ErnieImageAttention(int64_t query_dim,
                            int64_t heads,
                            int64_t dim_head,
                            float eps = 1e-6f)
            : num_heads(heads), head_dim(dim_head) {
            int64_t inner_dim  = heads * dim_head;
            blocks["to_q"]     = std::make_shared<Linear>(query_dim, inner_dim, false);
            blocks["to_k"]     = std::make_shared<Linear>(query_dim, inner_dim, false);
            blocks["to_v"]     = std::make_shared<Linear>(query_dim, inner_dim, false);
            blocks["norm_q"]   = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["norm_k"]   = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, query_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, S, hidden_size]
            // pe: [S, head_dim/2, 2, 2], generated in image-token-first order.
            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto norm_q   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            int64_t S = x->ne[1];
            int64_t N = x->ne[2];

            auto q = to_q->forward(ctx, x);
            auto k = to_k->forward(ctx, x);
            auto v = to_v->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, S, N);  // [N, S, heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, S, N);  // [N, S, heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, S, N);  // [N, S, heads, head_dim]

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            q = apply_rotary_emb(ctx->ggml_ctx, q, pe);
            k = apply_rotary_emb(ctx->ggml_ctx, k, pe);

            q = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, heads, S, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);

            k = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, heads, S, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, ctx->flash_attn_enabled);  // [N, S, hidden_size]
            x = to_out_0->forward(ctx, x);
            return x;
        }
    };

    struct ErnieImageFeedForward : public GGMLBlock {
    public:
        ErnieImageFeedForward(int64_t hidden_size, int64_t ffn_hidden_size) {
            blocks["gate_proj"]  = std::make_shared<Linear>(hidden_size, ffn_hidden_size, false);
            blocks["up_proj"]    = std::make_shared<Linear>(hidden_size, ffn_hidden_size, false);
            blocks["linear_fc2"] = std::make_shared<Linear>(ffn_hidden_size, hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gate_proj  = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj    = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);

            auto gate = gate_proj->forward(ctx, x);
            gate      = ggml_ext_gelu(ctx->ggml_ctx, gate);
            x         = up_proj->forward(ctx, x);
            x         = ggml_mul(ctx->ggml_ctx, x, gate);
            x         = linear_fc2->forward(ctx, x);
            return x;
        }
    };

    struct ErnieImageSharedAdaLNBlock : public GGMLBlock {
    public:
        ErnieImageSharedAdaLNBlock(int64_t hidden_size,
                                   int64_t num_heads,
                                   int64_t ffn_hidden_size,
                                   float eps = 1e-6f) {
            blocks["adaLN_sa_ln"]    = std::make_shared<RMSNorm>(hidden_size, eps);
            blocks["self_attention"] = std::make_shared<ErnieImageAttention>(hidden_size,
                                                                             num_heads,
                                                                             hidden_size / num_heads,
                                                                             eps);
            blocks["adaLN_mlp_ln"]   = std::make_shared<RMSNorm>(hidden_size, eps);
            blocks["mlp"]            = std::make_shared<ErnieImageFeedForward>(hidden_size, ffn_hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             const std::vector<ggml_tensor*>& temb,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, image_tokens + text_tokens, hidden_size]
            auto adaLN_sa_ln    = std::dynamic_pointer_cast<RMSNorm>(blocks["adaLN_sa_ln"]);
            auto self_attention = std::dynamic_pointer_cast<ErnieImageAttention>(blocks["self_attention"]);
            auto adaLN_mlp_ln   = std::dynamic_pointer_cast<RMSNorm>(blocks["adaLN_mlp_ln"]);
            auto mlp            = std::dynamic_pointer_cast<ErnieImageFeedForward>(blocks["mlp"]);

            auto shift_msa = temb[0];
            auto scale_msa = temb[1];
            auto gate_msa  = temb[2];
            auto shift_mlp = temb[3];
            auto scale_mlp = temb[4];
            auto gate_mlp  = temb[5];

            auto residual = x;
            x             = adaLN_sa_ln->forward(ctx, x);
            x             = Flux::modulate(ctx->ggml_ctx, x, shift_msa, scale_msa, true);
            auto attn_out = self_attention->forward(ctx, x, pe, attention_mask);
            x             = ggml_add(ctx->ggml_ctx, residual, ggml_mul(ctx->ggml_ctx, attn_out, gate_msa));

            residual = x;
            x        = adaLN_mlp_ln->forward(ctx, x);
            x        = Flux::modulate(ctx->ggml_ctx, x, shift_mlp, scale_mlp, true);
            x        = ggml_add(ctx->ggml_ctx, residual, ggml_mul(ctx->ggml_ctx, mlp->forward(ctx, x), gate_mlp));
            return x;
        }
    };

    struct ErnieImageAdaLNContinuous : public GGMLBlock {
    public:
        ErnieImageAdaLNContinuous(int64_t hidden_size, float eps = 1e-6f) {
            blocks["norm"]   = std::make_shared<LayerNorm>(hidden_size, eps, false);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, hidden_size * 2, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* conditioning) {
            auto norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto mods  = ggml_ext_chunk(ctx->ggml_ctx, linear->forward(ctx, conditioning), 2, 0);
            auto scale = mods[0];
            auto shift = mods[1];

            x = norm->forward(ctx, x);
            x = Flux::modulate(ctx->ggml_ctx, x, shift, scale);
            return x;
        }
    };

    struct ErnieImageParams {
        int64_t hidden_size       = 4096;
        int64_t num_heads         = 32;
        int64_t num_layers        = 36;
        int64_t ffn_hidden_size   = 12288;
        int64_t in_channels       = 128;
        int64_t out_channels      = 128;
        int patch_size            = 1;
        int64_t text_in_dim       = 3072;
        int theta                 = 256;
        std::vector<int> axes_dim = {32, 48, 48};
        int axes_dim_sum          = 128;
        float eps                 = 1e-6f;
    };

    class ErnieImageModel : public GGMLBlock {
    public:
        ErnieImageParams params;

        ErnieImageModel() = default;
        ErnieImageModel(ErnieImageParams params)
            : params(params) {
            blocks["x_embedder.proj"] = std::make_shared<Conv2d>(params.in_channels,
                                                                 params.hidden_size,
                                                                 std::pair<int, int>{params.patch_size, params.patch_size},
                                                                 std::pair<int, int>{params.patch_size, params.patch_size},
                                                                 std::pair<int, int>{0, 0},
                                                                 std::pair<int, int>{1, 1},
                                                                 true);
            if (params.text_in_dim != params.hidden_size) {
                blocks["text_proj"] = std::make_shared<Linear>(params.text_in_dim, params.hidden_size, false);
            }
            blocks["time_embedding"]     = std::make_shared<Qwen::TimestepEmbedding>(params.hidden_size, params.hidden_size);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(params.hidden_size, 6 * params.hidden_size, true);

            for (int i = 0; i < params.num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::make_shared<ErnieImageSharedAdaLNBlock>(params.hidden_size,
                                                                                                     params.num_heads,
                                                                                                     params.ffn_hidden_size,
                                                                                                     params.eps);
            }

            blocks["final_norm"]   = std::make_shared<ErnieImageAdaLNContinuous>(params.hidden_size, params.eps);
            blocks["final_linear"] = std::make_shared<Linear>(params.hidden_size,
                                                              params.patch_size * params.patch_size * params.out_channels,
                                                              true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            // x: [N, C, H, W]
            // context: [N, text_tokens, 3072]
            // pe: [image_tokens + text_tokens, head_dim/2, 2, 2]
            GGML_ASSERT(context != nullptr);
            GGML_ASSERT(x->ne[1] % params.patch_size == 0 && x->ne[0] % params.patch_size == 0);

            int64_t W     = x->ne[0];
            int64_t H     = x->ne[1];
            int64_t Hp    = H / params.patch_size;
            int64_t Wp    = W / params.patch_size;
            int64_t n_img = Hp * Wp;
            int64_t N     = x->ne[3];

            auto x_embedder_proj = std::dynamic_pointer_cast<Conv2d>(blocks["x_embedder.proj"]);
            auto time_embedding  = std::dynamic_pointer_cast<Qwen::TimestepEmbedding>(blocks["time_embedding"]);
            auto adaLN_mod       = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);
            auto final_norm      = std::dynamic_pointer_cast<ErnieImageAdaLNContinuous>(blocks["final_norm"]);
            auto final_linear    = std::dynamic_pointer_cast<Linear>(blocks["final_linear"]);

            auto img = x_embedder_proj->forward(ctx, x);                                                  // [N, hidden_size, Hp, Wp]
            img      = ggml_reshape_3d(ctx->ggml_ctx, img, img->ne[0] * img->ne[1], img->ne[2], N);       // [N, hidden_size, image_tokens]
            img      = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));  // [N, image_tokens, hidden_size]

            auto txt       = context;
            auto text_proj = std::dynamic_pointer_cast<Linear>(blocks["text_proj"]);
            if (text_proj) {
                txt = text_proj->forward(ctx, txt);
            }

            auto hidden_states = ggml_concat(ctx->ggml_ctx, img, txt, 1);  // [N, image_tokens + text_tokens, hidden_size]

            auto sample = timestep_embedding_sin_cos(ctx->ggml_ctx, timestep, static_cast<int>(params.hidden_size));
            auto c      = time_embedding->forward(ctx, sample);  // [N, hidden_size]

            auto mod_params = adaLN_mod->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 6 * hidden_size]
            sd::ggml_graph_cut::mark_graph_cut(hidden_states, "ernie_image.prelude", "hidden_states");
            // sd::ggml_graph_cut::mark_graph_cut(mod_params, "ernie_image.prelude", "mod_params");
            auto chunks = ggml_ext_chunk(ctx->ggml_ctx, mod_params, 6, 0);
            std::vector<ggml_tensor*> temb;
            temb.reserve(6);
            for (auto chunk : chunks) {
                temb.push_back(ggml_reshape_3d(ctx->ggml_ctx, chunk, chunk->ne[0], 1, chunk->ne[1]));  // [N, 1, hidden_size]
            }

            for (int i = 0; i < params.num_layers; i++) {
                auto layer    = std::dynamic_pointer_cast<ErnieImageSharedAdaLNBlock>(blocks["layers." + std::to_string(i)]);
                hidden_states = layer->forward(ctx, hidden_states, pe, temb);
                sd::ggml_graph_cut::mark_graph_cut(hidden_states, "ernie_image.layers." + std::to_string(i), "hidden_states");
            }

            hidden_states = final_norm->forward(ctx, hidden_states, c);
            hidden_states = final_linear->forward(ctx, hidden_states);                  // [N, image_tokens, p*p*out_channels]
            auto patches  = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, 0, n_img);  // [N, image_tokens, hidden_size]

            auto out = DiT::unpatchify(ctx->ggml_ctx,
                                       patches,
                                       Hp,
                                       Wp,
                                       params.patch_size,
                                       params.patch_size,
                                       false);  // [N, out_channels, H, W]
            return out;
        }
    };

    struct ErnieImageRunner : public GGMLRunner {
        ErnieImageParams ernie_params;
        ErnieImageModel ernie_image;
        std::vector<float> pe_vec;

        ErnieImageRunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "")
            : GGMLRunner(backend, offload_params_to_cpu) {
            ernie_params.num_layers = 0;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "x_embedder.proj.weight") && tensor_storage.n_dims == 4) {
                    ernie_params.patch_size  = static_cast<int>(tensor_storage.ne[0]);
                    ernie_params.in_channels = tensor_storage.ne[2];
                    ernie_params.hidden_size = tensor_storage.ne[3];
                } else if (ends_with(name, "text_proj.weight") && tensor_storage.n_dims == 2) {
                    ernie_params.text_in_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "layers.0.self_attention.norm_q.weight")) {
                    int64_t head_dim       = tensor_storage.ne[0];
                    ernie_params.num_heads = ernie_params.hidden_size / head_dim;
                } else if (ends_with(name, "layers.0.mlp.gate_proj.weight") && tensor_storage.n_dims == 2) {
                    ernie_params.ffn_hidden_size = tensor_storage.ne[1];
                } else if (ends_with(name, "final_linear.weight") && tensor_storage.n_dims == 2) {
                    int64_t out_dim           = tensor_storage.ne[1];
                    ernie_params.out_channels = out_dim / ernie_params.patch_size / ernie_params.patch_size;
                }

                size_t pos = name.find("layers.");
                if (pos != std::string::npos) {
                    std::string layer_name = name.substr(pos);
                    auto items             = split_string(layer_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > ernie_params.num_layers) {
                            ernie_params.num_layers = block_index + 1;
                        }
                    }
                }
            }
            if (ernie_params.num_layers == 0) {
                ernie_params.num_layers = 36;
            }
            ernie_params.axes_dim_sum = 0;
            for (int axis_dim : ernie_params.axes_dim) {
                ernie_params.axes_dim_sum += axis_dim;
            }

            LOG_INFO("ernie_image: layers = %" PRId64 ", hidden_size = %" PRId64 ", heads = %" PRId64
                     ", ffn_hidden_size = %" PRId64 ", in_channels = %" PRId64 ", out_channels = %" PRId64,
                     ernie_params.num_layers,
                     ernie_params.hidden_size,
                     ernie_params.num_heads,
                     ernie_params.ffn_hidden_size,
                     ernie_params.in_channels,
                     ernie_params.out_channels);

            ernie_image = ErnieImageModel(ernie_params);
            ernie_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "ernie_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            ernie_image.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(ERNIE_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            pe_vec      = Rope::gen_ernie_image_pe(static_cast<int>(x->ne[1]),
                                                   static_cast<int>(x->ne[0]),
                                                   ernie_params.patch_size,
                                                   static_cast<int>(x->ne[3]),
                                                   static_cast<int>(context->ne[1]),
                                                   ernie_params.theta,
                                                   circular_y_enabled,
                                                   circular_x_enabled,
                                                   ernie_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / ernie_params.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, ernie_params.axes_dim_sum, 1, pos_len, 2);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = ernie_image.forward(&runner_ctx, x, timesteps, context, pe);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), x.dim());
        }
    };
}  // namespace ErnieImage

#endif  // __SD_ERNIE_IMAGE_HPP__

#ifndef __SD_LENS_HPP__
#define __SD_LENS_HPP__

#include <memory>
#include <vector>

#include "common_block.hpp"
#include "diffusion_model.hpp"
#include "flux.hpp"
#include "qwen_image.hpp"
#include "rope.hpp"

namespace Lens {
    constexpr int LENS_GRAPH_SIZE = 40960;

    struct LensTimestepProjEmbeddings : public GGMLBlock {
        LensTimestepProjEmbeddings(int64_t embedding_dim) {
            blocks["timestep_embedder"] = std::make_shared<Qwen::TimestepEmbedding>(256, embedding_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto timestep_embedder = std::dynamic_pointer_cast<Qwen::TimestepEmbedding>(blocks["timestep_embedder"]);
            auto timesteps_proj    = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1000.f);
            return timestep_embedder->forward(ctx, timesteps_proj);
        }
    };

    struct LensGateMLP : public GGMLBlock {
        LensGateMLP(int64_t dim, int64_t hidden_dim) {
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            auto gate = ggml_silu(ctx->ggml_ctx, w1->forward(ctx, x));
            auto up   = w3->forward(ctx, x);
            x         = ggml_mul(ctx->ggml_ctx, gate, up);
            return w2->forward(ctx, x);
        }
    };

    struct LensJointAttention : public GGMLBlock {
        int64_t dim_head;
        int64_t num_heads;

        LensJointAttention(int64_t query_dim,
                           int64_t dim_head,
                           int64_t num_heads,
                           float eps = 1e-5f)
            : dim_head(dim_head), num_heads(num_heads) {
            int64_t inner_dim = dim_head * num_heads;
            blocks["img_qkv"] = std::make_shared<Linear>(query_dim, inner_dim * 3, true);
            blocks["txt_qkv"] = std::make_shared<Linear>(query_dim, inner_dim * 3, true);

            blocks["norm_q"]       = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["norm_k"]       = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["norm_added_q"] = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["norm_added_k"] = std::make_shared<RMSNorm>(dim_head, eps);

            blocks["to_out.0"]   = std::make_shared<Linear>(inner_dim, query_dim, true);
            blocks["to_add_out"] = std::make_shared<Linear>(inner_dim, query_dim, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* pe,
                                                      ggml_tensor* mask = nullptr) {
            auto img_qkv    = std::dynamic_pointer_cast<Linear>(blocks["img_qkv"]);
            auto txt_qkv    = std::dynamic_pointer_cast<Linear>(blocks["txt_qkv"]);
            auto norm_q     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto norm_add_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_q"]);
            auto norm_add_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_added_k"]);
            auto to_out_0   = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);
            int64_t n_img   = img->ne[1];
            int64_t n_txt   = txt->ne[1];
            int64_t N       = img->ne[2];
            int64_t inner   = dim_head * num_heads;

            auto img_qkv_vec = split_qkv(ctx->ggml_ctx, img_qkv->forward(ctx, img));
            auto txt_qkv_vec = split_qkv(ctx->ggml_ctx, txt_qkv->forward(ctx, txt));

            auto img_q = ggml_reshape_4d(ctx->ggml_ctx, img_qkv_vec[0], dim_head, num_heads, n_img, N);
            auto img_k = ggml_reshape_4d(ctx->ggml_ctx, img_qkv_vec[1], dim_head, num_heads, n_img, N);
            auto img_v = ggml_reshape_4d(ctx->ggml_ctx, img_qkv_vec[2], dim_head, num_heads, n_img, N);

            img_q = norm_q->forward(ctx, img_q);
            img_k = norm_k->forward(ctx, img_k);

            auto txt_q = ggml_reshape_4d(ctx->ggml_ctx, txt_qkv_vec[0], dim_head, num_heads, n_txt, N);
            auto txt_k = ggml_reshape_4d(ctx->ggml_ctx, txt_qkv_vec[1], dim_head, num_heads, n_txt, N);
            auto txt_v = ggml_reshape_4d(ctx->ggml_ctx, txt_qkv_vec[2], dim_head, num_heads, n_txt, N);

            txt_q = norm_add_q->forward(ctx, txt_q);
            txt_k = norm_add_k->forward(ctx, txt_k);

            auto q = ggml_concat(ctx->ggml_ctx, img_q, txt_q, 2);
            auto k = ggml_concat(ctx->ggml_ctx, img_k, txt_k, 2);
            auto v = ggml_concat(ctx->ggml_ctx, img_v, txt_v, 2);

            auto attn = Rope::attention(ctx, q, k, v, pe, mask, (1.0f / 128.f));

            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             inner,
                                             n_img,
                                             N,
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             inner,
                                             n_txt,
                                             N,
                                             attn->nb[1],
                                             attn->nb[2],
                                             n_img * attn->nb[1]);

            img_attn_out = to_out_0->forward(ctx, ggml_cont(ctx->ggml_ctx, img_attn_out));
            txt_attn_out = to_add_out->forward(ctx, ggml_cont(ctx->ggml_ctx, txt_attn_out));
            return {img_attn_out, txt_attn_out};
        }
    };

    struct LensTransformerBlock : public GGMLBlock {
        LensTransformerBlock(int64_t dim,
                             int64_t num_attention_heads,
                             int64_t attention_head_dim,
                             float eps = 1e-6f) {
            int64_t mlp_hidden_dim = dim / 3 * 8;
            blocks["img_mod.1"]    = std::make_shared<Linear>(dim, 6 * dim, true);
            blocks["txt_mod.1"]    = std::make_shared<Linear>(dim, 6 * dim, true);
            blocks["img_norm1"]    = std::make_shared<RMSNorm>(dim, eps);
            blocks["img_norm2"]    = std::make_shared<RMSNorm>(dim, eps);
            blocks["txt_norm1"]    = std::make_shared<RMSNorm>(dim, eps);
            blocks["txt_norm2"]    = std::make_shared<RMSNorm>(dim, eps);
            blocks["img_mlp"]      = std::make_shared<LensGateMLP>(dim, mlp_hidden_dim);
            blocks["txt_mlp"]      = std::make_shared<LensGateMLP>(dim, mlp_hidden_dim);
            blocks["attn"]         = std::make_shared<LensJointAttention>(dim, attention_head_dim, num_attention_heads);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img,
                                                      ggml_tensor* txt,
                                                      ggml_tensor* t_emb,
                                                      ggml_tensor* pe) {
            auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
            auto img_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["img_norm2"]);
            auto txt_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<LensGateMLP>(blocks["img_mlp"]);
            auto txt_mlp   = std::dynamic_pointer_cast<LensGateMLP>(blocks["txt_mlp"]);
            auto attn      = std::dynamic_pointer_cast<LensJointAttention>(blocks["attn"]);

            auto temb = ggml_silu(ctx->ggml_ctx, t_emb);

            auto img_mod_params = img_mod_1->forward(ctx, temb);
            auto img_mod_vec    = ggml_ext_chunk(ctx->ggml_ctx, img_mod_params, 6, 0);
            auto txt_mod_params = txt_mod_1->forward(ctx, temb);
            auto txt_mod_vec    = ggml_ext_chunk(ctx->ggml_ctx, txt_mod_params, 6, 0);

            auto img_normed    = img_norm1->forward(ctx, img);
            auto img_modulated = Flux::modulate(ctx->ggml_ctx, img_normed, img_mod_vec[0], img_mod_vec[1]);
            auto txt_normed    = txt_norm1->forward(ctx, txt);
            auto txt_modulated = Flux::modulate(ctx->ggml_ctx, txt_normed, txt_mod_vec[0], txt_mod_vec[1]);

            auto [img_attn_output, txt_attn_output] = attn->forward(ctx, img_modulated, txt_modulated, pe);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn_output, img_mod_vec[2]));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn_output, txt_mod_vec[2]));

            auto img_normed2    = img_norm2->forward(ctx, img);
            auto img_modulated2 = Flux::modulate(ctx->ggml_ctx, img_normed2, img_mod_vec[3], img_mod_vec[4]);
            auto txt_normed2    = txt_norm2->forward(ctx, txt);
            auto txt_modulated2 = Flux::modulate(ctx->ggml_ctx, txt_normed2, txt_mod_vec[3], txt_mod_vec[4]);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp->forward(ctx, img_modulated2), img_mod_vec[5]));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp->forward(ctx, txt_modulated2), txt_mod_vec[5]));
            return {img, txt};
        }
    };

    struct LensAdaLayerNormContinuous : public GGMLBlock {
        int64_t hidden_size;
        float eps;

        LensAdaLayerNormContinuous(int64_t hidden_size, float eps = 1e-6f)
            : hidden_size(hidden_size), eps(eps) {
            blocks["linear"] = std::make_shared<Linear>(hidden_size, hidden_size * 2, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* conditioning) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto mods   = ggml_ext_chunk(ctx->ggml_ctx, linear->forward(ctx, ggml_silu(ctx->ggml_ctx, conditioning)), 2, 0);
            auto scale  = mods[0];
            auto shift  = mods[1];
            x           = ggml_norm(ctx->ggml_ctx, x, eps);
            return Flux::modulate(ctx->ggml_ctx, x, shift, scale);
        }
    };

    struct LensParams {
        int patch_size              = 2;
        int64_t in_channels         = 128;
        int64_t out_channels        = 32;
        int num_layers              = 48;
        int64_t attention_head_dim  = 64;
        int64_t num_attention_heads = 24;
        int64_t joint_attention_dim = 2880;
        int selected_layer_count    = 4;
        int theta                   = 10000;
        std::vector<int> axes_dim   = {8, 28, 28};
        int axes_dim_sum            = 64;
    };

    class LensModel : public GGMLBlock {
    public:
        LensParams params;

        LensModel() = default;
        LensModel(LensParams params)
            : params(params) {
            int64_t inner_dim         = params.num_attention_heads * params.attention_head_dim;
            blocks["time_text_embed"] = std::make_shared<LensTimestepProjEmbeddings>(inner_dim);
            blocks["img_in"]          = std::make_shared<Linear>(params.in_channels, inner_dim, true);
            blocks["txt_in"]          = std::make_shared<Linear>(params.joint_attention_dim * params.selected_layer_count, inner_dim, true);
            for (int i = 0; i < params.selected_layer_count; ++i) {
                blocks["txt_norm." + std::to_string(i)] = std::make_shared<RMSNorm>(params.joint_attention_dim, 1e-5f);
            }
            for (int i = 0; i < params.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<LensTransformerBlock>(inner_dim,
                                                                                                           params.num_attention_heads,
                                                                                                           params.attention_head_dim);
            }
            blocks["norm_out"] = std::make_shared<LensAdaLayerNormContinuous>(inner_dim, 1e-6f);
            blocks["proj_out"] = std::make_shared<Linear>(inner_dim, params.patch_size * params.patch_size * params.out_channels, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            GGML_ASSERT(context != nullptr);
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            auto time_text_embed = std::dynamic_pointer_cast<LensTimestepProjEmbeddings>(blocks["time_text_embed"]);
            auto img_in          = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in          = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto norm_out        = std::dynamic_pointer_cast<LensAdaLayerNormContinuous>(blocks["norm_out"]);
            auto proj_out        = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto t_emb = time_text_embed->forward(ctx, timestep);

            auto img = ggml_reshape_3d(ctx->ggml_ctx, x, W * H, C, N);
            img      = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));
            img      = img_in->forward(ctx, img);

            std::vector<ggml_tensor*> txt_chunks = ggml_ext_chunk(ctx->ggml_ctx, context, params.selected_layer_count, 0);
            ggml_tensor* txt                     = nullptr;
            for (int i = 0; i < params.selected_layer_count; ++i) {
                auto txt_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm." + std::to_string(i)]);
                auto chunk    = txt_norm->forward(ctx, txt_chunks[i]);
                txt           = txt == nullptr ? chunk : ggml_concat(ctx->ggml_ctx, txt, chunk, 0);
            }
            txt = txt_in->forward(ctx, txt);

            sd::ggml_graph_cut::mark_graph_cut(img, "lens.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(txt, "lens.prelude", "txt");

            for (int i = 0; i < params.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<LensTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                auto out   = block->forward(ctx, img, txt, t_emb, pe);
                img        = out.first;
                txt        = out.second;
                sd::ggml_graph_cut::mark_graph_cut(img, "lens.transformer_blocks." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "lens.transformer_blocks." + std::to_string(i), "txt");
            }

            img = norm_out->forward(ctx, img, t_emb);
            img = proj_out->forward(ctx, img);

            auto out = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, img, 1, 0, 2, 3));
            out      = ggml_reshape_4d(ctx->ggml_ctx, out, W, H, params.patch_size * params.patch_size * params.out_channels, N);
            return out;
        }
    };

    struct LensRunner : public DiffusionModelRunner {
        LensParams lens_params;
        LensModel lens;
        std::vector<float> pe_vec;

        LensRunner(ggml_backend_t backend,
                   ggml_backend_t params_backend,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "")
            : DiffusionModelRunner(backend, params_backend, prefix) {
            lens_params.num_layers = 0;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "img_in.weight") && tensor_storage.n_dims == 2) {
                    lens_params.in_channels         = tensor_storage.ne[0];
                    int64_t inner_dim               = tensor_storage.ne[1];
                    lens_params.num_attention_heads = inner_dim / lens_params.attention_head_dim;
                } else if (ends_with(name, "txt_in.weight") && tensor_storage.n_dims == 2) {
                    lens_params.selected_layer_count = static_cast<int>(tensor_storage.ne[0] / lens_params.joint_attention_dim);
                } else if (ends_with(name, "proj_out.weight") && tensor_storage.n_dims == 2) {
                    lens_params.out_channels = tensor_storage.ne[1] / lens_params.patch_size / lens_params.patch_size;
                } else if (ends_with(name, "transformer_blocks.0.attn.norm_q.weight") && tensor_storage.n_dims == 1) {
                    lens_params.attention_head_dim = tensor_storage.ne[0];
                }

                size_t pos = name.find("transformer_blocks.");
                if (pos != std::string::npos) {
                    std::string layer_name = name.substr(pos);
                    auto items             = split_string(layer_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > lens_params.num_layers) {
                            lens_params.num_layers = block_index + 1;
                        }
                    }
                }
            }
            if (lens_params.num_layers == 0) {
                lens_params.num_layers = 48;
            }
            lens_params.axes_dim_sum = 0;
            for (int axis_dim : lens_params.axes_dim) {
                lens_params.axes_dim_sum += axis_dim;
            }

            LOG_INFO("lens: layers = %d, in_channels = %" PRId64 ", out_channels = %" PRId64
                     ", heads = %" PRId64 ", head_dim = %" PRId64,
                     lens_params.num_layers,
                     lens_params.in_channels,
                     lens_params.out_channels,
                     lens_params.num_attention_heads,
                     lens_params.attention_head_dim);

            lens = LensModel(lens_params);
            lens.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "lens";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            lens.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(LENS_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            pe_vec      = Rope::gen_lens_pe(static_cast<int>(x->ne[1]),
                                            static_cast<int>(x->ne[0]),
                                            static_cast<int>(x->ne[3]),
                                            static_cast<int>(context->ne[1]),
                                            lens_params.theta,
                                            circular_y_enabled,
                                            circular_x_enabled,
                                            lens_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / lens_params.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, lens_params.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = lens.forward(&runner_ctx, x, timesteps, context, pe);
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

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context));
        }
    };
}  // namespace Lens

#endif  // __SD_LENS_HPP__

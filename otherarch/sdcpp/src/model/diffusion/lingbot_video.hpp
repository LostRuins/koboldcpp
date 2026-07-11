#ifndef __SD_MODEL_DIFFUSION_LINGBOT_VIDEO_HPP__
#define __SD_MODEL_DIFFUSION_LINGBOT_VIDEO_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/util.h"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/qwen_image.hpp"

namespace LingBotVideo {
    constexpr int LINGBOT_VIDEO_GRAPH_SIZE = 65536;

    struct LingBotVideoConfig {
        int patch_t                   = 1;
        int patch_h                   = 2;
        int patch_w                   = 2;
        int64_t in_channels           = 16;
        int64_t out_channels          = 16;
        int64_t hidden_size           = 2048;
        int64_t num_attention_heads   = 16;
        int64_t depth                 = 24;
        int64_t intermediate_size     = 6144;
        int64_t text_dim              = 2560;
        int64_t freq_dim              = 256;
        float norm_eps                = 1e-6f;
        int rope_theta                = 256;
        std::vector<int> axes_dim     = {32, 48, 48};
        int axes_dim_sum              = 128;
        bool qkv_bias                 = false;
        bool out_bias                 = true;
        bool patch_embed_bias         = true;
        bool timestep_mlp_bias        = true;
        int64_t num_experts           = 0;
        int64_t num_experts_per_tok   = 8;
        int64_t moe_intermediate_size = 512;
        int64_t decoder_sparse_step   = 1;
        int64_t n_shared_experts      = 0;
        bool norm_topk_prob           = true;
        float routed_scaling_factor   = 1.0f;
        int64_t n_group               = 0;
        int64_t topk_group            = 0;
        std::set<int> sparse_layers;

        static LingBotVideoConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                      const std::string& prefix) {
            LingBotVideoConfig config;
            config.depth = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }

                if (ends_with(name, "patch_embedder.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_dim    = tensor_storage.ne[0];
                    config.hidden_size   = tensor_storage.ne[1];
                    int64_t patch_volume = config.patch_t * config.patch_h * config.patch_w;
                    if (patch_dim % patch_volume == 0) {
                        config.in_channels = patch_dim / patch_volume;
                    }
                } else if (ends_with(name, "text_embedder.linear_1.weight") && tensor_storage.n_dims == 2) {
                    config.text_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "time_embedder.linear_1.weight") && tensor_storage.n_dims == 2) {
                    config.freq_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "blocks.0.attn.norm_q.weight") && tensor_storage.n_dims == 1) {
                    int64_t head_dim = tensor_storage.ne[0];
                    if (head_dim > 0) {
                        config.num_attention_heads = config.hidden_size / head_dim;
                    }
                } else if (name.find(".attn.to_q.bias") != std::string::npos) {
                    config.qkv_bias = true;
                } else if (name.find(".ffn.gate_proj.weight") != std::string::npos && tensor_storage.n_dims == 2) {
                    config.intermediate_size = tensor_storage.ne[1];
                } else if (name.find(".ffn.experts.w1") != std::string::npos && tensor_storage.n_dims == 3) {
                    config.num_experts           = tensor_storage.ne[2];
                    config.moe_intermediate_size = tensor_storage.ne[1];
                } else if (name.find(".ffn.shared_experts.gate_proj.weight") != std::string::npos && tensor_storage.n_dims == 2) {
                    if (config.moe_intermediate_size > 0) {
                        config.n_shared_experts = tensor_storage.ne[1] / config.moe_intermediate_size;
                    }
                } else if (ends_with(name, "proj_out.weight") && tensor_storage.n_dims == 2) {
                    int64_t out_dim      = tensor_storage.ne[1];
                    int64_t patch_volume = config.patch_t * config.patch_h * config.patch_w;
                    config.out_channels  = patch_volume > 0 ? out_dim / patch_volume : config.out_channels;
                }

                size_t block_pos = name.find("blocks.");
                if (block_pos != std::string::npos) {
                    auto items = split_string(name.substr(block_pos), '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > config.depth) {
                            config.depth = block_index + 1;
                        }
                        if (name.find("blocks." + std::to_string(block_index) + ".ffn.experts.w1") != std::string::npos) {
                            config.sparse_layers.insert(block_index);
                        }
                    }
                }
            }

            if (config.depth == 0) {
                config.depth = 24;
            }
            config.axes_dim_sum = 0;
            for (int axis_dim : config.axes_dim) {
                config.axes_dim_sum += axis_dim;
            }
            if (!config.sparse_layers.empty()) {
                config.num_experts           = 128;
                config.num_experts_per_tok   = 8;
                config.moe_intermediate_size = 768;
                config.decoder_sparse_step   = 1;
                config.n_shared_experts      = 1;
                config.norm_topk_prob        = true;
                config.n_group               = 4;
                config.topk_group            = 2;
                config.routed_scaling_factor = 2.5f;
            }
            LOG_DEBUG("lingbot_video: depth = %" PRId64 ", hidden_size = %" PRId64 ", heads = %" PRId64 ", text_dim = %" PRId64 ", experts = %" PRId64 ", experts_per_tok = %" PRId64 ", n_group = %" PRId64 ", topk_group = %" PRId64 ", route_scale = %.2f, sparse_layers = %zu",
                      config.depth,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.text_dim,
                      config.num_experts,
                      config.num_experts_per_tok,
                      config.n_group,
                      config.topk_group,
                      config.routed_scaling_factor,
                      config.sparse_layers.size());
            return config;
        }
    };

    struct LingBotVideoTextEmbedder : public GGMLBlock {
        LingBotVideoTextEmbedder(int64_t text_dim,
                                 int64_t hidden_size,
                                 float eps = 1e-6f) {
            blocks["norm"]     = std::make_shared<RMSNorm>(text_dim, eps);
            blocks["linear_1"] = std::make_shared<Linear>(text_dim, hidden_size, true);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            x = norm->forward(ctx, x);
            x = linear_1->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = linear_2->forward(ctx, x);
            return x;
        }
    };

    struct LingBotVideoAttention : public GGMLBlock {
        int64_t num_heads;
        int64_t head_dim;

        LingBotVideoAttention(int64_t hidden_size,
                              int64_t num_heads,
                              bool qkv_bias = false,
                              bool out_bias = true,
                              float eps     = 1e-6f)
            : num_heads(num_heads),
              head_dim(hidden_size / num_heads) {
            int64_t inner_dim = num_heads * head_dim;
            blocks["to_q"]    = std::make_shared<Linear>(hidden_size, inner_dim, qkv_bias);
            blocks["to_k"]    = std::make_shared<Linear>(hidden_size, inner_dim, qkv_bias);
            blocks["to_v"]    = std::make_shared<Linear>(hidden_size, inner_dim, qkv_bias);
            blocks["norm_q"]  = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["norm_k"]  = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["to_out"]  = std::make_shared<Linear>(inner_dim, hidden_size, out_bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, video_tokens + text_tokens, hidden_size]
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto norm_q = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);

            int64_t S = x->ne[1];
            int64_t N = x->ne[2];

            auto q = to_q->forward(ctx, x);
            auto k = to_k->forward(ctx, x);
            auto v = to_v->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, S, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, S, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, S, N);

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            x = Rope::attention(ctx, q, k, v, pe, attention_mask);
            x = to_out->forward(ctx, x);
            return x;
        }
    };

    struct LingBotVideoMLP : public UnaryBlock {
        LingBotVideoMLP(int64_t hidden_size,
                        int64_t intermediate_size) {
            blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, false);
            blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, false);
            blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto gate = gate_proj->forward(ctx, x);
            gate      = ggml_silu_inplace(ctx->ggml_ctx, gate);
            auto up   = up_proj->forward(ctx, x);
            x         = ggml_mul(ctx->ggml_ctx, gate, up);
            x         = down_proj->forward(ctx, x);
            return x;
        }
    };

    struct LingBotVideoSparseMoeBlock : public UnaryBlock {
        int64_t hidden_size;
        int64_t intermediate_size;
        int64_t num_experts;
        int64_t num_experts_per_tok;
        bool has_shared_experts;
        bool norm_topk_prob;
        float routed_scaling_factor;
        int64_t n_group;
        int64_t topk_group;
        std::vector<float> group_expert_mask_vec;
        bool has_correction_bias = false;

        LingBotVideoSparseMoeBlock(const LingBotVideoConfig& config)
            : hidden_size(config.hidden_size),
              intermediate_size(config.moe_intermediate_size),
              num_experts(config.num_experts),
              num_experts_per_tok(config.num_experts_per_tok),
              has_shared_experts(config.n_shared_experts > 0),
              norm_topk_prob(config.norm_topk_prob),
              routed_scaling_factor(config.routed_scaling_factor),
              n_group(config.n_group),
              topk_group(config.topk_group) {
            if (n_group > 1) {
                GGML_ASSERT(num_experts % n_group == 0);
                int64_t experts_per_group = num_experts / n_group;
                group_expert_mask_vec.assign(static_cast<size_t>(num_experts * n_group), 0.f);
                for (int64_t group = 0; group < n_group; ++group) {
                    int64_t expert_begin = group * experts_per_group;
                    int64_t expert_end   = expert_begin + experts_per_group;
                    for (int64_t expert = expert_begin; expert < expert_end; ++expert) {
                        group_expert_mask_vec[static_cast<size_t>(group * num_experts + expert)] = 1.f;
                    }
                }
            }
            if (has_shared_experts) {
                blocks["shared_experts"] = std::make_shared<LingBotVideoMLP>(hidden_size,
                                                                             intermediate_size * config.n_shared_experts);
            }
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type router_type   = get_type(prefix + "router.weight", tensor_storage_map, GGML_TYPE_F32);
            ggml_type w1_type       = get_type(prefix + "experts.w1", tensor_storage_map, GGML_TYPE_F32);
            ggml_type w2_type       = get_type(prefix + "experts.w2", tensor_storage_map, GGML_TYPE_F32);
            ggml_type w3_type       = get_type(prefix + "experts.w3", tensor_storage_map, GGML_TYPE_F32);
            params["router.weight"] = ggml_new_tensor_2d(ctx, router_type, hidden_size, num_experts);
            if (tensor_storage_map.find(prefix + "router.e_score_correction_bias") != tensor_storage_map.end()) {
                params["router.e_score_correction_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_experts);
                has_correction_bias                      = true;
            }
            params["experts.w1"] = ggml_new_tensor_3d(ctx, w1_type, hidden_size, intermediate_size, num_experts);
            params["experts.w2"] = ggml_new_tensor_3d(ctx, w2_type, intermediate_size, hidden_size, num_experts);
            params["experts.w3"] = ggml_new_tensor_3d(ctx, w3_type, hidden_size, intermediate_size, num_experts);
        }

        ggml_tensor* expert_linear(GGMLRunnerContext* ctx,
                                   const std::string& weight_name,
                                   ggml_tensor* x,
                                   ggml_tensor* selected_experts) {
            return ggml_mul_mat_id(ctx->ggml_ctx, params[weight_name], x, selected_experts);
        }

        ggml_tensor* select_experts(GGMLRunnerContext* ctx, ggml_tensor* choice_scores) {
            ggml_context* gctx = ctx->ggml_ctx;
            if (n_group <= 1 || topk_group <= 0) {
                return ggml_argsort_top_k(gctx, choice_scores, static_cast<int>(num_experts_per_tok));
            }

            GGML_ASSERT(choice_scores->ne[0] == num_experts);
            GGML_ASSERT(num_experts % n_group == 0);
            GGML_ASSERT(topk_group > 0 && topk_group <= n_group);
            GGML_ASSERT(!group_expert_mask_vec.empty());

            const int64_t n_token_total     = choice_scores->ne[1];
            const int64_t experts_per_group = num_experts / n_group;
            const int group_score_k         = 2;
            GGML_ASSERT(experts_per_group >= group_score_k);

            ggml_tensor* grouped_scores = ggml_reshape_3d(gctx, choice_scores, experts_per_group, n_group, n_token_total);
            ggml_tensor* group_top_ids  = ggml_argsort_top_k(gctx, grouped_scores, group_score_k);
            grouped_scores              = ggml_reshape_3d(gctx, grouped_scores, 1, experts_per_group, n_group * n_token_total);
            group_top_ids               = ggml_cont(gctx, group_top_ids);
            group_top_ids               = ggml_reshape_2d(gctx, group_top_ids, group_score_k, n_group * n_token_total);

            ggml_tensor* group_top_values = ggml_get_rows(gctx, grouped_scores, group_top_ids);
            group_top_values              = ggml_reshape_3d(gctx, group_top_values, group_score_k, n_group, n_token_total);

            ggml_tensor* group_scores = nullptr;
            for (int rank = 0; rank < group_score_k; ++rank) {
                ggml_tensor* value = ggml_view_3d(gctx,
                                                  group_top_values,
                                                  1,
                                                  n_group,
                                                  n_token_total,
                                                  group_top_values->nb[1],
                                                  group_top_values->nb[2],
                                                  rank * group_top_values->nb[0]);
                group_scores       = group_scores == nullptr ? value : ggml_add(gctx, group_scores, value);
            }
            group_scores = ggml_reshape_2d(gctx, group_scores, n_group, n_token_total);

            ggml_tensor* selected_groups = ggml_argsort_top_k(gctx, group_scores, static_cast<int>(topk_group));
            selected_groups              = ggml_cont(gctx, selected_groups);

            ggml_tensor* group_expert_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, num_experts, n_group, 1);
            ctx->bind_backend_tensor_data(group_expert_mask, group_expert_mask_vec.data());
            ggml_tensor* group_expert_mask_template = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, num_experts, n_group, n_token_total);
            group_expert_mask                       = ggml_repeat(gctx, group_expert_mask, group_expert_mask_template);

            ggml_tensor* selected_group_masks = ggml_get_rows(gctx, group_expert_mask, selected_groups);
            ggml_tensor* selected_mask        = nullptr;
            for (int64_t rank = 0; rank < topk_group; ++rank) {
                ggml_tensor* mask = ggml_view_3d(gctx,
                                                 selected_group_masks,
                                                 num_experts,
                                                 1,
                                                 n_token_total,
                                                 selected_group_masks->nb[1],
                                                 selected_group_masks->nb[2],
                                                 rank * selected_group_masks->nb[1]);
                selected_mask     = selected_mask == nullptr ? mask : ggml_add(gctx, selected_mask, mask);
            }
            selected_mask = ggml_reshape_2d(gctx, selected_mask, num_experts, n_token_total);

            ggml_tensor* excluded_group_mask = ggml_sub(gctx, selected_mask, ggml_ext_ones_like(gctx, selected_mask));
            ggml_tensor* masked_scores       = ggml_add(gctx, choice_scores, ggml_scale(gctx, excluded_group_mask, 1.0e9f));
            return ggml_argsort_top_k(gctx, masked_scores, static_cast<int>(num_experts_per_tok));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, tokens, hidden_size]
            GGML_ASSERT(num_experts > 0);
            GGML_ASSERT(num_experts_per_tok > 0 && num_experts_per_tok <= num_experts);

            ggml_tensor* residual       = x;
            const int64_t n_token       = x->ne[1];
            const int64_t N             = x->ne[2];
            const int64_t n_token_total = n_token * N;

            ggml_tensor* router_logits = ggml_mul_mat(ctx->ggml_ctx, params["router.weight"], x);
            router_logits              = ggml_reshape_2d(ctx->ggml_ctx, router_logits, num_experts, n_token_total);
            ggml_tensor* probs         = ggml_sigmoid(ctx->ggml_ctx, router_logits);
            ggml_tensor* choice_scores = probs;
            if (has_correction_bias) {
                choice_scores = ggml_add(ctx->ggml_ctx, choice_scores, params["router.e_score_correction_bias"]);
            }

            ggml_tensor* selected_experts = select_experts(ctx, choice_scores);
            ggml_tensor* weights          = ggml_get_rows(ctx->ggml_ctx,
                                                          ggml_reshape_3d(ctx->ggml_ctx, probs, 1, num_experts, n_token_total),
                                                          selected_experts);
            weights                       = ggml_reshape_2d(ctx->ggml_ctx, weights, num_experts_per_tok, n_token_total);
            if (norm_topk_prob && num_experts_per_tok > 1) {
                auto weights_sum = ggml_sum_rows(ctx->ggml_ctx, weights);
                weights_sum      = ggml_clamp(ctx->ggml_ctx, weights_sum, 6.103515625e-5f, INFINITY);
                weights          = ggml_div(ctx->ggml_ctx, weights, weights_sum);
            }
            if (routed_scaling_factor != 1.0f) {
                weights = ggml_scale(ctx->ggml_ctx, weights, routed_scaling_factor);
            }
            weights = ggml_reshape_3d(ctx->ggml_ctx, weights, 1, num_experts_per_tok, n_token_total);

            x         = ggml_reshape_3d(ctx->ggml_ctx, x, hidden_size, 1, n_token_total);
            auto gate = expert_linear(ctx, "experts.w1", x, selected_experts);
            gate      = ggml_silu_inplace(ctx->ggml_ctx, gate);
            auto up   = expert_linear(ctx, "experts.w3", x, selected_experts);
            auto act  = ggml_mul(ctx->ggml_ctx, gate, up);
            auto out  = expert_linear(ctx, "experts.w2", act, selected_experts);
            out       = ggml_mul(ctx->ggml_ctx, out, weights);

            ggml_tensor* summed = nullptr;
            for (int64_t i = 0; i < num_experts_per_tok; ++i) {
                auto expert_out = ggml_view_2d(ctx->ggml_ctx,
                                               out,
                                               hidden_size,
                                               n_token_total,
                                               out->nb[2],
                                               i * out->nb[1]);
                summed          = summed == nullptr ? expert_out : ggml_add(ctx->ggml_ctx, summed, expert_out);
            }
            if (num_experts_per_tok == 1) {
                summed = ggml_cont(ctx->ggml_ctx, summed);
            }
            summed = ggml_reshape_3d(ctx->ggml_ctx, summed, hidden_size, n_token, N);

            if (has_shared_experts) {
                auto shared_experts = std::dynamic_pointer_cast<LingBotVideoMLP>(blocks["shared_experts"]);
                summed              = ggml_add(ctx->ggml_ctx, summed, shared_experts->forward(ctx, residual));
            }
            return summed;
        }
    };

    struct LingBotVideoBlock : public GGMLBlock {
        int64_t hidden_size;

        LingBotVideoBlock(const LingBotVideoConfig& config,
                          bool sparse)
            : hidden_size(config.hidden_size) {
            blocks["norm1"]          = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]           = std::make_shared<LingBotVideoAttention>(config.hidden_size,
                                                                     config.num_attention_heads,
                                                                     config.qkv_bias,
                                                                     config.out_bias,
                                                                     config.norm_eps);
            blocks["norm_post_attn"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"]          = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            if (sparse) {
                blocks["ffn"] = std::make_shared<LingBotVideoSparseMoeBlock>(config);
            } else {
                blocks["ffn"] = std::make_shared<LingBotVideoMLP>(config.hidden_size, config.intermediate_size);
            }
            blocks["norm_post_ffn"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size * 6, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* temb6,
                             ggml_tensor* pe,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, tokens, hidden_size], temb6: [N, tokens, 6 * hidden_size]
            auto norm1          = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto attn           = std::dynamic_pointer_cast<LingBotVideoAttention>(blocks["attn"]);
            auto norm_post_attn = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_post_attn"]);
            auto norm2          = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto ffn            = std::dynamic_pointer_cast<UnaryBlock>(blocks["ffn"]);
            auto norm_post_ffn  = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_post_ffn"]);

            auto table = ggml_reshape_3d(ctx->ggml_ctx, params["scale_shift_table"], hidden_size * 6, 1, 1);
            auto mod   = ggml_add(ctx->ggml_ctx, temb6, table);
            auto mods  = ggml_ext_chunk(ctx->ggml_ctx, mod, 6, 0);

            auto shift_msa = mods[0];
            auto scale_msa = mods[1];
            auto gate_msa  = ggml_tanh(ctx->ggml_ctx, mods[2]);
            auto shift_mlp = mods[3];
            auto scale_mlp = mods[4];
            auto gate_mlp  = ggml_tanh(ctx->ggml_ctx, mods[5]);

            auto attn_in  = Flux::modulate(ctx->ggml_ctx, norm1->forward(ctx, x), shift_msa, scale_msa, true);
            auto attn_out = attn->forward(ctx, attn_in, pe, attention_mask);
            attn_out      = norm_post_attn->forward(ctx, attn_out);
            x             = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, gate_msa, attn_out));

            auto ffn_in  = Flux::modulate(ctx->ggml_ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp, true);
            auto ffn_out = ffn->forward(ctx, ffn_in);
            ffn_out      = norm_post_ffn->forward(ctx, ffn_out);
            x            = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, gate_mlp, ffn_out));
            return x;
        }
    };

    class LingBotVideoModel : public GGMLBlock {
    public:
        LingBotVideoConfig config;

        LingBotVideoModel() = default;
        LingBotVideoModel(LingBotVideoConfig config)
            : config(config) {
            int64_t patch_dim           = config.patch_t * config.patch_h * config.patch_w * config.in_channels;
            blocks["patch_embedder"]    = std::make_shared<Linear>(patch_dim, config.hidden_size, config.patch_embed_bias);
            blocks["time_embedder"]     = std::make_shared<Qwen::TimestepEmbedding>(config.freq_dim,
                                                                                config.hidden_size,
                                                                                config.hidden_size,
                                                                                0,
                                                                                config.timestep_mlp_bias);
            blocks["time_modulation.1"] = std::make_shared<Linear>(config.hidden_size, 6 * config.hidden_size, true);
            blocks["text_embedder"]     = std::make_shared<LingBotVideoTextEmbedder>(config.text_dim,
                                                                                 config.hidden_size,
                                                                                 config.norm_eps);
            for (int i = 0; i < config.depth; i++) {
                bool sparse                           = config.sparse_layers.find(i) != config.sparse_layers.end();
                blocks["blocks." + std::to_string(i)] = std::make_shared<LingBotVideoBlock>(config, sparse);
            }
            blocks["norm_out"]              = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps, false);
            blocks["norm_out_modulation.1"] = std::make_shared<Linear>(config.hidden_size, 2 * config.hidden_size, true);
            blocks["proj_out"]              = std::make_shared<Linear>(config.hidden_size,
                                                          config.patch_t * config.patch_h * config.patch_w * config.out_channels,
                                                          true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            // x: [N*C, T, H, W], context: [N, text_tokens, text_dim]
            GGML_ASSERT(context != nullptr);
            GGML_ASSERT(x->ne[3] == config.in_channels);
            GGML_ASSERT(x->ne[2] % config.patch_t == 0);
            GGML_ASSERT(x->ne[1] % config.patch_h == 0);
            GGML_ASSERT(x->ne[0] % config.patch_w == 0);

            auto patch_embedder      = std::dynamic_pointer_cast<Linear>(blocks["patch_embedder"]);
            auto time_embedder       = std::dynamic_pointer_cast<Qwen::TimestepEmbedding>(blocks["time_embedder"]);
            auto time_modulation     = std::dynamic_pointer_cast<Linear>(blocks["time_modulation.1"]);
            auto text_embedder       = std::dynamic_pointer_cast<LingBotVideoTextEmbedder>(blocks["text_embedder"]);
            auto norm_out            = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);
            auto norm_out_modulation = std::dynamic_pointer_cast<Linear>(blocks["norm_out_modulation.1"]);
            auto proj_out            = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            int64_t W     = x->ne[0];
            int64_t H     = x->ne[1];
            int64_t T     = x->ne[2];
            int64_t N     = 1;
            int64_t t_len = T / config.patch_t;
            int64_t h_len = H / config.patch_h;
            int64_t w_len = W / config.patch_w;
            int64_t n_img = t_len * h_len * w_len;

            auto img = DiT::patchify_3d(ctx->ggml_ctx, x, config.patch_t, config.patch_h, config.patch_w, N, false);
            img      = patch_embedder->forward(ctx, img);

            auto txt           = text_embedder->forward(ctx, context);
            auto hidden_states = ggml_concat(ctx->ggml_ctx, img, txt, 1);

            auto timestep_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx,
                                                             timestep,
                                                             static_cast<int>(config.freq_dim),
                                                             10000,
                                                             1.0f);
            auto t_emb         = time_embedder->forward(ctx, timestep_proj);
            auto temb_template = ggml_new_tensor_3d(ctx->ggml_ctx, t_emb->type, t_emb->ne[0], hidden_states->ne[1], t_emb->ne[1]);
            auto temb_tokens   = ggml_repeat(ctx->ggml_ctx,
                                             ggml_reshape_3d(ctx->ggml_ctx, t_emb, t_emb->ne[0], 1, t_emb->ne[1]),
                                             temb_template);
            auto temb6         = time_modulation->forward(ctx, ggml_silu(ctx->ggml_ctx, temb_tokens));

            sd::ggml_graph_cut::mark_graph_cut(hidden_states, "lingbot_video.prelude", "hidden_states");

            for (int i = 0; i < config.depth; i++) {
                auto block    = std::dynamic_pointer_cast<LingBotVideoBlock>(blocks["blocks." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, temb6, pe);
                sd::ggml_graph_cut::mark_graph_cut(hidden_states, "lingbot_video.blocks." + std::to_string(i), "hidden_states");
            }

            auto final_mods = ggml_ext_chunk(ctx->ggml_ctx,
                                             norm_out_modulation->forward(ctx, ggml_silu(ctx->ggml_ctx, temb_tokens)),
                                             2,
                                             0);
            hidden_states   = norm_out->forward(ctx, hidden_states);
            hidden_states   = Flux::modulate(ctx->ggml_ctx, hidden_states, final_mods[0], final_mods[1], true);
            hidden_states   = proj_out->forward(ctx, hidden_states);
            hidden_states   = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, 0, n_img);

            auto out = DiT::unpatchify_3d(ctx->ggml_ctx,
                                          hidden_states,
                                          t_len,
                                          h_len,
                                          w_len,
                                          config.patch_t,
                                          config.patch_h,
                                          config.patch_w,
                                          false);
            return out;
        }
    };

    struct LingBotVideoRunner : public DiffusionModelRunner {
        LingBotVideoConfig config;
        LingBotVideoModel lingbot_video;
        std::vector<float> pe_vec;

        LingBotVideoRunner(ggml_backend_t backend,
                           const String2TensorStorage& tensor_storage_map      = {},
                           const std::string prefix                            = "",
                           std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                           const char* model_args                              = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(LingBotVideoConfig::detect_from_weights(tensor_storage_map, prefix)) {
            SD_UNUSED(model_args);

            lingbot_video = LingBotVideoModel(config);
            lingbot_video.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "lingbot_video";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            lingbot_video.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(LINGBOT_VIDEO_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x_tensor.dim() == 5);
            GGML_ASSERT(x->ne[3] == config.in_channels);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t N = x_tensor.shape()[4];
            GGML_ASSERT(N == 1);
            pe_vec      = Rope::gen_lingbot_video_pe(static_cast<int>(T),
                                                     static_cast<int>(H),
                                                     static_cast<int>(W),
                                                     config.patch_t,
                                                     config.patch_h,
                                                     config.patch_w,
                                                     static_cast<int>(N),
                                                     static_cast<int>(context->ne[1]),
                                                     config.rope_theta,
                                                     config.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = lingbot_video.forward(&runner_ctx, x, timesteps, context, pe);
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
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
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
}  // namespace LingBotVideo

#endif  // __SD_MODEL_DIFFUSION_LINGBOT_VIDEO_HPP__

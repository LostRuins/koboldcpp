#ifndef __LLM_HPP__
#define __LLM_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "json.hpp"
#include "rope.hpp"
#include "tokenizers/bpe_tokenizer.h"
#include "tokenizers/mistral_tokenizer.h"
#include "tokenizers/qwen2_tokenizer.h"

namespace LLM {
    constexpr int LLM_GRAPH_SIZE = 10240;

    enum class LLMArch {
        QWEN2_5_VL,
        QWEN3,
        QWEN3_VL,
        MISTRAL_SMALL_3_2,
        MINISTRAL_3_3B,
        ARCH_COUNT,
    };

    static const char* llm_arch_to_str[] = {
        "qwen2.5vl",
        "qwen3",
        "qwen3vl",
        "mistral_small3.2",
        "ministral3.3b",
    };

    enum class LLMVisionArch {
        QWEN2_5_VL,
        QWEN3_VL,
    };

    struct LLMVisionParams {
        LLMVisionArch arch                  = LLMVisionArch::QWEN2_5_VL;
        int num_layers                      = 32;
        int64_t hidden_size                 = 1280;
        int64_t intermediate_size           = 3420;
        int num_heads                       = 16;
        int64_t in_channels                 = 3;
        int64_t out_hidden_size             = 3584;
        int temporal_patch_size             = 2;
        int patch_size                      = 14;
        int spatial_merge_size              = 2;
        int window_size                     = 112;
        int num_position_embeddings         = 0;
        std::set<int> fullatt_block_indexes = {7, 15, 23, 31};
    };

    struct LLMParams {
        LLMArch arch              = LLMArch::QWEN2_5_VL;
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int num_heads             = 28;
        int num_kv_heads          = 4;
        int head_dim              = 128;
        bool qkv_bias             = true;
        bool qk_norm              = false;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
        LLMVisionParams vision;
    };

    struct MLP : public GGMLBlock {
    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, bias));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    static ggml_tensor* splice_image_embeds(GGMLRunnerContext* ctx,
                                            ggml_tensor* x,
                                            const std::vector<std::pair<int, ggml_tensor*>>& image_embeds) {
        if (image_embeds.empty()) {
            return x;
        }

        GGML_ASSERT(x->ne[2] == 1);  // N == 1

        auto raw_x               = ggml_cast(ctx->ggml_ctx, x, image_embeds[0].second->type);
        int64_t txt_token_start  = 0;
        int64_t txt_token_end    = 0;
        ggml_tensor* input_embed = nullptr;

        for (int i = 0; i < image_embeds.size(); i++) {
            if (i == 0) {
                txt_token_start = 0;
            } else {
                txt_token_start = image_embeds[i - 1].first + image_embeds[i - 1].second->ne[1];
            }
            txt_token_end = image_embeds[i].first;

            auto txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
            if (input_embed == nullptr) {
                input_embed = txt_embed;
            } else {
                input_embed = ggml_concat(ctx->ggml_ctx, input_embed, txt_embed, 1);
            }

            input_embed = ggml_concat(ctx->ggml_ctx, input_embed, image_embeds[i].second, 1);
        }

        txt_token_start = image_embeds[image_embeds.size() - 1].first + image_embeds[image_embeds.size() - 1].second->ne[1];
        txt_token_end   = raw_x->ne[1];

        auto final_txt_embed = ggml_ext_slice(ctx->ggml_ctx, raw_x, 1, txt_token_start, txt_token_end);
        input_embed          = ggml_concat(ctx->ggml_ctx, input_embed, final_txt_embed, 1);
        GGML_ASSERT(raw_x->ne[1] == input_embed->ne[1]);
        return input_embed;
    }

    struct VisionMLP : public GGMLBlock {
    protected:
        LLMVisionArch arch_;

    public:
        VisionMLP(LLMVisionArch arch, int64_t hidden_size, int64_t intermediate_size)
            : arch_(arch) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["linear_fc2"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
            } else {
                blocks["gate_proj"] = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["up_proj"]   = std::make_shared<Linear>(hidden_size, intermediate_size, true);
                blocks["down_proj"] = std::make_shared<Linear>(intermediate_size, hidden_size, true);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
                auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);
                x               = linear_fc1->forward(ctx, x);
                x               = ggml_ext_gelu(ctx->ggml_ctx, x);
                x               = linear_fc2->forward(ctx, x);
            } else {
                auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
                auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
                auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);
                auto h         = gate_proj->forward(ctx, x);
                h              = ggml_silu_inplace(ctx->ggml_ctx, h);
                h              = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
                x              = down_proj->forward(ctx, h);
            }
            return x;
        }
    };

    struct VisionPatchEmbed : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int patch_size;
        int temporal_patch_size;
        int64_t in_channels;
        int64_t embed_dim;

    public:
        VisionPatchEmbed(bool llama_cpp_style,
                         LLMVisionArch arch,
                         int patch_size          = 14,
                         int temporal_patch_size = 2,
                         int64_t in_channels     = 3,
                         int64_t embed_dim       = 1152)
            : llama_cpp_style(llama_cpp_style),
              patch_size(patch_size),
              temporal_patch_size(temporal_patch_size),
              in_channels(in_channels),
              embed_dim(embed_dim) {
            bool bias = arch == LLMVisionArch::QWEN3_VL;
            if (llama_cpp_style) {
                blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},
                                                                         {0, 0},
                                                                         {1, 1},
                                                                         bias));
                blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels,
                                                                         embed_dim,
                                                                         {patch_size, patch_size},
                                                                         {patch_size, patch_size},
                                                                         {0, 0},
                                                                         {1, 1},
                                                                         bias));
            } else {
                std::tuple<int, int, int> kernel_size = {(int)temporal_patch_size, (int)patch_size, (int)patch_size};
                blocks["proj"]                        = std::shared_ptr<GGMLBlock>(new Conv3d(in_channels,
                                                                                              embed_dim,
                                                                                              kernel_size,
                                                                                              kernel_size,
                                                                                              {0, 0, 0},
                                                                                              {1, 1, 1},
                                                                                              bias));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = ggml_reshape_4d(ctx->ggml_ctx,
                                x,
                                patch_size,
                                patch_size,
                                temporal_patch_size,
                                ggml_nelements(x) / (temporal_patch_size * patch_size * patch_size));

            if (llama_cpp_style) {
                auto proj_0 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.0"]);
                auto proj_1 = std::dynamic_pointer_cast<Conv2d>(blocks["proj.1"]);

                auto x0 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x0      = ggml_reshape_4d(ctx->ggml_ctx, x0, x0->ne[0], x0->ne[1], in_channels, x0->ne[3] / in_channels);
                x0      = proj_0->forward(ctx, x0);

                auto x1 = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, 2);
                x1      = ggml_reshape_4d(ctx->ggml_ctx, x1, x1->ne[0], x1->ne[1], in_channels, x1->ne[3] / in_channels);
                x1      = proj_1->forward(ctx, x1);

                x = ggml_add(ctx->ggml_ctx, x0, x1);
            } else {
                auto proj = std::dynamic_pointer_cast<Conv3d>(blocks["proj"]);

                x = proj->forward(ctx, x);
            }

            x = ggml_reshape_2d(ctx->ggml_ctx, x, embed_dim, ggml_nelements(x) / embed_dim);
            return x;
        }
    };

    struct VisionPatchMerger : public GGMLBlock {
    protected:
        LLMVisionArch arch_;
        int64_t hidden_size;

    public:
        VisionPatchMerger(LLMVisionArch arch,
                          int64_t dim,
                          int64_t context_dim,
                          int64_t spatial_merge_size)
            : arch_(arch),
              hidden_size(context_dim * spatial_merge_size * spatial_merge_size) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["norm"]       = std::make_shared<LayerNorm>(context_dim, 1e-6f);
                blocks["linear_fc1"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
                blocks["linear_fc2"] = std::make_shared<Linear>(hidden_size, dim, true);
            } else {
                blocks["ln_q"]  = std::make_shared<RMSNorm>(context_dim, 1e-6f);
                blocks["mlp.0"] = std::make_shared<Linear>(hidden_size, hidden_size);
                blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                auto norm       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
                auto linear_fc1 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc1"]);
                auto linear_fc2 = std::dynamic_pointer_cast<Linear>(blocks["linear_fc2"]);

                x = norm->forward(ctx, x);
                x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
                x = linear_fc1->forward(ctx, x);
                x = ggml_gelu_erf(ctx->ggml_ctx, x);
                x = linear_fc2->forward(ctx, x);
                return x;
            }

            auto ln_q  = std::dynamic_pointer_cast<RMSNorm>(blocks["ln_q"]);
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            x = ln_q->forward(ctx, x);
            x = ggml_reshape_2d(ctx->ggml_ctx, x, hidden_size, ggml_nelements(x) / hidden_size);
            x = mlp_0->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct VisionAttention : public GGMLBlock {
    protected:
        bool llama_cpp_style;
        int head_dim;
        int num_heads;

    public:
        VisionAttention(bool llama_cpp_style,
                        int64_t hidden_size,
                        int num_heads)
            : llama_cpp_style(llama_cpp_style), num_heads(num_heads) {
            head_dim = static_cast<int>(hidden_size / num_heads);
            GGML_ASSERT(num_heads * head_dim == hidden_size);
            if (llama_cpp_style) {
                blocks["q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
                blocks["v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
            } else {
                blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3));
            }
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto proj       = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            std::vector<ggml_tensor*> qkv_vec;
            if (llama_cpp_style) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);

                auto q = q_proj->forward(ctx, x);
                auto k = k_proj->forward(ctx, x);
                auto v = v_proj->forward(ctx, x);

                qkv_vec = {q, k, v};
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                auto qkv      = qkv_proj->forward(ctx, x);
                qkv_vec       = split_qkv(ctx->ggml_ctx, qkv);
            }

            auto q = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
            auto k = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
            auto v = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f, false);  // [N, n_token, hidden_size]

            x = proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct VisionBlock : public GGMLBlock {
    protected:
        LLMVisionArch arch_;

        ggml_tensor* forward_norm(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);
                return norm->forward(ctx, x);
            }
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks[name]);
            return norm->forward(ctx, x);
        }

    public:
        VisionBlock(bool llama_cpp_style,
                    LLMVisionArch arch,
                    int64_t hidden_size,
                    int64_t intermediate_size,
                    int num_heads,
                    float eps = 1e-6f)
            : arch_(arch) {
            blocks["attn"] = std::shared_ptr<GGMLBlock>(new VisionAttention(llama_cpp_style, hidden_size, num_heads));
            blocks["mlp"]  = std::shared_ptr<GGMLBlock>(new VisionMLP(arch_, hidden_size, intermediate_size));
            if (arch_ == LLMVisionArch::QWEN3_VL) {
                blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps));
                blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, eps));
            } else {
                blocks["norm1"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
                blocks["norm2"] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_size, eps));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto attn = std::dynamic_pointer_cast<VisionAttention>(blocks["attn"]);
            auto mlp  = std::dynamic_pointer_cast<VisionMLP>(blocks["mlp"]);

            auto residual = x;
            x             = forward_norm(ctx, "norm1", x);
            x             = attn->forward(ctx, x, pe, mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = forward_norm(ctx, "norm2", x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct VisionModel : public GGMLBlock {
    protected:
        LLMVisionArch arch_;
        int num_layers;
        int spatial_merge_size;
        int num_grid_per_side;
        std::set<int> fullatt_block_indexes;

    public:
        VisionModel(bool llama_cpp_style,
                    const LLMVisionParams& vision_params,
                    float eps = 1e-6f)
            : arch_(vision_params.arch),
              num_layers(vision_params.num_layers),
              spatial_merge_size(vision_params.spatial_merge_size),
              num_grid_per_side(vision_params.num_position_embeddings > 0 ? static_cast<int>(std::sqrt(vision_params.num_position_embeddings)) : 0),
              fullatt_block_indexes(vision_params.fullatt_block_indexes) {
            blocks["patch_embed"] = std::shared_ptr<GGMLBlock>(new VisionPatchEmbed(llama_cpp_style,
                                                                                    arch_,
                                                                                    vision_params.patch_size,
                                                                                    vision_params.temporal_patch_size,
                                                                                    vision_params.in_channels,
                                                                                    vision_params.hidden_size));
            if (vision_params.num_position_embeddings > 0) {
                blocks["pos_embed"] = std::make_shared<Embedding>(vision_params.num_position_embeddings, vision_params.hidden_size);
            }
            for (int i = 0; i < num_layers; i++) {
                blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new VisionBlock(llama_cpp_style,
                                                                                                   arch_,
                                                                                                   vision_params.hidden_size,
                                                                                                   vision_params.intermediate_size,
                                                                                                   vision_params.num_heads,
                                                                                                   eps));
            }
            blocks["merger"] = std::shared_ptr<GGMLBlock>(new VisionPatchMerger(arch_,
                                                                                vision_params.out_hidden_size,
                                                                                vision_params.hidden_size,
                                                                                spatial_merge_size));
        }

        std::shared_ptr<Embedding> pos_embedder() {
            auto it = blocks.find("pos_embed");
            if (it == blocks.end()) {
                return nullptr;
            }
            return std::dynamic_pointer_cast<Embedding>(it->second);
        }

        int get_num_grid_per_side() const {
            return num_grid_per_side;
        }

        int get_spatial_merge_size() const {
            return spatial_merge_size;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             ggml_tensor* pe,
                             ggml_tensor* window_index,
                             ggml_tensor* window_inverse_index,
                             ggml_tensor* window_mask,
                             ggml_tensor* pos_embeds = nullptr) {
            // pixel_values: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            // window_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_inverse_index: [grid_t*(H/mh/ph)*(W/mw/pw)]
            // window_mask: [grid_h*grid_w, grid_h*grid_w]
            auto patch_embed = std::dynamic_pointer_cast<VisionPatchEmbed>(blocks["patch_embed"]);
            auto merger      = std::dynamic_pointer_cast<VisionPatchMerger>(blocks["merger"]);

            auto x = patch_embed->forward(ctx, pixel_values);
            sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.prelude", "x");
            if (pos_embeds != nullptr) {
                x = ggml_add(ctx->ggml_ctx, x, pos_embeds);
            }

            if (window_index != nullptr) {
                x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] * spatial_merge_size * spatial_merge_size, x->ne[1] / spatial_merge_size / spatial_merge_size, x->ne[2], x->ne[3]);
                x = ggml_get_rows(ctx->ggml_ctx, x, window_index);
                x = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0] / spatial_merge_size / spatial_merge_size, x->ne[1] * spatial_merge_size * spatial_merge_size, x->ne[2], x->ne[3]);
            }

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<VisionBlock>(blocks["blocks." + std::to_string(i)]);

                auto mask = window_mask;
                if (fullatt_block_indexes.find(i) != fullatt_block_indexes.end()) {
                    mask = nullptr;
                }
                x = block->forward(ctx, x, pe, mask);
                if (i == 0) {
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.blocks." + std::to_string(i), "x");
            }

            x = merger->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "llm.vision.final", "x");

            if (window_inverse_index != nullptr) {
                x = ggml_get_rows(ctx->ggml_ctx, x, window_inverse_index);
            }

            return x;
        }
    };

    struct Attention : public GGMLBlock {
    protected:
        LLMArch arch;
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        Attention(const LLMParams& params)
            : arch(params.arch), num_heads(params.num_heads), num_kv_heads(params.num_kv_heads), head_dim(params.head_dim), qk_norm(params.qk_norm) {
            blocks["q_proj"] = std::make_shared<Linear>(params.hidden_size, num_heads * head_dim, params.qkv_bias);
            blocks["k_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["v_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, params.hidden_size, false);
            if (params.qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);

            auto q = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
            auto k = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            auto v = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            if (arch == LLMArch::MISTRAL_SMALL_3_2) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NORMAL, 8192, 1000000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::MINISTRAL_3_3B) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 262144, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3) {
                q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else if (arch == LLMArch::QWEN3_VL) {
                int sections[4] = {24, 20, 20, 0};
                q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_IMROPE, 262144, 5000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_IMROPE, 262144, 5000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            } else {
                int sections[4] = {16, 24, 24, 0};
                q               = ggml_rope_multi(ctx->ggml_ctx, q, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
                k               = ggml_rope_multi(ctx->ggml_ctx, k, input_pos, nullptr, head_dim, sections, GGML_ROPE_TYPE_MROPE, 128000, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            }

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, false);  // [N, n_token, hidden_size]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    public:
        TransformerBlock(const LLMParams& params) {
            blocks["self_attn"]                = std::make_shared<Attention>(params);
            blocks["mlp"]                      = std::make_shared<MLP>(params.hidden_size, params.intermediate_size);
            blocks["input_layernorm"]          = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
            blocks["post_attention_layernorm"] = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos, attention_mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;
        LLMParams params;

    public:
        TextModel(const LLMParams& params)
            : num_layers(params.num_layers), params(params) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size, params.hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(params));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.hidden_size, params.rms_norm_eps));
        }

        ggml_tensor* embed(GGMLRunnerContext* ctx,
                           ggml_tensor* input_ids) {
            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto x            = embed_tokens->forward(ctx, input_ids);
            return x;
        }

        ggml_tensor* forward_embeds(GGMLRunnerContext* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* input_pos,
                                    ggml_tensor* attention_mask,
                                    std::set<int> out_layers) {
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            std::vector<ggml_tensor*> intermediate_outputs;

            sd::ggml_graph_cut::mark_graph_cut(x, "llm.text.prelude", "x");
            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);

                x = block->forward(ctx, x, input_pos, attention_mask);
                if (out_layers.size() > 1) {
                    x = ggml_cont(ctx->ggml_ctx, x);
                }
                sd::ggml_graph_cut::mark_graph_cut(x, "llm.text.layers." + std::to_string(i), "x");
                if (out_layers.find(i + 1) != out_layers.end()) {
                    intermediate_outputs.push_back(x);
                }
            }

            if (!intermediate_outputs.empty()) {
                x = intermediate_outputs[0];
                for (int i = 1; i < intermediate_outputs.size(); i++) {
                    x = ggml_concat(ctx->ggml_ctx, x, intermediate_outputs[i], 0);
                }
                return x;
            }

            return norm->forward(ctx, x);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]
            auto x = embed(ctx, input_ids);
            x      = splice_image_embeds(ctx, x, image_embeds);
            return forward_embeds(ctx, x, input_pos, attention_mask, std::move(out_layers));
        }
    };

    struct LLM : public GGMLBlock {
        bool enable_vision;
        LLMParams params;

    public:
        LLM() = default;
        LLM(LLMParams params, bool enable_vision = false, bool llama_cpp_style = false)
            : enable_vision(enable_vision), params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new TextModel(params));
            if (enable_vision) {
                blocks["visual"] = std::shared_ptr<GGMLBlock>(new VisionModel(llama_cpp_style, params.vision));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers) {
            // input_ids: [N, n_token]
            auto model = std::dynamic_pointer_cast<TextModel>(blocks["model"]);

            auto x = model->forward(ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers);
            return x;
        }

        std::shared_ptr<VisionModel> vision_model() {
            GGML_ASSERT(enable_vision);
            return std::dynamic_pointer_cast<VisionModel>(blocks["visual"]);
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* pe,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask,
                                    ggml_tensor* pos_embeds = nullptr) {
            GGML_ASSERT(enable_vision);
            return vision_model()->forward(ctx, pixel_values, pe, window_index, window_inverse_index, window_mask, pos_embeds);
        }
    };

    struct LLMRunner : public GGMLRunner {
        LLMParams params;
        bool enable_vision;
        LLM model;

        std::vector<int> input_pos_vec;
        std::vector<float> attention_mask_vec;
        std::vector<float> window_mask_vec;
        std::vector<int> window_index_vec;
        std::vector<int> window_inverse_index_vec;
        std::vector<float> pe_vec;
        std::array<std::vector<int32_t>, 4> pos_embed_idx_data_;
        std::array<std::vector<float>, 4> pos_embed_weight_data_;

        static ggml_tensor* process_image_common(ggml_context* ctx,
                                                 ggml_tensor* image,
                                                 const LLMVisionParams& vision_params) {
            // image: [C, H, W]
            // return: [grid_t*(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw], grid_t == 1
            int64_t C  = image->ne[2];
            int64_t H  = image->ne[1];
            int64_t W  = image->ne[0];
            int64_t mh = vision_params.spatial_merge_size;
            int64_t mw = vision_params.spatial_merge_size;
            int64_t pt = vision_params.temporal_patch_size;
            int64_t ph = vision_params.patch_size;
            int64_t pw = vision_params.patch_size;

            image = ggml_reshape_4d(ctx, image, pw, mw, (W / mw / pw), H * C);                               // [C*H, (W/mw/pw), mw, pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [mw, C*H, (W/mw/pw), pw]
            image = ggml_reshape_4d(ctx, image, pw * (W / mw / pw), H, C, mw);                               // [mw, C, H, (W/mw/pw)*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 3, 1));                          // [H, mw, C, (W/mw/pw)*pw]
            image = ggml_reshape_4d(ctx, image, pw, (W / mw / pw) * C * mw, ph, mh * (H / mh / ph));         // [(H/mh/ph)*mh, ph, mw*C*(W/mw/pw), pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh, mw*C*(W/mw/pw), ph, pw]
            image = ggml_reshape_4d(ctx, image, pw * ph, (W / mw / pw), C, mw * mh * (H / mh / ph));         // [(H/mh/ph)*mh*mw, C, (W/mw/pw), ph*pw]
            image = ggml_concat(ctx, image, image, 0);                                                       // [(H/mh/ph)*mh*mw, C, (W/mw/pw), pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph)*mh*mw, (W/mw/pw), C, pt*ph*pw]
            image = ggml_reshape_4d(ctx, image, pw * ph * pt * C, (W / mw / pw), mw * mh, (H / mh / ph));    // [(H/mh/ph), mh*mw, (W/mw/pw), C*pt*ph*pw]
            image = ggml_cont(ctx, ggml_ext_torch_permute(ctx, image, 0, 2, 1, 3));                          // [(H/mh/ph), (W/mw/pw), mh*mw, C*pt*ph*pw]
            image = ggml_reshape_2d(ctx, image, pw * ph * pt * C, mw * mh * (W / mw / pw) * (H / mh / ph));  // [(H/mh/ph)*(W/mw/pw)*mh*mw, C*pt*ph*pw]
            return image;
        }

        static ggml_tensor* build_patch_pos_embeds_common(GGMLRunner* runner,
                                                          ggml_context* compute_ctx,
                                                          GGMLRunnerContext* runner_ctx,
                                                          std::shared_ptr<VisionModel> vision,
                                                          int grid_h,
                                                          int grid_w,
                                                          std::array<std::vector<int32_t>, 4>& pos_embed_idx_data,
                                                          std::array<std::vector<float>, 4>& pos_embed_weight_data) {
            auto pos_embed = vision->pos_embedder();
            GGML_ASSERT(pos_embed != nullptr);
            for (int i = 0; i < 4; ++i) {
                pos_embed_idx_data[i].clear();
                pos_embed_weight_data[i].clear();
                pos_embed_idx_data[i].reserve(static_cast<size_t>(grid_h * grid_w));
                pos_embed_weight_data[i].reserve(static_cast<size_t>(grid_h * grid_w));
            }

            int num_grid_per_side = vision->get_num_grid_per_side();
            double max_index      = static_cast<double>(num_grid_per_side - 1);
            int merge_size        = vision->get_spatial_merge_size();
            GGML_ASSERT(grid_h % merge_size == 0);
            GGML_ASSERT(grid_w % merge_size == 0);
            for (int bh = 0; bh < grid_h / merge_size; ++bh) {
                for (int bw = 0; bw < grid_w / merge_size; ++bw) {
                    for (int ih = 0; ih < merge_size; ++ih) {
                        int h        = bh * merge_size + ih;
                        double h_pos = grid_h == 1 ? 0.0 : max_index * h / static_cast<double>(grid_h - 1);
                        int h_floor  = static_cast<int>(std::floor(h_pos));
                        int h_ceil   = std::min(h_floor + 1, num_grid_per_side - 1);
                        double dh    = h_pos - h_floor;
                        for (int iw = 0; iw < merge_size; ++iw) {
                            int w        = bw * merge_size + iw;
                            double w_pos = grid_w == 1 ? 0.0 : max_index * w / static_cast<double>(grid_w - 1);
                            int w_floor  = static_cast<int>(std::floor(w_pos));
                            int w_ceil   = std::min(w_floor + 1, num_grid_per_side - 1);
                            double dw    = w_pos - w_floor;

                            pos_embed_idx_data[0].push_back(h_floor * num_grid_per_side + w_floor);
                            pos_embed_idx_data[1].push_back(h_floor * num_grid_per_side + w_ceil);
                            pos_embed_idx_data[2].push_back(h_ceil * num_grid_per_side + w_floor);
                            pos_embed_idx_data[3].push_back(h_ceil * num_grid_per_side + w_ceil);

                            pos_embed_weight_data[0].push_back(static_cast<float>((1.0 - dh) * (1.0 - dw)));
                            pos_embed_weight_data[1].push_back(static_cast<float>((1.0 - dh) * dw));
                            pos_embed_weight_data[2].push_back(static_cast<float>(dh * (1.0 - dw)));
                            pos_embed_weight_data[3].push_back(static_cast<float>(dh * dw));
                        }
                    }
                }
            }

            ggml_tensor* patch_pos_embeds = nullptr;
            for (int i = 0; i < 4; ++i) {
                auto idx_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(pos_embed_idx_data[i].size()));
                runner->set_backend_tensor_data(idx_tensor, pos_embed_idx_data[i].data());
                auto embed         = pos_embed->forward(runner_ctx, idx_tensor);
                auto weight_tensor = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 1, static_cast<int64_t>(pos_embed_weight_data[i].size()));
                runner->set_backend_tensor_data(weight_tensor, pos_embed_weight_data[i].data());
                embed            = ggml_mul(compute_ctx, embed, weight_tensor);
                patch_pos_embeds = patch_pos_embeds == nullptr ? embed : ggml_add(compute_ctx, patch_pos_embeds, embed);
            }
            return patch_pos_embeds;
        }

        static ggml_tensor* encode_image_common(GGMLRunner* runner,
                                                ggml_context* compute_ctx,
                                                GGMLRunnerContext* runner_ctx,
                                                ggml_tensor* image,
                                                const LLMVisionParams& vision_params,
                                                std::shared_ptr<VisionModel> vision_model,
                                                std::vector<int>& window_index_vec,
                                                std::vector<int>& window_inverse_index_vec,
                                                std::vector<float>& window_mask_vec,
                                                std::vector<float>& pe_vec,
                                                std::array<std::vector<int32_t>, 4>& pos_embed_idx_data,
                                                std::array<std::vector<float>, 4>& pos_embed_weight_data) {
            GGML_ASSERT(image->ne[1] % (vision_params.patch_size * vision_params.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (vision_params.patch_size * vision_params.spatial_merge_size) == 0);

            int grid_h = static_cast<int>(image->ne[1]) / vision_params.patch_size;
            int grid_w = static_cast<int>(image->ne[0]) / vision_params.patch_size;

            auto pixel_values = process_image_common(compute_ctx, image, vision_params);
            int head_dim      = static_cast<int>(vision_params.hidden_size / vision_params.num_heads);

            if (vision_params.arch == LLMVisionArch::QWEN3_VL) {
                auto pos_embeds = build_patch_pos_embeds_common(runner,
                                                                compute_ctx,
                                                                runner_ctx,
                                                                vision_model,
                                                                grid_h,
                                                                grid_w,
                                                                pos_embed_idx_data,
                                                                pos_embed_weight_data);
                window_index_vec.resize(static_cast<size_t>((grid_h / vision_params.spatial_merge_size) * (grid_w / vision_params.spatial_merge_size)));
                for (int i = 0; i < static_cast<int>(window_index_vec.size()); ++i) {
                    window_index_vec[static_cast<size_t>(i)] = i;
                }
                pe_vec      = Rope::gen_qwen2vl_pe(grid_h,
                                                   grid_w,
                                                   vision_params.spatial_merge_size,
                                                   window_index_vec,
                                                   10000,
                                                   {head_dim / 2, head_dim / 2});
                int pos_len = static_cast<int>(pe_vec.size() / head_dim / 2);
                auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
                runner->set_backend_tensor_data(pe, pe_vec.data());
                return vision_model->forward(runner_ctx, pixel_values, pe, nullptr, nullptr, nullptr, pos_embeds);
            }

            int llm_grid_h             = grid_h / vision_params.spatial_merge_size;
            int llm_grid_w             = grid_w / vision_params.spatial_merge_size;
            int vit_merger_window_size = vision_params.window_size / vision_params.patch_size / vision_params.spatial_merge_size;

            int inverse_index = 0;
            window_index_vec.resize(llm_grid_h * llm_grid_w);
            window_inverse_index_vec.resize(llm_grid_h * llm_grid_w);
            std::vector<int> seqlens;
            for (int ih = 0; ih < llm_grid_h; ih += vit_merger_window_size) {
                for (int iw = 0; iw < llm_grid_w; iw += vit_merger_window_size) {
                    int win_h = std::min(vit_merger_window_size, llm_grid_h - ih);
                    int win_w = std::min(vit_merger_window_size, llm_grid_w - iw);
                    for (int iy = 0; iy < win_h; iy++) {
                        for (int ix = 0; ix < win_w; ix++) {
                            int index                       = (ih + iy) * llm_grid_w + iw + ix;
                            window_index_vec[inverse_index] = index;
                            window_inverse_index_vec[index] = inverse_index;
                            inverse_index++;
                        }
                    }
                    seqlens.push_back(win_h * win_w * vision_params.spatial_merge_size * vision_params.spatial_merge_size);
                }
            }
            auto window_index         = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, llm_grid_h * llm_grid_w);
            auto window_inverse_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, llm_grid_h * llm_grid_w);
            runner->set_backend_tensor_data(window_index, window_index_vec.data());
            runner->set_backend_tensor_data(window_inverse_index, window_inverse_index_vec.data());

            window_mask_vec.resize((grid_h * grid_w) * (grid_h * grid_w));
            int window_start_index = 0;
            for (int seq_index = 0; seq_index < seqlens.size(); seq_index++) {
                int window_end_index = window_start_index + seqlens[seq_index];
                GGML_ASSERT(window_end_index <= grid_h * grid_w);
                for (int i = window_start_index; i < window_end_index; i++) {
                    for (int j = 0; j < grid_h * grid_w; j++) {
                        float mask_value = -INFINITY;
                        if (j >= window_start_index && j < window_end_index) {
                            mask_value = 0;
                        }
                        GGML_ASSERT((i * (grid_h * grid_w) + j) < window_mask_vec.size());
                        window_mask_vec[i * (grid_h * grid_w) + j] = mask_value;
                    }
                }
                window_start_index = window_end_index;
            }

            auto window_mask = ggml_new_tensor_2d(compute_ctx,
                                                  GGML_TYPE_F32,
                                                  grid_h * grid_w,
                                                  grid_h * grid_w);
            runner->set_backend_tensor_data(window_mask, window_mask_vec.data());

            pe_vec      = Rope::gen_qwen2vl_pe(grid_h,
                                               grid_w,
                                               vision_params.spatial_merge_size,
                                               window_inverse_index_vec,
                                               10000,
                                               {head_dim / 2, head_dim / 2});
            int pos_len = static_cast<int>(pe_vec.size() / head_dim / 2);

            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
            runner->set_backend_tensor_data(pe, pe_vec.data());

            return vision_model->forward(runner_ctx, pixel_values, pe, window_index, window_inverse_index, window_mask);
        }

    public:
        LLMRunner(LLMArch arch,
                  ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix,
                  bool enable_vision_ = false)
            : GGMLRunner(backend, offload_params_to_cpu), enable_vision(enable_vision_) {
            params.arch = arch;
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.rms_norm_eps = 1e-5f;
            } else if (arch == LLMArch::QWEN3) {
                params.head_dim     = 128;
                params.num_heads    = 32;
                params.num_kv_heads = 8;
                params.qkv_bias     = false;
                params.qk_norm      = true;
                params.rms_norm_eps = 1e-6f;
            }
            bool have_vision_weight = false;
            bool llama_cpp_style    = false;
            params.num_layers       = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("visual.");
                if (pos != std::string::npos) {
                    have_vision_weight = true;
                    if (contains(tensor_name, "attn.q_proj")) {
                        llama_cpp_style = true;
                    }
                    continue;
                }
                pos = tensor_name.find("layers.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > params.num_layers) {
                            params.num_layers = block_index + 1;
                        }
                    }
                }
                if (contains(tensor_name, "embed_tokens.weight")) {
                    params.hidden_size = pair.second.ne[0];
                    params.vocab_size  = pair.second.ne[1];
                }
                if (contains(tensor_name, "layers.0.mlp.gate_proj.weight")) {
                    params.intermediate_size = pair.second.ne[1];
                }
            }
            if (arch == LLMArch::QWEN3 && params.num_layers == 28) {  // Qwen3 2B
                params.num_heads = 16;
            }
            LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
                      params.num_layers,
                      params.vocab_size,
                      params.hidden_size,
                      params.intermediate_size);
            if (enable_vision && !have_vision_weight) {
                LOG_WARN("no vision weights detected, vision disabled");
                enable_vision = false;
            }
            if (enable_vision) {
                LOG_DEBUG("enable llm vision");
                if (llama_cpp_style) {
                    LOG_DEBUG("llama.cpp style vision weight");
                }
            }
            model = LLM(params, enable_vision, llama_cpp_style);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return llm_arch_to_str[static_cast<int>(params.arch)];
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* input_ids,
                             ggml_tensor* input_pos,
                             ggml_tensor* attention_mask,
                             std::vector<std::pair<int, ggml_tensor*>> image_embeds,
                             std::set<int> out_layers) {
            auto hidden_states = model.forward(ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers);  // [N, n_token, hidden_size]
            return hidden_states;
        }

        ggml_tensor* vision_forward(GGMLRunnerContext* ctx,
                                    ggml_tensor* pixel_values,
                                    ggml_tensor* input_pos,
                                    ggml_tensor* window_index,
                                    ggml_tensor* window_inverse_index,
                                    ggml_tensor* window_mask,
                                    ggml_tensor* pos_embeds = nullptr) {
            auto hidden_states = model.vision_forward(ctx, pixel_values, input_pos, window_index, window_inverse_index, window_mask, pos_embeds);
            return hidden_states;
        }

        ggml_cgraph* build_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                                 const sd::Tensor<float>& attention_mask_tensor,
                                 const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds_tensor,
                                 std::set<int> out_layers) {
            ggml_cgraph* gf        = ggml_new_graph(compute_ctx);
            ggml_tensor* input_ids = make_input(input_ids_tensor);
            std::vector<std::pair<int, ggml_tensor*>> image_embeds;
            image_embeds.reserve(image_embeds_tensor.size());
            for (const auto& [idx, embed_tensor] : image_embeds_tensor) {
                ggml_tensor* embed = make_input(embed_tensor);
                image_embeds.emplace_back(idx, embed);
            }

            int64_t n_tokens = input_ids->ne[0];
            if (params.arch == LLMArch::MISTRAL_SMALL_3_2 || params.arch == LLMArch::MINISTRAL_3_3B || params.arch == LLMArch::QWEN3) {
                input_pos_vec.resize(n_tokens);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i] = i;
                }
            } else {
                input_pos_vec.resize(n_tokens * 4);
                for (int i = 0; i < n_tokens; ++i) {
                    input_pos_vec[i]                = i;
                    input_pos_vec[n_tokens + i]     = i;
                    input_pos_vec[2 * n_tokens + i] = i;
                    input_pos_vec[3 * n_tokens + i] = 0;
                }
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx,
                                                GGML_TYPE_I32,
                                                input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            ggml_tensor* attention_mask = nullptr;
            if (!attention_mask_tensor.empty()) {
                attention_mask = make_input(attention_mask_tensor);
            } else {
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {
                    for (int i1 = 0; i1 < n_tokens; i1++) {
                        float value = 0.f;
                        if (i0 > i1) {
                            value = -INFINITY;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            auto runner_ctx = get_context();

            ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, attention_mask, image_embeds, out_layers);

            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<int32_t>& input_ids,
                                  const sd::Tensor<float>& attention_mask,
                                  const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                  std::set<int> out_layers) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(input_ids, attention_mask, image_embeds, out_layers);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true));
        }

        int64_t get_num_image_tokens(int64_t t, int64_t h, int64_t w) {
            int64_t grid_t     = 1;
            int64_t grid_h     = h / params.vision.patch_size;
            int64_t grid_w     = w / params.vision.patch_size;
            int64_t llm_grid_h = grid_h / params.vision.spatial_merge_size;
            int64_t llm_grid_w = grid_w / params.vision.spatial_merge_size;
            return grid_t * grid_h * grid_w;
        }

        ggml_tensor* process_image(ggml_context* ctx, ggml_tensor* image) {
            return process_image_common(ctx, image, params.vision);
        }

        ggml_tensor* build_patch_pos_embeds(GGMLRunnerContext* runner_ctx,
                                            std::shared_ptr<VisionModel> vision,
                                            int grid_h,
                                            int grid_w) {
            return build_patch_pos_embeds_common(this,
                                                 compute_ctx,
                                                 runner_ctx,
                                                 vision,
                                                 grid_h,
                                                 grid_w,
                                                 pos_embed_idx_data_,
                                                 pos_embed_weight_data_);
        }

        ggml_tensor* encode_image(GGMLRunnerContext* runner_ctx, ggml_tensor* image) {
            return encode_image_common(this,
                                       compute_ctx,
                                       runner_ctx,
                                       image,
                                       params.vision,
                                       model.vision_model(),
                                       window_index_vec,
                                       window_inverse_index_vec,
                                       window_mask_vec,
                                       pe_vec,
                                       pos_embed_idx_data_,
                                       pos_embed_weight_data_);
        }

        ggml_cgraph* build_encode_image_graph(const sd::Tensor<float>& image_tensor) {
            ggml_cgraph* gf    = new_graph_custom(LLM_GRAPH_SIZE);
            ggml_tensor* image = make_input(image_tensor);

            GGML_ASSERT(image->ne[1] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);
            GGML_ASSERT(image->ne[0] % (params.vision.patch_size * params.vision.spatial_merge_size) == 0);

            auto runnter_ctx           = get_context();
            ggml_tensor* hidden_states = encode_image(&runnter_ctx, image);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        sd::Tensor<float> encode_image(const int n_threads,
                                       const sd::Tensor<float>& image) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_encode_image_graph(image);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

    struct LLMEmbedder {
        std::shared_ptr<BPETokenizer> tokenizer;
        LLMRunner model;

        LLMEmbedder(LLMArch arch,
                    ggml_backend_t backend,
                    bool offload_params_to_cpu,
                    const String2TensorStorage& tensor_storage_map = {},
                    const std::string prefix                       = "",
                    bool enable_vision                             = false)
            : model(arch, backend, offload_params_to_cpu, tensor_storage_map, prefix, enable_vision) {
            if (arch == LLMArch::MISTRAL_SMALL_3_2 || arch == LLMArch::MINISTRAL_3_3B) {
                tokenizer = std::make_shared<MistralTokenizer>();
            } else {
                tokenizer = std::make_shared<Qwen2Tokenizer>();
            }
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void alloc_params_buffer() {
            model.alloc_params_buffer();
        }

        std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                                  std::pair<int, int> attn_range,
                                                                  size_t max_length = 0,
                                                                  bool padding      = false) {
            std::vector<std::pair<std::string, float>> parsed_attention;
            parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            {
                std::stringstream ss;
                ss << "[";
                for (const auto& item : parsed_attention) {
                    ss << "['" << item.first << "', " << item.second << "], ";
                }
                ss << "]";
                LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
            }

            std::vector<int> tokens;
            std::vector<float> weights;
            for (const auto& item : parsed_attention) {
                const std::string& curr_text = item.first;
                float curr_weight            = item.second;
                std::vector<int> curr_tokens = tokenizer->tokenize(curr_text, nullptr);
                tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
                weights.insert(weights.end(), curr_tokens.size(), curr_weight);
            }

            tokenizer->pad_tokens(tokens, &weights, nullptr, padding ? max_length : 0, padding ? max_length : 100000000, padding);

            // for (int i = 0; i < tokens.size(); i++) {
            //     std::cout << tokens[i] << ":" << weights[i] << ", ";
            // }
            // std::cout << std::endl;

            return {tokens, weights};
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);
            bool test_mistral          = false;
            bool test_qwen3            = true;
            bool test_vit              = false;
            bool test_decoder_with_vit = false;

            if (test_decoder_with_vit) {
                sd::Tensor<float> image_embed;
                {
                    auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                    print_sd_tensor(image, false, "image");
                    sd::Tensor<float> out;

                    int64_t t0   = ggml_time_ms();
                    auto out_opt = model.encode_image(8, image);
                    int64_t t1   = ggml_time_ms();

                    GGML_ASSERT(!out_opt.empty());
                    out = std::move(out_opt);
                    print_sd_tensor(out, false, "image_embed");
                    image_embed = out;
                    LOG_DEBUG("llm encode_image test done in %lldms", t1 - t0);
                }

                std::string placeholder  = "<|image_pad|>";
                std::string img_prompt   = "Picture 1: <|vision_start|>";  // [24669, 220, 16, 25, 220, 151652]
                int64_t num_image_tokens = image_embed.shape()[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int i = 0; i < num_image_tokens; i++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";

                std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
                image_embeds.emplace_back(64, image_embed);

                std::pair<int, int> prompt_attn_range;
                std::string text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                text += img_prompt;
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "change 'flux.cpp' to 'edit.cpp'";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";

                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), image_embeds, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_vit) {
                // auto image = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 280, 280, 3);
                // ggml_set_f32(image, 0.f);
                auto image = sd::load_tensor_from_file_as_tensor<float>("qwen2vl_normalized.bin");
                print_sd_tensor(image, false, "image");
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.encode_image(8, image);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out, false, "out");

                // auto ref_out = load_tensor_from_file(ctx, "qwen2vl.bin");
                // ggml_ext_tensor_diff(ref_out, out, 0.01f);

                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_mistral) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "[/INST]";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {10, 20, 30});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else if (test_qwen3) {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {35});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            } else {
                std::pair<int, int> prompt_attn_range;
                std::string text        = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
                prompt_attn_range.first = static_cast<int>(text.size());
                text += "a lovely cat";
                prompt_attn_range.second = static_cast<int>(text.size());
                text += "<|im_end|>\n<|im_start|>assistant\n";
                auto tokens_and_weights     = tokenize(text, prompt_attn_range, 0, false);
                std::vector<int>& tokens    = std::get<0>(tokens_and_weights);
                std::vector<float>& weights = std::get<1>(tokens_and_weights);
                for (auto token : tokens) {
                    printf("%d ", token);
                }
                printf("\n");
                auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = model.compute(8, input_ids, sd::Tensor<float>(), {}, {});
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("llm test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cpu f16: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_COUNT;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "text_encoders.llm.")) {
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

            LLMArch arch = LLMArch::QWEN3;

            std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(arch,
                                                                             backend,
                                                                             true,
                                                                             tensor_storage_map,
                                                                             "text_encoders.llm",
                                                                             true);

            llm->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            llm->get_param_tensors(tensors, "text_encoders.llm");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("llm model loaded");
            llm->test();
        }
    };
};  // LLM

#endif  // __LLM_HPP__

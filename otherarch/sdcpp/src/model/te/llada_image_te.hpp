#ifndef __SD_MODEL_TE_LLADA_IMAGE_TE_HPP__
#define __SD_MODEL_TE_LLADA_IMAGE_TE_HPP__

#include <algorithm>
#include <array>
#include <cmath>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "model/common/ggml_block.hpp"
#include "model_loader.h"

// The conditioning components LLaDA-Image puts around its LLaDA2-MoE backbone.
// Ref: LLaDAImageQueryFormerModel / LLaDAImageTextProjectionModel in
// https://github.com/inclusionAI/LLaDA-Image/blob/main/src/models/transformer_llada_image.py
//
// QueryFormer turns the LLaDA token embeddings into 256 learned queries that the pipeline
// appends to the backbone input; TextProjection maps the backbone hidden states to the
// denoiser's caption dimension. Neither uses RoPE, and every norm is parameter-free.
// Both MLPs use the tanh GELU approximation, so ggml_gelu (not ggml_gelu_erf).
//
// SigVQ is the editing-only image encoder: a 40-layer ViT whose output is quantized against a
// 16384-entry codebook, with the resulting ids embedded and projected into the semantic features
// the denoiser consumes. Its MLP uses the exact erf GELU, unlike the two above.

namespace LLaDAImageTE {
    constexpr int LLADA_IMAGE_TE_GRAPH_SIZE = 16384;

    struct QueryFormerConfig {
        int64_t num_queries       = 256;
        int64_t hidden_size       = 2048;
        int64_t num_layers        = 1;
        int64_t num_heads         = 16;
        int64_t intermediate_size = 8192;
        float norm_eps            = 1e-6f;
    };

    struct TextProjectionConfig {
        int64_t hidden_size       = 2048;
        int64_t intermediate_size = 8960;
        int64_t num_layers        = 6;
        int64_t num_heads         = 32;
        int64_t projection_dim    = 2560;
        float norm_eps            = 1e-6f;
    };

    // Cross-attention with a single fused in_proj over q (from the queries) and k/v (from the
    // token embeddings). The checkpoint stores in_proj as one [3*hidden, hidden] parameter.
    struct QueryAttention : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t num_heads;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         std::string prefix                             = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            enum ggml_type wtype     = get_type(prefix + "in_proj_weight", tensor_storage_map, GGML_TYPE_F32);
            params["in_proj_weight"] = ggml_new_tensor_2d(ctx, wtype, hidden_size, hidden_size * 3);
            params["in_proj_bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size * 3);
        }

    public:
        QueryAttention(int64_t hidden_size, int64_t num_heads)
            : hidden_size(hidden_size), num_heads(num_heads) {
            blocks["out_proj"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* query,
                             ggml_tensor* context,
                             ggml_tensor* mask = nullptr) {
            // query: [N, num_queries, hidden_size], context: [N, n_token, hidden_size]
            ggml_context* gctx = ctx->ggml_ctx;
            auto out_proj      = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            auto w = params["in_proj_weight"];
            auto b = params["in_proj_bias"];

            auto slice_w = [&](int64_t index) {
                return ggml_ext_slice(gctx, w, 1, index * hidden_size, (index + 1) * hidden_size);
            };
            auto slice_b = [&](int64_t index) {
                return ggml_ext_slice(gctx, b, 0, index * hidden_size, (index + 1) * hidden_size);
            };

            auto q = ggml_ext_linear(gctx, query, slice_w(0), slice_b(0));
            auto k = ggml_ext_linear(gctx, context, slice_w(1), slice_b(1));
            auto v = ggml_ext_linear(gctx, context, slice_w(2), slice_b(2));

            auto x = ggml_ext_attention_ext(ctx, q, k, v, num_heads, mask);  // [N, num_queries, hidden_size]
            return out_proj->forward(ctx, x);
        }
    };

    struct QueryFormerBlock : public GGMLBlock {
    protected:
        QueryFormerConfig config;

    public:
        QueryFormerBlock(const QueryFormerConfig& config)
            : config(config) {
            blocks["norm_q"]     = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps, false);
            blocks["norm_k"]     = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps, false);
            blocks["cross_attn"] = std::make_shared<QueryAttention>(config.hidden_size, config.num_heads);
            blocks["norm1"]      = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps, false);
            blocks["mlp.fc1"]    = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, true);
            blocks["mlp.fc2"]    = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* query,
                             ggml_tensor* context,
                             ggml_tensor* mask = nullptr) {
            auto norm_q     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_q"]);
            auto norm_k     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_k"]);
            auto cross_attn = std::dynamic_pointer_cast<QueryAttention>(blocks["cross_attn"]);
            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto fc1        = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"]);
            auto fc2        = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"]);

            // The reference overwrites query_embeds with its normalized value before the
            // residual add, so both residuals here are on normalized activations.
            query      = norm_q->forward(ctx, query);
            auto ctx_n = norm_k->forward(ctx, context);
            query      = ggml_add(ctx->ggml_ctx, query, cross_attn->forward(ctx, query, ctx_n, mask));
            query      = norm1->forward(ctx, query);

            auto h = fc1->forward(ctx, query);
            h      = ggml_gelu(ctx->ggml_ctx, h);
            h      = fc2->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, query, h);
        }
    };

    struct QueryFormerModel : public GGMLBlock {
    protected:
        QueryFormerConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["meta_queries"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, config.num_queries);
        }

    public:
        QueryFormerModel() = default;
        QueryFormerModel(const QueryFormerConfig& config)
            : config(config) {
            for (int i = 0; i < config.num_layers; i++) {
                blocks["query_blocks." + std::to_string(i)] = std::make_shared<QueryFormerBlock>(config);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* inputs_embeds,
                             ggml_tensor* mask = nullptr) {
            // inputs_embeds: [N, n_token, hidden_size] -> [N, num_queries, hidden_size]
            auto query = params["meta_queries"];
            query      = ggml_reshape_3d(ctx->ggml_ctx, query, config.hidden_size, config.num_queries, 1);

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<QueryFormerBlock>(blocks["query_blocks." + std::to_string(i)]);
                query      = block->forward(ctx, query, inputs_embeds, mask);
            }
            return query;
        }
    };

    struct TextProjectionAttention : public GGMLBlock {
    protected:
        int64_t num_heads;
        int64_t head_dim;

    public:
        TextProjectionAttention(const TextProjectionConfig& config)
            : num_heads(config.num_heads), head_dim(config.hidden_size / config.num_heads) {
            blocks["q_proj"]   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, true);
            blocks["k_proj"]   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, true);
            blocks["v_proj"]   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, true);
            blocks["out_proj"] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, true);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(head_dim, config.norm_eps, false);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(head_dim, config.norm_eps, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            ggml_context* gctx = ctx->ggml_ctx;
            int64_t n_token    = x->ne[1];
            int64_t N          = x->ne[2];

            auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

            auto q = q_proj->forward(ctx, x);
            auto k = k_proj->forward(ctx, x);
            auto v = v_proj->forward(ctx, x);

            q = ggml_reshape_4d(gctx, q, head_dim, num_heads, n_token, N);
            k = ggml_reshape_4d(gctx, k, head_dim, num_heads, n_token, N);
            q = q_norm->forward(ctx, q);
            k = k_norm->forward(ctx, k);
            q = ggml_reshape_3d(gctx, q, head_dim * num_heads, n_token, N);
            k = ggml_reshape_3d(gctx, k, head_dim * num_heads, n_token, N);

            auto out = ggml_ext_attention_ext(ctx, q, k, v, num_heads);
            return out_proj->forward(ctx, out);
        }
    };

    struct TextProjectionBlock : public GGMLBlock {
    public:
        TextProjectionBlock(const TextProjectionConfig& config) {
            blocks["self_attn"]   = std::make_shared<TextProjectionAttention>(config);
            blocks["layer_norm1"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps, false);
            blocks["layer_norm2"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps, false);
            blocks["mlp.fc1"]     = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, true);
            blocks["mlp.fc2"]     = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto self_attn   = std::dynamic_pointer_cast<TextProjectionAttention>(blocks["self_attn"]);
            auto layer_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["layer_norm1"]);
            auto layer_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["layer_norm2"]);
            auto fc1         = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"]);
            auto fc2         = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"]);

            x = ggml_add(ctx->ggml_ctx, x, self_attn->forward(ctx, layer_norm1->forward(ctx, x)));

            auto h = fc1->forward(ctx, layer_norm2->forward(ctx, x));
            h      = ggml_gelu(ctx->ggml_ctx, h);
            h      = fc2->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct TextProjectionModel : public GGMLBlock {
    protected:
        TextProjectionConfig config;

    public:
        TextProjectionModel() = default;
        TextProjectionModel(const TextProjectionConfig& config)
            : config(config) {
            for (int i = 0; i < config.num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::make_shared<TextProjectionBlock>(config);
            }
            blocks["projector"] = std::make_shared<Linear>(config.hidden_size, config.projection_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, n_token, hidden_size] -> [N, n_token, projection_dim]
            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TextProjectionBlock>(blocks["layers." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            auto projector = std::dynamic_pointer_cast<Linear>(blocks["projector"]);
            return projector->forward(ctx, x);
        }
    };

    struct SigVQConfig {
        int64_t image_size         = 2048;
        int patch_size             = 16;
        int64_t in_channels        = 3;
        int64_t hidden_size        = 1536;
        int64_t intermediate_size  = 6144;
        int64_t num_layers         = 40;
        int64_t num_heads          = 16;
        int64_t codebook_size      = 16384;
        int64_t codebook_embed_dim = 2048;
        int64_t semantic_embed_dim = 4096;
        float norm_eps             = 1e-6f;
    };

    struct SigVQAttention : public GGMLBlock {
    protected:
        int64_t num_heads;
        int64_t head_dim;

    public:
        SigVQAttention(const SigVQConfig& config)
            : num_heads(config.num_heads), head_dim(config.hidden_size / config.num_heads) {
            blocks["qkv"]  = std::make_shared<Linear>(config.hidden_size, config.hidden_size * 3, true);
            blocks["proj"] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            ggml_context* gctx = ctx->ggml_ctx;
            auto qkv_proj      = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out_proj      = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            int64_t hidden_size = num_heads * head_dim;
            auto qkv            = qkv_proj->forward(ctx, x);
            auto q              = ggml_ext_slice(gctx, qkv, 0, 0, hidden_size);
            auto k              = ggml_ext_slice(gctx, qkv, 0, hidden_size, hidden_size * 2);
            auto v              = ggml_ext_slice(gctx, qkv, 0, hidden_size * 2, hidden_size * 3);

            auto out = ggml_ext_attention_ext(ctx, q, k, v, num_heads);
            return out_proj->forward(ctx, out);
        }
    };

    struct SigVQBlock : public GGMLBlock {
    public:
        SigVQBlock(const SigVQConfig& config) {
            blocks["norm1"]   = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"]   = std::make_shared<LayerNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]    = std::make_shared<SigVQAttention>(config);
            blocks["mlp.fc1"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, true);
            blocks["mlp.fc2"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<SigVQAttention>(blocks["attn"]);
            auto fc1   = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"]);
            auto fc2   = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"]);

            x      = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, norm1->forward(ctx, x)));
            auto h = fc1->forward(ctx, norm2->forward(ctx, x));
            h      = ggml_gelu_erf(ctx->ggml_ctx, h);
            h      = fc2->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct SigVQModel : public GGMLBlock {
    protected:
        SigVQConfig config;

    public:
        SigVQModel() = default;
        SigVQModel(const SigVQConfig& config)
            : config(config) {
            blocks["visual.patch_embed.proj"] = std::make_shared<Conv2d>(config.in_channels,
                                                                         config.hidden_size,
                                                                         std::make_pair(config.patch_size, config.patch_size),
                                                                         std::make_pair(config.patch_size, config.patch_size));
            for (int i = 0; i < config.num_layers; i++) {
                blocks["visual.blocks." + std::to_string(i)] = std::make_shared<SigVQBlock>(config);
            }
            blocks["vqmodel.quant_conv"]         = std::make_shared<Conv2d>(config.hidden_size,
                                                                    config.codebook_embed_dim,
                                                                    std::make_pair(1, 1));
            blocks["prior_projector.net.0.proj"] = std::make_shared<Linear>(config.semantic_embed_dim, config.semantic_embed_dim, true);
            blocks["prior_projector.net.2"]      = std::make_shared<Linear>(config.semantic_embed_dim, config.semantic_embed_dim, true);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["visual.embeddings.position_embedding.weight"] =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, (config.image_size / config.patch_size) * (config.image_size / config.patch_size));
            params["vqmodel.quantize.embedding.weight"] =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.codebook_embed_dim, config.codebook_size);
            params["prior_token_embedding.weight"] =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.semantic_embed_dim, config.codebook_size);
        }

        // Bilinear-resamples the square position-embedding grid onto the image's patch grid.
        // The reference uses grid_sample(align_corners=False, padding_mode="border"); the source
        // coordinate for output index j is therefore (j + 0.5) * side / out - 0.5, clamped.
        ggml_tensor* resample_pos_embed(GGMLRunnerContext* ctx,
                                        ggml_tensor* pos_idx,
                                        ggml_tensor* pos_weight) {
            auto pos_embed = params["visual.embeddings.position_embedding.weight"];
            auto gathered  = ggml_get_rows(ctx->ggml_ctx, pos_embed, pos_idx);
            return ggml_mul(ctx->ggml_ctx, gathered, pos_weight);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* pixel_values,
                             const std::vector<ggml_tensor*>& pos_idx,
                             const std::vector<ggml_tensor*>& pos_weight) {
            // pixel_values: [N, in_channels, H, W] -> [N, grid_h * grid_w, semantic_embed_dim]
            ggml_context* gctx = ctx->ggml_ctx;

            auto patch_embed = std::dynamic_pointer_cast<Conv2d>(blocks["visual.patch_embed.proj"]);
            auto quant_conv  = std::dynamic_pointer_cast<Conv2d>(blocks["vqmodel.quant_conv"]);
            auto proj_0      = std::dynamic_pointer_cast<Linear>(blocks["prior_projector.net.0.proj"]);
            auto proj_2      = std::dynamic_pointer_cast<Linear>(blocks["prior_projector.net.2"]);

            auto x          = patch_embed->forward(ctx, pixel_values);  // [N, hidden_size, grid_h, grid_w]
            int64_t grid_w  = x->ne[0];
            int64_t grid_h  = x->ne[1];
            int64_t n_token = grid_h * grid_w;
            int64_t N       = x->ne[3];

            x = ggml_reshape_3d(gctx, x, n_token, config.hidden_size, N);
            x = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [N, n_token, hidden_size]

            ggml_tensor* pos = nullptr;
            for (size_t i = 0; i < pos_idx.size(); i++) {
                auto corner = resample_pos_embed(ctx, pos_idx[i], pos_weight[i]);
                pos         = pos == nullptr ? corner : ggml_add(gctx, pos, corner);
            }
            x = ggml_add(gctx, x, ggml_reshape_3d(gctx, pos, config.hidden_size, n_token, N));

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<SigVQBlock>(blocks["visual.blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }

            // quant_conv is 1x1, so run it as a per-token projection rather than reshaping to 2-D.
            x = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [N, hidden_size, n_token]
            x = ggml_reshape_4d(gctx, x, n_token, 1, config.hidden_size, N);
            x = quant_conv->forward(ctx, x);  // [N, codebook_embed_dim, 1, n_token]
            x = ggml_reshape_3d(gctx, x, n_token, config.codebook_embed_dim, N);
            x = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [N, n_token, codebook_embed_dim]

            // Both sides are L2-normalized, so the nearest codebook entry by euclidean distance
            // is the one with the largest dot product.
            auto codebook  = ggml_l2_norm(gctx, params["vqmodel.quantize.embedding.weight"], 1e-12f);
            auto normed    = ggml_l2_norm(gctx, x, 1e-12f);
            auto logits    = ggml_mul_mat(gctx, codebook, normed);  // [N, n_token, codebook_size]
            auto token_ids = ggml_argmax(gctx, ggml_reshape_2d(gctx, logits, config.codebook_size, n_token * N));

            auto semantic = ggml_get_rows(gctx, params["prior_token_embedding.weight"], token_ids);
            semantic      = ggml_reshape_3d(gctx, semantic, config.semantic_embed_dim, n_token, N);

            auto h = proj_0->forward(ctx, semantic);
            h      = ggml_silu(gctx, h);
            return proj_2->forward(ctx, h);
        }
    };

    struct QueryFormerRunner : public GGMLRunner {
    public:
        QueryFormerConfig config;
        QueryFormerModel query_former;

        QueryFormerRunner(ggml_backend_t backend,
                          const String2TensorStorage& tensor_storage_map      = {},
                          const std::string prefix                            = "",
                          std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager) {
            query_former = QueryFormerModel(config);
            query_former.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "llada_image_queryformer";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            query_former.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& inputs_embeds) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(LLADA_IMAGE_TE_GRAPH_SIZE);
                ggml_tensor* x   = make_input(inputs_embeds);
                auto runner_ctx  = get_context();
                ggml_tensor* out = query_former.forward(&runner_ctx, x);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, true),
                                                   inputs_embeds.dim());
        }
    };

    struct TextProjectionRunner : public GGMLRunner {
    public:
        TextProjectionConfig config;
        TextProjectionModel text_projection;

        TextProjectionRunner(ggml_backend_t backend,
                             const String2TensorStorage& tensor_storage_map      = {},
                             const std::string prefix                            = "",
                             std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager) {
            text_projection = TextProjectionModel(config);
            text_projection.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "llada_image_text_projection";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            text_projection.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& hidden_states) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(LLADA_IMAGE_TE_GRAPH_SIZE);
                ggml_tensor* x   = make_input(hidden_states);
                auto runner_ctx  = get_context();
                ggml_tensor* out = text_projection.forward(&runner_ctx, x);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, true),
                                                   hidden_states.dim());
        }
    };

    struct SigVQRunner : public GGMLRunner {
    public:
        SigVQConfig config;
        SigVQModel sigvq;
        std::array<std::vector<int32_t>, 4> pos_idx_data;
        std::array<std::vector<float>, 4> pos_weight_data;

        SigVQRunner(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map      = {},
                    const std::string prefix                            = "",
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager) {
            sigvq = SigVQModel(config);
            sigvq.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "llada_image_sigvq";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            sigvq.get_param_tensors(tensors, prefix);
        }

        // Precomputes the four bilinear taps that resample the square position-embedding grid
        // onto a grid_h x grid_w patch grid, matching grid_sample(align_corners=False,
        // padding_mode="border").
        void build_pos_embed_taps(int64_t grid_h, int64_t grid_w) {
            const int64_t side = config.image_size / config.patch_size;
            for (auto& v : pos_idx_data) {
                v.clear();
            }
            for (auto& v : pos_weight_data) {
                v.clear();
            }

            auto clamp_index = [side](int64_t v) {
                return static_cast<int32_t>(std::min<int64_t>(std::max<int64_t>(v, 0), side - 1));
            };

            for (int64_t i = 0; i < grid_h; ++i) {
                double src_h    = (static_cast<double>(i) + 0.5) * side / static_cast<double>(grid_h) - 0.5;
                int64_t h_floor = static_cast<int64_t>(std::floor(src_h));
                double dh       = src_h - static_cast<double>(h_floor);
                for (int64_t j = 0; j < grid_w; ++j) {
                    double src_w    = (static_cast<double>(j) + 0.5) * side / static_cast<double>(grid_w) - 0.5;
                    int64_t w_floor = static_cast<int64_t>(std::floor(src_w));
                    double dw       = src_w - static_cast<double>(w_floor);

                    int32_t h0 = clamp_index(h_floor);
                    int32_t h1 = clamp_index(h_floor + 1);
                    int32_t w0 = clamp_index(w_floor);
                    int32_t w1 = clamp_index(w_floor + 1);

                    pos_idx_data[0].push_back(h0 * static_cast<int32_t>(side) + w0);
                    pos_idx_data[1].push_back(h0 * static_cast<int32_t>(side) + w1);
                    pos_idx_data[2].push_back(h1 * static_cast<int32_t>(side) + w0);
                    pos_idx_data[3].push_back(h1 * static_cast<int32_t>(side) + w1);

                    pos_weight_data[0].push_back(static_cast<float>((1.0 - dh) * (1.0 - dw)));
                    pos_weight_data[1].push_back(static_cast<float>((1.0 - dh) * dw));
                    pos_weight_data[2].push_back(static_cast<float>(dh * (1.0 - dw)));
                    pos_weight_data[3].push_back(static_cast<float>(dh * dw));
                }
            }
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& pixel_values) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(LLADA_IMAGE_TE_GRAPH_SIZE);
                ggml_tensor* x  = make_input(pixel_values);

                int64_t grid_h = x->ne[1] / config.patch_size;
                int64_t grid_w = x->ne[0] / config.patch_size;
                build_pos_embed_taps(grid_h, grid_w);

                std::vector<ggml_tensor*> pos_idx;
                std::vector<ggml_tensor*> pos_weight;
                for (int i = 0; i < 4; i++) {
                    auto idx = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, static_cast<int64_t>(pos_idx_data[i].size()));
                    set_backend_tensor_data(idx, pos_idx_data[i].data());
                    auto w = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, 1, static_cast<int64_t>(pos_weight_data[i].size()));
                    set_backend_tensor_data(w, pos_weight_data[i].data());
                    pos_idx.push_back(idx);
                    pos_weight.push_back(w);
                }

                auto runner_ctx  = get_context();
                ggml_tensor* out = sigvq.forward(&runner_ctx, x, pos_idx, pos_weight);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, true), 3);
        }
    };

}  // namespace LLaDAImageTE

#endif  // __SD_MODEL_TE_LLADA_IMAGE_TE_HPP__

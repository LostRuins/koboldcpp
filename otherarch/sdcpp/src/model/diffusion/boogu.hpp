#ifndef __SD_MODEL_DIFFUSION_BOOGU_HPP__
#define __SD_MODEL_DIFFUSION_BOOGU_HPP__

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/qwen_image.hpp"
#include "model_loader.h"

namespace Boogu {
    constexpr int BOOGU_GRAPH_SIZE = 65536;

    struct BooguConfig {
        int patch_size                   = 2;
        int64_t in_channels              = 16;
        int64_t out_channels             = 16;
        int64_t hidden_size              = 3360;
        int64_t num_layers               = 32;
        int64_t num_double_stream_layers = 8;
        int64_t num_refiner_layers       = 2;
        int64_t num_attention_heads      = 28;
        int64_t num_kv_heads             = 7;
        int64_t head_dim                 = 120;
        int64_t multiple_of              = 256;
        int64_t instruction_feat_dim     = 4096;
        int64_t timestep_embed_dim       = 1024;
        int theta                        = 10000;
        float timestep_scale             = 1000.0f;
        float norm_eps                   = 1e-5f;
        std::vector<int> axes_dim        = {40, 40, 40};
        int64_t axes_dim_sum             = 120;

        static int64_t count_blocks(const String2TensorStorage& tensor_storage_map,
                                    const std::string& prefix,
                                    const std::string& block_prefix) {
            int64_t count = 0;
            for (const auto& [name, _] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t pos = name.find(block_prefix);
                if (pos == std::string::npos) {
                    continue;
                }
                auto items = split_string(name.substr(pos), '.');
                if (items.size() > 1) {
                    count = std::max<int64_t>(count, atoi(items[1].c_str()) + 1);
                }
            }
            return count;
        }

        static BooguConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            BooguConfig config;
            int64_t detected_head_dim = 0;
            int64_t detected_kv_dim   = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "x_embedder.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area = config.patch_size * config.patch_size;
                    config.in_channels = tensor_storage.ne[0] / patch_area;
                    config.hidden_size = tensor_storage.ne[1];
                } else if (ends_with(name, "time_caption_embed.caption_embedder.1.weight") && tensor_storage.n_dims == 2) {
                    config.instruction_feat_dim = tensor_storage.ne[0];
                    config.hidden_size          = tensor_storage.ne[1];
                } else if (ends_with(name, "single_stream_layers.0.attn.norm_q.weight") && tensor_storage.n_dims == 1) {
                    detected_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "double_stream_layers.0.img_self_attn.norm_q.weight") && tensor_storage.n_dims == 1) {
                    detected_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "single_stream_layers.0.attn.to_k.weight") && tensor_storage.n_dims == 2) {
                    detected_kv_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "double_stream_layers.0.img_instruct_attn.processor.img_to_k.weight") && tensor_storage.n_dims == 2) {
                    detected_kv_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "norm_out.linear_2.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area  = config.patch_size * config.patch_size;
                    config.out_channels = tensor_storage.ne[1] / patch_area;
                }
            }

            config.num_layers               = std::max<int64_t>(1, count_blocks(tensor_storage_map, prefix, "single_stream_layers."));
            config.num_double_stream_layers = std::max<int64_t>(0, count_blocks(tensor_storage_map, prefix, "double_stream_layers."));
            int64_t noise_refiner_layers    = count_blocks(tensor_storage_map, prefix, "noise_refiner.");
            int64_t ref_refiner_layers      = count_blocks(tensor_storage_map, prefix, "ref_image_refiner.");
            int64_t context_refiner_layers  = count_blocks(tensor_storage_map, prefix, "context_refiner.");
            config.num_refiner_layers       = std::max<int64_t>(1, std::max(noise_refiner_layers, std::max(ref_refiner_layers, context_refiner_layers)));

            if (detected_head_dim > 0) {
                config.head_dim            = detected_head_dim;
                config.num_attention_heads = config.hidden_size / config.head_dim;
                config.axes_dim_sum        = config.head_dim;
                if (detected_kv_dim > 0) {
                    config.num_kv_heads = detected_kv_dim / config.head_dim;
                }
                if (config.axes_dim_sum == 120) {
                    config.axes_dim = {40, 40, 40};
                } else if (config.axes_dim_sum % 3 == 0) {
                    int axis        = static_cast<int>(config.axes_dim_sum / 3);
                    config.axes_dim = {axis, axis, axis};
                }
            }
            config.timestep_embed_dim = std::min<int64_t>(config.hidden_size, 1024);

            LOG_DEBUG("boogu_image: layers=%" PRId64 ", double_stream_layers=%" PRId64 ", refiner_layers=%" PRId64 ", hidden=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", head_dim=%" PRId64 ", in_channels=%" PRId64 ", out_channels=%" PRId64,
                      config.num_layers,
                      config.num_double_stream_layers,
                      config.num_refiner_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.num_kv_heads,
                      config.head_dim,
                      config.in_channels,
                      config.out_channels);
            return config;
        }
    };

    __STATIC_INLINE__ ggml_tensor* scale_modulate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
        return ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    }

    __STATIC_INLINE__ ggml_tensor* gate_residual(ggml_context* ctx, ggml_tensor* residual, ggml_tensor* x, ggml_tensor* gate) {
        gate = ggml_tanh(ctx, gate);
        gate = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);
        x    = ggml_mul(ctx, x, gate);
        return ggml_add(ctx, residual, x);
    }

    struct LuminaCombinedTimestepCaptionEmbedding : public GGMLBlock {
        int64_t frequency_embedding_size;
        float timestep_scale;

        LuminaCombinedTimestepCaptionEmbedding(int64_t hidden_size,
                                               int64_t instruction_feat_dim,
                                               int64_t frequency_embedding_size,
                                               float norm_eps,
                                               float timestep_scale)
            : frequency_embedding_size(frequency_embedding_size),
              timestep_scale(timestep_scale) {
            blocks["timestep_embedder"]  = std::make_shared<Qwen::TimestepEmbedding>(frequency_embedding_size, std::min<int64_t>(hidden_size, 1024));
            blocks["caption_embedder.0"] = std::make_shared<RMSNorm>(instruction_feat_dim, norm_eps);
            blocks["caption_embedder.1"] = std::make_shared<Linear>(instruction_feat_dim, hidden_size, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* timestep, ggml_tensor* text_hidden_states) {
            auto timestep_embedder  = std::dynamic_pointer_cast<Qwen::TimestepEmbedding>(blocks["timestep_embedder"]);
            auto caption_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["caption_embedder.0"]);
            auto caption_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["caption_embedder.1"]);

            auto timestep_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, static_cast<int>(frequency_embedding_size), 10000, timestep_scale);
            auto time_embed    = timestep_embedder->forward(ctx, timestep_proj);
            auto caption_embed = caption_embedder_1->forward(ctx, caption_embedder_0->forward(ctx, text_hidden_states));
            return {time_embed, caption_embed};
        }
    };

    struct LuminaRMSNormZero : public GGMLBlock {
        LuminaRMSNormZero(int64_t embedding_dim, int64_t conditioning_embedding_dim, float norm_eps) {
            blocks["linear"] = std::make_shared<Linear>(conditioning_embedding_dim, 4 * embedding_dim, true);
            blocks["norm"]   = std::make_shared<RMSNorm>(embedding_dim, norm_eps);
        }

        std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* emb) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            emb       = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, emb));
            auto mods = ggml_ext_chunk(ctx->ggml_ctx, emb, 4, 0);

            auto scale_msa = mods[0];
            auto gate_msa  = mods[1];
            auto scale_mlp = mods[2];
            auto gate_mlp  = mods[3];

            x = scale_modulate(ctx->ggml_ctx, norm->forward(ctx, x), scale_msa);
            return {x, gate_msa, scale_mlp, gate_mlp};
        }
    };

    struct LuminaFeedForward : public GGMLBlock {
        LuminaFeedForward(int64_t dim, int64_t inner_dim, int64_t multiple_of) {
            inner_dim          = multiple_of * ((inner_dim + multiple_of - 1) / multiple_of);
            blocks["linear_1"] = std::make_shared<Linear>(dim, inner_dim, false);
            blocks["linear_2"] = std::make_shared<Linear>(inner_dim, dim, false);
            blocks["linear_3"] = std::make_shared<Linear>(dim, inner_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);
            auto linear_3 = std::dynamic_pointer_cast<Linear>(blocks["linear_3"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                linear_2->set_force_prec_f32(true);
            }

            auto h1 = linear_1->forward(ctx, x);
            auto h2 = linear_3->forward(ctx, x);
            x       = ggml_swiglu_split(ctx->ggml_ctx, h1, h2);
            x       = linear_2->forward(ctx, x);
            return x;
        }
    };

    struct LuminaLayerNormContinuous : public GGMLBlock {
        LuminaLayerNormContinuous(int64_t embedding_dim,
                                  int64_t conditioning_embedding_dim,
                                  int64_t out_dim) {
            blocks["linear_1"] = std::make_shared<Linear>(conditioning_embedding_dim, embedding_dim, true);
            blocks["norm"]     = std::make_shared<LayerNorm>(embedding_dim, 1e-6f, false);
            blocks["linear_2"] = std::make_shared<Linear>(embedding_dim, out_dim, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* conditioning_embedding) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto emb = linear_1->forward(ctx, ggml_silu(ctx->ggml_ctx, conditioning_embedding));
            x        = scale_modulate(ctx->ggml_ctx, norm->forward(ctx, x), emb);
            x        = linear_2->forward(ctx, x);
            return x;
        }
    };

    struct Attention : public GGMLBlock {
        int64_t dim_head;
        int64_t heads;
        int64_t kv_heads;

        Attention(int64_t query_dim, int64_t dim_head, int64_t heads, int64_t kv_heads, float eps = 1e-5f)
            : dim_head(dim_head), heads(heads), kv_heads(kv_heads) {
            blocks["to_q"]     = std::make_shared<Linear>(query_dim, heads * dim_head, false);
            blocks["to_k"]     = std::make_shared<Linear>(query_dim, kv_heads * dim_head, false);
            blocks["to_v"]     = std::make_shared<Linear>(query_dim, kv_heads * dim_head, false);
            blocks["norm_q"]   = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["norm_k"]   = std::make_shared<RMSNorm>(dim_head, eps);
            blocks["to_out.0"] = std::make_shared<Linear>(heads * dim_head, query_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* encoder_hidden_states,
                             ggml_tensor* rotary_emb,
                             ggml_tensor* attention_mask = nullptr) {
            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto norm_q   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k   = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                to_out_0->set_force_prec_f32(true);
            }

            int64_t N  = hidden_states->ne[2];
            int64_t Lq = hidden_states->ne[1];
            int64_t Lk = encoder_hidden_states->ne[1];

            auto q = to_q->forward(ctx, hidden_states);
            q      = ggml_reshape_4d(ctx->ggml_ctx, q, dim_head, heads, Lq, N);
            auto k = to_k->forward(ctx, encoder_hidden_states);
            k      = ggml_reshape_4d(ctx->ggml_ctx, k, dim_head, kv_heads, Lk, N);
            auto v = to_v->forward(ctx, encoder_hidden_states);
            v      = ggml_reshape_4d(ctx->ggml_ctx, v, dim_head, kv_heads, Lk, N);

            q = norm_q->forward(ctx, q);
            k = norm_k->forward(ctx, k);

            auto out = Rope::attention(ctx, q, k, v, rotary_emb, attention_mask);
            out      = to_out_0->forward(ctx, out);
            return out;
        }
    };

    struct BooguImageTransformerBlock : public GGMLBlock {
        bool modulation;

        BooguImageTransformerBlock(int64_t dim,
                                   int64_t num_attention_heads,
                                   int64_t num_kv_heads,
                                   int64_t multiple_of,
                                   float norm_eps,
                                   bool modulation)
            : modulation(modulation) {
            int64_t head_dim       = dim / num_attention_heads;
            blocks["attn"]         = std::make_shared<Attention>(dim, head_dim, num_attention_heads, num_kv_heads, 1e-5f);
            blocks["feed_forward"] = std::make_shared<LuminaFeedForward>(dim, 4 * dim, multiple_of);
            if (modulation) {
                blocks["norm1"] = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            } else {
                blocks["norm1"] = std::make_shared<RMSNorm>(dim, norm_eps);
            }
            blocks["ffn_norm1"] = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["norm2"]     = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["ffn_norm2"] = std::make_shared<RMSNorm>(dim, norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* hidden_states,
                             ggml_tensor* rotary_emb,
                             ggml_tensor* temb           = nullptr,
                             ggml_tensor* attention_mask = nullptr) {
            auto attn         = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto feed_forward = std::dynamic_pointer_cast<LuminaFeedForward>(blocks["feed_forward"]);
            auto ffn_norm1    = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto norm2        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto ffn_norm2    = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                auto norm1 = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["norm1"]);
                auto mods  = norm1->forward(ctx, hidden_states, temb);

                auto norm_hidden_states = std::get<0>(mods);
                auto gate_msa           = std::get<1>(mods);
                auto scale_mlp          = std::get<2>(mods);
                auto gate_mlp           = std::get<3>(mods);

                auto attn_output = attn->forward(ctx, norm_hidden_states, norm_hidden_states, rotary_emb, attention_mask);
                hidden_states    = gate_residual(ctx->ggml_ctx, hidden_states, norm2->forward(ctx, attn_output), gate_msa);

                auto mlp_input  = scale_modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, hidden_states), scale_mlp);
                auto mlp_output = feed_forward->forward(ctx, mlp_input);
                hidden_states   = gate_residual(ctx->ggml_ctx, hidden_states, ffn_norm2->forward(ctx, mlp_output), gate_mlp);
            } else {
                auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);

                auto norm_hidden_states = norm1->forward(ctx, hidden_states);
                auto attn_output        = attn->forward(ctx, norm_hidden_states, norm_hidden_states, rotary_emb, attention_mask);
                hidden_states           = ggml_add(ctx->ggml_ctx, hidden_states, norm2->forward(ctx, attn_output));

                auto mlp_output = feed_forward->forward(ctx, ffn_norm1->forward(ctx, hidden_states));
                hidden_states   = ggml_add(ctx->ggml_ctx, hidden_states, ffn_norm2->forward(ctx, mlp_output));
            }
            return hidden_states;
        }
    };

    struct BooguImageJointAttention : public GGMLBlock {
        int64_t dim_head;
        int64_t heads;
        int64_t kv_heads;

        BooguImageJointAttention(int64_t dim, int64_t dim_head, int64_t heads, int64_t kv_heads)
            : dim_head(dim_head), heads(heads), kv_heads(kv_heads) {
            blocks["norm_q"]                  = std::make_shared<RMSNorm>(dim_head, 1e-5f);
            blocks["norm_k"]                  = std::make_shared<RMSNorm>(dim_head, 1e-5f);
            blocks["to_out.0"]                = std::make_shared<Linear>(heads * dim_head, dim, false);
            blocks["processor.img_to_q"]      = std::make_shared<Linear>(dim, heads * dim_head, false);
            blocks["processor.img_to_k"]      = std::make_shared<Linear>(dim, kv_heads * dim_head, false);
            blocks["processor.img_to_v"]      = std::make_shared<Linear>(dim, kv_heads * dim_head, false);
            blocks["processor.instruct_to_q"] = std::make_shared<Linear>(dim, heads * dim_head, false);
            blocks["processor.instruct_to_k"] = std::make_shared<Linear>(dim, kv_heads * dim_head, false);
            blocks["processor.instruct_to_v"] = std::make_shared<Linear>(dim, kv_heads * dim_head, false);
            blocks["processor.instruct_out"]  = std::make_shared<Linear>(heads * dim_head, dim, false);
            blocks["processor.img_out"]       = std::make_shared<Linear>(heads * dim_head, dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* img_hidden_states,
                             ggml_tensor* instruct_hidden_states,
                             ggml_tensor* rotary_emb,
                             ggml_tensor* attention_mask = nullptr) {
            auto norm_q        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"]);
            auto norm_k        = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"]);
            auto to_out_0      = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto img_to_q      = std::dynamic_pointer_cast<Linear>(blocks["processor.img_to_q"]);
            auto img_to_k      = std::dynamic_pointer_cast<Linear>(blocks["processor.img_to_k"]);
            auto img_to_v      = std::dynamic_pointer_cast<Linear>(blocks["processor.img_to_v"]);
            auto instruct_to_q = std::dynamic_pointer_cast<Linear>(blocks["processor.instruct_to_q"]);
            auto instruct_to_k = std::dynamic_pointer_cast<Linear>(blocks["processor.instruct_to_k"]);
            auto instruct_to_v = std::dynamic_pointer_cast<Linear>(blocks["processor.instruct_to_v"]);
            auto instruct_out  = std::dynamic_pointer_cast<Linear>(blocks["processor.instruct_out"]);
            auto img_out       = std::dynamic_pointer_cast<Linear>(blocks["processor.img_out"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                to_out_0->set_force_prec_f32(true);
            }

            int64_t N          = img_hidden_states->ne[2];
            int64_t L_img      = img_hidden_states->ne[1];
            int64_t L_instruct = instruct_hidden_states->ne[1];

            auto img_q = img_to_q->forward(ctx, img_hidden_states);
            img_q      = ggml_reshape_4d(ctx->ggml_ctx, img_q, dim_head, heads, L_img, N);
            auto img_k = img_to_k->forward(ctx, img_hidden_states);
            img_k      = ggml_reshape_4d(ctx->ggml_ctx, img_k, dim_head, kv_heads, L_img, N);
            auto img_v = img_to_v->forward(ctx, img_hidden_states);
            img_v      = ggml_reshape_4d(ctx->ggml_ctx, img_v, dim_head, kv_heads, L_img, N);

            auto instruct_q = instruct_to_q->forward(ctx, instruct_hidden_states);
            instruct_q      = ggml_reshape_4d(ctx->ggml_ctx, instruct_q, dim_head, heads, L_instruct, N);
            auto instruct_k = instruct_to_k->forward(ctx, instruct_hidden_states);
            instruct_k      = ggml_reshape_4d(ctx->ggml_ctx, instruct_k, dim_head, kv_heads, L_instruct, N);
            auto instruct_v = instruct_to_v->forward(ctx, instruct_hidden_states);
            instruct_v      = ggml_reshape_4d(ctx->ggml_ctx, instruct_v, dim_head, kv_heads, L_instruct, N);

            auto q = ggml_concat(ctx->ggml_ctx, instruct_q, img_q, 2);
            auto k = ggml_concat(ctx->ggml_ctx, instruct_k, img_k, 2);
            auto v = ggml_concat(ctx->ggml_ctx, instruct_v, img_v, 2);
            q      = norm_q->forward(ctx, q);
            k      = norm_k->forward(ctx, k);

            auto hidden_states = Rope::attention(ctx, q, k, v, rotary_emb, attention_mask);
            auto instruct_attn = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, 0, L_instruct);
            auto img_attn      = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, L_instruct, L_instruct + L_img);

            instruct_attn = instruct_out->forward(ctx, instruct_attn);
            img_attn      = img_out->forward(ctx, img_attn);
            hidden_states = ggml_concat(ctx->ggml_ctx, instruct_attn, img_attn, 1);
            hidden_states = to_out_0->forward(ctx, hidden_states);
            return hidden_states;
        }
    };

    struct BooguImageDoubleStreamBlock : public GGMLBlock {
        BooguImageDoubleStreamBlock(int64_t dim,
                                    int64_t num_attention_heads,
                                    int64_t num_kv_heads,
                                    int64_t multiple_of,
                                    float norm_eps) {
            int64_t head_dim                = dim / num_attention_heads;
            blocks["img_instruct_attn"]     = std::make_shared<BooguImageJointAttention>(dim, head_dim, num_attention_heads, num_kv_heads);
            blocks["img_self_attn"]         = std::make_shared<Attention>(dim, head_dim, num_attention_heads, num_kv_heads, 1e-5f);
            blocks["img_feed_forward"]      = std::make_shared<LuminaFeedForward>(dim, 4 * dim, multiple_of);
            blocks["instruct_feed_forward"] = std::make_shared<LuminaFeedForward>(dim, 4 * dim, multiple_of);
            blocks["img_norm1"]             = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            blocks["img_norm2"]             = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            blocks["img_norm3"]             = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            blocks["instruct_norm1"]        = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            blocks["instruct_norm2"]        = std::make_shared<LuminaRMSNormZero>(dim, std::min<int64_t>(dim, 1024), norm_eps);
            blocks["img_attn_norm"]         = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["img_self_attn_norm"]    = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["img_ffn_norm1"]         = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["img_ffn_norm2"]         = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["instruct_attn_norm"]    = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["instruct_ffn_norm1"]    = std::make_shared<RMSNorm>(dim, norm_eps);
            blocks["instruct_ffn_norm2"]    = std::make_shared<RMSNorm>(dim, norm_eps);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* img_hidden_states,
                                                      ggml_tensor* instruct_hidden_states,
                                                      ggml_tensor* joint_rotary_emb,
                                                      ggml_tensor* img_rotary_emb,
                                                      ggml_tensor* temb) {
            auto img_instruct_attn     = std::dynamic_pointer_cast<BooguImageJointAttention>(blocks["img_instruct_attn"]);
            auto img_self_attn         = std::dynamic_pointer_cast<Attention>(blocks["img_self_attn"]);
            auto img_feed_forward      = std::dynamic_pointer_cast<LuminaFeedForward>(blocks["img_feed_forward"]);
            auto instruct_feed_forward = std::dynamic_pointer_cast<LuminaFeedForward>(blocks["instruct_feed_forward"]);
            auto img_norm1             = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["img_norm1"]);
            auto img_norm2             = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["img_norm2"]);
            auto img_norm3             = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["img_norm3"]);
            auto instruct_norm1        = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["instruct_norm1"]);
            auto instruct_norm2        = std::dynamic_pointer_cast<LuminaRMSNormZero>(blocks["instruct_norm2"]);
            auto img_attn_norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["img_attn_norm"]);
            auto img_self_attn_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["img_self_attn_norm"]);
            auto img_ffn_norm1         = std::dynamic_pointer_cast<RMSNorm>(blocks["img_ffn_norm1"]);
            auto img_ffn_norm2         = std::dynamic_pointer_cast<RMSNorm>(blocks["img_ffn_norm2"]);
            auto instruct_attn_norm    = std::dynamic_pointer_cast<RMSNorm>(blocks["instruct_attn_norm"]);
            auto instruct_ffn_norm1    = std::dynamic_pointer_cast<RMSNorm>(blocks["instruct_ffn_norm1"]);
            auto instruct_ffn_norm2    = std::dynamic_pointer_cast<RMSNorm>(blocks["instruct_ffn_norm2"]);

            int64_t L_instruct = instruct_hidden_states->ne[1];

            auto img_norm1_out_vec      = img_norm1->forward(ctx, img_hidden_states, temb);
            auto img_norm2_out_vec      = img_norm2->forward(ctx, img_hidden_states, temb);
            auto img_norm3_out_vec      = img_norm3->forward(ctx, img_hidden_states, temb);
            auto instruct_norm1_out_vec = instruct_norm1->forward(ctx, instruct_hidden_states, temb);
            auto instruct_norm2_out_vec = instruct_norm2->forward(ctx, instruct_hidden_states, temb);

            auto img_norm1_out = std::get<0>(img_norm1_out_vec);
            auto img_gate_msa  = std::get<1>(img_norm1_out_vec);
            auto img_scale_mlp = std::get<2>(img_norm1_out_vec);
            auto img_gate_mlp  = std::get<3>(img_norm1_out_vec);

            auto img_norm2_out = std::get<0>(img_norm2_out_vec);
            auto img_shift_mlp = std::get<1>(img_norm2_out_vec);

            auto img_norm3_out = std::get<0>(img_norm3_out_vec);
            auto img_gate_self = std::get<1>(img_norm3_out_vec);

            auto instruct_norm1_out = std::get<0>(instruct_norm1_out_vec);
            auto instruct_gate_msa  = std::get<1>(instruct_norm1_out_vec);
            auto instruct_scale_mlp = std::get<2>(instruct_norm1_out_vec);
            auto instruct_gate_mlp  = std::get<3>(instruct_norm1_out_vec);

            auto instruct_norm2_out = std::get<0>(instruct_norm2_out_vec);
            auto instruct_shift_mlp = std::get<1>(instruct_norm2_out_vec);

            auto joint_attn_out    = img_instruct_attn->forward(ctx, img_norm1_out, instruct_norm1_out, joint_rotary_emb);
            auto instruct_attn_out = ggml_ext_slice(ctx->ggml_ctx, joint_attn_out, 1, 0, L_instruct);
            auto img_attn_out      = ggml_ext_slice(ctx->ggml_ctx, joint_attn_out, 1, L_instruct, joint_attn_out->ne[1]);

            auto img_self_attn_out = img_self_attn->forward(ctx, img_norm3_out, img_norm3_out, img_rotary_emb);

            img_hidden_states = gate_residual(ctx->ggml_ctx, img_hidden_states, img_attn_norm->forward(ctx, img_attn_out), img_gate_msa);
            img_hidden_states = gate_residual(ctx->ggml_ctx, img_hidden_states, img_self_attn_norm->forward(ctx, img_self_attn_out), img_gate_self);

            auto img_mlp_input = scale_modulate(ctx->ggml_ctx, img_norm2_out, img_scale_mlp);
            img_shift_mlp      = ggml_reshape_3d(ctx->ggml_ctx, img_shift_mlp, img_shift_mlp->ne[0], 1, img_shift_mlp->ne[1]);
            img_mlp_input      = ggml_add(ctx->ggml_ctx, img_mlp_input, img_shift_mlp);
            auto img_mlp_out   = img_feed_forward->forward(ctx, img_ffn_norm1->forward(ctx, img_mlp_input));
            img_hidden_states  = gate_residual(ctx->ggml_ctx, img_hidden_states, img_ffn_norm2->forward(ctx, img_mlp_out), img_gate_mlp);

            instruct_hidden_states  = gate_residual(ctx->ggml_ctx, instruct_hidden_states, instruct_attn_norm->forward(ctx, instruct_attn_out), instruct_gate_msa);
            auto instruct_mlp_input = scale_modulate(ctx->ggml_ctx, instruct_norm2_out, instruct_scale_mlp);
            instruct_shift_mlp      = ggml_reshape_3d(ctx->ggml_ctx, instruct_shift_mlp, instruct_shift_mlp->ne[0], 1, instruct_shift_mlp->ne[1]);
            instruct_mlp_input      = ggml_add(ctx->ggml_ctx, instruct_mlp_input, instruct_shift_mlp);
            auto instruct_mlp_out   = instruct_feed_forward->forward(ctx, instruct_ffn_norm1->forward(ctx, instruct_mlp_input));
            instruct_hidden_states  = gate_residual(ctx->ggml_ctx, instruct_hidden_states, instruct_ffn_norm2->forward(ctx, instruct_mlp_out), instruct_gate_mlp);

            return {img_hidden_states, instruct_hidden_states};
        }
    };

    struct BooguImageModel : public GGMLBlock {
        BooguConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            GGML_UNUSED(tensor_storage_map);
            GGML_UNUSED(prefix);
            params["image_index_embedding"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, 5);
        }

        BooguImageModel() = default;
        BooguImageModel(BooguConfig config)
            : config(std::move(config)) {
            blocks["x_embedder"]               = std::make_shared<Linear>(this->config.patch_size * this->config.patch_size * this->config.in_channels, this->config.hidden_size, true);
            blocks["ref_image_patch_embedder"] = std::make_shared<Linear>(this->config.patch_size * this->config.patch_size * this->config.in_channels, this->config.hidden_size, true);
            blocks["time_caption_embed"]       = std::make_shared<LuminaCombinedTimestepCaptionEmbedding>(this->config.hidden_size,
                                                                                                    this->config.instruction_feat_dim,
                                                                                                    256,
                                                                                                    this->config.norm_eps,
                                                                                                    this->config.timestep_scale);

            for (int i = 0; i < this->config.num_refiner_layers; i++) {
                blocks["noise_refiner." + std::to_string(i)]     = std::make_shared<BooguImageTransformerBlock>(this->config.hidden_size,
                                                                                                            this->config.num_attention_heads,
                                                                                                            this->config.num_kv_heads,
                                                                                                            this->config.multiple_of,
                                                                                                            this->config.norm_eps,
                                                                                                            true);
                blocks["ref_image_refiner." + std::to_string(i)] = std::make_shared<BooguImageTransformerBlock>(this->config.hidden_size,
                                                                                                                this->config.num_attention_heads,
                                                                                                                this->config.num_kv_heads,
                                                                                                                this->config.multiple_of,
                                                                                                                this->config.norm_eps,
                                                                                                                true);
                blocks["context_refiner." + std::to_string(i)]   = std::make_shared<BooguImageTransformerBlock>(this->config.hidden_size,
                                                                                                              this->config.num_attention_heads,
                                                                                                              this->config.num_kv_heads,
                                                                                                              this->config.multiple_of,
                                                                                                              this->config.norm_eps,
                                                                                                              false);
            }

            for (int i = 0; i < this->config.num_double_stream_layers; i++) {
                blocks["double_stream_layers." + std::to_string(i)] = std::make_shared<BooguImageDoubleStreamBlock>(this->config.hidden_size,
                                                                                                                    this->config.num_attention_heads,
                                                                                                                    this->config.num_kv_heads,
                                                                                                                    this->config.multiple_of,
                                                                                                                    this->config.norm_eps);
            }

            for (int i = 0; i < this->config.num_layers; i++) {
                blocks["single_stream_layers." + std::to_string(i)] = std::make_shared<BooguImageTransformerBlock>(this->config.hidden_size,
                                                                                                                   this->config.num_attention_heads,
                                                                                                                   this->config.num_kv_heads,
                                                                                                                   this->config.multiple_of,
                                                                                                                   this->config.norm_eps,
                                                                                                                   true);
            }

            blocks["norm_out"] = std::make_shared<LuminaLayerNormContinuous>(this->config.hidden_size,
                                                                             this->config.timestep_embed_dim,
                                                                             this->config.patch_size * this->config.patch_size * this->config.out_channels);
        }

        ggml_tensor* image_index_embedding(GGMLRunnerContext* ctx, int index) {
            GGML_ASSERT(index >= 0 && index < 5);
            auto embedding = params["image_index_embedding"];
            auto out       = ggml_view_1d(ctx->ggml_ctx,
                                          embedding,
                                          config.hidden_size,
                                          index * config.hidden_size * ggml_element_size(embedding));
            out            = ggml_reshape_3d(ctx->ggml_ctx, out, config.hidden_size, 1, 1);
            return out;
        }

        ggml_tensor* embed_refs(GGMLRunnerContext* ctx, const std::vector<ggml_tensor*>& ref_latents) {
            if (ref_latents.empty()) {
                return nullptr;
            }
            auto ref_image_patch_embedder = std::dynamic_pointer_cast<Linear>(blocks["ref_image_patch_embedder"]);

            ggml_tensor* ref_img = nullptr;
            for (int i = 0; i < static_cast<int>(ref_latents.size()); i++) {
                auto ref = DiT::pad_and_patchify(ctx, ref_latents[i], config.patch_size, config.patch_size, false);
                ref      = ref_image_patch_embedder->forward(ctx, ref);
                ref      = ggml_add(ctx->ggml_ctx, ref, image_index_embedding(ctx, std::min(i, 4)));
                ref_img  = ref_img == nullptr ? ref : ggml_concat(ctx->ggml_ctx, ref_img, ref, 1);
            }
            return ref_img;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timesteps,
                             ggml_tensor* context,
                             ggml_tensor* pe,
                             std::vector<ggml_tensor*> ref_latents = {}) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t N = x->ne[3];
            GGML_ASSERT(N == 1);

            auto x_embedder         = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto time_caption_embed = std::dynamic_pointer_cast<LuminaCombinedTimestepCaptionEmbedding>(blocks["time_caption_embed"]);
            auto norm_out           = std::dynamic_pointer_cast<LuminaLayerNormContinuous>(blocks["norm_out"]);

            auto timestep = ggml_sub(ctx->ggml_ctx, ggml_ext_ones_like(ctx->ggml_ctx, timesteps), timesteps);
            auto embeds   = time_caption_embed->forward(ctx, timestep, context);
            auto temb     = embeds.first;
            auto txt      = embeds.second;

            auto img        = DiT::pad_and_patchify(ctx, x, config.patch_size, config.patch_size, false);
            int64_t img_len = img->ne[1];
            img             = x_embedder->forward(ctx, img);
            auto ref_img    = embed_refs(ctx, ref_latents);
            int64_t ref_len = ref_img != nullptr ? ref_img->ne[1] : 0;
            int64_t txt_len = txt->ne[1];

            GGML_ASSERT(pe->ne[3] == txt_len + ref_len + img_len);
            auto txt_pe   = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt_len);
            auto noise_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt_len + ref_len, txt_len + ref_len + img_len);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<BooguImageTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);
                txt        = block->forward(ctx, txt, txt_pe);
                sd::ggml_graph_cut::mark_graph_cut(txt, "boogu.context_refiner." + std::to_string(i), "txt");
            }

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<BooguImageTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);
                img        = block->forward(ctx, img, noise_pe, temb);
                sd::ggml_graph_cut::mark_graph_cut(img, "boogu.noise_refiner." + std::to_string(i), "img");
            }

            ggml_tensor* combined_img = img;
            if (ref_img != nullptr) {
                auto ref_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt_len, txt_len + ref_len);
                for (int i = 0; i < config.num_refiner_layers; i++) {
                    auto block = std::dynamic_pointer_cast<BooguImageTransformerBlock>(blocks["ref_image_refiner." + std::to_string(i)]);
                    ref_img    = block->forward(ctx, ref_img, ref_pe, temb);
                    sd::ggml_graph_cut::mark_graph_cut(ref_img, "boogu.ref_image_refiner." + std::to_string(i), "ref_img");
                }
                combined_img = ggml_concat(ctx->ggml_ctx, ref_img, img, 1);
            }

            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt_len, txt_len + combined_img->ne[1]);
            for (int i = 0; i < config.num_double_stream_layers; i++) {
                auto block   = std::dynamic_pointer_cast<BooguImageDoubleStreamBlock>(blocks["double_stream_layers." + std::to_string(i)]);
                auto result  = block->forward(ctx, combined_img, txt, pe, img_pe, temb);
                combined_img = result.first;
                txt          = result.second;
                sd::ggml_graph_cut::mark_graph_cut(combined_img, "boogu.double_stream_layers." + std::to_string(i), "img");
                sd::ggml_graph_cut::mark_graph_cut(txt, "boogu.double_stream_layers." + std::to_string(i), "txt");
            }

            auto hidden_states = ggml_concat(ctx->ggml_ctx, txt, combined_img, 1);
            for (int i = 0; i < config.num_layers; i++) {
                auto block    = std::dynamic_pointer_cast<BooguImageTransformerBlock>(blocks["single_stream_layers." + std::to_string(i)]);
                hidden_states = block->forward(ctx, hidden_states, pe, temb);
                sd::ggml_graph_cut::mark_graph_cut(hidden_states, "boogu.single_stream_layers." + std::to_string(i), "hidden_states");
            }

            hidden_states = norm_out->forward(ctx, hidden_states, temb);
            hidden_states = ggml_ext_slice(ctx->ggml_ctx, hidden_states, 1, hidden_states->ne[1] - img_len, hidden_states->ne[1]);
            hidden_states = DiT::unpatchify_and_crop(ctx->ggml_ctx, hidden_states, H, W, config.patch_size, config.patch_size, false);
            hidden_states = ggml_ext_scale(ctx->ggml_ctx, hidden_states, -1.f);
            return hidden_states;
        }
    };

    __STATIC_INLINE__ int patched_token_count(int64_t size, int patch_size) {
        int pad = (patch_size - (static_cast<int>(size) % patch_size)) % patch_size;
        return (static_cast<int>(size) + pad) / patch_size;
    }

    __STATIC_INLINE__ void append_spatial_ids(std::vector<std::vector<float>>& ids,
                                              int bs,
                                              int pe_shift,
                                              int h_tokens,
                                              int w_tokens) {
        std::vector<std::vector<float>> image_ids(h_tokens * w_tokens, std::vector<float>(3, 0.0f));
        for (int h = 0; h < h_tokens; h++) {
            for (int w = 0; w < w_tokens; w++) {
                image_ids[h * w_tokens + w][0] = static_cast<float>(pe_shift);
                image_ids[h * w_tokens + w][1] = static_cast<float>(h);
                image_ids[h * w_tokens + w][2] = static_cast<float>(w);
            }
        }
        for (int b = 0; b < bs; b++) {
            ids.insert(ids.end(), image_ids.begin(), image_ids.end());
        }
    }

    __STATIC_INLINE__ std::vector<float> gen_boogu_pe(int h,
                                                      int w,
                                                      int patch_size,
                                                      int bs,
                                                      int context_len,
                                                      const std::vector<ggml_tensor*>& ref_latents,
                                                      int theta,
                                                      const std::vector<int>& axes_dim) {
        std::vector<std::vector<float>> ids;
        ids.reserve(static_cast<size_t>(bs) * context_len);
        for (int b = 0; b < bs; b++) {
            for (int i = 0; i < context_len; i++) {
                float pos = static_cast<float>(i);
                ids.push_back({pos, pos, pos});
            }
        }

        int pe_shift = context_len;
        for (ggml_tensor* ref : ref_latents) {
            int ref_h_tokens = patched_token_count(ref->ne[1], patch_size);
            int ref_w_tokens = patched_token_count(ref->ne[0], patch_size);
            append_spatial_ids(ids, bs, pe_shift, ref_h_tokens, ref_w_tokens);
            pe_shift += std::max(ref_h_tokens, ref_w_tokens);
        }

        int h_tokens = patched_token_count(h, patch_size);
        int w_tokens = patched_token_count(w, patch_size);
        append_spatial_ids(ids, bs, pe_shift, h_tokens, w_tokens);

        return Rope::embed_nd(ids, bs, static_cast<float>(theta), axes_dim);
    }

    struct BooguImageRunner : public DiffusionModelRunner {
        BooguConfig config;
        BooguImageModel boogu;
        std::vector<float> pe_vec;

        BooguImageRunner(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map      = {},
                         const std::string prefix                            = "",
                         SDVersion version                                   = VERSION_BOOGU_IMAGE,
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(BooguConfig::detect_from_weights(tensor_storage_map, prefix)) {
            boogu = BooguImageModel(config);
            boogu.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "boogu_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            boogu.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {}) {
            ggml_cgraph* gf        = new_graph_custom(BOOGU_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            std::vector<ggml_tensor*> ref_latents;
            ref_latents.reserve(ref_latents_tensor.size());
            for (const auto& ref_latent_tensor : ref_latents_tensor) {
                ref_latents.push_back(make_input(ref_latent_tensor));
            }

            pe_vec      = gen_boogu_pe(static_cast<int>(x->ne[1]),
                                       static_cast<int>(x->ne[0]),
                                       config.patch_size,
                                       static_cast<int>(x->ne[3]),
                                       static_cast<int>(context->ne[1]),
                                       ref_latents,
                                       config.theta,
                                       config.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = boogu.forward(&runner_ctx, x, timesteps, context, pe, ref_latents);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const std::vector<sd::Tensor<float>>& ref_latents = {}) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            static const std::vector<sd::Tensor<float>> empty_ref_latents;
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context),
                           diffusion_params.ref_latents ? *diffusion_params.ref_latents : empty_ref_latents);
        }
    };
}  // namespace Boogu

#endif  // __SD_MODEL_DIFFUSION_BOOGU_HPP__

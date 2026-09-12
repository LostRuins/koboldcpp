#ifndef __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__
#define __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/ggml_extend.h"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"
#include "model/te/llm.hpp"
#include "model_loader.h"

namespace SenseNovaU1 {
    constexpr int SENSENOVA_U1_GRAPH_SIZE = 327680;

    struct SenseNovaU1Config {
        int64_t hidden_size                  = 4096;
        int64_t intermediate_size            = 12288;
        int64_t num_layers                   = 42;
        int64_t num_heads                    = 32;
        int64_t num_kv_heads                 = 8;
        int64_t head_dim                     = 128;
        int64_t vocab_size                   = 151936;
        int64_t max_position_embeddings      = 262144;
        int64_t max_position_embeddings_hw   = 10000;
        int64_t vision_hidden_size           = 1024;
        int64_t patch_size                   = 16;
        int64_t vision_downsample_factor     = 2;
        int64_t in_channels                  = 3;
        int64_t timestep_embedding_size      = 256;
        float rms_norm_eps                   = 1e-6f;
        float rope_theta                     = 5000000.f;
        float rope_theta_hw                  = 10000.f;
        float noise_scale_base_image_seq_len = 64.f;
        float noise_scale_max_value          = 16.f;
        float t_eps                          = 0.02f;
        bool add_noise_scale_embedding       = true;

        int64_t image_token_stride() const {
            return patch_size * vision_downsample_factor;
        }

        static SenseNovaU1Config detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                     const std::string& prefix) {
            SenseNovaU1Config config;
            config.num_layers      = 0;
            const std::string root = prefix.empty() ? "" : prefix + ".";

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, root)) {
                    continue;
                }
                if (ends_with(name, "language_model.model.embed_tokens.weight") && tensor_storage.n_dims == 2) {
                    config.hidden_size = tensor_storage.ne[0];
                    config.vocab_size  = tensor_storage.ne[1];
                } else if (ends_with(name, "language_model.model.layers.0.mlp.gate_proj.weight") && tensor_storage.n_dims == 2) {
                    config.intermediate_size = tensor_storage.ne[1];
                } else if (ends_with(name, "language_model.model.layers.0.self_attn.q_proj.weight") && tensor_storage.n_dims == 2) {
                    config.num_heads = tensor_storage.ne[1] / config.head_dim;
                } else if (ends_with(name, "language_model.model.layers.0.self_attn.k_proj.weight") && tensor_storage.n_dims == 2) {
                    config.num_kv_heads = tensor_storage.ne[1] / config.head_dim;
                } else if (ends_with(name, "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight") && tensor_storage.n_dims == 4) {
                    config.patch_size         = tensor_storage.ne[0];
                    config.in_channels        = tensor_storage.ne[2];
                    config.vision_hidden_size = tensor_storage.ne[3];
                } else if (ends_with(name, "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight") && tensor_storage.n_dims == 4) {
                    config.vision_downsample_factor = tensor_storage.ne[0];
                }

                const std::string layer_prefix = root + "language_model.model.layers.";
                if (starts_with(name, layer_prefix)) {
                    const char* index_begin = name.c_str() + layer_prefix.size();
                    config.num_layers       = std::max<int64_t>(config.num_layers, std::strtoll(index_begin, nullptr, 10) + 1);
                }
            }

            if (config.num_layers == 0) {
                config.num_layers = 42;
            }
            config.add_noise_scale_embedding = tensor_storage_map.find(root + "fm_modules.noise_scale_embedder.mlp.0.weight") != tensor_storage_map.end();

            LOG_DEBUG("sensenova-u1.5: layers=%" PRId64 ", hidden=%" PRId64 ", intermediate=%" PRId64 ", heads=%" PRId64 ", kv_heads=%" PRId64 ", patch=%" PRId64 "x%" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.intermediate_size,
                      config.num_heads,
                      config.num_kv_heads,
                      config.patch_size,
                      config.vision_downsample_factor);
            return config;
        }
    };

    class StorageConv2d : public Conv2d {
    protected:
        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            this->prefix     = prefix;
            ggml_type wtype  = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16);
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  kernel_size.second,
                                                  kernel_size.first,
                                                  in_channels,
                                                  out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        StorageConv2d(int64_t in_channels,
                      int64_t out_channels,
                      std::pair<int, int> kernel_size,
                      std::pair<int, int> stride  = {1, 1},
                      std::pair<int, int> padding = {0, 0},
                      bool bias                   = true)
            : Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     {1, 1},
                     bias) {}
    };

    struct TimestepEmbedder : public GGMLBlock {
        int64_t frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["mlp.0"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size, true);
            blocks["mlp.2"] = std::make_shared<Linear>(hidden_size, hidden_size, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
            auto x     = ggml_ext_timestep_embedding(ctx->ggml_ctx,
                                                     timesteps,
                                                     static_cast<int>(frequency_embedding_size),
                                                     10000.f,
                                                     1.f);
            x          = mlp_0->forward(ctx, x);
            x          = ggml_silu_inplace(ctx->ggml_ctx, x);
            return mlp_2->forward(ctx, x);
        }
    };

    inline ggml_tensor* apply_vision_rope(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* position_x,
                                          ggml_tensor* position_y,
                                          float theta,
                                          int max_position) {
        GGML_ASSERT(x->ne[0] % 2 == 0);
        // ggml_rope_ext addresses positions through ne[2]. The vision
        // embeddings arrive as [hidden, tokens, batch], so add the singleton
        // head axis used by the RoPE kernel: [hidden, 1, tokens, batch].
        x                  = ggml_reshape_4d(ctx->ggml_ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
        const int64_t half = x->ne[0] / 2;
        auto x_part        = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, half);
        auto y_part        = ggml_ext_slice(ctx->ggml_ctx, x, 0, half, x->ne[0]);
        x_part             = ggml_rope_ext(ctx->ggml_ctx,
                                           x_part,
                                           position_x,
                                           nullptr,
                                           static_cast<int>(half),
                                           GGML_ROPE_TYPE_NORMAL,
                                           max_position,
                                           theta,
                                           1.f,
                                           0.f,
                                           1.f,
                                           32.f,
                                           1.f);
        y_part             = ggml_rope_ext(ctx->ggml_ctx,
                                           y_part,
                                           position_y,
                                           nullptr,
                                           static_cast<int>(half),
                                           GGML_ROPE_TYPE_NORMAL,
                                           max_position,
                                           theta,
                                           1.f,
                                           0.f,
                                           1.f,
                                           32.f,
                                           1.f);
        return ggml_concat(ctx->ggml_ctx, x_part, y_part, 0);
    }

    struct VisionEmbeddings : public GGMLBlock {
        SenseNovaU1Config config;

        explicit VisionEmbeddings(const SenseNovaU1Config& config)
            : config(config) {
            blocks["patch_embedding"] = std::make_shared<StorageConv2d>(config.in_channels,
                                                                        config.vision_hidden_size,
                                                                        std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)},
                                                                        std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)},
                                                                        std::pair<int, int>{0, 0},
                                                                        true);
            blocks["dense_embedding"] = std::make_shared<StorageConv2d>(config.vision_hidden_size,
                                                                        config.hidden_size,
                                                                        std::pair<int, int>{static_cast<int>(config.vision_downsample_factor), static_cast<int>(config.vision_downsample_factor)},
                                                                        std::pair<int, int>{static_cast<int>(config.vision_downsample_factor), static_cast<int>(config.vision_downsample_factor)},
                                                                        std::pair<int, int>{0, 0},
                                                                        true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* image,
                             ggml_tensor* position_x,
                             ggml_tensor* position_y) {
            auto patch_embedding = std::dynamic_pointer_cast<StorageConv2d>(blocks["patch_embedding"]);
            auto dense_embedding = std::dynamic_pointer_cast<StorageConv2d>(blocks["dense_embedding"]);

            auto x = patch_embedding->forward(ctx, image);
            x      = ggml_gelu_erf(ctx->ggml_ctx, x);

            const int64_t grid_w = x->ne[0];
            const int64_t grid_h = x->ne[1];
            const int64_t batch  = x->ne[3];
            x                    = ggml_reshape_3d(ctx->ggml_ctx, x, grid_w * grid_h, x->ne[2], batch);
            x                    = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
            x                    = apply_vision_rope(ctx,
                                                     x,
                                                     position_x,
                                                     position_y,
                                                     config.rope_theta_hw,
                                                     static_cast<int>(config.max_position_embeddings_hw));
            x                    = ggml_reshape_4d(ctx->ggml_ctx, x, config.vision_hidden_size, grid_w, grid_h, batch);
            x                    = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
            x                    = dense_embedding->forward(ctx, x);

            const int64_t token_w = x->ne[0];
            const int64_t token_h = x->ne[1];
            x                     = ggml_reshape_3d(ctx->ggml_ctx, x, token_w * token_h, x->ne[2], x->ne[3]);
            return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        }
    };

    inline ggml_tensor* pixel_shuffle(GGMLRunnerContext* ctx,
                                      ggml_tensor* x,
                                      int upscale_factor) {
        GGML_ASSERT(upscale_factor > 0);
        const int64_t h = x->ne[1];
        const int64_t w = x->ne[0];
        GGML_ASSERT(x->ne[2] % (upscale_factor * upscale_factor) == 0);
        x = ggml_ext_cont(ctx->ggml_ctx,
                          ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);
        return DiT::unpatchify(ctx->ggml_ctx, x, h, w, upscale_factor, upscale_factor, true);
    }

    struct PixelDecoder : public GGMLBlock {
        explicit PixelDecoder(const SenseNovaU1Config& config) {
            blocks["conv1"] = std::make_shared<StorageConv2d>(config.hidden_size / 4,
                                                              1024,
                                                              std::pair<int, int>{3, 3},
                                                              std::pair<int, int>{1, 1},
                                                              std::pair<int, int>{1, 1},
                                                              true);
            blocks["conv2"] = std::make_shared<StorageConv2d>(256,
                                                              192,
                                                              std::pair<int, int>{3, 3},
                                                              std::pair<int, int>{1, 1},
                                                              std::pair<int, int>{1, 1},
                                                              true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv1 = std::dynamic_pointer_cast<StorageConv2d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<StorageConv2d>(blocks["conv2"]);
            x          = pixel_shuffle(ctx, x, 2);
            x          = conv1->forward(ctx, x);
            x          = ggml_gelu_erf(ctx->ggml_ctx, x);
            x          = pixel_shuffle(ctx, x, 2);
            x          = conv2->forward(ctx, x);
            return pixel_shuffle(ctx, x, 8);
        }
    };

    enum class Branch {
        UNDERSTANDING,
        GENERATION,
    };

    struct Attention : public GGMLBlock {
        SenseNovaU1Config config;
        int layer_index;

        Attention(const SenseNovaU1Config& config, int layer_index)
            : config(config), layer_index(layer_index) {
            blocks["q_proj"]         = std::make_shared<Linear>(config.hidden_size, config.num_heads * config.head_dim, false);
            blocks["k_proj"]         = std::make_shared<Linear>(config.hidden_size, config.num_kv_heads * config.head_dim, false);
            blocks["v_proj"]         = std::make_shared<Linear>(config.hidden_size, config.num_kv_heads * config.head_dim, false);
            blocks["o_proj"]         = std::make_shared<Linear>(config.num_heads * config.head_dim, config.hidden_size, false);
            blocks["q_proj_mot_gen"] = std::make_shared<Linear>(config.hidden_size, config.num_heads * config.head_dim, false);
            blocks["k_proj_mot_gen"] = std::make_shared<Linear>(config.hidden_size, config.num_kv_heads * config.head_dim, false);
            blocks["v_proj_mot_gen"] = std::make_shared<Linear>(config.hidden_size, config.num_kv_heads * config.head_dim, false);
            blocks["o_proj_mot_gen"] = std::make_shared<Linear>(config.num_heads * config.head_dim, config.hidden_size, false);

            const int64_t axis_dim      = config.head_dim / 2;
            blocks["q_norm"]            = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["k_norm"]            = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["q_norm_hw"]         = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["k_norm_hw"]         = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["q_norm_mot_gen"]    = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["k_norm_mot_gen"]    = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["q_norm_hw_mot_gen"] = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
            blocks["k_norm_hw_mot_gen"] = std::make_shared<LLM::LLMRMSNorm>(axis_dim, config.rms_norm_eps);
        }

        ggml_tensor* apply_axis_rope(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* positions,
                                     int dimensions,
                                     float theta,
                                     int max_position) {
            return ggml_rope_ext(ctx->ggml_ctx,
                                 x,
                                 positions,
                                 nullptr,
                                 dimensions,
                                 GGML_ROPE_TYPE_NEOX,
                                 max_position,
                                 theta,
                                 1.f,
                                 0.f,
                                 1.f,
                                 32.f,
                                 1.f);
        }

        ggml_tensor* normalize_and_rotate(GGMLRunnerContext* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* position_t,
                                          ggml_tensor* position_h,
                                          ggml_tensor* position_w,
                                          const std::string& norm_name,
                                          const std::string& norm_hw_name) {
            const int64_t temporal_dim = config.head_dim / 2;
            const int64_t spatial_dim  = config.head_dim - temporal_dim;
            const int64_t axis_dim     = spatial_dim / 2;

            auto temporal = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, temporal_dim);
            auto spatial  = ggml_ext_slice(ctx->ggml_ctx, x, 0, temporal_dim, config.head_dim);
            temporal      = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(blocks[norm_name])->forward(ctx, temporal);
            spatial       = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(blocks[norm_hw_name])->forward(ctx, spatial);

            auto height = ggml_ext_slice(ctx->ggml_ctx, spatial, 0, 0, axis_dim);
            auto width  = ggml_ext_slice(ctx->ggml_ctx, spatial, 0, axis_dim, spatial_dim);
            temporal    = apply_axis_rope(ctx,
                                          temporal,
                                          position_t,
                                          static_cast<int>(temporal_dim),
                                          config.rope_theta,
                                          static_cast<int>(config.max_position_embeddings));
            height      = apply_axis_rope(ctx,
                                          height,
                                          position_h,
                                          static_cast<int>(axis_dim),
                                          config.rope_theta_hw,
                                          static_cast<int>(config.max_position_embeddings_hw));
            width       = apply_axis_rope(ctx,
                                          width,
                                          position_w,
                                          static_cast<int>(axis_dim),
                                          config.rope_theta_hw,
                                          static_cast<int>(config.max_position_embeddings_hw));
            return ggml_concat(ctx->ggml_ctx,
                               ggml_concat(ctx->ggml_ctx, temporal, height, 0),
                               width,
                               0);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* position_t,
                             ggml_tensor* position_h,
                             ggml_tensor* position_w,
                             ggml_tensor* attention_mask,
                             Branch branch,
                             const std::string& cache_prefix) {
            const bool generation    = branch == Branch::GENERATION;
            const std::string suffix = generation ? "_mot_gen" : "";
            auto q_proj              = std::dynamic_pointer_cast<Linear>(blocks["q_proj" + suffix]);
            auto k_proj              = std::dynamic_pointer_cast<Linear>(blocks["k_proj" + suffix]);
            auto v_proj              = std::dynamic_pointer_cast<Linear>(blocks["v_proj" + suffix]);
            auto o_proj              = std::dynamic_pointer_cast<Linear>(blocks["o_proj" + suffix]);

            const int64_t n_tokens = x->ne[1];
            const int64_t batch    = x->ne[2];
            auto q                 = ggml_reshape_4d(ctx->ggml_ctx,
                                                     q_proj->forward(ctx, x),
                                                     config.head_dim,
                                                     config.num_heads,
                                                     n_tokens,
                                                     batch);
            auto k                 = ggml_reshape_4d(ctx->ggml_ctx,
                                                     k_proj->forward(ctx, x),
                                                     config.head_dim,
                                                     config.num_kv_heads,
                                                     n_tokens,
                                                     batch);
            auto v                 = ggml_reshape_4d(ctx->ggml_ctx,
                                                     v_proj->forward(ctx, x),
                                                     config.head_dim,
                                                     config.num_kv_heads,
                                                     n_tokens,
                                                     batch);

            q = normalize_and_rotate(ctx,
                                     q,
                                     position_t,
                                     position_h,
                                     position_w,
                                     "q_norm" + suffix,
                                     "q_norm_hw" + suffix);
            k = normalize_and_rotate(ctx,
                                     k,
                                     position_t,
                                     position_h,
                                     position_w,
                                     "k_norm" + suffix,
                                     "k_norm_hw" + suffix);

            const std::string layer_cache = cache_prefix + "." + std::to_string(layer_index);
            if (generation) {
                auto prefix_k = ctx->load_cache_tensor(layer_cache + ".k");
                auto prefix_v = ctx->load_cache_tensor(layer_cache + ".v");
                GGML_ASSERT(prefix_k != nullptr && prefix_v != nullptr);
                k = ggml_concat(ctx->ggml_ctx, prefix_k, k, 2);
                v = ggml_concat(ctx->ggml_ctx, prefix_v, v, 2);
            } else {
                // Keep dedicated graph outputs alive until the runner copies them
                // into its persistent cache buffer after graph execution.
                auto cache_k = ggml_dup_tensor(ctx->ggml_ctx, k);
                cache_k      = ggml_cpy(ctx->ggml_ctx, k, cache_k);
                ggml_set_output(cache_k);
                auto cache_v = ggml_dup_tensor(ctx->ggml_ctx, v);
                cache_v      = ggml_cpy(ctx->ggml_ctx, v, cache_v);
                ggml_set_output(cache_v);
                ctx->persist_cache_tensor(layer_cache + ".k", cache_k);
                ctx->persist_cache_tensor(layer_cache + ".v", cache_v);
            }

            q = ggml_cont(ctx->ggml_ctx,
                          ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);
            k = ggml_cont(ctx->ggml_ctx,
                          ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              config.num_heads,
                                              attention_mask,
                                              true,
                                              ctx->flash_attn_enabled);
            return o_proj->forward(ctx, out);
        }
    };

    struct TransformerBlock : public GGMLBlock {
        TransformerBlock(const SenseNovaU1Config& config, int layer_index) {
            blocks["self_attn"]                        = std::make_shared<Attention>(config, layer_index);
            blocks["mlp"]                              = std::make_shared<LLM::MLP>(config.hidden_size, config.intermediate_size, false);
            blocks["mlp_mot_gen"]                      = std::make_shared<LLM::MLP>(config.hidden_size, config.intermediate_size, false);
            blocks["input_layernorm"]                  = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["input_layernorm_mot_gen"]          = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["post_attention_layernorm"]         = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["post_attention_layernorm_mot_gen"] = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* position_t,
                             ggml_tensor* position_h,
                             ggml_tensor* position_w,
                             ggml_tensor* attention_mask,
                             Branch branch,
                             const std::string& cache_prefix) {
            const bool generation = branch == Branch::GENERATION;
            auto input_norm       = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(
                blocks[generation ? "input_layernorm_mot_gen" : "input_layernorm"]);
            auto post_norm = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(
                blocks[generation ? "post_attention_layernorm_mot_gen" : "post_attention_layernorm"]);
            auto attention = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp       = std::dynamic_pointer_cast<LLM::MLP>(blocks[generation ? "mlp_mot_gen" : "mlp"]);

            auto residual = x;
            x             = input_norm->forward(ctx, x);
            x             = attention->forward(ctx,
                                               x,
                                               position_t,
                                               position_h,
                                               position_w,
                                               attention_mask,
                                               branch,
                                               cache_prefix);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_norm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            return ggml_add_inplace(ctx->ggml_ctx, x, residual);
        }
    };

    struct TextModel : public GGMLBlock {
        SenseNovaU1Config config;

        explicit TextModel(const SenseNovaU1Config& config)
            : config(config) {
            blocks["embed_tokens"] = std::make_shared<Embedding>(config.vocab_size, config.hidden_size);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::make_shared<TransformerBlock>(config, i);
            }
            blocks["norm"]         = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
            blocks["norm_mot_gen"] = std::make_shared<LLM::LLMRMSNorm>(config.hidden_size, config.rms_norm_eps);
        }

        ggml_tensor* embed(GGMLRunnerContext* ctx, ggml_tensor* input_ids) {
            return std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"])->forward(ctx, input_ids);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* position_t,
                             ggml_tensor* position_h,
                             ggml_tensor* position_w,
                             ggml_tensor* attention_mask,
                             Branch branch,
                             const std::string& cache_prefix) {
            for (int i = 0; i < config.num_layers; ++i) {
                auto layer = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);
                x          = layer->forward(ctx,
                                            x,
                                            position_t,
                                            position_h,
                                            position_w,
                                            attention_mask,
                                            branch,
                                            cache_prefix);
            }
            auto norm = std::dynamic_pointer_cast<LLM::LLMRMSNorm>(
                blocks[branch == Branch::GENERATION ? "norm_mot_gen" : "norm"]);
            return norm->forward(ctx, x);
        }
    };

    struct SenseNovaU1Model : public GGMLBlock {
        SenseNovaU1Config config;

        explicit SenseNovaU1Model(const SenseNovaU1Config& config)
            : config(config) {
            blocks["language_model.model"]                       = std::make_shared<TextModel>(config);
            blocks["fm_modules.vision_model_mot_gen.embeddings"] = std::make_shared<VisionEmbeddings>(config);
            blocks["fm_modules.timestep_embedder"]               = std::make_shared<TimestepEmbedder>(config.hidden_size,
                                                                                        config.timestep_embedding_size);
            if (config.add_noise_scale_embedding) {
                blocks["fm_modules.noise_scale_embedder"] = std::make_shared<TimestepEmbedder>(config.hidden_size,
                                                                                               config.timestep_embedding_size);
            }
            blocks["fm_modules.fm_head"] = std::make_shared<PixelDecoder>(config);
        }

        std::shared_ptr<TextModel> text_model() {
            return std::dynamic_pointer_cast<TextModel>(blocks["language_model.model"]);
        }

        std::shared_ptr<VisionEmbeddings> vision_embeddings() {
            return std::dynamic_pointer_cast<VisionEmbeddings>(blocks["fm_modules.vision_model_mot_gen.embeddings"]);
        }

        std::shared_ptr<TimestepEmbedder> timestep_embedder() {
            return std::dynamic_pointer_cast<TimestepEmbedder>(blocks["fm_modules.timestep_embedder"]);
        }

        std::shared_ptr<TimestepEmbedder> noise_scale_embedder() {
            if (!config.add_noise_scale_embedding) {
                return nullptr;
            }
            return std::dynamic_pointer_cast<TimestepEmbedder>(blocks["fm_modules.noise_scale_embedder"]);
        }

        std::shared_ptr<PixelDecoder> pixel_decoder() {
            return std::dynamic_pointer_cast<PixelDecoder>(blocks["fm_modules.fm_head"]);
        }
    };

    struct SenseNovaU1Runner : public DiffusionModelRunner {
        SenseNovaU1Config config;
        SenseNovaU1Model model;
        std::unordered_set<uint64_t> cached_prefix_hashes;
        std::vector<int32_t> position_t_vec;
        std::vector<int32_t> position_h_vec;
        std::vector<int32_t> position_w_vec;
        std::vector<float> attention_mask_vec;
        std::vector<float> noise_scale_vec;

        SenseNovaU1Runner(ggml_backend_t backend,
                          const String2TensorStorage& tensor_storage_map      = {},
                          const std::string& prefix                           = "",
                          std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(SenseNovaU1Config::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "SenseNova U1.5";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        static uint64_t hash_input_ids(const sd::Tensor<int32_t>& input_ids) {
            uint64_t hash = 1469598103934665603ULL;
            for (int32_t token : input_ids.values()) {
                uint32_t value = static_cast<uint32_t>(token);
                for (int byte = 0; byte < 4; ++byte) {
                    hash ^= static_cast<uint8_t>(value & 0xffU);
                    hash *= 1099511628211ULL;
                    value >>= 8;
                }
            }
            hash ^= static_cast<uint64_t>(input_ids.numel());
            hash *= 1099511628211ULL;
            return hash;
        }

        static std::string cache_prefix(uint64_t hash) {
            return "snu15." + std::to_string(hash);
        }

        ggml_tensor* make_position_tensor(const std::vector<int32_t>& values,
                                          const std::string& name) {
            auto tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, values.size());
            ggml_set_name(tensor, name.c_str());
            set_backend_tensor_data(tensor, values.data());
            return tensor;
        }

        ggml_cgraph* build_prefix_graph(const sd::Tensor<int32_t>& input_ids_tensor,
                                        const std::string& prefix_cache) {
            ggml_cgraph* graph   = new_graph_custom(SENSENOVA_U1_GRAPH_SIZE);
            ggml_tensor* ids     = make_input(input_ids_tensor);
            const int64_t length = input_ids_tensor.numel();

            position_t_vec.resize(length);
            position_h_vec.assign(length, 0);
            position_w_vec.assign(length, 0);
            for (int64_t i = 0; i < length; ++i) {
                position_t_vec[i] = static_cast<int32_t>(i);
            }
            auto position_t = make_position_tensor(position_t_vec, "snu15.prefix.position_t");
            auto position_h = make_position_tensor(position_h_vec, "snu15.prefix.position_h");
            auto position_w = make_position_tensor(position_w_vec, "snu15.prefix.position_w");

            attention_mask_vec.assign(static_cast<size_t>(length * length), 0.f);
            for (int64_t query = 0; query < length; ++query) {
                for (int64_t key = query + 1; key < length; ++key) {
                    attention_mask_vec[static_cast<size_t>(query * length + key)] = -INFINITY;
                }
            }
            auto attention_mask = ggml_new_tensor_2d(compute_ctx,
                                                     GGML_TYPE_F32,
                                                     length,
                                                     length);
            ggml_set_name(attention_mask, "snu15.prefix.attention_mask");
            set_backend_tensor_data(attention_mask, attention_mask_vec.data());

            auto runner_ctx = get_context();
            auto text_model = model.text_model();
            auto hidden     = text_model->embed(&runner_ctx, ids);
            hidden          = text_model->forward(&runner_ctx,
                                                  hidden,
                                                  position_t,
                                                  position_h,
                                                  position_w,
                                                  attention_mask,
                                                  Branch::UNDERSTANDING,
                                                  prefix_cache);
            ggml_build_forward_expand(graph, hidden);
            return graph;
        }

        bool ensure_prefix_cache(int n_threads,
                                 const sd::Tensor<int32_t>& input_ids,
                                 std::string* prefix_cache) {
            const uint64_t hash = hash_input_ids(input_ids);
            *prefix_cache       = cache_prefix(hash);
            if (cached_prefix_hashes.find(hash) != cached_prefix_hashes.end() &&
                get_cache_tensor_by_name(*prefix_cache + ".0.k") != nullptr) {
                return true;
            }

            if (cached_prefix_hashes.size() >= 2) {
                free_cache_ctx_and_buffer();
                cached_prefix_hashes.clear();
            }
            auto get_graph = [&]() {
                return build_prefix_graph(input_ids, *prefix_cache);
            };
            auto result = GGMLRunner::compute(get_graph, n_threads, false, true);
            if (!result.has_value()) {
                LOG_ERROR("SenseNova U1.5 prefix cache computation failed");
                return false;
            }
            cached_prefix_hashes.insert(hash);
            return true;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timestep_tensor,
                                 const std::string& prefix_cache,
                                 int64_t prefix_length) {
            ggml_cgraph* graph = new_graph_custom(SENSENOVA_U1_GRAPH_SIZE);
            ggml_tensor* x     = make_input(x_tensor);
            ggml_tensor* t     = make_input(timestep_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(x->ne[0] % config.image_token_stride() == 0);
            GGML_ASSERT(x->ne[1] % config.image_token_stride() == 0);

            const int64_t grid_w  = x->ne[0] / config.patch_size;
            const int64_t grid_h  = x->ne[1] / config.patch_size;
            const int64_t token_w = grid_w / config.vision_downsample_factor;
            const int64_t token_h = grid_h / config.vision_downsample_factor;
            const int64_t tokens  = token_w * token_h;

            position_h_vec.resize(grid_w * grid_h);
            position_w_vec.resize(grid_w * grid_h);
            for (int64_t index = 0; index < grid_w * grid_h; ++index) {
                position_h_vec[index] = static_cast<int32_t>(index / grid_w);
                position_w_vec[index] = static_cast<int32_t>(index % grid_w);
            }
            auto vision_position_x = make_position_tensor(position_w_vec, "snu15.vision.position_x");
            auto vision_position_y = make_position_tensor(position_h_vec, "snu15.vision.position_y");

            auto runner_ctx     = get_context();
            auto hidden         = model.vision_embeddings()->forward(&runner_ctx,
                                                                     x,
                                                                     vision_position_x,
                                                                     vision_position_y);
            auto time_embedding = model.timestep_embedder()->forward(&runner_ctx, t);
            time_embedding      = ggml_reshape_3d(compute_ctx, time_embedding, config.hidden_size, 1, 1);
            hidden              = ggml_add(compute_ctx, hidden, time_embedding);

            if (config.add_noise_scale_embedding) {
                const float image_tokens = static_cast<float>(tokens);
                const float noise_scale  = std::min(config.noise_scale_max_value,
                                                    std::sqrt(image_tokens / config.noise_scale_base_image_seq_len));
                noise_scale_vec          = {noise_scale / config.noise_scale_max_value};
                auto noise_scale_tensor  = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
                ggml_set_name(noise_scale_tensor, "snu15.noise_scale");
                set_backend_tensor_data(noise_scale_tensor, noise_scale_vec.data());
                auto noise_embedding = model.noise_scale_embedder()->forward(&runner_ctx, noise_scale_tensor);
                noise_embedding      = ggml_reshape_3d(compute_ctx, noise_embedding, config.hidden_size, 1, 1);
                hidden               = ggml_add(compute_ctx, hidden, noise_embedding);
            }

            position_t_vec.assign(tokens, static_cast<int32_t>(prefix_length));
            position_h_vec.resize(tokens);
            position_w_vec.resize(tokens);
            for (int64_t index = 0; index < tokens; ++index) {
                position_h_vec[index] = static_cast<int32_t>(index / token_w);
                position_w_vec[index] = static_cast<int32_t>(index % token_w);
            }
            auto position_t = make_position_tensor(position_t_vec, "snu15.image.position_t");
            auto position_h = make_position_tensor(position_h_vec, "snu15.image.position_h");
            auto position_w = make_position_tensor(position_w_vec, "snu15.image.position_w");

            hidden            = model.text_model()->forward(&runner_ctx,
                                                            hidden,
                                                            position_t,
                                                            position_h,
                                                            position_w,
                                                            nullptr,
                                                            Branch::GENERATION,
                                                            prefix_cache);
            hidden            = ggml_reshape_4d(compute_ctx,
                                                hidden,
                                                config.hidden_size,
                                                token_w,
                                                token_h,
                                                x->ne[3]);
            hidden            = ggml_cont(compute_ctx, ggml_permute(compute_ctx, hidden, 2, 0, 1, 3));
            auto x_prediction = model.pixel_decoder()->forward(&runner_ctx, hidden);

            const float timestep = timestep_tensor.values()[0];
            const float denom    = std::max(1.f - timestep, config.t_eps);
            auto velocity        = ggml_scale(compute_ctx,
                                              ggml_sub(compute_ctx, x_prediction, x),
                                              1.f / denom);
            ggml_build_forward_expand(graph, velocity);
            return graph;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timestep,
                                  const sd::Tensor<int32_t>& input_ids) {
            std::string prefix_cache;
            if (!ensure_prefix_cache(n_threads, input_ids, &prefix_cache)) {
                return {};
            }
            auto get_graph = [&]() {
                return build_graph(x, timestep, prefix_cache, input_ids.numel());
            };
            return restore_trailing_singleton_dims(
                GGMLRunner::compute(get_graph, n_threads, false),
                x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            const auto* extra = diffusion_extra_as<SenseNovaU1DiffusionExtra>(diffusion_params);
            GGML_ASSERT(extra->input_ids != nullptr);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           *extra->input_ids);
        }
    };
}  // namespace SenseNovaU1

#endif  // __SD_MODEL_DIFFUSION_SENSENOVA_U1_H__

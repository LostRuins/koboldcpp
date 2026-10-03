#ifndef __SD_MODEL_DIFFUSION_PIXART_HPP__
#define __SD_MODEL_DIFFUSION_PIXART_HPP__

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <vector>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/mmdit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
// Ref: https://github.com/PixArt-alpha/PixArt-sigma

namespace PixArt {
    constexpr int PIXART_GRAPH_SIZE = 20480;
    constexpr int ADALN_EMBED_DIM   = 256;

    struct PixArtConfig {
        int64_t in_channels         = 4;
        int64_t out_channels        = 8;  // learn_sigma: noise prediction + learned variance
        int64_t hidden_size         = 1152;
        int64_t cross_attention_dim = 1152;
        int64_t caption_channels    = 4096;
        int64_t num_heads           = 16;
        int64_t patch_size          = 2;
        int64_t ffn_dim             = 4608;
        int64_t pos_embed_base_size = 64;
        float interpolation_scale   = 2.f;
        int num_layers              = 28;

        static PixArtConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            PixArtConfig config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = weights.find(prefix + "." + suffix);
                return it == weights.end() ? nullptr : &it->second;
            };
            if (auto w = find("x_embedder.proj.weight")) {
                config.hidden_size = w->ne[3];
                config.in_channels = w->ne[2];
                config.patch_size  = w->ne[0];
            }
            if (auto w = find("final_layer.linear.weight")) {
                config.out_channels = w->ne[1] / (config.patch_size * config.patch_size);
            }
            if (auto w = find("y_embedder.y_proj.fc1.weight")) {
                config.caption_channels = w->ne[0];
            }
            if (auto w = find("blocks.0.cross_attn.kv_linear.weight")) {
                config.cross_attention_dim = w->ne[0];
            }
            if (auto w = find("blocks.0.mlp.fc1.weight")) {
                config.ffn_dim = w->ne[1];
            }
            if (find("csize_embedder.mlp.0.weight") != nullptr) {
                LOG_WARN("pixart: resolution/aspect-ratio micro conditions are not supported; output may differ from the reference");
            }
            int layers                     = 0;
            const std::string block_prefix = prefix + ".blocks.";
            for (const auto& [name, _] : weights) {
                if (starts_with(name, block_prefix)) {
                    layers = std::max(layers, atoi(name.substr(block_prefix.size()).c_str()) + 1);
                }
            }
            if (layers > 0) {
                config.num_layers = layers;
                LOG_VERBOSE("pixart: layers = %d, hidden_size = %" PRId64,
                            layers, config.hidden_size);
            }
            return config;
        }
    };

    // Mirrors diffusers get_2d_sincos_pos_embed for a (gh, gw) patch grid.
    static std::vector<float> gen_2d_sincos_pos_embed(int64_t dim,
                                                      int64_t gh,
                                                      int64_t gw,
                                                      int64_t base_size,
                                                      float interpolation_scale) {
        // diffusers: meshgrid(grid_w, grid_h, indexing="xy") -> grid[0]=w, grid[1]=h,
        // embedding = concat(sincos(w), sincos(h))
        std::vector<float> out(static_cast<size_t>(gh) * gw * dim);
        int64_t quarter = dim / 4;
        for (int64_t h = 0; h < gh; ++h) {
            float pos_h = static_cast<float>(h) / (static_cast<float>(gh) / base_size) / interpolation_scale;
            for (int64_t w = 0; w < gw; ++w) {
                float pos_w  = static_cast<float>(w) / (static_cast<float>(gw) / base_size) / interpolation_scale;
                float* dst_w = out.data() + (h * gw + w) * dim;
                float* dst_h = dst_w + dim / 2;
                for (int64_t i = 0; i < quarter; ++i) {
                    float omega        = 1.f / powf(10000.f, static_cast<float>(i) / quarter);
                    dst_w[i]           = sinf(pos_w * omega);
                    dst_w[i + quarter] = cosf(pos_w * omega);
                    dst_h[i]           = sinf(pos_h * omega);
                    dst_h[i + quarter] = cosf(pos_h * omega);
                }
            }
        }
        return out;
    }

    class PixArtTimestepEmbedding : public GGMLBlock {
    public:
        PixArtTimestepEmbedding(int64_t in_channels, int64_t out_dim) {
            blocks["mlp.0"] = std::make_shared<Linear>(in_channels, out_dim);
            blocks["mlp.2"] = std::make_shared<Linear>(out_dim, out_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"])->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            return std::dynamic_pointer_cast<Linear>(blocks["mlp.2"])->forward(ctx, x);
        }
    };

    class PixArtAttention : public GGMLBlock {
        int64_t num_heads;
        bool self_attention;

    public:
        PixArtAttention(int64_t dim, int64_t num_heads, int64_t context_dim, bool self_attention)
            : num_heads(num_heads), self_attention(self_attention) {
            if (self_attention) {
                blocks["qkv"] = std::make_shared<Linear>(dim, 3 * dim);
            } else {
                blocks["q_linear"]  = std::make_shared<Linear>(dim, dim);
                blocks["kv_linear"] = std::make_shared<Linear>(context_dim, 2 * dim);
            }
            blocks["proj"] = std::make_shared<Linear>(dim, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, ggml_tensor* mask = nullptr) {
            std::vector<ggml_tensor*> qkv;
            if (self_attention) {
                auto projected = std::dynamic_pointer_cast<Linear>(blocks["qkv"])->forward(ctx, x);
                qkv            = ggml_ext_chunk(ctx->ggml_ctx, projected, 3, 0);
            } else {
                auto q     = std::dynamic_pointer_cast<Linear>(blocks["q_linear"])->forward(ctx, x);
                auto kv    = std::dynamic_pointer_cast<Linear>(blocks["kv_linear"])->forward(ctx, context);
                auto parts = ggml_ext_chunk(ctx->ggml_ctx, kv, 2, 0);
                qkv        = {q, parts[0], parts[1]};
            }
            auto out = ggml_ext_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads, mask, false, ctx->flash_attn_enabled);
            return std::dynamic_pointer_cast<Linear>(blocks["proj"])->forward(ctx, out);
        }
    };

    class PixArtBlock : public GGMLBlock {
        int64_t dim;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type wtype             = get_type(prefix + "scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, dim, 6);
        }

    public:
        PixArtBlock(int64_t dim, int64_t num_heads, int64_t context_dim, int64_t ffn_dim)
            : dim(dim) {
            blocks["attn"]       = std::make_shared<PixArtAttention>(dim, num_heads, dim, true);
            blocks["cross_attn"] = std::make_shared<PixArtAttention>(dim, num_heads, context_dim, false);
            blocks["mlp.fc1"]    = std::make_shared<Linear>(dim, ffn_dim);
            blocks["mlp.fc2"]    = std::make_shared<Linear>(ffn_dim, dim);
        }

        static ggml_tensor* norm(ggml_context* ctx, ggml_tensor* x) {
            return ggml_norm(ctx, x, 1e-6f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mod, ggml_tensor* context, ggml_tensor* context_mask) {
            // x: [N, n_token, dim]
            // mod: [N, 6 * dim], shared adaLN-single output
            int64_t N = x->ne[2];

            auto table = params["scale_shift_table"];
            if (table->type != GGML_TYPE_F32) {
                table = ggml_cast(ctx->ggml_ctx, table, GGML_TYPE_F32);
            }
            table   = ggml_reshape_3d(ctx->ggml_ctx, table, dim, 6, 1);
            auto m  = ggml_add(ctx->ggml_ctx, ggml_reshape_3d(ctx->ggml_ctx, mod, dim, 6, N), table);
            auto mv = ggml_ext_chunk(ctx->ggml_ctx, ggml_reshape_2d(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, m), dim * 6, N), 6, 0);

            auto attn1 = std::dynamic_pointer_cast<PixArtAttention>(blocks["attn"]);
            auto attn2 = std::dynamic_pointer_cast<PixArtAttention>(blocks["cross_attn"]);
            auto proj  = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"]);
            auto fc2   = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"]);

            auto gate = [&](ggml_tensor* y, ggml_tensor* g) {
                g = ggml_reshape_3d(ctx->ggml_ctx, g, dim, 1, N);
                return ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, y, g));
            };

            auto h = modulate(ctx->ggml_ctx, norm(ctx->ggml_ctx, x), mv[0], mv[1]);
            x      = gate(attn1->forward(ctx, h, h), mv[2]);
            // ada_norm_single: no norm before cross-attention (PixArtMS.py)
            x = ggml_add(ctx->ggml_ctx, x, attn2->forward(ctx, x, context, context_mask));
            h = modulate(ctx->ggml_ctx, norm(ctx->ggml_ctx, x), mv[3], mv[4]);
            h = proj->forward(ctx, h);
            h = ggml_ext_gelu(ctx->ggml_ctx, h, true);
            h = fc2->forward(ctx, h);
            return gate(h, mv[5]);
        }
    };

    class PixArtModel : public GGMLBlock {
        PixArtConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type wtype                         = get_type(prefix + "final_layer.scale_shift_table", tensor_storage_map, GGML_TYPE_F32);
            params["final_layer.scale_shift_table"] = ggml_new_tensor_2d(ctx, wtype, config.hidden_size, 2);
        }

    public:
        PixArtModel() = default;
        PixArtModel(const PixArtConfig& config)
            : config(config) {
            blocks["x_embedder.proj"]       = std::make_shared<Conv2d>(config.in_channels,
                                                                 config.hidden_size,
                                                                 std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)},
                                                                 std::pair<int, int>{static_cast<int>(config.patch_size), static_cast<int>(config.patch_size)});
            blocks["t_embedder"]            = std::make_shared<PixArtTimestepEmbedding>(ADALN_EMBED_DIM, config.hidden_size);
            blocks["t_block.1"]             = std::make_shared<Linear>(config.hidden_size, 6 * config.hidden_size);
            blocks["y_embedder.y_proj.fc1"] = std::make_shared<Linear>(config.caption_channels, config.hidden_size);
            blocks["y_embedder.y_proj.fc2"] = std::make_shared<Linear>(config.hidden_size, config.cross_attention_dim);
            for (int i = 0; i < config.num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] =
                    std::make_shared<PixArtBlock>(config.hidden_size, config.num_heads, config.cross_attention_dim, config.ffn_dim);
            }
            blocks["final_layer.norm_final"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["final_layer.linear"]     = std::make_shared<Linear>(config.hidden_size,
                                                                    config.patch_size * config.patch_size * config.out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timesteps,
                             ggml_tensor* context,
                             ggml_tensor* pos_embed,
                             ggml_tensor* context_mask) {
            // x: [N, C, H, W] latent, context: [N, n_ctx, caption_channels]
            int64_t W  = x->ne[0];
            int64_t H  = x->ne[1];
            int64_t N  = x->ne[3];
            int64_t p  = config.patch_size;
            int64_t wp = W / p;
            int64_t hp = H / p;

            auto h = std::dynamic_pointer_cast<Conv2d>(blocks["x_embedder.proj"])->forward(ctx, x);  // [N, hidden, hp, wp]
            h      = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, h, 1, 2, 0, 3));       // [N, hp, wp, hidden] -> [N, hp*wp, hidden]
            h      = ggml_reshape_3d(ctx->ggml_ctx, h, config.hidden_size, wp * hp, N);              // [N, hp*wp, hidden]
            h      = ggml_add(ctx->ggml_ctx, h, pos_embed);

            auto t   = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, ADALN_EMBED_DIM, 10000);
            auto emb = std::dynamic_pointer_cast<PixArtTimestepEmbedding>(blocks["t_embedder"])->forward(ctx, t);

            auto mod = std::dynamic_pointer_cast<Linear>(blocks["t_block.1"])
                           ->forward(ctx, ggml_silu(ctx->ggml_ctx, emb));  // [N, 6 * hidden]

            auto ctx_emb = std::dynamic_pointer_cast<Linear>(blocks["y_embedder.y_proj.fc1"])->forward(ctx, context);
            ctx_emb      = ggml_ext_gelu(ctx->ggml_ctx, ctx_emb, true);
            ctx_emb      = std::dynamic_pointer_cast<Linear>(blocks["y_embedder.y_proj.fc2"])->forward(ctx, ctx_emb);

            for (int i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<PixArtBlock>(blocks["blocks." + std::to_string(i)]);
                h          = block->forward(ctx, h, mod, ctx_emb, context_mask);
                sd::ggml_graph_cut::mark_graph_cut(h, "pixart.blocks." + std::to_string(i), "h");
            }

            // scale_shift_table + emb -> (shift, scale) for the affine-free final norm
            auto tail_table = params["final_layer.scale_shift_table"];
            if (tail_table->type != GGML_TYPE_F32) {
                tail_table = ggml_cast(ctx->ggml_ctx, tail_table, GGML_TYPE_F32);
            }
            auto ss    = ggml_add(ctx->ggml_ctx,
                                  ggml_reshape_3d(ctx->ggml_ctx, tail_table, config.hidden_size, 2, 1),
                                  ggml_reshape_3d(ctx->ggml_ctx, emb, config.hidden_size, 1, N));  // [2, hidden, N]
            auto parts = ggml_ext_chunk(ctx->ggml_ctx,
                                        ggml_reshape_2d(ctx->ggml_ctx, ggml_ext_cont(ctx->ggml_ctx, ss), config.hidden_size * 2, N),
                                        2, 0);
            h          = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer.norm_final"])->forward(ctx, h);
            h          = modulate(ctx->ggml_ctx, h, parts[0], parts[1]);
            h          = std::dynamic_pointer_cast<Linear>(blocks["final_layer.linear"])->forward(ctx, h);  // [N, hp*wp, p*p*out_ch]
            h          = DiT::unpatchify(ctx->ggml_ctx, h, hp, wp, static_cast<int>(p), static_cast<int>(p), false);
            return h;  // [N, out_channels, H, W]
        }
    };

    struct PixArtRunner : public DiffusionModelRunner {
        PixArtConfig config;
        PixArtModel model;
        std::vector<float> pos_vec;

        PixArtRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map      = {},
                     const std::string prefix                            = "",
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr,
                     const char* model_args                              = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(PixArtConfig::detect_from_weights(tensor_storage_map, prefix)) {
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                if (key == "pixart_pos_embed_base_size") {
                    int parsed = 0;
                    if (parse_strict_int(value, parsed)) {
                        config.pos_embed_base_size = parsed;
                    } else {
                        LOG_WARN("ignoring invalid PixArt model arg '%s=%s'", key.c_str(), value.c_str());
                    }
                } else if (key == "pixart_interpolation_scale") {
                    float parsed = 0.f;
                    if (parse_strict_float(value, parsed)) {
                        config.interpolation_scale = parsed;
                    } else {
                        LOG_WARN("ignoring invalid PixArt model arg '%s=%s'", key.c_str(), value.c_str());
                    }
                }
            }
            model = PixArtModel(config);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "pixart";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& mask_tensor) {
            ggml_cgraph* gf        = new_graph_custom(PIXART_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            ggml_tensor* context_mask = nullptr;
            if (!mask_tensor.empty()) {
                // additive attention bias over context tokens: 0 keep / -inf discard
                context_mask = ggml_reshape_4d(compute_ctx, make_input(mask_tensor), mask_tensor.shape()[0], 1, 1, 1);
            }

            int64_t W  = x->ne[0];
            int64_t H  = x->ne[1];
            int64_t wp = W / config.patch_size;
            int64_t hp = H / config.patch_size;

            pos_vec  = gen_2d_sincos_pos_embed(config.hidden_size, hp, wp,
                                               config.pos_embed_base_size, config.interpolation_scale);
            auto pos = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, config.hidden_size, wp * hp, 1);
            set_backend_tensor_data(pos, pos_vec.data());

            auto runner_ctx  = get_context();
            ggml_tensor* out = model.forward(&runner_ctx, x, timesteps, context, pos, context_mask);
            // learn_sigma: keep the noise prediction half of the output channels
            out = ggml_ext_slice(compute_ctx, out, 2, 0, config.in_channels);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& context_mask) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, context_mask);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            auto context     = tensor_or_empty(diffusion_params.context);
            auto context_msk = tensor_or_empty(diffusion_params.y);
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           context,
                           context_msk);
        }
    };
}  // namespace PixArt

#endif  // __SD_MODEL_DIFFUSION_PIXART_HPP__

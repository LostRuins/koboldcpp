#ifndef __SD_MODEL_DIFFUSION_IRIS_HPP__
#define __SD_MODEL_DIFFUSION_IRIS_HPP__

#include <array>
#include <stdexcept>
#include "model/diffusion/pid.hpp"

namespace Iris {
    struct IrisConfig {
        int64_t hidden_size    = 2560;
        int64_t depth          = 24;
        int64_t dual_depth     = 8;
        int64_t head_dim       = 128;
        int64_t num_heads      = 20;
        int64_t kv_heads       = 5;
        int64_t mlp_dim        = 6826;
        int64_t patch_size     = 16;
        int64_t text_dim       = 2560;
        int64_t text_len       = 300;
        int64_t text_layers    = 12;
        int64_t layer_heads    = 32;
        int64_t layer_mlp_dim  = 3328;
        int64_t pixel_dim      = 16;
        int64_t pixel_attn_dim = 1280;
        int64_t pixel_heads    = 10;
        int64_t pixel_depth    = 4;

        static IrisConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            IrisConfig config;
            auto find = [&](const std::string& name) -> const TensorStorage& {
                auto it = weights.find(prefix + "." + name);
                if (it == weights.end()) {
                    throw std::runtime_error("Iris-3B: missing weight " + prefix + "." + name);
                }
                return it->second;
            };
            config.hidden_size    = find("s_embedder.proj.weight").ne[1];
            config.patch_size     = static_cast<int64_t>(std::sqrt(find("s_embedder.proj.weight").ne[0] / 3));
            config.head_dim       = find("blocks.0.attn.q_norm_x.weight").ne[0];
            config.num_heads      = config.hidden_size / config.head_dim;
            config.kv_heads       = find("blocks.0.attn.k_proj_x.weight").ne[1] / config.head_dim;
            config.mlp_dim        = find("blocks.0.mlp_x.w1.weight").ne[1];
            config.text_dim       = find("y_embedder.refiner.proj.weight").ne[0];
            config.text_len       = find("y_pos_embedding").ne[1];
            config.text_layers    = find("y_embedder.layer_pool.weight").ne[0];
            config.layer_mlp_dim  = find("y_embedder.layer_blocks.0.mlp.0.weight").ne[1];
            config.pixel_dim      = find("pixel_embedder.proj.weight").ne[1];
            config.pixel_attn_dim = find("pixel_blocks.0.compress.weight").ne[1];
            config.pixel_heads    = config.pixel_attn_dim / find("pixel_blocks.0.attn.q_norm.weight").ne[0];
            config.depth = config.dual_depth = config.pixel_depth = 0;
            for (const auto& [name, tensor] : weights) {
                if (!starts_with(name, prefix + ".")) {
                    continue;
                }
                auto parts = split_string(name.substr(prefix.size() + 1), '.');
                if (parts.size() > 2 && parts[0] == "blocks") {
                    int64_t index = std::stoi(parts[1]);
                    config.depth  = std::max(config.depth, index + 1);
                    if (parts[2] == "adaln_img") {
                        config.dual_depth = std::max(config.dual_depth, index + 1);
                    }
                } else if (parts.size() > 2 && parts[0] == "pixel_blocks") {
                    config.pixel_depth = std::max(config.pixel_depth, int64_t(std::stoi(parts[1]) + 1));
                }
            }
            if (config.text_layers != 12 || config.text_dim != 2560 || config.text_len != 300 ||
                find("modulation_cores.adaln_img.weight").ne[1] != 6 * config.hidden_size ||
                find("pixel_blocks.0.adaln.weight").ne[1] != 4 * config.pixel_dim * config.patch_size * config.patch_size) {
                throw std::runtime_error("Iris-3B: unsupported checkpoint configuration");
            }
            LOG_VERBOSE("iris: hidden_size = %" PRId64 ", depth = %" PRId64 ", dual_depth = %" PRId64 ", heads = %" PRId64 "/%" PRId64 ", pixel_depth = %" PRId64,
                        config.hidden_size, config.depth, config.dual_depth, config.num_heads, config.kv_heads, config.pixel_depth);
            return config;
        }
    };

    struct SelfAttention : public GGMLBlock {
        int64_t dim;
        int64_t heads;
        bool qk_norm;

        SelfAttention(int64_t dim, int64_t heads, bool qk_norm = true)
            : dim(dim), heads(heads), qk_norm(qk_norm) {
            blocks["qkv"]  = std::make_shared<Linear>(dim, 3 * dim, false);
            blocks["proj"] = std::make_shared<Linear>(dim, dim, true);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(dim / heads, 1e-6f);
                blocks["k_norm"] = std::make_shared<RMSNorm>(dim / heads, 1e-6f);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe = nullptr, ggml_tensor* mask = nullptr) {
            auto c   = ctx->ggml_ctx;
            auto qkv = split_qkv(c, std::dynamic_pointer_cast<Linear>(blocks["qkv"])->forward(ctx, x));
            auto q   = ggml_reshape_4d(c, qkv[0], dim / heads, heads, x->ne[1], x->ne[2]);
            auto k   = ggml_reshape_4d(c, qkv[1], dim / heads, heads, x->ne[1], x->ne[2]);
            auto v   = ggml_reshape_4d(c, qkv[2], dim / heads, heads, x->ne[1], x->ne[2]);
            if (qk_norm) {
                q = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"])->forward(ctx, q);
                k = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"])->forward(ctx, k);
            }
            if (pe) {
                x = Rope::attention(ctx, q, k, v, pe, mask);
            } else {
                q = ggml_reshape_3d(c, q, dim, x->ne[1], x->ne[2]);
                k = ggml_reshape_3d(c, k, dim, x->ne[1], x->ne[2]);
                v = ggml_reshape_3d(c, v, dim, x->ne[1], x->ne[2]);
                x = ggml_ext_attention_ext(ctx, q, k, v, heads, mask, false, ctx->flash_attn_enabled);
            }
            return std::dynamic_pointer_cast<Linear>(blocks["proj"])->forward(ctx, x);
        }
    };

    struct TextBlock : public GGMLBlock {
        bool layerwise;

        TextBlock(int64_t dim, int64_t heads, int64_t mlp_dim, bool layerwise)
            : layerwise(layerwise) {
            blocks["norm1"] = std::make_shared<RMSNorm>(dim, 1e-6f);
            blocks["norm2"] = std::make_shared<RMSNorm>(dim, 1e-6f);
            blocks["attn"]  = std::make_shared<SelfAttention>(dim, heads, !layerwise);
            if (layerwise) {
                blocks["mlp.0"] = std::make_shared<Linear>(dim, mlp_dim, true);
                blocks["mlp.2"] = std::make_shared<Linear>(mlp_dim, dim, true);
            } else {
                blocks["mlp"] = std::make_shared<Pid::FeedForward>(dim, mlp_dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask = nullptr) {
            auto h = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"])->forward(ctx, x);
            h      = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"])->forward(ctx, h, nullptr, mask);
            x      = ggml_add(ctx->ggml_ctx, x, h);
            h      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"])->forward(ctx, x);
            if (layerwise) {
                h = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"])->forward(ctx, h);
                h = ggml_silu(ctx->ggml_ctx, h);
                h = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"])->forward(ctx, h);
            } else {
                h = std::dynamic_pointer_cast<Pid::FeedForward>(blocks["mlp"])->forward(ctx, h);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct TextEmbedder : public GGMLBlock {
        IrisConfig config;

        TextEmbedder(const IrisConfig& config)
            : config(config) {
            for (int i = 0; i < 2; ++i) {
                blocks["layer_blocks." + std::to_string(i)]   = std::make_shared<TextBlock>(config.text_dim, config.layer_heads, config.layer_mlp_dim, true);
                blocks["refiner.blocks." + std::to_string(i)] = std::make_shared<TextBlock>(config.hidden_size, config.num_heads, config.mlp_dim, false);
            }
            blocks["layer_pool"]   = std::make_shared<Linear>(config.text_layers, 1, true);
            blocks["refiner.proj"] = std::make_shared<Linear>(config.text_dim, config.hidden_size, true);
            blocks["refiner.norm"] = std::make_shared<RMSNorm>(config.hidden_size, 1e-6f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask) {
            auto c         = ctx->ggml_ctx;
            int64_t tokens = x->ne[1];
            int64_t batch  = x->ne[2];
            // The encoder concatenates selected layers within each token's feature dimension.
            x = ggml_reshape_3d(c, x, config.text_dim, config.text_layers, tokens * batch);
            for (int i = 0; i < 2; ++i) {
                x = std::dynamic_pointer_cast<TextBlock>(blocks["layer_blocks." + std::to_string(i)])->forward(ctx, x);
            }
            x = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));
            x = std::dynamic_pointer_cast<Linear>(blocks["layer_pool"])->forward(ctx, x);
            x = ggml_reshape_3d(c, x, config.text_dim, tokens, batch);
            x = std::dynamic_pointer_cast<Linear>(blocks["refiner.proj"])->forward(ctx, x);
            for (int i = 0; i < 2; ++i) {
                x = std::dynamic_pointer_cast<TextBlock>(blocks["refiner.blocks." + std::to_string(i)])->forward(ctx, x, mask);
            }
            return std::dynamic_pointer_cast<RMSNorm>(blocks["refiner.norm"])->forward(ctx, x);
        }
    };

    struct TrunkBlock : public GGMLBlock {
        IrisConfig config;
        bool dual;

        TrunkBlock(const IrisConfig& config, bool dual)
            : config(config), dual(dual) {
            int64_t d = config.hidden_size;
            for (const std::string stream : dual ? std::vector<std::string>{"x", "y"} : std::vector<std::string>{""}) {
                std::string suffix                                = stream.empty() ? "" : "_" + stream;
                std::string attn                                  = dual ? "attn." : "";
                blocks[attn + "q_proj" + suffix]                  = std::make_shared<Linear>(d, d, false);
                blocks[attn + "k_proj" + suffix]                  = std::make_shared<Linear>(d, config.kv_heads * config.head_dim, false);
                blocks[attn + "v_proj" + suffix]                  = std::make_shared<Linear>(d, config.kv_heads * config.head_dim, false);
                blocks[attn + "q_norm" + suffix]                  = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
                blocks[attn + "k_norm" + suffix]                  = std::make_shared<RMSNorm>(config.head_dim, 1e-6f);
                blocks[dual ? "attn.proj" + suffix : "attn_proj"] = std::make_shared<Linear>(d, d, true);
                blocks["attn_gate" + suffix]                      = std::make_shared<Linear>(d, d, false);
                blocks["norm" + suffix + "1"]                     = std::make_shared<RMSNorm>(d, 1e-6f);
                blocks["norm" + suffix + "2"]                     = std::make_shared<RMSNorm>(d, 1e-6f);
                blocks["attn_post_norm" + suffix]                 = std::make_shared<RMSNorm>(d, 1e-6f);
                blocks["mlp_post_norm" + suffix]                  = std::make_shared<RMSNorm>(d, 1e-6f);
                blocks["mlp" + suffix]                            = std::make_shared<Pid::FeedForward>(d, config.mlp_dim);
            }
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& = {}, const std::string = "") override {
            if (dual) {
                params["adaln_img.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * config.hidden_size);
                params["adaln_txt.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * config.hidden_size);
            } else {
                params["adaln.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6 * config.hidden_size);
            }
        }

        ggml_tensor* norm(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
            return std::dynamic_pointer_cast<RMSNorm>(blocks[name])->forward(ctx, x);
        }

        ggml_tensor* linear(GGMLRunnerContext* ctx, const std::string& name, ggml_tensor* x) {
            return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x);
        }

        std::array<ggml_tensor*, 3> project(GGMLRunnerContext* ctx, ggml_tensor* h, const std::string& suffix) {
            auto c        = ctx->ggml_ctx;
            std::string a = dual ? "attn." : "";
            auto q        = linear(ctx, a + "q_proj" + suffix, h);
            auto k        = linear(ctx, a + "k_proj" + suffix, h);
            auto v        = linear(ctx, a + "v_proj" + suffix, h);
            q             = ggml_reshape_4d(c, q, config.head_dim, config.num_heads, h->ne[1], h->ne[2]);
            k             = ggml_reshape_4d(c, k, config.head_dim, config.kv_heads, h->ne[1], h->ne[2]);
            v             = ggml_reshape_4d(c, v, config.head_dim, config.kv_heads, h->ne[1], h->ne[2]);
            return {norm(ctx, a + "q_norm" + suffix, q), norm(ctx, a + "k_norm" + suffix, k), v};
        }

        ggml_tensor* finish(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* h, ggml_tensor* out, const std::vector<ggml_tensor*>& mods, const std::string& suffix) {
            auto c = ctx->ggml_ctx;
            out    = ggml_mul(c, out, ggml_sigmoid(c, linear(ctx, "attn_gate" + suffix, h)));
            out    = linear(ctx, dual ? "attn.proj" + suffix : "attn_proj", out);
            out    = norm(ctx, "attn_post_norm" + suffix, out);
            x      = ggml_add(c, x, ggml_mul(c, out, mods[2]));
            h      = Pid::apply_adaln(c, norm(ctx, "norm" + suffix + "2", x), mods[3], mods[4]);
            h      = std::dynamic_pointer_cast<Pid::FeedForward>(blocks["mlp" + suffix])->forward(ctx, h);
            h      = norm(ctx, "mlp_post_norm" + suffix, h);
            return ggml_add(c, x, ggml_mul(c, h, mods[5]));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* y, ggml_tensor* img_mod, ggml_tensor* txt_mod, ggml_tensor* pe) {
            auto c     = ctx->ggml_ctx;
            int64_t nt = y->ne[1];
            auto mx    = ggml_ext_chunk(c, ggml_add(c, img_mod, params[dual ? "adaln_img.bias" : "adaln.bias"]), 6, 0);
            if (!dual) {
                auto tokens = ggml_concat(c, y, x, 1);
                auto h      = Pid::apply_adaln(c, norm(ctx, "norm1", tokens), mx[0], mx[1]);
                auto qkv    = project(ctx, h, "");
                auto out    = Rope::attention(ctx, qkv[0], qkv[1], qkv[2], pe, nullptr);
                tokens      = finish(ctx, tokens, h, out, mx, "");
                return {ggml_ext_slice(c, tokens, 1, nt, tokens->ne[1]), ggml_ext_slice(c, tokens, 1, 0, nt)};
            }
            auto my  = ggml_ext_chunk(c, ggml_add(c, txt_mod, params["adaln_txt.bias"]), 6, 0);
            auto hx  = Pid::apply_adaln(c, norm(ctx, "norm_x1", x), mx[0], mx[1]);
            auto hy  = Pid::apply_adaln(c, norm(ctx, "norm_y1", y), my[0], my[1]);
            auto qx  = project(ctx, hx, "_x");
            auto qy  = project(ctx, hy, "_y");
            auto out = Rope::attention(ctx, ggml_concat(c, qy[0], qx[0], 2), ggml_concat(c, qy[1], qx[1], 2),
                                       ggml_concat(c, qy[2], qx[2], 2), pe, nullptr);
            x        = finish(ctx, x, hx, ggml_ext_slice(c, out, 1, nt, out->ne[1]), mx, "_x");
            y        = finish(ctx, y, hy, ggml_ext_slice(c, out, 1, 0, nt), my, "_y");
            return {x, y};
        }
    };

    struct PixelBlock : public GGMLBlock {
        IrisConfig config;

        PixelBlock(const IrisConfig& config)
            : config(config) {
            auto d             = config.pixel_dim;
            auto flat          = d * config.patch_size * config.patch_size;
            blocks["norm1"]    = std::make_shared<RMSNorm>(d, 1e-6f);
            blocks["norm2"]    = std::make_shared<RMSNorm>(d, 1e-6f);
            blocks["compress"] = std::make_shared<Linear>(flat, config.pixel_attn_dim, true);
            blocks["expand"]   = std::make_shared<Linear>(config.pixel_attn_dim, flat, true);
            blocks["adaln"]    = std::make_shared<Linear>(config.hidden_size, 4 * flat, true);
            blocks["attn"]     = std::make_shared<SelfAttention>(config.pixel_attn_dim, config.pixel_heads);
            blocks["mlp.fc1"]  = std::make_shared<Linear>(d, d * 4, true);
            blocks["mlp.fc2"]  = std::make_shared<Linear>(d * 4, d, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* cond, ggml_tensor* pe) {
            auto c       = ctx->ggml_ctx;
            auto d       = config.pixel_dim;
            auto p2      = config.patch_size * config.patch_size;
            auto patches = cond->ne[1];
            auto batch   = cond->ne[2];
            auto mods    = std::dynamic_pointer_cast<Linear>(blocks["adaln"])->forward(ctx, cond);
            mods         = ggml_reshape_3d(c, mods, 4 * d, p2, patches * batch);
            auto m       = ggml_ext_chunk(c, mods, 4, 0);
            auto h       = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"])->forward(ctx, x);
            h            = ggml_reshape_3d(c, h, d * p2, patches, batch);
            h            = std::dynamic_pointer_cast<Linear>(blocks["compress"])->forward(ctx, h);
            h            = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"])->forward(ctx, h, pe);
            h            = std::dynamic_pointer_cast<Linear>(blocks["expand"])->forward(ctx, h);
            h            = ggml_reshape_3d(c, h, d, p2, patches * batch);
            x            = ggml_add(c, x, Pid::apply_adaln(c, h, m[1], m[0]));
            h            = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"])->forward(ctx, x);
            h            = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc1"])->forward(ctx, h);
            h            = ggml_gelu_erf(c, h);
            h            = std::dynamic_pointer_cast<Linear>(blocks["mlp.fc2"])->forward(ctx, h);
            return ggml_add(c, x, Pid::apply_adaln(c, h, m[3], m[2]));
        }
    };

    struct IrisModel : public GGMLBlock {
        IrisConfig config;

        IrisModel(const IrisConfig& config)
            : config(config) {
            blocks["s_embedder"]                 = std::make_shared<Pid::PatchTokenEmbedder>(3 * config.patch_size * config.patch_size, config.hidden_size);
            blocks["t_embedder"]                 = std::make_shared<Pid::PixelDiTTimestepEmbedder>(config.hidden_size);
            blocks["y_embedder"]                 = std::make_shared<TextEmbedder>(config);
            blocks["modulation_cores.adaln_img"] = std::make_shared<Linear>(config.hidden_size, 6 * config.hidden_size, true);
            blocks["modulation_cores.adaln_txt"] = std::make_shared<Linear>(config.hidden_size, 6 * config.hidden_size, true);
            for (int64_t i = 0; i < config.depth; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TrunkBlock>(config, i < config.dual_depth);
            }
            blocks["pixel_embedder"] = std::make_shared<Pid::PixelTokenEmbedder>(3, config.pixel_dim);
            for (int64_t i = 0; i < config.pixel_depth; ++i) {
                blocks["pixel_blocks." + std::to_string(i)] = std::make_shared<PixelBlock>(config);
            }
            blocks["final_layer"] = std::make_shared<Pid::FinalLayer>(config.pixel_dim, 3);
        }

        void init_params(ggml_context* ctx, const String2TensorStorage& = {}, const std::string = "") override {
            params["y_pos_embedding"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.hidden_size, config.text_len, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image, ggml_tensor* t, ggml_tensor* text, ggml_tensor* mask, ggml_tensor* joint_pe, ggml_tensor* pixel_pe, ggml_tensor* pixel_pos) {
            auto c    = ctx->ggml_ctx;
            int p     = static_cast<int>(config.patch_size);
            auto x    = DiT::patchify(c, image, p, p, true);
            x         = std::dynamic_pointer_cast<Pid::PatchTokenEmbedder>(blocks["s_embedder"])->forward(ctx, x);
            auto te   = std::dynamic_pointer_cast<Pid::PixelDiTTimestepEmbedder>(blocks["t_embedder"])->forward(ctx, t);
            te        = ggml_reshape_3d(c, te, config.hidden_size, 1, image->ne[3]);
            auto cond = ggml_silu(c, te);
            auto im   = std::dynamic_pointer_cast<Linear>(blocks["modulation_cores.adaln_img"])->forward(ctx, cond);
            auto tm   = std::dynamic_pointer_cast<Linear>(blocks["modulation_cores.adaln_txt"])->forward(ctx, cond);
            auto y    = std::dynamic_pointer_cast<TextEmbedder>(blocks["y_embedder"])->forward(ctx, text, mask);
            y         = ggml_add(c, y, params["y_pos_embedding"]);
            for (int64_t i = 0; i < config.depth; ++i) {
                auto block     = std::dynamic_pointer_cast<TrunkBlock>(blocks["blocks." + std::to_string(i)]);
                std::tie(x, y) = block->forward(ctx, x, y, im, tm, joint_pe);
                sd::ggml_graph_cut::mark_graph_cut(x, "iris.blocks." + std::to_string(i), "x");
                sd::ggml_graph_cut::mark_graph_cut(y, "iris.blocks." + std::to_string(i), "y");
            }
            cond = ggml_silu(c, ggml_add(c, x, te));
            x    = std::dynamic_pointer_cast<Pid::PixelTokenEmbedder>(blocks["pixel_embedder"])->forward(ctx, image, p, pixel_pos);
            for (int64_t i = 0; i < config.pixel_depth; ++i) {
                x = std::dynamic_pointer_cast<PixelBlock>(blocks["pixel_blocks." + std::to_string(i)])->forward(ctx, x, cond, pixel_pe);
                sd::ggml_graph_cut::mark_graph_cut(x, "iris.pixel_blocks." + std::to_string(i), "x");
            }
            x = std::dynamic_pointer_cast<Pid::FinalLayer>(blocks["final_layer"])->forward(ctx, x);
            x = ggml_reshape_3d(c, x, 3 * p * p, cond->ne[1], image->ne[3]);
            return DiT::unpatchify(c, x, image->ne[1] / p, image->ne[0] / p, p, p, false);
        }
    };

    inline std::vector<float> image_rope(int64_t height, int64_t width, int64_t dim) {
        std::vector<float> pe;
        pe.reserve(height * width * dim * 2);
        float step = 16.f / static_cast<float>(std::max<int64_t>(std::max(height, width) - 1, 1));
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t i = 0; i < dim / 4; ++i) {
                    float frequency = std::pow(10000.f, -4.f * static_cast<float>(i) / static_cast<float>(dim));
                    for (int64_t pos : {x, y}) {
                        float angle = static_cast<float>(pos) * step * frequency;
                        pe.insert(pe.end(), {std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle)});
                    }
                }
            }
        }
        return pe;
    }

    struct IrisRunner : public DiffusionModelRunner {
        IrisConfig config;
        IrisModel model;
        std::vector<float> joint_pe_data;
        std::vector<float> pixel_pe_data;
        std::vector<float> pixel_pos_data;
        std::vector<float> text_mask_data;

        IrisRunner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(IrisConfig::detect_from_weights(weights, prefix)),
              model(config) {
            model.init(params_ctx, weights, prefix);
        }

        std::string get_desc() override { return "Iris-3B"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            if (!inputs.x || !inputs.timesteps || !inputs.context || !inputs.y || inputs.y->empty()) {
                LOG_ERROR("Iris-3B requires text features and their padding mask");
                return {};
            }
            auto get_graph = [&]() {
                auto gf   = new_graph_custom(196608);
                auto x    = make_input(*inputs.x);
                auto t    = make_input(*inputs.timesteps);
                auto text = make_input(*inputs.context);
                int h     = static_cast<int>(x->ne[1]);
                int w     = static_cast<int>(x->ne[0]);
                int nt    = static_cast<int>(config.text_len);
                GGML_ASSERT(h % config.patch_size == 0 && w % config.patch_size == 0);
                GGML_ASSERT(text->ne[0] == config.text_dim * config.text_layers && text->ne[1] == nt);
                GGML_ASSERT(inputs.y->numel() == nt && x->ne[3] == 1);
                joint_pe_data = Pid::make_rope_1d(nt, static_cast<int>(config.head_dim), 10000.f);
                auto img      = image_rope(h / config.patch_size, w / config.patch_size, config.head_dim);
                joint_pe_data.insert(joint_pe_data.end(), img.begin(), img.end());
                pixel_pe_data  = image_rope(h / config.patch_size, w / config.patch_size, config.pixel_attn_dim / config.pixel_heads);
                pixel_pos_data = Pid::make_pixel_abs_pos(h, w, static_cast<int>(config.pixel_dim));
                text_mask_data.resize(nt * nt);
                for (int q = 0; q < nt; ++q) {
                    for (int k = 0; k < nt; ++k) {
                        text_mask_data[q * nt + k] = inputs.y->values()[k] != 0.f || q == k ? 0.f : -INFINITY;
                    }
                }
                auto pe   = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.head_dim / 2, nt + img.size() / (2 * config.head_dim));
                auto pp   = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.pixel_attn_dim / config.pixel_heads / 2, (h / config.patch_size) * (w / config.patch_size));
                auto pos  = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, config.pixel_dim, h * w, 1);
                auto mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, nt, nt);
                set_backend_tensor_data(pe, joint_pe_data.data());
                set_backend_tensor_data(pp, pixel_pe_data.data());
                set_backend_tensor_data(pos, pixel_pos_data.data());
                set_backend_tensor_data(mask, text_mask_data.data());
                auto ctx = get_context();
                auto out = model.forward(&ctx, x, t, text, mask, pe, pp, pos);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace Iris

#endif  // __SD_MODEL_DIFFUSION_IRIS_HPP__

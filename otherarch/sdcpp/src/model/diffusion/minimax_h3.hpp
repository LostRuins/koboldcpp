#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/ggml_graph_cut.h"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/model.hpp"

namespace MiniMaxH3 {

    constexpr int H3_GRAPH_SIZE          = 131072;
    constexpr float FRAME_RESCALE        = 5.f / 3.f;
    constexpr float VISUAL_COND_TIMESTEP = 0.999f;

    struct Config {
        int64_t hidden_size              = 5376;
        int64_t num_layers               = 50;
        int64_t token_refiner_num_layers = 2;
        int64_t num_attention_heads      = 56;
        int64_t attention_head_dim       = 128;
        int64_t ffn_hidden_size          = 14336;
        int64_t video_latent_channels    = 24;
        int64_t audio_latent_channels    = 32;
        int64_t text_dim                 = 5120;
        int64_t timestep_input_dim       = 256;
        int64_t time_embed_hidden_size   = 5376;
        int64_t time_embed_dim           = 2688;
        int64_t rope_inv_freq_len        = 16;
        int64_t adaln_curve_grid         = 0;
        int patch_t                      = 1;
        int patch_h                      = 2;
        int patch_w                      = 2;
        float norm_eps                   = 1e-5f;
        float qk_norm_eps                = 1e-5f;
        float final_norm_eps             = 1e-5f;

        bool uses_adaln_curves() const {
            return adaln_curve_grid > 0;
        }

        static int64_t count_blocks(const String2TensorStorage& tensors,
                                    const std::string& prefix) {
            std::set<int> indices;
            for (const auto& [name, _] : tensors) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t begin = prefix.size();
                size_t end   = name.find('.', begin);
                if (end != std::string::npos) {
                    indices.insert(std::atoi(name.substr(begin, end - begin).c_str()));
                }
            }
            return static_cast<int64_t>(indices.size());
        }

        static Config detect_from_weights(const String2TensorStorage& tensors,
                                          const std::string& prefix) {
            Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = tensors.find(prefix + "." + suffix);
                return it == tensors.end() ? nullptr : &it->second;
            };

            if (const auto* weight = find("video_patch_proj.weight")) {
                config.video_latent_channels = weight->ne[0] / 4;
                config.hidden_size           = weight->ne[1];
            }
            if (const auto* weight = find("audio_patch_proj.weight")) {
                config.audio_latent_channels = weight->ne[0];
            }
            config.num_layers               = count_blocks(tensors, prefix + ".blocks.");
            config.token_refiner_num_layers = count_blocks(tensors, prefix + ".token_refiner.blocks.");
            if (const auto* weight = find("blocks.0.attn.q_norm.weight")) {
                config.attention_head_dim = weight->ne[0];
            }
            if (const auto* weight = find("blocks.0.attn.qkv_proj.weight")) {
                config.num_attention_heads = weight->ne[1] / (3 * config.attention_head_dim);
            }
            if (const auto* weight = find("blocks.0.mlp.fc1.weight")) {
                config.ffn_hidden_size = weight->ne[1] / 2;
            }
            if (const auto* weight = find("condition_proj.weight")) {
                config.text_dim = weight->ne[0];
            }
            if (const auto* table = find("adaln_t_table")) {
                config.time_embed_dim   = table->ne[0];
                config.adaln_curve_grid = table->ne[1];
            } else {
                if (const auto* weight = find("time_embedder.proj_in.weight")) {
                    config.timestep_input_dim     = weight->ne[0];
                    config.time_embed_hidden_size = weight->ne[1];
                }
                if (const auto* weight = find("time_embedder.proj_out.weight")) {
                    config.time_embed_dim = weight->ne[1];
                }
            }
            if (const auto* inv_freq = find("rope.inv_freq")) {
                config.rope_inv_freq_len = inv_freq->ne[0];
            }

            LOG_DEBUG("minimax_h3: layers=%" PRId64 ", hidden=%" PRId64 ", heads=%" PRId64
                      ", head_dim=%" PRId64 ", ffn=%" PRId64 ", adaln_curve=%" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.attention_head_dim,
                      config.ffn_hidden_size,
                      config.adaln_curve_grid);
            return config;
        }
    };

    static float time_shift_sigma(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        return to_shift * base / (1.f + (to_shift - 1.f) * base);
    }

    static float time_shift_slope(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        float a    = 1.f + (from_shift - 1.f) * base;
        float b    = 1.f + (to_shift - 1.f) * base;
        return to_shift * a * a / (from_shift * b * b);
    }

    struct TimeEmbedder : public GGMLBlock {
        TimeEmbedder(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
            blocks["proj_in"]  = std::make_shared<Linear>(input_dim, hidden_dim, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(hidden_dim, output_dim, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            return proj_out->forward(ctx, ggml_silu(ctx->ggml_ctx, proj_in->forward(ctx, x)));
        }
    };

    struct MLP : public UnaryBlock {
        MLP(int64_t hidden_size,
            int64_t ffn_hidden_size) {
            blocks["fc1"] = std::make_shared<Linear>(hidden_size, ffn_hidden_size * 2, false, false, true, 1.f / 128.f);
            blocks["fc2"] = std::make_shared<Linear>(ffn_hidden_size, hidden_size, false, false, true, 1.f / 128.f);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
            auto uv  = ggml_ext_chunk(ctx->ggml_ctx, fc1->forward(ctx, x), 2, 0);
            return fc2->forward(ctx, ggml_mul(ctx->ggml_ctx,
                                              ggml_silu(ctx->ggml_ctx, uv[0]),
                                              uv[1]));
        }
    };

    static ggml_tensor* attention_layout(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
    }

    static ggml_tensor* apply_partial_rope(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* pe) {
        int64_t rot_dim = pe->ne[2] * 2;
        GGML_ASSERT(rot_dim <= x->ne[0]);
        auto rotated = Rope::apply_rope(ctx,
                                        ggml_ext_slice(ctx, x, 0, 0, rot_dim),
                                        pe,
                                        false);
        if (rot_dim == x->ne[0]) {
            return rotated;
        }
        auto tail = attention_layout(ctx, ggml_ext_slice(ctx, x, 0, rot_dim, x->ne[0]));
        return ggml_concat(ctx, rotated, tail, 0);
    }

    struct Attention : public GGMLBlock {
        int64_t heads;
        int64_t head_dim;

        Attention(int64_t hidden_size,
                  int64_t heads,
                  int64_t head_dim,
                  float eps)
            : heads(heads), head_dim(head_dim) {
            int64_t inner      = heads * head_dim;
            blocks["qkv_proj"] = std::make_shared<Linear>(hidden_size, inner * 3, false);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["out_proj"] = std::make_shared<Linear>(inner, hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe = nullptr) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            int64_t sequence = x->ne[1];
            int64_t batch    = x->ne[2] * x->ne[3];
            auto qkv         = ggml_ext_chunk(ctx->ggml_ctx, qkv_proj->forward(ctx, x), 3, 0);
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv[0], head_dim, heads, sequence, batch);
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv[1], head_dim, heads, sequence, batch);
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv[2], head_dim, heads, sequence, batch);
            q                = q_norm->forward(ctx, q);
            k                = k_norm->forward(ctx, k);
            if (pe != nullptr) {
                q = apply_partial_rope(ctx->ggml_ctx, q, pe);
                k = apply_partial_rope(ctx->ggml_ctx, k, pe);
            } else {
                q = attention_layout(ctx->ggml_ctx, q);
                k = attention_layout(ctx->ggml_ctx, k);
            }
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              static_cast<int>(heads),
                                              nullptr,
                                              true,
                                              ctx->flash_attn_enabled,
                                              1.f / 128.f);
            return out_proj->forward(ctx, out);
        }
    };

    struct TokenRefinerBlock : public GGMLBlock {
        TokenRefinerBlock(const Config& config) {
            blocks["norm1"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]  = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps);
            blocks["mlp"]   = std::make_shared<MLP>(config.hidden_size, config.ffn_hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            x          = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, norm1->forward(ctx, x)));
            return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x)));
        }
    };

    struct TokenRefiner : public GGMLBlock {
        int64_t num_layers;

        explicit TokenRefiner(const Config& config)
            : num_layers(config.token_refiner_num_layers) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TokenRefinerBlock>(config);
            }
            blocks["final_norm"] = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TokenRefinerBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            return std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm"])->forward(ctx, x);
        }
    };

    struct AdaLayerNormModulation : public GGMLBlock {
        int64_t hidden_size;
        int expand;
        int modalities;
        bool apply_silu;

        AdaLayerNormModulation(int64_t time_dim,
                               int64_t hidden_size,
                               int expand,
                               int modalities,
                               bool apply_silu,
                               bool force_f32)
            : hidden_size(hidden_size),
              expand(expand),
              modalities(modalities),
              apply_silu(apply_silu) {
            blocks["linear"] = std::make_shared<Linear>(time_dim,
                                                        hidden_size * expand * modalities,
                                                        true,
                                                        force_f32);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            if (apply_silu) {
                t_emb = ggml_silu(ctx->ggml_ctx, t_emb);
            }
            return std::dynamic_pointer_cast<Linear>(blocks["linear"])->forward(ctx, t_emb);
        }
    };

    struct TokenModulationSpan {
        int64_t start;
        int64_t end;
        int64_t modulation_row;
    };

    enum class SequenceKind {
        TEXT,
        CONDITION_VIDEO,
        CONDITION_AUDIO,
        TARGET_AUDIO,
        TARGET_VIDEO,
    };

    struct SequenceSegment {
        int64_t start;
        int64_t end;
        SequenceKind kind;
        int32_t source_index = -1;
    };

    static std::vector<ggml_tensor*> modulation_row(ggml_context* ctx,
                                                    ggml_tensor* projection,
                                                    int64_t hidden_size,
                                                    int expand,
                                                    int modalities,
                                                    int64_t row) {
        int64_t timestep_rows = projection->ne[1];
        auto reshaped         = ggml_reshape_2d(ctx,
                                                projection,
                                                hidden_size * expand,
                                                timestep_rows * modalities);
        auto selected         = ggml_ext_slice(ctx, reshaped, 1, row, row + 1);
        return ggml_ext_chunk(ctx, selected, expand, 0);
    }

    static ggml_tensor* modulate_segments(ggml_context* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* projection,
                                          const std::vector<TokenModulationSpan>& segments,
                                          int64_t hidden_size,
                                          int expand,
                                          int modalities,
                                          int shift_index,
                                          int scale_index) {
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx,
                                       projection,
                                       hidden_size,
                                       expand,
                                       modalities,
                                       segment.modulation_row);
            auto part = ggml_ext_slice(ctx, x, 1, segment.start, segment.end);
            part      = ggml_add(ctx,
                                 ggml_add(ctx, part, ggml_mul(ctx, part, mods[scale_index])),
                                 mods[shift_index]);
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    static ggml_tensor* gated_residual_segments(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* update,
                                                ggml_tensor* projection,
                                                const std::vector<TokenModulationSpan>& segments,
                                                int64_t hidden_size,
                                                int gate_index) {
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx, projection, hidden_size, 6, 3, segment.modulation_row);
            auto base = ggml_ext_slice(ctx, x, 1, segment.start, segment.end);
            auto add  = ggml_ext_slice(ctx, update, 1, segment.start, segment.end);
            auto part = ggml_add(ctx, base, ggml_mul(ctx, add, mods[gate_index]));
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    struct TransformerBlock : public GGMLBlock {
        Config config;

        explicit TransformerBlock(const Config& config)
            : config(config) {
            blocks["norm1"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]       = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps);
            blocks["mlp"]        = std::make_shared<MLP>(config.hidden_size,
                                                  config.ffn_hidden_size);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            6,
                                                                            3,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_emb,
                             const std::vector<TokenModulationSpan>& segments,
                             ggml_tensor* pe) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaln = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto mods  = adaln->forward(ctx, t_emb);

            auto h = modulate_segments(ctx->ggml_ctx,
                                       norm1->forward(ctx, x),
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       0,
                                       1);
            x      = gated_residual_segments(ctx->ggml_ctx,
                                             x,
                                             attn->forward(ctx, h, pe),
                                             mods,
                                             segments,
                                             config.hidden_size,
                                             2);
            h      = modulate_segments(ctx->ggml_ctx,
                                       norm2->forward(ctx, x),
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       3,
                                       4);
            return gated_residual_segments(ctx->ggml_ctx,
                                           x,
                                           mlp->forward(ctx, h),
                                           mods,
                                           segments,
                                           config.hidden_size,
                                           5);
        }
    };

    struct FinalLayer : public GGMLBlock {
        Config config;

        explicit FinalLayer(const Config& config)
            : config(config) {
            int64_t video_dim    = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["norm"]       = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            2,
                                                                            1,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
            blocks["video_out"]  = std::make_shared<Linear>(config.hidden_size, video_dim, true, true);
            blocks["audio_out"]  = std::make_shared<Linear>(config.hidden_size, config.audio_latent_channels, true, true);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* t_emb,
                                                      const TokenModulationSpan& video,
                                                      const TokenModulationSpan& audio) {
            auto norm      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto adaln     = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto video_out = std::dynamic_pointer_cast<Linear>(blocks["video_out"]);
            auto audio_out = std::dynamic_pointer_cast<Linear>(blocks["audio_out"]);
            auto mods      = adaln->forward(ctx, t_emb);
            auto apply     = [&](const TokenModulationSpan& segment) {
                auto row   = modulation_row(ctx->ggml_ctx, mods, config.hidden_size, 2, 1, segment.modulation_row);
                auto value = norm->forward(ctx, ggml_ext_slice(ctx->ggml_ctx, x, 1, segment.start, segment.end));
                return ggml_add(ctx->ggml_ctx,
                                    ggml_add(ctx->ggml_ctx, value, ggml_mul(ctx->ggml_ctx, value, row[1])),
                                    row[0]);
            };
            return {video_out->forward(ctx, apply(video)),
                    audio_out->forward(ctx, apply(audio))};
        }
    };

    struct MiniMaxH3Transformer3DModel : public GGMLBlock {
        Config config;

        explicit MiniMaxH3Transformer3DModel(const Config& config)
            : config(config) {
            int64_t video_dim          = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["video_patch_proj"] = std::make_shared<Linear>(video_dim, config.hidden_size, true, true);
            blocks["audio_patch_proj"] = std::make_shared<Linear>(config.audio_latent_channels, config.hidden_size, true, true);
            blocks["condition_proj"]   = std::make_shared<Linear>(config.text_dim, config.hidden_size, true);
            if (!config.uses_adaln_curves()) {
                blocks["time_embedder"] = std::make_shared<TimeEmbedder>(config.timestep_input_dim,
                                                                         config.time_embed_hidden_size,
                                                                         config.time_embed_dim);
            }
            blocks["token_refiner"] = std::make_shared<TokenRefiner>(config);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(config);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(config);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensors = {},
                         const std::string prefix            = "") override {
            GGMLBlock::init_params(ctx, tensors, prefix);
            params["rope.inv_freq"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.rope_inv_freq_len);
            if (config.uses_adaln_curves()) {
                params["adaln_t_table"] = ggml_new_tensor_2d(ctx,
                                                             GGML_TYPE_F32,
                                                             config.time_embed_dim,
                                                             config.adaln_curve_grid);
            }
        }

        ggml_tensor* refine_context(GGMLRunnerContext* ctx, ggml_tensor* context) {
            if (context->ne[0] == config.hidden_size) {
                return context;
            }
            GGML_ASSERT(context->ne[0] == config.text_dim);
            auto condition_proj = std::dynamic_pointer_cast<Linear>(blocks["condition_proj"]);
            auto token_refiner  = std::dynamic_pointer_cast<TokenRefiner>(blocks["token_refiner"]);
            return token_refiner->forward(ctx, condition_proj->forward(ctx, context));
        }

        ggml_tensor* time_embedding(GGMLRunnerContext* ctx,
                                    ggml_tensor* timestep_features,
                                    ggml_tensor* curve_indices,
                                    ggml_tensor* curve_upper_indices,
                                    ggml_tensor* curve_fractions) {
            if (!config.uses_adaln_curves()) {
                return std::dynamic_pointer_cast<TimeEmbedder>(blocks["time_embedder"])->forward(ctx, timestep_features);
            }
            auto lower = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_indices);
            auto upper = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_upper_indices);
            return ggml_add(ctx->ggml_ctx,
                            lower,
                            ggml_mul(ctx->ggml_ctx,
                                     ggml_sub(ctx->ggml_ctx, upper, lower),
                                     curve_fractions));
        }

        ggml_tensor* build_rope(GGMLRunnerContext* ctx,
                                ggml_tensor* position_ids) {
            auto inv            = ggml_reshape_2d(ctx->ggml_ctx,
                                                  params["rope.inv_freq"],
                                                  config.rope_inv_freq_len,
                                                  1);
            ggml_tensor* angles = nullptr;
            for (int axis = 0; axis < 3; ++axis) {
                auto pos          = ggml_ext_slice(ctx->ggml_ctx, position_ids, 0, axis, axis + 1);
                auto expanded_inv = ggml_repeat_4d(ctx->ggml_ctx,
                                                   inv,
                                                   inv->ne[0],
                                                   pos->ne[1],
                                                   1,
                                                   1);
                auto a            = ggml_mul(ctx->ggml_ctx, expanded_inv, pos);
                angles            = angles == nullptr ? a : ggml_concat(ctx->ggml_ctx, angles, a, 0);
            }
            auto c  = ggml_reshape_4d(ctx->ggml_ctx, ggml_cos(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto s  = ggml_reshape_4d(ctx->ggml_ctx, ggml_sin(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto ns = ggml_neg(ctx->ggml_ctx, s);
            auto pe = ggml_concat(ctx->ggml_ctx, c, ns, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, s, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, c, 0);
            return ggml_reshape_4d(ctx->ggml_ctx, pe, 2, 2, angles->ne[0], angles->ne[1]);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* video,
                                                      ggml_tensor* audio,
                                                      ggml_tensor* context,
                                                      const std::vector<ggml_tensor*>& condition_videos,
                                                      const std::vector<ggml_tensor*>& condition_audios,
                                                      ggml_tensor* position_ids,
                                                      ggml_tensor* timestep_features,
                                                      ggml_tensor* curve_indices,
                                                      ggml_tensor* curve_upper_indices,
                                                      ggml_tensor* curve_fractions,
                                                      const std::vector<TokenModulationSpan>& segments,
                                                      const std::vector<SequenceSegment>& sequence_segments,
                                                      const TokenModulationSpan& video_segment,
                                                      const TokenModulationSpan& audio_segment,
                                                      float audio_slope) {
            auto video_proj = std::dynamic_pointer_cast<Linear>(blocks["video_patch_proj"]);
            auto audio_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patch_proj"]);

            std::vector<std::pair<int64_t, int64_t>> condition_video_ranges;
            ggml_tensor* video_rows = nullptr;
            int64_t video_offset    = 0;
            for (auto condition : condition_videos) {
                auto rows = DiT::patchify_3d(ctx->ggml_ctx,
                                             condition,
                                             config.patch_t,
                                             config.patch_h,
                                             config.patch_w,
                                             1,
                                             true);
                condition_video_ranges.push_back({video_offset, video_offset + rows->ne[1]});
                video_offset += rows->ne[1];
                video_rows = video_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, video_rows, rows, 1);
            }
            auto target_video_rows                         = DiT::patchify_3d(ctx->ggml_ctx,
                                                                              video,
                                                                              config.patch_t,
                                                                              config.patch_h,
                                                                              config.patch_w,
                                                                              1,
                                                                              true);
            std::pair<int64_t, int64_t> target_video_range = {video_offset, video_offset + target_video_rows->ne[1]};
            video_rows                                     = video_rows == nullptr ? target_video_rows
                                                                                   : ggml_concat(ctx->ggml_ctx, video_rows, target_video_rows, 1);
            auto video_embeds                              = video_proj->forward(ctx, video_rows);

            auto pack_audio_rows = [&](ggml_tensor* value) {
                value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, value, 2, 0, 1, 3));
                return ggml_reshape_3d(ctx->ggml_ctx,
                                       value,
                                       value->ne[0],
                                       value->ne[1] * value->ne[2],
                                       value->ne[3]);
            };
            std::vector<std::pair<int64_t, int64_t>> condition_audio_ranges;
            ggml_tensor* audio_rows = nullptr;
            int64_t audio_offset    = 0;
            for (auto condition : condition_audios) {
                auto rows = pack_audio_rows(condition);
                condition_audio_ranges.push_back({audio_offset, audio_offset + rows->ne[1]});
                audio_offset += rows->ne[1];
                audio_rows = audio_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, audio_rows, rows, 1);
            }
            audio                                          = pack_audio_rows(audio);
            std::pair<int64_t, int64_t> target_audio_range = {audio_offset, audio_offset + audio->ne[1]};
            audio_rows                                     = audio_rows == nullptr ? audio : ggml_concat(ctx->ggml_ctx, audio_rows, audio, 1);
            auto audio_embeds                              = audio_proj->forward(ctx, audio_rows);
            context                                        = refine_context(ctx, context);

            ggml_tensor* h = nullptr;
            auto append    = [&](ggml_tensor* value) {
                h = h == nullptr ? value : ggml_concat(ctx->ggml_ctx, h, value, 1);
            };
            for (const auto& sequence : sequence_segments) {
                if (sequence.kind == SequenceKind::TEXT) {
                    append(context);
                } else if (sequence.kind == SequenceKind::CONDITION_VIDEO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_video_ranges.size()));
                    auto range = condition_video_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, video_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::CONDITION_AUDIO) {
                    GGML_ASSERT(sequence.source_index >= 0 &&
                                sequence.source_index < static_cast<int32_t>(condition_audio_ranges.size()));
                    auto range = condition_audio_ranges[static_cast<size_t>(sequence.source_index)];
                    append(ggml_ext_slice(ctx->ggml_ctx, audio_embeds, 1, range.first, range.second));
                } else if (sequence.kind == SequenceKind::TARGET_AUDIO) {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          audio_embeds,
                                          1,
                                          target_audio_range.first,
                                          target_audio_range.second));
                } else {
                    append(ggml_ext_slice(ctx->ggml_ctx,
                                          video_embeds,
                                          1,
                                          target_video_range.first,
                                          target_video_range.second));
                }
            }
            GGML_ASSERT(h != nullptr);

            auto t_emb = time_embedding(ctx,
                                        timestep_features,
                                        curve_indices,
                                        curve_upper_indices,
                                        curve_fractions);
            auto pe    = build_rope(ctx, position_ids);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                h          = block->forward(ctx, h, t_emb, segments, pe);
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   "minimax_h3.blocks." + std::to_string(i),
                                                   "hidden_states");
            }

            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            auto output      = final_layer->forward(ctx, h, t_emb, video_segment, audio_segment);
            auto video_out   = DiT::unpatchify_3d(ctx->ggml_ctx,
                                                  output.first,
                                                  video->ne[2] / config.patch_t,
                                                  video->ne[1] / config.patch_h,
                                                  video->ne[0] / config.patch_w,
                                                  config.patch_t,
                                                  config.patch_h,
                                                  config.patch_w,
                                                  true);
            auto audio_out   = ggml_reshape_4d(ctx->ggml_ctx,
                                               output.second,
                                               config.audio_latent_channels,
                                               audio->ne[1] / 2,
                                               2,
                                               audio->ne[2]);
            audio_out        = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, audio_out, 1, 2, 0, 3));
            video_out        = ggml_ext_scale(ctx->ggml_ctx, video_out, -1.f);
            audio_out        = ggml_ext_scale(ctx->ggml_ctx, audio_out, -audio_slope);
            return {video_out, audio_out};
        }
    };

    struct PackedSequenceLayout {
        std::vector<float> positions;
        std::vector<TokenModulationSpan> segments;
        std::vector<SequenceSegment> sequence_segments;
        TokenModulationSpan video_segment{};
        TokenModulationSpan audio_segment{};
        std::vector<float> timesteps;
    };

    static float video_span(int64_t frame) {
        static const int spans[5] = {1, 4, 4, 4, 4};
        return FRAME_RESCALE * spans[frame % 5];
    }

    static std::vector<float> spatial_axis(int64_t dim,
                                           float sqrt_area) {
        int64_t count = dim / 2;
        float ratio   = static_cast<float>(dim) / sqrt_area;
        std::vector<float> result(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            result[static_cast<size_t>(i)] =
                (static_cast<float>(i) * (ratio / count) + (1.f - ratio) * 0.5f) * 32.f;
        }
        return result;
    }

    static int find_or_add_timestep(std::vector<float>* values, float value) {
        auto it = std::find(values->begin(), values->end(), value);
        if (it != values->end()) {
            return static_cast<int>(it - values->begin());
        }
        values->push_back(value);
        return static_cast<int>(values->size() - 1);
    }

    static PackedSequenceLayout build_layout(int64_t text_len,
                                             int64_t latent_t,
                                             int64_t latent_h,
                                             int64_t latent_w,
                                             int64_t audio_t,
                                             const std::vector<sd::Tensor<float>>& condition_videos,
                                             const std::vector<sd::Tensor<float>>& condition_audios,
                                             const sd::Tensor<int32_t>& keyframe_indices,
                                             const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                             const sd::Tensor<int32_t>& text_tags,
                                             float video_t,
                                             float audio_timestep) {
        PackedSequenceLayout layout;
        float sqrt_area    = std::sqrt(static_cast<float>(latent_h * latent_w));
        auto h_axis        = spatial_axis(latent_h, sqrt_area);
        auto w_axis        = spatial_axis(latent_w, sqrt_area);
        int64_t frame_rows = static_cast<int64_t>(h_axis.size() * w_axis.size());
        int64_t row        = 0;

        auto append_position = [&](float t, float h, float w) {
            layout.positions.push_back(t);
            layout.positions.push_back(h);
            layout.positions.push_back(w);
        };
        for (int64_t i = 0; i < text_len; ++i) {
            append_position(static_cast<float>(i), 0.f, 0.f);
        }
        layout.sequence_segments.push_back({0, text_len, SequenceKind::TEXT});

        int video_time_row           = find_or_add_timestep(&layout.timesteps, video_t);
        int audio_time_row           = find_or_add_timestep(&layout.timesteps, audio_timestep);
        int condition_time_row       = find_or_add_timestep(&layout.timesteps,
                                                            std::max(video_t, VISUAL_COND_TIMESTEP));
        int audio_condition_time_row = find_or_add_timestep(&layout.timesteps,
                                                            std::max(audio_timestep, 1.f));

        int64_t run_start = 0;
        int current_tag   = text_tags.empty() ? 1 : text_tags[0];
        for (int64_t i = 1; i <= text_len; ++i) {
            int tag = i < text_len && !text_tags.empty() ? text_tags[i] : -1;
            if (i == text_len || tag != current_tag) {
                layout.segments.push_back({run_start,
                                           i,
                                           video_time_row * 3 + current_tag});
                run_start   = i;
                current_tag = tag;
            }
        }
        row = text_len;

        auto condition_spatial_axes = [&](const sd::Tensor<float>& condition) {
            float area = std::sqrt(static_cast<float>(condition.shape()[0] * condition.shape()[1]));
            return std::make_pair(spatial_axis(condition.shape()[1], area),
                                  spatial_axis(condition.shape()[0], area));
        };
        auto append_video_positions = [&](const sd::Tensor<float>& condition,
                                          float cursor) {
            auto axes = condition_spatial_axes(condition);
            for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                for (float h : axes.first) {
                    for (float w : axes.second) {
                        append_position(cursor, h, w);
                    }
                }
                cursor += video_span(t);
            }
            return cursor;
        };
        auto append_audio_positions = [&](int64_t length,
                                          float cursor,
                                          float w_low,
                                          float w_high) {
            for (int channel = 0; channel < 2; ++channel) {
                float w = channel == 0 ? w_low : w_high;
                for (int64_t t = 0; t < length; ++t) {
                    append_position(cursor + static_cast<float>(t), 0.f, w);
                }
            }
        };

        float cursor = static_cast<float>(text_len);
        if (reference_blocks.empty()) {
            float video_duration = 0.f;
            for (int64_t t = 0; t < latent_t; ++t) {
                video_duration += video_span(t);
            }
            for (size_t index = 0; index < condition_videos.size(); ++index) {
                const auto& condition = condition_videos[index];
                auto axes             = condition_spatial_axes(condition);
                int64_t count         = condition.shape()[2] *
                                static_cast<int64_t>(axes.first.size() * axes.second.size());
                bool is_first    = keyframe_indices.empty() || keyframe_indices[static_cast<int64_t>(index)] == 0;
                float keyframe_t = is_first ? static_cast<float>(text_len)
                                            : static_cast<float>(text_len) + video_duration - FRAME_RESCALE;
                for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                    for (float h : axes.first) {
                        for (float w : axes.second) {
                            append_position(keyframe_t, h, w);
                        }
                    }
                }
                layout.sequence_segments.push_back({row,
                                                    row + count,
                                                    SequenceKind::CONDITION_VIDEO,
                                                    static_cast<int32_t>(index)});
                layout.segments.push_back({row, row + count, condition_time_row * 3});
                row += count;
            }
        } else {
            for (const auto& block : reference_blocks) {
                const sd::Tensor<float>* ref_video = nullptr;
                const sd::Tensor<float>* ref_audio = nullptr;
                if (block.video_index >= 0) {
                    GGML_ASSERT(block.video_index < static_cast<int32_t>(condition_videos.size()));
                    ref_video = &condition_videos[static_cast<size_t>(block.video_index)];
                }
                if (block.audio_index >= 0) {
                    GGML_ASSERT(block.audio_index < static_cast<int32_t>(condition_audios.size()));
                    ref_audio = &condition_audios[static_cast<size_t>(block.audio_index)];
                }

                float block_end = cursor;
                if (block.kind == MiniMaxH3ReferenceKind::AUDIO ||
                    block.kind == MiniMaxH3ReferenceKind::VIDEO_AUDIO) {
                    GGML_ASSERT(ref_audio != nullptr);
                    float w_low  = w_axis.front();
                    float w_high = w_axis.back();
                    if (ref_video != nullptr) {
                        auto axes = condition_spatial_axes(*ref_video);
                        w_low     = axes.second.front();
                        w_high    = axes.second.back();
                    }
                    int64_t count = ref_audio->shape()[0] * 2;
                    append_audio_positions(ref_audio->shape()[0], cursor, w_low, w_high);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_AUDIO,
                                                        block.audio_index});
                    layout.segments.push_back({row,
                                               row + count,
                                               audio_condition_time_row * 3 + 2});
                    row += count;
                    block_end = std::max(block_end, cursor + static_cast<float>(ref_audio->shape()[0]));
                }

                if (block.kind != MiniMaxH3ReferenceKind::AUDIO) {
                    GGML_ASSERT(ref_video != nullptr);
                    auto axes     = condition_spatial_axes(*ref_video);
                    int64_t count = ref_video->shape()[2] *
                                    static_cast<int64_t>(axes.first.size() * axes.second.size());
                    float video_end = append_video_positions(*ref_video, cursor);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_VIDEO,
                                                        block.video_index});
                    layout.segments.push_back({row, row + count, condition_time_row * 3});
                    row += count;
                    block_end = block.kind == MiniMaxH3ReferenceKind::IMAGE
                                    ? std::max(block_end, cursor + 1.f)
                                    : std::max(block_end, video_end);
                }
                cursor = block_end;
            }
        }

        int64_t audio_start = row;
        append_audio_positions(audio_t, cursor, w_axis.front(), w_axis.back());
        layout.audio_segment = {audio_start, row + audio_t * 2, audio_time_row};
        layout.sequence_segments.push_back({audio_start,
                                            row + audio_t * 2,
                                            SequenceKind::TARGET_AUDIO});
        layout.segments.push_back({audio_start,
                                   row + audio_t * 2,
                                   audio_time_row * 3 + 2});
        row += audio_t * 2;

        int64_t video_start = row;
        for (int64_t t = 0; t < latent_t; ++t) {
            for (float h : h_axis) {
                for (float w : w_axis) {
                    append_position(cursor, h, w);
                }
            }
            cursor += video_span(t);
        }
        int64_t video_rows   = latent_t * frame_rows;
        layout.video_segment = {video_start, video_start + video_rows, video_time_row};
        layout.sequence_segments.push_back({video_start,
                                            video_start + video_rows,
                                            SequenceKind::TARGET_VIDEO});
        layout.segments.push_back({video_start,
                                   video_start + video_rows,
                                   video_time_row * 3});
        return layout;
    }

    struct MiniMaxH3Runner : public DiffusionModelRunner {
        Config config;
        MiniMaxH3Transformer3DModel model;
        sd::Tensor<float> video_input_cache;
        sd::Tensor<float> audio_input_cache;
        sd::Tensor<float> position_input_cache;
        sd::Tensor<float> timestep_feature_input_cache;
        sd::Tensor<int32_t> curve_index_input_cache;
        sd::Tensor<int32_t> curve_upper_index_input_cache;
        sd::Tensor<float> curve_fraction_input_cache;

        MiniMaxH3Runner(ggml_backend_t backend,
                        const String2TensorStorage& tensors,
                        const std::string& prefix                           = "model.diffusion_model",
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Config::detect_from_weights(tensors, prefix)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override {
            return "minimax_h3";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> split_av_latents(const sd::Tensor<float>& packed,
                                                                         int audio_length) const {
            GGML_ASSERT(packed.dim() == 4 || packed.dim() == 5);
            int64_t spatial      = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
            int64_t video_values = spatial * config.video_latent_channels;
            sd::Tensor<float> video({packed.shape()[0],
                                     packed.shape()[1],
                                     packed.shape()[2],
                                     config.video_latent_channels,
                                     1});
            std::copy_n(packed.data(), static_cast<size_t>(video_values), video.data());
            if (audio_length <= 0) {
                return {video, {}};
            }
            int64_t audio_values = audio_length * 2 * config.audio_latent_channels;
            GGML_ASSERT(packed.numel() >= video_values + audio_values);
            sd::Tensor<float> audio({audio_length, 2, config.audio_latent_channels, 1});
            std::copy_n(packed.data() + video_values,
                        static_cast<size_t>(audio_values),
                        audio.data());
            return {video, audio};
        }

        ggml_tensor* merge_av_latents(ggml_context* ctx,
                                      ggml_tensor* video,
                                      ggml_tensor* audio) const {
            int64_t divisor = video->ne[0] * video->ne[1] * video->ne[2];
            int64_t values  = ggml_nelements(audio);
            int64_t padding = (divisor - values % divisor) % divisor;
            audio           = ggml_reshape_4d(ctx, ggml_cont(ctx, audio), values, 1, 1, 1);
            if (padding > 0) {
                audio = ggml_ext_pad(ctx, audio, static_cast<int>(padding), 0, 0, 0);
            }
            audio = ggml_reshape_4d(ctx,
                                    audio,
                                    video->ne[0],
                                    video->ne[1],
                                    video->ne[2],
                                    (values + padding) / divisor);
            return ggml_concat(ctx, video, audio, 3);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& packed,
                                 const sd::Tensor<float>& timestep,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& condition_videos,
                                 const std::vector<sd::Tensor<float>>& condition_audios,
                                 const sd::Tensor<int32_t>& text_tags,
                                 const sd::Tensor<int32_t>& keyframe_indices,
                                 const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                 int audio_length,
                                 float video_shift,
                                 float audio_shift) {
            auto split        = split_av_latents(packed, audio_length);
            video_input_cache = std::move(split.first);
            audio_input_cache = std::move(split.second);
            GGML_ASSERT(!audio_input_cache.empty());
            GGML_ASSERT(!context_tensor.empty());

            auto video   = make_input(video_input_cache);
            auto audio   = make_input(audio_input_cache);
            auto context = make_input(context_tensor);
            std::vector<ggml_tensor*> condition_inputs;
            condition_inputs.reserve(condition_videos.size());
            for (const auto& condition : condition_videos) {
                condition_inputs.push_back(make_input(condition));
            }
            std::vector<ggml_tensor*> audio_condition_inputs;
            audio_condition_inputs.reserve(condition_audios.size());
            for (const auto& condition : condition_audios) {
                audio_condition_inputs.push_back(make_input(condition));
            }

            float sigma_v = std::clamp(timestep[0] / 1000.f, 1e-6f, 1.f);
            float t_v     = 1.f - sigma_v;
            float t_a     = 1.f - time_shift_sigma(sigma_v, video_shift, audio_shift);
            auto layout   = build_layout(context_tensor.shape()[1],
                                         video_input_cache.shape()[2],
                                         video_input_cache.shape()[1],
                                         video_input_cache.shape()[0],
                                         audio_length,
                                         condition_videos,
                                         condition_audios,
                                         keyframe_indices,
                                         reference_blocks,
                                         text_tags,
                                         t_v,
                                         t_a);

            position_input_cache = sd::Tensor<float>(
                {3, static_cast<int64_t>(layout.positions.size() / 3)},
                layout.positions);
            auto positions = make_input(position_input_cache);

            ggml_tensor* timestep_features   = nullptr;
            ggml_tensor* curve_indices       = nullptr;
            ggml_tensor* curve_upper_indices = nullptr;
            ggml_tensor* curve_fractions     = nullptr;
            if (config.uses_adaln_curves()) {
                std::vector<int32_t> indices(layout.timesteps.size());
                std::vector<int32_t> upper_indices(layout.timesteps.size());
                std::vector<float> fractions(layout.timesteps.size());
                for (size_t i = 0; i < layout.timesteps.size(); ++i) {
                    float position   = std::clamp(layout.timesteps[i], 0.f, 1.f) * (config.adaln_curve_grid - 1);
                    int index        = std::min(static_cast<int>(std::floor(position)),
                                                static_cast<int>(config.adaln_curve_grid - 2));
                    indices[i]       = index;
                    upper_indices[i] = index + 1;
                    fractions[i]     = position - index;
                }
                curve_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(indices.size())},
                    indices);
                curve_upper_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(upper_indices.size())},
                    upper_indices);
                curve_fraction_input_cache = sd::Tensor<float>(
                    {1, static_cast<int64_t>(fractions.size())},
                    fractions);
                curve_indices       = make_input(curve_index_input_cache);
                curve_upper_indices = make_input(curve_upper_index_input_cache);
                curve_fractions     = make_input(curve_fraction_input_cache);
            } else {
                timestep_feature_input_cache = sd::Tensor<float>(
                    {config.timestep_input_dim, static_cast<int64_t>(layout.timesteps.size())},
                    timestep_embedding(layout.timesteps,
                                       static_cast<int>(config.timestep_input_dim),
                                       10000,
                                       true,
                                       1.f));
                timestep_features = make_input(timestep_feature_input_cache);
            }

            auto runner_ctx = get_context();
            auto output     = model.forward(&runner_ctx,
                                            video,
                                            audio,
                                            context,
                                            condition_inputs,
                                            audio_condition_inputs,
                                            positions,
                                            timestep_features,
                                            curve_indices,
                                            curve_upper_indices,
                                            curve_fractions,
                                            layout.segments,
                                            layout.sequence_segments,
                                            layout.video_segment,
                                            layout.audio_segment,
                                            time_shift_slope(sigma_v, video_shift, audio_shift));
            auto merged     = merge_av_latents(compute_ctx, output.first, output.second);
            auto graph      = new_graph_custom(H3_GRAPH_SIZE);
            ggml_build_forward_expand(graph, merged);
            return graph;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& params) override {
            GGML_ASSERT(params.x != nullptr && params.timesteps != nullptr && params.context != nullptr);
            const auto* extra = diffusion_extra_as<MiniMaxH3DiffusionExtra>(params);
            static const std::vector<sd::Tensor<float>> empty_conditions;
            static const std::vector<MiniMaxH3ReferenceBlock> empty_reference_blocks;
            const auto& conditions       = params.ref_latents == nullptr ? empty_conditions : *params.ref_latents;
            const auto& audio_conditions = extra->reference_audio_latents == nullptr
                                               ? empty_conditions
                                               : *extra->reference_audio_latents;
            const auto& reference_blocks = extra->reference_blocks == nullptr
                                               ? empty_reference_blocks
                                               : *extra->reference_blocks;
            const sd::Tensor<int32_t> empty_int;
            auto get_graph = [&]() {
                return build_graph(*params.x,
                                   *params.timesteps,
                                   *params.context,
                                   conditions,
                                   audio_conditions,
                                   extra->text_token_tags == nullptr ? empty_int : *extra->text_token_tags,
                                   extra->keyframe_indices == nullptr ? empty_int : *extra->keyframe_indices,
                                   reference_blocks,
                                   extra->audio_length,
                                   extra->video_sigma_shift,
                                   extra->audio_sigma_shift);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph,
                                                                              n_threads,
                                                                              false,
                                                                              false,
                                                                              false),
                                                   params.x->dim());
        }
    };

}  // namespace MiniMaxH3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_HPP__

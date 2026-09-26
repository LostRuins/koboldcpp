#ifndef __SD_MODEL_DIFFUSION_LLADA_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_LLADA_IMAGE_HPP__

#include <algorithm>
#include <cinttypes>

#include "core/ggml_extend.h"
#include "core/ggml_runner.h"
#include "core/util.h"
#include "model/common/ggml_block.hpp"
#include "model/diffusion/model.hpp"
#include "model/diffusion/z_image.hpp"
#include "model_loader.h"

// Ref: https://github.com/inclusionAI/LLaDA-Image/blob/main/src/models/transformer_llada_image.py
//
// The denoiser is Lumina2/z_image's NextDiT with identical hyperparameters, so the blocks are
// reused from ZImage. Two things differ: every norm here is non-parametric (the checkpoint
// carries no norm weights at all), and latents arrive already patchified from the Flux2 VAE,
// so patch_size is 1 over 128 channels.

namespace LLaDAImage {
    constexpr int LLADA_IMAGE_GRAPH_SIZE = 20480;

    struct LLaDAImageConfig {
        int patch_size             = 1;
        int64_t hidden_size        = 3840;
        int64_t in_channels        = 128;
        int64_t out_channels       = 128;
        int64_t num_layers         = 30;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 128;
        int64_t num_heads          = 30;
        int64_t num_kv_heads       = 30;
        int64_t multiple_of        = 256;
        float ffn_dim_multiplier   = 8.0f / 3.0f;
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2560;
        int64_t semantic_feat_dim  = 4096;
        int theta                  = 256;
        std::vector<int> axes_dim  = {32, 48, 48};
        int64_t axes_dim_sum       = 128;

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

        static LLaDAImageConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            LLaDAImageConfig config;
            int64_t detected_q_dim  = 0;
            int64_t detected_kv_dim = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "x_embedder.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area = config.patch_size * config.patch_size;
                    config.in_channels = tensor_storage.ne[0] / patch_area;
                    config.hidden_size = tensor_storage.ne[1];
                } else if (ends_with(name, "cap_embedder.1.weight") && tensor_storage.n_dims == 2) {
                    config.cap_feat_dim = tensor_storage.ne[0];
                    config.hidden_size  = tensor_storage.ne[1];
                } else if (ends_with(name, "sigvq_embedder.1.weight") && tensor_storage.n_dims == 2) {
                    config.semantic_feat_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "layers.0.attention.to_q.weight") && tensor_storage.n_dims == 2) {
                    detected_q_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "layers.0.attention.to_k.weight") && tensor_storage.n_dims == 2) {
                    detected_kv_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "final_layer.linear.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area  = config.patch_size * config.patch_size;
                    config.out_channels = tensor_storage.ne[1] / patch_area;
                }
            }

            int64_t detected_layers  = count_blocks(tensor_storage_map, prefix, "layers.");
            int64_t detected_refiner = std::max(count_blocks(tensor_storage_map, prefix, "noise_refiner."),
                                                count_blocks(tensor_storage_map, prefix, "context_refiner."));
            if (detected_layers > 0) {
                config.num_layers = detected_layers;
            }
            if (detected_refiner > 0) {
                config.num_refiner_layers = detected_refiner;
            }
            if (detected_q_dim > 0) {
                config.num_heads = detected_q_dim / config.head_dim;
            }
            if (detected_kv_dim > 0) {
                config.num_kv_heads = detected_kv_dim / config.head_dim;
            } else if (detected_q_dim > 0) {
                config.num_kv_heads = config.num_heads;
            }

            LOG_VERBOSE("llada_image: num_layers = %" PRId64 ", num_refiner_layers = %" PRId64 ", hidden_size = %" PRId64 ", num_heads = %" PRId64 ", num_kv_heads = %" PRId64 ", in_channels = %" PRId64 ", out_channels = %" PRId64 ", cap_feat_dim = %" PRId64 ", semantic_feat_dim = %" PRId64,
                        config.num_layers,
                        config.num_refiner_layers,
                        config.hidden_size,
                        config.num_heads,
                        config.num_kv_heads,
                        config.in_channels,
                        config.out_channels,
                        config.cap_feat_dim,
                        config.semantic_feat_dim);
            return config;
        }
    };

    class LLaDAImageModel : public GGMLBlock {
    protected:
        LLaDAImageConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.hidden_size);
            params["x_pad_token"]     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.hidden_size);
            params["sigvq_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.hidden_size);
        }

        std::shared_ptr<ZImage::JointTransformerBlock> make_block(bool modulation) {
            return std::make_shared<ZImage::JointTransformerBlock>(0,
                                                                   config.hidden_size,
                                                                   config.head_dim,
                                                                   config.num_heads,
                                                                   config.num_kv_heads,
                                                                   config.multiple_of,
                                                                   config.ffn_dim_multiplier,
                                                                   config.norm_eps,
                                                                   config.qk_norm,
                                                                   modulation,
                                                                   false,
                                                                   true);
        }

    public:
        LLaDAImageModel() = default;
        LLaDAImageModel(LLaDAImageConfig config)
            : config(config) {
            blocks["x_embedder"] = std::make_shared<Linear>(config.patch_size * config.patch_size * config.in_channels, config.hidden_size);
            blocks["t_embedder"] = std::make_shared<TimestepEmbedder>(MIN(config.hidden_size, 1024), 256, ZImage::ADALN_EMBED_DIM);

            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(config.cap_feat_dim, config.norm_eps, false);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(config.cap_feat_dim, config.hidden_size);

            blocks["semantic_embedder.0"] = std::make_shared<RMSNorm>(config.semantic_feat_dim, config.norm_eps, false);
            blocks["semantic_embedder.1"] = std::make_shared<Linear>(config.semantic_feat_dim, config.hidden_size);
            blocks["sigvq_embedder.0"]    = std::make_shared<RMSNorm>(config.semantic_feat_dim, config.norm_eps, false);
            blocks["sigvq_embedder.1"]    = std::make_shared<Linear>(config.semantic_feat_dim, config.hidden_size);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                blocks["noise_refiner." + std::to_string(i)]   = make_block(true);
                blocks["context_refiner." + std::to_string(i)] = make_block(false);
                blocks["sigvq_refiner." + std::to_string(i)]   = make_block(false);
            }
            for (int i = 0; i < config.num_layers; i++) {
                blocks["layers." + std::to_string(i)] = make_block(true);
            }

            blocks["final_layer"] = std::make_shared<ZImage::FinalLayer>(config.hidden_size, config.patch_size, config.out_channels);
        }

        ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto final_layer    = std::dynamic_pointer_cast<ZImage::FinalLayer>(blocks["final_layer"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            // sdcpp's flow denoiser already hands over sigma * 1000, which is the range the
            // reference reaches via its own t_scale, so no further scaling here.
            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]
            sd::ggml_graph_cut::mark_graph_cut(txt, "llada_image.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(img, "llada_image.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(t_emb, "llada_image.prelude", "t_emb");

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), ZImage::SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), ZImage::SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);
            }

            GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]);

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt->ne[1], pe->ne[3]);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);

                txt = block->forward(ctx, txt, txt_pe, nullptr, nullptr);
                sd::ggml_graph_cut::mark_graph_cut(txt, "llada_image.context_refiner." + std::to_string(i), "txt");
            }

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);

                img = block->forward(ctx, img, img_pe, nullptr, t_emb);
                sd::ggml_graph_cut::mark_graph_cut(img, "llada_image.noise_refiner." + std::to_string(i), "img");
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);
            sd::ggml_graph_cut::mark_graph_cut(txt_img, "llada_image.prelude", "txt_img");

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["layers." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, pe, nullptr, t_emb);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "llada_image.layers." + std::to_string(i), "txt_img");
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);

            return ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, n_txt_token + n_txt_pad_token, n_txt_token + n_txt_pad_token + n_img_token);
        }

        ggml_tensor* pad_stream(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pad_token) {
            int64_t n_pad = Rope::bound_mod(static_cast<int>(x->ne[1]), ZImage::SEQ_MULTI_OF);
            if (n_pad == 0) {
                return x;
            }
            auto pads = ggml_repeat_4d(ctx->ggml_ctx, pad_token, pad_token->ne[0], n_pad, x->ne[2], 1);
            return ggml_concat(ctx->ggml_ctx, x, pads, 1);
        }

        // Editing runs one joint sequence carrying two timesteps: the caption and source latent
        // are clean (t = 0) while the second caption copy and the target latent are noisy. adaLN
        // is a linear map of the timestep embedding, so feeding a per-token embedding selects the
        // right modulation exactly, without duplicating the modulation projections.
        ggml_tensor* forward_editing(GGMLRunnerContext* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* timestep,
                                     ggml_tensor* context,
                                     ggml_tensor* semantic,
                                     ggml_tensor* source_latent,
                                     ggml_tensor* pe) {
            ggml_context* gctx = ctx->ggml_ctx;

            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto sigvq_embed_0  = std::dynamic_pointer_cast<RMSNorm>(blocks["sigvq_embedder.0"]);
            auto sigvq_embed_1  = std::dynamic_pointer_cast<Linear>(blocks["sigvq_embedder.1"]);
            auto final_layer    = std::dynamic_pointer_cast<ZImage::FinalLayer>(blocks["final_layer"]);

            auto t_noisy = t_embedder->forward(ctx, timestep);
            auto t_clean = t_embedder->forward(ctx, ggml_scale(gctx, timestep, 0.f));

            auto per_token = [&](ggml_tensor* emb, int64_t n) {
                return ggml_repeat_4d(gctx, emb, emb->ne[0], n, 1, 1);
            };

            auto cap        = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));
            cap             = pad_stream(ctx, cap, params["cap_pad_token"]);
            int64_t cap_len = cap->ne[1];
            cap             = ggml_concat(gctx, cap, cap, 1);

            auto src            = pad_stream(ctx, x_embedder->forward(ctx, source_latent), params["x_pad_token"]);
            auto tgt_embed      = x_embedder->forward(ctx, x);
            int64_t n_img_token = tgt_embed->ne[1];
            auto tgt            = pad_stream(ctx, tgt_embed, params["x_pad_token"]);
            int64_t img_len     = tgt->ne[1];
            auto img            = ggml_concat(gctx, src, tgt, 1);

            ggml_tensor* sig = nullptr;
            int64_t sig_len  = 0;
            if (semantic != nullptr) {
                sig     = sigvq_embed_1->forward(ctx, sigvq_embed_0->forward(ctx, semantic));
                sig     = pad_stream(ctx, sig, params["sigvq_pad_token"]);
                sig_len = sig->ne[1];
            }

            GGML_ASSERT(cap_len * 2 + img_len * 2 + sig_len == pe->ne[3]);

            auto cap_pe = ggml_ext_slice(gctx, pe, 3, 0, cap_len * 2);
            auto img_pe = ggml_ext_slice(gctx, pe, 3, cap_len * 2, cap_len * 2 + img_len * 2);

            auto img_adaln = ggml_concat(gctx, per_token(t_clean, img_len), per_token(t_noisy, img_len), 1);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);
                cap        = block->forward(ctx, cap, cap_pe, nullptr, nullptr);
            }
            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);
                img        = block->forward(ctx, img, img_pe, nullptr, img_adaln);
            }
            if (sig != nullptr) {
                auto sig_pe = ggml_ext_slice(gctx, pe, 3, cap_len * 2 + img_len * 2, pe->ne[3]);
                for (int i = 0; i < config.num_refiner_layers; i++) {
                    auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["sigvq_refiner." + std::to_string(i)]);
                    sig        = block->forward(ctx, sig, sig_pe, nullptr, nullptr);
                }
            }

            auto seq = ggml_concat(gctx, cap, img, 1);

            auto cap_adaln = ggml_concat(gctx, per_token(t_clean, cap_len), per_token(t_noisy, cap_len), 1);
            auto seq_adaln = ggml_concat(gctx, cap_adaln, img_adaln, 1);
            if (sig != nullptr) {
                seq       = ggml_concat(gctx, seq, sig, 1);
                seq_adaln = ggml_concat(gctx, seq_adaln, per_token(t_clean, sig_len), 1);
            }

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["layers." + std::to_string(i)]);
                seq        = block->forward(ctx, seq, pe, nullptr, seq_adaln);
                sd::ggml_graph_cut::mark_graph_cut(seq, "llada_image.layers." + std::to_string(i), "seq");
            }

            seq = final_layer->forward(ctx, seq, seq_adaln);

            // Only the target latent is denoised; the source half of the image stream is context.
            // The stream is padded to SEQ_MULTI_OF, so drop the pad tokens: they are not part of
            // the latent grid that unpatchify reconstructs.
            int64_t target_start = cap_len * 2 + img_len;
            return ggml_ext_slice(gctx, seq, 1, target_start, target_start + n_img_token);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, cap_feat_dim]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int patch_size = config.patch_size;

            auto img = DiT::pad_and_patchify(ctx, x, patch_size, patch_size, false);

            auto out = forward_core(ctx, img, timestep, context, pe);

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size, false);

            // The reference pipeline negates the model output before the scheduler step.
            return ggml_ext_scale(ctx->ggml_ctx, out, -1.f);
        }
    };

    struct LLaDAImageRunner : public DiffusionModelRunner {
    public:
        LLaDAImageConfig config;
        LLaDAImageModel llada_image;
        std::vector<float> pe_vec;

        LLaDAImageRunner(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map      = {},
                         const std::string prefix                            = "",
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(LLaDAImageConfig::detect_from_weights(tensor_storage_map, prefix)) {
            llada_image = LLaDAImageModel(config);
            llada_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "llada_image";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            llada_image.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(LLADA_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            pe_vec      = finish_rope_pe(Rope::gen_llada_image_pe(static_cast<int>(x->ne[1]),
                                                                  static_cast<int>(x->ne[0]),
                                                                  config.patch_size,
                                                                  static_cast<int>(x->ne[3]),
                                                                  static_cast<int>(context->ne[1]),
                                                                  ZImage::SEQ_MULTI_OF,
                                                                  config.theta,
                                                                  config.axes_dim));
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            ggml_tensor* out = llada_image.forward(&runner_ctx, x, timesteps, context, pe);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, cap_feat_dim]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };

            return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false), x.dim());
        }

        ggml_cgraph* build_edit_graph(const sd::Tensor<float>& x_tensor,
                                      const sd::Tensor<float>& timesteps_tensor,
                                      const sd::Tensor<float>& context_tensor,
                                      const sd::Tensor<float>& semantic_tensor,
                                      const sd::Tensor<float>& source_tensor) {
            ggml_cgraph* gf        = new_graph_custom(LLADA_IMAGE_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_input(context_tensor);
            ggml_tensor* semantic  = make_optional_input(semantic_tensor);
            ggml_tensor* source    = make_input(source_tensor);
            GGML_ASSERT(x->ne[3] == 1);

            pe_vec      = finish_rope_pe(Rope::gen_llada_image_edit_pe(static_cast<int>(x->ne[1]),
                                                                       static_cast<int>(x->ne[0]),
                                                                       config.patch_size,
                                                                       static_cast<int>(context->ne[1]),
                                                                  semantic != nullptr ? static_cast<int>(semantic->ne[1]) : 0,
                                                                       ZImage::SEQ_MULTI_OF,
                                                                       config.theta,
                                                                       config.axes_dim));
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());
            auto runner_ctx = get_context();

            int64_t W   = x->ne[0];
            int64_t H   = x->ne[1];
            auto target = DiT::pad_and_patchify(&runner_ctx, x, config.patch_size, config.patch_size, false);
            auto src    = DiT::pad_and_patchify(&runner_ctx, source, config.patch_size, config.patch_size, false);

            auto out = llada_image.forward_editing(&runner_ctx, target, timesteps, context, semantic, src, pe);
            out      = DiT::unpatchify_and_crop(runner_ctx.ggml_ctx, out, H, W, config.patch_size, config.patch_size, false);
            out      = ggml_ext_scale(runner_ctx.ggml_ctx, out, -1.f);

            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);

            const auto* extra   = std::get_if<LLaDAImageDiffusionExtra>(&diffusion_params.extra);
            bool has_semantic   = extra != nullptr && extra->semantic != nullptr && !extra->semantic->empty();
            bool has_ref_latent = diffusion_params.ref_latents != nullptr && !diffusion_params.ref_latents->empty();
            if (has_semantic && !has_ref_latent) {
                LOG_WARN("llada_image: SigVQ features without a reference latent are not supported; falling back to text to image");
            }
            if (has_ref_latent) {
                const auto& source = diffusion_params.ref_latents->front();
                if (source.shape() != diffusion_params.x->shape()) {
                    LOG_ERROR("llada_image: reference latent must match the target shape; use resize_vae_to_target=1");
                    return {};
                }
                auto get_graph = [&]() -> ggml_cgraph* {
                    return build_edit_graph(*diffusion_params.x,
                                            *diffusion_params.timesteps,
                                            tensor_or_empty(diffusion_params.context),
                                            tensor_or_empty(extra != nullptr ? extra->semantic : nullptr),
                                            source);
                };
                return restore_trailing_singleton_dims(GGMLRunner::compute(get_graph, n_threads, false),
                                                       diffusion_params.x->dim());
            }

            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context));
        }
    };

}  // namespace LLaDAImage

#endif  // __SD_MODEL_DIFFUSION_LLADA_IMAGE_HPP__

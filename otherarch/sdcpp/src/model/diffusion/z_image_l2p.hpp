#ifndef __SD_MODEL_DIFFUSION_Z_IMAGE_L2P_HPP__
#define __SD_MODEL_DIFFUSION_Z_IMAGE_L2P_HPP__

#include <cmath>

#include "z_image.hpp"

// Ref: https://github.com/TencentYoutuResearch/T2I-L2P/blob/main/diffsynth/models/z_image_dit_L2P.py

namespace ZImageL2P {
    struct ZImageL2PConfig : ZImage::ZImageConfig {
        static ZImageL2PConfig detect_from_weights(const String2TensorStorage& tensors, const std::string& prefix) {
            ZImageL2PConfig config;
            static_cast<ZImage::ZImageConfig&>(config) = ZImage::ZImageConfig::detect_from_weights(tensors, prefix);
            config.in_channels                         = 3;
            config.out_channels                        = 3;
            config.patch_size                          = 16;
            auto x_embedder                            = tensors.find(prefix + ".x_embedder.weight");
            if (x_embedder != tensors.end()) {
                config.patch_size = static_cast<int>(std::lround(std::sqrt(static_cast<double>(x_embedder->second.ne[0] / config.in_channels))));
            }
            // L2P ships split to_q/to_k/to_v, which name conversion maps to qkv.weight plus
            // .1/.2 parts; the base detection only sees the q part and undercounts kv heads.
            auto k_part = tensors.find(prefix + ".layers.0.attention.qkv.weight.1");
            if (k_part != tensors.end()) {
                config.num_kv_heads = k_part->second.ne[1] / config.head_dim;
            }
            LOG_VERBOSE("z_image_l2p: patch_size = %d, in_channels = %" PRId64 ", num_heads = %" PRId64 ", num_kv_heads = %" PRId64,
                        config.patch_size,
                        config.in_channels,
                        config.num_heads,
                        config.num_kv_heads);
            return config;
        }
    };

    // U-Net over the full-resolution noisy image. Its bottleneck sits at the DiT token
    // grid, so the number of pooling levels is tied to patch_size == 16.
    class LocalDecoder : public GGMLBlock {
        static constexpr int LEVELS                  = 4;
        static constexpr int64_t CHANNELS[LEVELS]    = {64, 128, 256, 512};
        static constexpr int64_t BOTTLENECK_CHANNELS = 512;

    public:
        LocalDecoder(int64_t in_channels, int64_t cond_channels) {
            int64_t prev = in_channels;
            for (int i = 0; i < LEVELS; ++i) {
                blocks["enc" + std::to_string(i + 1) + ".0"] = std::make_shared<Conv2d>(prev, CHANNELS[i], std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
                prev                                         = CHANNELS[i];
            }
            blocks["bottleneck.0"] = std::make_shared<Conv2d>(prev + cond_channels, BOTTLENECK_CHANNELS, std::pair{1, 1});
            prev                   = BOTTLENECK_CHANNELS;
            for (int i = LEVELS - 1; i >= 0; --i) {
                const std::string level      = std::to_string(i + 1);
                const int64_t out            = i == 0 ? CHANNELS[0] : CHANNELS[i - 1];
                blocks["up" + level + ".1"]  = std::make_shared<Conv2d>(prev, prev, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
                blocks["dec" + level + ".0"] = std::make_shared<Conv2d>(prev + CHANNELS[i], out, std::pair{3, 3}, std::pair{1, 1}, std::pair{1, 1});
                prev                         = out;
            }
            blocks["out_conv"] = std::make_shared<Conv2d>(prev, in_channels, std::pair{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* cond) {
            // x: [N, C, H, W]
            // cond: [N, cond_channels, H / 16, W / 16]
            // return: [N, C, H, W]
            auto gctx = ctx->ggml_ctx;
            auto conv = [&](const std::string& name, ggml_tensor* h) {
                return std::dynamic_pointer_cast<Conv2d>(blocks[name])->forward(ctx, h);
            };

            std::vector<ggml_tensor*> skips;
            auto h = x;
            for (int i = 0; i < LEVELS; ++i) {
                h = ggml_silu(gctx, conv("enc" + std::to_string(i + 1) + ".0", h));
                skips.push_back(h);
                h = ggml_pool_2d(gctx, h, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
            }

            h = ggml_concat(gctx, h, cond, 2);
            h = ggml_silu(gctx, conv("bottleneck.0", h));

            for (int i = LEVELS - 1; i >= 0; --i) {
                const std::string level = std::to_string(i + 1);
                h                       = ggml_upscale(gctx, h, 2, GGML_SCALE_MODE_NEAREST);
                h                       = conv("up" + level + ".1", h);
                h                       = ggml_concat(gctx, h, skips[i], 2);
                h                       = ggml_silu(gctx, conv("dec" + level + ".0", h));
            }
            return conv("out_conv", h);
        }
    };

    class ZImageL2PModel : public GGMLBlock {
        ZImageL2PConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.hidden_size);
            params["x_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.hidden_size);
        }

    public:
        explicit ZImageL2PModel(const ZImageL2PConfig& config)
            : config(config) {
            blocks["x_embedder"]     = std::make_shared<Linear>(config.patch_size * config.patch_size * config.in_channels, config.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(MIN(config.hidden_size, 1024), 256, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(config.cap_feat_dim, config.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(config.cap_feat_dim, config.hidden_size);
            auto add_blocks          = [&](const std::string& prefix, int64_t count, bool modulation) {
                for (int64_t i = 0; i < count; ++i) {
                    blocks[prefix + std::to_string(i)] = std::make_shared<ZImage::JointTransformerBlock>(
                        static_cast<int>(i), config.hidden_size, config.head_dim, config.num_heads,
                        config.num_kv_heads, config.multiple_of, config.ffn_dim_multiplier,
                        config.norm_eps, config.qk_norm, modulation, true, false, 1e-5f);
                }
            };
            add_blocks("noise_refiner.", config.num_refiner_layers, true);
            add_blocks("context_refiner.", config.num_refiner_layers, false);
            add_blocks("layers.", config.num_layers, true);
            blocks["local_decoder"] = std::make_shared<LocalDecoder>(config.in_channels, config.hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, ggml_tensor* pe) {
            // x: [N, C, H, W], H and W are multiples of patch_size
            // context: [N, L, cap_feat_dim]
            // return: [N, C, H, W]
            auto gctx           = ctx->ggml_ctx;
            const int64_t W     = x->ne[0];
            const int64_t H     = x->ne[1];
            const int64_t N     = x->ne[3];
            const int64_t w     = W / config.patch_size;
            const int64_t h     = H / config.patch_size;
            const int64_t n_img = w * h;
            const int64_t n_txt = context->ne[1];

            auto img = DiT::patchify(gctx, x, config.patch_size, config.patch_size, false);
            img      = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"])->forward(ctx, img);
            auto txt = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"])->forward(ctx, context);
            txt      = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"])->forward(ctx, txt);
            auto t   = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"])->forward(ctx, timestep);
            sd::ggml_graph_cut::mark_graph_cut(txt, "z_image_l2p.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(img, "z_image_l2p.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(t, "z_image_l2p.prelude", "t_emb");

            const int64_t n_txt_pad = Rope::bound_mod(static_cast<int>(n_txt), ZImage::SEQ_MULTI_OF);
            if (n_txt_pad > 0) {
                auto pad = params["cap_pad_token"];
                txt      = ggml_concat(gctx, txt, ggml_repeat_4d(gctx, pad, pad->ne[0], n_txt_pad, N, 1), 1);
            }
            const int64_t n_img_pad = Rope::bound_mod(static_cast<int>(n_img), ZImage::SEQ_MULTI_OF);
            if (n_img_pad > 0) {
                auto pad = params["x_pad_token"];
                img      = ggml_concat(gctx, img, ggml_repeat_4d(gctx, pad, pad->ne[0], n_img_pad, N, 1), 1);
            }
            GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]);

            auto txt_pe = ggml_ext_slice(gctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(gctx, pe, 3, txt->ne[1], pe->ne[3]);
            for (int64_t i = 0; i < config.num_refiner_layers; ++i) {
                txt = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)])->forward(ctx, txt, txt_pe);
                sd::ggml_graph_cut::mark_graph_cut(txt, "z_image_l2p.context_refiner." + std::to_string(i), "txt");
            }
            for (int64_t i = 0; i < config.num_refiner_layers; ++i) {
                img = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)])->forward(ctx, img, img_pe, nullptr, t);
                sd::ggml_graph_cut::mark_graph_cut(img, "z_image_l2p.noise_refiner." + std::to_string(i), "img");
            }

            auto txt_img = ggml_concat(gctx, txt, img, 1);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                txt_img = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["layers." + std::to_string(i)])->forward(ctx, txt_img, pe, nullptr, t);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "z_image_l2p.layers." + std::to_string(i), "txt_img");
            }

            // The local decoder consumes the raw last-layer hidden states: there is no final
            // norm or adaLN modulation in front of it, unlike Z-Image's final_layer.
            const int64_t img_start = n_txt + n_txt_pad;
            auto feat               = ggml_ext_slice(gctx, txt_img, 1, img_start, img_start + n_img);                           // [N, h*w, hidden_size]
            feat                    = ggml_reshape_4d(gctx, feat, config.hidden_size, w, h, N);                                 // [N, h, w, hidden_size]
            feat                    = ggml_cont(gctx, ggml_permute(gctx, feat, 2, 0, 1, 3));                                    // [N, hidden_size, h, w]
            auto out                = std::dynamic_pointer_cast<LocalDecoder>(blocks["local_decoder"])->forward(ctx, x, feat);  // [N, C, H, W]
            return ggml_ext_scale(gctx, out, -1.f);
        }
    };

    struct ZImageL2PRunner : public DiffusionModelRunner {
        ZImageL2PConfig config;
        ZImageL2PModel model;
        std::vector<float> pe_vec;

        ZImageL2PRunner(ggml_backend_t backend,
                        const String2TensorStorage& tensors,
                        const std::string& prefix,
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(ZImageL2PConfig::detect_from_weights(tensors, prefix)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override { return "z_image_l2p"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            GGML_ASSERT(inputs.x != nullptr);
            GGML_ASSERT(inputs.timesteps != nullptr);
            if (inputs.ref_latents != nullptr && !inputs.ref_latents->empty()) {
                LOG_ERROR("Z-Image L2P reference-image conditioning is not supported");
                return {};
            }
            if (inputs.context == nullptr || inputs.context->empty()) {
                LOG_ERROR("Z-Image L2P requires a text condition");
                return {};
            }
            auto graph = [&]() {
                auto gf      = new_graph_custom(ZImage::Z_IMAGE_GRAPH_SIZE);
                auto x       = make_input(*inputs.x);
                auto t       = make_input(*inputs.timesteps);
                auto context = make_input(*inputs.context);
                GGML_ASSERT(x->ne[3] == 1);
                GGML_ASSERT(x->ne[0] % config.patch_size == 0 && x->ne[1] % config.patch_size == 0);
                pe_vec      = finish_rope_pe(Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                                                  static_cast<int>(x->ne[0]),
                                                                  config.patch_size,
                                                                  static_cast<int>(x->ne[3]),
                                                                  static_cast<int>(context->ne[1]),
                                                                  ZImage::SEQ_MULTI_OF,
                                                                  {},
                                                                  Rope::RefIndexMode::FIXED,
                                                                  config.theta,
                                                                  config.axes_dim));
                int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
                auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
                set_backend_tensor_data(pe, pe_vec.data());
                auto ctx = get_context();
                auto out = model.forward(&ctx, x, t, context, pe);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}  // namespace ZImageL2P

#endif  // __SD_MODEL_DIFFUSION_Z_IMAGE_L2P_HPP__

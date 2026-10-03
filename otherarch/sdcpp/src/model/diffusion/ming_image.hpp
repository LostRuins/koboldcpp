#ifndef __SD_MODEL_DIFFUSION_MING_IMAGE_HPP__
#define __SD_MODEL_DIFFUSION_MING_IMAGE_HPP__

#include "z_image.hpp"

namespace MingImage {
    struct MingImageConfig : ZImage::ZImageConfig {
        bool split_qkv = true;

        static MingImageConfig detect_from_weights(const String2TensorStorage& tensors, const std::string& prefix) {
            MingImageConfig config;
            static_cast<ZImage::ZImageConfig&>(config) = ZImage::ZImageConfig::detect_from_weights(tensors, prefix);
            config.split_qkv                           = tensors.count(prefix + ".layers.0.attention.qkv.weight") == 0;
            return config;
        }
    };

    class MingImageModel : public GGMLBlock {
        MingImageConfig config;

    public:
        explicit MingImageModel(const MingImageConfig& config)
            : config(config) {
            blocks["x_embedder"]     = std::make_shared<Linear>(config.patch_size * config.patch_size * config.in_channels, config.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(1024, 256, std::min<int64_t>(config.hidden_size, 256));
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(config.cap_feat_dim, config.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(config.cap_feat_dim, config.hidden_size);
            auto add_blocks          = [&](const std::string& prefix, int64_t count, bool modulation) {
                for (int64_t i = 0; i < count; ++i) {
                    blocks[prefix + std::to_string(i)] = std::make_shared<ZImage::JointTransformerBlock>(
                        static_cast<int>(i), config.hidden_size, config.head_dim, config.num_heads,
                        config.num_kv_heads, config.multiple_of, config.ffn_dim_multiplier,
                        config.norm_eps, config.qk_norm, modulation, true, config.split_qkv, 1e-5f);
                }
            };
            add_blocks("noise_refiner.", config.num_refiner_layers, true);
            add_blocks("context_refiner.", config.num_refiner_layers, false);
            add_blocks("layers.", config.num_layers, true);
            blocks["final_layer"] = std::make_shared<ZImage::FinalLayer>(config.hidden_size, config.patch_size, config.out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, ggml_tensor* direct, ggml_tensor* pe) {
            auto gctx           = ctx->ggml_ctx;
            const int64_t width = x->ne[0], height = x->ne[1];
            auto img            = DiT::pad_and_patchify(ctx, x, config.patch_size, config.patch_size, false);
            img                 = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"])->forward(ctx, img);
            auto txt            = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"])->forward(ctx, context);
            txt                 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"])->forward(ctx, txt);
            txt                 = ggml_concat(gctx, txt, direct, 1);
            auto t              = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"])->forward(ctx, timestep);
            const int64_t n_txt = txt->ne[1], n_img = img->ne[1];
            auto txt_pe = ggml_ext_slice(gctx, pe, 3, 0, n_txt);
            auto img_pe = ggml_ext_slice(gctx, pe, 3, n_txt, n_txt + n_img);
            for (int64_t i = 0; i < config.num_refiner_layers; ++i) {
                txt = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)])->forward(ctx, txt, txt_pe);
                sd::ggml_graph_cut::mark_graph_cut(txt, "ming_image.context_refiner." + std::to_string(i), "txt");
            }
            for (int64_t i = 0; i < config.num_refiner_layers; ++i) {
                img = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)])->forward(ctx, img, img_pe, nullptr, t);
                sd::ggml_graph_cut::mark_graph_cut(img, "ming_image.noise_refiner." + std::to_string(i), "img");
            }
            auto combined = ggml_concat(gctx, txt, img, 1);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                combined = std::dynamic_pointer_cast<ZImage::JointTransformerBlock>(blocks["layers." + std::to_string(i)])->forward(ctx, combined, pe, nullptr, t);
                sd::ggml_graph_cut::mark_graph_cut(combined, "ming_image.layers." + std::to_string(i), "combined");
            }
            img = ggml_ext_slice(gctx, combined, 1, n_txt, n_txt + n_img);
            img = std::dynamic_pointer_cast<ZImage::FinalLayer>(blocks["final_layer"])->forward(ctx, img, t);
            img = DiT::unpatchify_and_crop(gctx, img, height, width, config.patch_size, config.patch_size, false);
            return ggml_scale(gctx, img, -1.f);
        }
    };

    struct MingImageRunner : DiffusionModelRunner {
        MingImageConfig config;
        MingImageModel model;
        std::vector<float> pe_values;

        MingImageRunner(ggml_backend_t backend, const String2TensorStorage& tensors, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(MingImageConfig::detect_from_weights(tensors, prefix)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override { return "ming_image"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            const auto* extra = diffusion_extra_as<MingImageDiffusionExtra>(inputs);
            if (inputs.ref_latents != nullptr && !inputs.ref_latents->empty()) {
                LOG_ERROR("Ming-Image reference-image conditioning is not supported");
                return {};
            }
            if (inputs.context == nullptr || extra->direct_context == nullptr) {
                LOG_ERROR("Ming-Image requires both query and direct text conditions");
                return {};
            }
            auto graph = [&]() {
                auto gf      = new_graph_custom(ZImage::Z_IMAGE_GRAPH_SIZE);
                auto x       = make_input(*inputs.x);
                auto t       = make_input(*inputs.timesteps);
                auto context = make_input(*inputs.context);
                auto direct  = make_input(*extra->direct_context);
                GGML_ASSERT(x->ne[3] == 1);
                const int64_t n_txt = context->ne[1] + direct->ne[1];
                const int64_t n_img = ((x->ne[0] + config.patch_size - 1) / config.patch_size) *
                                      ((x->ne[1] + config.patch_size - 1) / config.patch_size);
                auto padded = finish_rope_pe(Rope::gen_z_image_pe(
                    static_cast<int>(x->ne[1]), static_cast<int>(x->ne[0]), config.patch_size, 1,
                    static_cast<int>(n_txt), ZImage::SEQ_MULTI_OF, {}, Rope::RefIndexMode::FIXED,
                    config.theta, config.axes_dim));
                // Zero-masked alignment tokens cannot affect valid queries. Omit them while
                // retaining the padded caption length used to position image tokens.
                const size_t stride      = config.axes_dim_sum * 2;
                const int64_t padded_txt = n_txt + Rope::bound_mod(static_cast<int>(n_txt), ZImage::SEQ_MULTI_OF);
                pe_values.assign(padded.begin(), padded.begin() + n_txt * stride);
                pe_values.insert(pe_values.end(), padded.begin() + padded_txt * stride,
                                 padded.begin() + (padded_txt + n_img) * stride);
                auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, n_txt + n_img);
                set_backend_tensor_data(pe, pe_values.data());
                auto ctx = get_context();
                auto out = model.forward(&ctx, x, t, context, direct, pe);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute(graph, n_threads, false), inputs.x->dim());
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_MING_IMAGE_HPP__

#ifndef __SD_MODEL_DIFFUSION_MAGE_FLOW_HPP__
#define __SD_MODEL_DIFFUSION_MAGE_FLOW_HPP__

#include <cmath>
#include <memory>

#include "model/diffusion/qwen_image.hpp"

namespace MageFlow {
    constexpr int MAGE_FLOW_GRAPH_SIZE = 20480;

    // Mage-Flow was trained with BF16-rounded timestep frequencies; using Qwen's F32 projection degrades generation quality.
    struct MageFlowTimestepProjEmbeddings : public Qwen::QwenTimestepProjEmbeddings {
        static constexpr int TIMESTEP_DIM = 256;
        static constexpr int HALF_DIM     = TIMESTEP_DIM / 2;

        std::vector<float> frequencies;
        std::vector<float> timesteps_proj;

        explicit MageFlowTimestepProjEmbeddings(int64_t embedding_dim)
            : QwenTimestepProjEmbeddings(embedding_dim), frequencies(HALF_DIM) {
            for (int i = 0; i < HALF_DIM; ++i) {
                float frequency = std::exp(-std::log(10000.f) * static_cast<float>(i) / HALF_DIM);
                frequencies[i]  = ggml_bf16_to_fp32(ggml_fp32_to_bf16(frequency));
            }
        }

        void prepare(const sd::Tensor<float>& timesteps) {
            size_t num_timesteps = static_cast<size_t>(timesteps.numel());
            timesteps_proj.resize(static_cast<size_t>(TIMESTEP_DIM) * num_timesteps);
            for (size_t b = 0; b < num_timesteps; ++b) {
                float sigma = ggml_bf16_to_fp32(ggml_fp32_to_bf16(timesteps.values()[b] / 1000.f));
                for (int i = 0; i < HALF_DIM; ++i) {
                    float argument = sigma * frequencies[i] * 1000.f;
                    timesteps_proj[b * TIMESTEP_DIM + i] =
                        ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::cos(argument)));
                    timesteps_proj[b * TIMESTEP_DIM + HALF_DIM + i] =
                        ggml_bf16_to_fp32(ggml_fp32_to_bf16(std::sin(argument)));
                }
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* timesteps,
                             ggml_tensor* addition_t_cond = nullptr) override {
            GGML_ASSERT(addition_t_cond == nullptr);
            GGML_ASSERT(timesteps_proj.size() ==
                        static_cast<size_t>(TIMESTEP_DIM * ggml_nelements(timesteps)));
            auto projection = ggml_new_tensor_2d(ctx->ggml_ctx,
                                                 GGML_TYPE_F32,
                                                 TIMESTEP_DIM,
                                                 ggml_nelements(timesteps));
            ctx->bind_backend_tensor_data(projection, timesteps_proj.data());
            auto timestep_embedder = std::dynamic_pointer_cast<Qwen::TimestepEmbedding>(blocks["timestep_embedder"]);
            return timestep_embedder->forward(ctx, projection);
        }
    };

    struct MageFlowRunner : public DiffusionModelRunner {
    public:
        Qwen::QwenImageConfig config;
        Qwen::QwenImageModel mage_flow;
        std::shared_ptr<MageFlowTimestepProjEmbeddings> time_text_embed;
        std::vector<float> pe_vec;

        MageFlowRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map      = {},
                       const std::string prefix                            = "",
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager) {
            config.patch_size          = 1;
            config.in_channels         = 128;
            config.out_channels        = 128;
            config.num_layers          = 12;
            config.attention_head_dim  = 128;
            config.num_attention_heads = 24;
            config.joint_attention_dim = 2560;
            config.theta               = 10000;
            config.axes_dim            = {16, 56, 56};
            config.axes_dim_sum        = 128;
            time_text_embed            = std::make_shared<MageFlowTimestepProjEmbeddings>(
                config.num_attention_heads * config.attention_head_dim);
            mage_flow = Qwen::QwenImageModel(config, time_text_embed);
            mage_flow.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "mage_flow";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            mage_flow.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& ref_latents_tensor = {}) {
            ggml_cgraph* gf        = new_graph_custom(MAGE_FLOW_GRAPH_SIZE);
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

            int batch_size = static_cast<int>(x->ne[3]);
            pe_vec         = Rope::gen_mage_flow_pe(static_cast<int>(x->ne[1]),
                                                    static_cast<int>(x->ne[0]),
                                                    batch_size,
                                                    static_cast<int>(context->ne[1]),
                                                    ref_latents,
                                                    config.theta,
                                                    config.axes_dim);
            int pos_len    = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe        = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            time_text_embed->prepare(timesteps_tensor);
            auto runner_ctx = get_context();
            auto out        = mage_flow.forward(&runner_ctx,
                                                x,
                                                timesteps,
                                                nullptr,
                                                context,
                                                pe,
                                                ref_latents);
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
                           diffusion_params.ref_latents && diffusion_params.ref_image_params.pass_to_dit ? *diffusion_params.ref_latents : empty_ref_latents);
        }
    };
}  // namespace MageFlow

#endif  // __SD_MODEL_DIFFUSION_MAGE_FLOW_HPP__

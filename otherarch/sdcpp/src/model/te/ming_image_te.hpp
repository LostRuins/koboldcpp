#ifndef __SD_MODEL_TE_MING_IMAGE_TE_HPP__
#define __SD_MODEL_TE_MING_IMAGE_TE_HPP__

#include "llm.hpp"

namespace MingImageTE {
    struct MingImageTEConfig {
        LLM::LLMConfig backbone;
        LLM::LLMConfig connector;
        int64_t num_queries   = 256;
        int64_t caption_dim   = 2560;
        int64_t diffusion_dim = 3840;

        static MingImageTEConfig detect_from_weights(const String2TensorStorage& tensors, const std::string& prefix) {
            MingImageTEConfig config;
            for (const auto& entry : tensors) {
                if (starts_with(entry.first, prefix + ".backbone.") &&
                    contains(entry.first, ".mlp.experts.") && entry.second.type == GGML_TYPE_I8) {
                    throw std::runtime_error("Ming-Image INT8/W4A8 text encoder experts are not supported; use the BF16 text encoder");
                }
            }
            bool vision           = false;
            config.backbone       = LLM::LLMConfig::detect_from_weights(tensors, prefix + ".backbone.", LLM::LLMArch::BAILING_MOE, vision);
            config.connector      = LLM::LLMConfig::detect_from_weights(tensors, prefix + ".connector.", LLM::LLMArch::QWEN2, vision);
            const auto query      = tensors.find(prefix + ".query_tokens_dict.16x16");
            const auto projection = tensors.find(prefix + ".proj_out.weight");
            const auto direct     = tensors.find(prefix + ".proj_directvlm.1.weight");
            if (query == tensors.end() || projection == tensors.end() || direct == tensors.end()) {
                throw std::runtime_error("Ming-Image requires the learned queries, connector and both condition projections");
            }
            config.num_queries   = query->second.ne[1];
            config.caption_dim   = projection->second.ne[1];
            config.diffusion_dim = direct->second.ne[1];
            if (config.num_queries != 256 || config.caption_dim != 2560 || config.diffusion_dim != 3840 ||
                config.backbone.num_layers != 20 || config.backbone.hidden_size != 2048 ||
                config.connector.num_layers != 28 || config.connector.hidden_size != 1536) {
                throw std::runtime_error("unsupported Ming-Image text encoder configuration");
            }
            LOG_VERBOSE("ming_image_te: queries = %" PRId64 ", caption_dim = %" PRId64 ", diffusion_dim = %" PRId64,
                        config.num_queries, config.caption_dim, config.diffusion_dim);
            return config;
        }
    };

    struct ConnectorModel : LLM::TextModel {
        explicit ConnectorModel(const LLM::LLMConfig& config)
            : LLM::TextModel(config, "ming_image.connector") {
            blocks.erase("embed_tokens");
        }
    };

    class MingImageTextModel : public GGMLBlock {
        MingImageTEConfig config;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensors = {}, const std::string prefix = "") override {
            params["query_tokens_dict.16x16"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.backbone.hidden_size, config.num_queries);
        }

    public:
        explicit MingImageTextModel(const MingImageTEConfig& config)
            : config(config) {
            blocks["backbone"]         = std::make_shared<LLM::TextModel>(config.backbone, "ming_image.backbone");
            blocks["connector"]        = std::make_shared<ConnectorModel>(config.connector);
            blocks["proj_in"]          = std::make_shared<Linear>(config.backbone.hidden_size, config.connector.hidden_size);
            blocks["proj_out"]         = std::make_shared<Linear>(config.connector.hidden_size, config.caption_dim);
            blocks["proj_directvlm.0"] = std::make_shared<RMSNorm>(config.backbone.hidden_size * 3, 1e-5f);
            blocks["proj_directvlm.1"] = std::make_shared<Linear>(config.backbone.hidden_size * 3, config.diffusion_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* ids, int64_t prompt_length, ggml_tensor* mask, ggml_tensor* image_mask, ggml_tensor* cos, ggml_tensor* sin, ggml_tensor* connector_positions) {
            auto gctx                 = ctx->ggml_ctx;
            auto backbone             = std::dynamic_pointer_cast<LLM::TextModel>(blocks["backbone"]);
            auto connector            = std::dynamic_pointer_cast<ConnectorModel>(blocks["connector"]);
            auto x                    = backbone->embed(ctx, ids);
            const int64_t query_start = prompt_length + 1;
            auto before               = ggml_ext_slice(gctx, x, 1, 0, query_start);
            auto after                = ggml_ext_slice(gctx, x, 1, query_start + config.num_queries, x->ne[1]);
            x                         = ggml_concat(gctx, ggml_concat(gctx, before, params["query_tokens_dict.16x16"], 1), after, 1);
            // HF hidden_states[20] includes final RMSNorm; sd.cpp selects it as num_layers + 1.
            x            = backbone->forward_embeds(ctx, x, nullptr, mask, {5, 12, 21}, {}, nullptr, false, image_mask, cos, sin);
            auto direct  = ggml_ext_slice(gctx, x, 1, 0, prompt_length);
            direct       = std::dynamic_pointer_cast<RMSNorm>(blocks["proj_directvlm.0"])->forward(ctx, direct);
            direct       = std::dynamic_pointer_cast<Linear>(blocks["proj_directvlm.1"])->forward(ctx, direct);
            auto queries = ggml_ext_slice(gctx, x, 0, config.backbone.hidden_size * 2, config.backbone.hidden_size * 3);
            queries      = ggml_ext_slice(gctx, queries, 1, query_start, query_start + config.num_queries);
            queries      = std::dynamic_pointer_cast<Linear>(blocks["proj_in"])->forward(ctx, queries);
            queries      = connector->forward_embeds(ctx, queries, connector_positions, nullptr, {});
            queries      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, queries);
            queries      = ggml_pad(gctx, queries, static_cast<int>(config.diffusion_dim - config.caption_dim), 0, 0, 0);
            return ggml_concat(gctx, queries, direct, 1);
        }
    };

    struct MingImageTextRunner : GGMLRunner {
        MingImageTEConfig config;
        MingImageTextModel model;

        MingImageTextRunner(ggml_backend_t backend, const String2TensorStorage& tensors, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager), config(MingImageTEConfig::detect_from_weights(tensors, prefix)), model(config) {
            model.init(params_ctx, tensors, prefix);
        }

        std::string get_desc() override { return "ming_image_text"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        void get_param_tensor_ops(std::map<ggml_tensor*, enum ggml_op>& ops) {
            model.get_param_tensor_ops(ops);
        }

        sd::Tensor<float> compute(int n_threads, const std::vector<int>& tokens) {
            const int64_t prompt_length = tokens.size();
            const int64_t query_start   = prompt_length + 1;
            const int64_t total         = prompt_length + config.num_queries + 2;
            std::vector<int32_t> ids(tokens.begin(), tokens.end());
            ids.push_back(157158);
            ids.insert(ids.end(), config.num_queries, 157157);
            ids.push_back(157159);
            auto input = sd::Tensor<int32_t>({total}, std::move(ids));
            sd::Tensor<float> attention_mask({total, total});
            sd::Tensor<float> image_mask({1, total});
            sd::Tensor<float> cos({32, 1, total}), sin({32, 1, total});
            std::vector<int32_t> positions(config.num_queries);
            std::iota(positions.begin(), positions.end(), 0);
            auto connector_positions = sd::Tensor<int32_t>({config.num_queries}, std::move(positions));
            for (int64_t token = 0; token < total; ++token) {
                const bool query  = token >= query_start && token < query_start + config.num_queries;
                image_mask[token] = query ? 1.f : 0.f;
                for (int64_t key = 0; key < total; ++key) {
                    attention_mask[key + token * total] = key > token ? -INFINITY : 0.f;
                }
                // A 16x16 query bank is represented upstream as a [1, 2, 512] image grid,
                // then spatially merged and centered to [1, 1, 256].
                int64_t temporal = query ? query_start : (token == total - 1 ? query_start + 1 : token);
                int64_t width    = query ? token - 127 : temporal;
                for (int j = 0; j < 32; ++j) {
                    int64_t position    = query && j < 24 && j % 2 ? width : temporal;
                    float frequency     = 1.f / std::pow(600000.f, static_cast<float>(2 * j) / 64.f);
                    float angle         = static_cast<float>(position) * frequency;
                    cos[token * 32 + j] = std::cos(angle);
                    sin[token * 32 + j] = std::sin(angle);
                }
            }
            auto graph = [&]() {
                auto gf  = new_graph_custom(LLM::LLM_GRAPH_SIZE);
                auto ctx = get_context();
                auto out = model.forward(&ctx, make_input(input), prompt_length, make_input(attention_mask),
                                         make_input(image_mask), make_input(cos), make_input(sin), make_input(connector_positions));
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return take_or_empty(GGMLRunner::compute(graph, n_threads));
        }
    };
}

#endif  // __SD_MODEL_TE_MING_IMAGE_TE_HPP__

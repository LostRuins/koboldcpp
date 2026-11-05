#ifndef __SD_MODEL_TE_PH3_IMAGE_CONNECTOR_HPP__
#define __SD_MODEL_TE_PH3_IMAGE_CONNECTOR_HPP__

#include "core/ggml_runner.h"
#include "model/common/ggml_block.hpp"

struct PH3ImageTextConnector : public UnaryBlock {
    static constexpr int64_t input_dim  = 5 * 2560;
    static constexpr int64_t output_dim = 5120;

    PH3ImageTextConnector() {
        blocks["proj"] = std::make_shared<Linear>(input_dim, output_dim, false, true, true);
    }

    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        for (const auto* name : {"input_mean", "input_std"}) {
            params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
        }
        for (const auto* name : {"output_mean", "output_std", "sink_embedding"}) {
            params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
        x         = ggml_div(ctx->ggml_ctx, ggml_sub(ctx->ggml_ctx, x, params["input_mean"]), params["input_std"]);
        x         = proj->forward(ctx, x);
        x         = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, params["output_std"]), params["output_mean"]);
        auto sink = ggml_repeat(ctx->ggml_ctx, params["sink_embedding"], ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, 1));
        if (x->ne[1] == 1) {
            return sink;
        }
        return ggml_concat(ctx->ggml_ctx, sink, ggml_ext_slice(ctx->ggml_ctx, x, 1, 1, x->ne[1]), 1);
    }
};

struct PH3ImageTextConnectorRunner : public GGMLRunner {
    PH3ImageTextConnector model;
    static constexpr const char* prefix = "text_encoders.ph3_image_connector";

    PH3ImageTextConnectorRunner(ggml_backend_t backend,
                                const String2TensorStorage& tensor_storage_map,
                                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "ph3_image_text_connector";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors, prefix);
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& hidden_states) {
        auto get_graph = [&]() {
            auto gf         = ggml_new_graph(compute_ctx);
            auto x          = make_input(hidden_states);
            auto runner_ctx = get_context();
            ggml_build_forward_expand(gf, model.forward(&runner_ctx, x));
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, false));
    }
};

#endif  // __SD_MODEL_TE_PH3_IMAGE_CONNECTOR_HPP__

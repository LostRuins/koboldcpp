#ifndef __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__
#define __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"
#include "model_loader.h"

namespace IPAdapter {

    struct ImageProjModel : public GGMLBlock {
        int64_t num_tokens = 4;
        int64_t ctx_dim    = 768;
        int64_t clip_dim   = 1024;

        ImageProjModel() {}
        ImageProjModel(int64_t num_tokens, int64_t ctx_dim, int64_t clip_dim)
            : num_tokens(num_tokens), ctx_dim(ctx_dim), clip_dim(clip_dim) {
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(clip_dim, num_tokens * ctx_dim, true));
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(ctx_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image_embeds) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);

            int64_t n = image_embeds->ne[1];
            auto x    = proj->forward(ctx, image_embeds);
            x         = ggml_reshape_3d(ctx->ggml_ctx, x, ctx_dim, num_tokens, n);
            x         = norm->forward(ctx, x);
            return x;
        }
    };

    struct IPAdapterRunner : public GGMLRunner {
        ImageProjModel image_proj;
        int64_t num_tokens = 4;
        std::string prefix;

        IPAdapterRunner(ggml_backend_t backend,
                        const String2TensorStorage& tensor_storage_map,
                        const std::string prefix,
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager), prefix(prefix) {
            int64_t ctx_dim  = 768;
            int64_t clip_dim = 1024;
            int64_t out_dim  = 3072;
            auto norm_iter   = tensor_storage_map.find(prefix + ".image_proj.norm.weight");
            if (norm_iter != tensor_storage_map.end()) {
                ctx_dim = norm_iter->second.ne[0];
            }
            auto proj_iter = tensor_storage_map.find(prefix + ".image_proj.proj.weight");
            if (proj_iter != tensor_storage_map.end()) {
                clip_dim = proj_iter->second.ne[0];
                out_dim  = proj_iter->second.ne[1];
            }
            num_tokens = out_dim / ctx_dim;
            image_proj = ImageProjModel(num_tokens, ctx_dim, clip_dim);
            image_proj.init(params_ctx, tensor_storage_map, prefix + ".image_proj");
        }

        std::string get_desc() override {
            return "ip_adapter";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string = "") {
            image_proj.get_param_tensors(tensors, prefix + ".image_proj");
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& image_embeds_tensor) {
            ggml_cgraph* gf     = new_graph_custom(1024);
            ggml_tensor* embeds = make_input(image_embeds_tensor);
            auto runner_ctx     = get_context();
            ggml_tensor* out    = image_proj.forward(&runner_ctx, embeds);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& image_embeds) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(image_embeds);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true, true, true));
        }
    };

}  // namespace IPAdapter

#endif  // __SD_MODEL_ADAPTER_IP_ADAPTER_HPP__

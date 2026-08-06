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

    struct Resampler : public GGMLBlock {
        int64_t dim         = 1280;
        int64_t depth       = 4;
        int64_t num_queries = 16;
        int64_t embed_dim   = 1280;
        int64_t output_dim  = 2048;
        int64_t ff_inner    = 5120;
        int64_t dim_head    = 64;
        int64_t heads       = 20;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["latents"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, num_queries, 1);
        }

        Resampler() {}
        Resampler(int64_t dim, int64_t depth, int64_t num_queries, int64_t embed_dim, int64_t output_dim, int64_t ff_inner)
            : dim(dim), depth(depth), num_queries(num_queries), embed_dim(embed_dim), output_dim(output_dim), ff_inner(ff_inner) {
            heads              = dim / dim_head;
            blocks["proj_in"]  = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, dim, true));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(dim, output_dim, true));
            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new LayerNorm(output_dim));
            for (int64_t i = 0; i < depth; i++) {
                std::string p           = "layers." + std::to_string(i);
                blocks[p + ".0.norm1"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
                blocks[p + ".0.norm2"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
                blocks[p + ".0.to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, false));
                blocks[p + ".0.to_kv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 2, false));
                blocks[p + ".0.to_out"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, false));
                blocks[p + ".1.0"]      = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
                blocks[p + ".1.1"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, ff_inner, false));
                blocks[p + ".1.3"]      = std::shared_ptr<GGMLBlock>(new Linear(ff_inner, dim, false));
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* image_embeds) {
            int64_t N     = image_embeds->ne[2];
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            auto norm_out = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out"]);

            ggml_tensor* x       = proj_in->forward(ctx, image_embeds);
            ggml_tensor* latents = params["latents"];
            if (N > 1) {
                latents = ggml_repeat(ctx->ggml_ctx, latents, ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, dim, num_queries, N));
            }

            for (int64_t i = 0; i < depth; i++) {
                std::string p = "layers." + std::to_string(i);
                auto norm1    = std::dynamic_pointer_cast<LayerNorm>(blocks[p + ".0.norm1"]);
                auto norm2    = std::dynamic_pointer_cast<LayerNorm>(blocks[p + ".0.norm2"]);
                auto to_q     = std::dynamic_pointer_cast<Linear>(blocks[p + ".0.to_q"]);
                auto to_kv    = std::dynamic_pointer_cast<Linear>(blocks[p + ".0.to_kv"]);
                auto to_out   = std::dynamic_pointer_cast<Linear>(blocks[p + ".0.to_out"]);

                ggml_tensor* xn    = norm1->forward(ctx, x);
                ggml_tensor* ln    = norm2->forward(ctx, latents);
                ggml_tensor* q     = to_q->forward(ctx, ln);
                ggml_tensor* kv_in = ggml_concat(ctx->ggml_ctx, xn, ln, 1);
                ggml_tensor* kv    = to_kv->forward(ctx, kv_in);
                int64_t L          = kv->ne[1];
                ggml_tensor* k     = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, kv, dim, L, N, kv->nb[1], kv->nb[2], 0));
                ggml_tensor* v     = ggml_cont(ctx->ggml_ctx, ggml_view_3d(ctx->ggml_ctx, kv, dim, L, N, kv->nb[1], kv->nb[2], dim * kv->nb[0]));
                ggml_tensor* attn  = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, heads, nullptr, false, false);
                attn               = to_out->forward(ctx, attn);
                latents            = ggml_add(ctx->ggml_ctx, latents, attn);

                auto ff_norm   = std::dynamic_pointer_cast<LayerNorm>(blocks[p + ".1.0"]);
                auto ff_fc1    = std::dynamic_pointer_cast<Linear>(blocks[p + ".1.1"]);
                auto ff_fc2    = std::dynamic_pointer_cast<Linear>(blocks[p + ".1.3"]);
                ggml_tensor* h = ff_norm->forward(ctx, latents);
                h              = ff_fc1->forward(ctx, h);
                h              = ggml_gelu_erf(ctx->ggml_ctx, h);
                h              = ff_fc2->forward(ctx, h);
                latents        = ggml_add(ctx->ggml_ctx, latents, h);
            }

            latents = proj_out->forward(ctx, latents);
            latents = norm_out->forward(ctx, latents);
            return latents;
        }
    };

    struct IPAdapterRunner : public GGMLRunner {
        ImageProjModel image_proj;
        Resampler resampler;
        bool is_plus       = false;
        int64_t num_tokens = 4;
        std::string prefix;

        IPAdapterRunner(ggml_backend_t backend,
                        const String2TensorStorage& tensor_storage_map,
                        const std::string prefix,
                        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager), prefix(prefix) {
            is_plus = tensor_storage_map.find(prefix + ".image_proj.latents") != tensor_storage_map.end();
            if (is_plus) {
                int64_t dim         = 1280;
                int64_t num_queries = 16;
                int64_t embed_dim   = 1280;
                int64_t output_dim  = 2048;
                int64_t ff_inner    = 5120;
                auto latents_iter   = tensor_storage_map.find(prefix + ".image_proj.latents");
                if (latents_iter != tensor_storage_map.end()) {
                    dim         = latents_iter->second.ne[0];
                    num_queries = latents_iter->second.ne[1];
                }
                auto proj_in_iter = tensor_storage_map.find(prefix + ".image_proj.proj_in.weight");
                if (proj_in_iter != tensor_storage_map.end()) {
                    embed_dim = proj_in_iter->second.ne[0];
                }
                auto proj_out_iter = tensor_storage_map.find(prefix + ".image_proj.proj_out.weight");
                if (proj_out_iter != tensor_storage_map.end()) {
                    output_dim = proj_out_iter->second.ne[1];
                }
                auto ff_iter = tensor_storage_map.find(prefix + ".image_proj.layers.0.1.1.weight");
                if (ff_iter != tensor_storage_map.end()) {
                    ff_inner = ff_iter->second.ne[1];
                }
                int64_t depth = 0;
                while (tensor_storage_map.find(prefix + ".image_proj.layers." + std::to_string(depth) + ".0.to_q.weight") != tensor_storage_map.end()) {
                    depth++;
                }
                num_tokens = num_queries;
                resampler  = Resampler(dim, depth, num_queries, embed_dim, output_dim, ff_inner);
                resampler.init(params_ctx, tensor_storage_map, prefix + ".image_proj");
            } else {
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
        }

        std::string get_desc() override {
            return "ip_adapter";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string = "") {
            if (is_plus) {
                resampler.get_param_tensors(tensors, prefix + ".image_proj");
            } else {
                image_proj.get_param_tensors(tensors, prefix + ".image_proj");
            }
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& image_embeds_tensor) {
            ggml_cgraph* gf     = new_graph_custom(1024);
            ggml_tensor* embeds = make_input(image_embeds_tensor);
            auto runner_ctx     = get_context();
            ggml_tensor* out    = is_plus ? resampler.forward(&runner_ctx, embeds) : image_proj.forward(&runner_ctx, embeds);
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

#ifndef __PULID_HPP__
#define __PULID_HPP__

#include "core/ggml_extend.hpp"
#include "model/common/block.hpp"

class PuLIDPerceiverAttentionCA : public GGMLBlock {
public:
    static constexpr int64_t DEFAULT_DIM      = 3072;  // Flux hidden size
    static constexpr int64_t DEFAULT_DIM_HEAD = 128;
    static constexpr int64_t DEFAULT_HEADS    = 16;
    static constexpr int64_t DEFAULT_KV_DIM   = 2048;  // PuLID ID-embedding dim

protected:
    int64_t dim;
    int64_t dim_head;
    int64_t heads;
    int64_t kv_dim;
    int64_t inner_dim;

public:
    PuLIDPerceiverAttentionCA(int64_t dim      = DEFAULT_DIM,
                              int64_t dim_head = DEFAULT_DIM_HEAD,
                              int64_t heads    = DEFAULT_HEADS,
                              int64_t kv_dim   = DEFAULT_KV_DIM)
        : dim(dim),
          dim_head(dim_head),
          heads(heads),
          kv_dim(kv_dim),
          inner_dim(dim_head * heads) {
        blocks["norm1"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(kv_dim));
        blocks["norm2"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(dim));
        blocks["to_q"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, inner_dim, /*bias=*/false));
        blocks["to_kv"]  = std::shared_ptr<GGMLBlock>(new Linear(kv_dim, inner_dim * 2, /*bias=*/false));
        blocks["to_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, dim, /*bias=*/false));
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* id_embedding,
                         ggml_tensor* image_tokens) {
        auto norm1  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto norm2  = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
        auto to_kv  = std::dynamic_pointer_cast<Linear>(blocks["to_kv"]);
        auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);

        ggml_tensor* x_normed   = norm1->forward(ctx, id_embedding);
        ggml_tensor* lat_normed = norm2->forward(ctx, image_tokens);

        ggml_tensor* q  = to_q->forward(ctx, lat_normed);  // [N, T_img, 2048]
        ggml_tensor* kv = to_kv->forward(ctx, x_normed);   // [N, T_img, 3072]

        ggml_tensor* k = ggml_view_3d(ctx->ggml_ctx, kv,
                                      inner_dim, kv->ne[1], kv->ne[2],
                                      kv->nb[1], kv->nb[2],
                                      /*offset=*/0);
        ggml_tensor* v = ggml_view_3d(ctx->ggml_ctx, kv,
                                      inner_dim, kv->ne[1], kv->ne[2],
                                      kv->nb[1], kv->nb[2],
                                      /*offset=*/inner_dim * ggml_element_size(kv));
        k              = ggml_cont(ctx->ggml_ctx, k);
        v              = ggml_cont(ctx->ggml_ctx, v);

        ggml_tensor* attn_out = ggml_ext_attention_ext(
            ctx->ggml_ctx, ctx->backend,
            q, k, v,
            heads,
            /*mask=*/nullptr,
            /*diag_mask_inf=*/false);

        ggml_tensor* out = to_out->forward(ctx, attn_out);
        return out;
    }
};

#endif  // __PULID_HPP__

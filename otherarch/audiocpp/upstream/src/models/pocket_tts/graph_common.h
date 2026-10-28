#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/models/pocket_tts/backend_weights.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::pocket_tts::graph_common {

inline modules::AttentionWeights make_packed_attention_weights(
    core::ModuleBuildContext & ctx,
    const PocketTTSBackendPackedAttentionWeights & weights,
    int64_t hidden_size) {
    if (weights.in_proj_weight.shape.rank != 2 || weights.in_proj_weight.shape.dims[0] != hidden_size * 3
        || weights.in_proj_weight.shape.dims[1] != hidden_size) {
        throw std::runtime_error("Packed attention weight must have shape [3 * hidden_size, hidden_size]");
    }
    const auto view_shape = core::TensorShape::from_dims({hidden_size, hidden_size});
    const size_t row_stride = weights.in_proj_weight.tensor->nb[1];
    auto q = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, hidden_size, row_stride, 0),
        view_shape,
        weights.in_proj_weight.type);
    auto k = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, hidden_size, row_stride, static_cast<size_t>(hidden_size) * row_stride),
        view_shape,
        weights.in_proj_weight.type);
    auto v = core::wrap_tensor(
        ggml_view_2d(ctx.ggml, weights.in_proj_weight.tensor, hidden_size, hidden_size, row_stride, static_cast<size_t>(hidden_size * 2) * row_stride),
        view_shape,
        weights.in_proj_weight.type);
    modules::AttentionWeights attention = {};
    attention.q_weight = q;
    attention.k_weight = k;
    attention.v_weight = v;
    attention.out_weight = weights.out_proj_weight;
    return attention;
}

inline modules::TransformerEncoderBlockWeights make_transformer_block_weights(
    core::ModuleBuildContext & ctx,
    const PocketTTSBackendTransformerLayerWeights & weights,
    int64_t hidden_size) {
    return modules::TransformerEncoderBlockWeights{
        weights.norm1,
        make_packed_attention_weights(ctx, weights.self_attn, hidden_size),
        weights.layer_scale_1,
        weights.norm2,
        weights.feed_forward,
        weights.layer_scale_2,
    };
}

template <typename BuildFn, typename InitFn, typename ReadFn>
auto run_graph_with_backend(
    const char * label,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_context_bytes,
    BuildFn build_fn,
    InitFn init_fn,
    ReadFn read_fn)
    -> decltype(read_fn(std::declval<ggml_context *>())) {
    ggml_context * ggml_ctx = ggml_init({graph_context_bytes, nullptr, true});
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("Failed to initialize ggml context");
    }
    core::ModuleBuildContext ctx{ggml_ctx, "pocket_tts", backend_type};
    auto output = build_fn(ctx);
    ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, 32768, false);
    ggml_build_forward_expand(graph, output.tensor);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (gallocr == nullptr ||
        !ggml_gallocr_reserve(gallocr, graph) ||
        !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        ggml_free(ggml_ctx);
        throw std::runtime_error(std::string(label) + " graph allocation failed");
    }
    init_fn();
    core::set_backend_threads(backend, threads);
    const ggml_status status = engine::core::compute_backend_graph(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        ggml_free(ggml_ctx);
        throw std::runtime_error(std::string(label) + " compute failed: " + ggml_status_to_string(status));
    }
    auto result = read_fn(ggml_ctx);
    ggml_gallocr_free(gallocr);
    ggml_free(ggml_ctx);
    return result;
}

}  // namespace engine::models::pocket_tts::graph_common

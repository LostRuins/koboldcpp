#pragma once
// weight_ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf_weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstddef>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void * src;
        size_t nbytes;
        size_t offset;  // byte offset into dst tensor (0 for regular loads)
    };
    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data (kept alive until wctx_alloc)
    std::vector<std::vector<float>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t ctx_size = (size_t)n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    wctx->ctx = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
    }
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n",
            wctx->pending.size(), (float)total / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) ggml_backend_buffer_free(wctx->buffer);
    if (wctx->ctx) ggml_free(wctx->ctx);
    wctx->buffer = NULL;
    wctx->ctx = NULL;
}

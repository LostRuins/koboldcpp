#include "engine/framework/runtime/graph_executor.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::runtime {

namespace {

constexpr size_t kDefaultGraphNodeCapacity = 32768;

}

GraphExecutor::GraphExecutor(engine::core::ExecutionContext & execution_context)
    : execution_context_(execution_context),
      gallocr_(ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(execution_context_.backend()))) {}

GraphExecutor::~GraphExecutor() {
    if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
}

GraphRunResult GraphExecutor::run(
    ggml_context * ggml_ctx,
    ggml_tensor * output,
    int warmup,
    int iterations) {
    if (ggml_ctx == nullptr || output == nullptr) {
        throw std::runtime_error("GraphExecutor::run requires non-null graph context and output tensor");
    }

    if (cached_ggml_ctx_ != ggml_ctx || cached_output_ != output || cached_graph_ == nullptr) {
        cached_ggml_ctx_ = ggml_ctx;
        cached_output_ = output;
        cached_graph_ = ggml_new_graph_custom(ggml_ctx, kDefaultGraphNodeCapacity, false);
        ggml_build_forward_expand(cached_graph_, output);
        ggml_gallocr_alloc_graph(gallocr_, cached_graph_);
        cached_graph_allocated_ = true;
    } else if (!cached_graph_allocated_) {
        ggml_gallocr_alloc_graph(gallocr_, cached_graph_);
        cached_graph_allocated_ = true;
    }

    for (int i = 0; i < warmup; ++i) {
        engine::core::compute_backend_graph(execution_context_.backend(), cached_graph_);
    }

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < std::max(1, iterations); ++i) {
        engine::core::compute_backend_graph(execution_context_.backend(), cached_graph_);
    }
    const auto ended = std::chrono::steady_clock::now();

    GraphRunResult result;
    result.average_ms =
        std::chrono::duration<double, std::milli>(ended - started).count() / std::max(1, iterations);
    return result;
}

}  // namespace engine::runtime

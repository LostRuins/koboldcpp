#pragma once

#include "ggml-alloc.h"

namespace engine::core {
class ExecutionContext;
}

namespace engine::runtime {

struct GraphRunResult {
    double average_ms = 0.0;
};

class GraphExecutor {
public:
    explicit GraphExecutor(engine::core::ExecutionContext & execution_context);
    ~GraphExecutor();

    GraphExecutor(const GraphExecutor &) = delete;
    GraphExecutor & operator=(const GraphExecutor &) = delete;

    GraphRunResult run(
        ggml_context * ggml_ctx,
        ggml_tensor * output,
        int warmup,
        int iterations);

private:
    engine::core::ExecutionContext & execution_context_;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_context * cached_ggml_ctx_ = nullptr;
    ggml_tensor * cached_output_ = nullptr;
    ggml_cgraph * cached_graph_ = nullptr;
    bool cached_graph_allocated_ = false;
};

}  // namespace engine::runtime

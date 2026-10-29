#pragma once

#include "engine/framework/core/backend.h"

namespace engine::core {

class ExecutionContext {
public:
    explicit ExecutionContext(const BackendConfig & config);
    ~ExecutionContext();

    ExecutionContext(const ExecutionContext &) = delete;
    ExecutionContext & operator=(const ExecutionContext &) = delete;
    ExecutionContext(ExecutionContext && other) noexcept;
    ExecutionContext & operator=(ExecutionContext && other) noexcept;

    ggml_backend_t backend() const noexcept;
    const BackendConfig & config() const noexcept;
    BackendType backend_type() const noexcept;
    bool uses_host_graph_plan() const noexcept;
    BackendMemorySnapshot memory_snapshot() const;

private:
    BackendConfig config_;
    ggml_backend_t backend_ = nullptr;
};

}  // namespace engine::core

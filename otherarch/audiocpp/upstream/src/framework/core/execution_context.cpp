#include "engine/framework/core/execution_context.h"

#include <utility>

namespace engine::core {

ExecutionContext::ExecutionContext(const BackendConfig & config)
    : config_(config),
      backend_(init_backend(config)) {
    if (backend_ != nullptr && config_.threads > 0) {
        set_backend_threads(backend_, config_.threads);
    }
}

ExecutionContext::~ExecutionContext() {
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

ExecutionContext::ExecutionContext(ExecutionContext && other) noexcept
    : config_(other.config_),
      backend_(other.backend_) {
    other.backend_ = nullptr;
}

ExecutionContext & ExecutionContext::operator=(ExecutionContext && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
    config_ = other.config_;
    backend_ = other.backend_;
    other.backend_ = nullptr;
    return *this;
}

ggml_backend_t ExecutionContext::backend() const noexcept {
    return backend_;
}

const BackendConfig & ExecutionContext::config() const noexcept {
    return config_;
}

BackendType ExecutionContext::backend_type() const noexcept {
    if (config_.type != BackendType::BestAvailable) {
        return config_.type;
    }
    return core::backend_type(backend_);
}

bool ExecutionContext::uses_host_graph_plan() const noexcept {
    return core::uses_host_graph_plan(backend_);
}

BackendMemorySnapshot ExecutionContext::memory_snapshot() const {
    if (config_.type != BackendType::BestAvailable) {
        return query_backend_memory(config_);
    }
    return query_backend_memory(backend_, config_.device);
}

}  // namespace engine::core

#include "engine/framework/runtime/session_base.h"

namespace engine::runtime {

RuntimeSessionBase::RuntimeSessionBase(const SessionOptions & options)
    : options_(options),
      execution_context_(options.backend),
      graph_executor_(execution_context_) {}

engine::core::ExecutionContext & RuntimeSessionBase::execution_context() noexcept {
    return execution_context_;
}

const engine::core::ExecutionContext & RuntimeSessionBase::execution_context() const noexcept {
    return execution_context_;
}

ArtifactStore & RuntimeSessionBase::artifacts() noexcept {
    return artifacts_;
}

const ArtifactStore & RuntimeSessionBase::artifacts() const noexcept {
    return artifacts_;
}

RuntimeCache & RuntimeSessionBase::cache() noexcept {
    return cache_;
}

const RuntimeCache & RuntimeSessionBase::cache() const noexcept {
    return cache_;
}

RuntimeWorkspace & RuntimeSessionBase::workspace() noexcept {
    return workspace_;
}

const RuntimeWorkspace & RuntimeSessionBase::workspace() const noexcept {
    return workspace_;
}

GraphExecutor & RuntimeSessionBase::graph_executor() noexcept {
    return graph_executor_;
}

const GraphExecutor & RuntimeSessionBase::graph_executor() const noexcept {
    return graph_executor_;
}

void RuntimeSessionBase::mark_prepared() noexcept {
    prepared_ = true;
}

void RuntimeSessionBase::require_prepared(std::string_view operation) const {
    if (!prepared_) {
        throw std::runtime_error(
            "session prepare() must be called before " + std::string(operation));
    }
}

void RuntimeSessionBase::trace(engine::debug::LogLevel level, std::string_view category, std::string_view message) const {
    engine::debug::log_message(level, category, message);
}

engine::debug::ScopeTimer RuntimeSessionBase::profile(
    engine::debug::LogLevel level,
    std::string_view category,
    std::string_view name) const {
    return engine::debug::ScopeTimer(level, category, name);
}

const SessionOptions & RuntimeSessionBase::options() const noexcept {
    return options_;
}

}  // namespace engine::runtime

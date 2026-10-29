#pragma once

#include "engine/framework/runtime/kv_cache.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::runtime {

struct CachedDecodeStep {
    int64_t position = 0;
    int64_t valid_steps = 0;
    int64_t cache_steps = 0;
    int32_t cache_slot = 0;
    int64_t logical_start = 0;
    bool wrapped = false;
};

template <typename Graph, typename Policy>
class CachedDecodeRuntime {
public:
    using Factory = std::function<std::unique_ptr<Graph>(int64_t cache_steps)>;
    using Config = typename Policy::Config;

    CachedDecodeRuntime() = default;

    explicit CachedDecodeRuntime(Config config)
        : policy_(std::move(config)) {}

    bool prepare_for_prefill(int64_t required_steps, const Factory & factory) {
        const int64_t target_steps = target_cache_steps(required_steps);
        if (graph_ != nullptr && graph_->cache_steps() == target_steps) {
            return false;
        }
        graph_ = factory(target_steps);
        policy_.reset();
        return true;
    }

    bool grow_for_next_step(int64_t required_steps, const Factory & factory) {
        const int64_t target_steps = target_cache_steps(required_steps);
        if (graph_ != nullptr && graph_->cache_steps() >= target_steps) {
            return false;
        }
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " requires an initialized graph before cache growth");
        }
        auto state = graph_->export_state();
        graph_ = factory(target_steps);
        import_state(state);
        return true;
    }

    void import_state(const TransformerKVState & state) {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " requires a graph before importing KV state");
        }
        graph_->import_state(state);
        policy_.import_state(state, graph_->cache_steps());
    }

    void reset_to_empty_cache() {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " requires a graph before reset");
        }
        graph_->reset_state();
        policy_.reset_to_empty(graph_->cache_steps());
    }

    CachedDecodeStep next_step() const {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " requires an initialized graph before decode");
        }
        return policy_.next_step();
    }

    void advance_after_direct_append(int64_t steps) {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " requires an initialized graph before cache advance");
        }
        const auto before = policy_.next_step();
        if (policy_.should_advance_graph_cache(before, steps)) {
            graph_->advance_cache_after_direct_append(steps);
        }
        policy_.advance_after_direct_append(steps);
    }

    bool has_graph() const noexcept {
        return graph_ != nullptr;
    }

    Graph & graph() {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " graph is not initialized");
        }
        return *graph_;
    }

    const Graph & graph() const {
        if (graph_ == nullptr) {
            throw std::runtime_error(policy_.label() + " graph is not initialized");
        }
        return *graph_;
    }

    int64_t target_cache_steps(int64_t required_steps) const {
        return policy_.target_cache_steps(required_steps);
    }

private:
    Policy policy_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::runtime

#pragma once

#include "engine/framework/runtime/cached_decode.h"

#include <string>
#include <utility>

namespace engine::runtime {

struct BoundedStaticKVDecodeConfig {
    int64_t sliding_window = 0;
    int64_t min_cache_steps = 256;
    std::string label = "bounded static KV decode";
};

using BoundedStaticKVDecodeStep = CachedDecodeStep;

int64_t bounded_static_kv_cache_steps_for_required(
    const BoundedStaticKVDecodeConfig & config,
    int64_t required_steps);

class BoundedStaticKVDecodeCursor {
public:
    void import_state(const TransformerKVState & state, int64_t cache_steps);
    void reset_to_empty(int64_t cache_steps);
    BoundedStaticKVDecodeStep next_step() const;
    void advance_after_direct_append(int64_t steps);
    void reset() noexcept;

private:
    int64_t valid_steps_ = 0;
    int64_t absolute_end_ = 0;
    int64_t cache_steps_ = 0;
};

class BoundedStaticKVDecodePolicy {
public:
    using Config = BoundedStaticKVDecodeConfig;

    BoundedStaticKVDecodePolicy() = default;
    explicit BoundedStaticKVDecodePolicy(BoundedStaticKVDecodeConfig config)
        : config_(std::move(config)) {}

    int64_t target_cache_steps(int64_t required_steps) const {
        return bounded_static_kv_cache_steps_for_required(config_, required_steps);
    }

    void reset() noexcept {
        cursor_.reset();
    }

    void import_state(const TransformerKVState & state, int64_t cache_steps) {
        cursor_.import_state(state, cache_steps);
    }

    void reset_to_empty(int64_t cache_steps) {
        cursor_.reset_to_empty(cache_steps);
    }

    CachedDecodeStep next_step() const {
        return cursor_.next_step();
    }

    bool should_advance_graph_cache(const CachedDecodeStep & step, int64_t /*steps*/) const noexcept {
        return step.valid_steps < step.cache_steps;
    }

    void advance_after_direct_append(int64_t steps) {
        cursor_.advance_after_direct_append(steps);
    }

    const std::string & label() const noexcept {
        return config_.label;
    }

private:
    BoundedStaticKVDecodeConfig config_;
    BoundedStaticKVDecodeCursor cursor_;
};

template <typename Graph>
using BoundedStaticKVDecodeRuntime = CachedDecodeRuntime<Graph, BoundedStaticKVDecodePolicy>;

}  // namespace engine::runtime

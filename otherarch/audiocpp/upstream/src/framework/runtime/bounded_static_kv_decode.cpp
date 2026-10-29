#include "engine/framework/runtime/bounded_static_kv_decode.h"

#include <algorithm>
#include <stdexcept>

namespace engine::runtime {

int64_t bounded_static_kv_cache_steps_for_required(
    const BoundedStaticKVDecodeConfig & config,
    int64_t required_steps) {
    if (config.sliding_window <= 0) {
        throw std::runtime_error(config.label + " requires a positive sliding_window");
    }
    if (config.min_cache_steps <= 0) {
        throw std::runtime_error(config.label + " requires a positive min_cache_steps");
    }
    int64_t target = std::min<int64_t>(std::max<int64_t>(required_steps, 1), config.sliding_window);
    target = std::max<int64_t>(target, std::min<int64_t>(config.min_cache_steps, config.sliding_window));
    int64_t bucket = 1;
    while (bucket < target && bucket < config.sliding_window) {
        bucket *= 2;
    }
    return std::min<int64_t>(bucket, config.sliding_window);
}

void BoundedStaticKVDecodeCursor::import_state(const TransformerKVState & state, int64_t cache_steps) {
    if (cache_steps <= 0) {
        throw std::runtime_error("BoundedStaticKVDecodeCursor requires positive cache_steps");
    }
    const int64_t state_steps = state.layers.empty() ? 0 : state.layers.front().valid_steps;
    if (state_steps > cache_steps) {
        throw std::runtime_error("BoundedStaticKVDecodeCursor state exceeds cache capacity");
    }
    for (const auto & layer : state.layers) {
        if (layer.valid_steps != state_steps) {
            throw std::runtime_error("BoundedStaticKVDecodeCursor requires consistent per-layer valid_steps");
        }
    }
    valid_steps_ = state_steps;
    absolute_end_ = state.current_end;
    cache_steps_ = cache_steps;
}

void BoundedStaticKVDecodeCursor::reset_to_empty(int64_t cache_steps) {
    if (cache_steps <= 0) {
        throw std::runtime_error("BoundedStaticKVDecodeCursor requires positive cache_steps");
    }
    valid_steps_ = 0;
    absolute_end_ = 0;
    cache_steps_ = cache_steps;
}

BoundedStaticKVDecodeStep BoundedStaticKVDecodeCursor::next_step() const {
    if (cache_steps_ <= 0) {
        throw std::runtime_error("BoundedStaticKVDecodeCursor requires imported state before decode");
    }
    return {
        absolute_end_,
        valid_steps_,
        cache_steps_,
        static_cast<int32_t>(valid_steps_ < cache_steps_ ? valid_steps_ : (absolute_end_ % cache_steps_)),
    };
}

void BoundedStaticKVDecodeCursor::advance_after_direct_append(int64_t steps) {
    if (steps <= 0) {
        return;
    }
    if (cache_steps_ <= 0) {
        throw std::runtime_error("BoundedStaticKVDecodeCursor requires imported state before advance");
    }
    if (valid_steps_ < cache_steps_) {
        valid_steps_ = std::min<int64_t>(cache_steps_, valid_steps_ + steps);
    }
    absolute_end_ += steps;
}

void BoundedStaticKVDecodeCursor::reset() noexcept {
    valid_steps_ = 0;
    absolute_end_ = 0;
    cache_steps_ = 0;
}

}  // namespace engine::runtime

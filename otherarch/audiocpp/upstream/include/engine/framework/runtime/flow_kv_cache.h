#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::runtime {

struct FlowKVLayerCache {
    ggml_type type = GGML_TYPE_F32;
    std::vector<std::byte> key_bytes;
    std::vector<std::byte> value_bytes;
    int64_t valid_steps = 0;
    int64_t capacity_steps = 0;
    int64_t step_values = 0;
    size_t step_bytes = 0;
};

class RollingFlowKVCache {
public:
    RollingFlowKVCache() = default;

    RollingFlowKVCache(
        const std::vector<int64_t> & layer_capacities,
        int64_t step_values);
    RollingFlowKVCache(
        const std::vector<int64_t> & layer_capacities,
        int64_t step_values,
        ggml_type type);

    void reset(
        const std::vector<int64_t> & layer_capacities,
        int64_t step_values);
    void reset(
        const std::vector<int64_t> & layer_capacities,
        int64_t step_values,
        ggml_type type);

    void clear_valid_steps();

    size_t layers() const noexcept {
        return layers_.size();
    }

    const FlowKVLayerCache & layer(size_t index) const;
    FlowKVLayerCache & layer(size_t index);

    void append_tail(
        size_t layer_index,
        const std::vector<float> & current_key,
        const std::vector<float> & current_value,
        int64_t current_steps,
        int64_t append_steps,
        const std::string & label);
    void append_tail_bytes(
        size_t layer_index,
        const std::vector<std::byte> & current_key,
        const std::vector<std::byte> & current_value,
        int64_t current_steps,
        int64_t append_steps,
        const std::string & label);

private:
    std::vector<FlowKVLayerCache> layers_;
};

}  // namespace engine::runtime

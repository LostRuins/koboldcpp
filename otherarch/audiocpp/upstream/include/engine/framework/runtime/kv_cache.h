#pragma once

#include "engine/framework/core/module.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::runtime {

struct KVLayerState {
    int64_t valid_steps = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct TransformerKVState {
    int64_t current_end = 0;
    std::vector<KVLayerState> layers;
};

// Absolute position of the i-th stored row in a pinned-prefix ring:
// pinned prefix first, then oldest-to-newest tail.
inline int64_t ring_stored_position(
    int64_t row,
    int64_t valid_steps,
    int64_t current_end,
    int64_t pinned_prefix_steps) {
    return row < pinned_prefix_steps ? row : current_end - valid_steps + row;
}

// Bucketed capacity for a static-cache AR loop: prompt rows plus a small
// generated window to start, doubling buckets up to prompt + max_tokens. A
// family that sizes its step graph from this number builds (and attends over)
// the cache the request can fill, not the model's full cache length.
inline int64_t bucketed_cache_steps(
    int64_t prompt_steps,
    int64_t max_tokens,
    int64_t initial_generated_steps = 128,
    int64_t minimum_bucket_steps = 128) {
    const int64_t maximum = prompt_steps + max_tokens;
    const int64_t required = prompt_steps + std::min(max_tokens, initial_generated_steps);
    int64_t bucket = minimum_bucket_steps;
    while (bucket < required && bucket <= maximum / 2) {
        bucket *= 2;
    }
    if (bucket < required) {
        bucket = required;
    }
    return std::min(bucket, maximum);
}

struct TransformerKVCacheOptions {
    bool allow_f16_storage = false;
    bool allow_bf16_storage = false;
    // In-place ring: once full, appends overwrite the oldest unpinned
    // entries instead of failing. The first ring_pinned_steps slots are
    // reserved (e.g. prompt / attention sinks); positions stay absolute.
    bool ring_mode = false;
    int64_t ring_pinned_steps = 0;
    // Device-only users can defer host staging allocation until import_state.
    bool lazy_import_scratch = false;
};

class TransformerKVCache {
public:
    TransformerKVCache() = default;
    TransformerKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values);
    TransformerKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values,
        TransformerKVCacheOptions options);

    void import_state(const TransformerKVState & state);
    // Import a contiguous, same-precision prefix starting at position zero.
    // Backend copies preserve device residency; unused cache rows are zeroed.
    void import_prefix(
        const std::vector<core::TensorValue> & keys,
        const std::vector<core::TensorValue> & values,
        int64_t steps);
    void clear_on_backend();
    TransformerKVState export_state() const;

    void advance_after_direct_append(int64_t steps);
    void retain_prefix(int64_t prefix_steps);

    int64_t slot_for_position(int64_t position) const;

    int64_t valid_steps() const noexcept;
    int64_t current_end() const noexcept;
    int64_t cache_steps() const noexcept;

    const core::TensorValue & key_tensor(size_t layer) const;
    const core::TensorValue & value_tensor(size_t layer) const;

    void trace_log_state(const std::string & name, int64_t num_heads, int64_t head_dim) const;

private:
    struct LayerCache {
        core::TensorValue key_tensor;
        core::TensorValue value_tensor;
        std::vector<float> import_key_scratch;
        std::vector<float> import_value_scratch;
    };

    int64_t cache_steps_ = 0;
    int64_t step_elems_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    TransformerKVCacheOptions options_;
    std::vector<LayerCache> layers_;
};

struct BatchedKVLayerState {
    int64_t valid_steps = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct TransformerBatchedKVState {
    int64_t batch_size = 0;
    int64_t current_end = 0;
    std::vector<int64_t> current_end_by_batch;
    std::vector<int64_t> valid_steps_by_batch;
    std::vector<BatchedKVLayerState> layers;
};

class TransformerBatchedKVCache {
public:
    TransformerBatchedKVCache() = default;
    TransformerBatchedKVCache(
        int64_t cache_steps,
        int64_t batch_size,
        int64_t row_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values);
    TransformerBatchedKVCache(
        int64_t cache_steps,
        int64_t batch_size,
        int64_t row_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values,
        TransformerKVCacheOptions options);

    void import_state(const TransformerBatchedKVState & state);
    TransformerBatchedKVState export_state() const;

    void advance_after_direct_append(int64_t steps);

    int64_t batch_size() const noexcept;
    int64_t valid_steps() const noexcept;
    int64_t current_end() const noexcept;
    int64_t cache_steps() const noexcept;
    const std::vector<int64_t> & valid_steps_by_batch() const noexcept;
    const std::vector<int64_t> & current_end_by_batch() const noexcept;

private:
    struct LayerCache {
        core::TensorValue key_tensor;
        core::TensorValue value_tensor;
        std::vector<float> import_key_scratch;
        std::vector<float> import_value_scratch;
    };

    int64_t cache_steps_ = 0;
    int64_t batch_size_ = 0;
    int64_t row_elems_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    std::vector<int64_t> valid_steps_by_batch_;
    std::vector<int64_t> current_end_by_batch_;
    TransformerKVCacheOptions options_;
    std::vector<LayerCache> layers_;
};

core::TensorValue view_transformer_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    const char * label,
    ggml_type view_type = GGML_TYPE_F32);

}  // namespace engine::runtime

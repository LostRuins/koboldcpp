#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/types.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace engine::models::yue2 {

struct Yue2ARSamplingWindow {
    int32_t begin = 0;
    int32_t end = 0;
    int32_t stop_token = 0;
    int64_t min_tokens = 0;
    int64_t max_tokens = 0;
    Yue2SamplingConfig sampling;
};

struct Yue2ARDevicePrefixState {
    int64_t current_end = 0;
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

class Yue2ARRuntime {
public:
    Yue2ARRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes);
    ~Yue2ARRuntime();

    std::vector<int32_t> generate(
        const std::vector<int32_t> & prefix,
        const Yue2ARSamplingWindow & window,
        uint64_t seed,
        const std::vector<int32_t> & forced = {});

    std::vector<int32_t> generate_cfg(
        const std::vector<int32_t> & positive_prefix,
        const std::vector<int32_t> & negative_prefix,
        const Yue2ARSamplingWindow & window,
        float guidance_scale,
        uint64_t seed,
        const std::vector<int32_t> & forced = {});

    runtime::TransformerKVState prefill_state(const std::vector<int32_t> & tokens);
    Yue2ARDevicePrefixState prefill_device_state(const std::vector<int32_t> & tokens);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2

#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/neutts/assets.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::neutts {

struct NeuTTSGenerationOptions {
    int64_t max_tokens = 0;
    int64_t min_tokens = 50;
    float temperature = 1.0F;
    int64_t top_k = 50;
    uint64_t seed = 0;
};

struct NeuTTSGeneratedCodes {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> speech_codes;
};

class NeuTTSQwen3ARRuntime {
public:
    struct Impl;

    NeuTTSQwen3ARRuntime(
        std::shared_ptr<const NeuTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~NeuTTSQwen3ARRuntime();

    NeuTTSGeneratedCodes generate(
        const std::vector<int32_t> & prompt_ids,
        int32_t speech_token_start,
        int32_t speech_token_end,
        int32_t speech_generation_end,
        const NeuTTSGenerationOptions & options,
        const std::function<void(int32_t)> & on_speech_code = {});
    void release_runtime_graphs();

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::neutts

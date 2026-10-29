#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/community_models/soprano_tts/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::soprano_tts {
struct SopranoQwen3Weights;

// Autoregressive Qwen3 causal LM wrapper. Captures the last-layer 512-dim
// hidden state of every generated token (the per-frame audio features) plus the
// sampled token ids, stopping on EOS.
class SopranoQwen3Generator {
public:
    SopranoQwen3Generator(
        const SopranoTTSAssets & assets,
        engine::core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~SopranoQwen3Generator();

    struct Result {
        std::vector<float> features;   // frames x hidden (frame-major)
        std::vector<int32_t> tokens;   // generated token ids (excluding EOS)
        int64_t frames = 0;
    };

    Result generate(const std::vector<int32_t> & prompt_ids,
                    const SopranoGenerationOptions & options);

    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::soprano_tts

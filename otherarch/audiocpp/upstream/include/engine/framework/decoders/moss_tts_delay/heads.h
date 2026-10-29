#pragma once

#include "engine/framework/decoders/moss_tts_delay/config.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::decoders {

// The delay family reads 1 + n_vq heads off the same backbone hidden state each step:
// lm_heads.0 predicts the next text token, lm_heads.1..n_vq predict one RVQ code each.
struct MossTtsDelayStepLogits {
    std::vector<float> text;                      // [text_vocab_size]
    std::vector<std::vector<float>> audio;        // n_vq x [audio_vocab_size + 1]
};

class MossTtsDelayHeadsRuntime {
public:
    MossTtsDelayHeadsRuntime(
        MossTtsDelayConfig config,
        std::shared_ptr<const assets::TensorSource> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~MossTtsDelayHeadsRuntime();

    MossTtsDelayHeadsRuntime(const MossTtsDelayHeadsRuntime &) = delete;
    MossTtsDelayHeadsRuntime & operator=(const MossTtsDelayHeadsRuntime &) = delete;

    // Evaluates every head for one position. The graph is built on first use and reused.
    void evaluate(const std::vector<float> & hidden_state, MossTtsDelayStepLogits & out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::decoders

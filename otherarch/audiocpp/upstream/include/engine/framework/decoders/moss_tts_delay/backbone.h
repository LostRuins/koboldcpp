#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::decoders {

// Qwen3 backbone (language_model.*) runtime: loads the language-model weights and runs
// a prefill forward that returns the final hidden states. Text tokens are embedded in
// the graph; the summed audio-codebook contribution is supplied as a precomputed bias,
// mirroring MossTTSDelayModel.get_input_embeddings.
//
// Adapted from src/models/moss/moss_tts_local/backbone.cpp. The two differ only in the
// weight prefix and in what consumes the hidden state — moss_tts_local hands it to a
// depth transformer, the delay family hands it to 1 + n_vq heads. Kept local per the
// maintainer's local-first request; an obvious candidate for sharing once MOSS-TTSD
// lands and there is a third caller.
class MossTtsDelayBackboneRuntime {
public:
    MossTtsDelayBackboneRuntime(
        MossTtsDelayConfig config,
        std::shared_ptr<const assets::TensorSource> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type,
        ggml_type cache_type = GGML_TYPE_F32);
    ~MossTtsDelayBackboneRuntime();

    MossTtsDelayBackboneRuntime(const MossTtsDelayBackboneRuntime &) = delete;
    MossTtsDelayBackboneRuntime & operator=(const MossTtsDelayBackboneRuntime &) = delete;

    int64_t hidden_size() const noexcept;

    // Incremental (KV-cached) generation. Call begin_generation once to size and allocate the
    // per-layer cache, prefill the prompt in a single batched forward, then step one position
    // at a time. This runs generation in O(T) work per step instead of the O(T^2) cost of
    // re-forwarding the whole sequence every frame.
    //
    // begin_generation sizes the cache for max_positions and builds the reusable
    // single-position step graph.
    void begin_generation(int64_t max_positions) const;
    // prefill forwards the whole prompt at once, writes every position's K/V into the cache,
    // and returns the last position's hidden state ([hidden_size]) to seed the first frame.
    // audio_bias is the summed audio-codebook embedding per position ([token_ids.size() *
    // hidden_size] row-major), mirroring MossTTSLocalModel._build_inputs_embeds.
    std::vector<float> prefill(const std::vector<int32_t> & token_ids, const std::vector<float> & audio_bias) const;
    // step forwards one fused position (its text token embedded in-graph plus audio_bias_row),
    // appends that position's K/V to the cache, and returns its hidden state ([hidden_size]).
    std::vector<float> step(int32_t token_id, const std::vector<float> & audio_bias_row) const;
    void step_into(int32_t token_id, const std::vector<float> & audio_bias_row, std::vector<float> & hidden_state) const;
    int64_t cached_positions() const noexcept;
    int64_t release_cached_step_graph() const;
    void reset_timing() const;
    void log_timing() const;

private:
    void build_step_graph(int64_t cache_steps) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::decoders

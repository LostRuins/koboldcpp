#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

enum class CausalDecoderOutputMode {
    Logits,
    Hidden,
};

struct CausalDecoderRuntimeConfig {
    std::string trace_name = "qwen_causal_decode";
    CausalDecoderConfig decoder;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    CausalDecoderOutputMode output_mode = CausalDecoderOutputMode::Logits;
    bool return_hidden = false;
    std::optional<ggml_type> readback_round_type;
    std::vector<int32_t> logits_readback_token_ids;
    int64_t sliding_window = 0;
    bool evict_cuda_graph_cache_on_release = false;
};

struct CausalDecoderRuntimeWeights {
    core::TensorValue token_embedding;
    DecoderStackWeights stack;
    NormWeights final_norm;
    std::optional<LinearWeights> lm_head;
};

struct CausalDecoderPrefillResult {
    std::vector<float> logits;
    std::vector<float> hidden;
    runtime::TransformerKVState state;
};

struct CausalDecoderPrefillIntoDecodeResult {
    std::vector<float> logits;
    std::vector<float> hidden;
    int64_t current_end = 0;
    int64_t valid_steps = 0;
};

struct CausalDecoderBatchedPrefillResult {
    std::vector<float> logits;
    std::vector<float> hidden;
    runtime::TransformerBatchedKVState state;
};

struct CausalDecoderStepResult {
    std::vector<float> logits;
    std::vector<float> hidden;
};

class CausalDecoderRuntime {
public:
    CausalDecoderRuntime(
        core::ExecutionContext & execution,
        CausalDecoderRuntimeConfig config,
        CausalDecoderRuntimeWeights weights);
    ~CausalDecoderRuntime();

    CausalDecoderRuntime(const CausalDecoderRuntime &) = delete;
    CausalDecoderRuntime & operator=(const CausalDecoderRuntime &) = delete;

    CausalDecoderPrefillResult prefill_tokens(const std::vector<int32_t> & token_ids);
    CausalDecoderPrefillResult prefill_embeddings(const std::vector<float> & embeddings, int64_t steps);
    CausalDecoderPrefillIntoDecodeResult prefill_tokens_into_decode_cache(
        const std::vector<int32_t> & token_ids,
        int64_t required_cache_steps);
    CausalDecoderPrefillIntoDecodeResult prefill_embeddings_into_decode_cache(
        const std::vector<float> & embeddings,
        int64_t steps,
        int64_t required_cache_steps);

    // Prefill bounded blocks directly into the token-decode cache on the backend.
    // No host KV export/import; subsequent decode_token calls continue this state.
    // keep_prefix_steps: the caller guarantees the first rows of `embeddings`
    // equal those of the previous prefill on this runtime; their cached K/V
    // rows are retained and only the remainder is computed. Falls back to a
    // full prefill when the cache no longer holds that prefix.
    CausalDecoderStepResult prefill_embeddings_into_cache(
        const std::vector<float> & embeddings, int64_t steps, int64_t cache_steps, int64_t chunk_steps,
        int64_t keep_prefix_steps = 0);
    // How many of keep_prefix_steps the next prefill_embeddings_into_cache call
    // with these arguments would actually retain (0 when the cache would be
    // cleared or rebuilt), so the caller can skip preparing kept rows.
    int64_t retainable_prefix_steps(
        int64_t steps, int64_t cache_steps, int64_t chunk_steps, int64_t keep_prefix_steps) const;

    CausalDecoderBatchedPrefillResult prefill_tokens_batched(
        const std::vector<int32_t> & token_ids,
        int64_t batch_size,
        int64_t steps);
    CausalDecoderBatchedPrefillResult prefill_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size,
        int64_t steps);

    void start_decode_tokens(const runtime::TransformerKVState & state, int64_t required_cache_steps);
    void start_decode_embeddings(const runtime::TransformerKVState & state, int64_t required_cache_steps);
    CausalDecoderStepResult decode_token(int32_t token);
    void decode_token_into(int32_t token, CausalDecoderStepResult & out);
    CausalDecoderStepResult decode_embedding(const std::vector<float> & embedding);

    void start_decode_tokens_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps);
    void start_decode_embeddings_batched(
        const runtime::TransformerBatchedKVState & state,
        int64_t required_cache_steps);
    CausalDecoderStepResult decode_tokens_batched(const std::vector<int32_t> & tokens);
    CausalDecoderStepResult decode_embeddings_batched(
        const std::vector<float> & embeddings,
        int64_t batch_size);

    // Snapshot of the batched decode KV cache (host vectors), suitable for
    // replication and re-import via start_decode_*_batched with a different
    // batch size — the runtime rebuilds its decode graphs for the new batch.
    runtime::TransformerBatchedKVState export_batched_decode_state() const;

    int64_t decode_cache_steps() const noexcept;
    int64_t decode_current_end() const noexcept;
    int64_t decode_valid_steps() const noexcept;
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules

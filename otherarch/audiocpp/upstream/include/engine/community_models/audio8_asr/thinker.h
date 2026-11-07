#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/greedy_causal_decoder.h"
#include "engine/community_models/audio8_asr/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::audio8_asr {

// The Audio8 8-layer Qwen2-style decoder, expressed through the framework's
// shared greedy Qwen decoder runtime (prefill with audio-embedding injection
// plus static-cache step decode). Owns only the family-specific spec.
class Audio8Qwen2ThinkerRuntime {
public:
    Audio8Qwen2ThinkerRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const Audio8ASRDecoderConfig & config,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~Audio8Qwen2ThinkerRuntime();

    Audio8Qwen2ThinkerRuntime(const Audio8Qwen2ThinkerRuntime &) = delete;
    Audio8Qwen2ThinkerRuntime & operator=(const Audio8Qwen2ThinkerRuntime &) = delete;

    Audio8ASRGeneratedTokens generate(
        const Audio8ASRPrompt & prompt,
        const Audio8ASRAudioEmbeddings & audio_embeddings,
        const Audio8ASRGenerationOptions & options);

private:
    runtime::GreedyCausalDecoderRuntime qwen2_runtime_;
    std::shared_ptr<const Audio8ASRDecoderConfig> config_;
};

}  // namespace engine::community_models::audio8_asr

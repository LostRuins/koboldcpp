#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/audio8_asr/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::audio8_asr {

// Runs the Audio8 adapter tail over Qwen3-ASR encoder output: the residual
// MLP tower, an adaptive average pool down to the prompt's audio token count,
// and the LayerNorm + linear projector into the decoder hidden size.
class Audio8ProjectorRuntime {
public:
    Audio8ProjectorRuntime(
        std::shared_ptr<const assets::TensorSource> weights_source,
        const Audio8TowerConfig & config,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~Audio8ProjectorRuntime();

    Audio8ProjectorRuntime(const Audio8ProjectorRuntime &) = delete;
    Audio8ProjectorRuntime & operator=(const Audio8ProjectorRuntime &) = delete;

    // input: [encoder_tokens, tower.input_size] float values (token-major);
    // returns [audio_tokens, tower.output_size] float values (token-major).
    Audio8ASRAudioEmbeddings project(
        const std::vector<float> & encoder_output,
        int64_t encoder_tokens,
        int64_t audio_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::audio8_asr

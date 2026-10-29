#pragma once

#include "engine/models/pocket_tts/mimi_decoder.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;

class AudioDecoder {
public:
    explicit AudioDecoder(MimiDecoderConfig config = {});

    std::vector<float> decode(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & normalized_latents,
        int64_t steps,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes,
        int64_t full_chunk_frames,
        int64_t stage2_chunk_frames,
        bool use_full_sequence_path) const;

    void reset_streaming_state() const;

    std::vector<float> decode_streaming_step(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & normalized_latent,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes) const;

    void clear_runtime_cache() const noexcept;

private:
    MimiDecoder decoder_;
};

}  // namespace engine::models::pocket_tts

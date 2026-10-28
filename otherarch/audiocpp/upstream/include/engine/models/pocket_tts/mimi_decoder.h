#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;

struct MimiDecoderConfig {
    int64_t latent_size = 32;
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t intermediate_size = 2048;
    int64_t transformer_layers = 2;
    int64_t encoder_upsample_stride = 16;
};

class MimiDecoder {
public:
    explicit MimiDecoder(MimiDecoderConfig config = {});
    ~MimiDecoder();

    const MimiDecoderConfig & config() const noexcept;

    std::vector<float> decode(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & latents,
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
        const std::vector<float> & latent,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes) const;

    void clear_runtime_cache() const noexcept;

private:
    struct RuntimeCache;
    struct StreamingState;
    MimiDecoderConfig config_;
    mutable std::unique_ptr<RuntimeCache> runtime_cache_;
    mutable std::unique_ptr<StreamingState> streaming_state_;
};

}  // namespace engine::models::pocket_tts

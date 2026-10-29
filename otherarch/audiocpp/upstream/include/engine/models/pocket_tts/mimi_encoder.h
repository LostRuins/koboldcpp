#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;

struct MimiPromptEmbedding {
    std::vector<float> values;
    int64_t frames = 0;
};

class MimiEncoder {
public:
    MimiEncoder() = default;

    MimiPromptEmbedding encode_prompt_embedding(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & waveform,
        size_t graph_context_bytes) const;
};

}  // namespace engine::models::pocket_tts

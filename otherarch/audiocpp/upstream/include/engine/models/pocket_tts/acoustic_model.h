#pragma once

#include "engine/models/pocket_tts/flow_lm.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;

struct AcousticGenerationConfig {
    int max_steps = 0;
    int frames_after_eos = -1;
    float temperature = 0.7F;
    float noise_clamp = -1.0F;
    float eos_threshold = -4.0F;
    uint32_t seed = 1234;
    std::vector<float> noise_schedule;
};

struct AcousticModelResult {
    int generated_steps = 0;
    std::vector<float> latents;
    std::vector<float> eos_logits;
};

struct AcousticPreparedRuntime {
    int64_t prompt_steps = 0;
    std::shared_ptr<FlowLMStepRuntime> step_runtime;
};

struct AcousticStreamState {
    AcousticPreparedRuntime runtime;
    AcousticGenerationConfig config;
    std::vector<float> current_input;
    std::mt19937 rng;
    int step = 0;
    int eos_step = -1;
    int generated_steps = 0;
    bool done = false;
};

class AcousticModel {
public:
    explicit AcousticModel(FlowLMConfig config = {});

    const FlowLMConfig & config() const noexcept;

    AcousticPreparedRuntime prepare_runtime(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config,
        int64_t prompt_capacity,
        int64_t initial_cache_capacity,
        int max_steps_capacity,
        size_t flow_weights_view_context_bytes,
        size_t flow_step_graph_context_bytes) const;

    AcousticModelResult generate(
        const AcousticPreparedRuntime & runtime,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config) const;

    AcousticStreamState start_stream(
        const AcousticPreparedRuntime & runtime,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & text_embeddings,
        const FlowLMState & initial_state,
        const AcousticGenerationConfig & config) const;

    std::optional<FlowLMStepResult> next_stream_step(AcousticStreamState & state) const;

    void clear_runtime_cache() const noexcept;
    int64_t prepared_prompt_capacity() const noexcept;
    int prepared_max_steps_capacity() const noexcept;

private:
    struct RuntimeCache {
        ggml_backend_t backend = nullptr;
        int threads = 1;
        const PocketTTSAssets * manifest = nullptr;
        int64_t prompt_capacity = 0;
        int64_t initial_cache_capacity = 0;
        int max_steps_capacity = 0;
        size_t flow_weights_view_context_bytes = 0;
        size_t flow_step_graph_context_bytes = 0;
        std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime;
        std::shared_ptr<FlowLMStepRuntime> runtime;
    };

    FlowLM flow_lm_;
    mutable RuntimeCache runtime_cache_;
};

}  // namespace engine::models::pocket_tts

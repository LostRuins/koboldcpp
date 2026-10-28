#pragma once

#include "engine/framework/runtime/kv_cache.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;
struct VoiceStateAssets;

struct FlowLMConfig {
    int64_t hidden_size = 1024;
    int64_t num_heads = 16;
    int64_t intermediate_size = 4096;
    int64_t latent_size = 32;
    int64_t flow_hidden_size = 512;
    int64_t layers = 6;
    float norm_eps = 1.0e-5F;
    float flow_eps = 1.0e-6F;
    int64_t flow_depth = 6;  // flow-net residual blocks, see PocketTTSModelConfig::flow_depth
};

using FlowLMCacheState = runtime::KVLayerState;
using FlowLMState = runtime::TransformerKVState;

struct FlowLMStepResult {
    std::vector<float> next_latent;
    float eos_logit = 0.0F;
};

struct FlowLMRuntimeTiming {
    double prompt_host_setup_ms = 0.0;
    double prompt_import_ms = 0.0;
    double prompt_graph_ms = 0.0;
    double prompt_finalize_ms = 0.0;
    double prompt_mask_refresh_ms = 0.0;
    double step_input_write_ms = 0.0;
    double step_graph_ms = 0.0;
    double step_output_read_ms = 0.0;
    double step_kv_update_ms = 0.0;
};

struct FlowLMPlanTiming {
    double prompt_plan_create_ms = 0.0;
    double step_plan_create_ms = 0.0;
};

class FlowLMStepRuntime;
class FlowLMWeightsRuntime;

class FlowLM {
public:
    explicit FlowLM(FlowLMConfig config = {});

    const FlowLMConfig & config() const noexcept;

    FlowLMState make_empty_state() const;
    FlowLMState make_state(const VoiceStateAssets & voice_assets) const;

    void apply_prompt(
        FlowLMStepRuntime & runtime,
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state) const;

    FlowLMState prompt_state(
        ggml_backend_t backend,
        int threads,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state,
        size_t weights_view_context_bytes,
        size_t step_graph_context_bytes) const;

    std::shared_ptr<const FlowLMWeightsRuntime> create_weights_runtime(
        ggml_backend_t backend,
        int threads,
        const PocketTTSBackendWeights & weights,
        size_t weights_view_context_bytes) const;

    std::shared_ptr<FlowLMStepRuntime> create_step_runtime(
        ggml_backend_t backend,
        int threads,
        std::shared_ptr<const FlowLMWeightsRuntime> weights,
        int64_t cache_steps,
        int64_t prompt_steps,
        int64_t prompt_prefix_steps,
        size_t step_graph_context_bytes) const;

    FlowLMStepResult run_step_in_place(
        FlowLMStepRuntime & runtime,
        const std::vector<float> & input_latent,
        const std::vector<float> & noise) const;

    FlowLMRuntimeTiming runtime_timing(const FlowLMStepRuntime & runtime) const;
    FlowLMPlanTiming runtime_plan_timing(const FlowLMStepRuntime & runtime) const;

private:
    FlowLMConfig config_;
};

}  // namespace engine::models::pocket_tts

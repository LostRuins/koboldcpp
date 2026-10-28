#pragma once

#include "engine/models/pocket_tts/types.h"
#include "engine/models/pocket_tts/flow_lm.h"
#include "engine/models/pocket_tts/mimi_encoder.h"

#include "ggml-backend.h"

#include <cstddef>

namespace engine::models::pocket_tts {

struct PocketTTSAssets;
struct PocketTTSBackendWeights;

struct VoiceConditioningResult {
    VoiceConditioningPlan plan;
    FlowLMState acoustic_state;
};

class VoiceConditioner {
public:
    explicit VoiceConditioner(FlowLMConfig flow_config = {});

    VoiceConditioningResult prepare(
        const VoiceConditioningPlan & plan,
        const PocketTTSAssets & manifest,
        const PocketTTSBackendWeights & weights,
        ggml_backend_t backend,
        int threads,
        size_t mimi_encoder_graph_context_bytes,
        size_t flow_weights_view_context_bytes,
        size_t flow_step_graph_context_bytes) const;

private:
    FlowLM flow_lm_;
    MimiEncoder mimi_encoder_;
};

}  // namespace engine::models::pocket_tts

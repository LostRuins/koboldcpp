#pragma once

#include "engine/models/vibevoice/diffusion_head.h"
#include "engine/models/vibevoice/scheduler.h"
#include "engine/models/vibevoice/tokenizer_audio.h"

#include <cstdint>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoiceSpeechDiffusionInput {
    std::vector<float> positive_condition;
    std::vector<float> negative_condition;
    std::vector<float> initial_speech;
    int64_t batch_size = 0;
    int64_t hidden_size = 0;
    int64_t latent_size = 0;
    int64_t inference_steps = 0;
    float guidance_scale = 1.0F;
};

std::vector<VibeVoiceTokenizerLatents> sample_vibevoice_speech_latents(
    const VibeVoiceDiffusionHeadWeightsRuntime & prediction_head,
    VibeVoiceDPMSolverScheduler & scheduler,
    const VibeVoiceSpeechDiffusionInput & input);

}  // namespace engine::models::vibevoice

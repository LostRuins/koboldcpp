#include "engine/models/vibevoice/diffusion_sampler.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace {

void require_payload_size(const std::vector<float> & values, int64_t expected, const char * label) {
    if (expected < 0 || static_cast<int64_t>(values.size()) != expected) {
        throw std::runtime_error(std::string("VibeVoice diffusion sampler payload size mismatch for ") + label);
    }
}

std::vector<float> concat_conditions(const VibeVoiceSpeechDiffusionInput & input) {
    std::vector<float> condition;
    condition.reserve(static_cast<size_t>(2 * input.batch_size * input.hidden_size));
    condition.insert(condition.end(), input.positive_condition.begin(), input.positive_condition.end());
    condition.insert(condition.end(), input.negative_condition.begin(), input.negative_condition.end());
    return condition;
}

std::vector<float> duplicate_positive_half(const std::vector<float> & speech, int64_t batch_size, int64_t latent_size) {
    std::vector<float> combined(static_cast<size_t>(2 * batch_size * latent_size), 0.0F);
    const size_t half_values = static_cast<size_t>(batch_size * latent_size);
    std::copy(speech.begin(), speech.begin() + static_cast<std::ptrdiff_t>(half_values), combined.begin());
    std::copy(speech.begin(), speech.begin() + static_cast<std::ptrdiff_t>(half_values), combined.begin() + static_cast<std::ptrdiff_t>(half_values));
    return combined;
}

std::vector<float> apply_cfg(
    const VibeVoiceDiffusionPrediction & prediction,
    int64_t batch_size,
    int64_t latent_size,
    float cfg_scale) {
    if (prediction.frames != 2 * batch_size || prediction.latent_size != latent_size) {
        throw std::runtime_error("VibeVoice diffusion sampler prediction shape mismatch");
    }
    const size_t half_values = static_cast<size_t>(batch_size * latent_size);
    if (prediction.values.size() != 2 * half_values) {
        throw std::runtime_error("VibeVoice diffusion sampler prediction payload size mismatch");
    }
    std::vector<float> eps(2 * half_values, 0.0F);
    for (size_t i = 0; i < half_values; ++i) {
        const float cond = prediction.values[i];
        const float uncond = prediction.values[half_values + i];
        const float guided = uncond + cfg_scale * (cond - uncond);
        eps[i] = guided;
        eps[half_values + i] = guided;
    }
    return eps;
}

std::vector<VibeVoiceTokenizerLatents> first_half_as_latents(
    const std::vector<float> & speech,
    int64_t batch_size,
    int64_t latent_size) {
    std::vector<VibeVoiceTokenizerLatents> out(static_cast<size_t>(batch_size));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto & latents = out[static_cast<size_t>(batch)];
        latents.frames = 1;
        latents.dim = latent_size;
        latents.values.resize(static_cast<size_t>(latent_size));
        const size_t base = static_cast<size_t>(batch * latent_size);
        std::copy(
            speech.begin() + static_cast<std::ptrdiff_t>(base),
            speech.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(latent_size)),
            latents.values.begin());
    }
    return out;
}

}  // namespace

std::vector<VibeVoiceTokenizerLatents> sample_vibevoice_speech_latents(
    const VibeVoiceDiffusionHeadWeightsRuntime & prediction_head,
    VibeVoiceDPMSolverScheduler & scheduler,
    const VibeVoiceSpeechDiffusionInput & input) {
    if (input.batch_size <= 0 || input.hidden_size <= 0 || input.latent_size <= 0) {
        throw std::runtime_error("VibeVoice diffusion sampler requires positive dimensions");
    }
    if (input.inference_steps <= 0) {
        throw std::runtime_error("VibeVoice diffusion sampler requires positive inference steps");
    }
    require_payload_size(
        input.positive_condition,
        input.batch_size * input.hidden_size,
        "positive_condition");
    require_payload_size(
        input.negative_condition,
        input.batch_size * input.hidden_size,
        "negative_condition");
    require_payload_size(
        input.initial_speech,
        2 * input.batch_size * input.latent_size,
        "initial_speech");
    if (static_cast<int64_t>(scheduler.timesteps().size()) != input.inference_steps) {
        throw std::runtime_error("VibeVoice diffusion scheduler timesteps do not match request inference steps");
    }

    const auto condition = concat_conditions(input);
    auto speech = input.initial_speech;
    scheduler.reset_step_state();
    for (const int64_t timestep : scheduler.timesteps()) {
        auto combined = duplicate_positive_half(speech, input.batch_size, input.latent_size);
        auto prediction = prediction_head.predict(
            combined,
            2 * input.batch_size,
            input.latent_size,
            static_cast<float>(timestep),
            condition,
            input.hidden_size);
        auto eps = apply_cfg(prediction, input.batch_size, input.latent_size, input.guidance_scale);
        speech = scheduler.step(eps, timestep, speech).prev_sample;
    }
    return first_half_as_latents(speech, input.batch_size, input.latent_size);
}

}  // namespace engine::models::vibevoice

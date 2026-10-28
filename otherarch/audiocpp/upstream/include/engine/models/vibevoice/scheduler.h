#pragma once

#include "engine/models/vibevoice/assets.h"

#include <cstdint>
#include <vector>

namespace engine::models::vibevoice {

struct VibeVoiceSchedulerStepResult {
    std::vector<float> prev_sample;
};

class VibeVoiceDPMSolverScheduler final {
public:
    explicit VibeVoiceDPMSolverScheduler(const VibeVoiceDiffusionHeadConfig & config);

    void set_timesteps(int64_t inference_steps);
    void reset_step_state();
    const std::vector<int64_t> & timesteps() const noexcept;

    VibeVoiceSchedulerStepResult step(
        const std::vector<float> & model_output,
        int64_t timestep,
        const std::vector<float> & sample);

private:
    void validate_supported_config() const;
    void init_step_index(int64_t timestep);
    int64_t index_for_timestep(int64_t timestep) const;
    std::vector<float> convert_model_output(
        const std::vector<float> & model_output,
        const std::vector<float> & sample) const;
    std::vector<float> first_order_update(
        const std::vector<float> & model_output,
        const std::vector<float> & sample) const;
    std::vector<float> second_order_update(const std::vector<float> & sample) const;

    VibeVoiceDiffusionHeadConfig config_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> sigmas_;
    std::vector<int64_t> timesteps_;
    std::vector<std::vector<float>> model_outputs_;
    int64_t num_inference_steps_ = 0;
    int64_t step_index_ = -1;
    int64_t lower_order_nums_ = 0;
};

}  // namespace engine::models::vibevoice

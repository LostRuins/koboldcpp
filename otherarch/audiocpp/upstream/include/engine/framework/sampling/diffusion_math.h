#pragma once

#include <cstdint>
#include <vector>

namespace engine::sampling {

bool timestep_in_interval(float timestep, float start, float end);

std::vector<float> cfg_guidance(
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float guidance_scale);

std::vector<float> apg_guidance(
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float guidance_scale,
    int64_t frames,
    int64_t channels,
    std::vector<float> & momentum,
    float momentum_coeff = -0.75F,
    double norm_threshold = 2.5);

std::vector<float> adg_guidance(
    const std::vector<float> & latents,
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float sigma,
    float guidance_scale,
    int64_t frames,
    int64_t channels,
    float angle_clip = 3.14F / 6.0F);

void clamp_velocity_norm(
    std::vector<float> & velocity,
    const std::vector<float> & reference,
    float threshold);

std::vector<float> euler_step(
    const std::vector<float> & x,
    const std::vector<float> & velocity,
    float dt);

void euler_step_in_place(
    std::vector<float> & x,
    const std::vector<float> & velocity,
    float dt);

std::vector<float> denoise_from_velocity(
    const std::vector<float> & x,
    const std::vector<float> & velocity,
    float t);

std::vector<float> renoise(
    const std::vector<float> & denoised,
    const std::vector<float> & noise,
    float t);

std::vector<float> heun_combine_velocity(
    const std::vector<float> & velocity_first,
    const std::vector<float> & velocity_second);

std::vector<float> heun_step(
    const std::vector<float> & x_before,
    const std::vector<float> & velocity_first,
    const std::vector<float> & velocity_second,
    float dt);

std::vector<float> build_soft_mask(
    const std::vector<int32_t> & mask,
    int64_t crossfade_frames);

std::vector<float> blend_by_mask(
    const std::vector<float> & generated,
    const std::vector<float> & source,
    const std::vector<float> & soft_mask,
    int64_t channels);

void blend_by_mask_in_place(
    std::vector<float> & generated,
    const std::vector<float> & source,
    const std::vector<float> & soft_mask,
    int64_t channels);

std::vector<float> repaint_step_injection(
    const std::vector<float> & generated,
    const std::vector<float> & clean_source,
    const std::vector<int32_t> & repaint_mask,
    float next_timestep,
    const std::vector<float> & noise,
    int64_t channels);

}  // namespace engine::sampling

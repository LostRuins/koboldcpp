#include "engine/models/vibevoice/scheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::models::vibevoice {
namespace {

constexpr int64_t kSolverOrder = 2;
constexpr float kMaxBeta = 0.999F;
constexpr float kPi = 3.14159265358979323846F;

float alpha_bar_cosine(float t) {
    const float value = std::cos((t + 0.008F) / 1.008F * kPi * 0.5F);
    return value * value;
}

std::vector<float> cosine_betas(int64_t timesteps) {
    if (timesteps <= 0) {
        throw std::runtime_error("VibeVoice scheduler requires positive diffusion timesteps");
    }
    std::vector<float> betas;
    betas.reserve(static_cast<size_t>(timesteps));
    for (int64_t i = 0; i < timesteps; ++i) {
        const float t1 = static_cast<float>(i) / static_cast<float>(timesteps);
        const float t2 = static_cast<float>(i + 1) / static_cast<float>(timesteps);
        betas.push_back(std::min(1.0F - alpha_bar_cosine(t2) / alpha_bar_cosine(t1), kMaxBeta));
    }
    return betas;
}

std::vector<float> build_alphas_cumprod(int64_t timesteps) {
    auto betas = cosine_betas(timesteps);
    std::vector<float> out;
    out.reserve(betas.size());
    float cumulative = 1.0F;
    for (float beta : betas) {
        cumulative *= (1.0F - beta);
        out.push_back(cumulative);
    }
    return out;
}

std::vector<float> base_sigmas(const std::vector<float> & alphas_cumprod) {
    std::vector<float> out;
    out.reserve(alphas_cumprod.size());
    for (float alpha : alphas_cumprod) {
        out.push_back(std::sqrt((1.0F - alpha) / alpha));
    }
    return out;
}

int64_t round_to_int64(double value) {
    return static_cast<int64_t>(std::nearbyint(value));
}

std::vector<int64_t> linspace_timesteps(int64_t last_timestep, int64_t inference_steps) {
    if (last_timestep <= 0 || inference_steps <= 0) {
        throw std::runtime_error("VibeVoice scheduler requires positive timestep counts");
    }
    std::vector<int64_t> forward;
    forward.reserve(static_cast<size_t>(inference_steps + 1));
    const double start = 0.0;
    const double stop = static_cast<double>(last_timestep - 1);
    const double denom = static_cast<double>(inference_steps);
    for (int64_t i = 0; i <= inference_steps; ++i) {
        const double value = start + (stop - start) * static_cast<double>(i) / denom;
        forward.push_back(round_to_int64(value));
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(inference_steps));
    for (int64_t i = inference_steps; i >= 1; --i) {
        out.push_back(forward[static_cast<size_t>(i)]);
    }
    return out;
}

std::vector<float> interpolate_sigmas(const std::vector<float> & sigmas, const std::vector<int64_t> & timesteps) {
    std::vector<float> out;
    out.reserve(timesteps.size() + 1);
    for (int64_t timestep : timesteps) {
        if (timestep < 0 || timestep >= static_cast<int64_t>(sigmas.size())) {
            throw std::runtime_error("VibeVoice scheduler timestep is outside sigma range");
        }
        out.push_back(sigmas[static_cast<size_t>(timestep)]);
    }
    out.push_back(0.0F);
    return out;
}

void sigma_to_alpha_sigma(float sigma, float & alpha_t, float & sigma_t) {
    alpha_t = 1.0F / std::sqrt(sigma * sigma + 1.0F);
    sigma_t = sigma * alpha_t;
}

void validate_same_size(const std::vector<float> & lhs, const std::vector<float> & rhs, const char * label) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error(std::string("VibeVoice scheduler size mismatch for ") + label);
    }
}

}  // namespace

VibeVoiceDPMSolverScheduler::VibeVoiceDPMSolverScheduler(const VibeVoiceDiffusionHeadConfig & config)
    : config_(config),
      alphas_cumprod_(build_alphas_cumprod(config.ddpm_num_steps)),
      model_outputs_(static_cast<size_t>(kSolverOrder)) {
    validate_supported_config();
}

void VibeVoiceDPMSolverScheduler::validate_supported_config() const {
    if (config_.diffusion_type != "ddpm") {
        throw std::runtime_error("VibeVoice scheduler only supports ddpm diffusion_type");
    }
    if (config_.ddpm_beta_schedule != "cosine") {
        throw std::runtime_error("VibeVoice scheduler only supports cosine beta schedule");
    }
    if (config_.prediction_type != "v_prediction") {
        throw std::runtime_error("VibeVoice scheduler only supports v_prediction");
    }
    if (config_.ddpm_num_steps <= 0 || config_.ddpm_num_inference_steps <= 0) {
        throw std::runtime_error("VibeVoice scheduler requires positive DDPM step counts");
    }
}

void VibeVoiceDPMSolverScheduler::set_timesteps(int64_t inference_steps) {
    if (inference_steps <= 0) {
        throw std::runtime_error("VibeVoice scheduler inference steps must be positive");
    }
    const int64_t last_timestep = config_.ddpm_num_steps;
    timesteps_ = linspace_timesteps(last_timestep, inference_steps);
    sigmas_ = interpolate_sigmas(base_sigmas(alphas_cumprod_), timesteps_);
    num_inference_steps_ = static_cast<int64_t>(timesteps_.size());
    reset_step_state();
}

void VibeVoiceDPMSolverScheduler::reset_step_state() {
    model_outputs_.assign(static_cast<size_t>(kSolverOrder), {});
    lower_order_nums_ = 0;
    step_index_ = -1;
}

const std::vector<int64_t> & VibeVoiceDPMSolverScheduler::timesteps() const noexcept {
    return timesteps_;
}

int64_t VibeVoiceDPMSolverScheduler::index_for_timestep(int64_t timestep) const {
    for (size_t i = 0; i < timesteps_.size(); ++i) {
        if (timesteps_[i] == timestep) {
            return static_cast<int64_t>(i);
        }
    }
    return static_cast<int64_t>(timesteps_.size()) - 1;
}

void VibeVoiceDPMSolverScheduler::init_step_index(int64_t timestep) {
    if (timesteps_.empty()) {
        throw std::runtime_error("VibeVoice scheduler timesteps were not initialized");
    }
    step_index_ = index_for_timestep(timestep);
}

std::vector<float> VibeVoiceDPMSolverScheduler::convert_model_output(
    const std::vector<float> & model_output,
    const std::vector<float> & sample) const {
    validate_same_size(model_output, sample, "convert_model_output");
    if (step_index_ < 0 || step_index_ >= static_cast<int64_t>(sigmas_.size())) {
        throw std::runtime_error("VibeVoice scheduler step index is invalid");
    }
    float alpha_t = 0.0F;
    float sigma_t = 0.0F;
    sigma_to_alpha_sigma(sigmas_[static_cast<size_t>(step_index_)], alpha_t, sigma_t);
    std::vector<float> out(model_output.size());
    for (size_t i = 0; i < model_output.size(); ++i) {
        out[i] = alpha_t * sample[i] - sigma_t * model_output[i];
    }
    return out;
}

std::vector<float> VibeVoiceDPMSolverScheduler::first_order_update(
    const std::vector<float> & model_output,
    const std::vector<float> & sample) const {
    validate_same_size(model_output, sample, "first_order_update");
    const size_t idx = static_cast<size_t>(step_index_);
    float alpha_t = 0.0F;
    float sigma_t = 0.0F;
    float alpha_s = 0.0F;
    float sigma_s = 0.0F;
    sigma_to_alpha_sigma(sigmas_.at(idx + 1), alpha_t, sigma_t);
    sigma_to_alpha_sigma(sigmas_.at(idx), alpha_s, sigma_s);
    const float lambda_t = std::log(alpha_t) - std::log(sigma_t);
    const float lambda_s = std::log(alpha_s) - std::log(sigma_s);
    const float h = lambda_t - lambda_s;
    const float exp_neg_h = std::exp(-h);
    const float sample_coeff = sigma_t / sigma_s;
    const float model_coeff = -alpha_t * (exp_neg_h - 1.0F);
    std::vector<float> out(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        out[i] = sample_coeff * sample[i] + model_coeff * model_output[i];
    }
    return out;
}

std::vector<float> VibeVoiceDPMSolverScheduler::second_order_update(const std::vector<float> & sample) const {
    if (model_outputs_.size() != static_cast<size_t>(kSolverOrder) || model_outputs_[0].empty() || model_outputs_[1].empty()) {
        throw std::runtime_error("VibeVoice scheduler second-order update requires two model outputs");
    }
    validate_same_size(model_outputs_[0], model_outputs_[1], "second_order_model_outputs");
    validate_same_size(model_outputs_[1], sample, "second_order_sample");
    const size_t idx = static_cast<size_t>(step_index_);
    float alpha_t = 0.0F;
    float sigma_t = 0.0F;
    float alpha_s0 = 0.0F;
    float sigma_s0 = 0.0F;
    float alpha_s1 = 0.0F;
    float sigma_s1 = 0.0F;
    sigma_to_alpha_sigma(sigmas_.at(idx + 1), alpha_t, sigma_t);
    sigma_to_alpha_sigma(sigmas_.at(idx), alpha_s0, sigma_s0);
    sigma_to_alpha_sigma(sigmas_.at(idx - 1), alpha_s1, sigma_s1);
    const float lambda_t = std::log(alpha_t) - std::log(sigma_t);
    const float lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0);
    const float lambda_s1 = std::log(alpha_s1) - std::log(sigma_s1);
    const float h = lambda_t - lambda_s0;
    const float h0 = lambda_s0 - lambda_s1;
    const float r0 = h0 / h;
    const float exp_neg_h = std::exp(-h);
    const float sample_coeff = sigma_t / sigma_s0;
    const float d0_coeff = -alpha_t * (exp_neg_h - 1.0F);
    std::vector<float> out(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        const float d0 = model_outputs_[1][i];
        const float d1 = (1.0F / r0) * (model_outputs_[1][i] - model_outputs_[0][i]);
        out[i] = sample_coeff * sample[i] + d0_coeff * d0 + 0.5F * d0_coeff * d1;
    }
    return out;
}

VibeVoiceSchedulerStepResult VibeVoiceDPMSolverScheduler::step(
    const std::vector<float> & model_output,
    int64_t timestep,
    const std::vector<float> & sample) {
    if (num_inference_steps_ <= 0) {
        throw std::runtime_error("VibeVoice scheduler set_timesteps must be called before step");
    }
    if (step_index_ < 0) {
        init_step_index(timestep);
    }
    const bool lower_order_final = (step_index_ == static_cast<int64_t>(timesteps_.size()) - 1);
    const bool lower_order_second =
        (step_index_ == static_cast<int64_t>(timesteps_.size()) - 2) &&
        static_cast<int64_t>(timesteps_.size()) < 15;
    auto converted = convert_model_output(model_output, sample);
    for (size_t i = 0; i + 1 < model_outputs_.size(); ++i) {
        model_outputs_[i] = model_outputs_[i + 1];
    }
    model_outputs_.back() = converted;

    std::vector<float> prev_sample;
    if (lower_order_nums_ < 1 || lower_order_final) {
        prev_sample = first_order_update(model_outputs_.back(), sample);
    } else if (kSolverOrder == 2 || lower_order_nums_ < 2 || lower_order_second) {
        prev_sample = second_order_update(sample);
    } else {
        throw std::runtime_error("VibeVoice scheduler unexpected third-order path");
    }

    if (lower_order_nums_ < kSolverOrder) {
        ++lower_order_nums_;
    }
    ++step_index_;
    return {std::move(prev_sample)};
}

}  // namespace engine::models::vibevoice

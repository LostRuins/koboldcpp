#ifndef __UCACHE_HPP__
#define __UCACHE_HPP__

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "condition_cache_utils.hpp"
#include "denoiser.hpp"
#include "ggml_extend.hpp"
#include "tensor.hpp"

struct UCacheConfig {
    bool enabled                = false;
    float reuse_threshold       = 1.0f;
    float start_percent         = 0.15f;
    float end_percent           = 0.95f;
    float error_decay_rate      = 1.0f;
    bool use_relative_threshold = true;
    bool adaptive_threshold     = true;
    float early_step_multiplier = 0.5f;
    float late_step_multiplier  = 1.5f;
    float relative_norm_gain    = 1.6f;
    bool reset_error_on_compute = true;
};

struct UCacheCacheEntry {
    std::vector<float> diff;
};

struct UCacheState {
    UCacheConfig config;
    Denoiser* denoiser           = nullptr;
    float start_sigma            = std::numeric_limits<float>::max();
    float end_sigma              = 0.0f;
    bool initialized             = false;
    bool initial_step            = true;
    bool skip_current_step       = false;
    bool step_active             = false;
    const void* anchor_condition = nullptr;
    std::unordered_map<const void*, UCacheCacheEntry> cache_diffs;
    std::vector<float> prev_input;
    std::vector<float> prev_output;
    float output_prev_norm                = 0.0f;
    bool has_prev_input                   = false;
    bool has_prev_output                  = false;
    bool has_output_prev_norm             = false;
    bool has_relative_transformation_rate = false;
    float relative_transformation_rate    = 0.0f;
    float last_input_change               = 0.0f;
    bool has_last_input_change            = false;
    float output_change_ema               = 0.0f;
    bool has_output_change_ema            = false;
    int total_steps_skipped               = 0;
    int current_step_index                = -1;
    int steps_computed_since_active       = 0;
    int expected_total_steps              = 0;
    int consecutive_skipped_steps         = 0;
    float accumulated_error               = 0.0f;

    struct BlockMetrics {
        float sum_transformation_rate = 0.0f;
        float sum_output_norm         = 0.0f;
        int sample_count              = 0;
        float min_change_rate         = std::numeric_limits<float>::max();
        float max_change_rate         = 0.0f;

        void reset() {
            sum_transformation_rate = 0.0f;
            sum_output_norm         = 0.0f;
            sample_count            = 0;
            min_change_rate         = std::numeric_limits<float>::max();
            max_change_rate         = 0.0f;
        }

        void record(float change_rate, float output_norm) {
            if (std::isfinite(change_rate) && change_rate > 0.0f) {
                sum_transformation_rate += change_rate;
                sum_output_norm += output_norm;
                sample_count++;
                if (change_rate < min_change_rate)
                    min_change_rate = change_rate;
                if (change_rate > max_change_rate)
                    max_change_rate = change_rate;
            }
        }

        float avg_transformation_rate() const {
            return (sample_count > 0) ? (sum_transformation_rate / sample_count) : 0.0f;
        }

        float avg_output_norm() const {
            return (sample_count > 0) ? (sum_output_norm / sample_count) : 0.0f;
        }
    };
    BlockMetrics block_metrics;
    int total_active_steps = 0;

    void reset_runtime() {
        initial_step      = true;
        skip_current_step = false;
        step_active       = false;
        anchor_condition  = nullptr;
        cache_diffs.clear();
        prev_input.clear();
        prev_output.clear();
        output_prev_norm                 = 0.0f;
        has_prev_input                   = false;
        has_prev_output                  = false;
        has_output_prev_norm             = false;
        has_relative_transformation_rate = false;
        relative_transformation_rate     = 0.0f;
        last_input_change                = 0.0f;
        has_last_input_change            = false;
        output_change_ema                = 0.0f;
        has_output_change_ema            = false;
        total_steps_skipped              = 0;
        current_step_index               = -1;
        steps_computed_since_active      = 0;
        expected_total_steps             = 0;
        consecutive_skipped_steps        = 0;
        accumulated_error                = 0.0f;
        block_metrics.reset();
        total_active_steps = 0;
    }

    void init(const UCacheConfig& cfg, Denoiser* d) {
        config      = cfg;
        denoiser    = d;
        initialized = cfg.enabled && d != nullptr;
        reset_runtime();
        if (initialized) {
            start_sigma = percent_to_sigma(config.start_percent);
            end_sigma   = percent_to_sigma(config.end_percent);
        }
    }

    void set_sigmas(const std::vector<float>& sigmas) {
        if (!initialized || sigmas.size() < 2) {
            return;
        }
        size_t n_steps       = sigmas.size() - 1;
        expected_total_steps = static_cast<int>(n_steps);

        size_t start_step = static_cast<size_t>(config.start_percent * n_steps);
        size_t end_step   = static_cast<size_t>(config.end_percent * n_steps);

        if (start_step >= n_steps)
            start_step = n_steps - 1;
        if (end_step >= n_steps)
            end_step = n_steps - 1;

        start_sigma = sigmas[start_step];
        end_sigma   = sigmas[end_step];

        if (start_sigma < end_sigma) {
            std::swap(start_sigma, end_sigma);
        }
    }

    bool enabled() const {
        return initialized && config.enabled;
    }

    float percent_to_sigma(float percent) const {
        if (!denoiser) {
            return 0.0f;
        }
        if (percent <= 0.0f) {
            return std::numeric_limits<float>::max();
        }
        if (percent >= 1.0f) {
            return 0.0f;
        }
        float t = (1.0f - percent) * (TIMESTEPS - 1);
        return denoiser->t_to_sigma(t);
    }

    void begin_step(int step_index, float sigma) {
        if (!enabled()) {
            return;
        }
        if (step_index == current_step_index) {
            return;
        }
        current_step_index    = step_index;
        skip_current_step     = false;
        has_last_input_change = false;
        step_active           = false;

        if (sigma > start_sigma) {
            return;
        }
        if (!(sigma > end_sigma)) {
            return;
        }
        step_active = true;
        total_active_steps++;
    }

    bool step_is_active() const {
        return enabled() && step_active;
    }

    bool is_step_skipped() const {
        return enabled() && step_active && skip_current_step;
    }

    float get_adaptive_threshold(int estimated_total_steps = 0) const {
        float base_threshold = config.reuse_threshold;

        if (!config.adaptive_threshold) {
            return base_threshold;
        }

        int effective_total = estimated_total_steps;
        if (effective_total <= 0) {
            effective_total = expected_total_steps;
        }
        if (effective_total <= 0) {
            effective_total = std::max(20, steps_computed_since_active * 2);
        }

        float progress = (effective_total > 0) ? (static_cast<float>(steps_computed_since_active) / effective_total) : 0.0f;
        progress       = std::max(0.0f, std::min(1.0f, progress));

        float multiplier = 1.0f;
        if (progress < 0.2f) {
            multiplier = config.early_step_multiplier;
        } else if (progress > 0.8f) {
            multiplier = config.late_step_multiplier;
        }

        return base_threshold * multiplier;
    }

    bool has_cache(const void* cond) const {
        auto it = cache_diffs.find(cond);
        return it != cache_diffs.end() && !it->second.diff.empty();
    }

    void update_cache(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        UCacheCacheEntry& entry = cache_diffs[cond];
        sd::store_condition_cache_diff(&entry.diff, input, output);
    }

    void apply_cache(const void* cond, const sd::Tensor<float>& input, sd::Tensor<float>* output) {
        auto it = cache_diffs.find(cond);
        if (it == cache_diffs.end() || it->second.diff.empty()) {
            return;
        }
        sd::apply_condition_cache_diff(it->second.diff, input, output);
    }

    bool before_condition(const void* cond,
                          const sd::Tensor<float>& input,
                          sd::Tensor<float>* output,
                          float sigma,
                          int step_index) {
        if (!enabled() || step_index < 0 || output == nullptr) {
            return false;
        }
        if (step_index != current_step_index) {
            begin_step(step_index, sigma);
        }
        if (!step_active) {
            return false;
        }

        if (initial_step) {
            anchor_condition = cond;
            initial_step     = false;
        }

        bool is_anchor = (cond == anchor_condition);

        if (skip_current_step) {
            if (has_cache(cond)) {
                apply_cache(cond, input, output);
                return true;
            }
            return false;
        }

        if (!is_anchor) {
            return false;
        }

        if (!has_prev_input || !has_prev_output || !has_cache(cond)) {
            return false;
        }

        size_t ne = static_cast<size_t>(input.numel());
        if (prev_input.size() != ne) {
            return false;
        }

        const float* input_data = input.data();
        last_input_change       = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            last_input_change += std::fabs(input_data[i] - prev_input[i]);
        }
        if (ne > 0) {
            last_input_change /= static_cast<float>(ne);
        }
        has_last_input_change = true;

        if (has_output_prev_norm && has_relative_transformation_rate &&
            last_input_change > 0.0f && output_prev_norm > 0.0f) {
            float approx_output_change = relative_transformation_rate * last_input_change;
            float approx_output_change_rate;
            if (config.use_relative_threshold) {
                float base_scale          = std::max(output_prev_norm, 1e-6f);
                float dyn_scale           = has_output_change_ema
                                                ? std::max(output_change_ema * std::max(1.0f, config.relative_norm_gain), 1e-6f)
                                                : base_scale;
                float scale               = std::sqrt(base_scale * dyn_scale);
                approx_output_change_rate = approx_output_change / scale;
            } else {
                approx_output_change_rate = approx_output_change;
            }
            // Increase estimated error with skip horizon to avoid long extrapolation streaks
            approx_output_change_rate *= (1.0f + 0.50f * consecutive_skipped_steps);
            accumulated_error = accumulated_error * config.error_decay_rate + approx_output_change_rate;

            float effective_threshold = get_adaptive_threshold();
            if (!config.use_relative_threshold && output_prev_norm > 0.0f) {
                effective_threshold = effective_threshold * output_prev_norm;
            }

            if (accumulated_error < effective_threshold) {
                skip_current_step = true;
                total_steps_skipped++;
                consecutive_skipped_steps++;
                apply_cache(cond, input, output);
                return true;
            } else if (config.reset_error_on_compute) {
                accumulated_error = 0.0f;
            }
        }

        return false;
    }

    void after_condition(const void* cond, const sd::Tensor<float>& input, const sd::Tensor<float>& output) {
        if (!step_is_active()) {
            return;
        }

        update_cache(cond, input, output);

        if (cond != anchor_condition) {
            return;
        }
        steps_computed_since_active++;
        consecutive_skipped_steps = 0;

        size_t ne            = static_cast<size_t>(input.numel());
        const float* in_data = input.data();
        prev_input.resize(ne);
        for (size_t i = 0; i < ne; ++i) {
            prev_input[i] = in_data[i];
        }
        has_prev_input = true;

        const float* out_data = output.data();
        float output_change   = 0.0f;
        if (has_prev_output && prev_output.size() == ne) {
            for (size_t i = 0; i < ne; ++i) {
                output_change += std::fabs(out_data[i] - prev_output[i]);
            }
            if (ne > 0) {
                output_change /= static_cast<float>(ne);
            }
        }
        if (std::isfinite(output_change) && output_change > 0.0f) {
            if (!has_output_change_ema) {
                output_change_ema     = output_change;
                has_output_change_ema = true;
            } else {
                output_change_ema = 0.8f * output_change_ema + 0.2f * output_change;
            }
        }

        prev_output.resize(ne);
        for (size_t i = 0; i < ne; ++i) {
            prev_output[i] = out_data[i];
        }
        has_prev_output = true;

        float mean_abs = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            mean_abs += std::fabs(out_data[i]);
        }
        output_prev_norm     = (ne > 0) ? (mean_abs / static_cast<float>(ne)) : 0.0f;
        has_output_prev_norm = output_prev_norm > 0.0f;

        if (has_last_input_change && last_input_change > 0.0f && output_change > 0.0f) {
            float rate = output_change / last_input_change;
            if (std::isfinite(rate)) {
                relative_transformation_rate     = rate;
                has_relative_transformation_rate = true;
                block_metrics.record(rate, output_prev_norm);
            }
        }

        has_last_input_change = false;
    }

    void log_block_metrics() const {
        if (block_metrics.sample_count > 0) {
            LOG_INFO("UCacheBlockMetrics: samples=%d, avg_rate=%.4f, min=%.4f, max=%.4f, avg_norm=%.4f",
                     block_metrics.sample_count,
                     block_metrics.avg_transformation_rate(),
                     block_metrics.min_change_rate,
                     block_metrics.max_change_rate,
                     block_metrics.avg_output_norm());
        }
    }
};

#endif  // __UCACHE_HPP__

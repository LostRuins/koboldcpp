#ifndef __UCACHE_HPP__
#define __UCACHE_HPP__

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "denoiser.hpp"
#include "ggml_extend.hpp"

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
    bool reset_error_on_compute = true;
};

struct UCacheCacheEntry {
    std::vector<float> diff;
};

struct UCacheState {
    UCacheConfig config;
    Denoiser* denoiser                  = nullptr;
    float start_sigma                   = std::numeric_limits<float>::max();
    float end_sigma                     = 0.0f;
    bool initialized                    = false;
    bool initial_step                   = true;
    bool skip_current_step              = false;
    bool step_active                    = false;
    const SDCondition* anchor_condition = nullptr;
    std::unordered_map<const SDCondition*, UCacheCacheEntry> cache_diffs;
    std::vector<float> prev_input;
    std::vector<float> prev_output;
    float output_prev_norm                = 0.0f;
    bool has_prev_input                   = false;
    bool has_prev_output                  = false;
    bool has_output_prev_norm             = false;
    bool has_relative_transformation_rate = false;
    float relative_transformation_rate    = 0.0f;
    float cumulative_change_rate          = 0.0f;
    float last_input_change               = 0.0f;
    bool has_last_input_change            = false;
    int total_steps_skipped               = 0;
    int current_step_index                = -1;
    int steps_computed_since_active       = 0;
    float accumulated_error               = 0.0f;
    float reference_output_norm           = 0.0f;

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
        cumulative_change_rate           = 0.0f;
        last_input_change                = 0.0f;
        has_last_input_change            = false;
        total_steps_skipped              = 0;
        current_step_index               = -1;
        steps_computed_since_active      = 0;
        accumulated_error                = 0.0f;
        reference_output_norm            = 0.0f;
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
        size_t n_steps = sigmas.size() - 1;

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
            effective_total = std::max(20, steps_computed_since_active * 2);
        }

        float progress = (effective_total > 0) ? (static_cast<float>(steps_computed_since_active) / effective_total) : 0.0f;

        float multiplier = 1.0f;
        if (progress < 0.2f) {
            multiplier = config.early_step_multiplier;
        } else if (progress > 0.8f) {
            multiplier = config.late_step_multiplier;
        }

        return base_threshold * multiplier;
    }

    bool has_cache(const SDCondition* cond) const {
        auto it = cache_diffs.find(cond);
        return it != cache_diffs.end() && !it->second.diff.empty();
    }

    void update_cache(const SDCondition* cond, ggml_tensor* input, ggml_tensor* output) {
        UCacheCacheEntry& entry = cache_diffs[cond];
        size_t ne               = static_cast<size_t>(ggml_nelements(output));
        entry.diff.resize(ne);
        float* out_data = (float*)output->data;
        float* in_data  = (float*)input->data;

        for (size_t i = 0; i < ne; ++i) {
            entry.diff[i] = out_data[i] - in_data[i];
        }
    }

    void apply_cache(const SDCondition* cond, ggml_tensor* input, ggml_tensor* output) {
        auto it = cache_diffs.find(cond);
        if (it == cache_diffs.end() || it->second.diff.empty()) {
            return;
        }

        copy_ggml_tensor(output, input);
        float* out_data                = (float*)output->data;
        const std::vector<float>& diff = it->second.diff;
        for (size_t i = 0; i < diff.size(); ++i) {
            out_data[i] += diff[i];
        }
    }

    bool before_condition(const SDCondition* cond,
                          ggml_tensor* input,
                          ggml_tensor* output,
                          float sigma,
                          int step_index) {
        if (!enabled() || step_index < 0) {
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

        size_t ne = static_cast<size_t>(ggml_nelements(input));
        if (prev_input.size() != ne) {
            return false;
        }

        float* input_data = (float*)input->data;
        last_input_change = 0.0f;
        for (size_t i = 0; i < ne; ++i) {
            last_input_change += std::fabs(input_data[i] - prev_input[i]);
        }
        if (ne > 0) {
            last_input_change /= static_cast<float>(ne);
        }
        has_last_input_change = true;

        if (has_output_prev_norm && has_relative_transformation_rate &&
            last_input_change > 0.0f && output_prev_norm > 0.0f) {
            float approx_output_change_rate = (relative_transformation_rate * last_input_change) / output_prev_norm;
            accumulated_error               = accumulated_error * config.error_decay_rate + approx_output_change_rate;

            float effective_threshold = get_adaptive_threshold();
            if (config.use_relative_threshold && reference_output_norm > 0.0f) {
                effective_threshold = effective_threshold * reference_output_norm;
            }

            if (accumulated_error < effective_threshold) {
                skip_current_step = true;
                total_steps_skipped++;
                apply_cache(cond, input, output);
                return true;
            } else if (config.reset_error_on_compute) {
                accumulated_error = 0.0f;
            }
        }

        return false;
    }

    void after_condition(const SDCondition* cond, ggml_tensor* input, ggml_tensor* output) {
        if (!step_is_active()) {
            return;
        }

        update_cache(cond, input, output);

        if (cond != anchor_condition) {
            return;
        }

        size_t ne      = static_cast<size_t>(ggml_nelements(input));
        float* in_data = (float*)input->data;
        prev_input.resize(ne);
        for (size_t i = 0; i < ne; ++i) {
            prev_input[i] = in_data[i];
        }
        has_prev_input = true;

        float* out_data     = (float*)output->data;
        float output_change = 0.0f;
        if (has_prev_output && prev_output.size() == ne) {
            for (size_t i = 0; i < ne; ++i) {
                output_change += std::fabs(out_data[i] - prev_output[i]);
            }
            if (ne > 0) {
                output_change /= static_cast<float>(ne);
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

        if (reference_output_norm == 0.0f) {
            reference_output_norm = output_prev_norm;
        }

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

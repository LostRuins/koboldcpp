#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "denoiser.hpp"
#include "ggml_extend.hpp"

struct EasyCacheConfig {
    bool enabled          = false;
    float reuse_threshold = 0.2f;
    float start_percent   = 0.15f;
    float end_percent     = 0.95f;
};

struct EasyCacheCacheEntry {
    std::vector<float> diff;
};

struct EasyCacheState {
    EasyCacheConfig config;
    Denoiser* denoiser                  = nullptr;
    float start_sigma                   = std::numeric_limits<float>::max();
    float end_sigma                     = 0.0f;
    bool initialized                    = false;
    bool initial_step                   = true;
    bool skip_current_step              = false;
    bool step_active                    = false;
    const SDCondition* anchor_condition = nullptr;
    std::unordered_map<const SDCondition*, EasyCacheCacheEntry> cache_diffs;
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
    }

    void init(const EasyCacheConfig& cfg, Denoiser* d) {
        config      = cfg;
        denoiser    = d;
        initialized = cfg.enabled && d != nullptr;
        reset_runtime();
        if (initialized) {
            start_sigma = percent_to_sigma(config.start_percent);
            end_sigma   = percent_to_sigma(config.end_percent);
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
    }

    bool step_is_active() const {
        return enabled() && step_active;
    }

    bool is_step_skipped() const {
        return enabled() && step_active && skip_current_step;
    }

    bool has_cache(const SDCondition* cond) const {
        auto it = cache_diffs.find(cond);
        return it != cache_diffs.end() && !it->second.diff.empty();
    }

    void update_cache(const SDCondition* cond, ggml_tensor* input, ggml_tensor* output) {
        EasyCacheCacheEntry& entry = cache_diffs[cond];
        size_t ne                  = static_cast<size_t>(ggml_nelements(output));
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

        if (has_output_prev_norm && has_relative_transformation_rate && last_input_change > 0.0f && output_prev_norm > 0.0f) {
            float approx_output_change_rate = (relative_transformation_rate * last_input_change) / output_prev_norm;
            cumulative_change_rate += approx_output_change_rate;
            if (cumulative_change_rate < config.reuse_threshold) {
                skip_current_step = true;
                total_steps_skipped++;
                apply_cache(cond, input, output);
                return true;
            } else {
                cumulative_change_rate = 0.0f;
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

        if (has_last_input_change && last_input_change > 0.0f && output_change > 0.0f) {
            float rate = output_change / last_input_change;
            if (std::isfinite(rate)) {
                relative_transformation_rate     = rate;
                has_relative_transformation_rate = true;
            }
        }
        cumulative_change_rate = 0.0f;
        has_last_input_change  = false;
    }
};
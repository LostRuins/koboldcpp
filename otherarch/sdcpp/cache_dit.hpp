#ifndef __CACHE_DIT_HPP__
#define __CACHE_DIT_HPP__

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml_extend.hpp"

struct DBCacheConfig {
    bool enabled                        = false;
    int Fn_compute_blocks               = 8;
    int Bn_compute_blocks               = 0;
    float residual_diff_threshold       = 0.08f;
    int max_warmup_steps                = 8;
    int max_cached_steps                = -1;
    int max_continuous_cached_steps     = -1;
    float max_accumulated_residual_diff = -1.0f;
    std::vector<int> steps_computation_mask;
    bool scm_policy_dynamic = true;
};

struct TaylorSeerConfig {
    bool enabled            = false;
    int n_derivatives       = 1;
    int max_warmup_steps    = 2;
    int skip_interval_steps = 1;
};

struct CacheDitConfig {
    DBCacheConfig dbcache;
    TaylorSeerConfig taylorseer;
    int double_Fn_blocks = -1;
    int double_Bn_blocks = -1;
    int single_Fn_blocks = -1;
    int single_Bn_blocks = -1;
};

struct TaylorSeerState {
    int n_derivatives      = 1;
    int current_step       = -1;
    int last_computed_step = -1;
    std::vector<std::vector<float>> dY_prev;
    std::vector<std::vector<float>> dY_current;

    void init(int n_deriv, size_t hidden_size) {
        n_derivatives = n_deriv;
        int order     = n_derivatives + 1;
        dY_prev.resize(order);
        dY_current.resize(order);
        for (int i = 0; i < order; i++) {
            dY_prev[i].clear();
            dY_current[i].clear();
        }
        current_step       = -1;
        last_computed_step = -1;
    }

    void reset() {
        for (auto& v : dY_prev)
            v.clear();
        for (auto& v : dY_current)
            v.clear();
        current_step       = -1;
        last_computed_step = -1;
    }

    bool can_approximate() const {
        return last_computed_step >= n_derivatives && !dY_prev.empty() && !dY_prev[0].empty();
    }

    void update_derivatives(const float* Y, size_t size, int step) {
        int order = n_derivatives + 1;
        dY_prev   = dY_current;
        dY_current[0].resize(size);
        for (size_t i = 0; i < size; i++) {
            dY_current[0][i] = Y[i];
        }

        int window = step - last_computed_step;
        if (window <= 0)
            window = 1;

        for (int d = 0; d < n_derivatives; d++) {
            if (!dY_prev[d].empty() && dY_prev[d].size() == size) {
                dY_current[d + 1].resize(size);
                for (size_t i = 0; i < size; i++) {
                    dY_current[d + 1][i] = (dY_current[d][i] - dY_prev[d][i]) / static_cast<float>(window);
                }
            } else {
                dY_current[d + 1].clear();
            }
        }

        current_step       = step;
        last_computed_step = step;
    }

    void approximate(float* output, size_t size, int target_step) const {
        if (!can_approximate() || dY_prev[0].size() != size) {
            return;
        }

        int elapsed = target_step - last_computed_step;
        if (elapsed <= 0)
            elapsed = 1;

        std::fill(output, output + size, 0.0f);
        float factorial = 1.0f;
        int order       = static_cast<int>(dY_prev.size());

        for (int o = 0; o < order; o++) {
            if (dY_prev[o].empty() || dY_prev[o].size() != size)
                continue;
            if (o > 0)
                factorial *= static_cast<float>(o);
            float coeff = ::powf(static_cast<float>(elapsed), static_cast<float>(o)) / factorial;
            for (size_t i = 0; i < size; i++) {
                output[i] += coeff * dY_prev[o][i];
            }
        }
    }
};

struct BlockCacheEntry {
    std::vector<float> residual_img;
    std::vector<float> residual_txt;
    std::vector<float> residual;
    std::vector<float> prev_img;
    std::vector<float> prev_txt;
    std::vector<float> prev_output;
    bool has_prev = false;
};

struct CacheDitState {
    CacheDitConfig config;
    bool initialized = false;

    int total_double_blocks = 0;
    int total_single_blocks = 0;
    size_t hidden_size      = 0;

    int current_step     = -1;
    int total_steps      = 0;
    int warmup_remaining = 0;
    std::vector<int> cached_steps;
    int continuous_cached_steps     = 0;
    float accumulated_residual_diff = 0.0f;

    std::vector<BlockCacheEntry> double_block_cache;
    std::vector<BlockCacheEntry> single_block_cache;

    std::vector<float> Fn_residual_img;
    std::vector<float> Fn_residual_txt;
    std::vector<float> prev_Fn_residual_img;
    std::vector<float> prev_Fn_residual_txt;
    bool has_prev_Fn_residual = false;

    std::vector<float> Bn_buffer_img;
    std::vector<float> Bn_buffer_txt;
    std::vector<float> Bn_buffer;
    bool has_Bn_buffer = false;

    TaylorSeerState taylor_state;

    bool can_cache_this_step  = false;
    bool is_caching_this_step = false;

    int total_blocks_computed = 0;
    int total_blocks_cached   = 0;

    void init(const CacheDitConfig& cfg, int num_double_blocks, int num_single_blocks, size_t h_size) {
        config              = cfg;
        total_double_blocks = num_double_blocks;
        total_single_blocks = num_single_blocks;
        hidden_size         = h_size;

        initialized = cfg.dbcache.enabled || cfg.taylorseer.enabled;

        if (!initialized)
            return;

        warmup_remaining = cfg.dbcache.max_warmup_steps;
        double_block_cache.resize(total_double_blocks);
        single_block_cache.resize(total_single_blocks);

        if (cfg.taylorseer.enabled) {
            taylor_state.init(cfg.taylorseer.n_derivatives, h_size);
        }

        reset_runtime();
    }

    void reset_runtime() {
        current_step     = -1;
        total_steps      = 0;
        warmup_remaining = config.dbcache.max_warmup_steps;
        cached_steps.clear();
        continuous_cached_steps   = 0;
        accumulated_residual_diff = 0.0f;

        for (auto& entry : double_block_cache) {
            entry.residual_img.clear();
            entry.residual_txt.clear();
            entry.prev_img.clear();
            entry.prev_txt.clear();
            entry.has_prev = false;
        }

        for (auto& entry : single_block_cache) {
            entry.residual.clear();
            entry.prev_output.clear();
            entry.has_prev = false;
        }

        Fn_residual_img.clear();
        Fn_residual_txt.clear();
        prev_Fn_residual_img.clear();
        prev_Fn_residual_txt.clear();
        has_prev_Fn_residual = false;

        Bn_buffer_img.clear();
        Bn_buffer_txt.clear();
        Bn_buffer.clear();
        has_Bn_buffer = false;

        taylor_state.reset();

        can_cache_this_step  = false;
        is_caching_this_step = false;

        total_blocks_computed = 0;
        total_blocks_cached   = 0;
    }

    bool enabled() const {
        return initialized && (config.dbcache.enabled || config.taylorseer.enabled);
    }

    void begin_step(int step_index, float sigma = 0.0f) {
        if (!enabled())
            return;
        if (step_index == current_step)
            return;

        current_step = step_index;
        total_steps++;

        bool in_warmup = warmup_remaining > 0;
        if (in_warmup) {
            warmup_remaining--;
        }

        bool scm_allows_cache = true;
        if (!config.dbcache.steps_computation_mask.empty()) {
            if (step_index < static_cast<int>(config.dbcache.steps_computation_mask.size())) {
                scm_allows_cache = (config.dbcache.steps_computation_mask[step_index] == 0);
                if (!config.dbcache.scm_policy_dynamic && scm_allows_cache) {
                    can_cache_this_step  = true;
                    is_caching_this_step = false;
                    return;
                }
            }
        }

        bool max_cached_ok = (config.dbcache.max_cached_steps < 0) ||
                             (static_cast<int>(cached_steps.size()) < config.dbcache.max_cached_steps);

        bool max_cont_ok = (config.dbcache.max_continuous_cached_steps < 0) ||
                           (continuous_cached_steps < config.dbcache.max_continuous_cached_steps);

        bool accum_ok = (config.dbcache.max_accumulated_residual_diff < 0.0f) ||
                        (accumulated_residual_diff < config.dbcache.max_accumulated_residual_diff);

        can_cache_this_step  = !in_warmup && scm_allows_cache && max_cached_ok && max_cont_ok && accum_ok && has_prev_Fn_residual;
        is_caching_this_step = false;
    }

    void end_step(bool was_cached) {
        if (was_cached) {
            cached_steps.push_back(current_step);
            continuous_cached_steps++;
        } else {
            continuous_cached_steps = 0;
        }
    }

    static float calculate_residual_diff(const float* prev, const float* curr, size_t size) {
        if (size == 0)
            return 0.0f;

        float sum_diff = 0.0f;
        float sum_abs  = 0.0f;

        for (size_t i = 0; i < size; i++) {
            sum_diff += std::fabs(prev[i] - curr[i]);
            sum_abs += std::fabs(prev[i]);
        }

        return sum_diff / (sum_abs + 1e-6f);
    }

    static float calculate_residual_diff(const std::vector<float>& prev, const std::vector<float>& curr) {
        if (prev.size() != curr.size() || prev.empty())
            return 1.0f;
        return calculate_residual_diff(prev.data(), curr.data(), prev.size());
    }

    int get_double_Fn_blocks() const {
        return (config.double_Fn_blocks >= 0) ? config.double_Fn_blocks : config.dbcache.Fn_compute_blocks;
    }

    int get_double_Bn_blocks() const {
        return (config.double_Bn_blocks >= 0) ? config.double_Bn_blocks : config.dbcache.Bn_compute_blocks;
    }

    int get_single_Fn_blocks() const {
        return (config.single_Fn_blocks >= 0) ? config.single_Fn_blocks : config.dbcache.Fn_compute_blocks;
    }

    int get_single_Bn_blocks() const {
        return (config.single_Bn_blocks >= 0) ? config.single_Bn_blocks : config.dbcache.Bn_compute_blocks;
    }

    bool is_Fn_double_block(int block_idx) const {
        return block_idx < get_double_Fn_blocks();
    }

    bool is_Bn_double_block(int block_idx) const {
        int Bn = get_double_Bn_blocks();
        return Bn > 0 && block_idx >= (total_double_blocks - Bn);
    }

    bool is_Mn_double_block(int block_idx) const {
        return !is_Fn_double_block(block_idx) && !is_Bn_double_block(block_idx);
    }

    bool is_Fn_single_block(int block_idx) const {
        return block_idx < get_single_Fn_blocks();
    }

    bool is_Bn_single_block(int block_idx) const {
        int Bn = get_single_Bn_blocks();
        return Bn > 0 && block_idx >= (total_single_blocks - Bn);
    }

    bool is_Mn_single_block(int block_idx) const {
        return !is_Fn_single_block(block_idx) && !is_Bn_single_block(block_idx);
    }

    void store_Fn_residual(const float* img, const float* txt, size_t img_size, size_t txt_size, const float* input_img, const float* input_txt) {
        Fn_residual_img.resize(img_size);
        Fn_residual_txt.resize(txt_size);

        for (size_t i = 0; i < img_size; i++) {
            Fn_residual_img[i] = img[i] - input_img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            Fn_residual_txt[i] = txt[i] - input_txt[i];
        }
    }

    bool check_cache_decision() {
        if (!can_cache_this_step) {
            is_caching_this_step = false;
            return false;
        }

        if (!has_prev_Fn_residual || prev_Fn_residual_img.empty()) {
            is_caching_this_step = false;
            return false;
        }

        float diff_img = calculate_residual_diff(prev_Fn_residual_img, Fn_residual_img);
        float diff_txt = calculate_residual_diff(prev_Fn_residual_txt, Fn_residual_txt);
        float diff     = (diff_img + diff_txt) / 2.0f;

        if (diff < config.dbcache.residual_diff_threshold) {
            is_caching_this_step = true;
            accumulated_residual_diff += diff;
            return true;
        }

        is_caching_this_step = false;
        return false;
    }

    void update_prev_Fn_residual() {
        prev_Fn_residual_img = Fn_residual_img;
        prev_Fn_residual_txt = Fn_residual_txt;
        has_prev_Fn_residual = !prev_Fn_residual_img.empty();
    }

    void store_double_block_residual(int block_idx, const float* img, const float* txt, size_t img_size, size_t txt_size, const float* prev_img, const float* prev_txt) {
        if (block_idx < 0 || block_idx >= static_cast<int>(double_block_cache.size()))
            return;

        BlockCacheEntry& entry = double_block_cache[block_idx];

        entry.residual_img.resize(img_size);
        entry.residual_txt.resize(txt_size);
        for (size_t i = 0; i < img_size; i++) {
            entry.residual_img[i] = img[i] - prev_img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            entry.residual_txt[i] = txt[i] - prev_txt[i];
        }

        entry.prev_img.resize(img_size);
        entry.prev_txt.resize(txt_size);
        for (size_t i = 0; i < img_size; i++) {
            entry.prev_img[i] = img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            entry.prev_txt[i] = txt[i];
        }
        entry.has_prev = true;
    }

    void apply_double_block_cache(int block_idx, float* img, float* txt, size_t img_size, size_t txt_size) {
        if (block_idx < 0 || block_idx >= static_cast<int>(double_block_cache.size()))
            return;

        const BlockCacheEntry& entry = double_block_cache[block_idx];
        if (entry.residual_img.size() != img_size || entry.residual_txt.size() != txt_size)
            return;

        for (size_t i = 0; i < img_size; i++) {
            img[i] += entry.residual_img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            txt[i] += entry.residual_txt[i];
        }

        total_blocks_cached++;
    }

    void store_single_block_residual(int block_idx, const float* output, size_t size, const float* input) {
        if (block_idx < 0 || block_idx >= static_cast<int>(single_block_cache.size()))
            return;

        BlockCacheEntry& entry = single_block_cache[block_idx];

        entry.residual.resize(size);
        for (size_t i = 0; i < size; i++) {
            entry.residual[i] = output[i] - input[i];
        }

        entry.prev_output.resize(size);
        for (size_t i = 0; i < size; i++) {
            entry.prev_output[i] = output[i];
        }
        entry.has_prev = true;
    }

    void apply_single_block_cache(int block_idx, float* output, size_t size) {
        if (block_idx < 0 || block_idx >= static_cast<int>(single_block_cache.size()))
            return;

        const BlockCacheEntry& entry = single_block_cache[block_idx];
        if (entry.residual.size() != size)
            return;

        for (size_t i = 0; i < size; i++) {
            output[i] += entry.residual[i];
        }

        total_blocks_cached++;
    }

    void store_Bn_buffer(const float* img, const float* txt, size_t img_size, size_t txt_size, const float* Bn_start_img, const float* Bn_start_txt) {
        Bn_buffer_img.resize(img_size);
        Bn_buffer_txt.resize(txt_size);

        for (size_t i = 0; i < img_size; i++) {
            Bn_buffer_img[i] = img[i] - Bn_start_img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            Bn_buffer_txt[i] = txt[i] - Bn_start_txt[i];
        }
        has_Bn_buffer = true;
    }

    void apply_Bn_buffer(float* img, float* txt, size_t img_size, size_t txt_size) {
        if (!has_Bn_buffer)
            return;
        if (Bn_buffer_img.size() != img_size || Bn_buffer_txt.size() != txt_size)
            return;

        for (size_t i = 0; i < img_size; i++) {
            img[i] += Bn_buffer_img[i];
        }
        for (size_t i = 0; i < txt_size; i++) {
            txt[i] += Bn_buffer_txt[i];
        }
    }

    void taylor_update(const float* hidden_state, size_t size) {
        if (!config.taylorseer.enabled)
            return;
        taylor_state.update_derivatives(hidden_state, size, current_step);
    }

    bool taylor_can_approximate() const {
        return config.taylorseer.enabled && taylor_state.can_approximate();
    }

    void taylor_approximate(float* output, size_t size) {
        if (!config.taylorseer.enabled)
            return;
        taylor_state.approximate(output, size, current_step);
    }

    bool should_use_taylor_this_step() const {
        if (!config.taylorseer.enabled)
            return false;
        if (current_step < config.taylorseer.max_warmup_steps)
            return false;

        int interval = config.taylorseer.skip_interval_steps;
        if (interval <= 0)
            interval = 1;

        return (current_step % (interval + 1)) != 0;
    }

    void log_metrics() const {
        if (!enabled())
            return;

        int total_blocks  = total_blocks_computed + total_blocks_cached;
        float cache_ratio = (total_blocks > 0) ? (static_cast<float>(total_blocks_cached) / total_blocks * 100.0f) : 0.0f;

        float step_cache_ratio = (total_steps > 0) ? (static_cast<float>(cached_steps.size()) / total_steps * 100.0f) : 0.0f;

        LOG_INFO("CacheDIT: steps_cached=%zu/%d (%.1f%%), blocks_cached=%d/%d (%.1f%%), accum_diff=%.4f",
                 cached_steps.size(), total_steps, step_cache_ratio,
                 total_blocks_cached, total_blocks, cache_ratio,
                 accumulated_residual_diff);
    }

    std::string get_summary() const {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "CacheDIT[thresh=%.2f]: cached %zu/%d steps, %d/%d blocks",
                 config.dbcache.residual_diff_threshold,
                 cached_steps.size(), total_steps,
                 total_blocks_cached, total_blocks_computed + total_blocks_cached);
        return std::string(buf);
    }
};

inline std::vector<int> parse_scm_mask(const std::string& mask_str) {
    std::vector<int> mask;
    if (mask_str.empty())
        return mask;

    size_t pos   = 0;
    size_t start = 0;
    while ((pos = mask_str.find(',', start)) != std::string::npos) {
        std::string token = mask_str.substr(start, pos - start);
        mask.push_back(std::stoi(token));
        start = pos + 1;
    }
    if (start < mask_str.length()) {
        mask.push_back(std::stoi(mask_str.substr(start)));
    }

    return mask;
}

inline std::vector<int> generate_scm_mask(
    const std::vector<int>& compute_bins,
    const std::vector<int>& cache_bins,
    int total_steps) {
    std::vector<int> mask;
    size_t c_idx = 0, cache_idx = 0;

    while (static_cast<int>(mask.size()) < total_steps) {
        if (c_idx < compute_bins.size()) {
            for (int i = 0; i < compute_bins[c_idx] && static_cast<int>(mask.size()) < total_steps; i++) {
                mask.push_back(1);
            }
            c_idx++;
        }
        if (cache_idx < cache_bins.size()) {
            for (int i = 0; i < cache_bins[cache_idx] && static_cast<int>(mask.size()) < total_steps; i++) {
                mask.push_back(0);
            }
            cache_idx++;
        }
        if (c_idx >= compute_bins.size() && cache_idx >= cache_bins.size())
            break;
    }

    if (!mask.empty()) {
        mask.back() = 1;
    }

    return mask;
}

inline std::vector<int> get_scm_preset(const std::string& preset, int total_steps) {
    struct Preset {
        std::vector<int> compute_bins;
        std::vector<int> cache_bins;
    };

    Preset slow   = {{8, 3, 3, 2, 1, 1}, {1, 2, 2, 2, 3}};
    Preset medium = {{6, 2, 2, 2, 2, 1}, {1, 3, 3, 3, 3}};
    Preset fast   = {{6, 1, 1, 1, 1, 1}, {1, 3, 4, 5, 4}};
    Preset ultra  = {{4, 1, 1, 1, 1}, {2, 5, 6, 7}};

    Preset* p = nullptr;
    if (preset == "slow" || preset == "s" || preset == "S")
        p = &slow;
    else if (preset == "medium" || preset == "m" || preset == "M")
        p = &medium;
    else if (preset == "fast" || preset == "f" || preset == "F")
        p = &fast;
    else if (preset == "ultra" || preset == "u" || preset == "U")
        p = &ultra;
    else
        return {};

    if (total_steps != 28 && total_steps > 0) {
        float scale = static_cast<float>(total_steps) / 28.0f;
        std::vector<int> scaled_compute, scaled_cache;

        for (int v : p->compute_bins) {
            scaled_compute.push_back(std::max(1, static_cast<int>(v * scale + 0.5f)));
        }
        for (int v : p->cache_bins) {
            scaled_cache.push_back(std::max(1, static_cast<int>(v * scale + 0.5f)));
        }

        return generate_scm_mask(scaled_compute, scaled_cache, total_steps);
    }

    return generate_scm_mask(p->compute_bins, p->cache_bins, total_steps);
}

inline float get_preset_threshold(const std::string& preset) {
    if (preset == "slow" || preset == "s" || preset == "S")
        return 0.20f;
    if (preset == "medium" || preset == "m" || preset == "M")
        return 0.25f;
    if (preset == "fast" || preset == "f" || preset == "F")
        return 0.30f;
    if (preset == "ultra" || preset == "u" || preset == "U")
        return 0.34f;
    return 0.08f;
}

inline int get_preset_warmup(const std::string& preset) {
    if (preset == "slow" || preset == "s" || preset == "S")
        return 8;
    if (preset == "medium" || preset == "m" || preset == "M")
        return 6;
    if (preset == "fast" || preset == "f" || preset == "F")
        return 6;
    if (preset == "ultra" || preset == "u" || preset == "U")
        return 4;
    return 8;
}

inline int get_preset_Fn(const std::string& preset) {
    if (preset == "slow" || preset == "s" || preset == "S")
        return 8;
    if (preset == "medium" || preset == "m" || preset == "M")
        return 8;
    if (preset == "fast" || preset == "f" || preset == "F")
        return 6;
    if (preset == "ultra" || preset == "u" || preset == "U")
        return 4;
    return 8;
}

inline int get_preset_Bn(const std::string& preset) {
    (void)preset;
    return 0;
}

inline void parse_dbcache_options(const std::string& opts, DBCacheConfig& cfg) {
    if (opts.empty())
        return;

    int Fn = 8, Bn = 0, warmup = 8, max_cached = -1, max_cont = -1;
    float thresh = 0.08f;

    sscanf(opts.c_str(), "%d,%d,%f,%d,%d,%d",
           &Fn, &Bn, &thresh, &warmup, &max_cached, &max_cont);

    cfg.Fn_compute_blocks           = Fn;
    cfg.Bn_compute_blocks           = Bn;
    cfg.residual_diff_threshold     = thresh;
    cfg.max_warmup_steps            = warmup;
    cfg.max_cached_steps            = max_cached;
    cfg.max_continuous_cached_steps = max_cont;
}

inline void parse_taylorseer_options(const std::string& opts, TaylorSeerConfig& cfg) {
    if (opts.empty())
        return;

    int n_deriv = 1, warmup = 2, interval = 1;
    sscanf(opts.c_str(), "%d,%d,%d", &n_deriv, &warmup, &interval);

    cfg.n_derivatives       = n_deriv;
    cfg.max_warmup_steps    = warmup;
    cfg.skip_interval_steps = interval;
}

struct CacheDitConditionState {
    DBCacheConfig config;
    TaylorSeerConfig taylor_config;
    bool initialized = false;

    int current_step_index = -1;
    bool step_active       = false;
    bool skip_current_step = false;
    bool initial_step      = true;
    int warmup_remaining   = 0;
    std::vector<int> cached_steps;
    int continuous_cached_steps     = 0;
    float accumulated_residual_diff = 0.0f;
    int total_steps_skipped         = 0;

    const void* anchor_condition = nullptr;

    struct CacheEntry {
        std::vector<float> diff;
        std::vector<float> prev_input;
        std::vector<float> prev_output;
        bool has_prev = false;
    };
    std::unordered_map<const void*, CacheEntry> cache_diffs;

    TaylorSeerState taylor_state;

    float start_sigma = std::numeric_limits<float>::max();
    float end_sigma   = 0.0f;

    void reset_runtime() {
        current_step_index = -1;
        step_active        = false;
        skip_current_step  = false;
        initial_step       = true;
        warmup_remaining   = config.max_warmup_steps;
        cached_steps.clear();
        continuous_cached_steps   = 0;
        accumulated_residual_diff = 0.0f;
        total_steps_skipped       = 0;
        anchor_condition          = nullptr;
        cache_diffs.clear();
        taylor_state.reset();
    }

    void init(const DBCacheConfig& dbcfg, const TaylorSeerConfig& tcfg) {
        config        = dbcfg;
        taylor_config = tcfg;
        initialized   = dbcfg.enabled || tcfg.enabled;
        reset_runtime();

        if (taylor_config.enabled) {
            taylor_state.init(taylor_config.n_derivatives, 0);
        }
    }

    void set_sigmas(const std::vector<float>& sigmas) {
        if (!initialized || sigmas.size() < 2)
            return;

        float start_percent = 0.15f;
        float end_percent   = 0.95f;

        size_t n_steps    = sigmas.size() - 1;
        size_t start_step = static_cast<size_t>(start_percent * n_steps);
        size_t end_step   = static_cast<size_t>(end_percent * n_steps);

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
        return initialized && (config.enabled || taylor_config.enabled);
    }

    void begin_step(int step_index, float sigma) {
        if (!enabled())
            return;
        if (step_index == current_step_index)
            return;

        current_step_index = step_index;
        skip_current_step  = false;
        step_active        = false;

        if (sigma > start_sigma)
            return;
        if (!(sigma > end_sigma))
            return;

        step_active = true;

        if (warmup_remaining > 0) {
            warmup_remaining--;
            return;
        }

        if (!config.steps_computation_mask.empty()) {
            if (step_index < static_cast<int>(config.steps_computation_mask.size())) {
                if (config.steps_computation_mask[step_index] == 1) {
                    return;
                }
            }
        }

        if (config.max_cached_steps >= 0 &&
            static_cast<int>(cached_steps.size()) >= config.max_cached_steps) {
            return;
        }

        if (config.max_continuous_cached_steps >= 0 &&
            continuous_cached_steps >= config.max_continuous_cached_steps) {
            return;
        }
    }

    bool step_is_active() const {
        return enabled() && step_active;
    }

    bool is_step_skipped() const {
        return enabled() && step_active && skip_current_step;
    }

    bool has_cache(const void* cond) const {
        auto it = cache_diffs.find(cond);
        return it != cache_diffs.end() && !it->second.diff.empty();
    }

    void update_cache(const void* cond, const float* input, const float* output, size_t size) {
        CacheEntry& entry = cache_diffs[cond];
        entry.diff.resize(size);
        for (size_t i = 0; i < size; i++) {
            entry.diff[i] = output[i] - input[i];
        }

        entry.prev_input.resize(size);
        entry.prev_output.resize(size);
        for (size_t i = 0; i < size; i++) {
            entry.prev_input[i]  = input[i];
            entry.prev_output[i] = output[i];
        }
        entry.has_prev = true;
    }

    void apply_cache(const void* cond, const float* input, float* output, size_t size) {
        auto it = cache_diffs.find(cond);
        if (it == cache_diffs.end() || it->second.diff.empty())
            return;
        if (it->second.diff.size() != size)
            return;

        for (size_t i = 0; i < size; i++) {
            output[i] = input[i] + it->second.diff[i];
        }
    }

    bool before_condition(const void* cond, struct ggml_tensor* input, struct ggml_tensor* output, float sigma, int step_index) {
        if (!enabled() || step_index < 0)
            return false;

        if (step_index != current_step_index) {
            begin_step(step_index, sigma);
        }

        if (!step_active)
            return false;

        if (initial_step) {
            anchor_condition = cond;
            initial_step     = false;
        }

        bool is_anchor = (cond == anchor_condition);

        if (skip_current_step) {
            if (has_cache(cond)) {
                apply_cache(cond, (float*)input->data, (float*)output->data,
                            static_cast<size_t>(ggml_nelements(output)));
                return true;
            }
            return false;
        }

        if (!is_anchor)
            return false;

        auto it = cache_diffs.find(cond);
        if (it == cache_diffs.end() || !it->second.has_prev)
            return false;

        size_t ne = static_cast<size_t>(ggml_nelements(input));
        if (it->second.prev_input.size() != ne)
            return false;

        float* input_data = (float*)input->data;
        float diff        = CacheDitState::calculate_residual_diff(
                   it->second.prev_input.data(), input_data, ne);

        float effective_threshold = config.residual_diff_threshold;
        if (config.Fn_compute_blocks > 0) {
            float fn_confidence = 1.0f + 0.02f * (config.Fn_compute_blocks - 8);
            fn_confidence       = std::max(0.5f, std::min(2.0f, fn_confidence));
            effective_threshold *= fn_confidence;
        }
        if (config.Bn_compute_blocks > 0) {
            float bn_quality = 1.0f - 0.03f * config.Bn_compute_blocks;
            bn_quality       = std::max(0.5f, std::min(1.0f, bn_quality));
            effective_threshold *= bn_quality;
        }

        if (diff < effective_threshold) {
            skip_current_step = true;
            total_steps_skipped++;
            cached_steps.push_back(current_step_index);
            continuous_cached_steps++;
            accumulated_residual_diff += diff;
            apply_cache(cond, input_data, (float*)output->data, ne);
            return true;
        }

        continuous_cached_steps = 0;
        return false;
    }

    void after_condition(const void* cond, struct ggml_tensor* input, struct ggml_tensor* output) {
        if (!step_is_active())
            return;

        size_t ne = static_cast<size_t>(ggml_nelements(output));
        update_cache(cond, (float*)input->data, (float*)output->data, ne);

        if (cond == anchor_condition && taylor_config.enabled) {
            taylor_state.update_derivatives((float*)output->data, ne, current_step_index);
        }
    }

    void log_metrics() const {
        if (!enabled())
            return;

        LOG_INFO("CacheDIT: steps_skipped=%d/%d (%.1f%%), accum_residual_diff=%.4f",
                 total_steps_skipped,
                 current_step_index + 1,
                 (current_step_index > 0) ? (100.0f * total_steps_skipped / (current_step_index + 1)) : 0.0f,
                 accumulated_residual_diff);
    }
};

#endif

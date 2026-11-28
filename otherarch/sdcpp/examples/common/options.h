#ifndef SD_EXAMPLES_OPTIONS_H
#define SD_EXAMPLES_OPTIONS_H

#include <cmath>
#include <cstddef>
#include <json.hpp>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace nlohmann {
    template <typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& out, const std::optional<T>& value) {
            out = value ? json(*value) : json(nullptr);
        }
        static void from_json(const json& in, std::optional<T>& value) {
            value = in.is_null() ? std::nullopt : std::optional<T>(in.get<T>());
        }
    };
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::optional<T>& value) {
    if (!value)
        return out << "unset";
    if constexpr (std::is_same_v<T, bool>)
        return out << (*value ? "true" : "false");
    else
        return out << *value;
}

template <typename T>
class OptionTarget {
    T* value_                = nullptr;
    std::optional<T>* input_ = nullptr;

public:
    OptionTarget(T* value)
        : value_(value) {}
    OptionTarget(std::optional<T>* input)
        : input_(input) {}
    T get() const { return input_ ? input_->value_or(T{}) : *value_; }
    void set(T value) const {
        if (input_)
            *input_ = std::move(value);
        else
            *value_ = std::move(value);
    }
};

template <typename T>
struct InputType {
    using type = T;
};
template <typename T>
struct InputType<std::optional<T>> {
    using type = T;
};

struct SDTilingParams {
    std::optional<bool> enabled;
    std::optional<bool> temporal_tiling;
    std::optional<int> tile_size_w;
    std::optional<int> tile_size_h;
    std::optional<float> target_overlap;
    std::optional<float> rel_size_w;
    std::optional<float> rel_size_h;
    const char* extra_tiling_args = nullptr;
};

struct SDSLGParams {
    std::optional<float> layer_start;
    std::optional<float> layer_end;
    std::optional<float> scale;
};

struct SDGuidanceParams {
    std::optional<float> txt_cfg;
    std::optional<float> img_cfg;
    std::optional<float> distilled_guidance;
    SDSLGParams slg;
};

struct SDSampleParams {
    SDGuidanceParams guidance;
    std::optional<std::string> scheduler;
    std::optional<std::string> sample_method;
    std::optional<int> sample_steps;
    std::optional<float> eta;
    std::optional<int> shifted_timestep;
    std::optional<float> flow_shift;
};

struct SDCacheParams {
    std::optional<std::string> mode;
    std::optional<float> reuse_threshold;
    std::optional<float> start_percent;
    std::optional<float> end_percent;
    std::optional<float> error_decay_rate;
    std::optional<bool> use_relative_threshold;
    std::optional<bool> reset_error_on_compute;
    std::optional<int> Fn_compute_blocks;
    std::optional<int> Bn_compute_blocks;
    std::optional<float> residual_diff_threshold;
    std::optional<int> max_warmup_steps;
    std::optional<int> max_cached_steps;
    std::optional<int> max_continuous_cached_steps;
    std::optional<int> taylorseer_n_derivatives;
    std::optional<int> taylorseer_skip_interval;
    const char* scm_mask = nullptr;
    std::optional<bool> scm_policy_dynamic;
    std::optional<float> spectrum_w;
    std::optional<int> spectrum_m;
    std::optional<float> spectrum_lam;
    std::optional<int> spectrum_window_size;
    std::optional<float> spectrum_flex_window;
    std::optional<int> spectrum_warmup_steps;
    std::optional<float> spectrum_stop_percent;
};

namespace example_options {
    inline constexpr const char* sample_methods[] = {
        "euler",
        "euler_a",
        "heun",
        "dpm2",
        "dpm++2s_a",
        "dpm++2m",
        "dpm++2mv2",
        "ipndm",
        "ipndm_v",
        "lcm",
        "ddim_trailing",
        "tcd",
        "res_multistep",
        "res_2s",
        "er_sde",
        "euler_cfg_pp",
        "euler_a_cfg_pp",
        "euler_ge",
        "dpm++2m_sde",
        "dpm++2m_sde_bt",
        "lms",
    };

    inline constexpr const char* schedulers[] = {
        "discrete",
        "karras",
        "exponential",
        "ays",
        "gits",
        "sgm_uniform",
        "simple",
        "smoothstep",
        "kl_optimal",
        "lcm",
        "bong_tangent",
        "ltx2",
        "logit_normal",
        "flux2",
        "flux",
        "beta",
        "llada_image",
    };

    inline constexpr const char* preview_modes[] = {
        "none",
        "proj",
        "tae",
        "vae",
    };

    template <size_t N>
    bool contains(const char* const (&values)[N], const std::string& value) {
        for (const char* entry : values)
            if (value == entry)
                return true;
        return false;
    }
}

#endif

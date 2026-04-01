#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include <cmath>

#include "ggml_extend.hpp"
#include "gits_noise.inl"
#include "tensor.hpp"

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000
#define FLUX_TIMESTEPS 1000

struct SigmaScheduler {
    typedef std::function<float(float)> t_to_sigma_t;

    virtual std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) = 0;
};

struct DiscreteScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct ExponentialScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> sigmas;

        // Calculate step size
        float log_sigma_min = std::log(sigma_min);
        float log_sigma_max = std::log(sigma_max);
        float step          = (log_sigma_max - log_sigma_min) / (n - 1);

        // Fill sigmas with exponential values
        for (uint32_t i = 0; i < n; ++i) {
            float sigma = std::exp(log_sigma_max - step * i);
            sigmas.push_back(sigma);
        }

        sigmas.push_back(0.0f);

        return sigmas;
    }
};

/* interp and linear_interp adapted from dpilger26's NumCpp library:
 * https://github.com/dpilger26/NumCpp/tree/5e40aab74d14e257d65d3dc385c9ff9e2120c60e */
constexpr double interp(double left, double right, double perc) noexcept {
    return (left * (1. - perc)) + (right * perc);
}

/* This will make the assumption that the reference x and y values are
 * already sorted in ascending order because they are being generated as
 * such in the calling function */
inline std::vector<double> linear_interp(std::vector<float> new_x,
                                         const std::vector<float> ref_x,
                                         const std::vector<float> ref_y) {
    const size_t len_x = new_x.size();
    size_t i           = 0;
    size_t j           = 0;
    std::vector<double> new_y(len_x);

    if (ref_x.size() != ref_y.size()) {
        LOG_ERROR("Linear Interpolation Failed: length mismatch");
        return new_y;
    }

    /* Adjusted bounds checking to ensure new_x is within ref_x range */
    if (new_x[0] < ref_x[0]) {
        new_x[0] = ref_x[0];
    }
    if (new_x.back() > ref_x.back()) {
        new_x.back() = ref_x.back();
    }

    while (i < len_x) {
        if ((ref_x[j] > new_x[i]) || (new_x[i] > ref_x[j + 1])) {
            j++;
            continue;
        }

        const double perc = static_cast<double>(new_x[i] - ref_x[j]) / static_cast<double>(ref_x[j + 1] - ref_x[j]);

        new_y[i] = interp(ref_y[j], ref_y[j + 1], perc);
        i++;
    }

    return new_y;
}

inline std::vector<float> linear_space(const float start, const float end, const size_t num_points) {
    std::vector<float> result(num_points);
    const float inc = (end - start) / (static_cast<float>(num_points - 1));

    if (num_points > 0) {
        result[0] = start;

        for (size_t i = 1; i < num_points; i++) {
            result[i] = result[i - 1] + inc;
        }
    }

    return result;
}

inline std::vector<float> log_linear_interpolation(std::vector<float> sigma_in,
                                                   const size_t new_len) {
    const size_t s_len        = sigma_in.size();
    std::vector<float> x_vals = linear_space(0.f, 1.f, s_len);
    std::vector<float> y_vals(s_len);

    /* Reverses the input array to be ascending instead of descending,
     * also hits it with a log, it is log-linear interpolation after all */
    for (size_t i = 0; i < s_len; i++) {
        y_vals[i] = std::log(sigma_in[s_len - i - 1]);
    }

    std::vector<float> new_x_vals  = linear_space(0.f, 1.f, new_len);
    std::vector<double> new_y_vals = linear_interp(new_x_vals, x_vals, y_vals);
    std::vector<float> results(new_len);

    for (size_t i = 0; i < new_len; i++) {
        results[i] = static_cast<float>(std::exp(new_y_vals[new_len - i - 1]));
    }

    return results;
}

/*
https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
*/
struct AYSScheduler : SigmaScheduler {
    SDVersion version;
    explicit AYSScheduler(SDVersion version)
        : version(version) {}
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        const std::vector<float> noise_levels[] = {
            /* SD1.5 */
            {14.6146412293f, 6.4745760956f, 3.8636745985f, 2.6946151520f,
             1.8841921177f, 1.3943805092f, 0.9642583904f, 0.6523686016f,
             0.3977456272f, 0.1515232662f, 0.0291671582f},
            /* SDXL */
            {14.6146412293f, 6.3184485287f, 3.7681790315f, 2.1811480769f,
             1.3405244945f, 0.8620721141f, 0.5550693289f, 0.3798540708f,
             0.2332364134f, 0.1114188177f, 0.0291671582f},
            /* SVD */
            {700.00f, 54.5f, 15.886f, 7.977f, 4.248f, 1.789f, 0.981f, 0.403f,
             0.173f, 0.034f, 0.002f},
        };

        std::vector<float> inputs;
        std::vector<float> results(n + 1);

        if (sd_version_is_sd2((SDVersion)version)) {
            LOG_WARN("AYS_SCHEDULER not designed for SD2.X models");
        } /* fallthrough */
        else if (sd_version_is_sd1((SDVersion)version)) {
            LOG_INFO("AYS_SCHEDULER using SD1.5 noise levels");
            inputs = noise_levels[0];
        } else if (sd_version_is_sdxl((SDVersion)version)) {
            LOG_INFO("AYS_SCHEDULER using SDXL noise levels");
            inputs = noise_levels[1];
        } else if (version == VERSION_SVD) {
            LOG_INFO("AYS_SCHEDULER using SVD noise levels");
            inputs = noise_levels[2];
        } else {
            LOG_ERROR("Version not compatible with AYS_SCHEDULER scheduler");
            return results;
        }

        /* Stretches those pre-calculated reference levels out to the desired
         * size using log-linear interpolation */
        if ((n + 1) != inputs.size()) {
            results = log_linear_interpolation(inputs, n + 1);
        } else {
            results = inputs;
        }

        /* Not sure if this is strictly neccessary */
        results[n] = 0.0f;

        return results;
    }
};

/*
 * GITS Scheduler: https://github.com/zju-pi/diff-sampler/tree/main/gits-main
 */
struct GITSScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        if (sigma_max <= 0.0f) {
            return std::vector<float>{};
        }

        std::vector<float> sigmas;

        // Assume coeff is provided (replace 1.20 with your dynamic coeff)
        float coeff = 1.20f;  // Default coefficient
        // Normalize coeff to the closest value in the array (0.80 to 1.50)
        coeff = std::round(coeff * 20.0f) / 20.0f;  // Round to the nearest 0.05
        // Calculate the index based on the coefficient
        int index = static_cast<int>((coeff - 0.80f) / 0.05f);
        // Ensure the index is within bounds
        index                                                 = std::max(0, std::min(index, static_cast<int>(GITS_NOISE.size() - 1)));
        const std::vector<std::vector<float>>& selected_noise = *GITS_NOISE[index];

        if (n <= 20) {
            sigmas = (selected_noise)[n - 2];
        } else {
            sigmas = log_linear_interpolation(selected_noise.back(), n + 1);
        }

        sigmas[n] = 0.0f;
        return sigmas;
    }
};

struct SGMUniformScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min_in, float sigma_max_in, t_to_sigma_t t_to_sigma_func) override {
        std::vector<float> result;
        if (n == 0) {
            result.push_back(0.0f);
            return result;
        }
        result.reserve(n + 1);
        int t_max                    = TIMESTEPS - 1;
        int t_min                    = 0;
        std::vector<float> timesteps = linear_space(static_cast<float>(t_max), static_cast<float>(t_min), n + 1);
        for (uint32_t i = 0; i < n; i++) {
            result.push_back(t_to_sigma_func(timesteps[i]));
        }
        result.push_back(0.0f);
        return result;
    }
};

struct LCMScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        result.reserve(n + 1);
        const int original_steps = 50;
        const int k              = TIMESTEPS / original_steps;
        for (uint32_t i = 0; i < n; i++) {
            // the rounding ensures we match the training schedule of the LCM model
            int index    = (i * original_steps) / n;
            int timestep = (original_steps - index) * k - 1;
            result.push_back(t_to_sigma(static_cast<float>(timestep)));
        }
        result.push_back(0.0f);
        return result;
    }
};

struct KarrasScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float rho = 7.f;

        if (sigma_min <= 1e-6f) {
            sigma_min = 1e-6f;
        }

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

struct SimpleScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result_sigmas;

        if (n == 0) {
            return result_sigmas;
        }

        result_sigmas.reserve(n + 1);

        int model_sigmas_len = TIMESTEPS;

        float step_factor = static_cast<float>(model_sigmas_len) / static_cast<float>(n);

        for (uint32_t i = 0; i < n; ++i) {
            int offset_from_start_of_py_array = static_cast<int>(static_cast<float>(i) * step_factor);
            int timestep_index                = model_sigmas_len - 1 - offset_from_start_of_py_array;

            if (timestep_index < 0) {
                timestep_index = 0;
            }

            result_sigmas.push_back(t_to_sigma(static_cast<float>(timestep_index)));
        }
        result_sigmas.push_back(0.0f);
        return result_sigmas;
    }
};

// Close to Beta Scheduler, but increadably simple in code.
struct SmoothStepScheduler : SigmaScheduler {
    static constexpr float smoothstep(float x) {
        return x * x * (3.0f - 2.0f * x);
    }

    std::vector<float> get_sigmas(uint32_t n, float /*sigma_min*/, float /*sigma_max*/, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        result.reserve(n + 1);

        const int t_max = TIMESTEPS - 1;
        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0.f);
            return result;
        }

        for (uint32_t i = 0; i < n; i++) {
            float u = 1.f - float(i) / float(n);
            result.push_back(t_to_sigma(std::round(smoothstep(u) * t_max)));
        }

        result.push_back(0.f);
        return result;
    }
};

struct BongTangentScheduler : SigmaScheduler {
    static constexpr float kPi = 3.14159265358979323846f;

    static std::vector<float> get_bong_tangent_sigmas(int steps, float slope, float pivot, float start, float end) {
        std::vector<float> sigmas;
        if (steps <= 0) {
            return sigmas;
        }

        float smax   = ((2.0f / kPi) * atanf(-slope * (0.0f - pivot)) + 1.0f) * 0.5f;
        float smin   = ((2.0f / kPi) * atanf(-slope * ((float)(steps - 1) - pivot)) + 1.0f) * 0.5f;
        float srange = smax - smin;
        float sscale = start - end;

        sigmas.reserve(steps);

        if (fabsf(srange) < 1e-8f) {
            if (steps == 1) {
                sigmas.push_back(start);
                return sigmas;
            }
            for (int i = 0; i < steps; ++i) {
                float t = (float)i / (float)(steps - 1);
                sigmas.push_back(start + (end - start) * t);
            }
            return sigmas;
        }

        float inv_srange = 1.0f / srange;
        for (int x = 0; x < steps; ++x) {
            float v     = ((2.0f / kPi) * atanf(-slope * ((float)x - pivot)) + 1.0f) * 0.5f;
            float sigma = ((v - smin) * inv_srange) * sscale + end;
            sigmas.push_back(sigma);
        }

        return sigmas;
    }

    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t /*t_to_sigma*/) override {
        std::vector<float> result;
        if (n == 0) {
            return result;
        }

        float start  = sigma_max;
        float end    = sigma_min;
        float middle = sigma_min + (sigma_max - sigma_min) * 0.5f;

        float pivot_1 = 0.6f;
        float pivot_2 = 0.6f;
        float slope_1 = 0.2f;
        float slope_2 = 0.2f;

        int steps     = static_cast<int>(n) + 2;
        int midpoint  = static_cast<int>(((float)steps * pivot_1 + (float)steps * pivot_2) * 0.5f);
        int pivot_1_i = static_cast<int>((float)steps * pivot_1);
        int pivot_2_i = static_cast<int>((float)steps * pivot_2);

        float slope_scale = (float)steps / 40.0f;
        slope_1           = slope_1 / slope_scale;
        slope_2           = slope_2 / slope_scale;

        int stage_2_len = steps - midpoint;
        int stage_1_len = steps - stage_2_len;

        std::vector<float> sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, (float)pivot_1_i, start, middle);
        std::vector<float> sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, (float)(pivot_2_i - stage_1_len), middle, end);

        if (!sigmas_1.empty()) {
            sigmas_1.pop_back();
        }

        result.reserve(n + 1);
        result.insert(result.end(), sigmas_1.begin(), sigmas_1.end());
        result.insert(result.end(), sigmas_2.begin(), sigmas_2.end());

        if (result.size() < n + 1) {
            while (result.size() < n + 1) {
                result.push_back(end);
            }
        } else if (result.size() > n + 1) {
            result.resize(n + 1);
        }

        result[n] = 0.0f;
        return result;
    }
};

struct KLOptimalScheduler : SigmaScheduler {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> sigmas;

        if (n == 0) {
            return sigmas;
        }

        if (n == 1) {
            sigmas.push_back(sigma_max);
            sigmas.push_back(0.0f);
            return sigmas;
        }

        if (sigma_min <= 1e-6f) {
            sigma_min = 1e-6f;
        }

        sigmas.reserve(n + 1);

        float alpha_min = std::atan(sigma_min);
        float alpha_max = std::atan(sigma_max);

        for (uint32_t i = 0; i < n; ++i) {
            float t     = static_cast<float>(i) / static_cast<float>(n - 1);
            float angle = t * alpha_min + (1.0f - t) * alpha_max;
            sigmas.push_back(std::tan(angle));
        }

        sigmas.push_back(0.0f);

        return sigmas;
    }
};

struct Denoiser {
    virtual float sigma_min()                                                        = 0;
    virtual float sigma_max()                                                        = 0;
    virtual float sigma_to_t(float sigma)                                            = 0;
    virtual float t_to_sigma(float t)                                                = 0;
    virtual std::vector<float> get_scalings(float sigma)                             = 0;
    virtual sd::Tensor<float> noise_scaling(float sigma,
                                            const sd::Tensor<float>& noise,
                                            const sd::Tensor<float>& latent)         = 0;
    virtual sd::Tensor<float> inverse_noise_scaling(float sigma,
                                                    const sd::Tensor<float>& latent) = 0;

    virtual std::vector<float> get_sigmas(uint32_t n, int /*image_seq_len*/, scheduler_t scheduler_type, SDVersion version) {
        auto bound_t_to_sigma = std::bind(&Denoiser::t_to_sigma, this, std::placeholders::_1);
        std::shared_ptr<SigmaScheduler> scheduler;
        switch (scheduler_type) {
            case DISCRETE_SCHEDULER:
                LOG_INFO("get_sigmas with discrete scheduler");
                scheduler = std::make_shared<DiscreteScheduler>();
                break;
            case KARRAS_SCHEDULER:
                LOG_INFO("get_sigmas with Karras scheduler");
                scheduler = std::make_shared<KarrasScheduler>();
                break;
            case EXPONENTIAL_SCHEDULER:
                LOG_INFO("get_sigmas exponential scheduler");
                scheduler = std::make_shared<ExponentialScheduler>();
                break;
            case AYS_SCHEDULER:
                LOG_INFO("get_sigmas with Align-Your-Steps scheduler");
                scheduler = std::make_shared<AYSScheduler>(version);
                break;
            case GITS_SCHEDULER:
                LOG_INFO("get_sigmas with GITS scheduler");
                scheduler = std::make_shared<GITSScheduler>();
                break;
            case SGM_UNIFORM_SCHEDULER:
                LOG_INFO("get_sigmas with SGM Uniform scheduler");
                scheduler = std::make_shared<SGMUniformScheduler>();
                break;
            case SIMPLE_SCHEDULER:
                LOG_INFO("get_sigmas with Simple scheduler");
                scheduler = std::make_shared<SimpleScheduler>();
                break;
            case SMOOTHSTEP_SCHEDULER:
                LOG_INFO("get_sigmas with SmoothStep scheduler");
                scheduler = std::make_shared<SmoothStepScheduler>();
                break;
            case BONG_TANGENT_SCHEDULER:
                LOG_INFO("get_sigmas with bong_tangent scheduler");
                scheduler = std::make_shared<BongTangentScheduler>();
                break;
            case KL_OPTIMAL_SCHEDULER:
                LOG_INFO("get_sigmas with KL Optimal scheduler");
                scheduler = std::make_shared<KLOptimalScheduler>();
                break;
            case LCM_SCHEDULER:
                LOG_INFO("get_sigmas with LCM scheduler");
                scheduler = std::make_shared<LCMScheduler>();
                break;
            default:
                LOG_INFO("get_sigmas with discrete scheduler (default)");
                scheduler = std::make_shared<DiscreteScheduler>();
                break;
        }
        return scheduler->get_sigmas(n, sigma_min(), sigma_max(), bound_t_to_sigma);
    }
};

struct CompVisDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float sigma_data = 1.0f;

    float sigma_min() override {
        return sigmas[0];
    }

    float sigma_max() override {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) override {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) override {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }

    std::vector<float> get_scalings(float sigma) override {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }

    virtual sd::Tensor<float> noise_scaling(float sigma,
                                            const sd::Tensor<float>& noise,
                                            const sd::Tensor<float>& latent) override {
        GGML_ASSERT(noise.numel() == latent.numel());
        return latent + noise * sigma;
    }

    sd::Tensor<float> inverse_noise_scaling(float sigma, const sd::Tensor<float>& latent) override {
        SD_UNUSED(sigma);
        return latent;
    }
};

struct CompVisVDenoiser : public CompVisDenoiser {
    std::vector<float> get_scalings(float sigma) override {
        float c_skip = sigma_data * sigma_data / (sigma * sigma + sigma_data * sigma_data);
        float c_out  = -sigma * sigma_data / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }
};

struct EDMVDenoiser : public CompVisVDenoiser {
    float min_sigma = 0.002f;
    float max_sigma = 120.0f;

    EDMVDenoiser(float min_sigma = 0.002, float max_sigma = 120.0)
        : min_sigma(min_sigma), max_sigma(max_sigma) {
    }

    float t_to_sigma(float t) override {
        return std::exp(t * 4 / (float)TIMESTEPS);
    }

    float sigma_to_t(float s) override {
        return 0.25f * std::log(s);
    }

    float sigma_min() override {
        return min_sigma;
    }

    float sigma_max() override {
        return max_sigma;
    }
};

inline float time_snr_shift(float alpha, float t) {
    if (alpha == 1.0f) {
        return t;
    }
    return alpha * t / (1 + (alpha - 1) * t);
}

struct DiscreteFlowDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float shift = 3.0f;

    float sigma_data = 1.0f;

    DiscreteFlowDenoiser(float shift = 3.0f) {
        set_shift(shift);
    }

    void set_parameters() {
        for (int i = 1; i < TIMESTEPS + 1; i++) {
            sigmas[i - 1] = t_to_sigma(static_cast<float>(i));
        }
    }

    void set_shift(float shift) {
        this->shift = shift;
        set_parameters();
    }

    float sigma_min() override {
        return sigmas[0];
    }

    float sigma_max() override {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) override {
        return sigma * 1000.f;
    }

    float t_to_sigma(float t) override {
        t = t + 1;
        return time_snr_shift(shift, t / 1000.f);
    }

    std::vector<float> get_scalings(float sigma) override {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    sd::Tensor<float> noise_scaling(float sigma,
                                    const sd::Tensor<float>& noise,
                                    const sd::Tensor<float>& latent) override {
        GGML_ASSERT(noise.numel() == latent.numel());
        return latent * (1.0f - sigma) + noise * sigma;
    }
    sd::Tensor<float> inverse_noise_scaling(float sigma, const sd::Tensor<float>& latent) override {
        return latent * (1.0f / (1.0f - sigma));
    }
};

inline float flux_time_shift(float mu, float sigma, float t) {
    return ::expf(mu) / (::expf(mu) + ::powf((1.0f / t - 1.0f), sigma));
}

struct FluxFlowDenoiser : public DiscreteFlowDenoiser {
    FluxFlowDenoiser() = default;

    float sigma_to_t(float sigma) override {
        return sigma;
    }

    float t_to_sigma(float t) override {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }
};

struct Flux2FlowDenoiser : public FluxFlowDenoiser {
    Flux2FlowDenoiser() = default;

    float compute_empirical_mu(uint32_t n, int image_seq_len) {
        const float a1 = 8.73809524e-05f;
        const float b1 = 1.89833333f;
        const float a2 = 0.00016927f;
        const float b2 = 0.45666666f;

        if (image_seq_len > 4300) {
            float mu = a2 * image_seq_len + b2;
            return mu;
        }

        float m_200 = a2 * image_seq_len + b2;
        float m_10  = a1 * image_seq_len + b1;

        float a  = (m_200 - m_10) / 190.0f;
        float b  = m_200 - 200.0f * a;
        float mu = a * n + b;

        return mu;
    }

    std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t scheduler_type, SDVersion version) override {
        float mu = compute_empirical_mu(n, image_seq_len);
        LOG_DEBUG("Flux2FlowDenoiser: set shift to %.3f", mu);
        set_shift(mu);
        return Denoiser::get_sigmas(n, image_seq_len, scheduler_type, version);
    }
};

typedef std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)> denoise_cb_t;

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static sd::Tensor<float> sample_k_diffusion(sample_method_t method,
                                            denoise_cb_t model,
                                            sd::Tensor<float> x,
                                            std::vector<float> sigmas,
                                            std::shared_ptr<RNG> rng,
                                            float eta) {
    size_t steps = sigmas.size() - 1;
    switch (method) {
        case EULER_A_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                float sigma       = sigmas[i];
                auto denoised_opt = model(x, sigma, i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                sd::Tensor<float> d        = (x - denoised) / sigma;
                float sigma_up             = std::min(sigmas[i + 1],
                                                      std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down           = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                float dt                   = sigma_down - sigmas[i];
                x += d * dt;
                if (sigmas[i + 1] > 0) {
                    x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
                }
            }
            return x;
        }
        case EULER_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                float sigma       = sigmas[i];
                auto denoised_opt = model(x, sigma, i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                sd::Tensor<float> d        = (x - denoised) / sigma;
                float dt                   = sigmas[i + 1] - sigma;
                x += d * dt;
            }
            return x;
        }
        case HEUN_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], -(i + 1));
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                sd::Tensor<float> d        = (x - denoised) / sigmas[i];
                float dt                   = sigmas[i + 1] - sigmas[i];
                if (sigmas[i + 1] == 0) {
                    x += d * dt;
                } else {
                    sd::Tensor<float> x2 = x + d * dt;
                    auto denoised2_opt   = model(x2, sigmas[i + 1], i + 1);
                    if (denoised2_opt.empty()) {
                        return {};
                    }
                    sd::Tensor<float> denoised2 = std::move(denoised2_opt);
                    d                           = (d + (x2 - denoised2) / sigmas[i + 1]) / 2.0f;
                    x += d * dt;
                }
            }
            return x;
        }
        case DPM2_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], -(i + 1));
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                sd::Tensor<float> d        = (x - denoised) / sigmas[i];
                if (sigmas[i + 1] == 0) {
                    float dt = sigmas[i + 1] - sigmas[i];
                    x += d * dt;
                } else {
                    float sigma_mid      = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                    float dt_1           = sigma_mid - sigmas[i];
                    float dt_2           = sigmas[i + 1] - sigmas[i];
                    sd::Tensor<float> x2 = x + d * dt_1;
                    auto denoised2_opt   = model(x2, sigma_mid, i + 1);
                    if (denoised2_opt.empty()) {
                        return {};
                    }
                    sd::Tensor<float> denoised2 = std::move(denoised2_opt);
                    x += ((x2 - denoised2) / sigma_mid) * dt_2;
                }
            }
            return x;
        }
        case DPMPP2S_A_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], -(i + 1));
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                float sigma_up             = std::min(sigmas[i + 1],
                                                      std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down           = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                auto t_fn                  = [](float sigma) -> float { return -log(sigma); };
                auto sigma_fn              = [](float t) -> float { return exp(-t); };

                if (sigma_down == 0) {
                    x = denoised;
                } else {
                    float t              = t_fn(sigmas[i]);
                    float t_next         = t_fn(sigma_down);
                    float h              = t_next - t;
                    float s              = t + 0.5f * h;
                    sd::Tensor<float> x2 = (sigma_fn(s) / sigma_fn(t)) * x - (exp(-h * 0.5f) - 1) * denoised;
                    auto denoised2_opt   = model(x2, sigmas[i + 1], i + 1);
                    if (denoised2_opt.empty()) {
                        return {};
                    }
                    sd::Tensor<float> denoised2 = std::move(denoised2_opt);
                    x                           = (sigma_fn(t_next) / sigma_fn(t)) * (x) - (exp(-h) - 1) * denoised2;
                }

                if (sigmas[i + 1] > 0) {
                    x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
                }
            }
            return x;
        }
        case DPMPP2M_SAMPLE_METHOD: {
            sd::Tensor<float> old_denoised = x;
            auto t_fn                      = [](float sigma) -> float { return -log(sigma); };
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                float t                    = t_fn(sigmas[i]);
                float t_next               = t_fn(sigmas[i + 1]);
                float h                    = t_next - t;
                float a                    = sigmas[i + 1] / sigmas[i];
                float b                    = exp(-h) - 1.f;

                if (i == 0 || sigmas[i + 1] == 0) {
                    x = a * (x)-b * denoised;
                } else {
                    float h_last                 = t - t_fn(sigmas[i - 1]);
                    float r                      = h_last / h;
                    sd::Tensor<float> denoised_d = (1.f + 1.f / (2.f * r)) * denoised - (1.f / (2.f * r)) * old_denoised;
                    x                            = a * (x)-b * denoised_d;
                }
                old_denoised = denoised;
            }
            return x;
        }
        case DPMPP2Mv2_SAMPLE_METHOD: {
            sd::Tensor<float> old_denoised = x;
            auto t_fn                      = [](float sigma) -> float { return -log(sigma); };
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);
                float t                    = t_fn(sigmas[i]);
                float t_next               = t_fn(sigmas[i + 1]);
                float h                    = t_next - t;
                float a                    = sigmas[i + 1] / sigmas[i];
                if (i == 0 || sigmas[i + 1] == 0) {
                    float b = exp(-h) - 1.f;
                    x       = a * (x)-b * denoised;
                } else {
                    float h_last                 = t - t_fn(sigmas[i - 1]);
                    float h_min                  = std::min(h_last, h);
                    float h_max                  = std::max(h_last, h);
                    float r                      = h_max / h_min;
                    float h_d                    = (h_max + h_min) / 2.f;
                    float b                      = exp(-h_d) - 1.f;
                    sd::Tensor<float> denoised_d = (1.f + 1.f / (2.f * r)) * denoised - (1.f / (2.f * r)) * old_denoised;
                    x                            = a * (x)-b * denoised_d;
                }
                old_denoised = denoised;
            }
            return x;
        }
        case LCM_SAMPLE_METHOD: {
            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);

                x = denoised;
                if (sigmas[i + 1] > 0) {
                    x += sd::Tensor<float>::randn_like(x, rng) * sigmas[i + 1];
                }
            }
            return x;
        }
        case IPNDM_SAMPLE_METHOD: {
            int max_order                       = 4;
            std::vector<sd::Tensor<float>> hist = {};
            for (int i = 0; i < steps; i++) {
                float sigma      = sigmas[i];
                float sigma_next = sigmas[i + 1];

                auto denoised_opt = model(x, sigma, i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);

                sd::Tensor<float> d_cur = (x - denoised) / sigma;
                int order               = std::min(max_order, i + 1);
                float dt                = sigma_next - sigma;

                switch (order) {
                    case 1:
                        x += d_cur * dt;
                        break;
                    case 2:
                        x += ((3.f * d_cur - hist.back()) / 2.f) * dt;
                        break;
                    case 3:
                        x += ((23.f * d_cur - 16.f * hist[hist.size() - 1] + 5.f * hist[hist.size() - 2]) / 12.f) * dt;
                        break;
                    case 4:
                        x += ((55.f * d_cur - 59.f * hist[hist.size() - 1] + 37.f * hist[hist.size() - 2] - 9.f * hist[hist.size() - 3]) / 24.f) * dt;
                        break;
                }

                if (hist.size() == static_cast<size_t>(max_order - 1)) {
                    hist.erase(hist.begin());
                }
                hist.push_back(std::move(d_cur));
            }
            return x;
        }
        case IPNDM_V_SAMPLE_METHOD: {
            int max_order                       = 4;
            std::vector<sd::Tensor<float>> hist = {};
            for (int i = 0; i < steps; i++) {
                float sigma  = sigmas[i];
                float t_next = sigmas[i + 1];

                auto denoised_opt = model(x, sigma, i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);

                sd::Tensor<float> d_cur = (x - denoised) / sigma;
                int order               = std::min(max_order, i + 1);
                float h_n               = t_next - sigma;
                float h_n_1             = (i > 0) ? (sigma - sigmas[i - 1]) : h_n;

                switch (order) {
                    case 1:
                        x += d_cur * h_n;
                        break;
                    case 2:
                        x += (((2.f + (h_n / h_n_1)) * d_cur - (h_n / h_n_1) * hist.back()) / 2.f) * h_n;
                        break;
                    case 3:
                        x += ((23.f * d_cur - 16.f * hist[hist.size() - 1] + 5.f * hist[hist.size() - 2]) / 12.f) * h_n;
                        break;
                    case 4:
                        x += ((55.f * d_cur - 59.f * hist[hist.size() - 1] + 37.f * hist[hist.size() - 2] - 9.f * hist[hist.size() - 3]) / 24.f) * h_n;
                        break;
                }

                if (hist.size() == static_cast<size_t>(max_order - 1)) {
                    hist.erase(hist.begin());
                }
                hist.push_back(std::move(d_cur));
            }
            return x;
        }
        case RES_MULTISTEP_SAMPLE_METHOD: {
            sd::Tensor<float> old_denoised = x;

            bool have_old_sigma  = false;
            float old_sigma_down = 0.0f;

            auto t_fn     = [](float sigma) -> float { return -logf(sigma); };
            auto sigma_fn = [](float t) -> float { return expf(-t); };
            auto phi1_fn  = [](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 1.0f + t * 0.5f + (t * t) / 6.0f;
                }
                return (expf(t) - 1.0f) / t;
            };
            auto phi2_fn = [&](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 0.5f + t / 6.0f + (t * t) / 24.0f;
                }
                float phi1_val = phi1_fn(t);
                return (phi1_val - 1.0f) / t;
            };

            for (int i = 0; i < steps; i++) {
                auto denoised_opt = model(x, sigmas[i], i + 1);
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);

                float sigma_from = sigmas[i];
                float sigma_to   = sigmas[i + 1];
                float sigma_up   = 0.0f;
                float sigma_down = sigma_to;

                if (eta > 0.0f) {
                    float sigma_from_sq = sigma_from * sigma_from;
                    float sigma_to_sq   = sigma_to * sigma_to;
                    if (sigma_from_sq > 0.0f) {
                        float term = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq;
                        if (term > 0.0f) {
                            sigma_up = eta * std::sqrt(term);
                        }
                    }
                    sigma_up            = std::min(sigma_up, sigma_to);
                    float sigma_down_sq = sigma_to_sq - sigma_up * sigma_up;
                    sigma_down          = sigma_down_sq > 0.0f ? std::sqrt(sigma_down_sq) : 0.0f;
                }

                if (sigma_down == 0.0f || !have_old_sigma) {
                    x += ((x - denoised) / sigma_from) * (sigma_down - sigma_from);
                } else {
                    float t      = t_fn(sigma_from);
                    float t_old  = t_fn(old_sigma_down);
                    float t_next = t_fn(sigma_down);
                    float t_prev = t_fn(sigmas[i - 1]);
                    float h      = t_next - t;
                    float c2     = (t_prev - t_old) / h;

                    float phi1_val = phi1_fn(-h);
                    float phi2_val = phi2_fn(-h);
                    float b1       = phi1_val - phi2_val / c2;
                    float b2       = phi2_val / c2;

                    if (!std::isfinite(b1)) {
                        b1 = 0.0f;
                    }
                    if (!std::isfinite(b2)) {
                        b2 = 0.0f;
                    }

                    x = sigma_fn(h) * (x) + h * (b1 * denoised + b2 * old_denoised);
                }

                if (sigmas[i + 1] > 0 && sigma_up > 0.0f) {
                    x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
                }

                old_denoised   = denoised;
                old_sigma_down = sigma_down;
                have_old_sigma = true;
            }
            return x;
        }
        case RES_2S_SAMPLE_METHOD: {
            const float c2 = 0.5f;
            auto t_fn      = [](float sigma) -> float { return -logf(sigma); };
            auto phi1_fn   = [](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 1.0f + t * 0.5f + (t * t) / 6.0f;
                }
                return (expf(t) - 1.0f) / t;
            };
            auto phi2_fn = [&](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 0.5f + t / 6.0f + (t * t) / 24.0f;
                }
                float phi1_val = phi1_fn(t);
                return (phi1_val - 1.0f) / t;
            };

            for (int i = 0; i < steps; i++) {
                float sigma_from = sigmas[i];
                float sigma_to   = sigmas[i + 1];

                auto denoised_opt = model(x, sigma_from, -(i + 1));
                if (denoised_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> denoised = std::move(denoised_opt);

                float sigma_up   = 0.0f;
                float sigma_down = sigma_to;
                if (eta > 0.0f) {
                    float sigma_from_sq = sigma_from * sigma_from;
                    float sigma_to_sq   = sigma_to * sigma_to;
                    if (sigma_from_sq > 0.0f) {
                        float term = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq;
                        if (term > 0.0f) {
                            sigma_up = eta * std::sqrt(term);
                        }
                    }
                    sigma_up            = std::min(sigma_up, sigma_to);
                    float sigma_down_sq = sigma_to_sq - sigma_up * sigma_up;
                    sigma_down          = sigma_down_sq > 0.0f ? std::sqrt(sigma_down_sq) : 0.0f;
                }

                sd::Tensor<float> x0 = x;
                if (sigma_down == 0.0f || sigma_from == 0.0f) {
                    x = denoised;
                } else {
                    float t      = t_fn(sigma_from);
                    float t_next = t_fn(sigma_down);
                    float h      = t_next - t;

                    float a21      = c2 * phi1_fn(-h * c2);
                    float phi1_val = phi1_fn(-h);
                    float phi2_val = phi2_fn(-h);
                    float b2       = phi2_val / c2;
                    float b1       = phi1_val - b2;

                    float sigma_c2         = expf(-(t + h * c2));
                    sd::Tensor<float> eps1 = denoised - x0;
                    sd::Tensor<float> x2   = x0 + eps1 * (h * a21);

                    auto denoised2_opt = model(x2, sigma_c2, i + 1);
                    if (denoised2_opt.empty()) {
                        return {};
                    }
                    sd::Tensor<float> denoised2 = std::move(denoised2_opt);
                    sd::Tensor<float> eps2      = denoised2 - x0;
                    x                           = x0 + h * (b1 * eps1 + b2 * eps2);
                }

                if (sigmas[i + 1] > 0 && sigma_up > 0.0f) {
                    x += sd::Tensor<float>::randn_like(x, rng) * sigma_up;
                }
            }
            return x;
        }
        case DDIM_TRAILING_SAMPLE_METHOD: {
            float beta_start = 0.00085f;
            float beta_end   = 0.0120f;
            std::vector<double> alphas_cumprod(TIMESTEPS);
            std::vector<double> compvis_sigmas(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                                  (sqrtf(beta_end) - sqrtf(beta_start)) *
                                      ((float)i / (TIMESTEPS - 1)),
                              2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            }

            for (int i = 0; i < steps; i++) {
                int timestep      = static_cast<int>(roundf(TIMESTEPS - i * ((float)TIMESTEPS / steps))) - 1;
                int prev_timestep = timestep - TIMESTEPS / static_cast<int>(steps);
                float sigma       = static_cast<float>(compvis_sigmas[timestep]);
                if (i == 0) {
                    x *= std::sqrt(sigma * sigma + 1) / sigma;
                } else {
                    x *= std::sqrt(sigma * sigma + 1);
                }

                auto model_output_opt = model(x, sigma, i + 1);
                if (model_output_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> model_output = std::move(model_output_opt);
                model_output                   = (x - model_output) * (1.0f / sigma);

                float alpha_prod_t      = static_cast<float>(alphas_cumprod[timestep]);
                float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
                float beta_prod_t       = 1.0f - alpha_prod_t;

                sd::Tensor<float> pred_original_sample = ((x / std::sqrt(sigma * sigma + 1)) -
                                                          std::sqrt(beta_prod_t) * model_output) *
                                                         (1.0f / std::sqrt(alpha_prod_t));

                float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;
                float variance         = (beta_prod_t_prev / beta_prod_t) *
                                 (1.0f - alpha_prod_t / alpha_prod_t_prev);
                float std_dev_t = eta * std::sqrt(variance);

                x = std::sqrt(alpha_prod_t_prev) * pred_original_sample +
                    std::sqrt(1.0f - alpha_prod_t_prev - std::pow(std_dev_t, 2)) * model_output;

                if (eta > 0) {
                    x += std_dev_t * sd::Tensor<float>::randn_like(x, rng);
                }
            }
            return x;
        }
        case TCD_SAMPLE_METHOD: {
            float beta_start = 0.00085f;
            float beta_end   = 0.0120f;
            std::vector<double> alphas_cumprod(TIMESTEPS);
            std::vector<double> compvis_sigmas(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                                  (sqrtf(beta_end) - sqrtf(beta_start)) *
                                      ((float)i / (TIMESTEPS - 1)),
                              2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            }
            int original_steps = 50;
            for (int i = 0; i < steps; i++) {
                int timestep      = TIMESTEPS - 1 - (TIMESTEPS / original_steps) * (int)floor(i * ((float)original_steps / steps));
                int prev_timestep = i >= steps - 1 ? 0 : TIMESTEPS - 1 - (TIMESTEPS / original_steps) * (int)floor((i + 1) * ((float)original_steps / steps));
                int timestep_s    = (int)floor((1 - eta) * prev_timestep);
                float sigma       = static_cast<float>(compvis_sigmas[timestep]);

                if (i == 0) {
                    x *= std::sqrt(sigma * sigma + 1) / sigma;
                } else {
                    x *= std::sqrt(sigma * sigma + 1);
                }

                auto model_output_opt = model(x, sigma, i + 1);
                if (model_output_opt.empty()) {
                    return {};
                }
                sd::Tensor<float> model_output = std::move(model_output_opt);
                model_output                   = (x - model_output) * (1.0f / sigma);

                float alpha_prod_t      = static_cast<float>(alphas_cumprod[timestep]);
                float beta_prod_t       = 1.0f - alpha_prod_t;
                float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
                float alpha_prod_s      = static_cast<float>(alphas_cumprod[timestep_s]);
                float beta_prod_s       = 1.0f - alpha_prod_s;

                sd::Tensor<float> pred_original_sample = ((x / std::sqrt(sigma * sigma + 1)) -
                                                          std::sqrt(beta_prod_t) * model_output) *
                                                         (1.0f / std::sqrt(alpha_prod_t));

                x = std::sqrt(alpha_prod_s) * pred_original_sample +
                    std::sqrt(beta_prod_s) * model_output;

                if (eta > 0 && i != steps - 1) {
                    x = std::sqrt(alpha_prod_t_prev / alpha_prod_s) * (x) +
                        std::sqrt(1.0f - alpha_prod_t_prev / alpha_prod_s) * sd::Tensor<float>::randn_like(x, rng);
                }
            }
            return x;
        }
        default:
            return {};
    }
}

#endif  // __DENOISER_HPP__

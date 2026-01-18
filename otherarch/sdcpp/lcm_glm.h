/*================================================= LCMSchedule =================================================*/

struct LCMSchedule : SigmaSchedule {
    // The number of steps in the original, reference schedule that LCM was trained on.
    // This is a default value from the Diffusers implementation.
    const uint32_t original_inference_steps = 50;
    // The 'rho' parameter for the Karras schedule used to generate the reference sigmas.
    const float rho = 7.0f;

    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t /* t_to_sigma */) override {
        // Note: The t_to_sigma function is not used here, as LCM's schedule is defined
        // directly in sigma-space, not based on a model's timestep-to-sigma conversion.

        if (n == 0) {
            // Return an empty vector if no steps are requested.
            return {};
        }

        // 1. Generate the "original" Karras schedule.
        // This is a full reference schedule of `original_inference_steps` that LCM was
        // trained to condense. The sigmas are ordered from high to low.
        std::vector<float> original_sigmas(original_inference_steps);
        float min_inv_rho = std::pow(sigma_min, (1.0f / rho));
        float max_inv_rho = std::pow(sigma_max, (1.0f / rho));
        for (uint32_t i = 0; i < original_inference_steps; i++) {
            // Formula for Karras schedule: sigma = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
            // This correctly produces a list of sigmas from sigma_max (at i=0) to sigma_min (at i=N-1).
            original_sigmas[i] = std::pow(max_inv_rho + (static_cast<float>(i) / (original_inference_steps - 1.0f)) * (min_inv_rho - max_inv_rho), rho);
        }

        std::vector<float> result;
        result.reserve(n + 1);

        // 2. Select `n` evenly spaced points from the original schedule.
        // We sample indices from 0 to `original_inference_steps - 1` and pick the
        // corresponding sigmas. This ensures we start with sigma_max and end with sigma_min.
        if (n == 1) {
            // Special case for a single step: just take the start and end.
            result.push_back(original_sigmas.front()); // sigma_max
            result.push_back(0.0f);
            return result;
        }

        float step_size = static_cast<float>(original_inference_steps - 1) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            // Calculate the index into the original_sigmas array.
            // We use round for a more even distribution of indices.
            int idx = static_cast<int>(std::round(step_size * static_cast<float>(i)));
            
            // Clamp index to be safe against floating point precision issues.
            idx = std::max(0, std::min(static_cast<int>(original_inference_steps - 1), idx));
            
            result.push_back(original_sigmas[idx]);
        }

        // 3. Append the final zero sigma, representing a fully denoised latent.
        result.push_back(0.0f);

        return result;
    }
};

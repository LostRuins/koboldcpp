/* LCM (Latent Consistency Model) Schedule
 * Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py
 * 
 * LCM uses a simple linear spacing of timesteps from training timesteps down to 0,
 * matching the original training distribution for few-step inference.
 */
struct LCMSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) override {
        std::vector<float> result;
        
        if (n == 0) {
            result.push_back(0.0f);
            return result;
        }
        
        result.reserve(n + 1);
        
        // Handle n == 1 as a special case to avoid division by zero in linear_space
        if (n == 1) {
            result.push_back(t_to_sigma(static_cast<float>(TIMESTEPS - 1)));
            result.push_back(0.0f);
            return result;
        }
        
        // LCM uses linearly spaced timesteps from TIMESTEPS-1 down to 0
        // This matches the Python implementation's behavior:
        // timesteps = np.linspace(0, train_timesteps - 1, num_inference_steps)[::-1]
        std::vector<float> timesteps = linear_space(static_cast<float>(TIMESTEPS - 1), 0.0f, n);
        
        for (uint32_t i = 0; i < n; ++i) {
            result.push_back(t_to_sigma(timesteps[i]));
        }
        
        // Append the final sigma of 0 as required by the sampling loop
        result.push_back(0.0f);
        return result;
    }
};

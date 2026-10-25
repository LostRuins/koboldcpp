#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct MingPatchNetworkConfig {
    int64_t hidden_size = 0;
    int64_t heads = 0;
    int64_t depth = 0;
    int64_t intermediate_size = 0;
};

struct MingFlowConfig {
    MingPatchNetworkConfig network;
    MingPatchNetworkConfig aggregator;
    int64_t latent_dim = 0;
    int64_t conditioning_dim = 0;
    std::string network_prefix = "flowloss.cfm.model";
    std::string aggregator_prefix = "linear_proj_audio";
};

class MingFlowRuntime {
public:
    MingFlowRuntime(std::shared_ptr<const assets::TensorSource> source,
        const MingFlowConfig & config, core::ExecutionContext & execution);
    ~MingFlowRuntime();

    std::vector<float> predict_velocity(const std::vector<float> & latent, const std::vector<float> & conditioning,
        const std::vector<float> & history, float time);

    std::vector<float> sample(const std::vector<float> & conditioning, const std::vector<float> & history,
        const std::vector<float> & noise, const std::vector<float> & sde_noise,
        float cfg = 2.0f, float sigma = 0.25f, float temperature = 0.0f);
    std::vector<float> aggregate(const std::vector<float> & latent);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules

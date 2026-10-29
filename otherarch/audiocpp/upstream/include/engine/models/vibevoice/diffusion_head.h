#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/vibevoice/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models::vibevoice {

class VibeVoiceDiffusionHeadGraph;

struct VibeVoiceDiffusionPrediction {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t latent_size = 0;
};

struct VibeVoiceDiffusionHeadLayerWeights {
    modules::LinearWeights gate_proj;
    modules::LinearWeights up_proj;
    modules::LinearWeights down_proj;
    assets::TensorDataF32 norm;
    modules::LinearWeights ada_ln;
};

struct VibeVoiceDiffusionHeadFinalWeights {
    modules::LinearWeights ada_ln;
    modules::LinearWeights linear;
};

struct VibeVoiceDiffusionHeadWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights noisy_images_proj;
    modules::LinearWeights cond_proj;
    modules::LinearWeights timestep_fc1;
    modules::LinearWeights timestep_fc2;
    std::vector<VibeVoiceDiffusionHeadLayerWeights> layers;
    VibeVoiceDiffusionHeadFinalWeights final_layer;
    core::TensorValue time_freqs;
    core::TensorValue ones;
};

class VibeVoiceDiffusionHeadWeightsRuntime final {
public:
    VibeVoiceDiffusionHeadWeightsRuntime(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 64ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 64ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);

    ~VibeVoiceDiffusionHeadWeightsRuntime();

    VibeVoiceDiffusionHeadWeightsRuntime(const VibeVoiceDiffusionHeadWeightsRuntime &) = delete;
    VibeVoiceDiffusionHeadWeightsRuntime & operator=(const VibeVoiceDiffusionHeadWeightsRuntime &) = delete;

    const VibeVoiceAssets & assets() const noexcept;
    const VibeVoiceDiffusionHeadWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    core::ConstantTensorCache & constants() const noexcept;
    int threads() const noexcept;

    VibeVoiceDiffusionPrediction predict(
        const std::vector<float> & noisy_images,
        int64_t frames,
        int64_t latent_size,
        float timestep,
        const std::vector<float> & condition,
        int64_t condition_hidden_size) const;

private:
    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::shared_ptr<const VibeVoiceDiffusionHeadWeights> weights_;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<VibeVoiceDiffusionHeadGraph> graph_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
};

VibeVoiceDiffusionHeadWeights load_vibevoice_diffusion_head_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

core::TensorValue build_vibevoice_diffusion_head(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & noisy_images,
    const core::TensorValue & timesteps,
    const core::TensorValue & condition,
    const VibeVoiceDiffusionHeadWeights & weights,
    const VibeVoiceDiffusionHeadConfig & config,
    core::ConstantTensorCache & constants);

}  // namespace engine::models::vibevoice

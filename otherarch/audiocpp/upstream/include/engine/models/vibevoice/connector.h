#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/models/vibevoice/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models::vibevoice {

class VibeVoiceConnectorGraph;
struct VibeVoiceTokenizerLatents;

struct VibeVoiceConnectorOutput {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t hidden_size = 0;
};

struct VibeVoiceConnectorWeights {
    modules::LinearWeights fc1;
    assets::TensorDataF32 norm;
    modules::LinearWeights fc2;
    int64_t input_dim = 0;
    int64_t hidden_size = 0;
};

struct VibeVoiceConnectorWeightsBundle {
    std::shared_ptr<core::BackendWeightStore> store;
    VibeVoiceConnectorWeights acoustic;
    VibeVoiceConnectorWeights semantic;
};

class VibeVoiceConnectorWeightsRuntime final {
public:
    VibeVoiceConnectorWeightsRuntime(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 64ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 32ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);

    ~VibeVoiceConnectorWeightsRuntime();

    VibeVoiceConnectorWeightsRuntime(const VibeVoiceConnectorWeightsRuntime &) = delete;
    VibeVoiceConnectorWeightsRuntime & operator=(const VibeVoiceConnectorWeightsRuntime &) = delete;

    const VibeVoiceAssets & assets() const noexcept;
    const VibeVoiceConnectorWeightsBundle & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int threads() const noexcept;

    VibeVoiceConnectorOutput project_acoustic(
        const std::vector<float> & features,
        int64_t frames,
        int64_t input_dim) const;
    std::vector<VibeVoiceConnectorOutput> project_acoustic_batch(
        const std::vector<VibeVoiceTokenizerLatents> & features) const;
    VibeVoiceConnectorOutput project_semantic(
        const std::vector<float> & features,
        int64_t frames,
        int64_t input_dim) const;
    std::vector<VibeVoiceConnectorOutput> project_semantic_batch(
        const std::vector<VibeVoiceTokenizerLatents> & features) const;

private:
    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::shared_ptr<const VibeVoiceConnectorWeightsBundle> weights_;
    std::unique_ptr<core::ConstantTensorCache> acoustic_constants_;
    std::unique_ptr<core::ConstantTensorCache> semantic_constants_;
    mutable std::unique_ptr<VibeVoiceConnectorGraph> acoustic_graph_;
    mutable std::unique_ptr<VibeVoiceConnectorGraph> semantic_graph_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
};

VibeVoiceConnectorWeightsBundle load_vibevoice_connector_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

core::TensorValue build_vibevoice_connector(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const VibeVoiceConnectorWeights & weights,
    core::ConstantTensorCache & constants);

}  // namespace engine::models::vibevoice

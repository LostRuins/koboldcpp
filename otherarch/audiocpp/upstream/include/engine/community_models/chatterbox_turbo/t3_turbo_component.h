#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/chatterbox_turbo/t3_turbo_types.h"

#include <memory>

namespace engine::community_models::chatterbox_turbo {

std::shared_ptr<const T3TurboInferenceWeights> load_t3_turbo_inference_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType graph_weight_storage_type = engine::assets::TensorStorageType::Native,
    bool load_reference_f32_graph_weights = true);

class T3TurboInferenceComponent {
public:
    explicit T3TurboInferenceComponent(
        std::shared_ptr<const T3TurboInferenceWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    T3TurboGenerateOutputs generate_speech_tokens(const T3TurboGenerateRequest & request) const;
    void release_runtime_graphs() const;
    void release_runtime_cache() const;

private:
    struct State;

    std::shared_ptr<const T3TurboInferenceWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

}  // namespace engine::community_models::chatterbox_turbo

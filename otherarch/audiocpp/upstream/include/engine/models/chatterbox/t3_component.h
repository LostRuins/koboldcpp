#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/chatterbox/t3_types.h"

#include <memory>

namespace engine::models::chatterbox {

std::shared_ptr<const T3InferenceWeights> load_t3_inference_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType graph_weight_storage_type = engine::assets::TensorStorageType::Native,
    bool load_reference_f32_graph_weights = true);

class T3InferenceComponent {
public:
    explicit T3InferenceComponent(
        std::shared_ptr<const T3InferenceWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    T3GenerateOutputs generate_speech_tokens(const T3GenerateRequest & request) const;
    void release_runtime_graphs() const;
    void release_runtime_cache() const;

private:
    struct ConditioningInput;
    struct PreparedInput;
    struct State;

    PreparedInput prepare_conditioning_inputs(const ConditioningInput & input) const;

    std::shared_ptr<const T3InferenceWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::chatterbox

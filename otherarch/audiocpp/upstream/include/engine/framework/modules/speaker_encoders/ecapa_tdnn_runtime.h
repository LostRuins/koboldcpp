#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/speaker_encoders/ecapa_tdnn_speaker.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules::ecapa_tdnn {

struct EcapaBackendWeights;

struct EcapaClassifyResult {
    std::string label;
    int index = -1;
    float score = 0.0f;
    std::vector<float> embedding;
    std::vector<float> logits;
};

class EcapaRuntime {
public:
    EcapaRuntime(
        std::shared_ptr<const EcapaWeights> weights,
        std::vector<std::string> labels,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~EcapaRuntime();

    std::vector<float> embed_features(const std::vector<float> & features, int64_t frames);
    std::vector<float> embed_audio(const runtime::AudioBuffer & audio);
    EcapaClassifyResult classify_audio(const runtime::AudioBuffer & audio);

private:
    class Graph;
    Graph & ensure_graph(int64_t frames);

    std::shared_ptr<const EcapaWeights> weights_;
    std::shared_ptr<const EcapaBackendWeights> backend_weights_;
    std::vector<std::string> labels_;
    std::vector<float> normalized_classifier_weight_;
    std::vector<float> centered_embedding_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::modules::ecapa_tdnn

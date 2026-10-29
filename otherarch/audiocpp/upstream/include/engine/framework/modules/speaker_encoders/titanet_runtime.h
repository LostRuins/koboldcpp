#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/speaker_encoders/titanet_speaker.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules::titanet {

struct TitaNetBackendWeights;

struct TitaNetClassifyResult {
    std::string label;
    int index = -1;
    float score = 0.0f;
    std::vector<float> embedding;
    std::vector<float> logits;
};

class TitaNetRuntime {
public:
    TitaNetRuntime(
        std::shared_ptr<const TitaNetWeights> weights,
        std::vector<std::string> labels,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~TitaNetRuntime();

    std::vector<float> embed_features(const std::vector<float> & features, int64_t frames);
    std::vector<float> embed_audio(const runtime::AudioBuffer & audio);
    TitaNetClassifyResult classify_audio(const runtime::AudioBuffer & audio);

private:
    class Graph;
    Graph & ensure_graph(int64_t frames);

    std::shared_ptr<const TitaNetWeights> weights_;
    std::shared_ptr<const TitaNetBackendWeights> backend_weights_;
    std::vector<std::string> labels_;
    std::vector<float> normalized_classifier_weight_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::modules::titanet

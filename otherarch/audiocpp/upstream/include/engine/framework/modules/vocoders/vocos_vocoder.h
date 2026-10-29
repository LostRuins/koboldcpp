#pragma once
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/convnext_modules.h"
#include <memory>
#include <string>
#include <vector>

namespace engine::modules {
struct VocosBackboneWeights {
    Conv1dWeights embed;
    NormWeights input_norm, final_norm;
    LinearWeights head;
    std::vector<ConvNeXt1dWeights> blocks;
};
// Shared F5/ZipVoice backbone, logical [batch, frames, mel] -> spectral rows.
core::TensorValue build_vocos_backbone(core::ModuleBuildContext & ctx,
    const core::TensorValue & mel, const VocosBackboneWeights & weights);

// Generalized F5/Vocos mel-24khz graph: Conv1d, ConvNeXt, LN, spectral head.
// Owns backend weights and one runtime graph/ISTFT workspace. The backend is
// borrowed and must outlive this object. Callers serialize decode calls.
class VocosVocoder {
public:
    VocosVocoder(const std::string & checkpoint, ggml_backend_t backend, int threads);
    ~VocosVocoder();
    VocosVocoder(const VocosVocoder &) = delete;
    VocosVocoder & operator=(const VocosVocoder &) = delete;
    std::vector<float> decode(const std::vector<float> & mel_rows);
    size_t cached_graph_count() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}

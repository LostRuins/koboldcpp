#pragma once

#include "engine/community_models/soprano_tts/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}
namespace engine::assets {
enum class TensorStorageType;
}
namespace engine::runtime {
struct AudioBuffer;
}

namespace engine::community_models::soprano_tts {

struct SopranoVocosWeights;
struct SopranoVocosGraph;

// Non-iterative Vocos-style decoder (SopranoDecoder): linear upsample x4 over
// the frame axis, a ConvNeXt backbone (embed Conv1d, 8 blocks, final LN) and a
// single ISTFT head (Linear(dim -> n_fft+2), exp(mag), cos/sin phase, 1 ISTFT).
class SopranoVocosRuntime final {
public:
    SopranoVocosRuntime(
        const SopranoTTSAssets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~SopranoVocosRuntime();

    // frames x hidden -> 32 kHz mono audio.
    runtime::AudioBuffer decode(const std::vector<float> & features, int64_t frames) const;



private:
    const SopranoTTSConfig & config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const SopranoVocosWeights> weights_;
    mutable std::unique_ptr<SopranoVocosGraph> graph_;
};

}  // namespace engine::community_models::soprano_tts

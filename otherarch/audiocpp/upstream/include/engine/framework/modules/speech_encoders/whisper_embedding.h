#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstdint>
#include <vector>

namespace engine::modules {

struct WhisperEmbeddingConfig {
    int64_t n_mels = 0;
    int64_t n_audio_ctx = 0;
    int64_t n_audio_state = 0;
    int64_t n_audio_head = 0;
    int64_t n_audio_layer = 0;
    float layer_norm_eps = 1.0e-5F;
};

struct WhisperAttentionWeights {
    LinearWeights query;
    LinearWeights key;
    LinearWeights value;
    LinearWeights out;
};

struct WhisperEncoderLayerWeights {
    NormWeights attention_norm;
    WhisperAttentionWeights attention;
    NormWeights mlp_norm;
    FeedForwardWeights mlp;
};

struct WhisperEmbeddingWeights {
    Conv1dWeights conv1;
    Conv1dWeights conv2;
    core::TensorValue positional_embedding;
    std::vector<WhisperEncoderLayerWeights> layers;
    NormWeights final_norm;
};

class WhisperEmbeddingModule {
public:
    explicit WhisperEmbeddingModule(WhisperEmbeddingConfig config);

    const WhisperEmbeddingConfig & config() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & log_mel,
        const WhisperEmbeddingWeights & weights) const;
    core::TensorValue build_pre_norm(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & log_mel,
        const WhisperEmbeddingWeights & weights) const;

private:
    WhisperEmbeddingConfig config_;
};

}  // namespace engine::modules

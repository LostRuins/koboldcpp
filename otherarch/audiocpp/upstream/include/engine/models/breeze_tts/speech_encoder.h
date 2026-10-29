#pragma once

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/speech_decoder.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models {

namespace breeze_tts {

struct BreezeSpeechEncoderWeights;
class BreezeSpeechEncoderConvGraph;
class BreezeSpeechEncoderTransformerGraph;

struct BreezeSpeechEncoderOutput {
    BreezeSpeechCodes codes;
    std::vector<float> semantic_projected;
    std::vector<float> acoustic_projected;
};

class BreezeSpeechEncoderRuntime {
public:
    BreezeSpeechEncoderRuntime(
        std::shared_ptr<const BreezeTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        core::AttentionPreference attention_preference = core::AttentionPreference::Auto);
    ~BreezeSpeechEncoderRuntime();

    BreezeSpeechCodes encode(const runtime::AudioBuffer & audio) const;
    void release_runtime_graphs() const;

private:
    std::shared_ptr<const BreezeTTSAssets> assets_;
    std::shared_ptr<const BreezeSpeechEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    bool allow_flash_attention_ = true;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<BreezeSpeechEncoderConvGraph> conv_graph_;
    mutable std::unique_ptr<BreezeSpeechEncoderTransformerGraph> transformer_graph_;
};

}  // namespace breeze_tts
}  // namespace engine::models

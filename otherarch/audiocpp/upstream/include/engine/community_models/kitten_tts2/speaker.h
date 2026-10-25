#pragma once
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include <memory>
#include <vector>

namespace engine::community_models::kitten_tts2 {
// XVectorSincNet identity encoder and Kitten's learned Qwen speaker projection.
class SpeakerEncoder {
public:
    SpeakerEncoder(const assets::TensorSource & source, const assets::TensorSource & lm,
        const core::ExecutionContext & execution);
    ~SpeakerEncoder();
    std::vector<float> embed(const std::vector<float> & mono_16k) const;
    std::vector<float> project(const std::vector<float> & embedding) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

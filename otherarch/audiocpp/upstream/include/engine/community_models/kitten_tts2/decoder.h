#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

namespace engine::community_models::kitten_tts2 {
class WaveformDecoder {
public:
    WaveformDecoder(std::shared_ptr<const assets::TensorSource> source,
        const core::ExecutionContext & execution);
    ~WaveformDecoder();
    std::vector<float> decode(const std::vector<int32_t> & codes,
        const std::vector<int64_t> & prompt_tokens, const std::vector<float> & prompt_mel,
        const std::vector<float> & embedding, uint64_t seed);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

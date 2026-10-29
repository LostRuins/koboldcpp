#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/cosyvoice3/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::cosyvoice3 {

struct CosyVoice3FlowRequest {
    std::vector<int32_t> speech_tokens;
    std::vector<int32_t> prompt_speech_tokens;
    std::vector<float> prompt_mel;
    int64_t prompt_mel_frames = 0;
    std::vector<float> speaker_embedding;
    uint32_t seed = 1986;
    int64_t num_inference_steps = 10;
};

struct CosyVoice3FlowOutput {
    std::vector<float> mel;
    int64_t frames = 0;
};

class CosyVoice3DiTFlowRuntime {
public:
    CosyVoice3DiTFlowRuntime(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        const std::vector<std::string> & rknn_engine_paths = {});
    ~CosyVoice3DiTFlowRuntime();

    CosyVoice3DiTFlowRuntime(const CosyVoice3DiTFlowRuntime &) = delete;
    CosyVoice3DiTFlowRuntime & operator=(const CosyVoice3DiTFlowRuntime &) = delete;

    CosyVoice3FlowOutput generate(const CosyVoice3FlowRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::cosyvoice3

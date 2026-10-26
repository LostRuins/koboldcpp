#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"
#include "engine/models/cosyvoice3/assets.h"

#include <memory>
#include <vector>

namespace engine::models::cosyvoice3 {

struct CosyVoice3ReferenceFeatures {
    std::vector<int32_t> speech_tokens;
    int64_t speech_token_count = 0;
    std::vector<float> prompt_mel;
    int64_t prompt_mel_frames = 0;
    std::vector<float> speaker_embedding;
};

class CosyVoice3Frontend {
public:
    CosyVoice3Frontend(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        size_t reference_cache_slots);
    ~CosyVoice3Frontend();

    CosyVoice3Frontend(const CosyVoice3Frontend &) = delete;
    CosyVoice3Frontend & operator=(const CosyVoice3Frontend &) = delete;

    const CosyVoice3ReferenceFeatures & prepare_reference(const engine::runtime::AudioBuffer & audio);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::cosyvoice3

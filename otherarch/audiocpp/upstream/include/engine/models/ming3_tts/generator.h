#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/ming3_tts/assets.h"
#include "engine/models/ming3_tts/prompt.h"

#include <memory>
#include <vector>

namespace engine::models::ming3_tts {

struct MingGenerationOptions {
    uint64_t seed = 1895;
    int64_t max_tokens = 1000;
    float cfg = 2.0f;
    float sigma = 0.25f;
    float temperature = 0.0f;
};

class MingGenerator {
public:
    MingGenerator(std::shared_ptr<const assets::TensorSource> source, const MingConfig & config,
        core::ExecutionContext & execution, core::AttentionPreference attention = core::AttentionPreference::Auto);
    ~MingGenerator();

    std::vector<float> prepare_inputs(const MingPrompt & prompt,
        const std::vector<std::vector<float>> & speakers = {}, const std::vector<float> & prompt_latents = {},
        bool zero_speaker = false);

    std::vector<float> generate(const std::vector<float> & embeddings, const MingGenerationOptions & options,
        uint64_t & random_offset, const std::vector<float> & initial_history = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_tts

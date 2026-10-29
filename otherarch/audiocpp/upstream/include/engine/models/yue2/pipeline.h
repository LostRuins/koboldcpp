#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/oobleck_audio_vae_runtime.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/request.h"
#include "engine/models/yue2/tokenizer_text.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::yue2 {

struct Yue2RunResult {
    std::optional<runtime::AudioBuffer> audio;
    std::string plan_abc_text;
    bool plan_abc_truncated = false;
    std::vector<int32_t> semantic_codes;
    bool semantic_truncated = false;
};

class Yue2PipelineRuntime {
public:
    Yue2PipelineRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType model_weight_type,
        assets::TensorStorageType vae_weight_type,
        size_t model_weight_context_bytes,
        size_t vae_weight_context_bytes,
        size_t ar_prefill_graph_arena_bytes,
        size_t ar_decode_graph_arena_bytes,
        size_t nar_graph_arena_bytes,
        size_t vae_graph_arena_bytes,
        core::AttentionPreference attention_preference = core::AttentionPreference::Auto,
        int64_t nar_attention_tile_rows = 0);
    ~Yue2PipelineRuntime();

    Yue2Plan plan(const Yue2Request & request);
    Yue2SemanticResult generate_semantic(const Yue2Request & request, Yue2Plan plan);
    std::vector<float> synthesize_latents(
        const Yue2SemanticResult & semantic,
        const Yue2GenerationConfig & generation,
        uint64_t seed);
    runtime::AudioBuffer decode_audio(const std::vector<float> & latents, int64_t frames);
    Yue2RunResult run(const Yue2Request & request);
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2

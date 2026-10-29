#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/breeze_tts/assets.h"
#include "engine/models/breeze_tts/tokenizer_text.h"

#include <memory>
#include <vector>

namespace engine::models::breeze_tts {

struct BreezeProjectedText {
    int64_t tokens = 0;
    std::vector<float> values;
};

class BreezeT5Gemma2TextEncoderRuntime {
public:
    BreezeT5Gemma2TextEncoderRuntime(
        std::shared_ptr<const BreezeTTSAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~BreezeT5Gemma2TextEncoderRuntime();

    BreezeProjectedText encode(const std::vector<int32_t> & input_ids);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::breeze_tts

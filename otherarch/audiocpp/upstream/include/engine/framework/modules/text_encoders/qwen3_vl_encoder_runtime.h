#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/decoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

enum class Qwen3VlEncoderOutputMode {
    Hidden,
    Logits,
};

struct Qwen3VlEncoderRuntimeConfig {
    std::string trace_name = "qwen3_vl_encoder";
    std::string model_prefix = "model.language_model";
    std::string final_norm_weight_name;
    std::string lm_head_weight_name = "lm_head.weight";
    std::string lm_head_bias_name = "lm_head.bias";
    DecoderStackConfig stack;
    int64_t vocab_size = 0;
    int64_t logits_size = 0;
    size_t weight_context_bytes = 0;
    size_t graph_arena_bytes = 0;
    size_t input_arena_bytes = 16ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    Qwen3VlEncoderOutputMode output_mode = Qwen3VlEncoderOutputMode::Hidden;
    bool return_hidden = false;
    bool use_lm_head_bias = false;
    ggml_prec lm_head_precision = GGML_PREC_DEFAULT;
    std::optional<ggml_type> lm_head_input_type;
    bool use_cuda_fast_projection = true;
    std::optional<ggml_type> readback_round_type;
};

struct Qwen3VlEncoderOptions {
    bool layerwise = false;
    int64_t layerwise_batch = 1;
};

struct Qwen3VlEncoderResult {
    std::vector<float> hidden;
    std::vector<float> logits;
    int64_t steps = 0;
    int64_t hidden_size = 0;
    int64_t logits_size = 0;
};

class Qwen3VlEncoderRuntime {
public:
    Qwen3VlEncoderRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> tensor_source,
        Qwen3VlEncoderRuntimeConfig config);
    ~Qwen3VlEncoderRuntime();

    Qwen3VlEncoderRuntime(const Qwen3VlEncoderRuntime &) = delete;
    Qwen3VlEncoderRuntime & operator=(const Qwen3VlEncoderRuntime &) = delete;

    Qwen3VlEncoderResult encode_text(
        const std::vector<int32_t> & ids,
        Qwen3VlEncoderOptions options = {});
    void release_runtime_graphs();
    void release_weights();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules

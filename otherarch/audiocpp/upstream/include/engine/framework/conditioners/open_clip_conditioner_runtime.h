#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::conditioners {

struct OpenClipTextConfig {
    int64_t vocab_size = 49408;
    int64_t context_length = 77;
    int64_t hidden_size = 1024;
    int64_t layers = 24;
    int64_t heads = 16;
    int64_t intermediate_size = 4096;
    float layer_norm_eps = 1.0e-5F;
    bool causal_text_attention = true;
    bool normalize_output = true;
    std::string tensor_prefix;
};

struct OpenClipImageConfig {
    int64_t image_channels = 3;
    int64_t image_size = 378;
    int64_t patch_size = 14;
    int64_t hidden_size = 1280;
    int64_t output_dim = 1024;
    int64_t layers = 32;
    int64_t heads = 16;
    int64_t intermediate_size = 5120;
    float layer_norm_eps = 1.0e-5F;
    bool normalize_output = true;
    std::string tensor_prefix;
};

struct OpenClipRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 3ull * 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    bool load_text = true;
    bool load_image = false;
};

struct OpenClipTextHidden {
    std::vector<float> values;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden = 0;
};

struct OpenClipImageEmbedding {
    std::vector<float> values;
    int64_t batch = 0;
    int64_t features = 0;
};

class OpenClipConditionerRuntime {
public:
    OpenClipConditionerRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        OpenClipTextConfig config = {},
        OpenClipRuntimeOptions options = {},
        OpenClipImageConfig image_config = {});
    ~OpenClipConditionerRuntime();

    OpenClipConditionerRuntime(const OpenClipConditionerRuntime &) = delete;
    OpenClipConditionerRuntime & operator=(const OpenClipConditionerRuntime &) = delete;
    OpenClipConditionerRuntime(OpenClipConditionerRuntime &&) noexcept;
    OpenClipConditionerRuntime & operator=(OpenClipConditionerRuntime &&) noexcept;

    OpenClipTextHidden encode_text(const std::vector<int32_t> & token_ids, int64_t batch, int64_t tokens);
    OpenClipImageEmbedding encode_image(const std::vector<float> & images, int64_t batch, int64_t height, int64_t width);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners

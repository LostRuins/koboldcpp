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

struct CavMaeStVisualConfig {
    int64_t image_channels = 3;
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t hidden_size = 768;
    int64_t visual_layers = 11;
    int64_t shared_layers = 1;
    int64_t heads = 12;
    int64_t intermediate_size = 3072;
    float layer_norm_eps = 1.0e-5F;
    std::string tensor_prefix = "module";
};

struct CavMaeStRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
};

struct CavMaeStVisualFeatures {
    std::vector<float> values;
    int64_t batch = 0;
    int64_t patches = 0;
    int64_t hidden = 0;
};

class CavMaeStConditionerRuntime {
public:
    CavMaeStConditionerRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        CavMaeStVisualConfig config = {},
        CavMaeStRuntimeOptions options = {});
    ~CavMaeStConditionerRuntime();

    CavMaeStConditionerRuntime(const CavMaeStConditionerRuntime &) = delete;
    CavMaeStConditionerRuntime & operator=(const CavMaeStConditionerRuntime &) = delete;
    CavMaeStConditionerRuntime(CavMaeStConditionerRuntime &&) noexcept;
    CavMaeStConditionerRuntime & operator=(CavMaeStConditionerRuntime &&) noexcept;

    CavMaeStVisualFeatures encode_visual(const std::vector<float> & images, int64_t batch, int64_t height, int64_t width);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners

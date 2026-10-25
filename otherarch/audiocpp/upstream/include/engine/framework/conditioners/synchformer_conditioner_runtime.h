#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <ggml.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::conditioners {

struct SynchformerConfig {
    int64_t image_channels = 3;
    int64_t image_size = 224;
    int64_t frames = 16;
    int64_t patch_size = 16;
    int64_t temporal_patch_size = 2;
    int64_t hidden_size = 768;
    int64_t layers = 12;
    int64_t heads = 12;
    int64_t intermediate_size = 3072;
    float layer_norm_eps = 1e-6F;
    std::string tensor_prefix = "vfeat_extractor";
};

struct SynchformerRuntimeOptions {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    size_t weight_context_bytes = 512ull * 1024ull * 1024ull;
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_F32;
};

struct SynchformerVideoFeatures {
    int64_t batch_segments = 0;
    int64_t temporal_tokens = 0;
    int64_t hidden = 0;
    std::vector<float> values;
};

class SynchformerConditionerRuntime {
public:
    SynchformerConditionerRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SynchformerConfig config = {},
        SynchformerRuntimeOptions options = {});
    ~SynchformerConditionerRuntime();

    SynchformerConditionerRuntime(const SynchformerConditionerRuntime &) = delete;
    SynchformerConditionerRuntime & operator=(const SynchformerConditionerRuntime &) = delete;

    SynchformerVideoFeatures encode_segments(
        const std::vector<float> & video,
        int64_t batch_segments,
        int64_t frames,
        int64_t height,
        int64_t width);

    SynchformerVideoFeatures encode_frames(
        const std::vector<float> & video,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t stride_frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners

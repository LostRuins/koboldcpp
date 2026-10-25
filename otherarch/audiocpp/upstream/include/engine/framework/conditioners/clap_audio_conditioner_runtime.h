#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::conditioners {

struct ClapAudioConfig {
    int64_t sample_rate = 48000;
    int64_t max_samples = 480000;
    int64_t n_fft = 1024;
    int64_t hop_length = 480;
    int64_t win_length = 1024;
    int64_t mel_bins = 64;
    int64_t spec_size = 256;
    int64_t patch_size = 4;
    int64_t patch_stride = 4;
    int64_t embed_dim = 128;
    int64_t output_dim = 512;
    int64_t class_count = 527;
    float layer_norm_eps = 1.0e-5F;
    float batch_norm_eps = 1.0e-5F;
    std::string tensor_prefix;
};

struct ClapAudioRuntimeOptions {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    size_t weight_context_bytes = 768ull * 1024ull * 1024ull;
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_F32;
};

struct ClapAudioEmbedding {
    int64_t batch = 0;
    int64_t features = 0;
    std::vector<float> values;
};

class ClapAudioConditionerRuntime {
public:
    ClapAudioConditionerRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        ClapAudioConfig config = {},
        ClapAudioRuntimeOptions options = {});
    ~ClapAudioConditionerRuntime();

    ClapAudioConditionerRuntime(const ClapAudioConditionerRuntime &) = delete;
    ClapAudioConditionerRuntime & operator=(const ClapAudioConditionerRuntime &) = delete;

    ClapAudioEmbedding encode_audio(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        size_t threads = 0);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::conditioners

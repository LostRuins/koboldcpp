#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs {

struct FishDacCodes {
    std::vector<int32_t> codes;
    int64_t codebooks = 0;
    int64_t frames = 0;
};

struct FishDacLatents {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

struct FishDacCodecConfig {
    int sample_rate = 44100;
    int64_t semantic_codebook_size = 4096;
    int64_t residual_codebook_size = 1024;
    int64_t quantizer_codebooks = 9;
    int64_t total_codebooks = 10;
    int64_t codebook_dim = 8;
    int64_t latent_dim = 1024;
    int64_t frame_length = 2048;
};

struct FishDacCodecWeightBinding {
    std::string encoder_prefix = "encoder";
    std::string quantizer_prefix = "quantizer";
    std::string decoder_prefix = "decoder";
};

struct FishDacCodecRuntimeOptions {
    size_t graph_arena_bytes = 512ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type = assets::TensorStorageType::Native;
    bool encoder_bf16_compute = false;
};

class FishDacCodecComponent {
public:
    static std::shared_ptr<const FishDacCodecComponent> load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        FishDacCodecConfig config,
        FishDacCodecWeightBinding binding,
        ggml_backend_t backend,
        core::BackendType backend_type,
        FishDacCodecRuntimeOptions options);

    ~FishDacCodecComponent();

    const FishDacCodecConfig & config() const noexcept;

private:
    struct Impl;
    explicit FishDacCodecComponent(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;

    friend class FishDacCodecRuntime;
};

class FishDacCodecRuntime {
public:
    FishDacCodecRuntime(
        std::shared_ptr<const FishDacCodecComponent> component,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        bool encoder_f32_accumulation = false,
        int64_t encoder_chunk_samples = 0);
    ~FishDacCodecRuntime();

    FishDacCodecRuntime(const FishDacCodecRuntime &) = delete;
    FishDacCodecRuntime & operator=(const FishDacCodecRuntime &) = delete;

    FishDacCodes encode_codes(const runtime::AudioBuffer & audio);
    FishDacLatents encode_latents(const runtime::AudioBuffer & audio);
    runtime::AudioBuffer decode_codes(const FishDacCodes & codes);
    // Decode one frame with persistent causal attention/convolution state.
    runtime::AudioBuffer decode_stream(const FishDacCodes & codes);
    void reset_decode_stream();
    runtime::AudioBuffer decode_latents(const FishDacLatents & latents);
    runtime::AudioBuffer decode_latents(const std::vector<float> & values, int64_t frames);
    void release_decode_graphs();
    void release_encode_graph();
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs

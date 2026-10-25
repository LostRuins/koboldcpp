#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::modules {

struct NemoNanoCodecConfig {
    int64_t sample_rate = 22050;
    int64_t input_dim = 32;
    int64_t base_channels = 864;
    int64_t audio_codebooks = 0;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<int64_t> resblock_dilation_sizes;
    std::vector<int32_t> fsq_num_levels;
    std::vector<int32_t> fsq_dim_base_index;
};

struct NemoNanoCodecRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 2048ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

class NemoNanoCodecRuntime {
public:
    NemoNanoCodecRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        NemoNanoCodecConfig config,
        NemoNanoCodecRuntimeOptions options);
    ~NemoNanoCodecRuntime();

    NemoNanoCodecRuntime(const NemoNanoCodecRuntime &) = delete;
    NemoNanoCodecRuntime & operator=(const NemoNanoCodecRuntime &) = delete;

    runtime::AudioBuffer decode_codes(const std::vector<int32_t> & codes);
    void release_runtime_graph();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules

#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs {

struct RedAeCodecConfig {
    int64_t sample_rate = 24000;
    int64_t audio_patch_size = 480;
    int64_t bottleneck_dim = 64;
    int64_t enc_hidden_size = 896;
    int64_t enc_intermediate_size = 3584;
    int64_t enc_layers = 18;
    int64_t enc_heads = 14;
    int64_t enc_kv_heads = 2;
    int64_t enc_head_dim = 64;
    int64_t enc_sliding_window = 0;
    int64_t enc_extra_downsample_rate = 2;
    int64_t enc_downsample_layers = 4;
    int64_t dec_hidden_size = 896;
    int64_t dec_intermediate_size = 3584;
    int64_t dec_layers = 18;
    int64_t dec_heads = 14;
    int64_t dec_kv_heads = 2;
    int64_t dec_head_dim = 64;
    int64_t dec_sliding_window = 0;
};

struct RedAeCodecSources {
    std::shared_ptr<const assets::TensorSource> encoder;
    std::shared_ptr<const assets::TensorSource> decoder;
};

struct RedAeCodecWeightBinding {
    std::string trace_prefix = "redae";
    std::string weight_store_name = "redae.weights";
    int64_t qwen_vocab_size = 151936;
    std::string enc_in0 = "encoder.in_proj.0";
    std::string enc_in1 = "encoder.in_proj.1";
    std::string encoder_qwen = "encoder.qwen3";
    std::string downsample_cls = "encoder.downsample.cls_tok";
    std::string downsample_qwen = "encoder.downsample.qwen3";
    std::string enc_out = "encoder.out_proj";
    std::string dec_in = "decoder.in_proj";
    std::string decoder_qwen = "decoder.qwen3";
    std::string istft_head = "decoder.istft_head.out";
    std::string istft_window = "decoder.istft_head.istft.window";
};

struct RedAeCodecRuntimeOptions {
    size_t graph_arena_bytes = 1024ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

class RedAeCodecRuntime {
public:
    RedAeCodecRuntime(
        RedAeCodecSources sources,
        core::ExecutionContext & execution,
        RedAeCodecConfig config,
        RedAeCodecWeightBinding binding,
        RedAeCodecRuntimeOptions options);
    ~RedAeCodecRuntime();

    RedAeCodecRuntime(const RedAeCodecRuntime &) = delete;
    RedAeCodecRuntime & operator=(const RedAeCodecRuntime &) = delete;

    std::vector<float> encode(const std::vector<float> & audio_24k);
    runtime::AudioBuffer decode(const std::vector<float> & latents);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs

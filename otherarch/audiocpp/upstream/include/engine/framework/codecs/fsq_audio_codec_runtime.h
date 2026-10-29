#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ggml-backend.h>

namespace engine::core {
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::codecs {

struct FsqAudioCodecConfig {
    int64_t sample_rate = 16000;
    int64_t output_sample_rate = 24000;
    int64_t hop_length = 480;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t quantization_dim = 0;
    int64_t prior_blocks = 2;
    int64_t post_blocks = 2;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::vector<int64_t> quantization_levels;
    std::string trace_name = "fsq_audio_codec";
};

struct FsqAudioCodecWeightBinding {
    std::string quantizer_project_out = "quantizer.project_out";
    std::string acoustic_fc = "acoustic_decoder.fc";
    std::string embed = "acoustic_decoder.embed";
    std::string prior_block_prefix = "acoustic_decoder.prior_net.";
    std::string transformer_layer_prefix = "acoustic_decoder.layers.";
    std::string post_block_prefix = "acoustic_decoder.post_net.";
    std::string final_norm = "acoustic_decoder.norm";
    std::string istft_head = "acoustic_decoder.head.linear";
};

struct FsqAudioCodecResnetBlockWeights {
    modules::NormWeights norm1;
    modules::Conv1dWeights conv1;
    modules::NormWeights norm2;
    modules::Conv1dWeights conv2;
};

struct FsqAudioCodecTransformerLayerWeights {
    modules::NormWeights attention_norm;
    std::optional<modules::LinearWeights> qkv_proj;
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
    modules::NormWeights ffn_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
};

struct FsqAudioCodecDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights quantizer_project_out;
    modules::LinearWeights acoustic_fc;
    modules::Conv1dWeights embed;
    std::vector<FsqAudioCodecResnetBlockWeights> prior_blocks;
    std::vector<FsqAudioCodecTransformerLayerWeights> transformer_layers;
    std::vector<FsqAudioCodecResnetBlockWeights> post_blocks;
    modules::NormWeights final_norm;
    modules::LinearWeights istft_head;
};

struct FsqAudioCodecHead {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t out_dim = 0;
};

class FsqAudioCodecDecoderRuntime {
public:
    FsqAudioCodecDecoderRuntime(
        FsqAudioCodecConfig config,
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_storage_type,
        assets::TensorStorageType conv_storage_type,
        FsqAudioCodecWeightBinding weight_binding = {});
    ~FsqAudioCodecDecoderRuntime();

    FsqAudioCodecDecoderRuntime(const FsqAudioCodecDecoderRuntime &) = delete;
    FsqAudioCodecDecoderRuntime & operator=(const FsqAudioCodecDecoderRuntime &) = delete;
    FsqAudioCodecDecoderRuntime(FsqAudioCodecDecoderRuntime &&) noexcept;
    FsqAudioCodecDecoderRuntime & operator=(FsqAudioCodecDecoderRuntime &&) noexcept;

    FsqAudioCodecHead decode_head(const std::vector<int32_t> & codes);
    std::vector<float> decode_audio(const std::vector<int32_t> & codes);
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

FsqAudioCodecDecoderWeights load_fsq_audio_codec_decoder_weights(
    const assets::TensorSource & source,
    const FsqAudioCodecConfig & config,
    FsqAudioCodecWeightBinding weight_binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type);

std::vector<float> decode_fsq_audio_codec_levels(
    const std::vector<int32_t> & codes,
    const std::vector<int64_t> & levels);

}  // namespace engine::codecs

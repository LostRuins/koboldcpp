#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/codecs/neural_audio.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/runtime/session.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::codecs {

struct MimiCodecConfig {
    int sample_rate = 24000;
    float frame_rate = 12.5F;
    int64_t channels = 1;
    int64_t hidden_size = 512;
    int64_t num_heads = 8;
    int64_t intermediate_size = 2048;
    int64_t transformer_layers = 8;
    int64_t context = 250;
    int64_t latent_size = 256;
    int64_t codebooks = 8;
    int64_t total_codebooks = 32;
    int64_t codebook_size = 2048;
    int64_t encoder_upsample_stride = 16;
};

struct MimiCodecTransformerLayerWeights {
    modules::NormWeights norm1;
    modules::AttentionWeights self_attn;
    modules::LayerScaleWeights layer_scale1;
    modules::NormWeights norm2;
    modules::FeedForwardWeights feed_forward;
    modules::LayerScaleWeights layer_scale2;
};

struct MimiCodecResidualWeights {
    modules::StreamingConv1dWeights conv1;
    modules::StreamingConv1dWeights conv2;
};

struct MimiCodecEncoderWeights {
    modules::StreamingConv1dWeights input_projection;
    std::vector<MimiCodecResidualWeights> residual_blocks;
    std::vector<modules::StreamingConv1dWeights> downsample_layers;
    modules::StreamingConv1dWeights output_projection;
    std::vector<MimiCodecTransformerLayerWeights> transformer_layers;
    modules::StreamingConv1dWeights downsample_conv;
};

struct MimiCodecCodebookWeights {
    core::TensorValue embedding;
    std::vector<float> host_embedding;
};

struct MimiCodecQuantizerWeights {
    std::vector<MimiCodecCodebookWeights> semantic_codebooks;
    std::vector<MimiCodecCodebookWeights> acoustic_codebooks;
    modules::Conv1dWeights semantic_input_proj;
    modules::Conv1dWeights acoustic_input_proj;
    modules::Conv1dWeights semantic_output_proj;
    modules::Conv1dWeights acoustic_output_proj;
};

struct MimiCodecDecoderWeights {
    std::vector<MimiCodecTransformerLayerWeights> transformer_layers;
    modules::DepthwiseConvTranspose1dWeights upsample_conv;
    SEANetDecoderWeights seanet;
    std::vector<std::vector<float>> seanet_upsample_bias_values;
};

struct MimiCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    MimiCodecEncoderWeights encoder;
    MimiCodecQuantizerWeights quantizer;
    MimiCodecDecoderWeights decoder;
};

struct MimiCodecWeightBinding {
    std::string encoder_input_projection_weight = "encoder.model.0.conv.conv.weight";
    std::string encoder_input_projection_bias = "encoder.model.0.conv.conv.bias";
    std::vector<std::string> encoder_residual_prefixes = {
        "encoder.model.1",
        "encoder.model.4",
        "encoder.model.7",
        "encoder.model.10",
    };
    std::vector<std::string> encoder_downsample_weight_names = {
        "encoder.model.3.conv.conv.weight",
        "encoder.model.6.conv.conv.weight",
        "encoder.model.9.conv.conv.weight",
        "encoder.model.12.conv.conv.weight",
    };
    std::vector<std::string> encoder_downsample_bias_names = {
        "encoder.model.3.conv.conv.bias",
        "encoder.model.6.conv.conv.bias",
        "encoder.model.9.conv.conv.bias",
        "encoder.model.12.conv.conv.bias",
    };
    std::string encoder_output_projection_weight = "encoder.model.14.conv.conv.weight";
    std::string encoder_output_projection_bias = "encoder.model.14.conv.conv.bias";
    std::string encoder_transformer_layer_prefix = "encoder_transformer.transformer.layers.";
    std::string downsample_weight = "downsample.conv.conv.conv.weight";

    std::string semantic_codebook_prefix = "quantizer.rvq_first.vq.layers.0._codebook.";
    std::string acoustic_codebook_layer_prefix = "quantizer.rvq_rest.vq.layers.";
    std::string acoustic_codebook_suffix = "._codebook.";
    std::string semantic_input_projection_weight = "quantizer.rvq_first.input_proj.weight";
    std::string acoustic_input_projection_weight = "quantizer.rvq_rest.input_proj.weight";
    std::string semantic_output_projection_weight = "quantizer.rvq_first.output_proj.weight";
    std::string acoustic_output_projection_weight = "quantizer.rvq_rest.output_proj.weight";

    std::string decoder_transformer_layer_prefix = "decoder_transformer.transformer.layers.";
    std::string decoder_upsample_weight = "upsample.convtr.convtr.convtr.weight";
    std::string decoder_input_projection_weight = "decoder.model.0.conv.conv.weight";
    std::string decoder_input_projection_bias = "decoder.model.0.conv.conv.bias";
    std::vector<std::string> decoder_upsample_weight_names = {
        "decoder.model.2.convtr.convtr.weight",
        "decoder.model.5.convtr.convtr.weight",
        "decoder.model.8.convtr.convtr.weight",
        "decoder.model.11.convtr.convtr.weight",
    };
    std::vector<std::string> decoder_upsample_bias_names = {
        "decoder.model.2.convtr.convtr.bias",
        "decoder.model.5.convtr.convtr.bias",
        "decoder.model.8.convtr.convtr.bias",
        "decoder.model.11.convtr.convtr.bias",
    };
    std::vector<std::string> decoder_residual_prefixes = {
        "decoder.model.3",
        "decoder.model.6",
        "decoder.model.9",
        "decoder.model.12",
    };
    std::string decoder_output_projection_weight = "decoder.model.14.conv.conv.weight";
    std::string decoder_output_projection_bias = "decoder.model.14.conv.conv.bias";
};

class MimiCodecComponent {
public:
    static MimiCodecComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        MimiCodecConfig config,
        MimiCodecWeightBinding weight_binding,
        ggml_backend_t backend,
        core::BackendType backend_type,
        size_t context_bytes,
        assets::TensorStorageType storage_type);

    MimiCodecComponent() = default;
    MimiCodecComponent(MimiCodecConfig config, std::shared_ptr<const MimiCodecWeights> weights);

    const MimiCodecConfig & config() const noexcept;
    const std::shared_ptr<const MimiCodecWeights> & weights() const noexcept;

private:
    MimiCodecConfig config_;
    std::shared_ptr<const MimiCodecWeights> weights_;
};

class MimiDecoderRuntime {
public:
    MimiDecoderRuntime(
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes);
    ~MimiDecoderRuntime();

    runtime::AudioBuffer decode(const std::vector<int32_t> & codes, int64_t frames);
    void reset_streaming();
    runtime::AudioBuffer decode_streaming(const std::vector<int32_t> & codes, int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MimiEncoderRuntime {
public:
    MimiEncoderRuntime(
        std::shared_ptr<const MimiCodecWeights> weights,
        MimiCodecConfig config,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_arena_bytes);
    ~MimiEncoderRuntime();

    std::vector<int32_t> encode(const runtime::AudioBuffer & audio);
    void reset_streaming();
    std::vector<int32_t> encode_streaming(const runtime::AudioBuffer & audio, bool flush);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<const MimiCodecWeights> load_mimi_codec_weights(
    const assets::TensorSource & source,
    const MimiCodecConfig & config,
    MimiCodecWeightBinding weight_binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type);

}  // namespace engine::codecs

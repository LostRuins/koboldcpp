#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/pocket_tts/assets.h"

#include "ggml-backend.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace engine::models::pocket_tts {

struct PocketTTSBackendResidualBlockWeights {
    modules::StreamingConv1dWeights conv1;
    modules::StreamingConv1dWeights conv2;
};

struct PocketTTSBackendPackedAttentionWeights {
    core::TensorValue in_proj_weight;
    core::TensorValue out_proj_weight;
};

struct PocketTTSBackendTransformerLayerWeights {
    modules::NormWeights norm1;
    PocketTTSBackendPackedAttentionWeights self_attn;
    std::optional<modules::LayerScaleWeights> layer_scale_1;
    modules::NormWeights norm2;
    modules::FeedForwardWeights feed_forward;
    std::optional<modules::LayerScaleWeights> layer_scale_2;
};

struct PocketTTSBackendFlowWeights {
    modules::LinearWeights input_linear;
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    modules::TimedConditionedFlowMLPWeights flow_net;
    modules::NormWeights out_norm;
    modules::LinearWeights out_eos;
    core::TensorValue speaker_proj_weight;
};

struct PocketTTSBackendMimiEncoderWeights {
    modules::StreamingConv1dWeights input_conv;
    PocketTTSBackendResidualBlockWeights block0;
    modules::StreamingConv1dWeights downsample0;
    PocketTTSBackendResidualBlockWeights block1;
    modules::StreamingConv1dWeights downsample1;
    PocketTTSBackendResidualBlockWeights block2;
    modules::StreamingConv1dWeights downsample2;
    modules::StreamingConv1dWeights output_conv;
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    modules::StreamingConv1dWeights downsample_conv;
};

struct PocketTTSBackendMimiDecoderWeights {
    std::vector<PocketTTSBackendTransformerLayerWeights> transformer_layers;
    core::TensorValue quantizer_output_proj_weight;
    core::TensorValue encoder_upsample_weight;
    modules::StreamingConv1dWeights input_projection;
    modules::ConvTranspose1dWeights stage0_upsample;
    std::vector<float> stage0_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage0_block;
    modules::ConvTranspose1dWeights stage1_upsample;
    std::vector<float> stage1_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage1_block;
    modules::ConvTranspose1dWeights stage2_upsample;
    std::vector<float> stage2_upsample_bias_values;
    PocketTTSBackendResidualBlockWeights stage2_block;
    modules::StreamingConv1dWeights output_projection;
};

struct PocketTTSBackendWeights {
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> flow_store;
    std::shared_ptr<core::BackendWeightStore> mimi_encoder_store;
    std::shared_ptr<core::BackendWeightStore> mimi_decoder_store;
    PocketTTSHostWeights host;
    PocketTTSBackendFlowWeights flow;
    PocketTTSBackendMimiEncoderWeights mimi_encoder;
    PocketTTSBackendMimiDecoderWeights mimi_decoder;
};

std::shared_ptr<const PocketTTSBackendWeights> load_pocket_tts_backend_weights(
    const PocketTTSAssets & manifest,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t flow_context_bytes,
    size_t mimi_encoder_context_bytes,
    size_t mimi_decoder_context_bytes);

}  // namespace engine::models::pocket_tts

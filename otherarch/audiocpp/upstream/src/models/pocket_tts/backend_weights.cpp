#include "engine/models/pocket_tts/backend_weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::pocket_tts {
namespace {

namespace binding = engine::modules::binding;

PocketTTSBackendPackedAttentionWeights load_backend_packed_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    const std::string in_name = prefix + "self_attn.in_proj.weight";
    const std::string out_name = prefix + "self_attn.out_proj.weight";
    return {
        binding::tensor_from_named_source(store, source, in_name, storage_type),
        binding::tensor_from_named_source(store, source, out_name, storage_type),
    };
}

PocketTTSBackendTransformerLayerWeights load_backend_transformer_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    PocketTTSBackendTransformerLayerWeights layer;
    layer.norm1 = binding::norm_from_named_source(store, source, prefix + "norm1.weight", prefix + "norm1.bias");
    layer.self_attn = load_backend_packed_attention(store, source, prefix, storage_type);
    layer.layer_scale_1 = binding::optional_layer_scale_from_named_source(store, source, prefix + "layer_scale_1.scale");
    layer.norm2 = binding::norm_from_named_source(store, source, prefix + "norm2.weight", prefix + "norm2.bias");
    layer.feed_forward = modules::FeedForwardWeights{
        binding::tensor_from_named_source(store, source, prefix + "linear1.weight", storage_type),
        std::nullopt,
        binding::tensor_from_named_source(store, source, prefix + "linear2.weight", storage_type),
        std::nullopt,
    };
    layer.layer_scale_2 = binding::optional_layer_scale_from_named_source(store, source, prefix + "layer_scale_2.scale");
    return layer;
}

PocketTTSBackendResidualBlockWeights load_backend_residual_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType conv_storage_type) {
    return {
        binding::conv1d_from_named_source(store, source, prefix + ".1.conv.weight", prefix + ".1.conv.bias", conv_storage_type),
        binding::conv1d_from_named_source(store, source, prefix + ".3.conv.weight", prefix + ".3.conv.bias", conv_storage_type),
    };
}

modules::TimestepEmbeddingWeights load_backend_timestep_embedding(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return modules::TimestepEmbeddingWeights{
        binding::f32_tensor_from_named_source(store, source, prefix + ".freqs"),
        binding::linear_from_named_source(store, source, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias", storage_type),
        binding::linear_from_named_source(store, source, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias", storage_type),
        binding::f32_tensor_from_named_source(store, source, prefix + ".mlp.3.alpha"),
    };
}

modules::TimedConditionedFlowMLPWeights load_backend_flow_net_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layers,
    assets::TensorStorageType storage_type) {
    modules::TimedConditionedFlowMLPWeights weights;
    weights.start_time_embedding = load_backend_timestep_embedding(store, source, "flow_lm.flow_net.time_embed.0", storage_type);
    weights.end_time_embedding = load_backend_timestep_embedding(store, source, "flow_lm.flow_net.time_embed.1", storage_type);
    weights.input_projection = binding::linear_from_named_source(
        store,
        source,
        "flow_lm.flow_net.input_proj.weight",
        "flow_lm.flow_net.input_proj.bias",
        storage_type);
    weights.condition_projection = binding::linear_from_named_source(
        store,
        source,
        "flow_lm.flow_net.cond_embed.weight",
        "flow_lm.flow_net.cond_embed.bias",
        storage_type);
    weights.residual_layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "flow_lm.flow_net.res_blocks." + std::to_string(layer);
        weights.residual_layers.push_back(modules::AdaLNResidualMLPWeights{
            binding::norm_from_named_source(store, source, prefix + ".in_ln.weight", prefix + ".in_ln.bias"),
            modules::AdaLNModulationWeights{
                binding::linear_from_named_source(store, source, prefix + ".adaLN_modulation.1.weight", prefix + ".adaLN_modulation.1.bias", storage_type),
            },
            binding::linear_from_named_source(store, source, prefix + ".mlp.0.weight", prefix + ".mlp.0.bias", storage_type),
            binding::linear_from_named_source(store, source, prefix + ".mlp.2.weight", prefix + ".mlp.2.bias", storage_type),
        });
    }
    weights.output_projection = modules::FinalAdaLNProjectionWeights{
        modules::AdaLNModulationWeights{
            binding::linear_from_named_source(
                store,
                source,
                "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight",
                "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias",
                storage_type),
        },
        binding::linear_from_named_source(
            store,
            source,
            "flow_lm.flow_net.final_layer.linear.weight",
            "flow_lm.flow_net.final_layer.linear.bias",
            storage_type),
    };
    return weights;
}

PocketTTSBackendFlowWeights load_backend_flow_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType storage_type) {
    PocketTTSBackendFlowWeights weights;
    weights.input_linear = modules::LinearWeights{
        binding::tensor_from_named_source(store, source, "flow_lm.input_linear.weight", storage_type),
        std::nullopt,
    };
    weights.transformer_layers.reserve(static_cast<size_t>(config.flow_layers));
    for (int64_t layer = 0; layer < config.flow_layers; ++layer) {
        const std::string prefix = "flow_lm.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, storage_type));
    }
    weights.flow_net = load_backend_flow_net_weights(store, source, config.flow_depth, storage_type);
    weights.out_norm = binding::norm_from_named_source(store, source, "flow_lm.out_norm.weight", "flow_lm.out_norm.bias");
    weights.out_eos = binding::linear_from_named_source(store, source, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias", storage_type);
    weights.speaker_proj_weight =
        binding::tensor_from_named_source(store, source, "flow_lm.speaker_proj_weight", storage_type);
    return weights;
}

PocketTTSBackendMimiEncoderWeights load_backend_mimi_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    PocketTTSBackendMimiEncoderWeights weights;
    weights.input_conv = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.0.conv.weight", "mimi.encoder.model.0.conv.bias", conv_storage_type);
    weights.block0 = load_backend_residual_block(store, source, "mimi.encoder.model.1.block", conv_storage_type);
    weights.downsample0 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.3.conv.weight", "mimi.encoder.model.3.conv.bias", conv_storage_type);
    weights.block1 = load_backend_residual_block(store, source, "mimi.encoder.model.4.block", conv_storage_type);
    weights.downsample1 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.6.conv.weight", "mimi.encoder.model.6.conv.bias", conv_storage_type);
    weights.block2 = load_backend_residual_block(store, source, "mimi.encoder.model.7.block", conv_storage_type);
    weights.downsample2 = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.9.conv.weight", "mimi.encoder.model.9.conv.bias", conv_storage_type);
    weights.output_conv = binding::conv1d_from_named_source(store, source, "mimi.encoder.model.11.conv.weight", "mimi.encoder.model.11.conv.bias", conv_storage_type);
    weights.transformer_layers.reserve(static_cast<size_t>(config.mimi_layers));
    for (int64_t layer = 0; layer < config.mimi_layers; ++layer) {
        const std::string prefix = "mimi.encoder_transformer.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, matmul_storage_type));
    }
    weights.downsample_conv = modules::StreamingConv1dWeights{
        binding::tensor_from_named_source(store, source, "mimi.downsample.conv.conv.weight", conv_storage_type),
        std::nullopt,
    };
    return weights;
}

PocketTTSBackendMimiDecoderWeights load_backend_mimi_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const PocketTTSModelConfig & config,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    core::BackendType backend_type) {
    PocketTTSBackendMimiDecoderWeights weights;
    weights.transformer_layers.reserve(static_cast<size_t>(config.mimi_layers));
    for (int64_t layer = 0; layer < config.mimi_layers; ++layer) {
        const std::string prefix = "mimi.decoder_transformer.transformer.layers." + std::to_string(layer) + ".";
        weights.transformer_layers.push_back(load_backend_transformer_layer(store, source, prefix, matmul_storage_type));
    }
    weights.quantizer_output_proj_weight =
        binding::tensor_from_named_source(store, source, "mimi.quantizer.output_proj.weight", conv_storage_type);
    const int64_t upsample_kernel_size = config.mimi_encoder_upsample_stride * 2;
    auto upsample_weight = source.require_f32("mimi.upsample.convtr.convtr.weight");
    if (static_cast<int64_t>(upsample_weight.size()) != config.mimi_dim * upsample_kernel_size) {
        throw std::runtime_error("mimi.upsample.convtr.convtr.weight has invalid depthwise conv transpose weight size");
    }
    if (core::uses_host_graph_plan(backend_type)) {
        // The depthwise module implements transpose convolution with a reversed FIR.
        for (int64_t channel = 0; channel < config.mimi_dim; ++channel) {
            auto * kernel = upsample_weight.data() + channel * upsample_kernel_size;
            std::reverse(kernel, kernel + upsample_kernel_size);
        }
        weights.encoder_upsample_weight = store.make_f32(
            core::TensorShape::from_dims({config.mimi_dim, 1, 1, upsample_kernel_size}),
            upsample_weight);
    } else {
        // Preserve the existing GPU GEMM lowering and its arithmetic.
        std::vector<float> dense(static_cast<size_t>(config.mimi_dim * config.mimi_dim * upsample_kernel_size), 0.0F);
        for (int64_t channel = 0; channel < config.mimi_dim; ++channel) {
            std::copy_n(
                upsample_weight.data() + channel * upsample_kernel_size,
                upsample_kernel_size,
                dense.data() + (channel * config.mimi_dim + channel) * upsample_kernel_size);
        }
        weights.encoder_upsample_weight = store.make_f32(
            core::TensorShape::from_dims({config.mimi_dim, config.mimi_dim, upsample_kernel_size}),
            dense);
    }
    weights.input_projection = binding::conv1d_from_named_source(store, source, "mimi.decoder.model.0.conv.weight", "mimi.decoder.model.0.conv.bias", conv_storage_type);
    weights.stage0_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.2.convtr.weight", "mimi.decoder.model.2.convtr.bias", conv_storage_type);
    weights.stage0_upsample_bias_values = source.require_f32("mimi.decoder.model.2.convtr.bias");
    weights.stage0_block = load_backend_residual_block(store, source, "mimi.decoder.model.3.block", conv_storage_type);
    weights.stage1_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.5.convtr.weight", "mimi.decoder.model.5.convtr.bias", conv_storage_type);
    weights.stage1_upsample_bias_values = source.require_f32("mimi.decoder.model.5.convtr.bias");
    weights.stage1_block = load_backend_residual_block(store, source, "mimi.decoder.model.6.block", conv_storage_type);
    weights.stage2_upsample = binding::conv_transpose1d_from_named_source(store, source, "mimi.decoder.model.8.convtr.weight", "mimi.decoder.model.8.convtr.bias", conv_storage_type);
    weights.stage2_upsample_bias_values = source.require_f32("mimi.decoder.model.8.convtr.bias");
    weights.stage2_block = load_backend_residual_block(store, source, "mimi.decoder.model.9.block", conv_storage_type);
    weights.output_projection = binding::conv1d_from_named_source(store, source, "mimi.decoder.model.11.conv.weight", "mimi.decoder.model.11.conv.bias", conv_storage_type);
    return weights;
}

}  // namespace

std::shared_ptr<const PocketTTSBackendWeights> load_pocket_tts_backend_weights(
    const PocketTTSAssets & manifest,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type,
    size_t flow_context_bytes,
    size_t mimi_encoder_context_bytes,
    size_t mimi_decoder_context_bytes) {
    if (manifest.model_weights == nullptr) {
        throw std::runtime_error("PocketTTS model weights source is not loaded");
    }
    auto weights = std::make_shared<PocketTTSBackendWeights>();
    weights->backend_type = backend_type;
    weights->host = manifest.host_weights;
    weights->flow_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.flow_lm.weights",
        flow_context_bytes);
    weights->flow =
        load_backend_flow_weights(*weights->flow_store, *manifest.model_weights, manifest.model_config, matmul_storage_type);
    weights->flow_store->upload();

    weights->mimi_encoder_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.mimi_encoder.weights",
        mimi_encoder_context_bytes);
    weights->mimi_encoder =
        load_backend_mimi_encoder_weights(
            *weights->mimi_encoder_store,
            *manifest.model_weights,
            manifest.model_config,
            matmul_storage_type,
            conv_storage_type);
    weights->mimi_encoder_store->upload();

    weights->mimi_decoder_store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "PocketTTS.mimi_decoder.weights",
        mimi_decoder_context_bytes);
    weights->mimi_decoder =
        load_backend_mimi_decoder_weights(
            *weights->mimi_decoder_store,
            *manifest.model_weights,
            manifest.model_config,
            matmul_storage_type,
            conv_storage_type,
            backend_type);
    weights->mimi_decoder_store->upload();

    manifest.model_weights->release_storage();
    return weights;
}

}  // namespace engine::models::pocket_tts

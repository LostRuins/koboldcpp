#include "engine/models/pocket_tts/mimi_encoder.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/pocket_tts/backend_weights.h"
#include "graph_common.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::pocket_tts {
namespace {

constexpr int64_t kMimiInputChannels = 1;

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    int64_t hidden_size,
    int64_t layer) {
    return graph_common::make_transformer_block_weights(
        ctx,
        weights.mimi_encoder.transformer_layers.at(static_cast<size_t>(layer)),
        hidden_size);
}

core::TensorValue build_residual_block(
    core::ModuleBuildContext & ctx,
    const PocketTTSBackendResidualBlockWeights & weights,
    const core::TensorValue & input,
    int64_t channels,
    int64_t hidden_channels) {
    auto x = modules::EluModule().build(ctx, input);
    x = modules::StreamingConv1dModule({
        channels,
        hidden_channels,
        3,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(ctx, x, {
        weights.conv1.weight,
        weights.conv1.bias,
    });
    x = modules::EluModule().build(ctx, x);
    x = modules::StreamingConv1dModule({
        hidden_channels,
        channels,
        1,
        1,
        1,
        true,
        modules::StreamingPadMode::Constant,
    }).build(ctx, x, {
        weights.conv2.weight,
        weights.conv2.bias,
    });
    return modules::ResidualAddModule().build(ctx, input, x);
}

core::TensorValue build_causal_context_mask(
    core::ModuleBuildContext & ctx,
    int64_t query_frames,
    int64_t key_frames) {
    return core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, query_frames, key_frames}));
}

std::vector<float> build_causal_context_mask_values(int64_t query_frames, int64_t key_frames, int64_t context) {
    std::vector<float> values(static_cast<size_t>(query_frames * key_frames), -INFINITY);
    for (int64_t q = 0; q < query_frames; ++q) {
        const int64_t first_k = std::max<int64_t>(0, q - context + 1);
        for (int64_t k = first_k; k <= q && k < key_frames; ++k) {
            values[static_cast<size_t>(q * key_frames + k)] = 0.0F;
        }
    }
    return values;
}

}  // namespace

MimiPromptEmbedding MimiEncoder::encode_prompt_embedding(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSBackendWeights & weights,
    const std::vector<float> & waveform,
    size_t graph_context_bytes) const {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    const int64_t mimi_dimension = manifest.model_config.mimi_seanet_dimension;
    const int64_t mimi_base_channels = manifest.model_config.mimi_base_filters;
    const int64_t mimi_heads = manifest.model_config.mimi_heads;
    const int64_t mimi_intermediate = manifest.model_config.mimi_intermediate_size;
    const int64_t mimi_transformer_layers = manifest.model_config.mimi_layers;
    const int64_t mimi_inner_dimension = manifest.model_config.mimi_inner_dim;
    const int64_t mimi_context = manifest.model_config.mimi_context;
    struct Capture {
        core::TensorValue input;
        core::TensorValue positions;
        core::TensorValue attention_mask;
        std::vector<int32_t> position_values;
        std::vector<float> attention_mask_values;
        core::TensorValue seanet;
        core::TensorValue transformer;
        core::TensorValue downsample;
        core::TensorValue conditioning;
    } capture;

    const auto started = std::chrono::steady_clock::now();
    auto result = graph_common::run_graph_with_backend(
        "mimi_encoder.encode_audio_prompt",
        backend,
        weights.backend_type,
        threads,
        graph_context_bytes,
        [&](core::ModuleBuildContext & ctx) {
            auto input = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, static_cast<int64_t>(waveform.size())}));
            capture.input = input;

            auto x = modules::StreamingConv1dModule({
                kMimiInputChannels,
                mimi_base_channels,
                7,
                1,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, input, {
                weights.mimi_encoder.input_conv.weight,
                weights.mimi_encoder.input_conv.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block0, x, mimi_base_channels, mimi_base_channels / 2);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels,
                mimi_base_channels * 2,
                8,
                4,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample0.weight,
                weights.mimi_encoder.downsample0.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block1, x, mimi_base_channels * 2, mimi_base_channels);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels * 2,
                mimi_base_channels * 4,
                10,
                5,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample1.weight,
                weights.mimi_encoder.downsample1.bias,
            });
            x = build_residual_block(ctx, weights.mimi_encoder.block2, x, mimi_base_channels * 4, mimi_base_channels * 2);
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_base_channels * 4,
                mimi_dimension,
                12,
                6,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample2.weight,
                weights.mimi_encoder.downsample2.bias,
            });
            x = modules::EluModule().build(ctx, x);
            x = modules::StreamingConv1dModule({
                mimi_dimension,
                mimi_dimension,
                3,
                1,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {
                weights.mimi_encoder.output_conv.weight,
                weights.mimi_encoder.output_conv.bias,
            });
            capture.seanet = x;

            const int64_t encoder_frames = x.shape.dims[2];
            auto positions = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({encoder_frames}));
            capture.positions = positions;
            capture.position_values.resize(static_cast<size_t>(encoder_frames));
            for (int64_t i = 0; i < encoder_frames; ++i) {
                capture.position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            std::vector<modules::TransformerEncoderBlockWeights> transformer_layers;
            transformer_layers.reserve(static_cast<size_t>(mimi_transformer_layers));
            for (int64_t layer = 0; layer < mimi_transformer_layers; ++layer) {
                transformer_layers.push_back(make_transformer_layer_weights(ctx, weights, mimi_dimension, layer));
            }
            auto attention_mask = build_causal_context_mask(ctx, encoder_frames, encoder_frames);
            capture.attention_mask = attention_mask;
            capture.attention_mask_values = build_causal_context_mask_values(encoder_frames, encoder_frames, mimi_context);
            x = modules::StreamingProjectedTransformerModule({
                mimi_dimension,
                mimi_dimension,
                mimi_dimension,
                mimi_heads,
                mimi_intermediate,
                mimi_transformer_layers,
                1.0e-5F,
                false,
            }).build(ctx, x, positions, {
                std::nullopt,
                std::move(transformer_layers),
                std::nullopt,
            }, std::nullopt, attention_mask).output;
            capture.transformer = x;
            x = modules::StreamingConv1dModule({
                mimi_dimension,
                mimi_inner_dimension,
                32,
                16,
                1,
                false,
                modules::StreamingPadMode::Replicate,
            }).build(ctx, x, {
                weights.mimi_encoder.downsample_conv.weight,
                weights.mimi_encoder.downsample_conv.bias,
            });
            capture.downsample = x;

            auto latents_btf = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            capture.conditioning = modules::LinearModule({mimi_inner_dimension, flow_hidden, false}).build(
                ctx,
                latents_btf,
                {
                    weights.flow.speaker_proj_weight,
                    std::nullopt,
                });
            return capture.conditioning;
        },
        [&]() {
            core::write_tensor_f32(capture.input, waveform);
            core::write_tensor_i32(capture.positions, capture.position_values);
            core::write_tensor_f32(capture.attention_mask, capture.attention_mask_values);
        },
        [&](ggml_context *) {
            MimiPromptEmbedding result;
            result.values = core::read_tensor_f32(capture.conditioning.tensor);
            result.frames = static_cast<int64_t>(result.values.size() / static_cast<size_t>(flow_hidden));
            return result;
        });
    const double encode_ms = engine::debug::elapsed_ms(started);
    engine::debug::timing_log_scalar("pocket_tts.mimi_encoder_ms", encode_ms);
    return result;
}

}  // namespace engine::models::pocket_tts

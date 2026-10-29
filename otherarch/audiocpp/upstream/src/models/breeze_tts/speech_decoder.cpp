#include "engine/models/breeze_tts/speech_decoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::breeze_tts {
namespace json = engine::io::json;
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

constexpr int64_t kSampleRate = 24000;
constexpr int64_t kDecodeSamplesPerCode = 1920;
constexpr int64_t kChunkCodes = 300;
constexpr int64_t kLeftContextCodes = 25;
constexpr std::array<int64_t, 2> kStrixHaloCachedChunkFrames{300, 105};
#if defined(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS)
constexpr bool kStrixHaloGraphCacheEnabled = true;
#else
constexpr bool kStrixHaloGraphCacheEnabled = false;
#endif
constexpr float kCodebookEps = 1.0e-5F;
constexpr float kMaskNegInf = -1.0e9F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TransformerLayerWeights {
    modules::AttentionWeights attention;
    modules::GatedFeedForwardWeights mlp;
    modules::NormWeights input_norm;
    modules::NormWeights post_norm;
    modules::LayerScaleWeights attn_scale;
    modules::LayerScaleWeights mlp_scale;
};

struct ConvNeXtWeights {
    modules::Conv1dWeights dwconv;
    modules::NormWeights norm;
    modules::LinearWeights pwconv1;
    modules::LinearWeights pwconv2;
    modules::LayerScaleWeights gamma;
};

struct ResidualUnitWeights {
    std::vector<float> act1_alpha;
    std::vector<float> act1_beta;
    modules::Conv1dWeights conv1;
    std::vector<float> act2_alpha;
    std::vector<float> act2_beta;
    modules::Conv1dWeights conv2;
};

struct UpsampleStageWeights {
    modules::ConvTranspose1dWeights upconv;
    ConvNeXtWeights convnext;
};

struct DecoderBlockWeights {
    std::vector<float> input_alpha;
    std::vector<float> input_beta;
    modules::ConvTranspose1dWeights upconv;
    std::vector<ResidualUnitWeights> residual_units;
};

struct DecoderConfig {
    int64_t codebook_size = 0;
    int64_t codebook_dim = 0;
    int64_t latent_dim = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t decoder_dim = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t num_layers = 0;
    int64_t num_quantizers = 0;
    int64_t num_semantic_quantizers = 1;
    int64_t sliding_window = 0;
    int64_t head_dim = 0;
    float rope_theta = 10000.0F;
    float rms_norm_eps = 1.0e-5F;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsampling_ratios;
};

}  // namespace

struct BreezeSpeechDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    DecoderConfig config;
    std::vector<core::TensorValue> semantic_codebooks;
    std::vector<core::TensorValue> acoustic_codebooks;
    modules::LinearWeights semantic_output_proj;
    modules::LinearWeights acoustic_output_proj;
    modules::Conv1dWeights pre_conv;
    modules::LinearWeights transformer_input_proj;
    std::vector<TransformerLayerWeights> transformer_layers;
    modules::NormWeights transformer_norm;
    modules::LinearWeights transformer_output_proj;
    std::vector<UpsampleStageWeights> upsample_stages;
    modules::Conv1dWeights decoder_input_conv;
    std::vector<DecoderBlockWeights> decoder_blocks;
    std::vector<float> output_alpha;
    std::vector<float> output_beta;
    modules::Conv1dWeights output_conv;
};

namespace {

core::TensorValue normalized_codebook(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t size,
    int64_t dim) {
    const auto cluster_usage = source.require_f32(prefix + "cluster_usage", {size});
    const auto embedding_sum = source.require_f32(prefix + "embedding_sum", {size, dim});
    std::vector<float> embedding(embedding_sum.size(), 0.0F);
    for (int64_t code = 0; code < size; ++code) {
        const float denom = std::max(cluster_usage[static_cast<size_t>(code)], kCodebookEps);
        for (int64_t col = 0; col < dim; ++col) {
            const size_t offset = static_cast<size_t>(code * dim + col);
            embedding[offset] = embedding_sum[offset] / denom;
        }
    }
    return store.make_f32(core::TensorShape::from_dims({size, dim}), embedding);
}

DecoderConfig load_decoder_config(const BreezeTTSAssets & assets) {
    const auto root = assets.resources.parse_json("audio_tokenizer_config_json");
    const auto & decoder = root.require("decoder_config");
    DecoderConfig config;
    config.codebook_size = json::require_i64(decoder, "codebook_size");
    config.codebook_dim = json::require_i64(decoder, "codebook_dim");
    config.latent_dim = json::require_i64(decoder, "latent_dim");
    config.hidden_size = json::require_i64(decoder, "hidden_size");
    config.intermediate_size = json::require_i64(decoder, "intermediate_size");
    config.decoder_dim = json::require_i64(decoder, "decoder_dim");
    config.num_heads = json::require_i64(decoder, "num_attention_heads");
    config.num_kv_heads = json::require_i64(decoder, "num_key_value_heads");
    config.num_layers = json::require_i64(decoder, "num_hidden_layers");
    config.num_quantizers = json::require_i64(decoder, "num_quantizers");
    config.num_semantic_quantizers = json::require_i64(decoder, "num_semantic_quantizers");
    config.sliding_window = json::require_i64(decoder, "sliding_window");
    config.head_dim = json::require_i64(decoder, "head_dim");
    config.rope_theta = json::optional_f32(decoder, "rope_theta", config.rope_theta);
    config.rms_norm_eps = json::optional_f32(decoder, "rms_norm_eps", config.rms_norm_eps);
    config.upsample_rates = json::require_i64_array(decoder, "upsample_rates");
    config.upsampling_ratios = json::require_i64_array(decoder, "upsampling_ratios");
    return config;
}

modules::LinearWeights load_conv1x1_as_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t input_dim,
    int64_t output_dim) {
    modules::LinearWeights weights;
    const auto data = source.require_tensor_as_shape(
        prefix + ".weight",
        storage_type,
        {output_dim, input_dim, 1},
        {output_dim, input_dim});
    weights.weight = store.make_tensor(data.shape, data.type, data.bytes.data(), data.bytes.size());
    return weights;
}

std::shared_ptr<const BreezeSpeechDecoderWeights> load_weights(
    const BreezeTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    const auto & source = *assets.weights;
    auto weights = std::make_shared<BreezeSpeechDecoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "breeze_tts.speech_decoder.weights",
        32ull * 1024ull * 1024ull);
    weights->config = load_decoder_config(assets);
    const auto & config = weights->config;
    const int64_t split_dim = config.codebook_dim / 2;

    for (int64_t layer = 0; layer < config.num_semantic_quantizers; ++layer) {
        const std::string prefix = "codec_model.decoder.quantizer.rvq_first.vq.layers." + std::to_string(layer) + "._codebook.";
        weights->semantic_codebooks.push_back(normalized_codebook(*weights->store, source, prefix, config.codebook_size, split_dim));
    }
    for (int64_t layer = 0; layer < config.num_quantizers - config.num_semantic_quantizers; ++layer) {
        const std::string prefix = "codec_model.decoder.quantizer.rvq_rest.vq.layers." + std::to_string(layer) + "._codebook.";
        weights->acoustic_codebooks.push_back(normalized_codebook(*weights->store, source, prefix, config.codebook_size, split_dim));
    }
    weights->semantic_output_proj = load_conv1x1_as_linear(
        *weights->store,
        source,
        "codec_model.decoder.quantizer.rvq_first.output_proj",
        linear_weight_storage_type,
        split_dim,
        config.hidden_size);
    weights->acoustic_output_proj = load_conv1x1_as_linear(
        *weights->store,
        source,
        "codec_model.decoder.quantizer.rvq_rest.output_proj",
        linear_weight_storage_type,
        split_dim,
        config.hidden_size);
    weights->pre_conv = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.decoder.pre_conv.conv",
        conv_weight_storage_type,
        config.latent_dim,
        config.hidden_size,
        3,
        true);
    weights->transformer_input_proj = binding::linear_from_source(
        *weights->store,
        source,
        "codec_model.decoder.pre_transformer.input_proj",
        linear_weight_storage_type,
        config.hidden_size,
        config.latent_dim,
        true);
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "codec_model.decoder.pre_transformer.layers." + std::to_string(layer);
        TransformerLayerWeights block;
        block.input_norm = binding::norm_weight_from_source(*weights->store, source, prefix + ".input_layernorm", config.hidden_size);
        block.post_norm = binding::norm_weight_from_source(*weights->store, source, prefix + ".post_attention_layernorm", config.hidden_size);
        block.attention.q_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            linear_weight_storage_type,
            {config.num_heads * config.head_dim, config.hidden_size});
        block.attention.k_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            linear_weight_storage_type,
            {config.num_kv_heads * config.head_dim, config.hidden_size});
        block.attention.v_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            linear_weight_storage_type,
            {config.num_kv_heads * config.head_dim, config.hidden_size});
        block.attention.out_weight = weights->store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            linear_weight_storage_type,
            {config.hidden_size, config.num_heads * config.head_dim});
        block.mlp.gate_proj = binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.gate_proj",
            linear_weight_storage_type,
            config.intermediate_size,
            config.hidden_size,
            false);
        block.mlp.up_proj = binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.up_proj",
            linear_weight_storage_type,
            config.intermediate_size,
            config.hidden_size,
            false);
        block.mlp.down_proj = binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.down_proj",
            linear_weight_storage_type,
            config.hidden_size,
            config.intermediate_size,
            false);
        block.attn_scale = binding::layer_scale_from_named_source(*weights->store, source, prefix + ".self_attn_layer_scale.scale");
        block.mlp_scale = binding::layer_scale_from_named_source(*weights->store, source, prefix + ".mlp_layer_scale.scale");
        weights->transformer_layers.push_back(std::move(block));
    }
    weights->transformer_norm = binding::norm_weight_from_source(
        *weights->store,
        source,
        "codec_model.decoder.pre_transformer.norm",
        config.hidden_size);
    weights->transformer_output_proj = binding::linear_from_source(
        *weights->store,
        source,
        "codec_model.decoder.pre_transformer.output_proj",
        linear_weight_storage_type,
        config.latent_dim,
        config.hidden_size,
        true);

    for (size_t i = 0; i < config.upsampling_ratios.size(); ++i) {
        const std::string prefix = "codec_model.decoder.upsample." + std::to_string(i);
        UpsampleStageWeights stage;
        stage.upconv = binding::conv_transpose1d_from_source(
            *weights->store,
            source,
            prefix + ".0.conv",
            conv_weight_storage_type,
            config.latent_dim,
            config.latent_dim,
            config.upsampling_ratios[i],
            true);
        stage.convnext.dwconv = binding::conv1d_from_source(
            *weights->store,
            source,
            prefix + ".1.dwconv.conv",
            conv_weight_storage_type,
            config.latent_dim,
            1,
            7,
            true);
        stage.convnext.norm = binding::norm_from_source(*weights->store, source, prefix + ".1.norm", config.latent_dim);
        stage.convnext.pwconv1 = binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".1.pwconv1",
            linear_weight_storage_type,
            config.latent_dim * 4,
            config.latent_dim,
            true);
        stage.convnext.pwconv2 = binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".1.pwconv2",
            linear_weight_storage_type,
            config.latent_dim,
            config.latent_dim * 4,
            true);
        stage.convnext.gamma = binding::layer_scale_from_named_source(*weights->store, source, prefix + ".1.gamma");
        weights->upsample_stages.push_back(std::move(stage));
    }

    weights->decoder_input_conv = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.decoder.decoder.0.conv",
        conv_weight_storage_type,
        config.decoder_dim,
        config.latent_dim,
        7,
        true);
    int64_t channels = config.decoder_dim;
    for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
        const std::string prefix = "codec_model.decoder.decoder." + std::to_string(i + 1) + ".block";
        const int64_t out_channels = channels / 2;
        DecoderBlockWeights block;
        block.input_alpha = source.require_f32(prefix + ".0.alpha", {channels});
        block.input_beta = source.require_f32(prefix + ".0.beta", {channels});
        block.upconv = binding::conv_transpose1d_from_source(
            *weights->store,
            source,
            prefix + ".1.conv",
            conv_weight_storage_type,
            channels,
            out_channels,
            config.upsample_rates[i] * 2,
            true);
        for (int unit_index = 0; unit_index < 3; ++unit_index) {
            const std::string unit = prefix + "." + std::to_string(unit_index + 2);
            ResidualUnitWeights residual;
            residual.act1_alpha = source.require_f32(unit + ".act1.alpha", {out_channels});
            residual.act1_beta = source.require_f32(unit + ".act1.beta", {out_channels});
            residual.conv1 = binding::conv1d_from_source(
                *weights->store,
                source,
                unit + ".conv1.conv",
                conv_weight_storage_type,
                out_channels,
                out_channels,
                7,
                true);
            residual.act2_alpha = source.require_f32(unit + ".act2.alpha", {out_channels});
            residual.act2_beta = source.require_f32(unit + ".act2.beta", {out_channels});
            residual.conv2 = binding::conv1d_from_source(
                *weights->store,
                source,
                unit + ".conv2.conv",
                conv_weight_storage_type,
                out_channels,
                out_channels,
                1,
                true);
            block.residual_units.push_back(std::move(residual));
        }
        weights->decoder_blocks.push_back(std::move(block));
        channels = out_channels;
    }
    weights->output_alpha = source.require_f32("codec_model.decoder.decoder.5.alpha", {channels});
    weights->output_beta = source.require_f32("codec_model.decoder.decoder.5.beta", {channels});
    weights->output_conv = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.decoder.decoder.6.conv",
        conv_weight_storage_type,
        1,
        channels,
        7,
        true);
    weights->store->upload();
    return weights;
}

core::TensorValue causal_conv1d(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t dilation,
    int64_t groups,
    bool use_bias) {
    const int64_t in_channels = input.shape.dims[1];
    const int64_t kernel_extent = (kernel - 1) * dilation + 1;
    const int64_t left_pad = kernel_extent - stride;
    const int64_t length = input.shape.dims[2];
    const float n_frames = static_cast<float>(length - kernel_extent + left_pad) / static_cast<float>(stride) + 1.0F;
    const int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride + (kernel_extent - left_pad);
    const int64_t right_pad = std::max<int64_t>(0, ideal_length - length);
    auto * padded = ggml_pad_ext(
        build_ctx.ggml,
        input.tensor,
        static_cast<int>(left_pad),
        static_cast<int>(right_pad),
        0,
        0,
        0,
        0,
        0,
        0);
    auto padded_value = core::wrap_tensor(
        padded,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + left_pad + right_pad}),
        GGML_TYPE_F32);
    if (weights.weight.type != GGML_TYPE_F32 && weights.weight.type != GGML_TYPE_F16) {
        throw std::runtime_error(
            std::string("Breeze speech decoder depthwise conv does not support weight type: ") +
            ggml_type_name(weights.weight.type));
    }
    ggml_tensor * result = nullptr;
    if (groups == in_channels) {
        ggml_tensor * bias = nullptr;
        if (use_bias) {
            if (!weights.bias.has_value()) {
                throw std::runtime_error("Breeze speech decoder depthwise conv requires bias");
            }
            bias = core::reshape_tensor(build_ctx, *weights.bias, core::TensorShape::from_dims({out_channels, 1})).tensor;
        }
        for (int64_t batch = 0; batch < padded_value.shape.dims[0]; ++batch) {
            auto * batch_input = ggml_view_2d(
                build_ctx.ggml,
                padded,
                padded->ne[0],
                padded->ne[1],
                padded->nb[1],
                static_cast<size_t>(batch) * padded->nb[2]);
            auto * batch_output = ggml_conv_1d_dw(
                build_ctx.ggml,
                weights.weight.tensor,
                core::has_backend_addressable_layout(batch_input) ? batch_input : ggml_cont(build_ctx.ggml, batch_input),
                static_cast<int>(stride),
                0,
                static_cast<int>(dilation));
            if (bias != nullptr) {
                batch_output = ggml_add(build_ctx.ggml, batch_output, bias);
            }
            batch_output = ggml_reshape_3d(build_ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1);
            result = result == nullptr ? batch_output : ggml_concat(build_ctx.ggml, result, batch_output, 2);
        }
        return core::wrap_tensor(
            result,
            core::TensorShape::from_dims({input.shape.dims[0], out_channels, result->ne[0]}),
            GGML_TYPE_F32);
    }
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        static_cast<int>(stride),
        0,
        static_cast<int>(dilation),
        use_bias,
    }).build(build_ctx, padded_value, weights);
}

core::TensorValue causal_conv_transpose1d(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    bool use_bias) {
    const int64_t right_trim = kernel - stride;
    auto output_bct = modules::ConvTranspose1dModule({
        input.shape.dims[1],
        out_channels,
        kernel,
        static_cast<int>(stride),
        0,
        1,
        use_bias,
    }).build(build_ctx, input, weights);
    if (right_trim <= 0) {
        return output_bct;
    }
    const int64_t trimmed_frames = output_bct.tensor->ne[0] - right_trim;
    return core::wrap_tensor(
        ggml_cont(
            build_ctx.ggml,
            ggml_view_3d(
                build_ctx.ggml,
                output_bct.tensor,
                trimmed_frames,
                out_channels,
                input.shape.dims[0],
                output_bct.tensor->nb[1],
                output_bct.tensor->nb[2],
                0)),
        core::TensorShape::from_dims({input.shape.dims[0], out_channels, trimmed_frames}),
        GGML_TYPE_F32);
}

core::TensorValue snake_beta(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    core::ConstantTensorCache & constants,
    const std::vector<float> & alpha,
    const std::vector<float> & beta) {
    std::vector<float> alpha_exp_values(alpha.size());
    std::transform(alpha.begin(), alpha.end(), alpha_exp_values.begin(), [](float value) { return std::exp(value); });
    std::vector<float> inv_beta_exp_values(beta.size());
    std::transform(beta.begin(), beta.end(), inv_beta_exp_values.begin(), [](float value) {
        return 1.0F / (std::exp(value) + 1.0e-9F);
    });
    auto alpha_exp = constants.make_f32(
        core::TensorShape::from_dims({1, static_cast<int64_t>(alpha.size()), 1}),
        alpha_exp_values);
    auto inv_beta_exp = constants.make_f32(
        core::TensorShape::from_dims({1, static_cast<int64_t>(beta.size()), 1}),
        inv_beta_exp_values);
    auto * periodic = ggml_sqr(
        build_ctx.ggml,
        ggml_sin(build_ctx.ggml, ggml_mul(build_ctx.ggml, input.tensor, alpha_exp.tensor)));
    return core::wrap_tensor(
        ggml_add(build_ctx.ggml, input.tensor, ggml_mul(build_ctx.ggml, periodic, inv_beta_exp.tensor)),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue quantizer_decode(
    ggml_context * ctx,
    core::ModuleBuildContext & build_ctx,
    ggml_tensor * codes_t_q_b,
    const BreezeSpeechDecoderWeights & weights) {
    const auto & config = weights.config;
    const int64_t split_dim = config.codebook_dim / 2;
    core::TensorValue semantic_sum;
    for (int64_t group = 0; group < config.num_semantic_quantizers; ++group) {
        auto * code_slice = ggml_view_2d(
            ctx,
            codes_t_q_b,
            codes_t_q_b->ne[0],
            codes_t_q_b->ne[2],
            codes_t_q_b->nb[2],
            static_cast<size_t>(group) * codes_t_q_b->nb[1]);
        auto indices = core::wrap_tensor(
            code_slice,
            core::TensorShape::from_dims({code_slice->ne[1], code_slice->ne[0]}),
            GGML_TYPE_I32);
        indices = core::ensure_backend_addressable_layout(build_ctx, indices);
        auto decoded = modules::CodebookLookupModule({config.codebook_size, split_dim})
                           .build(build_ctx, indices, weights.semantic_codebooks[static_cast<size_t>(group)]);
        semantic_sum = semantic_sum.valid() ? modules::AddModule{}.build(build_ctx, semantic_sum, decoded) : decoded;
    }
    semantic_sum = modules::LinearModule(binding::linear_config(split_dim, config.hidden_size, false))
                       .build(build_ctx, semantic_sum, weights.semantic_output_proj);

    core::TensorValue acoustic_sum;
    for (int64_t group = 0; group < config.num_quantizers - config.num_semantic_quantizers; ++group) {
        const int64_t source_group = config.num_semantic_quantizers + group;
        auto * code_slice = ggml_view_2d(
            ctx,
            codes_t_q_b,
            codes_t_q_b->ne[0],
            codes_t_q_b->ne[2],
            codes_t_q_b->nb[2],
            static_cast<size_t>(source_group) * codes_t_q_b->nb[1]);
        auto indices = core::wrap_tensor(
            code_slice,
            core::TensorShape::from_dims({code_slice->ne[1], code_slice->ne[0]}),
            GGML_TYPE_I32);
        indices = core::ensure_backend_addressable_layout(build_ctx, indices);
        auto decoded = modules::CodebookLookupModule({config.codebook_size, split_dim})
                           .build(build_ctx, indices, weights.acoustic_codebooks[static_cast<size_t>(group)]);
        acoustic_sum = acoustic_sum.valid() ? modules::AddModule{}.build(build_ctx, acoustic_sum, decoded) : decoded;
    }
    acoustic_sum = modules::LinearModule(binding::linear_config(split_dim, config.hidden_size, false))
                       .build(build_ctx, acoustic_sum, weights.acoustic_output_proj);
    return modules::AddModule{}.build(build_ctx, semantic_sum, acoustic_sum);
}

core::TensorValue attention(
    ggml_context * ctx,
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    ggml_tensor * positions,
    const core::TensorValue & attention_mask,
    const modules::AttentionWeights & weights,
    const DecoderConfig & config,
    modules::ScaledDotProductAttentionLowering lowering = modules::ScaledDotProductAttentionLowering::Flash) {
    const int64_t kv_repeat = config.num_heads / config.num_kv_heads;
    auto q_value = modules::LinearModule(binding::linear_config(config.hidden_size, config.num_heads * config.head_dim, false))
                       .build(build_ctx, input, {weights.q_weight, weights.q_bias});
    auto k_value = modules::LinearModule(binding::linear_config(config.hidden_size, config.num_kv_heads * config.head_dim, false))
                       .build(build_ctx, input, {weights.k_weight, weights.k_bias});
    auto v_value = modules::LinearModule(binding::linear_config(config.hidden_size, config.num_kv_heads * config.head_dim, false))
                       .build(build_ctx, input, {weights.v_weight, weights.v_bias});
    auto * q = q_value.tensor;
    auto * k = k_value.tensor;
    auto * v = v_value.tensor;
    const int64_t seq = q->ne[1];
    const int64_t batch = q->ne[2];
    q = ggml_reshape_4d(ctx, q, config.head_dim, config.num_heads, seq, batch);
    k = ggml_reshape_4d(ctx, k, config.head_dim, config.num_kv_heads, seq, batch);
    v = ggml_reshape_4d(ctx, v, config.head_dim, config.num_kv_heads, seq, batch);
    auto position_value = core::wrap_tensor(positions, core::TensorShape::from_dims({seq}), GGML_TYPE_I32);
    q = modules::RoPEModule({
        config.head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.rope_theta,
    }).build(
        build_ctx,
        core::wrap_tensor(q, core::TensorShape::from_dims({batch, seq, config.num_heads, config.head_dim}), GGML_TYPE_F32),
        position_value)
            .tensor;
    k = modules::RoPEModule({
        config.head_dim,
        GGML_ROPE_TYPE_NEOX,
        config.rope_theta,
    }).build(
        build_ctx,
        core::wrap_tensor(k, core::TensorShape::from_dims({batch, seq, config.num_kv_heads, config.head_dim}), GGML_TYPE_F32),
        position_value)
            .tensor;
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(
        build_ctx,
        core::wrap_tensor(q, core::TensorShape::from_dims({batch, seq, config.num_heads, config.head_dim}), GGML_TYPE_F32));
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(
        build_ctx,
        core::wrap_tensor(k, core::TensorShape::from_dims({batch, seq, config.num_kv_heads, config.head_dim}), GGML_TYPE_F32));
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(
        build_ctx,
        core::wrap_tensor(v, core::TensorShape::from_dims({batch, seq, config.num_kv_heads, config.head_dim}), GGML_TYPE_F32));
    if (kv_repeat > 1) {
        std::vector<core::TensorValue> repeated_k;
        std::vector<core::TensorValue> repeated_v;
        repeated_k.reserve(static_cast<size_t>(config.num_heads));
        repeated_v.reserve(static_cast<size_t>(config.num_heads));
        for (int64_t head = 0; head < config.num_kv_heads; ++head) {
            auto one_k = modules::SliceModule({1, head, 1}).build(build_ctx, k_heads);
            auto one_v = modules::SliceModule({1, head, 1}).build(build_ctx, v_heads);
            for (int64_t repeat = 0; repeat < kv_repeat; ++repeat) {
                repeated_k.push_back(one_k);
                repeated_v.push_back(one_v);
            }
        }
        k_heads = repeated_k.front();
        v_heads = repeated_v.front();
        for (size_t index = 1; index < repeated_k.size(); ++index) {
            k_heads = modules::ConcatModule({1}).build(build_ctx, k_heads, repeated_k[index]);
            v_heads = modules::ConcatModule({1}).build(build_ctx, v_heads, repeated_v[index]);
        }
    }
    auto context = modules::ScaledDotProductAttentionModule({
        config.head_dim,
        lowering,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(
        build_ctx,
        q_heads,
        k_heads,
        v_heads,
        attention_mask);
    context = core::ensure_backend_addressable_layout(build_ctx, context);
    return modules::LinearModule(binding::linear_config(config.num_heads * config.head_dim, config.hidden_size, false))
        .build(
            build_ctx,
            core::reshape_tensor(build_ctx, context, core::TensorShape::from_dims({batch, seq, config.num_heads * config.head_dim})),
            {weights.out_weight, weights.out_bias});
}

core::TensorValue convnext(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input_bct,
    const ConvNeXtWeights & weights) {
    const int64_t channels = input_bct.shape.dims[1];
    auto hidden = causal_conv1d(build_ctx, input_bct, weights.dwconv, channels, 7, 1, 1, channels, true);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
    hidden = modules::LayerNormModule({channels, 1.0e-6F, true, true})
                 .build(build_ctx, hidden, weights.norm);
    hidden = modules::LinearModule(binding::linear_config(channels, channels * 4, true))
                 .build(build_ctx, hidden, weights.pwconv1);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(build_ctx, hidden);
    hidden = modules::LinearModule(binding::linear_config(channels * 4, channels, true))
                 .build(build_ctx, hidden, weights.pwconv2);
    hidden = modules::LayerScaleModule{}.build(
        build_ctx,
        hidden,
        weights.gamma);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
    return modules::AddModule{}.build(build_ctx, input_bct, hidden);
}

core::TensorValue residual_unit(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t dilation,
    core::ConstantTensorCache & constants) {
    const int64_t channels = input.shape.dims[1];
    auto hidden = snake_beta(
        build_ctx,
        input,
        constants,
        weights.act1_alpha,
        weights.act1_beta);
    hidden = causal_conv1d(build_ctx, hidden, weights.conv1, channels, 7, 1, dilation, 1, true);
    hidden = snake_beta(
        build_ctx,
        hidden,
        constants,
        weights.act2_alpha,
        weights.act2_beta);
    hidden = causal_conv1d(build_ctx, hidden, weights.conv2, channels, 1, 1, 1, 1, true);
    return modules::AddModule{}.build(build_ctx, input, hidden);
}

std::vector<float> make_mask(int64_t frames, int64_t window) {
    std::vector<float> mask(static_cast<size_t>(frames * frames), kMaskNegInf);
    for (int64_t q = 0; q < frames; ++q) {
        const int64_t min_k = std::max<int64_t>(0, q - window + 1);
        for (int64_t k = min_k; k <= q; ++k) {
            mask[static_cast<size_t>(k + frames * q)] = 0.0F;
        }
    }
    return mask;
}

int graph_node_capacity(const DecoderConfig & config) {
    return static_cast<int>(4096 + config.num_layers * config.num_heads * 16 + config.upsample_rates.size() * 512);
}

}  // namespace

class BreezeSpeechDecoderGraph {
public:
    BreezeSpeechDecoderGraph(
        std::shared_ptr<const BreezeSpeechDecoderWeights> weights,
        int64_t code_frames,
        core::ExecutionContext & execution_context,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes,
        bool allow_flash_attention = true)
        : weights_(std::move(weights)),
          code_frames_(code_frames),
          backend_(execution_context.backend()),
          allow_flash_attention_(allow_flash_attention),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Breeze speech decoder graph requires weights");
        }
        if (code_frames_ <= 0) {
            throw std::runtime_error("Breeze speech decoder graph requires positive frame count");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Breeze speech decoder backend is not initialized");
        }
        const auto & config = weights_->config;
        waveform_frames_ = code_frames_;
        for (const auto factor : config.upsampling_ratios) {
            waveform_frames_ *= factor;
        }
        for (const auto factor : config.upsample_rates) {
            waveform_frames_ *= factor;
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Breeze speech decoder ggml context");
        }

        codes_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_I32, code_frames_, config.num_quantizers, 1);
        ggml_set_input(codes_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, code_frames_);
        ggml_set_input(positions_);
        mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, code_frames_, code_frames_, 1, 1);
        ggml_set_input(mask_);

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "breeze_tts.speech_decoder",
            execution_context.backend_type(),
        };
        constants.begin_graph();
        auto hidden = quantizer_decode(ctx_.get(), build_ctx, codes_, *weights_);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        hidden = causal_conv1d(build_ctx, hidden, weights_->pre_conv, config.latent_dim, 3, 1, 1, 1, true);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        hidden = modules::LinearModule(binding::linear_config(config.latent_dim, config.hidden_size, true))
                     .build(build_ctx, hidden, weights_->transformer_input_proj);
        for (const auto & layer : weights_->transformer_layers) {
            auto attn_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                               .build(build_ctx, hidden, layer.input_norm);
            auto attention_mask = core::wrap_tensor(
                mask_,
                core::TensorShape::from_dims({1, 1, code_frames_, code_frames_}),
                GGML_TYPE_F16);
            auto attn_out = attention(
                ctx_.get(),
                build_ctx,
                attn_in,
                positions_,
                attention_mask,
                layer.attention,
                config,
                allow_flash_attention_ ? modules::ScaledDotProductAttentionLowering::Flash
                                       : modules::ScaledDotProductAttentionLowering::Explicit);
            attn_out = modules::LayerScaleModule{}.build(
                build_ctx,
                attn_out,
                layer.attn_scale);
            hidden = modules::AddModule{}.build(build_ctx, hidden, attn_out);
            auto mlp_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                              .build(build_ctx, hidden, layer.post_norm);
            auto mlp_out = modules::GatedFeedForwardModule({
                config.hidden_size,
                config.intermediate_size,
                false,
                modules::GatedFeedForwardActivation::Silu,
            }).build(build_ctx, mlp_in, layer.mlp);
            mlp_out = modules::LayerScaleModule{}.build(
                build_ctx,
                mlp_out,
                layer.mlp_scale);
            hidden = modules::AddModule{}.build(build_ctx, hidden, mlp_out);
        }
        hidden = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                     .build(build_ctx, hidden, weights_->transformer_norm);
        hidden = modules::LinearModule(binding::linear_config(config.hidden_size, config.latent_dim, true))
                     .build(build_ctx, hidden, weights_->transformer_output_proj);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        for (size_t i = 0; i < weights_->upsample_stages.size(); ++i) {
            const auto & stage = weights_->upsample_stages[i];
            hidden = causal_conv_transpose1d(
                build_ctx,
                hidden,
                stage.upconv,
                config.latent_dim,
                config.upsampling_ratios[i],
                config.upsampling_ratios[i],
                true);
            hidden = convnext(build_ctx, hidden, stage.convnext);
        }
        hidden = causal_conv1d(build_ctx, hidden, weights_->decoder_input_conv, config.decoder_dim, 7, 1, 1, 1, true);
        int64_t decoder_channels = config.decoder_dim;
        for (size_t block_index = 0; block_index < weights_->decoder_blocks.size(); ++block_index) {
            const auto & block = weights_->decoder_blocks[block_index];
            const int64_t out_channels = decoder_channels / 2;
            hidden = snake_beta(
                build_ctx,
                hidden,
                constants,
                block.input_alpha,
                block.input_beta);
            hidden = causal_conv_transpose1d(
                build_ctx,
                hidden,
                block.upconv,
                out_channels,
                config.upsample_rates[block_index] * 2,
                config.upsample_rates[block_index],
                true);
            for (size_t unit_index = 0; unit_index < block.residual_units.size(); ++unit_index) {
                const int64_t dilation = unit_index == 0 ? 1 : unit_index == 1 ? 3 : 9;
                hidden = residual_unit(
                    build_ctx,
                    hidden,
                    block.residual_units[unit_index],
                    dilation,
                    constants);
            }
            decoder_channels = out_channels;
        }
        hidden = snake_beta(
            build_ctx,
            hidden,
            constants,
            weights_->output_alpha,
            weights_->output_beta);
        output_ = ggml_clamp(ctx_.get(), causal_conv1d(build_ctx, hidden, weights_->output_conv, 1, 7, 1, 1, 1, true).tensor, -1.0F, 1.0F);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), graph_node_capacity(config), false);
        ggml_build_forward_expand(graph_, output_);
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Breeze speech decoder graph");
        }
        positions_data_.resize(static_cast<size_t>(code_frames_));
        for (int64_t i = 0; i < code_frames_; ++i) {
            positions_data_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        const auto mask = make_mask(code_frames_, config.sliding_window);
        mask_f16_data_.resize(mask.size());
        for (size_t index = 0; index < mask.size(); ++index) {
            mask_f16_data_[index] = ggml_fp32_to_fp16(mask[index]);
        }
        upload_static_inputs();
    }

    ~BreezeSpeechDecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        const BreezeSpeechDecoderWeights & weights,
        int64_t code_frames,
        ggml_backend_t backend,
        int threads) const {
        const bool frame_match = code_frames_ == code_frames;
        return weights_.get() == &weights && frame_match && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const int32_t * codes, size_t code_count) {
        const int64_t input_frames = static_cast<int64_t>(code_count / weights_->config.num_quantizers);
        const size_t expected = static_cast<size_t>(code_frames_ * weights_->config.num_quantizers);
        if (code_count % static_cast<size_t>(weights_->config.num_quantizers) != 0 || input_frames <= 0 || input_frames > code_frames_) {
            throw std::runtime_error("Breeze speech decoder code count exceeds graph capacity");
        }
        // Cached GGML graphs may reuse backend allocations whose input contents are
        // not guaranteed to survive a prior execution. Restore every declared input,
        // not only the request-varying codes, before replaying a retained graph.
        upload_static_inputs();
        std::vector<int32_t> tensor_codes(expected, 0);
        for (int64_t frame = 0; frame < input_frames; ++frame) {
            for (int64_t group = 0; group < weights_->config.num_quantizers; ++group) {
                tensor_codes[static_cast<size_t>(frame + code_frames_ * group)] =
                    codes[static_cast<size_t>(frame * weights_->config.num_quantizers + group)];
            }
        }
        ggml_backend_tensor_set(codes_, tensor_codes.data(), 0, tensor_codes.size() * sizeof(int32_t));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Breeze speech decoder graph compute failed");
        }
        std::vector<float> audio(static_cast<size_t>(waveform_frames_), 0.0F);
        ggml_backend_tensor_get(output_, audio.data(), 0, audio.size() * sizeof(float));
        return audio;
    }

private:
    void upload_static_inputs() {
        ggml_backend_tensor_set(
            positions_,
            positions_data_.data(),
            0,
            positions_data_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            mask_,
            mask_f16_data_.data(),
            0,
            mask_f16_data_.size() * sizeof(ggml_fp16_t));
    }

    std::shared_ptr<const BreezeSpeechDecoderWeights> weights_;
    int64_t code_frames_ = 0;
    int64_t waveform_frames_ = 0;
    ggml_backend_t backend_ = nullptr;
    bool allow_flash_attention_ = true;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * codes_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_data_;
    std::vector<ggml_fp16_t> mask_f16_data_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

BreezeSpeechDecoderRuntime::BreezeSpeechDecoderRuntime(
    std::shared_ptr<const BreezeTTSAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type,
    core::AttentionPreference attention_preference)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("BreezeTTS speech decoder requires assets");
    }
    weights_ = load_weights(
        *assets_,
        execution_context_->backend(),
        execution_context_->backend_type(),
        linear_weight_storage_type,
        conv_weight_storage_type);
    allow_flash_attention_ = core::resolve_flash_attention(
        execution_context_->backend(), weights_->config.head_dim, attention_preference);
    engine::debug::trace_log_scalar("breeze_tts.attention.allow_decoder_flash", allow_flash_attention_);
    constants_ = std::make_unique<core::ConstantTensorCache>(
        execution_context_->backend(),
        std::max(1, execution_context_->config().threads),
        "breeze_tts.speech_decoder.constants",
        constant_context_bytes);
}

BreezeSpeechDecoderRuntime::~BreezeSpeechDecoderRuntime() = default;

std::vector<float> BreezeSpeechDecoderRuntime::decode_window_samples(
    const std::vector<int32_t> & chunk,
    int64_t chunk_frames,
    int64_t context_frames) const {
    if (chunk_frames <= 0 || context_frames < 0 || context_frames > chunk_frames) {
        throw std::runtime_error("Breeze speech decoder received invalid chunk shape");
    }
    if (static_cast<int64_t>(chunk.size()) != chunk_frames * weights_->config.num_quantizers) {
        throw std::runtime_error("Breeze speech decoder chunk payload size mismatch");
    }
    const int threads = std::max(1, execution_context_->config().threads);
    auto * graph_slot = &graph_;
#if defined(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS)
    const bool optimized_cache_enabled =
        kStrixHaloGraphCacheEnabled && execution_context_->backend_type() == core::BackendType::Hip;
    if (optimized_cache_enabled) {
        for (size_t index = 0; index < kStrixHaloCachedChunkFrames.size(); ++index) {
            if (chunk_frames == kStrixHaloCachedChunkFrames[index]) {
                graph_slot = &optimized_graphs_[index];
                break;
            }
        }
    }
#endif
    auto & graph = *graph_slot;
    const bool graph_rebuilt =
        graph == nullptr || !graph->matches(*weights_, chunk_frames, execution_context_->backend(), threads);
    if (graph_rebuilt) {
        graph.reset();
        graph = std::make_unique<BreezeSpeechDecoderGraph>(
            weights_,
            chunk_frames,
            *execution_context_,
            *constants_,
            graph_arena_bytes_,
            allow_flash_attention_);
    }
    auto decoded = graph->run(chunk.data(), chunk.size());
    const int64_t drop = context_frames * kDecodeSamplesPerCode;
    if (drop > static_cast<int64_t>(decoded.size())) {
        throw std::runtime_error("Breeze speech decoder chunk context exceeds decoded waveform");
    }
    const int64_t valid_samples = chunk_frames * kDecodeSamplesPerCode;
    if (valid_samples < drop || valid_samples > static_cast<int64_t>(decoded.size())) {
        throw std::runtime_error("Breeze speech decoder valid sample range exceeds decoded waveform");
    }
    return std::vector<float>(
        decoded.begin() + static_cast<std::ptrdiff_t>(drop),
        decoded.begin() + static_cast<std::ptrdiff_t>(valid_samples));
}

runtime::AudioBuffer BreezeSpeechDecoderRuntime::decode(const BreezeSpeechCodes & codec_codes) const {
    const auto total_start = Clock::now();
    if (codec_codes.frames <= 0 || codec_codes.code_groups != weights_->config.num_quantizers) {
        throw std::runtime_error("Breeze speech decoder received invalid codec shape");
    }
    if (static_cast<int64_t>(codec_codes.codes.size()) != codec_codes.frames * codec_codes.code_groups) {
        throw std::runtime_error("Breeze speech decoder codec payload size mismatch");
    }
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(codec_codes.frames * kDecodeSamplesPerCode));
    for (int64_t start = 0; start < codec_codes.frames; start += kChunkCodes) {
        const int64_t end = std::min<int64_t>(start + kChunkCodes, codec_codes.frames);
        const int64_t context = start > kLeftContextCodes ? kLeftContextCodes : start;
        const int64_t chunk_start = start - context;
        const int64_t chunk_frames = end - chunk_start;
        std::vector<int32_t> chunk(static_cast<size_t>(chunk_frames * codec_codes.code_groups), 0);
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            const int64_t src_frame = chunk_start + frame;
            const auto src = codec_codes.codes.begin() + static_cast<std::ptrdiff_t>(src_frame * codec_codes.code_groups);
            const auto dst = chunk.begin() + static_cast<std::ptrdiff_t>(frame * codec_codes.code_groups);
            std::copy(src, src + codec_codes.code_groups, dst);
        }
        auto decoded = decode_window_samples(chunk, chunk_frames, context);
        samples.insert(samples.end(), decoded.begin(), decoded.end());
    }
    debug::timing_log_scalar("breeze_tts.speech_decoder.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return runtime::AudioBuffer{kSampleRate, 1, std::move(samples)};
}

runtime::AudioBuffer BreezeSpeechDecoderRuntime::decode_stream_chunk(
    const BreezeSpeechCodes & codec_codes,
    int64_t context_frames,
    int64_t new_frames,
    int64_t frames_per_chunk) const {
    const auto total_start = Clock::now();
    if (frames_per_chunk <= 0 || context_frames < 0 || context_frames > kLeftContextFrames ||
        new_frames <= 0 || context_frames + new_frames > codec_codes.frames ||
        codec_codes.frames - context_frames > frames_per_chunk ||
        codec_codes.code_groups != weights_->config.num_quantizers) {
        throw std::runtime_error("Breeze speech decoder received invalid streaming codec shape");
    }
    if (static_cast<int64_t>(codec_codes.codes.size()) != codec_codes.frames * codec_codes.code_groups) {
        throw std::runtime_error("Breeze speech decoder streaming codec payload size mismatch");
    }
    const bool initial = context_frames == 0;
    const int64_t capacity = initial ? codec_codes.frames : kLeftContextFrames + frames_per_chunk;
    auto & graph = initial ? stream_initial_graph_ : stream_graph_;
    const int threads = std::max(1, execution_context_->config().threads);
    if (graph == nullptr || !graph->matches(*weights_, capacity, execution_context_->backend(), threads)) {
        graph.reset();
        graph = std::make_unique<BreezeSpeechDecoderGraph>(
            weights_, capacity, *execution_context_, *constants_, graph_arena_bytes_, allow_flash_attention_);
    }
    // Causal attention and convolutions make right padding independent of valid samples.
    auto samples = graph->run(codec_codes.codes.data(), codec_codes.codes.size());
    samples.resize(static_cast<size_t>((context_frames + new_frames) * kDecodeSamplesPerCode));
    samples.erase(samples.begin(), samples.begin() + context_frames * kDecodeSamplesPerCode);
    engine::debug::timing_log_scalar("breeze_tts.speech_decoder.streaming_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return runtime::AudioBuffer{kSampleRate, 1, std::move(samples)};
}

runtime::AudioBuffer BreezeSpeechDecoderRuntime::decode_and_trim_reference(
    const BreezeSpeechCodes & reference_codes,
    const BreezeSpeechCodes & generated_codes) const {
    if (reference_codes.code_groups != generated_codes.code_groups) {
        throw std::runtime_error("Breeze speech decoder reference/generated code group mismatch");
    }
    if (reference_codes.frames < 0 || generated_codes.frames < 0 || reference_codes.code_groups <= 0) {
        throw std::runtime_error("Breeze speech decoder reference/generated code shape is invalid");
    }
    if (reference_codes.frames > std::numeric_limits<int64_t>::max() - generated_codes.frames) {
        throw std::runtime_error("Breeze speech decoder combined frame count is too large");
    }
    BreezeSpeechCodes combined;
    combined.frames = reference_codes.frames + generated_codes.frames;
    combined.code_groups = reference_codes.code_groups;
    if (combined.frames > std::numeric_limits<int64_t>::max() / combined.code_groups) {
        throw std::runtime_error("Breeze speech decoder combined code count is too large");
    }
    const int64_t combined_code_count = combined.frames * combined.code_groups;
    if (static_cast<uint64_t>(combined_code_count) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Breeze speech decoder combined code count exceeds host size limits");
    }
    combined.codes.reserve(static_cast<size_t>(combined_code_count));
    combined.codes.insert(combined.codes.end(), reference_codes.codes.begin(), reference_codes.codes.end());
    combined.codes.insert(combined.codes.end(), generated_codes.codes.begin(), generated_codes.codes.end());
    auto audio = decode(combined);
    if (reference_codes.frames > std::numeric_limits<int64_t>::max() / kDecodeSamplesPerCode) {
        throw std::runtime_error("Breeze speech decoder reference sample count is too large");
    }
    const int64_t cut = reference_codes.frames * kDecodeSamplesPerCode;
    if (static_cast<uint64_t>(cut) > audio.samples.size()) {
        throw std::runtime_error("Breeze speech decoder reference trim is out of range");
    }
    audio.samples.erase(audio.samples.begin(), audio.samples.begin() + static_cast<std::ptrdiff_t>(cut));
    return audio;
}

void BreezeSpeechDecoderRuntime::release_runtime_graphs() const {
    graph_.reset();
    stream_initial_graph_.reset();
    stream_graph_.reset();
    for (auto & graph : optimized_graphs_) {
        graph.reset();
    }
}

}  // namespace engine::models::breeze_tts

#include "engine/models/sheetsage/runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::sheetsage {
namespace {

namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct SheetSage2AttentionWeights {
    modules::LinearWeights q_proj;
    modules::LinearWeights k_proj;
    modules::LinearWeights v_proj;
    modules::LinearWeights out_proj;
};

struct SheetSage2DecoderLayerWeights {
    SheetSage2AttentionWeights self_attn;
    modules::NormWeights self_attn_layer_norm;
    SheetSage2AttentionWeights encoder_attn;
    modules::NormWeights encoder_attn_layer_norm;
    modules::LinearWeights fc1;
    modules::LinearWeights fc2;
    modules::NormWeights final_layer_norm;
};

struct SheetSage2DecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    core::TensorValue position_embedding;
    modules::NormWeights layernorm_embedding;
    modules::LinearWeights encoder_projection;
    std::vector<SheetSage2DecoderLayerWeights> layers;
};

struct SheetSage2ConvNextLayerWeights {
    modules::DepthwiseConv1dWeights depthwise;
    modules::NormWeights norm;
    modules::LinearWeights up;
    modules::NormWeights grn;
    modules::LinearWeights down;
};

struct SheetSage2SubsamplingBlockWeights {
    std::optional<modules::NormWeights> resample_norm;
    std::optional<modules::Conv1dWeights> resample_conv;
    std::vector<SheetSage2ConvNextLayerWeights> layers;
};

struct SheetSage2EncoderLayerWeights {
    modules::NormWeights ffn1_norm;
    modules::LinearWeights ffn1_w1;
    modules::LinearWeights ffn1_w2;
    modules::NormWeights attn_norm;
    SheetSage2AttentionWeights attn;
    modules::NormWeights conv_norm;
    modules::Conv1dWeights conv_pw_in;
    modules::DepthwiseConv1dWeights conv_depthwise;
    modules::NormWeights conv_depthwise_norm;
    modules::Conv1dWeights conv_pw_out;
    modules::NormWeights ffn2_norm;
    modules::LinearWeights ffn2_w1;
    modules::LinearWeights ffn2_w2;
    modules::NormWeights final_norm;
};

struct SheetSage2EncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue mel_mean;
    core::TensorValue mel_std;
    core::TensorValue half;
    std::vector<core::TensorValue> layer_weights;
    std::vector<float> mel_mean_host;
    std::vector<float> mel_std_host;
    std::vector<float> layer_weight_host;
    std::vector<SheetSage2SubsamplingBlockWeights> subsampling;
    std::vector<SheetSage2EncoderLayerWeights> layers;
};

void validate_config(const SheetSage2DecoderConfig & config) {
    if (config.vocab_size <= 0 || config.hidden_size <= 0 || config.encoder_hidden_size <= 0 ||
        config.intermediate_size <= 0 || config.decoder_layers <= 0 ||
        config.num_attention_heads <= 0 || config.max_position_embeddings <= 0) {
        throw std::runtime_error("SheetSage2 decoder config dimensions must be positive");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("SheetSage2 decoder hidden size must be divisible by head count");
    }
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden}),
    };
}

SheetSage2AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t hidden) {
    return {
        load_linear(store, source, prefix + ".q_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".k_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".v_proj", storage_type, hidden, hidden, true),
        load_linear(store, source, prefix + ".out_proj", storage_type, hidden, hidden, true),
    };
}

SheetSage2DecoderWeights load_weights(
    const assets::TensorSource & source,
    const SheetSage2DecoderConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const SheetSage2DecoderRuntimeOptions & options) {
    SheetSage2DecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "models.sheetsage2.decoder.weights",
        options.weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "token_embedding.weight",
        options.weight_storage_type,
        {config.vocab_size, config.hidden_size});
    weights.position_embedding = weights.store->load_tensor(
        source,
        "decoder.embed_positions.weight",
        options.weight_storage_type,
        {config.max_position_embeddings + 2, config.hidden_size});
    weights.layernorm_embedding = load_norm(*weights.store, source, "decoder.layernorm_embedding", config.hidden_size);
    weights.encoder_projection = load_linear(
        *weights.store,
        source,
        "encoder_projection",
        options.weight_storage_type,
        config.hidden_size,
        config.encoder_hidden_size,
        true);
    weights.layers.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t i = 0; i < config.decoder_layers; ++i) {
        const std::string prefix = "decoder.layers." + std::to_string(i);
        SheetSage2DecoderLayerWeights layer;
        layer.self_attn = load_attention(*weights.store, source, prefix + ".self_attn", options.weight_storage_type, config.hidden_size);
        layer.self_attn_layer_norm = load_norm(*weights.store, source, prefix + ".self_attn_layer_norm", config.hidden_size);
        layer.encoder_attn = load_attention(*weights.store, source, prefix + ".encoder_attn", options.weight_storage_type, config.hidden_size);
        layer.encoder_attn_layer_norm = load_norm(*weights.store, source, prefix + ".encoder_attn_layer_norm", config.hidden_size);
        layer.fc1 = load_linear(*weights.store, source, prefix + ".fc1", options.weight_storage_type, config.intermediate_size, config.hidden_size, true);
        layer.fc2 = load_linear(*weights.store, source, prefix + ".fc2", options.weight_storage_type, config.hidden_size, config.intermediate_size, true);
        layer.final_layer_norm = load_norm(*weights.store, source, prefix + ".final_layer_norm", config.hidden_size);
        weights.layers.push_back(std::move(layer));
    }
    weights.store->upload();
    return weights;
}

modules::Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    bool use_bias) {
    modules::Conv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::DepthwiseConv1dWeights load_depthwise_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t kernel,
    bool use_bias) {
    modules::DepthwiseConv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {channels, 1, kernel});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {channels});
    }
    return weights;
}

SheetSage2EncoderWeights load_encoder_weights(
    const assets::TensorSource & source,
    const SheetSage2DecoderConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const SheetSage2DecoderRuntimeOptions & options) {
    SheetSage2EncoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "models.sheetsage2.encoder.weights",
        options.weight_context_bytes * 3);
    weights.mel_mean_host = source.require_f32("feature_extractor.mel_mean", {config.mel_bins});
    weights.mel_std_host = source.require_f32("feature_extractor.mel_std", {config.mel_bins});
    weights.layer_weight_host = source.require_f32("layer_weight", {config.encoder_layers + 1});
    float max_weight = *std::max_element(weights.layer_weight_host.begin(), weights.layer_weight_host.end());
    float sum = 0.0F;
    for (float & value : weights.layer_weight_host) {
        value = std::exp(value - max_weight);
        sum += value;
    }
    for (float & value : weights.layer_weight_host) {
        value /= sum;
    }
    weights.mel_mean = weights.store->make_f32(core::TensorShape::from_dims({1, 1, config.mel_bins}), weights.mel_mean_host);
    weights.mel_std = weights.store->make_f32(core::TensorShape::from_dims({1, 1, config.mel_bins}), weights.mel_std_host);
    weights.half = weights.store->make_f32(core::TensorShape::from_dims({1, 1, 1}), {0.5F});
    weights.layer_weights.reserve(weights.layer_weight_host.size());
    for (const float value : weights.layer_weight_host) {
        weights.layer_weights.push_back(weights.store->make_f32(core::TensorShape::from_dims({1, 1, 1}), {value}));
    }

    const int64_t block_channels[] = {128, 512, 1024};
    const int64_t block_inputs[] = {128, 128, 512};
    const int64_t block_depths[] = {3, 4, 5};
    const int64_t block_strides[] = {1, 2, 2};
    weights.subsampling.reserve(3);
    for (int64_t block = 0; block < 3; ++block) {
        SheetSage2SubsamplingBlockWeights out;
        const std::string prefix = "subsampling_module." + std::to_string(block);
        if (block_inputs[block] != block_channels[block] || block_strides[block] > 1) {
            out.resample_norm = load_norm(*weights.store, source, prefix + ".resampling_layer.0", block_inputs[block]);
            out.resample_conv = load_conv1d(
                *weights.store,
                source,
                prefix + ".resampling_layer.2",
                options.weight_storage_type,
                block_channels[block],
                block_inputs[block],
                2,
                true);
        }
        out.layers.reserve(static_cast<size_t>(block_depths[block]));
        for (int64_t layer = 0; layer < block_depths[block]; ++layer) {
            const std::string layer_prefix = prefix + ".convnext_layers." + std::to_string(layer);
            SheetSage2ConvNextLayerWeights cw;
            cw.depthwise = load_depthwise_conv1d(
                *weights.store,
                source,
                layer_prefix + ".depthwise_block.1",
                options.weight_storage_type,
                block_channels[block],
                config.convnext_kernel_size,
                true);
            cw.norm = load_norm(*weights.store, source, layer_prefix + ".pointwise_block.0", block_channels[block]);
            cw.up = load_linear(
                *weights.store,
                source,
                layer_prefix + ".pointwise_block.1",
                options.weight_storage_type,
                block_channels[block] * 4,
                block_channels[block],
                true);
            cw.grn = {
                weights.store->load_f32_tensor(source, layer_prefix + ".pointwise_block.3.weight", {1, 1, block_channels[block] * 4}),
                weights.store->load_f32_tensor(source, layer_prefix + ".pointwise_block.3.bias", {1, 1, block_channels[block] * 4}),
            };
            cw.down = load_linear(
                *weights.store,
                source,
                layer_prefix + ".pointwise_block.4",
                options.weight_storage_type,
                block_channels[block],
                block_channels[block] * 4,
                true);
            out.layers.push_back(std::move(cw));
        }
        weights.subsampling.push_back(std::move(out));
    }
    weights.layers.reserve(static_cast<size_t>(config.encoder_layers));
    for (int64_t i = 0; i < config.encoder_layers; ++i) {
        const std::string prefix = "layers." + std::to_string(i);
        SheetSage2EncoderLayerWeights layer;
        layer.ffn1_norm = load_norm(*weights.store, source, prefix + ".ffn1_layer_norm", config.encoder_hidden_size);
        layer.ffn1_w1 = load_linear(*weights.store, source, prefix + ".ffn1.w_1", options.weight_storage_type, config.encoder_intermediate_size, config.encoder_hidden_size, true);
        layer.ffn1_w2 = load_linear(*weights.store, source, prefix + ".ffn1.w_2", options.weight_storage_type, config.encoder_hidden_size, config.encoder_intermediate_size, true);
        layer.attn_norm = load_norm(*weights.store, source, prefix + ".attn_layer_norm", config.encoder_hidden_size);
        layer.attn = {
            load_linear(*weights.store, source, prefix + ".attn.query_proj", options.weight_storage_type, config.encoder_hidden_size, config.encoder_hidden_size, true),
            load_linear(*weights.store, source, prefix + ".attn.key_proj", options.weight_storage_type, config.encoder_hidden_size, config.encoder_hidden_size, true),
            load_linear(*weights.store, source, prefix + ".attn.value_proj", options.weight_storage_type, config.encoder_hidden_size, config.encoder_hidden_size, true),
            load_linear(*weights.store, source, prefix + ".attn.out_proj", options.weight_storage_type, config.encoder_hidden_size, config.encoder_hidden_size, true),
        };
        layer.conv_norm = load_norm(*weights.store, source, prefix + ".conv_module.layer_norm", config.encoder_hidden_size);
        layer.conv_pw_in = load_conv1d(*weights.store, source, prefix + ".conv_module.conv_block.1", options.weight_storage_type, config.encoder_hidden_size * 2, config.encoder_hidden_size, 1, false);
        layer.conv_depthwise = load_depthwise_conv1d(*weights.store, source, prefix + ".conv_module.conv_block.3", options.weight_storage_type, config.encoder_hidden_size, config.conformer_conv_kernel_size, false);
        layer.conv_depthwise_norm = load_norm(*weights.store, source, prefix + ".conv_module.conv_block.4.1", config.encoder_hidden_size);
        layer.conv_pw_out = load_conv1d(*weights.store, source, prefix + ".conv_module.conv_block.6", options.weight_storage_type, config.encoder_hidden_size, config.encoder_hidden_size, 1, false);
        layer.ffn2_norm = load_norm(*weights.store, source, prefix + ".ffn2_layer_norm", config.encoder_hidden_size);
        layer.ffn2_w1 = load_linear(*weights.store, source, prefix + ".ffn2.w_1", options.weight_storage_type, config.encoder_intermediate_size, config.encoder_hidden_size, true);
        layer.ffn2_w2 = load_linear(*weights.store, source, prefix + ".ffn2.w_2", options.weight_storage_type, config.encoder_hidden_size, config.encoder_intermediate_size, true);
        layer.final_norm = load_norm(*weights.store, source, prefix + ".final_layer_norm", config.encoder_hidden_size);
        weights.layers.push_back(std::move(layer));
    }
    weights.store->upload();
    return weights;
}

core::TensorValue split_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    auto shaped = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
    return modules::TransposeModule({{0, 2, 1, 3}, shaped.shape.rank}).build(ctx, shaped);
}

core::TensorValue merge_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t hidden) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}));
}


core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & key_value,
    const SheetSage2AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads,
    bool causal) {
    const int64_t head_dim = hidden_size / heads;
    auto q = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.q_proj);
    auto k = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, key_value, weights.k_proj);
    auto v = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, key_value, weights.v_proj);
    q = split_heads(ctx, q, heads, head_dim);
    k = split_heads(ctx, k, heads, head_dim);
    v = split_heads(ctx, v, heads, head_dim);
    auto context = modules::ScaledDotProductAttentionModule({
        head_dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        causal ? modules::AttentionCausality::Causal : modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v);
    context = merge_heads(ctx, context, hidden_size);
    return modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, context, weights.out_proj);
}

struct CachedSelfAttentionOutput {
    core::TensorValue output;
    core::TensorValue key_store;
    core::TensorValue value_store;
};

CachedSelfAttentionOutput cached_self_attention_step(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & cached_key_steps,
    const core::TensorValue & cached_value_steps,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    const SheetSage2AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads) {
    const int64_t head_dim = hidden_size / heads;
    auto q = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.q_proj);
    auto k = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.k_proj);
    auto v = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.v_proj);
    q = split_heads(ctx, q, heads, head_dim);
    k = split_heads(ctx, k, heads, head_dim);
    v = split_heads(ctx, v, heads, head_dim);
    auto k_store = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_store = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::BackendViewOptimized});
    auto updated_key_cache = set_rows.build(ctx, cached_key_steps, k_store, cache_slot);
    auto updated_value_cache = set_rows.build(ctx, cached_value_steps, v_store, cache_slot);
    auto all_k = modules::TransposeModule({{0, 2, 1, 3}, updated_key_cache.shape.rank}).build(ctx, updated_key_cache);
    auto all_v = modules::TransposeModule({{0, 2, 1, 3}, updated_value_cache.shape.rank}).build(ctx, updated_value_cache);
    auto context = modules::ScaledDotProductAttentionModule({
        head_dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, all_k, all_v, attention_mask);
    context = merge_heads(ctx, context, hidden_size);
    return {
        modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, context, weights.out_proj),
        k_store,
        v_store,
    };
}

struct CrossAttentionKeyValue {
    core::TensorValue key;
    core::TensorValue value;
};

CrossAttentionKeyValue cross_attention_key_value(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & memory,
    const SheetSage2AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads) {
    const int64_t head_dim = hidden_size / heads;
    auto k = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, memory, weights.k_proj);
    auto v = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, memory, weights.v_proj);
    k = split_heads(ctx, k, heads, head_dim);
    v = split_heads(ctx, v, heads, head_dim);
    return {k, v};
}

core::TensorValue cached_cross_attention_step(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const CrossAttentionKeyValue & key_value,
    const SheetSage2AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads) {
    const int64_t head_dim = hidden_size / heads;
    auto q = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, hidden, weights.q_proj);
    q = split_heads(ctx, q, heads, head_dim);
    auto context = modules::ScaledDotProductAttentionModule({
        head_dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, key_value.key, key_value.value);
    context = merge_heads(ctx, context, hidden_size);
    return modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, context, weights.out_proj);
}

core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & memory,
    const SheetSage2DecoderLayerWeights & weights,
    const SheetSage2DecoderConfig & config) {
    auto hidden = modules::ResidualAddModule{}.build(
        ctx,
        attention(ctx, input, input, weights.self_attn, config.hidden_size, config.num_attention_heads, true),
        input);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.self_attn_layer_norm);
    auto cross = attention(ctx, hidden, memory, weights.encoder_attn, config.hidden_size, config.num_attention_heads, false);
    hidden = modules::ResidualAddModule{}.build(ctx, cross, hidden);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.encoder_attn_layer_norm);
    auto ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true}).build(ctx, hidden, weights.fc1);
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true}).build(ctx, ff, weights.fc2);
    hidden = modules::ResidualAddModule{}.build(ctx, ff, hidden);
    return modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.final_layer_norm);
}

struct CachedDecoderLayerOutput {
    core::TensorValue hidden;
    core::TensorValue key_store;
    core::TensorValue value_store;
};

CachedDecoderLayerOutput cached_decoder_layer_step(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CrossAttentionKeyValue & cross_key_value,
    const core::TensorValue & cached_key_steps,
    const core::TensorValue & cached_value_steps,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask,
    const SheetSage2DecoderLayerWeights & weights,
    const SheetSage2DecoderConfig & config) {
    auto self = cached_self_attention_step(
        ctx,
        input,
        cached_key_steps,
        cached_value_steps,
        cache_slot,
        attention_mask,
        weights.self_attn,
        config.hidden_size,
        config.num_attention_heads);
    auto hidden = modules::ResidualAddModule{}.build(ctx, self.output, input);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.self_attn_layer_norm);
    auto cross = cached_cross_attention_step(
        ctx,
        hidden,
        cross_key_value,
        weights.encoder_attn,
        config.hidden_size,
        config.num_attention_heads);
    hidden = modules::ResidualAddModule{}.build(ctx, cross, hidden);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.encoder_attn_layer_norm);
    auto ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true}).build(ctx, hidden, weights.fc1);
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true}).build(ctx, ff, weights.fc2);
    hidden = modules::ResidualAddModule{}.build(ctx, ff, hidden);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true}).build(
        ctx,
        hidden,
        weights.final_layer_norm);
    return {hidden, self.key_store, self.value_store};
}

core::TensorValue global_response_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & eps,
    const modules::NormWeights & weights) {
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({1}).build(ctx, squared);
    auto magnitude = core::wrap_tensor(ggml_sqrt(ctx.ggml, sum.tensor), sum.shape, GGML_TYPE_F32);
    auto mean = modules::ReduceMeanModule({2}).build(ctx, magnitude);
    auto denom = modules::AddModule().build(ctx, mean, eps);
    auto denom_full = modules::RepeatModule({magnitude.shape}).build(ctx, denom);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, magnitude.tensor, denom_full.tensor), magnitude.shape, GGML_TYPE_F32);
    auto normalized_full = modules::RepeatModule({input.shape}).build(ctx, normalized);
    auto scaled = modules::MulModule().build(ctx, input, normalized_full);
    auto weight = modules::RepeatModule({input.shape}).build(ctx, *weights.weight);
    auto bias = modules::RepeatModule({input.shape}).build(ctx, *weights.bias);
    auto out = modules::MulModule().build(ctx, scaled, weight);
    out = modules::AddModule().build(ctx, out, bias);
    return modules::AddModule().build(ctx, out, input);
}

core::TensorValue convnext_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & grn_eps,
    const SheetSage2ConvNextLayerWeights & weights,
    int64_t channels,
    float eps) {
    auto bct = modules::TransposeModule({{0, 2, 1, 3}, input.shape.rank}).build(ctx, input);
    bct = modules::DepthwiseConv1dModule({channels, 7, 1, 3, 1, true}).build(ctx, bct, weights.depthwise);
    auto hidden = modules::TransposeModule({{0, 2, 1, 3}, bct.shape.rank}).build(ctx, bct);
    hidden = modules::LayerNormModule({channels, eps, true, true}).build(ctx, hidden, weights.norm);
    hidden = modules::LinearModule({channels, channels * 4, true}).build(ctx, hidden, weights.up);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = global_response_norm(ctx, hidden, grn_eps, weights.grn);
    hidden = modules::LinearModule({channels * 4, channels, true}).build(ctx, hidden, weights.down);
    return modules::ResidualAddModule().build(ctx, hidden, input);
}

core::TensorValue subsampling_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & grn_eps,
    const SheetSage2SubsamplingBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int stride,
    float eps) {
    auto hidden = input;
    if (weights.resample_norm.has_value() && weights.resample_conv.has_value()) {
        hidden = modules::LayerNormModule({in_channels, eps, true, true}).build(ctx, hidden, *weights.resample_norm);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
        hidden = modules::Conv1dModule({in_channels, out_channels, 2, stride, 0, 1, true}).build(ctx, hidden, *weights.resample_conv);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    }
    for (const auto & layer : weights.layers) {
        hidden = convnext_layer(ctx, hidden, grn_eps, layer, out_channels, eps);
    }
    return hidden;
}

core::TensorValue encoder_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const SheetSage2AttentionWeights & weights,
    const SheetSage2DecoderConfig & config,
    int64_t query_tile_rows) {
    const int64_t head_dim = config.encoder_hidden_size / config.encoder_attention_heads;
    auto q = modules::LinearModule({config.encoder_hidden_size, config.encoder_hidden_size, true}).build(ctx, hidden, weights.q_proj);
    auto k = modules::LinearModule({config.encoder_hidden_size, config.encoder_hidden_size, true}).build(ctx, hidden, weights.k_proj);
    auto v = modules::LinearModule({config.encoder_hidden_size, config.encoder_hidden_size, true}).build(ctx, hidden, weights.v_proj);
    q = split_heads(ctx, q, config.encoder_attention_heads, head_dim);
    k = split_heads(ctx, k, config.encoder_attention_heads, head_dim);
    v = split_heads(ctx, v, config.encoder_attention_heads, head_dim);
    q = modules::SplitRoPEModule({head_dim}).build(ctx, q, cos, sin);
    k = modules::SplitRoPEModule({head_dim}).build(ctx, k, cos, sin);
    const modules::ScaledDotProductAttentionModule attention({
        head_dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    });
    core::TensorValue context;
    // The padded 300-second window has 7,500 positions: full FP32 attention
    // scores occupy 3.6 GB, exceeding some Vulkan devices' per-buffer limit.
    // Explicit tiling splits only query rows on any backend; every row still
    // attends to all keys and values. Zero retains the original full attention.
    if (query_tile_rows > 0 && q.shape.dims[2] > query_tile_rows) {
        for (int64_t first = 0; first < q.shape.dims[2]; first += query_tile_rows) {
            const int64_t rows = std::min(query_tile_rows, q.shape.dims[2] - first);
            auto query = modules::SliceModule({2, first, rows}).build(ctx, q);
            query = core::ensure_backend_addressable_layout(ctx, query);
            auto tile = attention.build(ctx, query, k, v);
            context = context.valid() ? modules::ConcatModule({1}).build(ctx, context, tile) : tile;
        }
    } else {
        context = attention.build(ctx, q, k, v);
    }
    context = merge_heads(ctx, context, config.encoder_hidden_size);
    return modules::LinearModule({config.encoder_hidden_size, config.encoder_hidden_size, true}).build(ctx, context, weights.out_proj);
}

core::TensorValue feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const modules::LinearWeights & w1,
    const modules::LinearWeights & w2,
    const SheetSage2DecoderConfig & config) {
    auto out = modules::LinearModule({config.encoder_hidden_size, config.encoder_intermediate_size, true}).build(ctx, hidden, w1);
    out = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, out);
    return modules::LinearModule({config.encoder_intermediate_size, config.encoder_hidden_size, true}).build(ctx, out, w2);
}

core::TensorValue conformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin,
    const core::TensorValue & half,
    const SheetSage2EncoderLayerWeights & weights,
    const SheetSage2DecoderConfig & config,
    int64_t attention_tile_rows) {
    auto hidden = modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, input, weights.ffn1_norm);
    hidden = feed_forward(ctx, hidden, weights.ffn1_w1, weights.ffn1_w2, config);
    auto out = modules::ResidualAddModule().build(
        ctx,
        modules::MulModule().build(ctx, hidden, modules::RepeatModule({hidden.shape}).build(ctx, half)),
        input);
    hidden = modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, out, weights.attn_norm);
    hidden = encoder_attention(ctx, hidden, cos, sin, weights.attn, config, attention_tile_rows);
    out = modules::ResidualAddModule().build(ctx, hidden, out);
    hidden = modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, out, weights.conv_norm);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    hidden = modules::Conv1dModule({config.encoder_hidden_size, config.encoder_hidden_size * 2, 1, 1, 0, 1, false}).build(ctx, hidden, weights.conv_pw_in);
    auto gate_a = modules::SliceModule({1, 0, config.encoder_hidden_size}).build(ctx, hidden);
    auto gate_b = modules::SliceModule({1, config.encoder_hidden_size, config.encoder_hidden_size}).build(ctx, hidden);
    gate_b = core::wrap_tensor(ggml_sigmoid(ctx.ggml, gate_b.tensor), gate_b.shape, GGML_TYPE_F32);
    hidden = modules::MulModule().build(ctx, gate_a, gate_b);
    hidden = modules::DepthwiseConv1dModule({config.encoder_hidden_size, config.conformer_conv_kernel_size, 1, static_cast<int>((config.conformer_conv_kernel_size - 1) / 2), 1, false}).build(ctx, hidden, weights.conv_depthwise);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    hidden = modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, hidden, weights.conv_depthwise_norm);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    hidden = modules::Conv1dModule({config.encoder_hidden_size, config.encoder_hidden_size, 1, 1, 0, 1, false}).build(ctx, hidden, weights.conv_pw_out);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    out = modules::ResidualAddModule().build(ctx, hidden, out);
    hidden = modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, out, weights.ffn2_norm);
    hidden = feed_forward(ctx, hidden, weights.ffn2_w1, weights.ffn2_w2, config);
    out = modules::ResidualAddModule().build(
        ctx,
        modules::MulModule().build(ctx, hidden, modules::RepeatModule({hidden.shape}).build(ctx, half)),
        out);
    return modules::LayerNormModule({config.encoder_hidden_size, config.encoder_layer_norm_eps, true, true}).build(ctx, out, weights.final_norm);
}

}  // namespace

struct Mert2EncoderRuntime::Impl {
    class EncoderGraph;

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config,
        SheetSage2DecoderRuntimeOptions options)
        : source(std::move(source)),
          execution(&execution),
          config(config),
          options(options) {
        if (!this->source) {
            throw std::runtime_error("MERT2 encoder runtime requires tensor source");
        }
        if (options.attention_tile_rows < 0) {
            throw std::runtime_error("SheetSage2 attention_tile_rows must be nonnegative");
        }
        validate_config(this->config);
    }

    const SheetSage2EncoderWeights & require_encoder_weights() {
        if (!encoder_weights) {
            encoder_weights = std::make_unique<SheetSage2EncoderWeights>(load_encoder_weights(
                *source,
                config,
                execution->backend(),
                execution->backend_type(),
                options));
            source->release_storage();
        }
        return *encoder_weights;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext * execution = nullptr;
    SheetSage2DecoderConfig config;
    SheetSage2DecoderRuntimeOptions options;
    std::unique_ptr<SheetSage2EncoderWeights> encoder_weights;
    std::unique_ptr<EncoderGraph> encoder_graph;
};

class Mert2EncoderRuntime::Impl::EncoderGraph {
public:
    EncoderGraph(
        core::ExecutionContext & execution,
        const SheetSage2DecoderConfig & config,
        const SheetSage2DecoderRuntimeOptions & options,
        const SheetSage2EncoderWeights & weights,
        int64_t batch,
        int64_t mel_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          batch_(batch),
          mel_frames_(mel_frames) {
        if (backend_ == nullptr || batch_ != 1 || mel_frames_ <= 0) {
            throw std::runtime_error("SheetSage2 encoder graph initialization failed");
        }
        encoded_frames_ = ((mel_frames_ - 2) / 2 + 1 - 2) / 2 + 1;
        if (encoded_frames_ <= 0) {
            throw std::runtime_error("SheetSage2 encoder graph computed no frames");
        }
        build();
    }

    ~EncoderGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t mel_frames) const noexcept {
        return batch == batch_ && mel_frames == mel_frames_;
    }

    int64_t encoded_frames() const noexcept {
        return encoded_frames_;
    }

    std::vector<float> run(const std::vector<float> & normalized_mel) const {
        if (static_cast<int64_t>(normalized_mel.size()) != batch_ * mel_frames_ * config_.mel_bins) {
            throw std::runtime_error("SheetSage2 normalized mel shape mismatch");
        }
        core::write_tensor_f32(mel_, normalized_mel);
        core::write_tensor_f32(rope_cos_, rope_values(true));
        core::write_tensor_f32(rope_sin_, rope_values(false));
        core::write_tensor_f32(grn_eps_, std::vector<float>(static_cast<size_t>(batch_), 1.0e-6F));
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "models.sheetsage2.encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SheetSage2 encoder graph compute failed");
        }
        return core::read_tensor_f32(mixed_);
    }

private:
    std::vector<float> rope_values(bool cosine) const {
        const int64_t head_dim = config_.encoder_hidden_size / config_.encoder_attention_heads;
        const int64_t half_dim = head_dim / 2;
        std::vector<float> values(static_cast<size_t>(config_.encoder_attention_heads * encoded_frames_ * half_dim));
        for (int64_t h = 0; h < config_.encoder_attention_heads; ++h) {
            for (int64_t t = 0; t < encoded_frames_; ++t) {
                for (int64_t i = 0; i < half_dim; ++i) {
                    const float inv = std::pow(config_.rotary_embedding_base, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
                    const float value = cosine ? std::cos(static_cast<float>(t) * inv) : std::sin(static_cast<float>(t) * inv);
                    values[static_cast<size_t>((h * encoded_frames_ + t) * half_dim + i)] = value;
                }
            }
        }
        return values;
    }

    void build() {
        ggml_init_params params{options_.graph_arena_bytes * 8, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("SheetSage2 encoder ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "models.sheetsage2.encoder.inputs", backend_type_};
        mel_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, mel_frames_, config_.mel_bins}));
        const int64_t head_dim = config_.encoder_hidden_size / config_.encoder_attention_heads;
        rope_cos_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.encoder_attention_heads, encoded_frames_, head_dim / 2}));
        rope_sin_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config_.encoder_attention_heads, encoded_frames_, head_dim / 2}));
        grn_eps_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, 1}));
        ggml_set_input(mel_.tensor);
        ggml_set_input(rope_cos_.tensor);
        ggml_set_input(rope_sin_.tensor);
        ggml_set_input(grn_eps_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "models.sheetsage2.encoder", backend_type_};
        auto mixed = build_graph_output(build_ctx);
        mixed_ = mixed.tensor;
        ggml_set_output(mixed_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, mixed_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("SheetSage2 encoder backend buffer allocation failed");
        }
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        auto hidden = mel_;
        hidden = subsampling_block(ctx, hidden, grn_eps_, weights_.subsampling[0], 128, 128, 1, config_.subsampling_layer_norm_eps);
        hidden = subsampling_block(ctx, hidden, grn_eps_, weights_.subsampling[1], 128, 512, 2, config_.subsampling_layer_norm_eps);
        hidden = subsampling_block(ctx, hidden, grn_eps_, weights_.subsampling[2], 512, 1024, 2, config_.subsampling_layer_norm_eps);
        auto mixed = modules::MulModule().build(
            ctx,
            hidden,
            modules::RepeatModule({hidden.shape}).build(ctx, weights_.layer_weights[0]));
        for (int64_t i = 0; i < config_.encoder_layers; ++i) {
            hidden = conformer_layer(
                ctx,
                hidden,
                rope_cos_,
                rope_sin_,
                weights_.half,
                weights_.layers[static_cast<size_t>(i)],
                config_,
                options_.attention_tile_rows);
            const auto layer_weight = modules::RepeatModule({hidden.shape}).build(
                ctx,
                weights_.layer_weights[static_cast<size_t>(i + 1)]);
            mixed = modules::AddModule().build(ctx, mixed, modules::MulModule().build(ctx, hidden, layer_weight));
        }
        return mixed;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    SheetSage2DecoderConfig config_;
    SheetSage2DecoderRuntimeOptions options_;
    const SheetSage2EncoderWeights & weights_;
    int64_t batch_ = 1;
    int64_t mel_frames_ = 0;
    int64_t encoded_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue mel_;
    core::TensorValue rope_cos_;
    core::TensorValue rope_sin_;
    core::TensorValue grn_eps_;
    ggml_tensor * mixed_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

struct SheetSage2DecoderRuntime::Impl {
    class DecodeGraph;
    class CachedDecodeGraph;

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config,
        SheetSage2DecoderRuntimeOptions options)
        : source(std::move(source)),
          execution(&execution),
          config(config),
          options(options) {
        if (!this->source) {
            throw std::runtime_error("SheetSage2 decoder runtime requires tensor source");
        }
        validate_config(this->config);
    }

    const SheetSage2DecoderWeights & require_weights() {
        if (!weights) {
            weights = std::make_unique<SheetSage2DecoderWeights>(load_weights(
                *source,
                config,
                execution->backend(),
                execution->backend_type(),
                options));
            source->release_storage();
        }
        return *weights;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext * execution = nullptr;
    SheetSage2DecoderConfig config;
    SheetSage2DecoderRuntimeOptions options;
    std::unique_ptr<SheetSage2DecoderWeights> weights;
    std::unique_ptr<DecodeGraph> graph;
    std::unique_ptr<CachedDecodeGraph> cached_graph;
};

class SheetSage2DecoderRuntime::Impl::DecodeGraph {
public:
    DecodeGraph(
        core::ExecutionContext & execution,
        const SheetSage2DecoderConfig & config,
        const SheetSage2DecoderRuntimeOptions & options,
        const SheetSage2DecoderWeights & weights,
        int64_t batch,
        int64_t memory_steps,
        int64_t decoder_steps)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          batch_(batch),
          memory_steps_(memory_steps),
          decoder_steps_(decoder_steps) {
        if (backend_ == nullptr || batch_ <= 0 || memory_steps_ <= 0 || decoder_steps_ <= 0) {
            throw std::runtime_error("SheetSage2 decoder graph initialization failed");
        }
        if (decoder_steps_ > config_.max_position_embeddings) {
            throw std::runtime_error("SheetSage2 decoder steps exceed max position embeddings");
        }
        build();
    }

    ~DecodeGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t memory_steps, int64_t decoder_steps) const noexcept {
        return batch == batch_ && memory_steps == memory_steps_ && decoder_steps == decoder_steps_;
    }

    std::vector<float> run(
        const std::vector<float> & mixed_encoder_state,
        const std::vector<int32_t> & decoder_input_ids) const {
        if (static_cast<int64_t>(mixed_encoder_state.size()) != batch_ * memory_steps_ * config_.encoder_hidden_size) {
            throw std::runtime_error("SheetSage2 mixed encoder state shape mismatch");
        }
        if (static_cast<int64_t>(decoder_input_ids.size()) != batch_ * decoder_steps_) {
            throw std::runtime_error("SheetSage2 decoder input id shape mismatch");
        }
        core::write_tensor_f32(mixed_encoder_state_, mixed_encoder_state);
        core::write_tensor_i32(decoder_input_ids_, decoder_input_ids);
        core::write_tensor_i32(position_ids_, position_ids());
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "models.sheetsage2.decoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SheetSage2 decoder graph compute failed");
        }
        return core::read_tensor_f32(logits_);
    }

private:
    std::vector<int32_t> position_ids() const {
        std::vector<int32_t> ids(static_cast<size_t>(batch_ * decoder_steps_));
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t t = 0; t < decoder_steps_; ++t) {
                ids[static_cast<size_t>(b * decoder_steps_ + t)] = static_cast<int32_t>(t + 2);
            }
        }
        return ids;
    }

    void build() {
        ggml_init_params params{options_.graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("SheetSage2 decoder ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "models.sheetsage2.decoder.inputs", backend_type_};
        mixed_encoder_state_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, memory_steps_, config_.encoder_hidden_size}));
        decoder_input_ids_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({batch_, decoder_steps_}));
        position_ids_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({batch_, decoder_steps_}));
        ggml_set_input(mixed_encoder_state_.tensor);
        ggml_set_input(decoder_input_ids_.tensor);
        ggml_set_input(position_ids_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "models.sheetsage2.decoder", backend_type_};
        auto logits = build_graph_output(build_ctx);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, logits_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("SheetSage2 decoder backend buffer allocation failed");
        }
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        auto memory = modules::LinearModule({config_.encoder_hidden_size, config_.hidden_size, true}).build(
            ctx,
            mixed_encoder_state_,
            weights_.encoder_projection);
        auto hidden = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(
            ctx,
            decoder_input_ids_,
            weights_.token_embedding);
        auto positions = modules::EmbeddingModule({config_.max_position_embeddings + 2, config_.hidden_size}).build(
            ctx,
            position_ids_,
            weights_.position_embedding);
        hidden = modules::AddModule{}.build(ctx, hidden, positions);
        hidden = modules::LayerNormModule({config_.hidden_size, config_.layer_norm_eps, true, true}).build(
            ctx,
            hidden,
            weights_.layernorm_embedding);
        for (const auto & layer : weights_.layers) {
            hidden = decoder_layer(ctx, hidden, memory, layer, config_);
        }
        auto logits = modules::LinearModule({config_.hidden_size, config_.vocab_size, false}).build(
            ctx,
            hidden,
            {weights_.token_embedding, std::nullopt});
        logits = modules::SliceModule({1, decoder_steps_ - 1, 1}).build(ctx, logits);
        return core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({config_.vocab_size}));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    SheetSage2DecoderConfig config_;
    SheetSage2DecoderRuntimeOptions options_;
    const SheetSage2DecoderWeights & weights_;
    int64_t batch_ = 0;
    int64_t memory_steps_ = 0;
    int64_t decoder_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue mixed_encoder_state_;
    core::TensorValue decoder_input_ids_;
    core::TensorValue position_ids_;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class SheetSage2DecoderRuntime::Impl::CachedDecodeGraph {
public:
    CachedDecodeGraph(
        core::ExecutionContext & execution,
        const SheetSage2DecoderConfig & config,
        const SheetSage2DecoderRuntimeOptions & options,
        const SheetSage2DecoderWeights & weights,
        int64_t memory_steps,
        int64_t cache_steps)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          config_(config),
          options_(options),
          weights_(weights),
          memory_steps_(memory_steps),
          cache_steps_(cache_steps),
          head_dim_(config.hidden_size / config.num_attention_heads) {
        if (backend_ == nullptr || memory_steps_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("SheetSage2 cached decoder graph initialization failed");
        }
        if (cache_steps_ > config_.max_position_embeddings) {
            throw std::runtime_error("SheetSage2 cached decoder steps exceed max position embeddings");
        }
        build();
    }

    ~CachedDecodeGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (backend_ != nullptr && state_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, state_graph_);
        }
        if (backend_ != nullptr && cross_graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, cross_graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(int64_t memory_steps, int64_t cache_steps) const noexcept {
        return memory_steps == memory_steps_ && cache_steps == cache_steps_;
    }

    std::vector<float> prefill(const std::vector<int32_t> & tokens) {
        const int64_t steps = static_cast<int64_t>(tokens.size());
        if (steps <= 0) {
            throw std::runtime_error("SheetSage2 cached decoder prefill requires tokens");
        }
        if (steps > cache_steps_) {
            throw std::runtime_error("SheetSage2 cached decoder prefill exceeds cache capacity");
        }
        for (int64_t i = 0; i + 1 < steps; ++i) {
            run_step(tokens[static_cast<size_t>(i)], false);
        }
        return run_step(tokens.back(), true);
    }

    void reset(const std::vector<float> & mixed_encoder_state) {
        if (static_cast<int64_t>(mixed_encoder_state.size()) != memory_steps_ * config_.encoder_hidden_size) {
            throw std::runtime_error("SheetSage2 cached decoder memory shape mismatch");
        }
        core::write_tensor_f32(mixed_encoder_state_, mixed_encoder_state);
        const auto cross_start = std::chrono::steady_clock::now();
        const ggml_status cross_status =
            core::compute_backend_graph(backend_, cross_graph_, nullptr, "models.sheetsage2.decoder.cross_cache");
        if (cross_status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SheetSage2 cached decoder cross-cache graph compute failed");
        }
        engine::debug::timing_log_scalar(
            "sheetsage2.decoder.cross_cache_ms",
            engine::debug::elapsed_ms(cross_start));
        runtime::TransformerKVState empty;
        empty.current_end = 0;
        empty.layers.resize(static_cast<size_t>(config_.decoder_layers));
        for (auto & layer : empty.layers) {
            layer.valid_steps = 0;
        }
        step_cache_.import_state(empty);
        attention_mask_values_.assign(
            static_cast<size_t>(config_.num_attention_heads * cache_steps_),
            ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(
            attention_mask_.tensor,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
    }

    std::vector<float> run_step(int32_t token, bool read_logits = true) {
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("SheetSage2 cached decoder cache exhausted");
        }
        core::write_tensor_i32(token_id_, std::vector<int32_t>{token});
        core::write_tensor_i32(position_id_, std::vector<int32_t>{static_cast<int32_t>(step_cache_.current_end() + 2)});
        core::write_tensor_i32(cache_slot_, std::vector<int32_t>{static_cast<int32_t>(step_cache_.valid_steps())});
        const size_t valid = static_cast<size_t>(step_cache_.valid_steps());
        for (int64_t head = 0; head < config_.num_attention_heads; ++head) {
            attention_mask_values_[static_cast<size_t>(head * cache_steps_) + valid] = ggml_fp32_to_fp16(0.0F);
        }
        ggml_backend_tensor_set(
            attention_mask_.tensor,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(
            backend_,
            read_logits ? graph_ : state_graph_,
            nullptr,
            read_logits ? "models.sheetsage2.decoder.cached_step" : "models.sheetsage2.decoder.cached_state_step");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("SheetSage2 cached decoder graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        step_cache_.advance_after_direct_append(1);
        return read_logits ? core::read_tensor_f32(logits_) : std::vector<float>{};
    }

private:
    void build() {
        ggml_init_params params{options_.graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("SheetSage2 cached decoder ggml context initialization failed");
        }
        core::ModuleBuildContext input_ctx{ctx_.get(), "models.sheetsage2.decoder.cached.inputs", backend_type_};
        mixed_encoder_state_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, memory_steps_, config_.encoder_hidden_size}));
        token_id_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({1, 1}));
        position_id_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({1, 1}));
        cache_slot_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_I32,
            core::TensorShape::from_dims({1}));
        attention_mask_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F16,
            core::TensorShape::from_dims({1, config_.num_attention_heads, 1, cache_steps_}));
        ggml_set_input(mixed_encoder_state_.tensor);
        ggml_set_input(token_id_.tensor);
        ggml_set_input(position_id_.tensor);
        ggml_set_input(cache_slot_.tensor);
        ggml_set_input(attention_mask_.tensor);

        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(static_cast<size_t>(config_.decoder_layers));
        cache_values.reserve(static_cast<size_t>(config_.decoder_layers));
        cross_keys_.reserve(static_cast<size_t>(config_.decoder_layers));
        cross_values_.reserve(static_cast<size_t>(config_.decoder_layers));
        for (int64_t layer = 0; layer < config_.decoder_layers; ++layer) {
            cache_keys.push_back(core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps_, config_.num_attention_heads, head_dim_})));
            cache_values.push_back(core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps_, config_.num_attention_heads, head_dim_})));
            ggml_set_input(cache_keys.back().tensor);
            ggml_set_input(cache_values.back().tensor);
            cross_keys_.push_back(core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, config_.num_attention_heads, memory_steps_, head_dim_})));
            cross_values_.push_back(core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, config_.num_attention_heads, memory_steps_, head_dim_})));
        }
        runtime::TransformerKVCacheOptions cache_options;
        cache_options.allow_f16_storage = true;
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_,
            config_.num_attention_heads * head_dim_,
            std::move(cache_keys),
            std::move(cache_values),
            cache_options);

        core::ModuleBuildContext build_ctx{ctx_.get(), "models.sheetsage2.decoder.cached", backend_type_};
        build_cross_cache_graph(build_ctx);
        auto hidden = build_hidden_output(build_ctx);
        hidden_ = hidden.tensor;
        ggml_set_output(hidden_);
        state_graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(state_graph_, hidden_);
        auto logits = build_logits_output(build_ctx, hidden);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("SheetSage2 cached decoder backend buffer allocation failed");
        }
    }

    void build_cross_cache_graph(core::ModuleBuildContext & ctx) {
        auto memory = modules::LinearModule({config_.encoder_hidden_size, config_.hidden_size, true}).build(
            ctx,
            mixed_encoder_state_,
            weights_.encoder_projection);
        cross_graph_ = ggml_new_graph_custom(ctx.ggml, 131072, false);
        for (int64_t layer = 0; layer < config_.decoder_layers; ++layer) {
            const auto kv = cross_attention_key_value(
                ctx,
                memory,
                weights_.layers[static_cast<size_t>(layer)].encoder_attn,
                config_.hidden_size,
                config_.num_attention_heads);
            auto key_copy = core::wrap_tensor(
                ggml_cpy(ctx.ggml, kv.key.tensor, cross_keys_[static_cast<size_t>(layer)].tensor),
                cross_keys_[static_cast<size_t>(layer)].shape,
                GGML_TYPE_F16);
            auto value_copy = core::wrap_tensor(
                ggml_cpy(ctx.ggml, kv.value.tensor, cross_values_[static_cast<size_t>(layer)].tensor),
                cross_values_[static_cast<size_t>(layer)].shape,
                GGML_TYPE_F16);
            ggml_set_output(key_copy.tensor);
            ggml_set_output(value_copy.tensor);
            ggml_build_forward_expand(cross_graph_, key_copy.tensor);
            ggml_build_forward_expand(cross_graph_, value_copy.tensor);
        }
    }

    core::TensorValue build_hidden_output(core::ModuleBuildContext & ctx) {
        auto hidden = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(
            ctx,
            token_id_,
            weights_.token_embedding);
        auto positions = modules::EmbeddingModule({config_.max_position_embeddings + 2, config_.hidden_size}).build(
            ctx,
            position_id_,
            weights_.position_embedding);
        hidden = modules::AddModule{}.build(ctx, hidden, positions);
        hidden = modules::LayerNormModule({config_.hidden_size, config_.layer_norm_eps, true, true}).build(
            ctx,
            hidden,
            weights_.layernorm_embedding);
        for (int64_t layer = 0; layer < config_.decoder_layers; ++layer) {
            auto out = cached_decoder_layer_step(
                ctx,
                hidden,
                {cross_keys_[static_cast<size_t>(layer)], cross_values_[static_cast<size_t>(layer)]},
                step_cache_.key_tensor(static_cast<size_t>(layer)),
                step_cache_.value_tensor(static_cast<size_t>(layer)),
                cache_slot_,
                attention_mask_,
                weights_.layers[static_cast<size_t>(layer)],
                config_);
            hidden = out.hidden;
        }
        return core::ensure_backend_addressable_layout(ctx, hidden);
    }

    core::TensorValue build_logits_output(core::ModuleBuildContext & ctx, const core::TensorValue & hidden) {
        auto logits = modules::LinearModule({config_.hidden_size, config_.vocab_size, false}).build(
            ctx,
            hidden,
            {weights_.token_embedding, std::nullopt});
        return core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({config_.vocab_size}));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    SheetSage2DecoderConfig config_;
    SheetSage2DecoderRuntimeOptions options_;
    const SheetSage2DecoderWeights & weights_;
    int64_t memory_steps_ = 0;
    int64_t cache_steps_ = 0;
    int64_t head_dim_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue mixed_encoder_state_;
    core::TensorValue token_id_;
    core::TensorValue position_id_;
    core::TensorValue cache_slot_;
    core::TensorValue attention_mask_;
    std::vector<core::TensorValue> cross_keys_;
    std::vector<core::TensorValue> cross_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_cgraph * state_graph_ = nullptr;
    ggml_cgraph * cross_graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

Mert2EncoderRuntime::Mert2EncoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    SheetSage2DecoderConfig config,
    SheetSage2DecoderRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, config, options)) {}

Mert2EncoderRuntime::~Mert2EncoderRuntime() = default;
Mert2EncoderRuntime::Mert2EncoderRuntime(Mert2EncoderRuntime &&) noexcept = default;
Mert2EncoderRuntime & Mert2EncoderRuntime::operator=(Mert2EncoderRuntime &&) noexcept = default;

void Mert2EncoderRuntime::prepare(int64_t batch, int64_t mel_frames) {
    const auto & weights = impl_->require_encoder_weights();
    if (!impl_->encoder_graph || !impl_->encoder_graph->matches(batch, mel_frames)) {
        impl_->encoder_graph = std::make_unique<Impl::EncoderGraph>(
            *impl_->execution,
            impl_->config,
            impl_->options,
            weights,
            batch,
            mel_frames);
    }
}

std::vector<float> Mert2EncoderRuntime::encode_mel(
    const std::vector<float> & normalized_mel,
    int64_t mel_frames) {
    if (mel_frames <= 0) {
        throw std::runtime_error("MERT2 mel frames must be positive");
    }
    if (static_cast<int64_t>(normalized_mel.size()) % (mel_frames * impl_->config.mel_bins) != 0) {
        throw std::runtime_error("MERT2 normalized mel does not divide into batches");
    }
    const int64_t batch = static_cast<int64_t>(normalized_mel.size()) / (mel_frames * impl_->config.mel_bins);
    prepare(batch, mel_frames);
    return impl_->encoder_graph->run(normalized_mel);
}

void Mert2EncoderRuntime::release_runtime_graphs() {
    impl_->encoder_graph.reset();
}

SheetSage2DecoderRuntime::SheetSage2DecoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    SheetSage2DecoderConfig config,
    SheetSage2DecoderRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, config, options)) {}

SheetSage2DecoderRuntime::~SheetSage2DecoderRuntime() = default;
SheetSage2DecoderRuntime::SheetSage2DecoderRuntime(SheetSage2DecoderRuntime &&) noexcept = default;
SheetSage2DecoderRuntime & SheetSage2DecoderRuntime::operator=(SheetSage2DecoderRuntime &&) noexcept = default;

void SheetSage2DecoderRuntime::prepare(int64_t batch, int64_t memory_steps, int64_t decoder_steps) {
    const auto & weights = impl_->require_weights();
    if (!impl_->graph || !impl_->graph->matches(batch, memory_steps, decoder_steps)) {
        impl_->graph = std::make_unique<Impl::DecodeGraph>(
            *impl_->execution,
            impl_->config,
            impl_->options,
            weights,
            batch,
            memory_steps,
            decoder_steps);
    }
}

std::vector<float> SheetSage2DecoderRuntime::decode_logits(
    const std::vector<float> & mixed_encoder_state,
    int64_t memory_steps,
    const std::vector<int32_t> & decoder_input_ids) {
    if (memory_steps <= 0) {
        throw std::runtime_error("SheetSage2 memory steps must be positive");
    }
    if (decoder_input_ids.empty()) {
        throw std::runtime_error("SheetSage2 decoder input ids must not be empty");
    }
    const int64_t memory_values_per_batch = memory_steps * impl_->config.encoder_hidden_size;
    if (static_cast<int64_t>(mixed_encoder_state.size()) % memory_values_per_batch != 0) {
        throw std::runtime_error("SheetSage2 mixed encoder state does not divide into batches");
    }
    const int64_t batch = static_cast<int64_t>(mixed_encoder_state.size()) / memory_values_per_batch;
    if (static_cast<int64_t>(decoder_input_ids.size()) % batch != 0) {
        throw std::runtime_error("SheetSage2 decoder input ids do not divide by batch");
    }
    const int64_t decoder_steps = static_cast<int64_t>(decoder_input_ids.size()) / batch;
    prepare(batch, memory_steps, decoder_steps);
    return impl_->graph->run(mixed_encoder_state, decoder_input_ids);
}

void SheetSage2DecoderRuntime::reset_cached_decode(
    const std::vector<float> & mixed_encoder_state,
    int64_t memory_steps,
    int64_t cache_steps) {
    if (memory_steps <= 0 || cache_steps <= 0) {
        throw std::runtime_error("SheetSage2 cached decode requires positive memory/cache steps");
    }
    const int64_t expected = memory_steps * impl_->config.encoder_hidden_size;
    if (static_cast<int64_t>(mixed_encoder_state.size()) != expected) {
        throw std::runtime_error("SheetSage2 cached decode memory shape mismatch");
    }
    const auto & weights = impl_->require_weights();
    if (!impl_->cached_graph || !impl_->cached_graph->matches(memory_steps, cache_steps)) {
        impl_->cached_graph = std::make_unique<Impl::CachedDecodeGraph>(
            *impl_->execution,
            impl_->config,
            impl_->options,
            weights,
            memory_steps,
            cache_steps);
    }
    impl_->cached_graph->reset(mixed_encoder_state);
}

std::vector<float> SheetSage2DecoderRuntime::decode_cached_step(int32_t token) {
    if (!impl_->cached_graph) {
        throw std::runtime_error("SheetSage2 cached decode graph is not prepared");
    }
    return impl_->cached_graph->run_step(token);
}

std::vector<float> SheetSage2DecoderRuntime::prefill_cached_decode(const std::vector<int32_t> & token_ids) {
    if (!impl_->cached_graph) {
        throw std::runtime_error("SheetSage2 cached decode graph is not prepared");
    }
    return impl_->cached_graph->prefill(token_ids);
}

void SheetSage2DecoderRuntime::release_runtime_graphs() {
    impl_->graph.reset();
    impl_->cached_graph.reset();
}

}  // namespace engine::models::sheetsage

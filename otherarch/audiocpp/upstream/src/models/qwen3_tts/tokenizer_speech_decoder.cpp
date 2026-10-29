#include "engine/models/qwen3_tts/tokenizer_speech_decoder.h"

#include "engine/models/qwen3_tts/rknn_speech_decoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/attention/streaming_frame_attention_cache.h"
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
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::qwen3_tts {
namespace json = engine::io::json;
namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

constexpr int64_t kSampleRate = 24000;
constexpr int64_t kDecodeSamplesPerCode = 1920;
constexpr int64_t kChunkCodes = 300;
constexpr int64_t kLeftContextCodes = Qwen3SpeechTokenizerDecoderRuntime::kLeftContextFrames;
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

struct LinearWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t input_dim = 0;
    int64_t output_dim = 0;
    bool use_bias = true;
};

struct Conv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t groups = 1;
    bool use_bias = true;
};

struct ConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    bool use_bias = true;
};

struct LayerNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    float eps = 1.0e-6F;
};

struct RmsNormWeights {
    std::vector<float> weight;
    float eps = 1.0e-5F;
};

struct CodebookWeights {
    std::vector<float> embedding;
};

struct AttentionWeights {
    LinearWeights q;
    LinearWeights k;
    LinearWeights v;
    LinearWeights o;
};

struct MlpWeights {
    LinearWeights gate;
    LinearWeights up;
    LinearWeights down;
};

struct TransformerLayerWeights {
    AttentionWeights attention;
    MlpWeights mlp;
    RmsNormWeights input_norm;
    RmsNormWeights post_norm;
    std::vector<float> attn_scale;
    std::vector<float> mlp_scale;
};

struct ConvNeXtWeights {
    Conv1dWeights dwconv;
    LayerNormWeights norm;
    LinearWeights pwconv1;
    LinearWeights pwconv2;
    std::vector<float> gamma;
};

struct ResidualUnitWeights {
    std::vector<float> act1_alpha;
    std::vector<float> act1_beta;
    Conv1dWeights conv1;
    std::vector<float> act2_alpha;
    std::vector<float> act2_beta;
    Conv1dWeights conv2;
};

struct UpsampleStageWeights {
    ConvTranspose1dWeights upconv;
    ConvNeXtWeights convnext;
};

struct DecoderBlockWeights {
    std::vector<float> input_alpha;
    std::vector<float> input_beta;
    ConvTranspose1dWeights upconv;
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

struct Qwen3SpeechTokenizerDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    DecoderConfig config;
    std::vector<CodebookWeights> semantic_codebooks;
    std::vector<CodebookWeights> acoustic_codebooks;
    LinearWeights semantic_output_proj;
    LinearWeights acoustic_output_proj;
    Conv1dWeights pre_conv;
    LinearWeights transformer_input_proj;
    std::vector<TransformerLayerWeights> transformer_layers;
    RmsNormWeights transformer_norm;
    LinearWeights transformer_output_proj;
    std::vector<UpsampleStageWeights> upsample_stages;
    Conv1dWeights decoder_input_conv;
    std::vector<DecoderBlockWeights> decoder_blocks;
    std::vector<float> output_alpha;
    std::vector<float> output_beta;
    Conv1dWeights output_conv;
};

// Persistent stream tensors are separate from graph scratch: grow chunks and
// short tails each use their exact frame count, and every graph uses the same
// session-owned history. Weight addresses identify layers within this session.
class Qwen3SpeechDecoderStreamState {
public:
    explicit Qwen3SpeechDecoderStreamState(size_t arena_bytes) {
        ctx_.reset(ggml_init({arena_bytes, nullptr, true}));
        if (!ctx_) throw std::runtime_error("failed to initialize Qwen3 decoder stream state");
    }

    ~Qwen3SpeechDecoderStreamState() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
    }

    core::TensorValue tensor(const void * layer, size_t slot, const core::TensorShape & shape) {
        auto & tensors = tensors_[layer];
        if (slot < tensors.size()) {
            if (tensors[slot].shape.dims != shape.dims || tensors[slot].shape.rank != shape.rank) {
                throw std::runtime_error("Qwen3 decoder stream cache shape changed");
            }
            return tensors[slot];
        }
        if (buffer_ || slot != tensors.size()) {
            throw std::runtime_error("Qwen3 decoder stream cache layout changed");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_tts.speech_decoder.stream_state"};
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        ggml_set_input(value.tensor);
        ggml_set_output(value.tensor);
        tensors.push_back(value);
        return value;
    }

    void allocate(ggml_backend_t backend) {
        if (buffer_) return;
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend);
        if (!buffer_) throw std::runtime_error("failed to allocate Qwen3 decoder stream state");
        reset();
    }

    void reset() {
        if (buffer_) ggml_backend_buffer_clear(buffer_, 0);
        frames = 0;
    }

    int64_t frames = 0;

private:
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::map<const void *, std::vector<core::TensorValue>, std::less<const void *>> tensors_;
};

namespace {

// Graph-local bindings into session-owned state. Each retained graph has an
// exact chunk length, so the framework's tail slices commit only real frames.
class DecoderStreamGraphState {
public:
    explicit DecoderStreamGraphState(Qwen3SpeechDecoderStreamState & state) : state_(state) {}

    core::TensorValue convolution_input(
        core::ModuleBuildContext & ctx, const void * layer,
        const core::TensorValue & input, int64_t history) {
        if (history == 0) return input;
        auto shape = input.shape;
        shape.dims[2] = history;
        auto cache = state_.tensor(layer, 0, shape);
        auto context = modules::ConcatModule({2}).build(ctx, cache, input);
        auto next = modules::SliceModule({2, context.shape.dims[2] - history, history}).build(ctx, context);
        updates_.push_back({cache.tensor, next.tensor});
        return context;
    }

    core::TensorValue transposed_convolution_output(
        core::ModuleBuildContext & ctx, const void * layer,
        const core::TensorValue & raw, int64_t emit, int64_t overlap) {
        if (overlap == 0) return raw;
        auto shape = raw.shape;
        shape.dims[2] = overlap;
        auto carry = state_.tensor(layer, 0, shape);
        core::TensorValue output, next;
        if (overlap <= emit) {
            auto head = modules::SliceModule({2, 0, overlap}).build(ctx, raw);
            output = modules::AddModule{}.build(ctx, head, carry);
            if (emit > overlap) {
                auto body = modules::SliceModule({2, overlap, emit - overlap}).build(ctx, raw);
                output = modules::ConcatModule({2}).build(ctx, output, body);
            }
            next = modules::SliceModule({2, emit, overlap}).build(ctx, raw);
        } else {
            // Kernels wider than one emitted chunk retain the unconsumed carry.
            auto padded = core::wrap_tensor(
                ggml_pad(ctx.ggml, carry.tensor, emit, 0, 0, 0), raw.shape, GGML_TYPE_F32);
            auto mixed = modules::AddModule{}.build(ctx, raw, padded);
            output = modules::SliceModule({2, 0, emit}).build(ctx, mixed);
            next = modules::SliceModule({2, emit, overlap}).build(ctx, mixed);
        }
        updates_.push_back({carry.tensor, next.tensor});
        return output;
    }

    std::pair<core::TensorValue, core::TensorValue> attention_input(
        core::ModuleBuildContext & ctx, const void * layer,
        const core::TensorValue & key, const core::TensorValue & value, int64_t window) {
        auto shape = key.shape;
        shape.dims[2] = window;
        auto key_cache = state_.tensor(layer, 0, shape);
        auto value_cache = state_.tensor(layer, 1, shape);
        auto result = modules::StreamingFrameAttentionCacheModule({window}).build(
            ctx, key_cache, value_cache, key, value);
        updates_.push_back({key_cache.tensor, result.next_key_cache.tensor});
        updates_.push_back({value_cache.tensor, result.next_value_cache.tensor});
        return {result.key_context, result.value_context};
    }

    void expand(core::ModuleBuildContext & ctx, ggml_cgraph * graph) {
        // Read every old state before any writes. The waveform is expanded first.
        for (const auto & update : updates_) ggml_build_forward_expand(graph, update.next);
        for (const auto & update : updates_) {
            ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, update.next, update.cache));
        }
    }

private:
    struct Update {
        ggml_tensor * cache;
        ggml_tensor * next;
    };
    Qwen3SpeechDecoderStreamState & state_;
    std::vector<Update> updates_;
};

std::vector<float> normalized_codebook(
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
    return embedding;
}

DecoderConfig load_decoder_config(const Qwen3TTSAssets & assets) {
    const auto root = assets.resources.parse_json("speech_tokenizer_config");
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

Conv1dWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride = 1,
    int64_t dilation = 1,
    int64_t groups = 1,
    bool use_bias = true) {
    Conv1dWeights weights;
    weights.in_channels = in_channels;
    weights.out_channels = out_channels;
    weights.kernel = kernel;
    weights.stride = stride;
    weights.dilation = dilation;
    weights.groups = groups;
    weights.use_bias = use_bias;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels / groups, kernel});
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    }
    return weights;
}

ConvTranspose1dWeights load_conv_transpose(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    bool use_bias = true) {
    ConvTranspose1dWeights weights;
    weights.in_channels = in_channels;
    weights.out_channels = out_channels;
    weights.kernel = kernel;
    weights.stride = stride;
    weights.use_bias = use_bias;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {in_channels, out_channels, kernel});
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    }
    return weights;
}

LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t input_dim,
    int64_t output_dim,
    bool use_bias = true) {
    LinearWeights weights;
    weights.input_dim = input_dim;
    weights.output_dim = output_dim;
    weights.use_bias = use_bias;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {output_dim, input_dim});
    if (use_bias) {
        weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {output_dim});
    }
    return weights;
}

LinearWeights load_conv1x1_as_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t input_dim,
    int64_t output_dim) {
    LinearWeights weights;
    weights.input_dim = input_dim;
    weights.output_dim = output_dim;
    weights.use_bias = false;
    const auto data = source.require_tensor_as_shape(
        prefix + ".weight",
        storage_type,
        {output_dim, input_dim, 1},
        {output_dim, input_dim});
    weights.weight = store.make_tensor(data.shape, data.type, data.bytes.data(), data.bytes.size());
    return weights;
}

RmsNormWeights load_rms_norm(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    float eps) {
    return {source.require_f32(prefix + ".weight", {hidden}), eps};
}

LayerNormWeights load_layer_norm(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    float eps = 1.0e-6F) {
    return {
        source.require_f32(prefix + ".weight", {hidden}),
        source.require_f32(prefix + ".bias", {hidden}),
        eps,
    };
}

std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> load_weights(
    const Qwen3TTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    const auto & source = *assets.speech_tokenizer_weights;
    auto weights = std::make_shared<Qwen3SpeechTokenizerDecoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "qwen3_tts.speech_tokenizer_decoder.weights",
        32ull * 1024ull * 1024ull);
    weights->config = load_decoder_config(assets);
    const auto & config = weights->config;
    const int64_t split_dim = config.codebook_dim / 2;

    for (int64_t layer = 0; layer < config.num_semantic_quantizers; ++layer) {
        const std::string prefix = "decoder.quantizer.rvq_first.vq.layers." + std::to_string(layer) + "._codebook.";
        weights->semantic_codebooks.push_back({normalized_codebook(source, prefix, config.codebook_size, split_dim)});
    }
    for (int64_t layer = 0; layer < config.num_quantizers - config.num_semantic_quantizers; ++layer) {
        const std::string prefix = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(layer) + "._codebook.";
        weights->acoustic_codebooks.push_back({normalized_codebook(source, prefix, config.codebook_size, split_dim)});
    }
    weights->semantic_output_proj = load_conv1x1_as_linear(
        *weights->store,
        source,
        "decoder.quantizer.rvq_first.output_proj",
        linear_weight_storage_type,
        split_dim,
        config.hidden_size);
    weights->acoustic_output_proj = load_conv1x1_as_linear(
        *weights->store,
        source,
        "decoder.quantizer.rvq_rest.output_proj",
        linear_weight_storage_type,
        split_dim,
        config.hidden_size);
    weights->pre_conv = load_conv(*weights->store, source, "decoder.pre_conv.conv", conv_weight_storage_type, config.hidden_size, config.latent_dim, 3);
    weights->transformer_input_proj = load_linear(
        *weights->store,
        source,
        "decoder.pre_transformer.input_proj",
        linear_weight_storage_type,
        config.latent_dim,
        config.hidden_size);
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "decoder.pre_transformer.layers." + std::to_string(layer);
        TransformerLayerWeights block;
        block.input_norm = load_rms_norm(source, prefix + ".input_layernorm", config.hidden_size, config.rms_norm_eps);
        block.post_norm = load_rms_norm(source, prefix + ".post_attention_layernorm", config.hidden_size, config.rms_norm_eps);
        block.attention.q = load_linear(*weights->store, source, prefix + ".self_attn.q_proj", linear_weight_storage_type, config.hidden_size, config.num_heads * config.head_dim, false);
        block.attention.k = load_linear(*weights->store, source, prefix + ".self_attn.k_proj", linear_weight_storage_type, config.hidden_size, config.num_kv_heads * config.head_dim, false);
        block.attention.v = load_linear(*weights->store, source, prefix + ".self_attn.v_proj", linear_weight_storage_type, config.hidden_size, config.num_kv_heads * config.head_dim, false);
        block.attention.o = load_linear(*weights->store, source, prefix + ".self_attn.o_proj", linear_weight_storage_type, config.num_heads * config.head_dim, config.hidden_size, false);
        block.mlp.gate = load_linear(*weights->store, source, prefix + ".mlp.gate_proj", linear_weight_storage_type, config.hidden_size, config.intermediate_size, false);
        block.mlp.up = load_linear(*weights->store, source, prefix + ".mlp.up_proj", linear_weight_storage_type, config.hidden_size, config.intermediate_size, false);
        block.mlp.down = load_linear(*weights->store, source, prefix + ".mlp.down_proj", linear_weight_storage_type, config.intermediate_size, config.hidden_size, false);
        block.attn_scale = source.require_f32(prefix + ".self_attn_layer_scale.scale", {config.hidden_size});
        block.mlp_scale = source.require_f32(prefix + ".mlp_layer_scale.scale", {config.hidden_size});
        weights->transformer_layers.push_back(std::move(block));
    }
    weights->transformer_norm = load_rms_norm(source, "decoder.pre_transformer.norm", config.hidden_size, config.rms_norm_eps);
    weights->transformer_output_proj = load_linear(
        *weights->store,
        source,
        "decoder.pre_transformer.output_proj",
        linear_weight_storage_type,
        config.hidden_size,
        config.latent_dim);

    for (size_t i = 0; i < config.upsampling_ratios.size(); ++i) {
        const std::string prefix = "decoder.upsample." + std::to_string(i);
        UpsampleStageWeights stage;
        stage.upconv = load_conv_transpose(
            *weights->store,
            source,
            prefix + ".0.conv",
            conv_weight_storage_type,
            config.latent_dim,
            config.latent_dim,
            config.upsampling_ratios[i],
            config.upsampling_ratios[i]);
        stage.convnext.dwconv = load_conv(
            *weights->store,
            source,
            prefix + ".1.dwconv.conv",
            conv_weight_storage_type,
            config.latent_dim,
            config.latent_dim,
            7,
            1,
            1,
            config.latent_dim);
        stage.convnext.norm = load_layer_norm(source, prefix + ".1.norm", config.latent_dim);
        stage.convnext.pwconv1 = load_linear(*weights->store, source, prefix + ".1.pwconv1", linear_weight_storage_type, config.latent_dim, config.latent_dim * 4);
        stage.convnext.pwconv2 = load_linear(*weights->store, source, prefix + ".1.pwconv2", linear_weight_storage_type, config.latent_dim * 4, config.latent_dim);
        stage.convnext.gamma = source.require_f32(prefix + ".1.gamma", {config.latent_dim});
        weights->upsample_stages.push_back(std::move(stage));
    }

    weights->decoder_input_conv = load_conv(*weights->store, source, "decoder.decoder.0.conv", conv_weight_storage_type, config.latent_dim, config.decoder_dim, 7);
    int64_t channels = config.decoder_dim;
    for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
        const std::string prefix = "decoder.decoder." + std::to_string(i + 1) + ".block";
        const int64_t out_channels = channels / 2;
        DecoderBlockWeights block;
        block.input_alpha = source.require_f32(prefix + ".0.alpha", {channels});
        block.input_beta = source.require_f32(prefix + ".0.beta", {channels});
        block.upconv = load_conv_transpose(
            *weights->store,
            source,
            prefix + ".1.conv",
            conv_weight_storage_type,
            channels,
            out_channels,
            config.upsample_rates[i] * 2,
            config.upsample_rates[i]);
        for (int unit_index = 0; unit_index < 3; ++unit_index) {
            const std::string unit = prefix + "." + std::to_string(unit_index + 2);
            ResidualUnitWeights residual;
            residual.act1_alpha = source.require_f32(unit + ".act1.alpha", {out_channels});
            residual.act1_beta = source.require_f32(unit + ".act1.beta", {out_channels});
            residual.conv1 = load_conv(*weights->store, source, unit + ".conv1.conv", conv_weight_storage_type, out_channels, out_channels, 7, 1, unit_index == 0 ? 1 : unit_index == 1 ? 3 : 9);
            residual.act2_alpha = source.require_f32(unit + ".act2.alpha", {out_channels});
            residual.act2_beta = source.require_f32(unit + ".act2.beta", {out_channels});
            residual.conv2 = load_conv(*weights->store, source, unit + ".conv2.conv", conv_weight_storage_type, out_channels, out_channels, 1);
            block.residual_units.push_back(std::move(residual));
        }
        weights->decoder_blocks.push_back(std::move(block));
        channels = out_channels;
    }
    weights->output_alpha = source.require_f32("decoder.decoder.5.alpha", {channels});
    weights->output_beta = source.require_f32("decoder.decoder.5.beta", {channels});
    weights->output_conv = load_conv(*weights->store, source, "decoder.decoder.6.conv", conv_weight_storage_type, channels, 1, 7);
    weights->store->upload();
    return weights;
}

core::TensorValue causal_conv1d(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const Conv1dWeights & weights,
    core::ConstantTensorCache & constants,
    DecoderStreamGraphState * stream = nullptr) {
    const int64_t kernel_extent = (weights.kernel - 1) * weights.dilation + 1;
    const int64_t left_pad = kernel_extent - weights.stride;
    const int64_t length = input.shape.dims[2];
    const float n_frames = static_cast<float>(length - kernel_extent + left_pad) / static_cast<float>(weights.stride) + 1.0F;
    const int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * weights.stride + (kernel_extent - left_pad);
    const int64_t right_pad = std::max<int64_t>(0, ideal_length - length);
    core::TensorValue padded_value;
    if (stream != nullptr) {
        padded_value = stream->convolution_input(build_ctx, &weights, input, left_pad);
    } else {
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
        padded_value = core::wrap_tensor(
            padded,
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + left_pad + right_pad}),
            GGML_TYPE_F32);
    }
    auto * padded = padded_value.tensor;
    if (weights.weight.type != GGML_TYPE_F32 && weights.weight.type != GGML_TYPE_F16) {
        throw std::runtime_error(
            std::string("Qwen3 speech decoder depthwise conv does not support weight type: ") +
            ggml_type_name(weights.weight.type));
    }
    ggml_tensor * result = nullptr;
    if (weights.groups == weights.in_channels) {
        ggml_tensor * bias = nullptr;
        if (weights.use_bias) {
            if (!weights.bias.has_value()) {
                throw std::runtime_error("Qwen3 speech decoder depthwise conv requires bias");
            }
            bias = core::reshape_tensor(build_ctx, *weights.bias, core::TensorShape::from_dims({weights.out_channels, 1})).tensor;
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
                static_cast<int>(weights.stride),
                0,
                static_cast<int>(weights.dilation));
            if (bias != nullptr) {
                batch_output = ggml_add(build_ctx.ggml, batch_output, bias);
            }
            batch_output = ggml_reshape_3d(build_ctx.ggml, batch_output, batch_output->ne[0], batch_output->ne[1], 1);
            result = result == nullptr ? batch_output : ggml_concat(build_ctx.ggml, result, batch_output, 2);
        }
        return core::wrap_tensor(
            result,
            core::TensorShape::from_dims({input.shape.dims[0], weights.out_channels, result->ne[0]}),
            GGML_TYPE_F32);
    } else {
        return modules::Conv1dModule({
            weights.in_channels,
            weights.out_channels,
            weights.kernel,
            static_cast<int>(weights.stride),
            0,
            static_cast<int>(weights.dilation),
            weights.use_bias,
        }).build(build_ctx, padded_value, binding::conv1d_data(constants, weights.weight, weights.bias));
    }
}

core::TensorValue causal_conv_transpose1d(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const ConvTranspose1dWeights & weights,
    core::ConstantTensorCache & constants,
    DecoderStreamGraphState * stream = nullptr) {
    const int64_t right_trim = weights.kernel - weights.stride;
    if (stream != nullptr) {
        if (right_trim < 0) throw std::runtime_error("Qwen3 stateful transposed convolution requires kernel >= stride");
        auto raw = modules::ConvTranspose1dModule({
            weights.in_channels, weights.out_channels, weights.kernel,
            static_cast<int>(weights.stride), 0, 1, false,
        }).build(build_ctx, input, binding::conv_transpose1d_data(constants, weights.weight));
        auto output = stream->transposed_convolution_output(
            build_ctx, &weights, raw, input.shape.dims[2] * weights.stride, right_trim);
        if (weights.use_bias) {
            if (!weights.bias) throw std::runtime_error("Qwen3 transposed convolution requires bias");
            auto bias = core::reshape_tensor(build_ctx, *weights.bias,
                core::TensorShape::from_dims({1, weights.out_channels, 1}));
            output = modules::AddModule(false, true).build(build_ctx, output, bias);
        }
        return output;
    }
    auto output_bct = modules::ConvTranspose1dModule({
        weights.in_channels,
        weights.out_channels,
        weights.kernel,
        static_cast<int>(weights.stride),
        0,
        1,
        weights.use_bias,
    }).build(build_ctx, input, binding::conv_transpose1d_data(constants, weights.weight, weights.bias));
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
                weights.out_channels,
                input.shape.dims[0],
                output_bct.tensor->nb[1],
                output_bct.tensor->nb[2],
                0)),
        core::TensorShape::from_dims({input.shape.dims[0], weights.out_channels, trimmed_frames}),
        GGML_TYPE_F32);
}

std::vector<float> snake_alpha_exp(const std::vector<float> & alpha) {
    std::vector<float> out(alpha.size());
    std::transform(alpha.begin(), alpha.end(), out.begin(), [](float value) { return std::exp(value); });
    return out;
}

std::vector<float> snake_inv_beta_exp(const std::vector<float> & beta) {
    std::vector<float> out(beta.size());
    std::transform(beta.begin(), beta.end(), out.begin(), [](float value) { return 1.0F / (std::exp(value) + 1.0e-9F); });
    return out;
}

core::TensorValue snake_beta(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    core::ConstantTensorCache & constants,
    const std::vector<float> & alpha,
    const std::vector<float> & beta) {
    auto alpha_exp = constants.make_f32(
        core::TensorShape::from_dims({1, static_cast<int64_t>(alpha.size()), 1}),
        snake_alpha_exp(alpha));
    auto inv_beta_exp = constants.make_f32(
        core::TensorShape::from_dims({1, static_cast<int64_t>(beta.size()), 1}),
        snake_inv_beta_exp(beta));
    auto * periodic = ggml_sqr(
        build_ctx.ggml,
        ggml_sin(build_ctx.ggml, ggml_mul(build_ctx.ggml, input.tensor, alpha_exp.tensor)));
    return core::wrap_tensor(
        ggml_add(build_ctx.ggml, input.tensor, ggml_mul(build_ctx.ggml, periodic, inv_beta_exp.tensor)),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue codebook_decode(
    core::ModuleBuildContext & build_ctx,
    ggml_tensor * codes_t_b,
    const CodebookWeights & codebook,
    int64_t dim,
    int64_t size,
    core::ConstantTensorCache & constants) {
    auto indices = core::wrap_tensor(
        codes_t_b,
        core::TensorShape::from_dims({codes_t_b->ne[1], codes_t_b->ne[0]}),
        GGML_TYPE_I32);
    indices = core::ensure_backend_addressable_layout(build_ctx, indices);
    const auto table = constants.make_f32(core::TensorShape::from_dims({size, dim}), codebook.embedding);
    return modules::CodebookLookupModule({size, dim}).build(build_ctx, indices, table);
}

core::TensorValue quantizer_decode(
    ggml_context * ctx,
    core::ModuleBuildContext & build_ctx,
    ggml_tensor * codes_t_q_b,
    const Qwen3SpeechTokenizerDecoderWeights & weights,
    core::ConstantTensorCache & constants) {
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
        auto decoded = codebook_decode(
            build_ctx,
            code_slice,
            weights.semantic_codebooks[static_cast<size_t>(group)],
            split_dim,
            config.codebook_size,
            constants);
        semantic_sum = semantic_sum.valid() ? modules::AddModule{}.build(build_ctx, semantic_sum, decoded) : decoded;
    }
    semantic_sum = modules::LinearModule(binding::linear_config(
        weights.semantic_output_proj.input_dim,
        weights.semantic_output_proj.output_dim,
        weights.semantic_output_proj.use_bias))
        .build(build_ctx, semantic_sum, binding::linear_data(constants, weights.semantic_output_proj.weight, weights.semantic_output_proj.bias));

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
        auto decoded = codebook_decode(
            build_ctx,
            code_slice,
            weights.acoustic_codebooks[static_cast<size_t>(group)],
            split_dim,
            config.codebook_size,
            constants);
        acoustic_sum = acoustic_sum.valid() ? modules::AddModule{}.build(build_ctx, acoustic_sum, decoded) : decoded;
    }
    acoustic_sum = modules::LinearModule(binding::linear_config(
        weights.acoustic_output_proj.input_dim,
        weights.acoustic_output_proj.output_dim,
        weights.acoustic_output_proj.use_bias))
        .build(build_ctx, acoustic_sum, binding::linear_data(constants, weights.acoustic_output_proj.weight, weights.acoustic_output_proj.bias));
    return modules::AddModule{}.build(build_ctx, semantic_sum, acoustic_sum);
}

core::TensorValue attention(
    ggml_context * ctx,
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    ggml_tensor * positions,
    const core::TensorValue & attention_mask,
    const AttentionWeights & weights,
    const DecoderConfig & config,
    core::ConstantTensorCache & constants,
    Qwen3TTSPerfMode perf_mode,
    DecoderStreamGraphState * stream = nullptr) {
    const int64_t kv_repeat = config.num_heads / config.num_kv_heads;
    auto q_value = modules::LinearModule(binding::linear_config(weights.q.input_dim, weights.q.output_dim, weights.q.use_bias))
                       .build(build_ctx, input, binding::linear_data(constants, weights.q.weight, weights.q.bias));
    auto k_value = modules::LinearModule(binding::linear_config(weights.k.input_dim, weights.k.output_dim, weights.k.use_bias))
                       .build(build_ctx, input, binding::linear_data(constants, weights.k.weight, weights.k.bias));
    auto v_value = modules::LinearModule(binding::linear_config(weights.v.input_dim, weights.v.output_dim, weights.v.use_bias))
                       .build(build_ctx, input, binding::linear_data(constants, weights.v.weight, weights.v.bias));
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
    if (stream != nullptr) {
        auto context = stream->attention_input(build_ctx, &weights, k_heads, v_heads, config.sliding_window);
        k_heads = context.first;
        v_heads = context.second;
    }
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
        perf_mode == Qwen3TTSPerfMode::FlashAttention
            ? modules::ScaledDotProductAttentionLowering::Flash
            : modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::NonCausal,
    }).build(
        build_ctx,
        q_heads,
        k_heads,
        v_heads,
        attention_mask);
    context = core::ensure_backend_addressable_layout(build_ctx, context);
    return modules::LinearModule(binding::linear_config(weights.o.input_dim, weights.o.output_dim, weights.o.use_bias))
        .build(
            build_ctx,
            core::reshape_tensor(build_ctx, context, core::TensorShape::from_dims({batch, seq, config.num_heads * config.head_dim})),
            binding::linear_data(constants, weights.o.weight, weights.o.bias));
}

core::TensorValue mlp(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const MlpWeights & weights,
    core::ConstantTensorCache & constants) {
    auto gate_linear = modules::LinearModule(binding::linear_config(weights.gate.input_dim, weights.gate.output_dim, weights.gate.use_bias))
                           .build(build_ctx, input, binding::linear_data(constants, weights.gate.weight, weights.gate.bias));
    auto gate = modules::SiluModule{}.build(build_ctx, gate_linear);
    auto up = modules::LinearModule(binding::linear_config(weights.up.input_dim, weights.up.output_dim, weights.up.use_bias))
                  .build(build_ctx, input, binding::linear_data(constants, weights.up.weight, weights.up.bias));
    return modules::LinearModule(binding::linear_config(weights.down.input_dim, weights.down.output_dim, weights.down.use_bias))
        .build(build_ctx, modules::MulModule{}.build(build_ctx, gate, up), binding::linear_data(constants, weights.down.weight, weights.down.bias));
}

core::TensorValue layer_scale(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const std::vector<float> & scale,
    core::ConstantTensorCache & constants) {
    return core::wrap_tensor(
        ggml_mul(
            build_ctx.ggml,
            input.tensor,
            constants.make_f32(core::TensorShape::from_dims({1, 1, static_cast<int64_t>(scale.size())}), scale).tensor),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue convnext(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input_bct,
    const ConvNeXtWeights & weights,
    core::ConstantTensorCache & constants,
    DecoderStreamGraphState * stream = nullptr) {
    auto hidden = causal_conv1d(build_ctx, input_bct, weights.dwconv, constants, stream);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
    hidden = modules::LayerNormModule({static_cast<int64_t>(weights.norm.weight.size()), weights.norm.eps, true, true})
                 .build(build_ctx, hidden, binding::norm(constants, weights.norm.weight, weights.norm.bias));
    hidden = modules::LinearModule(binding::linear_config(weights.pwconv1.input_dim, weights.pwconv1.output_dim, weights.pwconv1.use_bias))
                 .build(build_ctx, hidden, binding::linear_data(constants, weights.pwconv1.weight, weights.pwconv1.bias));
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(build_ctx, hidden);
    hidden = modules::LinearModule(binding::linear_config(weights.pwconv2.input_dim, weights.pwconv2.output_dim, weights.pwconv2.use_bias))
                 .build(build_ctx, hidden, binding::linear_data(constants, weights.pwconv2.weight, weights.pwconv2.bias));
    hidden = core::wrap_tensor(
        ggml_mul(
            build_ctx.ggml,
            hidden.tensor,
            constants.make_f32(core::TensorShape::from_dims({1, 1, static_cast<int64_t>(weights.gamma.size())}), weights.gamma).tensor),
        hidden.shape,
        GGML_TYPE_F32);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
    return modules::AddModule{}.build(build_ctx, input_bct, hidden);
}

core::TensorValue residual_unit(
    core::ModuleBuildContext & build_ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    core::ConstantTensorCache & constants,
    DecoderStreamGraphState * stream = nullptr) {
    auto hidden = snake_beta(
        build_ctx,
        input,
        constants,
        weights.act1_alpha,
        weights.act1_beta);
    hidden = causal_conv1d(build_ctx, hidden, weights.conv1, constants, stream);
    hidden = snake_beta(
        build_ctx,
        hidden,
        constants,
        weights.act2_alpha,
        weights.act2_beta);
    hidden = causal_conv1d(build_ctx, hidden, weights.conv2, constants, stream);
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

class Qwen3SpeechTokenizerDecoderGraph {
public:
    Qwen3SpeechTokenizerDecoderGraph(
        std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights,
        int64_t code_frames,
        core::ExecutionContext & execution_context,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes,
        Qwen3TTSPerfMode perf_mode,
        Qwen3SpeechDecoderStreamState * stream_state = nullptr)
        : weights_(std::move(weights)),
          stream_state_(stream_state),
          code_frames_(code_frames),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)),
          perf_mode_(perf_mode) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Qwen3 speech decoder graph requires weights");
        }
        if (code_frames_ <= 0) {
            throw std::runtime_error("Qwen3 speech decoder graph requires positive frame count");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 speech decoder backend is not initialized");
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
            throw std::runtime_error("failed to initialize Qwen3 speech decoder ggml context");
        }

        if (stream_state_ != nullptr) {
            stream_ = std::make_unique<DecoderStreamGraphState>(*stream_state_);
        }
        const int64_t context_frames = code_frames_ + (stream_state_ != nullptr ? config.sliding_window : 0);
        codes_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_I32, code_frames_, config.num_quantizers, 1);
        ggml_set_input(codes_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, code_frames_);
        ggml_set_input(positions_);
        if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention) {
            mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, context_frames, code_frames_, 1, 1);
        } else {
            mask_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, context_frames, code_frames_);
        }
        ggml_set_input(mask_);

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "qwen3_tts.speech_decoder",
            execution_context.backend_type(),
        };
        constants.begin_graph();
        auto hidden = quantizer_decode(ctx_.get(), build_ctx, codes_, *weights_, constants);
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        hidden = causal_conv1d(build_ctx, hidden, weights_->pre_conv, constants, stream_.get());
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        hidden = modules::LinearModule(binding::linear_config(
            weights_->transformer_input_proj.input_dim,
            weights_->transformer_input_proj.output_dim,
            weights_->transformer_input_proj.use_bias))
            .build(build_ctx, hidden, binding::linear_data(constants, weights_->transformer_input_proj.weight, weights_->transformer_input_proj.bias));
        for (const auto & layer : weights_->transformer_layers) {
            auto attn_in = modules::RMSNormModule({static_cast<int64_t>(layer.input_norm.weight.size()), layer.input_norm.eps, true, false})
                               .build(build_ctx, hidden, binding::norm(constants, layer.input_norm.weight));
            auto attention_mask = perf_mode_ == Qwen3TTSPerfMode::FlashAttention
                ? core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, code_frames_, context_frames}), GGML_TYPE_F16)
                : core::wrap_tensor(mask_, core::TensorShape::from_dims({code_frames_, context_frames}), GGML_TYPE_F32);
            auto attn_out = attention(ctx_.get(), build_ctx, attn_in, positions_, attention_mask, layer.attention, config, constants, perf_mode_, stream_.get());
            hidden = modules::AddModule{}.build(build_ctx, hidden, layer_scale(build_ctx, attn_out, layer.attn_scale, constants));
            auto mlp_in = modules::RMSNormModule({static_cast<int64_t>(layer.post_norm.weight.size()), layer.post_norm.eps, true, false})
                              .build(build_ctx, hidden, binding::norm(constants, layer.post_norm.weight));
            auto mlp_out = mlp(build_ctx, mlp_in, layer.mlp, constants);
            hidden = modules::AddModule{}.build(build_ctx, hidden, layer_scale(build_ctx, mlp_out, layer.mlp_scale, constants));
        }
        hidden = modules::RMSNormModule({static_cast<int64_t>(weights_->transformer_norm.weight.size()), weights_->transformer_norm.eps, true, false})
                     .build(build_ctx, hidden, binding::norm(constants, weights_->transformer_norm.weight));
        hidden = modules::LinearModule(binding::linear_config(
            weights_->transformer_output_proj.input_dim,
            weights_->transformer_output_proj.output_dim,
            weights_->transformer_output_proj.use_bias))
            .build(build_ctx, hidden, binding::linear_data(constants, weights_->transformer_output_proj.weight, weights_->transformer_output_proj.bias));
        hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build_ctx, hidden);
        for (const auto & stage : weights_->upsample_stages) {
            hidden = causal_conv_transpose1d(build_ctx, hidden, stage.upconv, constants, stream_.get());
            hidden = convnext(build_ctx, hidden, stage.convnext, constants, stream_.get());
        }
        hidden = causal_conv1d(build_ctx, hidden, weights_->decoder_input_conv, constants, stream_.get());
        for (const auto & block : weights_->decoder_blocks) {
            hidden = snake_beta(
                build_ctx,
                hidden,
                constants,
                block.input_alpha,
                block.input_beta);
            hidden = causal_conv_transpose1d(build_ctx, hidden, block.upconv, constants, stream_.get());
            for (const auto & unit : block.residual_units) {
                hidden = residual_unit(build_ctx, hidden, unit, constants, stream_.get());
            }
        }
        hidden = snake_beta(
            build_ctx,
            hidden,
            constants,
            weights_->output_alpha,
            weights_->output_beta);
        output_ = ggml_clamp(ctx_.get(), causal_conv1d(build_ctx, hidden, weights_->output_conv, constants, stream_.get()).tensor, -1.0F, 1.0F);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), graph_node_capacity(config), false);
        ggml_build_forward_expand(graph_, output_);
        if (stream_ != nullptr) {
            stream_->expand(build_ctx, graph_);
            stream_state_->allocate(backend_);
        }
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Qwen3 speech decoder graph");
        }
        positions_data_.resize(static_cast<size_t>(code_frames_));
        for (int64_t i = 0; i < code_frames_; ++i) {
            positions_data_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        const auto mask = stream_state_ != nullptr
            ? std::vector<float>(static_cast<size_t>(context_frames * code_frames_), kMaskNegInf)
            : make_mask(code_frames_, config.sliding_window);
        if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention) {
            mask_f16_data_.resize(mask.size());
            for (size_t index = 0; index < mask.size(); ++index) {
                mask_f16_data_[index] = ggml_fp32_to_fp16(mask[index]);
            }
        } else {
            mask_f32_data_ = mask;
        }
        upload_static_inputs();
    }

    ~Qwen3SpeechTokenizerDecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        const Qwen3SpeechTokenizerDecoderWeights & weights,
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
            throw std::runtime_error("Qwen3 speech decoder code count exceeds graph capacity");
        }
        if (stream_state_ != nullptr && input_frames != code_frames_) {
            throw std::runtime_error("Qwen3 stateful decoder requires an exact-length graph");
        }
        const auto upload_start = Clock::now();
        // Cached GGML graphs may reuse backend allocations whose input contents are
        // not guaranteed to survive a prior execution. Restore every declared input,
        // not only the request-varying codes, before replaying a retained graph.
        if (stream_ != nullptr) {
            const int64_t start = stream_state_->frames;
            const int64_t window = weights_->config.sliding_window;
            const int64_t keys = window + code_frames_;
            for (int64_t q = 0; q < code_frames_; ++q) {
                positions_data_[static_cast<size_t>(q)] = static_cast<int32_t>(start + q);
                for (int64_t k = 0; k < keys; ++k) {
                    const int64_t position = start - window + k;
                    const float value = position >= 0 && position <= start + q && position > start + q - window
                        ? 0.0F : kMaskNegInf;
                    const size_t index = static_cast<size_t>(k + keys * q);
                    if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention) mask_f16_data_[index] = ggml_fp32_to_fp16(value);
                    else mask_f32_data_[index] = value;
                }
            }
        }
        upload_static_inputs();
        std::vector<int32_t> tensor_codes(expected, 0);
        for (int64_t frame = 0; frame < input_frames; ++frame) {
            for (int64_t group = 0; group < weights_->config.num_quantizers; ++group) {
                tensor_codes[static_cast<size_t>(frame + code_frames_ * group)] =
                    codes[static_cast<size_t>(frame * weights_->config.num_quantizers + group)];
            }
        }
        ggml_backend_tensor_set(codes_, tensor_codes.data(), 0, tensor_codes.size() * sizeof(int32_t));
        last_input_upload_ms_ = engine::debug::elapsed_ms(upload_start, Clock::now());
        const auto compute_start = Clock::now();
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        last_graph_compute_ms_ = engine::debug::elapsed_ms(compute_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 speech decoder graph compute failed");
        }
        const auto read_start = Clock::now();
        std::vector<float> audio(static_cast<size_t>(waveform_frames_), 0.0F);
        ggml_backend_tensor_get(output_, audio.data(), 0, audio.size() * sizeof(float));
        last_output_read_ms_ = engine::debug::elapsed_ms(read_start, Clock::now());
        if (stream_state_ != nullptr) stream_state_->frames += input_frames;
        return audio;
    }

    double last_input_upload_ms() const noexcept {
        return last_input_upload_ms_;
    }

    double last_graph_compute_ms() const noexcept {
        return last_graph_compute_ms_;
    }

    double last_output_read_ms() const noexcept {
        return last_output_read_ms_;
    }

private:
    void upload_static_inputs() {
        ggml_backend_tensor_set(
            positions_,
            positions_data_.data(),
            0,
            positions_data_.size() * sizeof(int32_t));
        if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention) {
            ggml_backend_tensor_set(
                mask_,
                mask_f16_data_.data(),
                0,
                mask_f16_data_.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(
                mask_,
                mask_f32_data_.data(),
                0,
                mask_f32_data_.size() * sizeof(float));
        }
    }

    std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights_;
    Qwen3SpeechDecoderStreamState * stream_state_ = nullptr;
    std::unique_ptr<DecoderStreamGraphState> stream_;
    int64_t code_frames_ = 0;
    int64_t waveform_frames_ = 0;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * codes_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_data_;
    std::vector<float> mask_f32_data_;
    std::vector<ggml_fp16_t> mask_f16_data_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    Qwen3TTSPerfMode perf_mode_ = Qwen3TTSPerfMode::Standard;
    double last_input_upload_ms_ = 0.0;
    double last_graph_compute_ms_ = 0.0;
    double last_output_read_ms_ = 0.0;
};

Qwen3SpeechTokenizerDecoderRuntime::Qwen3SpeechTokenizerDecoderRuntime(
    std::shared_ptr<const Qwen3TTSAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type,
    Qwen3TTSPerfMode perf_mode,
    Qwen3SpeechTokenizerDecoderRknn rknn,
    bool stateful_stream)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes),
      perf_mode_(perf_mode) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Qwen3 speech tokenizer decoder requires assets");
    }
    if (stateful_stream && !rknn.model_path.empty()) {
        throw std::runtime_error("qwen3_tts.stream_decoder=stateful requires the GGML speech decoder");
    }
    if (stateful_stream) {
        stream_state_ = std::make_unique<Qwen3SpeechDecoderStreamState>(graph_arena_bytes_);
    }
    if (!rknn.model_path.empty()) {
        // The NPU path owns the whole decoder stage, so the ggml decoder
        // weights (hundreds of MB) are never loaded; only the config is needed,
        // for shape validation and the RoPE/KV-window geometry.
        auto config_only = std::make_shared<Qwen3SpeechTokenizerDecoderWeights>();
        config_only->config = load_decoder_config(*assets_);
        weights_ = std::move(config_only);
        rknn_ = std::make_unique<RknnSpeechDecoder>(
            std::move(rknn.model_path),
            std::move(rknn.first_model_path),
            std::move(rknn.window_path),
            weights_->config.num_quantizers,
            weights_->config.head_dim,
            weights_->config.sliding_window,
            weights_->config.rope_theta);
        return;
    }
    weights_ = load_weights(
        *assets_,
        execution_context_->backend(),
        execution_context_->backend_type(),
        linear_weight_storage_type,
        conv_weight_storage_type);
    constants_ = std::make_unique<core::ConstantTensorCache>(
        execution_context_->backend(),
        std::max(1, execution_context_->config().threads),
        "qwen3_tts.speech_tokenizer_decoder.constants",
        constant_context_bytes);
}

Qwen3SpeechTokenizerDecoderRuntime::~Qwen3SpeechTokenizerDecoderRuntime() = default;

runtime::AudioBuffer Qwen3SpeechTokenizerDecoderRuntime::decode(const Qwen3SpeechCodes & codec_codes) const {
    const auto total_start = Clock::now();
    if (codec_codes.frames <= 0 || codec_codes.code_groups != weights_->config.num_quantizers) {
        throw std::runtime_error("Qwen3 speech decoder received invalid codec shape");
    }
    if (static_cast<int64_t>(codec_codes.codes.size()) != codec_codes.frames * codec_codes.code_groups) {
        throw std::runtime_error("Qwen3 speech decoder codec payload size mismatch");
    }
    if (rknn_ != nullptr) {
        const auto start = Clock::now();
        auto samples = rknn_->decode(
            codec_codes.codes.data(), codec_codes.frames, codec_codes.code_groups);
        debug::timing_log_scalar(
            "qwen3_tts.speech_decoder.rknn.decode_ms",
            engine::debug::elapsed_ms(start, Clock::now()));
        return runtime::AudioBuffer{kSampleRate, 1, std::move(samples)};
    }
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(codec_codes.frames * kDecodeSamplesPerCode));
    double graph_build_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    int64_t chunks = 0;
    int64_t graph_rebuilds = 0;
    int64_t max_chunk_frames = 0;
    const bool optimized_cache_enabled =
        kStrixHaloGraphCacheEnabled && execution_context_->backend_type() == core::BackendType::Hip;
    for (int64_t start = 0; start < codec_codes.frames; start += kChunkCodes) {
        const int64_t end = std::min<int64_t>(start + kChunkCodes, codec_codes.frames);
        const int64_t context = start > kLeftContextCodes ? kLeftContextCodes : start;
        const int64_t chunk_start = start - context;
        const int64_t chunk_frames = end - chunk_start;
        max_chunk_frames = std::max(max_chunk_frames, chunk_frames);
        std::vector<int32_t> chunk(static_cast<size_t>(chunk_frames * codec_codes.code_groups), 0);
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            const int64_t src_frame = chunk_start + frame;
            const auto src = codec_codes.codes.begin() + static_cast<std::ptrdiff_t>(src_frame * codec_codes.code_groups);
            const auto dst = chunk.begin() + static_cast<std::ptrdiff_t>(frame * codec_codes.code_groups);
            std::copy(src, src + codec_codes.code_groups, dst);
        }
        const int threads = std::max(1, execution_context_->config().threads);
        auto * graph_slot = &graph_;
#if defined(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS)
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
            const auto build_start = Clock::now();
            graph.reset();
            auto replacement = std::make_unique<Qwen3SpeechTokenizerDecoderGraph>(
                weights_,
                chunk_frames,
                *execution_context_,
                *constants_,
                graph_arena_bytes_,
                perf_mode_);
            graph_build_ms += engine::debug::elapsed_ms(build_start, Clock::now());
            graph = std::move(replacement);
            ++graph_rebuilds;
        }
        auto decoded = graph->run(chunk.data(), chunk.size());
        input_upload_ms += graph->last_input_upload_ms();
        graph_compute_ms += graph->last_graph_compute_ms();
        output_read_ms += graph->last_output_read_ms();
        ++chunks;
        const int64_t drop = context * kDecodeSamplesPerCode;
        if (drop > static_cast<int64_t>(decoded.size())) {
            throw std::runtime_error("Qwen3 speech decoder chunk context exceeds decoded waveform");
        }
        const int64_t valid_samples = chunk_frames * kDecodeSamplesPerCode;
        if (valid_samples < drop || valid_samples > static_cast<int64_t>(decoded.size())) {
            throw std::runtime_error("Qwen3 speech decoder valid sample range exceeds decoded waveform");
        }
        samples.insert(
            samples.end(),
            decoded.begin() + static_cast<std::ptrdiff_t>(drop),
            decoded.begin() + static_cast<std::ptrdiff_t>(valid_samples));
    }
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.rebuilds", graph_rebuilds);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.build_ms", graph_build_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.input_upload_ms", input_upload_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.compute_ms", graph_compute_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.output_read_ms", output_read_ms);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return runtime::AudioBuffer{kSampleRate, 1, std::move(samples)};
}

runtime::AudioBuffer Qwen3SpeechTokenizerDecoderRuntime::decode_stream_chunk(
    const Qwen3SpeechCodes & codec_codes,
    int64_t context_frames,
    int64_t frames_per_chunk) const {
    if (frames_per_chunk <= 0 || context_frames < 0 || context_frames > kLeftContextCodes ||
        codec_codes.frames <= context_frames || codec_codes.frames - context_frames > frames_per_chunk ||
        codec_codes.code_groups != weights_->config.num_quantizers ||
        static_cast<int64_t>(codec_codes.codes.size()) != codec_codes.frames * codec_codes.code_groups) {
        throw std::runtime_error("Qwen3 streaming speech decoder received invalid codec shape");
    }
    const auto begin = Clock::now();
    if (rknn_ != nullptr) {
        // NPU path: on the stateful engines the left context is already in the
        // carried cache, so only the tail is fed and audio comes out in whole
        // model groups (the controller drops empty chunks); with a window engine
        // loaded the whole window is voiced per event from a zeroed cache and
        // the context's audio is dropped (see RknnSpeechDecoder::step).
        auto samples = rknn_->step(
            codec_codes.codes.data(),
            context_frames,
            codec_codes.frames - context_frames,
            codec_codes.code_groups,
            false);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.rknn.chunk_ms", engine::debug::elapsed_ms(begin));
        return runtime::AudioBuffer{kSampleRate, 1, std::move(samples)};
    }
    if (stream_state_ != nullptr) {
        if (stream_graph_capacity_ != frames_per_chunk) {
            stateful_graphs_.clear();
            stream_graph_capacity_ = frames_per_chunk;
        }
        double graph_build_ms = 0.0, graph_compute_ms = 0.0, input_upload_ms = 0.0, output_read_ms = 0.0;
        int64_t rebuilds = 0;
        auto decode_frames = [&](const int32_t * codes, int64_t frames) {
            const auto build_start = Clock::now();
            auto & graph = stateful_graphs_[frames];
            if (graph == nullptr) {
                graph = std::make_unique<Qwen3SpeechTokenizerDecoderGraph>(
                    weights_, frames, *execution_context_, *constants_, graph_arena_bytes_, perf_mode_, stream_state_.get());
                ++rebuilds;
            }
            graph_build_ms += engine::debug::elapsed_ms(build_start);
            auto samples = graph->run(codes, static_cast<size_t>(frames * codec_codes.code_groups));
            graph_compute_ms += graph->last_graph_compute_ms();
            input_upload_ms += graph->last_input_upload_ms();
            output_read_ms += graph->last_output_read_ms();
            return samples;
        };
        // Prime the reference tail once; subsequent windows repeat old context
        // that is already represented by convolution and attention state.
        if (stream_state_->frames == 0) {
            for (int64_t offset = 0; offset < context_frames; offset += frames_per_chunk) {
                decode_frames(codec_codes.codes.data() + offset * codec_codes.code_groups,
                              std::min(frames_per_chunk, context_frames - offset));
            }
        }
        auto decoded = decode_frames(codec_codes.codes.data() + context_frames * codec_codes.code_groups,
                                     codec_codes.frames - context_frames);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.rebuilds", rebuilds);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.build_ms", graph_build_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.compute_ms", graph_compute_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.input_upload_ms", input_upload_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.output_read_ms", output_read_ms);
        debug::timing_log_scalar("qwen3_tts.speech_decoder.total_ms", engine::debug::elapsed_ms(begin));
        return runtime::AudioBuffer{kSampleRate, 1, std::move(decoded)};
    }
    const bool initial = context_frames == 0;
    const int64_t capacity = initial ? codec_codes.frames : kLeftContextCodes + frames_per_chunk;
    auto & graph = initial ? stream_initial_graph_ : stream_graph_;
    const int threads = std::max(1, execution_context_->config().threads);
    const bool rebuild = graph == nullptr ||
        !graph->matches(*weights_, capacity, execution_context_->backend(), threads);
    if (rebuild) {
        graph.reset();
        graph = std::make_unique<Qwen3SpeechTokenizerDecoderGraph>(
            weights_, capacity, *execution_context_, *constants_, graph_arena_bytes_, perf_mode_);
    }
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.rebuilds", static_cast<int64_t>(rebuild));
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.build_ms", engine::debug::elapsed_ms(begin));
    // All decoder layers are causal. Right padding allows the initial and final
    // short chunks to use the same graph without changing their valid samples.
    auto decoded = graph->run(codec_codes.codes.data(), codec_codes.codes.size());
    debug::timing_log_scalar("qwen3_tts.speech_decoder.graph.compute_ms", graph->last_graph_compute_ms());
    debug::timing_log_scalar("qwen3_tts.speech_decoder.input_upload_ms", graph->last_input_upload_ms());
    debug::timing_log_scalar("qwen3_tts.speech_decoder.output_read_ms", graph->last_output_read_ms());
    decoded.resize(static_cast<size_t>(codec_codes.frames * kDecodeSamplesPerCode));
    decoded.erase(decoded.begin(), decoded.begin() + context_frames * kDecodeSamplesPerCode);
    debug::timing_log_scalar("qwen3_tts.speech_decoder.total_ms", engine::debug::elapsed_ms(begin));
    return runtime::AudioBuffer{kSampleRate, 1, std::move(decoded)};
}

void Qwen3SpeechTokenizerDecoderRuntime::reset_stream() const {
    if (stream_state_ != nullptr) stream_state_->reset();
    if (rknn_ != nullptr) {
        rknn_->reset();
    }
}

runtime::AudioBuffer Qwen3SpeechTokenizerDecoderRuntime::flush_stream() const {
    if (rknn_ == nullptr) {
        return runtime::AudioBuffer{kSampleRate, 1, {}};
    }
    // The stateful model voices whole groups; whatever is left below one group
    // at the end of the utterance is padded and voiced here.
    return runtime::AudioBuffer{kSampleRate, 1, rknn_->step(nullptr, 0, 0, weights_->config.num_quantizers, true)};
}

runtime::AudioBuffer Qwen3SpeechTokenizerDecoderRuntime::decode_and_trim_reference(
    const Qwen3SpeechCodes & reference_codes,
    const Qwen3SpeechCodes & generated_codes) const {
    if (reference_codes.code_groups != generated_codes.code_groups) {
        throw std::runtime_error("Qwen3 speech decoder reference/generated code group mismatch");
    }
    if (reference_codes.frames < 0 || generated_codes.frames < 0 || reference_codes.code_groups <= 0) {
        throw std::runtime_error("Qwen3 speech decoder reference/generated code shape is invalid");
    }
    if (reference_codes.frames > std::numeric_limits<int64_t>::max() - generated_codes.frames) {
        throw std::runtime_error("Qwen3 speech decoder combined frame count is too large");
    }
    Qwen3SpeechCodes combined;
    combined.frames = reference_codes.frames + generated_codes.frames;
    combined.code_groups = reference_codes.code_groups;
    if (combined.frames > std::numeric_limits<int64_t>::max() / combined.code_groups) {
        throw std::runtime_error("Qwen3 speech decoder combined code count is too large");
    }
    const int64_t combined_code_count = combined.frames * combined.code_groups;
    if (static_cast<uint64_t>(combined_code_count) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Qwen3 speech decoder combined code count exceeds host size limits");
    }
    combined.codes.reserve(static_cast<size_t>(combined_code_count));
    combined.codes.insert(combined.codes.end(), reference_codes.codes.begin(), reference_codes.codes.end());
    combined.codes.insert(combined.codes.end(), generated_codes.codes.begin(), generated_codes.codes.end());
    auto audio = decode(combined);
    if (reference_codes.frames > std::numeric_limits<int64_t>::max() / kDecodeSamplesPerCode) {
        throw std::runtime_error("Qwen3 speech decoder reference sample count is too large");
    }
    const int64_t cut = reference_codes.frames * kDecodeSamplesPerCode;
    if (static_cast<uint64_t>(cut) > audio.samples.size()) {
        throw std::runtime_error("Qwen3 speech decoder reference trim is out of range");
    }
    audio.samples.erase(audio.samples.begin(), audio.samples.begin() + static_cast<std::ptrdiff_t>(cut));
    return audio;
}

}  // namespace engine::models::qwen3_tts

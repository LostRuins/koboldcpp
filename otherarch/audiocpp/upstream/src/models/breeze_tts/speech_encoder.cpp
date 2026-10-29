#include "engine/models/breeze_tts/speech_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::breeze_tts {

namespace {

using Clock = std::chrono::steady_clock;
namespace binding = modules::binding;

}  // namespace

constexpr int64_t kSampleRate = 24000;
constexpr int64_t kDownsampleRate = 1920;
constexpr int64_t kHiddenSize = 512;
constexpr int64_t kQuantizerDim = 256;
constexpr int64_t kCodebookSize = 2048;
constexpr int64_t kValidQuantizers = 16;
constexpr float kCodebookEps = 1.0e-5F;
constexpr std::array<modules::Conv1dConfig, 6> kEncoderConvConfigs{{
    {1, 64, 7, 1, 0, 1, true},
    {64, 128, 8, 4, 0, 1, true},
    {128, 256, 10, 5, 0, 1, true},
    {256, 512, 12, 6, 0, 1, true},
    {512, 1024, 16, 8, 0, 1, true},
    {1024, 512, 3, 1, 0, 1, true},
}};
constexpr std::array<modules::StreamingPadMode, 6> kEncoderConvPadModes{{
    modules::StreamingPadMode::Constant,
    modules::StreamingPadMode::Constant,
    modules::StreamingPadMode::Constant,
    modules::StreamingPadMode::Constant,
    modules::StreamingPadMode::Constant,
    modules::StreamingPadMode::Constant,
}};
constexpr modules::Conv1dConfig kDownsampleConvConfig{512, 512, 4, 2, 0, 1, false};
constexpr modules::Conv1dConfig kSemanticProjectionConfig{512, 256, 1, 1, 0, 1, false};
constexpr modules::Conv1dConfig kAcousticProjectionConfig{512, 256, 1, 1, 0, 1, false};

// The conv stack downsamples 960x before the transformer (strides 4*5*6*8).
constexpr int64_t kTransformerStride = 960;
// Conv chunks are kChunkSamples of new audio preceded by kChunkOverlapSamples of
// left context. The exact left context the stack needs is the sum of each
// layer's left pad scaled by the cumulative stride:
// 6+2 + 4+2*4 + 5*4+2*20 + 6*20+2*120 + 8*120+2*960 + 2*960 = 5240 samples.
constexpr int64_t kChunkSamples = 120000;  // 5 s at 24 kHz, 125 transformer frames
constexpr int64_t kChunkOverlapSamples = 9600;  // 10 transformer frames > 5240
constexpr int64_t kChunkCapacity = kChunkSamples + kChunkOverlapSamples;
static_assert(kChunkSamples % kTransformerStride == 0);
static_assert(kChunkOverlapSamples % kTransformerStride == 0);
constexpr int64_t kChunkFrames = kChunkCapacity / kTransformerStride;
// The transformer graph is built at frame capacities rounded up to this
// bucket, so reference lengths within a bucket share one graph instead of
// rebuilding per exact length.
constexpr int64_t kTransformerFrameBucket = kChunkSamples / kTransformerStride;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ResBlockWeights {
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
};

struct TransformerLayerWeights {
    modules::AttentionWeights attention;
    modules::FeedForwardWeights feed_forward;
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::LayerScaleWeights scale1;
    modules::LayerScaleWeights scale2;
};

struct BreezeSpeechEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<modules::Conv1dWeights> encoder_convs;
    std::vector<ResBlockWeights> residual_blocks;
    std::vector<TransformerLayerWeights> transformer_layers;
    modules::Conv1dWeights downsample;
    modules::Conv1dWeights semantic_projection;
    modules::Conv1dWeights acoustic_projection;
    std::vector<std::vector<float>> semantic_codebooks;
    std::vector<std::vector<float>> acoustic_codebooks;
};

core::TensorValue speech_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    modules::Conv1dConfig config,
    modules::StreamingPadMode pad_mode) {
    const int64_t effective_kernel = (config.kernel_size - 1) * config.dilation + 1;
    const int64_t left_pad = effective_kernel - config.stride;
    const int64_t right_pad = (config.stride - (input.shape.dims[2] % config.stride)) % config.stride;
    auto padded = input;
    if (left_pad > 0) {
        auto prefix = modules::SliceModule({2, 0, 1}).build(ctx, input);
        prefix = core::ensure_backend_addressable_layout(ctx, prefix);
        if (pad_mode == modules::StreamingPadMode::Constant) {
            prefix = core::wrap_tensor(ggml_scale(ctx.ggml, prefix.tensor, 0.0F), prefix.shape, GGML_TYPE_F32);
        }
        prefix = modules::RepeatModule({
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], left_pad}),
        }).build(ctx, prefix);
        padded = modules::ConcatModule({2}).build(ctx, prefix, padded);
    }
    if (right_pad > 0) {
        auto suffix = modules::SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
        suffix = core::ensure_backend_addressable_layout(ctx, suffix);
        if (pad_mode == modules::StreamingPadMode::Constant) {
            suffix = core::wrap_tensor(ggml_scale(ctx.ggml, suffix.tensor, 0.0F), suffix.shape, GGML_TYPE_F32);
        }
        suffix = modules::RepeatModule({
            core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], right_pad}),
        }).build(ctx, suffix);
        padded = modules::ConcatModule({2}).build(ctx, padded, suffix);
    }
    config.padding = 0;
    return modules::Conv1dModule(config).build(ctx, padded, weights);
}

core::TensorValue speech_residual_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResBlockWeights & block,
    core::ConstantTensorCache &) {
    const int64_t channels = input.shape.dims[1];
    auto x = modules::EluModule{}.build(ctx, input);
    x = speech_conv(ctx, x, block.conv1, {channels, channels / 2, 3, 1, 0, 1, true}, modules::StreamingPadMode::Constant);
    x = modules::EluModule{}.build(ctx, x);
    x = speech_conv(ctx, x, block.conv2, {channels / 2, channels, 1, 1, 0, 1, true}, modules::StreamingPadMode::Constant);
    return modules::AddModule{}.build(ctx, input, x);
}

core::TensorValue mimi_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    modules::ScaledDotProductAttentionLowering lowering = modules::ScaledDotProductAttentionLowering::Flash) {
    constexpr int64_t kHeads = 8;
    constexpr int64_t kHeadDim = 64;
    auto q = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, {weights.attention.q_weight, weights.attention.q_bias});
    auto k = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, {weights.attention.k_weight, weights.attention.k_bias});
    auto v = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, {weights.attention.v_weight, weights.attention.v_bias});
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], kHeads, kHeadDim}));
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], kHeads, kHeadDim}));
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], kHeads, kHeadDim}));
    q = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NEOX}).build(ctx, q, positions);
    k = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NEOX}).build(ctx, k, positions);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        kHeadDim,
        lowering,
        GGML_PREC_F32,
        modules::AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kHiddenSize}));
    return modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
        .build(ctx, context, {weights.attention.out_weight, weights.attention.out_bias});
}

core::TensorValue transformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    const std::optional<core::TensorValue> & attention_mask,
    modules::ScaledDotProductAttentionLowering lowering = modules::ScaledDotProductAttentionLowering::Flash) {
    const modules::LayerNormModule norm({kHiddenSize, 1.0e-5F, true, true});
    auto x = norm.build(ctx, input, weights.norm1);
    auto attn_out = modules::LayerScaleModule{}.build(
        ctx,
        mimi_self_attention(ctx, x, positions, weights, attention_mask, lowering),
        weights.scale1);
    x = modules::AddModule{}.build(ctx, input, attn_out);
    auto y = norm.build(ctx, x, weights.norm2);
    y = modules::FeedForwardModule({
        kHiddenSize,
        2048,
        false,
        modules::GeluApproximation::ExactErf,
    }).build(ctx, y, weights.feed_forward);
    y = modules::LayerScaleModule{}.build(ctx, y, weights.scale2);
    return modules::AddModule{}.build(ctx, x, y);
}

std::vector<float> codebook_embedding(
    const assets::TensorSource & source,
    const std::string & prefix) {
    const auto cluster_usage = source.require_f32(prefix + "cluster_usage", {kCodebookSize});
    const auto embedding_sum = source.require_f32(prefix + "embed_sum", {kCodebookSize, kQuantizerDim});
    std::vector<float> embedding(embedding_sum.size(), 0.0F);
    for (int64_t code = 0; code < kCodebookSize; ++code) {
        const float denom = std::max(cluster_usage[static_cast<size_t>(code)], kCodebookEps);
        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            const size_t offset = static_cast<size_t>(code * kQuantizerDim + dim);
            embedding[offset] = embedding_sum[offset] / denom;
        }
    }
    return embedding;
}

int32_t nearest_code(const std::vector<float> & residual, const std::vector<float> & embedding) {
    int32_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity();
    for (int64_t code = 0; code < kCodebookSize; ++code) {
        float distance = 0.0F;
        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            const float diff =
                residual[static_cast<size_t>(dim)] - embedding[static_cast<size_t>(code * kQuantizerDim + dim)];
            distance += diff * diff;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = static_cast<int32_t>(code);
        }
    }
    return best;
}

std::vector<int32_t> quantize_projected(
    const std::vector<float> & semantic,
    const std::vector<float> & acoustic,
    int64_t frames,
    const BreezeSpeechEncoderWeights & weights) {
    if (weights.semantic_codebooks.empty() || static_cast<int64_t>(weights.acoustic_codebooks.size()) < kValidQuantizers - 1) {
        throw std::runtime_error("Breeze speech encoder has insufficient quantizer codebooks");
    }
    if (static_cast<int64_t>(semantic.size()) != kQuantizerDim * frames ||
        static_cast<int64_t>(acoustic.size()) != kQuantizerDim * frames) {
        throw std::runtime_error("Breeze speech encoder projected tensor size mismatch");
    }

    std::vector<int32_t> codes(static_cast<size_t>(frames * kValidQuantizers), 0);
    std::vector<float> residual(static_cast<size_t>(kQuantizerDim), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            residual[static_cast<size_t>(dim)] = semantic[static_cast<size_t>(dim * frames + frame)];
        }
        const int32_t semantic_code = nearest_code(residual, weights.semantic_codebooks[0]);
        codes[static_cast<size_t>(frame * kValidQuantizers)] = semantic_code;

        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            residual[static_cast<size_t>(dim)] = acoustic[static_cast<size_t>(dim * frames + frame)];
        }
        for (int64_t group = 1; group < kValidQuantizers; ++group) {
            const auto & embedding = weights.acoustic_codebooks[static_cast<size_t>(group - 1)];
            const int32_t code = nearest_code(residual, embedding);
            codes[static_cast<size_t>(frame * kValidQuantizers + group)] = code;
            for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
                residual[static_cast<size_t>(dim)] -= embedding[static_cast<size_t>(code * kQuantizerDim + dim)];
            }
        }
    }
    return codes;
}

std::shared_ptr<const BreezeSpeechEncoderWeights> load_weights(
    const BreezeTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    const auto & source = *assets.weights;
    auto weights = std::make_shared<BreezeSpeechEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "breeze_tts.speech_encoder.weights",
        32ull * 1024ull * 1024ull);

    const char * conv_prefixes[] = {
        "codec_model.encoder.encoder.layers.0.conv",
        "codec_model.encoder.encoder.layers.3.conv",
        "codec_model.encoder.encoder.layers.6.conv",
        "codec_model.encoder.encoder.layers.9.conv",
        "codec_model.encoder.encoder.layers.12.conv",
        "codec_model.encoder.encoder.layers.14.conv",
    };
    for (size_t i = 0; i < std::size(conv_prefixes); ++i) {
        const auto & conv = kEncoderConvConfigs[i];
        weights->encoder_convs.push_back(
            binding::conv1d_from_source(
                *weights->store,
                source,
                conv_prefixes[i],
                conv_weight_storage_type,
                conv.out_channels,
                conv.in_channels,
                conv.kernel_size,
                conv.use_bias));
    }

    const int residual_indices[] = {1, 4, 7, 10};
    const int64_t residual_channels[] = {64, 128, 256, 512};
    for (size_t i = 0; i < std::size(residual_indices); ++i) {
        const int idx = residual_indices[i];
        const int64_t channels = residual_channels[i];
        ResBlockWeights block;
        block.conv1 = binding::conv1d_from_source(
            *weights->store,
            source,
            "codec_model.encoder.encoder.layers." + std::to_string(idx) + ".block.1.conv",
            conv_weight_storage_type,
            channels / 2,
            channels,
            3,
            true);
        block.conv2 = binding::conv1d_from_source(
            *weights->store,
            source,
            "codec_model.encoder.encoder.layers." + std::to_string(idx) + ".block.3.conv",
            conv_weight_storage_type,
            channels,
            channels / 2,
            1,
            true);
        weights->residual_blocks.push_back(std::move(block));
    }

    for (int layer = 0; layer < 8; ++layer) {
        const std::string prefix = "codec_model.encoder.encoder_transformer.layers." + std::to_string(layer);
        TransformerLayerWeights block;
        block.attention.q_weight = weights->store->load_tensor(source, prefix + ".self_attn.q_proj.weight", linear_weight_storage_type, {512, 512});
        block.attention.k_weight = weights->store->load_tensor(source, prefix + ".self_attn.k_proj.weight", linear_weight_storage_type, {512, 512});
        block.attention.v_weight = weights->store->load_tensor(source, prefix + ".self_attn.v_proj.weight", linear_weight_storage_type, {512, 512});
        block.attention.out_weight = weights->store->load_tensor(source, prefix + ".self_attn.o_proj.weight", linear_weight_storage_type, {512, 512});
        block.feed_forward.fc1_weight = weights->store->load_tensor(source, prefix + ".mlp.fc1.weight", linear_weight_storage_type, {2048, 512});
        block.feed_forward.fc2_weight = weights->store->load_tensor(source, prefix + ".mlp.fc2.weight", linear_weight_storage_type, {512, 2048});
        block.norm1 = binding::norm_from_source(*weights->store, source, prefix + ".input_layernorm", 512);
        block.norm2 = binding::norm_from_source(*weights->store, source, prefix + ".post_attention_layernorm", 512);
        block.scale1 = binding::layer_scale_from_named_source(*weights->store, source, prefix + ".self_attn_layer_scale.scale");
        block.scale2 = binding::layer_scale_from_named_source(*weights->store, source, prefix + ".mlp_layer_scale.scale");
        weights->transformer_layers.push_back(std::move(block));
    }

    weights->downsample = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.encoder.downsample.conv",
        conv_weight_storage_type,
        kDownsampleConvConfig.out_channels,
        kDownsampleConvConfig.in_channels,
        kDownsampleConvConfig.kernel_size,
        kDownsampleConvConfig.use_bias);
    weights->semantic_projection = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
        conv_weight_storage_type,
        kSemanticProjectionConfig.out_channels,
        kSemanticProjectionConfig.in_channels,
        kSemanticProjectionConfig.kernel_size,
        kSemanticProjectionConfig.use_bias);
    weights->acoustic_projection = binding::conv1d_from_source(
        *weights->store,
        source,
        "codec_model.encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
        conv_weight_storage_type,
        kAcousticProjectionConfig.out_channels,
        kAcousticProjectionConfig.in_channels,
        kAcousticProjectionConfig.kernel_size,
        kAcousticProjectionConfig.use_bias);

    weights->semantic_codebooks.push_back(codebook_embedding(
        source,
        "codec_model.encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook."));

    for (int layer = 0; layer < 31; ++layer) {
        const std::string prefix =
            "codec_model.encoder.quantizer.acoustic_residual_vector_quantizer.layers." + std::to_string(layer) + ".codebook.";
        weights->acoustic_codebooks.push_back(codebook_embedding(source, prefix));
    }

    weights->store->upload();
    return weights;
}

// Conv stack runs on fixed-size chunks (plus left overlap), so its graph
// memory is constant regardless of reference length. Chunk outputs stitch
// exactly: chunk lengths are multiples of kTransformerStride, so no per-stage
// right padding occurs, and discarded overlap frames absorb the zero left
// pads that represent audio start in the first chunk.
class BreezeSpeechEncoderConvGraph {
public:
    BreezeSpeechEncoderConvGraph(
        std::shared_ptr<const BreezeSpeechEncoderWeights> weights,
        core::ExecutionContext & execution_context,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Breeze speech encoder conv graph requires weights");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Breeze speech encoder backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Breeze speech encoder conv ggml context");
        }

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "breeze_tts.speech_encoder.conv",
            execution_context.backend_type(),
        };
        auto x = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, kChunkCapacity}));
        input_ = x.tensor;

        constants.begin_graph();
        x = speech_conv(build_ctx, x, weights_->encoder_convs[0], kEncoderConvConfigs[0], kEncoderConvPadModes[0]);
        for (size_t i = 0; i < weights_->residual_blocks.size(); ++i) {
            x = speech_residual_block(build_ctx, x, weights_->residual_blocks[i], constants);
            x = modules::EluModule{}.build(build_ctx, x);
            x = speech_conv(build_ctx, x, weights_->encoder_convs[i + 1], kEncoderConvConfigs[i + 1], kEncoderConvPadModes[i + 1]);
        }
        x = modules::EluModule{}.build(build_ctx, x);
        x = speech_conv(build_ctx, x, weights_->encoder_convs.back(), kEncoderConvConfigs.back(), kEncoderConvPadModes.back());
        auto seq = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
        seq = core::ensure_backend_addressable_layout(build_ctx, seq);
        output_ = seq.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Breeze speech encoder conv graph");
        }
    }

    ~BreezeSpeechEncoderConvGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const BreezeSpeechEncoderWeights & weights, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const std::vector<float> & chunk_input) {
        if (static_cast<int64_t>(chunk_input.size()) != kChunkCapacity) {
            throw std::runtime_error("Breeze speech encoder conv chunk size mismatch");
        }
        ggml_backend_tensor_set(input_, chunk_input.data(), 0, chunk_input.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Breeze speech encoder conv graph compute failed");
        }
        std::vector<float> features(static_cast<size_t>(kHiddenSize * kChunkFrames));
        ggml_backend_tensor_get(output_, features.data(), 0, features.size() * sizeof(float));
        return features;
    }

private:
    std::shared_ptr<const BreezeSpeechEncoderWeights> weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

// Transformer + downsample + projections run once over the full frame
// sequence; at frame scale (960x downsampled) this graph is a few tens of MiB
// even for minute-long references. Attention is causal, so frames computed
// from right-padded tail regions never affect earlier frames.
class BreezeSpeechEncoderTransformerGraph {
public:
    BreezeSpeechEncoderTransformerGraph(
        std::shared_ptr<const BreezeSpeechEncoderWeights> weights,
        int64_t frames,
        core::ExecutionContext & execution_context,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes,
        bool allow_flash_attention = true)
        : weights_(std::move(weights)),
          allow_flash_attention_(allow_flash_attention),
          frames_(frames),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Breeze speech encoder transformer graph requires weights");
        }
        if (frames_ <= 0) {
            throw std::runtime_error("Breeze speech encoder transformer graph requires positive frame count");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Breeze speech encoder backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Breeze speech encoder transformer ggml context");
        }

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "breeze_tts.speech_encoder.transformer",
            execution_context.backend_type(),
        };
        auto seq = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, frames_, kHiddenSize}));
        input_ = seq.tensor;

        constants.begin_graph();
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, frames_);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({frames_}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, frames_, frames_, 1, 1);
        const auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, frames_, frames_}),
            GGML_TYPE_F16);
        for (const auto & layer : weights_->transformer_layers) {
            seq = transformer_block(
                build_ctx,
                seq,
                positions_value,
                layer,
                attention_mask,
                allow_flash_attention_ ? modules::ScaledDotProductAttentionLowering::Flash
                                       : modules::ScaledDotProductAttentionLowering::Explicit);
        }
        auto x = modules::TransposeModule({{0, 2, 1, 3}, seq.shape.rank}).build(build_ctx, seq);
        x = core::ensure_backend_addressable_layout(build_ctx, x);
        x = speech_conv(build_ctx, x, weights_->downsample, kDownsampleConvConfig, modules::StreamingPadMode::Replicate);
        auto semantic = speech_conv(build_ctx, x, weights_->semantic_projection, kSemanticProjectionConfig, modules::StreamingPadMode::Constant);
        auto acoustic = speech_conv(build_ctx, x, weights_->acoustic_projection, kAcousticProjectionConfig, modules::StreamingPadMode::Constant);
        output_frames_ = semantic.shape.dims[2];
        semantic_output_ = semantic.tensor;
        acoustic_output_ = acoustic.tensor;
        ggml_set_output(semantic_output_);
        ggml_set_output(acoustic_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, semantic_output_);
        ggml_build_forward_expand(graph_, acoustic_output_);
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Breeze speech encoder transformer graph");
        }
        positions_data_.resize(static_cast<size_t>(frames_));
        for (int64_t i = 0; i < frames_; ++i) {
            positions_data_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        if (attention_mask_ != nullptr) {
            auto mask = modules::causal_prefill_mask_values(1, frames_);
            attention_mask_data_ = std::move(mask);
        }
        upload_static_inputs();
    }

    ~BreezeSpeechEncoderTransformerGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const BreezeSpeechEncoderWeights & weights, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && frames_ == frames && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    BreezeSpeechEncoderOutput run(const std::vector<float> & features) {
        if (static_cast<int64_t>(features.size()) != kHiddenSize * frames_) {
            throw std::runtime_error("Breeze speech encoder transformer input size mismatch");
        }
        upload_static_inputs();
        ggml_backend_tensor_set(input_, features.data(), 0, features.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Breeze speech encoder transformer graph compute failed");
        }
        BreezeSpeechEncoderOutput out;
        out.semantic_projected.resize(static_cast<size_t>(kQuantizerDim * output_frames_));
        out.acoustic_projected.resize(static_cast<size_t>(kQuantizerDim * output_frames_));
        ggml_backend_tensor_get(semantic_output_, out.semantic_projected.data(), 0, out.semantic_projected.size() * sizeof(float));
        ggml_backend_tensor_get(acoustic_output_, out.acoustic_projected.data(), 0, out.acoustic_projected.size() * sizeof(float));
        return out;
    }

    int64_t output_frames() const noexcept {
        return output_frames_;
    }

private:
    void upload_static_inputs() {
        ggml_backend_tensor_set(
            positions_,
            positions_data_.data(),
            0,
            positions_data_.size() * sizeof(int32_t));
        if (attention_mask_ != nullptr) {
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_data_.data(),
                0,
                attention_mask_data_.size() * sizeof(ggml_fp16_t));
        }
    }

    std::shared_ptr<const BreezeSpeechEncoderWeights> weights_;
    bool allow_flash_attention_ = true;
    int64_t frames_ = 0;
    int64_t output_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * semantic_output_ = nullptr;
    ggml_tensor * acoustic_output_ = nullptr;
    std::vector<int32_t> positions_data_;
    std::vector<ggml_fp16_t> attention_mask_data_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

BreezeSpeechEncoderRuntime::BreezeSpeechEncoderRuntime(
    std::shared_ptr<const BreezeTTSAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type,
    core::AttentionPreference attention_preference)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Breeze speech encoder requires assets");
    }
    weights_ = load_weights(
        *assets_,
        execution_context_->backend(),
        execution_context_->backend_type(),
        linear_weight_storage_type,
        conv_weight_storage_type);
    // Mimi encoder self-attention head dim (kHeadDim in mimi_self_attention).
    allow_flash_attention_ = core::resolve_flash_attention(execution_context_->backend(), 64, attention_preference);
    engine::debug::trace_log_scalar("breeze_tts.attention.allow_encoder_flash", allow_flash_attention_);
    constants_ = std::make_unique<core::ConstantTensorCache>(
        execution_context_->backend(),
        std::max(1, execution_context_->config().threads),
        "breeze_tts.speech_encoder.constants",
        768ull * 1024ull * 1024ull);
}

BreezeSpeechEncoderRuntime::~BreezeSpeechEncoderRuntime() = default;

BreezeSpeechCodes BreezeSpeechEncoderRuntime::encode(const runtime::AudioBuffer & audio) const {
    const auto start = Clock::now();
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Breeze speech encoder execution context is missing");
    }
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("Breeze speech encoder requires non-empty reference audio");
    }
    const auto waveform = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(kSampleRate));
    const int64_t valid_samples = static_cast<int64_t>(waveform.size());
    const int64_t frames = std::max<int64_t>(1, (valid_samples + kDownsampleRate - 1) / kDownsampleRate);
    const int64_t transformer_frames = (valid_samples + kTransformerStride - 1) / kTransformerStride;
    const int64_t graph_frames =
        (transformer_frames + kTransformerFrameBucket - 1) / kTransformerFrameBucket * kTransformerFrameBucket;
    const int threads = std::max(1, execution_context_->config().threads);
    if (conv_graph_ == nullptr || !conv_graph_->matches(*weights_, execution_context_->backend(), threads)) {
        conv_graph_.reset();
        conv_graph_ = std::make_unique<BreezeSpeechEncoderConvGraph>(
            weights_,
            *execution_context_,
            *constants_,
            graph_arena_bytes_);
    }
    if (transformer_graph_ == nullptr ||
        !transformer_graph_->matches(*weights_, graph_frames, execution_context_->backend(), threads)) {
        transformer_graph_.reset();
        transformer_graph_ = std::make_unique<BreezeSpeechEncoderTransformerGraph>(
            weights_,
            graph_frames,
            *execution_context_,
            *constants_,
            graph_arena_bytes_,
            allow_flash_attention_);
    }

    std::vector<float> features(static_cast<size_t>(kHiddenSize * graph_frames));
    std::vector<float> chunk_input(static_cast<size_t>(kChunkCapacity));
    int64_t dst_frame = 0;
    for (int64_t pos = 0; pos < valid_samples; pos += kChunkSamples) {
        const int64_t overlap = pos > 0 ? kChunkOverlapSamples : 0;
        const int64_t fresh = std::min(kChunkSamples, valid_samples - pos);
        std::fill(chunk_input.begin(), chunk_input.end(), 0.0F);
        std::copy(
            waveform.begin() + (pos - overlap),
            waveform.begin() + (pos + fresh),
            chunk_input.begin());
        const auto chunk_features = conv_graph_->run(chunk_input);
        const int64_t skip_frames = overlap / kTransformerStride;
        const int64_t keep_frames = (fresh + kTransformerStride - 1) / kTransformerStride;
        std::copy_n(
            chunk_features.begin() + skip_frames * kHiddenSize,
            keep_frames * kHiddenSize,
            features.begin() + dst_frame * kHiddenSize);
        dst_frame += keep_frames;
    }
    // Pad unused bucket frames with the last real frame (not zeros): causal
    // attention keeps padding invisible to real frames, and the downsample
    // conv's Replicate right pad then sees the same value as an exact-length
    // graph would produce.
    for (int64_t f = transformer_frames; f < graph_frames; ++f) {
        std::copy_n(
            features.begin() + (transformer_frames - 1) * kHiddenSize,
            kHiddenSize,
            features.begin() + f * kHiddenSize);
    }

    auto out = transformer_graph_->run(features);
    const int64_t produced_frames = transformer_graph_->output_frames();
    if (produced_frames != frames) {
        // Projected outputs are channel-major with stride produced_frames;
        // drop the padding frames before quantization.
        auto slice_frames = [frames, produced_frames](std::vector<float> & projected) {
            std::vector<float> sliced(static_cast<size_t>(kQuantizerDim * frames));
            for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
                std::copy_n(
                    projected.begin() + dim * produced_frames,
                    frames,
                    sliced.begin() + dim * frames);
            }
            projected = std::move(sliced);
        };
        slice_frames(out.semantic_projected);
        slice_frames(out.acoustic_projected);
    }
    out.codes.frames = frames;
    out.codes.code_groups = kValidQuantizers;
    out.codes.codes = quantize_projected(out.semantic_projected, out.acoustic_projected, frames, *weights_);
    debug::timing_log_scalar("breeze_tts.speech_encoder.total_ms", engine::debug::elapsed_ms(start));
    return out.codes;
}

void BreezeSpeechEncoderRuntime::release_runtime_graphs() const {
    conv_graph_.reset();
    transformer_graph_.reset();
}

}  // namespace engine::models::breeze_tts

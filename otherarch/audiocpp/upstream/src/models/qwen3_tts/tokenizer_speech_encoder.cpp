#include "engine/models/qwen3_tts/tokenizer_speech_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::qwen3_tts {

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

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ConvWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int stride = 1;
    int dilation = 1;
    bool use_bias = true;
    modules::StreamingPadMode pad_mode = modules::StreamingPadMode::Constant;
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

struct ResBlockWeights {
    ConvWeights conv1;
    ConvWeights conv2;
};

struct TransformerLayerWeights {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
    core::TensorValue o;
    core::TensorValue fc1;
    core::TensorValue fc2;
    assets::TensorDataF32 norm1_weight;
    assets::TensorDataF32 norm1_bias;
    assets::TensorDataF32 norm2_weight;
    assets::TensorDataF32 norm2_bias;
    assets::TensorDataF32 scale1;
    assets::TensorDataF32 scale2;
};

struct CodebookWeights {
    std::vector<float> cluster_usage;
    std::vector<float> embedding_sum;
};

struct Qwen3SpeechTokenizerEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<ConvWeights> encoder_convs;
    std::vector<ResBlockWeights> residual_blocks;
    std::vector<TransformerLayerWeights> transformer_layers;
    ConvWeights downsample;
    ConvWeights semantic_projection;
    ConvWeights acoustic_projection;
    std::vector<CodebookWeights> semantic_codebooks;
    std::vector<CodebookWeights> acoustic_codebooks;
};

int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

ConvWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int stride = 1,
    int dilation = 1,
    bool use_bias = true,
    modules::StreamingPadMode pad_mode = modules::StreamingPadMode::Constant) {
    ConvWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.pad_mode = pad_mode;
    conv.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        conv.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    }
    return conv;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue repeat_frame(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t frame,
    int64_t count,
    bool zero) {
    auto one = modules::SliceModule({2, frame, 1}).build(ctx, input);
    one = core::ensure_backend_addressable_layout(ctx, one);
    if (zero) {
        one = core::wrap_tensor(ggml_scale(ctx.ggml, one.tensor, 0.0F), one.shape, GGML_TYPE_F32);
    }
    return modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], count})})
        .build(ctx, one);
}

core::TensorValue speech_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvWeights & conv,
    core::ConstantTensorCache & constants) {
    const int64_t effective_kernel = (conv.kernel - 1) * conv.dilation + 1;
    const int64_t left_pad = effective_kernel - conv.stride;
    const int64_t right_pad = (conv.stride - (input.shape.dims[2] % conv.stride)) % conv.stride;
    auto padded = input;
    if (left_pad > 0) {
        const bool zero = conv.pad_mode == modules::StreamingPadMode::Constant;
        auto prefix = repeat_frame(ctx, input, 0, left_pad, zero);
        padded = modules::ConcatModule({2}).build(ctx, prefix, padded);
    }
    if (right_pad > 0) {
        const bool zero = conv.pad_mode == modules::StreamingPadMode::Constant;
        auto suffix = repeat_frame(ctx, input, input.shape.dims[2] - 1, right_pad, zero);
        padded = modules::ConcatModule({2}).build(ctx, padded, suffix);
    }
    return modules::Conv1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        conv.stride,
        0,
        conv.dilation,
        conv.use_bias,
    }).build(ctx, padded, binding::conv1d_data(constants, conv.weight, conv.bias));
}

core::TensorValue speech_residual_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResBlockWeights & block,
    core::ConstantTensorCache & constants) {
    auto x = modules::EluModule{}.build(ctx, input);
    x = speech_conv(ctx, x, block.conv1, constants);
    x = modules::EluModule{}.build(ctx, x);
    x = speech_conv(ctx, x, block.conv2, constants);
    return modules::AddModule{}.build(ctx, input, x);
}

core::TensorValue mimi_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    core::ConstantTensorCache & constants,
    Qwen3TTSPerfMode perf_mode,
    const std::optional<core::TensorValue> & attention_mask) {
    constexpr int64_t kHeads = 8;
    constexpr int64_t kHeadDim = 64;
    auto q = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, binding::linear_data(constants, weights.q));
    auto k = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, binding::linear_data(constants, weights.k));
    auto v = modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
                 .build(ctx, input, binding::linear_data(constants, weights.v));
    q = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NEOX}).build(ctx, reshape_heads(ctx, q, kHeads, kHeadDim), positions);
    k = modules::RoPEModule({kHeadDim, GGML_ROPE_TYPE_NEOX}).build(ctx, reshape_heads(ctx, k, kHeads, kHeadDim), positions);
    v = reshape_heads(ctx, v, kHeads, kHeadDim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        kHeadDim,
        perf_mode == Qwen3TTSPerfMode::FlashAttention
            ? modules::ScaledDotProductAttentionLowering::Flash
            : modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kHiddenSize}));
    return modules::LinearModule(binding::linear_config(kHiddenSize, kHiddenSize, false))
        .build(ctx, context, binding::linear_data(constants, weights.o));
}

core::TensorValue transformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerLayerWeights & weights,
    core::ConstantTensorCache & constants,
    Qwen3TTSPerfMode perf_mode,
    const std::optional<core::TensorValue> & attention_mask) {
    const modules::LayerNormModule norm({kHiddenSize, 1.0e-5F, true, true});
    auto x = norm.build(ctx, input, binding::norm_data(constants, weights.norm1_weight, weights.norm1_bias));
    auto attn_out = modules::LayerScaleModule{}.build(
        ctx,
        mimi_self_attention(ctx, x, positions, weights, constants, perf_mode, attention_mask),
        binding::layer_scale_data(constants, weights.scale1));
    x = modules::AddModule{}.build(ctx, input, attn_out);
    auto y = norm.build(ctx, x, binding::norm_data(constants, weights.norm2_weight, weights.norm2_bias));
    y = modules::LinearModule(binding::linear_config(kHiddenSize, 2048, false))
            .build(ctx, y, binding::linear_data(constants, weights.fc1));
    y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
    y = modules::LinearModule(binding::linear_config(2048, kHiddenSize, false))
            .build(ctx, y, binding::linear_data(constants, weights.fc2));
    y = modules::LayerScaleModule{}.build(ctx, y, binding::layer_scale_data(constants, weights.scale2));
    return modules::AddModule{}.build(ctx, x, y);
}

std::vector<float> codebook_embedding(const CodebookWeights & weights) {
    if (static_cast<int64_t>(weights.cluster_usage.size()) != kCodebookSize ||
        static_cast<int64_t>(weights.embedding_sum.size()) != kCodebookSize * kQuantizerDim) {
        throw std::runtime_error("Qwen3 speech tokenizer codebook has invalid shape");
    }
    std::vector<float> embedding(weights.embedding_sum.size(), 0.0F);
    for (int64_t code = 0; code < kCodebookSize; ++code) {
        const float denom = std::max(weights.cluster_usage[static_cast<size_t>(code)], kCodebookEps);
        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            const size_t offset = static_cast<size_t>(code * kQuantizerDim + dim);
            embedding[offset] = weights.embedding_sum[offset] / denom;
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
    const Qwen3SpeechTokenizerEncoderWeights & weights) {
    std::vector<std::vector<float>> semantic_embeddings;
    std::vector<std::vector<float>> acoustic_embeddings;
    semantic_embeddings.reserve(weights.semantic_codebooks.size());
    acoustic_embeddings.reserve(weights.acoustic_codebooks.size());
    for (const auto & codebook : weights.semantic_codebooks) {
        semantic_embeddings.push_back(codebook_embedding(codebook));
    }
    for (const auto & codebook : weights.acoustic_codebooks) {
        acoustic_embeddings.push_back(codebook_embedding(codebook));
    }

    if (semantic_embeddings.empty() || static_cast<int64_t>(acoustic_embeddings.size()) < kValidQuantizers - 1) {
        throw std::runtime_error("Qwen3 speech tokenizer has insufficient quantizer codebooks");
    }
    if (static_cast<int64_t>(semantic.size()) != kQuantizerDim * frames ||
        static_cast<int64_t>(acoustic.size()) != kQuantizerDim * frames) {
        throw std::runtime_error("Qwen3 speech tokenizer projected tensor size mismatch");
    }

    std::vector<int32_t> codes(static_cast<size_t>(frames * kValidQuantizers), 0);
    std::vector<float> residual(static_cast<size_t>(kQuantizerDim), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            residual[static_cast<size_t>(dim)] = semantic[static_cast<size_t>(dim * frames + frame)];
        }
        const int32_t semantic_code = nearest_code(residual, semantic_embeddings[0]);
        codes[static_cast<size_t>(frame * kValidQuantizers)] = semantic_code;

        for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
            residual[static_cast<size_t>(dim)] = acoustic[static_cast<size_t>(dim * frames + frame)];
        }
        for (int64_t group = 1; group < kValidQuantizers; ++group) {
            const auto & embedding = acoustic_embeddings[static_cast<size_t>(group - 1)];
            const int32_t code = nearest_code(residual, embedding);
            codes[static_cast<size_t>(frame * kValidQuantizers + group)] = code;
            for (int64_t dim = 0; dim < kQuantizerDim; ++dim) {
                residual[static_cast<size_t>(dim)] -= embedding[static_cast<size_t>(code * kQuantizerDim + dim)];
            }
        }
    }
    return codes;
}

std::shared_ptr<const Qwen3SpeechTokenizerEncoderWeights> load_weights(
    const Qwen3TTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    const auto & source = *assets.speech_tokenizer_weights;
    auto weights = std::make_shared<Qwen3SpeechTokenizerEncoderWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "qwen3_tts.speech_tokenizer_encoder.weights",
        32ull * 1024ull * 1024ull);

    const char * conv_prefixes[] = {
        "encoder.encoder.layers.0.conv",
        "encoder.encoder.layers.3.conv",
        "encoder.encoder.layers.6.conv",
        "encoder.encoder.layers.9.conv",
        "encoder.encoder.layers.12.conv",
        "encoder.encoder.layers.14.conv",
    };
    const int64_t conv_channels[][4] = {
        {1, 64, 7, 1},
        {64, 128, 8, 4},
        {128, 256, 10, 5},
        {256, 512, 12, 6},
        {512, 1024, 16, 8},
        {1024, 512, 3, 1},
    };
    for (size_t i = 0; i < std::size(conv_prefixes); ++i) {
        weights->encoder_convs.push_back(load_conv(
            *weights->store,
            source,
            conv_prefixes[i],
            conv_weight_storage_type,
            conv_channels[i][1],
            conv_channels[i][0],
            conv_channels[i][2],
            static_cast<int>(conv_channels[i][3])));
    }

    const int residual_indices[] = {1, 4, 7, 10};
    const int64_t residual_channels[] = {64, 128, 256, 512};
    for (size_t i = 0; i < std::size(residual_indices); ++i) {
        const int idx = residual_indices[i];
        const int64_t channels = residual_channels[i];
        ResBlockWeights block;
        block.conv1 = load_conv(
            *weights->store,
            source,
            "encoder.encoder.layers." + std::to_string(idx) + ".block.1.conv",
            conv_weight_storage_type,
            channels / 2,
            channels,
            3);
        block.conv2 = load_conv(
            *weights->store,
            source,
            "encoder.encoder.layers." + std::to_string(idx) + ".block.3.conv",
            conv_weight_storage_type,
            channels,
            channels / 2,
            1);
        weights->residual_blocks.push_back(std::move(block));
    }

    for (int layer = 0; layer < 8; ++layer) {
        const std::string prefix = "encoder.encoder_transformer.layers." + std::to_string(layer);
        TransformerLayerWeights block;
        block.q = weights->store->load_tensor(source, prefix + ".self_attn.q_proj.weight", linear_weight_storage_type, {512, 512});
        block.k = weights->store->load_tensor(source, prefix + ".self_attn.k_proj.weight", linear_weight_storage_type, {512, 512});
        block.v = weights->store->load_tensor(source, prefix + ".self_attn.v_proj.weight", linear_weight_storage_type, {512, 512});
        block.o = weights->store->load_tensor(source, prefix + ".self_attn.o_proj.weight", linear_weight_storage_type, {512, 512});
        block.fc1 = weights->store->load_tensor(source, prefix + ".mlp.fc1.weight", linear_weight_storage_type, {2048, 512});
        block.fc2 = weights->store->load_tensor(source, prefix + ".mlp.fc2.weight", linear_weight_storage_type, {512, 2048});
        block.norm1_weight = source.require_f32_tensor(prefix + ".input_layernorm.weight", {512});
        block.norm1_bias = source.require_f32_tensor(prefix + ".input_layernorm.bias", {512});
        block.norm2_weight = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {512});
        block.norm2_bias = source.require_f32_tensor(prefix + ".post_attention_layernorm.bias", {512});
        block.scale1 = source.require_f32_tensor(prefix + ".self_attn_layer_scale.scale", {512});
        block.scale2 = source.require_f32_tensor(prefix + ".mlp_layer_scale.scale", {512});
        weights->transformer_layers.push_back(std::move(block));
    }

    weights->downsample = load_conv(
        *weights->store,
        source,
        "encoder.downsample.conv",
        conv_weight_storage_type,
        512,
        512,
        4,
        2,
        1,
        false,
        modules::StreamingPadMode::Replicate);
    weights->semantic_projection = load_conv(
        *weights->store,
        source,
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
        conv_weight_storage_type,
        256,
        512,
        1,
        1,
        1,
        false);
    weights->acoustic_projection = load_conv(
        *weights->store,
        source,
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
        conv_weight_storage_type,
        256,
        512,
        1,
        1,
        1,
        false);

    CodebookWeights semantic;
    semantic.cluster_usage = source.require_f32(
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage",
        {2048});
    semantic.embedding_sum = source.require_f32(
        "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
        {2048, 256});
    weights->semantic_codebooks.push_back(std::move(semantic));

    for (int layer = 0; layer < 31; ++layer) {
        const std::string prefix =
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers." + std::to_string(layer) + ".codebook.";
        CodebookWeights acoustic;
        acoustic.cluster_usage = source.require_f32(prefix + "cluster_usage", {2048});
        acoustic.embedding_sum = source.require_f32(prefix + "embed_sum", {2048, 256});
        weights->acoustic_codebooks.push_back(std::move(acoustic));
    }

    weights->store->upload();
    return weights;
}

class Qwen3SpeechTokenizerEncoderGraph {
public:
    Qwen3SpeechTokenizerEncoderGraph(
        std::shared_ptr<const Qwen3SpeechTokenizerEncoderWeights> weights,
        int64_t sample_capacity,
        core::ExecutionContext & execution_context,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes,
        Qwen3TTSPerfMode perf_mode)
        : weights_(std::move(weights)),
          sample_capacity_(sample_capacity),
          frames_(ceil_div(sample_capacity, kDownsampleRate)),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)),
          perf_mode_(perf_mode) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Qwen3 speech tokenizer graph requires weights");
        }
        if (sample_capacity_ <= 0) {
            throw std::runtime_error("Qwen3 speech tokenizer graph requires positive sample capacity");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 speech tokenizer backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 speech tokenizer ggml context");
        }

        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "qwen3_tts.speech_tokenizer_encoder",
            execution_context.backend_type(),
        };
        auto x = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, sample_capacity_}));
        input_ = x.tensor;

        constants.begin_graph();
        x = speech_conv(build_ctx, x, weights_->encoder_convs[0], constants);
        for (size_t i = 0; i < weights_->residual_blocks.size(); ++i) {
            x = speech_residual_block(build_ctx, x, weights_->residual_blocks[i], constants);
            x = modules::EluModule{}.build(build_ctx, x);
            x = speech_conv(build_ctx, x, weights_->encoder_convs[i + 1], constants);
        }
        x = modules::EluModule{}.build(build_ctx, x);
        x = speech_conv(build_ctx, x, weights_->encoder_convs.back(), constants);

        auto seq = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
        seq = core::ensure_backend_addressable_layout(build_ctx, seq);
        transformer_frames_ = seq.shape.dims[1];
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, transformer_frames_);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({transformer_frames_}), GGML_TYPE_I32);
        std::optional<core::TensorValue> attention_mask = std::nullopt;
        if (perf_mode_ == Qwen3TTSPerfMode::FlashAttention) {
            attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, transformer_frames_, transformer_frames_, 1, 1);
            attention_mask = core::wrap_tensor(
                attention_mask_,
                core::TensorShape::from_dims({1, 1, transformer_frames_, transformer_frames_}),
                GGML_TYPE_F16);
        }
        for (const auto & layer : weights_->transformer_layers) {
            seq = transformer_block(build_ctx, seq, positions_value, layer, constants, perf_mode_, attention_mask);
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, seq.shape.rank}).build(build_ctx, seq);
        x = core::ensure_backend_addressable_layout(build_ctx, x);
        x = speech_conv(build_ctx, x, weights_->downsample, constants);
        auto semantic = speech_conv(build_ctx, x, weights_->semantic_projection, constants);
        auto acoustic = speech_conv(build_ctx, x, weights_->acoustic_projection, constants);
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
            throw std::runtime_error("failed to allocate Qwen3 speech tokenizer graph");
        }
        std::vector<int32_t> positions(static_cast<size_t>(transformer_frames_), 0);
        for (int64_t i = 0; i < transformer_frames_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        if (attention_mask_ != nullptr) {
            auto mask = modules::causal_prefill_mask_values(1, transformer_frames_);
            ggml_backend_tensor_set(attention_mask_, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
    }

    ~Qwen3SpeechTokenizerEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const Qwen3SpeechTokenizerEncoderWeights & weights, int64_t samples, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && sample_capacity_ == samples && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    std::pair<std::vector<float>, std::vector<float>> run(const std::vector<float> & waveform) {
        if (static_cast<int64_t>(waveform.size()) > sample_capacity_) {
            throw std::runtime_error("Qwen3 speech tokenizer waveform exceeds graph capacity");
        }
        std::vector<float> padded(static_cast<size_t>(sample_capacity_), 0.0F);
        std::copy(waveform.begin(), waveform.end(), padded.begin());
        ggml_backend_tensor_set(input_, padded.data(), 0, padded.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 speech tokenizer graph compute failed");
        }
        std::vector<float> semantic(static_cast<size_t>(kQuantizerDim * frames_), 0.0F);
        std::vector<float> acoustic(static_cast<size_t>(kQuantizerDim * frames_), 0.0F);
        ggml_backend_tensor_get(semantic_output_, semantic.data(), 0, semantic.size() * sizeof(float));
        ggml_backend_tensor_get(acoustic_output_, acoustic.data(), 0, acoustic.size() * sizeof(float));
        return {std::move(semantic), std::move(acoustic)};
    }

    int64_t frames() const noexcept {
        return frames_;
    }

private:
    std::shared_ptr<const Qwen3SpeechTokenizerEncoderWeights> weights_;
    int64_t sample_capacity_ = 0;
    int64_t frames_ = 0;
    int64_t transformer_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * semantic_output_ = nullptr;
    ggml_tensor * acoustic_output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    Qwen3TTSPerfMode perf_mode_ = Qwen3TTSPerfMode::Standard;
    ggml_gallocr_t gallocr_ = nullptr;
};

Qwen3SpeechTokenizerEncoderRuntime::Qwen3SpeechTokenizerEncoderRuntime(
    std::shared_ptr<const Qwen3TTSAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    assets::TensorStorageType linear_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type,
    Qwen3TTSPerfMode perf_mode)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes),
      perf_mode_(perf_mode) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Qwen3 speech tokenizer encoder requires assets");
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
        "qwen3_tts.speech_tokenizer_encoder.constants",
        768ull * 1024ull * 1024ull);
}

Qwen3SpeechTokenizerEncoderRuntime::~Qwen3SpeechTokenizerEncoderRuntime() = default;

Qwen3SpeechCodes Qwen3SpeechTokenizerEncoderRuntime::encode(const runtime::AudioBuffer & audio) const {
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Qwen3 speech tokenizer execution context is missing");
    }
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("Qwen3 speech tokenizer requires non-empty reference audio");
    }
    const auto waveform = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        static_cast<int>(kSampleRate));
    const int64_t valid_samples = static_cast<int64_t>(waveform.size());
    const int64_t frames = std::max<int64_t>(1, ceil_div(valid_samples, kDownsampleRate));
    const int64_t sample_capacity = valid_samples;
    const int threads = std::max(1, execution_context_->config().threads);
    if (graph_ == nullptr || !graph_->matches(*weights_, sample_capacity, execution_context_->backend(), threads)) {
        const auto build_start = Clock::now();
        graph_.reset();
        graph_ = std::make_unique<Qwen3SpeechTokenizerEncoderGraph>(
            weights_,
            sample_capacity,
            *execution_context_,
            *constants_,
            graph_arena_bytes_,
            perf_mode_);
        debug::timing_log_scalar(
            "qwen3_tts.speech_tokenizer_encoder.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    } else {
        debug::timing_log_scalar("qwen3_tts.speech_tokenizer_encoder.graph.build_ms", 0.0);
    }
    auto projected = graph_->run(waveform);
    Qwen3SpeechCodes out;
    out.frames = frames;
    out.code_groups = kValidQuantizers;
    out.codes = quantize_projected(projected.first, projected.second, frames, *weights_);
    return out;
}

}  // namespace engine::models::qwen3_tts

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include "engine/framework/codecs/s3gen_runtime.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/sampling/torch_random.h"
#include <stdexcept>
#include <random>
#include "engine/framework/core/backend_weight_store.h"
#include <vector>

namespace engine::codecs::s3gen {
namespace {

struct S3FlowEncoderWeights {
    S3FlowEncoderConfig config;
    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };
    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
    };
    struct RelativeAttentionWeights {
        LinearWeights q;
        LinearWeights k;
        LinearWeights v;
        LinearWeights out;
        LinearWeights pos;
        engine::core::TensorValue pos_bias_u_tensor;
        engine::core::TensorValue pos_bias_v_tensor;
    };
    struct FeedForwardWeights {
        LinearWeights w1;
        LinearWeights w2;
    };
    struct EncoderLayerWeights {
        LayerNormWeights norm_mha;
        RelativeAttentionWeights attn;
        LayerNormWeights norm_ff;
        FeedForwardWeights ff;
    };
    LinearWeights speaker_affine;
    LinearWeights encoder_proj;
    LinearWeights embed_linear;
    LayerNormWeights embed_norm;
    Conv1dWeights prelook_conv1;
    Conv1dWeights prelook_conv2;
    std::vector<EncoderLayerWeights> encoders;
    Conv1dWeights up_layer_conv;
    LinearWeights up_embed_linear;
    LayerNormWeights up_embed_norm;
    std::vector<EncoderLayerWeights> up_encoders;
    LayerNormWeights after_norm;
    engine::core::TensorValue input_embedding_tensor;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct S3FlowDecoderWeights {
    S3FlowDecoderConfig config;
    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };
    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        bool use_bias = false;
    };
    struct CausalBlockWeights {
        Conv1dWeights conv;
        LayerNormWeights norm;
    };
    struct ResnetBlockWeights {
        LinearWeights time_mlp;
        CausalBlockWeights block1;
        CausalBlockWeights block2;
        Conv1dWeights res_conv;
    };
    struct TransformerBlockWeights {
        int64_t heads = 0;
        int64_t head_dim = 0;
        LayerNormWeights norm1;
        LinearWeights attn_q;
        LinearWeights attn_k;
        LinearWeights attn_v;
        LinearWeights attn_out;
        LayerNormWeights norm3;
        LinearWeights ff_proj_in;
        LinearWeights ff_proj_out;
    };
    struct DownBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
        Conv1dWeights downsample;
    };
    struct MidBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
    };
    struct UpBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
        Conv1dWeights upsample;
    };

    LinearWeights time_mlp_1;
    LinearWeights time_mlp_2;
    std::vector<DownBlockWeights> down_blocks;
    std::vector<MidBlockWeights> mid_blocks;
    std::vector<UpBlockWeights> up_blocks;
    CausalBlockWeights final_block;
    Conv1dWeights final_proj;
    // Meanflow-distilled decoders (Chatterbox Turbo) mix a second "r" (end-time) sinusoidal
    // embedding into the time embedding via this diagonal-init, no-bias linear layer before
    // feeding the UNet1D estimator; see decoder.py::get_intmeanflow_time_mixer upstream. Unset
    // (weight_tensor.tensor == nullptr) for the base Chatterbox 10-step CFG decoder.
    bool meanflow = false;
    LinearWeights time_embed_mixer;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct S3FlowDecoderOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t storage_frames = 0;
};

struct S3FlowCFMOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
    int64_t storage_frames = 0;
};

class S3FlowSessionCache {
public:
    explicit S3FlowSessionCache(engine::core::BackendConfig backend = {});
    ~S3FlowSessionCache();

    S3FlowSessionCache(const S3FlowSessionCache &) = delete;
    S3FlowSessionCache & operator=(const S3FlowSessionCache &) = delete;

    void release_encoder_graphs();
    void release_decoder_graphs();

private:
    struct State;
    std::unique_ptr<State> state_;

    friend S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
        S3FlowSessionCache & cache,
        const S3FlowEncoderWeights & weights,
        const std::vector<float> & input,
        int64_t frames,
        int64_t capacity_frames,
        int64_t hidden_size,
        engine::core::BackendConfig backend);
    friend S3FlowCFMOutputs compute_s3_flow_cfm_euler(
        S3FlowSessionCache & cache,
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & noise,
        const std::vector<float> & mask,
        const std::vector<float> & mu,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t batch,
        int64_t frames,
        int64_t capacity_frames,
        int64_t num_steps,
        float cfg_rate,
        bool cosine_schedule,
        engine::core::BackendConfig backend,
        S3FlowCFMTimingBreakdown * timing);
    friend S3FlowCFMOutputs compute_s3_flow_cfm_meanflow(
        S3FlowSessionCache & cache,
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & noise,
        const std::vector<float> & mask,
        const std::vector<float> & mu,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t batch,
        int64_t frames,
        int64_t capacity_frames,
        int64_t num_steps,
        engine::core::BackendConfig backend);
};

std::shared_ptr<const S3FlowEncoderWeights> load_s3_flow_encoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    const S3GenConfig & config,
    const std::string & tensor_prefix,
    engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);
S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowEncoderWeights & weights,
    const std::vector<float> & input,
    int64_t frames,
    int64_t capacity_frames,
    int64_t hidden_size,
    engine::core::BackendConfig backend = {});
std::shared_ptr<const S3FlowDecoderWeights> load_s3_flow_decoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    const S3GenConfig & config,
    const std::string & tensor_prefix,
    engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);
S3FlowCFMOutputs compute_s3_flow_cfm_euler(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    engine::core::BackendConfig backend = {},
    S3FlowCFMTimingBreakdown * timing = nullptr);

// Non-CFG, non-batch-doubled Euler solve for meanflow-distilled decoders (Chatterbox Turbo);
// see S3FlowDecoderWeights::meanflow. num_steps defaults to 2 in upstream Python (tts_turbo.py).
S3FlowCFMOutputs compute_s3_flow_cfm_meanflow(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps = 2,
    engine::core::BackendConfig backend = {});

class S3GenSessionCache {
public:
    explicit S3GenSessionCache(engine::core::BackendConfig backend = {}, bool retain_encoder_graphs = false);
    ~S3GenSessionCache();

    S3GenSessionCache(const S3GenSessionCache &) = delete;
    S3GenSessionCache & operator=(const S3GenSessionCache &) = delete;

    void release_runtime_graphs();

private:
    friend class engine::codecs::s3gen::S3GenRuntime;
    struct State;
    std::unique_ptr<State> state_;

    friend S3Token2MelOutputs compute_s3_token2mel_inference(
        S3GenSessionCache & cache,
        const S3FlowEncoderWeights & encoder_weights,
        const S3FlowDecoderWeights & decoder_weights,
        const S3GenConditioning & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        int64_t speech_token_count,
        int64_t num_steps,
        float cfg_rate,
        bool cosine_schedule,
        const std::vector<float> & full_noise,
        uint64_t flow_seed,
        engine::core::BackendConfig backend,
        S3GenTimingBreakdown * timing);
};

namespace {

S3FlowEncoderWeights::LayerNormWeights load_flow_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    S3FlowEncoderWeights::LayerNormWeights weights;
    weights.weight_tensor = store.load_f32_tensor(source, prefix + ".weight", {channels});
    weights.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {channels});
    return weights;
}

S3FlowEncoderWeights::LinearWeights load_flow_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowEncoderWeights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return linear;
}

S3FlowEncoderWeights::Conv1dWeights load_flow_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowEncoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return conv;
}

S3FlowDecoderWeights::LayerNormWeights load_decoder_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    S3FlowDecoderWeights::LayerNormWeights weights;
    weights.weight_tensor = store.load_f32_tensor(source, prefix + ".weight", {channels});
    weights.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {channels});
    return weights;
}

S3FlowDecoderWeights::LinearWeights load_decoder_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowDecoderWeights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return linear;
}

S3FlowDecoderWeights::Conv1dWeights load_decoder_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowDecoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.use_bias = use_bias;
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return conv;
}

std::vector<float> decoder_sinusoidal_pos_emb(
    const std::vector<float> & timesteps,
    int64_t batch,
    int64_t dim,
    float scale = 1000.0f) {
    if ((dim % 2) != 0) {
        throw std::runtime_error("decoder sinusoidal embedding requires even dim");
    }
    const int64_t half_dim = dim / 2;
    std::vector<float> out(static_cast<size_t>(batch * dim), 0.0f);
    const double emb_scale = std::log(10000.0) / static_cast<double>(half_dim - 1);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < half_dim; ++i) {
            const double freq = std::exp(-static_cast<double>(i) * emb_scale);
            const double angle = static_cast<double>(scale) * static_cast<double>(timesteps[static_cast<size_t>(b)]) * freq;
            out[static_cast<size_t>(b * dim + i)] = static_cast<float>(std::sin(angle));
            out[static_cast<size_t>(b * dim + half_dim + i)] = static_cast<float>(std::cos(angle));
        }
    }
    return out;
}

std::vector<float> flow_relative_positional_encoding(int64_t frames, int64_t channels) {
    const int64_t rel_frames = (frames * 2) - 1;
    std::vector<float> positive(static_cast<size_t>(frames * channels), 0.0f);
    std::vector<float> negative(static_cast<size_t>(frames * channels), 0.0f);
    for (int64_t pos = 0; pos < frames; ++pos) {
        for (int64_t i = 0; i < channels; i += 2) {
            const double div = std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(channels));
            positive[static_cast<size_t>(pos * channels + i)] = static_cast<float>(std::sin(static_cast<double>(pos) * div));
            negative[static_cast<size_t>(pos * channels + i)] = static_cast<float>(std::sin(-static_cast<double>(pos) * div));
            if (i + 1 < channels) {
                positive[static_cast<size_t>(pos * channels + i + 1)] = static_cast<float>(std::cos(static_cast<double>(pos) * div));
                negative[static_cast<size_t>(pos * channels + i + 1)] = static_cast<float>(std::cos(-static_cast<double>(pos) * div));
            }
        }
    }
    std::vector<float> out(static_cast<size_t>(rel_frames * channels), 0.0f);
    for (int64_t pos = 0; pos < frames; ++pos) {
        std::copy_n(
            positive.data() + static_cast<std::ptrdiff_t>((frames - 1 - pos) * channels),
            channels,
            out.data() + static_cast<std::ptrdiff_t>(pos * channels));
    }
    for (int64_t pos = 1; pos < frames; ++pos) {
        std::copy_n(
            negative.data() + static_cast<std::ptrdiff_t>(pos * channels),
            channels,
            out.data() + static_cast<std::ptrdiff_t>((frames - 1 + pos) * channels));
    }
    return out;
}

bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

using Clock = std::chrono::steady_clock;

void copy_backend_tensor(
    const engine::core::TensorValue & src,
    const engine::core::TensorValue & dst) {
    ggml_backend_tensor_copy(src.tensor, dst.tensor);
}

void release_graph_resources(
    const engine::core::ExecutionContext * execution_context,
    ggml_cgraph *& graph,
    ggml_context *& ggml,
    ggml_gallocr_t & gallocr) {
    if (execution_context != nullptr && graph != nullptr) {
        engine::core::release_backend_graph_resources(execution_context->backend(), graph);
        graph = nullptr;
    }
    if (gallocr != nullptr) {
        ggml_gallocr_free(gallocr);
        gallocr = nullptr;
    }
    if (ggml != nullptr) {
        ggml_free(ggml);
        ggml = nullptr;
    }
}

void allocate_graph_resources(
    const engine::core::ExecutionContext & execution_context,
    ggml_cgraph * graph,
    ggml_gallocr_t & gallocr,
    const char * label) {
    gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context.backend()));
    if (gallocr == nullptr ||
        !ggml_gallocr_reserve(gallocr, graph) ||
        !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        throw std::runtime_error(std::string("failed to allocate backend tensors for ") + label);
    }
}

int64_t valid_frames_from_mask(const std::vector<float> & mask, int64_t batch, int64_t frames) {
    if (batch <= 0 || frames <= 0) {
        throw std::runtime_error("S3 valid_frames_from_mask requires positive batch and frames");
    }
    if (static_cast<int64_t>(mask.size()) != batch * frames) {
        throw std::runtime_error("S3 mask size mismatch");
    }
    int64_t valid = 0;
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float value = mask[static_cast<size_t>(frame)];
        if (value >= 0.5f) {
            ++valid;
            continue;
        }
        break;
    }
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            const float value = mask[static_cast<size_t>(batch_index * frames + frame)];
            const float expected = frame < valid ? 1.0f : 0.0f;
            if (std::fabs(value - expected) > 1.0e-6f) {
                throw std::runtime_error("S3 mask must be a shared prefix-ones suffix-zeros layout");
            }
        }
    }
    return valid;
}

std::vector<float> make_attention_key_mask(
    int64_t batch,
    int64_t query_frames,
    int64_t key_frames,
    int64_t valid_key_frames,
    int64_t heads = 1) {
    std::vector<float> values(static_cast<size_t>(batch * heads * query_frames * key_frames), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t query_frame = 0; query_frame < query_frames; ++query_frame) {
                for (int64_t key_frame = valid_key_frames; key_frame < key_frames; ++key_frame) {
                    values[static_cast<size_t>(((batch_index * heads + head) * query_frames + query_frame) * key_frames + key_frame)] = -10000.0f;
                }
            }
        }
    }
    return values;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::ensure_backend_addressable_layout(ctx, input);
}

engine::core::TensorValue make_f32_graph_constant(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorShape & shape,
    const std::vector<float> & values,
    engine::core::DeferredTensorWriter & writer) {
    auto tensor = writer.make_f32_tensor(ctx, shape, values);
    ggml_set_input(tensor.tensor);
    ggml_set_output(tensor.tensor);
    return tensor;
}

engine::core::TensorValue add_tensor_values(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto lhs_contiguous = contiguous(ctx, lhs);
    auto rhs_contiguous = contiguous(ctx, rhs);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue mul_tensor_values(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto lhs_contiguous = contiguous(ctx, lhs);
    auto rhs_contiguous = contiguous(ctx, rhs);
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue permute_tensor(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    std::array<int, engine::core::kMaxTensorRank> axes) {
    std::array<int, engine::core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    engine::core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;
    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        output_shape.dims[out_logical_axis] = input.shape.dims[in_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = engine::core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
    }
    return engine::core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

engine::core::TensorValue transpose_last_two(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    std::array<int, engine::core::kMaxTensorRank> axes = {0, 1, 2, 3};
    const size_t last = input.shape.rank - 1;
    const size_t second_last = input.shape.rank - 2;
    axes[second_last] = static_cast<int>(last);
    axes[last] = static_cast<int>(second_last);
    return permute_tensor(ctx, input, axes);
}

engine::core::TensorValue batched_matmul(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto rhs_transposed = transpose_last_two(ctx, rhs);
    rhs_transposed = contiguous(ctx, rhs_transposed);
    auto output_shape = lhs.shape;
    output_shape.dims[lhs.shape.rank - 1] = rhs.shape.dims[rhs.shape.rank - 1];
    return engine::core::wrap_tensor(ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor), output_shape, GGML_TYPE_F32);
}

template <typename LinearWeights>
engine::core::TensorValue linear_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const LinearWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t rows = input.shape.prefix_elements();
    auto flat = engine::core::reshape_tensor(
        ctx,
        contiguous(ctx, input),
        engine::core::TensorShape::from_dims({rows, input.shape.last_dim()}));
    auto projected = engine::core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, weights.weight_tensor.tensor, flat.tensor),
        engine::core::TensorShape::from_dims({rows, weights.out_features}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        const auto bias_view = engine::core::reshape_tensor(
            ctx,
            weights.bias_tensor,
            engine::core::TensorShape::from_dims({1, weights.out_features}));
        projected = engine::core::wrap_tensor(
            ggml_add(ctx.ggml, projected.tensor, bias_view.tensor),
            projected.shape,
            GGML_TYPE_F32);
    }
    return engine::core::reshape_tensor(ctx, projected, input.shape.with_last_dim(weights.out_features));
}

template <typename LayerNormWeights>
engine::core::TensorValue layer_norm_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const LayerNormWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    float eps = 1.0e-5f) {
    (void) writer;
    auto normalized = engine::core::wrap_tensor(ggml_norm(ctx.ggml, contiguous(ctx, input).tensor, eps), input.shape, GGML_TYPE_F32);
    auto scaled = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, weights.weight_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, scaled.tensor, weights.bias_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue transpose_bct_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return permute_tensor(ctx, input, {0, 2, 1});
}

engine::core::TensorValue zero_prefix_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t prefix_frames);

engine::core::TensorValue bct_add_bias(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & bias_tensor) {
    const int64_t channels = input.shape.dims[1];
    const auto bias = engine::core::reshape_tensor(
        ctx,
        bias_tensor,
        engine::core::TensorShape::from_dims({1, channels, 1}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, bias.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue add_attention_bias(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & bias_tensor,
    int64_t heads,
    int64_t head_dim) {
    const auto bias_view = engine::core::reshape_tensor(
        ctx,
        bias_tensor,
        engine::core::TensorShape::from_dims({1, heads, 1, head_dim}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, bias_view.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue zero_like_last_column(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        1,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->ne[3],
        input.tensor->nb[1],
        input.tensor->nb[2],
        input.tensor->nb[3],
        0);
    auto first = engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2], 1}),
        GGML_TYPE_F32);
    auto first_contiguous = contiguous(ctx, first);
    return engine::core::wrap_tensor(ggml_scale(ctx.ggml, first_contiguous.tensor, 0.0f), first.shape, GGML_TYPE_F32);
}

engine::core::TensorValue slice_last_dim_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t count);

engine::core::TensorValue relative_shift(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    const int64_t query = input.shape.dims[2];
    const int64_t pos = input.shape.dims[3];
    auto zero_col = zero_like_last_column(ctx, input);
    auto padded = engine::core::wrap_tensor(
        // ESPnet relative shift prepends the zero column before flattening.
        ggml_concat(ctx.ggml, zero_col.tensor, input.tensor, engine::core::logical_axis_to_ggml_axis(4, 3)),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query, pos + 1}),
        GGML_TYPE_F32);
    auto padded_contiguous = contiguous(ctx, padded);
    auto flattened = engine::core::reshape_tensor(
        ctx,
        padded_contiguous,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query * (pos + 1)}));
    auto * shifted_view = ggml_view_3d(
        ctx.ggml,
        flattened.tensor,
        flattened.tensor->ne[0] - query,
        flattened.tensor->ne[1],
        flattened.tensor->ne[2],
        flattened.tensor->nb[1],
        flattened.tensor->nb[2],
        static_cast<size_t>(query) * sizeof(float));
    auto shifted = engine::core::wrap_tensor(
        shifted_view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], (query * (pos + 1)) - query}),
        GGML_TYPE_F32);
    shifted = contiguous(ctx, shifted);
    auto reshaped = engine::core::reshape_tensor(
        ctx,
        shifted,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query, pos}));
    return slice_last_dim_4d(ctx, reshaped, 0, query);
}

engine::core::TensorValue slice_last_dim_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t count) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        count,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->ne[3],
        input.tensor->nb[1],
        input.tensor->nb[2],
        input.tensor->nb[3],
        static_cast<size_t>(start) * sizeof(float));
    return engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2], count}),
        GGML_TYPE_F32);
}

engine::core::TensorValue mish_activation(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    engine::core::DeferredTensorWriter & writer) {
    static const std::vector<float> kOne = {1.0f};
    const auto one = make_f32_graph_constant(ctx, engine::core::TensorShape::from_dims({1}), kOne, writer);
    const auto input_contiguous = contiguous(ctx, input);
    auto ex = engine::core::wrap_tensor(ggml_exp(ctx.ggml, input_contiguous.tensor), input.shape, GGML_TYPE_F32);
    auto softplus = engine::core::wrap_tensor(ggml_log(ctx.ggml, ggml_add(ctx.ggml, ex.tensor, one.tensor)), input.shape, GGML_TYPE_F32);
    auto squashed = engine::core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), input.shape, GGML_TYPE_F32);
    return mul_tensor_values(ctx, input_contiguous, squashed);
}

engine::core::TensorValue causal_conv1d_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowDecoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[2];
    if (weights.kernel == 1 && weights.stride == 1) {
        auto input_btc = transpose_bct_btc(ctx, input);
        const int64_t rows = batch * frames;
        auto flat = engine::core::reshape_tensor(
            ctx,
            contiguous(ctx, input_btc),
            engine::core::TensorShape::from_dims({rows, weights.in_channels}));
        const auto weight = engine::core::reshape_tensor(
            ctx,
            weights.weight_tensor,
            engine::core::TensorShape::from_dims({weights.out_channels, weights.in_channels}));
        auto projected = engine::core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, weight.tensor, flat.tensor),
            engine::core::TensorShape::from_dims({rows, weights.out_channels}),
            GGML_TYPE_F32);
        if (weights.use_bias) {
            const auto bias_view = engine::core::reshape_tensor(
                ctx,
                weights.bias_tensor,
                engine::core::TensorShape::from_dims({1, weights.out_channels}));
            projected = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, projected.tensor, bias_view.tensor),
                projected.shape,
                GGML_TYPE_F32);
        }
        auto output_btc = engine::core::reshape_tensor(
            ctx,
            projected,
            engine::core::TensorShape::from_dims({batch, frames, weights.out_channels}));
        return transpose_bct_btc(ctx, output_btc);
    }
    const int64_t left_pad = weights.kernel - 1;
    const auto weight = weights.weight_tensor;
    engine::core::TensorValue bias;
    engine::core::TensorValue bias_view;
    if (weights.use_bias) {
        bias = weights.bias_tensor;
        bias_view = engine::core::reshape_tensor(
            ctx,
            bias,
            engine::core::TensorShape::from_dims({1, weights.out_channels, 1}));
    }
    if (batch > 1) {
        engine::core::TensorValue output;
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            auto input_single = engine::modules::SliceModule({0, batch_index, 1}).build(ctx, input);
            auto padded_single = engine::modules::ConcatModule({2}).build(
                ctx,
                zero_prefix_bct(ctx, input_single, left_pad),
                input_single);
            auto output_single = engine::core::wrap_tensor(
                ggml_conv_1d(ctx.ggml, contiguous(ctx, weight).tensor, contiguous(ctx, padded_single).tensor, 1, 0, 1),
                engine::core::TensorShape::from_dims({1, weights.out_channels, frames}),
                GGML_TYPE_F32);
            if (weights.use_bias) {
                output_single = engine::core::wrap_tensor(
                    ggml_add(ctx.ggml, output_single.tensor, bias_view.tensor),
                    output_single.shape,
                    GGML_TYPE_F32);
            }
            output = output.valid()
                ? engine::modules::ConcatModule({0}).build(ctx, output, output_single)
                : output_single;
        }
        return contiguous(ctx, output);
    }
    auto padded = engine::modules::ConcatModule({2}).build(
        ctx,
        zero_prefix_bct(ctx, input, left_pad),
        input);
    auto output = engine::core::wrap_tensor(
        ggml_conv_1d(ctx.ggml, contiguous(ctx, weight).tensor, contiguous(ctx, padded).tensor, 1, 0, 1),
        engine::core::TensorShape::from_dims({batch, weights.out_channels, frames}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        output = engine::core::wrap_tensor(
            ggml_add(ctx.ggml, output.tensor, bias_view.tensor),
            output.shape,
            GGML_TYPE_F32);
    }
    return output;
}

engine::core::TensorValue causal_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const S3FlowDecoderWeights::CausalBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    auto x = causal_conv1d_bct(ctx, input_bct, weights.conv, writer);
    x = transpose_bct_btc(ctx, x);
    x = layer_norm_lastdim(ctx, x, weights.norm, writer);
    x = transpose_bct_btc(ctx, x);
    return mish_activation(ctx, x, writer);
}

engine::core::TensorValue repeat_spks_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & spks_bt,
    int64_t frames) {
    auto spks_bct = engine::core::reshape_tensor(
        ctx,
        spks_bt,
        engine::core::TensorShape::from_dims({spks_bt.shape.dims[0], spks_bt.shape.dims[1], 1}));
    auto target = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({spks_bt.shape.dims[0], spks_bt.shape.dims[1], frames}));
    return engine::core::wrap_tensor(ggml_repeat(ctx.ggml, spks_bct.tensor, target.tensor), target.shape, GGML_TYPE_F32);
}

engine::core::TensorValue cat_channels_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & a,
    const engine::core::TensorValue & b) {
    return engine::core::wrap_tensor(
        ggml_concat(ctx.ggml, a.tensor, b.tensor, engine::core::logical_axis_to_ggml_axis(3, 1)),
        engine::core::TensorShape::from_dims({a.shape.dims[0], a.shape.dims[1] + b.shape.dims[1], a.shape.dims[2]}),
        GGML_TYPE_F32);
}

engine::core::TensorValue add_time_bias_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const engine::core::TensorValue & time_bt) {
    auto time_bct = engine::core::reshape_tensor(
        ctx,
        time_bt,
        engine::core::TensorShape::from_dims({time_bt.shape.dims[0], time_bt.shape.dims[1], 1}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input_bct.tensor, time_bct.tensor),
        input_bct.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue zero_prefix_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t prefix_frames) {
    auto first = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
    auto prefix = engine::modules::RepeatModule({
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})})
                      .build(ctx, first);
    auto prefix_contiguous = contiguous(ctx, prefix);
    return engine::core::wrap_tensor(ggml_scale(ctx.ggml, prefix_contiguous.tensor, 0.0f), prefix.shape, GGML_TYPE_F32);
}

engine::core::TensorValue self_attention_no_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const S3FlowDecoderWeights::TransformerBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t heads = weights.heads;
    const int64_t head_dim = weights.head_dim;
    const int64_t inner_dim = heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q = linear_lastdim(ctx, input_btc, weights.attn_q, writer);
    auto k = linear_lastdim(ctx, input_btc, weights.attn_k, writer);
    auto v = linear_lastdim(ctx, input_btc, weights.attn_v, writer);
    const int64_t batch = input_btc.shape.dims[0];
    const int64_t frames = input_btc.shape.dims[1];
    q = engine::core::reshape_tensor(ctx, q, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, k, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, v, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    q = permute_tensor(ctx, q, {0, 2, 1, 3});
    k = permute_tensor(ctx, k, {0, 2, 1, 3});
    v = permute_tensor(ctx, v, {0, 2, 1, 3});
    engine::core::TensorValue context;
    if (attention_mask.has_value()) {
        auto q_contiguous = contiguous(ctx, q);
        auto k_contiguous = contiguous(ctx, k);
        auto v_contiguous = contiguous(ctx, v);
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_contiguous.tensor,
            k_contiguous.tensor,
            v_contiguous.tensor,
            attention_mask->tensor,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = engine::core::wrap_tensor(
            flash,
            engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto k_t = permute_tensor(ctx, k, {0, 1, 3, 2});
        auto scores = batched_matmul(ctx, q, k_t);
        scores = engine::core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
        context = batched_matmul(ctx, attn, v);
        context = permute_tensor(ctx, context, {0, 2, 1, 3});
    }
    context = contiguous(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({batch, frames, inner_dim}));
    return linear_lastdim(ctx, context, weights.attn_out, writer);
}

engine::core::TensorValue transformer_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const S3FlowDecoderWeights::TransformerBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) {
    auto x = layer_norm_lastdim(ctx, input_btc, weights.norm1, writer);
    auto attn = self_attention_no_mask(ctx, x, weights, writer, attention_mask);
    auto y = add_tensor_values(ctx, input_btc, attn);
    x = layer_norm_lastdim(ctx, y, weights.norm3, writer);
    auto ff = linear_lastdim(ctx, x, weights.ff_proj_in, writer);
    ff = engine::core::wrap_tensor(ggml_gelu_erf(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
    ff = linear_lastdim(ctx, ff, weights.ff_proj_out, writer);
    return add_tensor_values(ctx, y, ff);
}

engine::core::TensorValue resnet_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const engine::core::TensorValue & time_emb,
    const S3FlowDecoderWeights::ResnetBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    auto h = causal_block_forward(ctx, input_bct, weights.block1, writer);
    auto time_proj = mish_activation(ctx, time_emb, writer);
    time_proj = linear_lastdim(ctx, time_proj, weights.time_mlp, writer);
    h = add_time_bias_bct(ctx, h, time_proj);
    h = causal_block_forward(ctx, h, weights.block2, writer);
    auto residual = causal_conv1d_bct(ctx, input_bct, weights.res_conv, writer);
    return add_tensor_values(ctx, h, residual);
}

engine::core::TensorValue flow_conv1d_bct_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowEncoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t out_frames =
        (input.shape.dims[2] + 2 * weights.padding - weights.kernel) / weights.stride + 1;
    auto output = engine::core::wrap_tensor(
        ggml_conv_1d(
            ctx.ggml,
            contiguous(ctx, weights.weight_tensor).tensor,
            contiguous(ctx, input).tensor,
            static_cast<int>(weights.stride),
            static_cast<int>(weights.padding),
            1),
        engine::core::TensorShape::from_dims({input.shape.dims[0], weights.out_channels, out_frames}),
        GGML_TYPE_F32);
    output = bct_add_bias(ctx, output, weights.bias_tensor);
    return output;
}

engine::core::TensorValue flow_prelookahead_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::Conv1dWeights & conv1,
    const S3FlowEncoderWeights::Conv1dWeights & conv2,
    engine::core::DeferredTensorWriter & writer) {
    auto input_bct = transpose_bct_btc(ctx, input_btc);
    auto padded1 = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous(ctx, input_bct).tensor, 0, conv1.kernel - 1, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames + conv1.kernel - 1}),
        GGML_TYPE_F32);
    auto h = flow_conv1d_bct_backend(ctx, padded1, conv1, writer);
    h = engine::core::wrap_tensor(ggml_leaky_relu(ctx.ggml, h.tensor, 0.01f, false), h.shape, GGML_TYPE_F32);
    auto padded2 = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous(ctx, h).tensor, conv2.kernel - 1, 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames + conv2.kernel - 1}),
        GGML_TYPE_F32);
    auto y_bct = flow_conv1d_bct_backend(ctx, padded2, conv2, writer);
    auto y_btc = transpose_bct_btc(ctx, y_bct);
    return add_tensor_values(ctx, input_btc, y_btc);
}

engine::core::TensorValue flow_relative_attention_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const engine::core::TensorValue & pos_emb_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::RelativeAttentionWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) {
    const auto heads = weights.pos_bias_u_tensor.shape.dims[0];
    const auto head_dim = weights.pos_bias_u_tensor.shape.dims[1];
    auto q = linear_lastdim(ctx, input_btc, weights.q, writer);
    auto k = linear_lastdim(ctx, input_btc, weights.k, writer);
    auto v = linear_lastdim(ctx, input_btc, weights.v, writer);
    const int64_t pos_frames = (frames * 2) - 1;
    auto p = linear_lastdim(ctx, pos_emb_btc, weights.pos, writer);

    q = engine::core::reshape_tensor(ctx, q, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, k, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, v, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    p = engine::core::reshape_tensor(ctx, p, engine::core::TensorShape::from_dims({1, pos_frames, heads, head_dim}));

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});
    auto p_heads = permute_tensor(ctx, p, {0, 2, 1, 3});

    auto q_u = add_attention_bias(ctx, q_heads, weights.pos_bias_u_tensor, heads, head_dim);
    auto q_v = add_attention_bias(ctx, q_heads, weights.pos_bias_v_tensor, heads, head_dim);

    auto matrix_ac = batched_matmul(ctx, q_u, permute_tensor(ctx, k_heads, {0, 1, 3, 2}));
    auto matrix_bd = batched_matmul(ctx, q_v, permute_tensor(ctx, p_heads, {0, 1, 3, 2}));
    matrix_bd = relative_shift(ctx, matrix_bd);
    matrix_bd = slice_last_dim_4d(ctx, matrix_bd, 0, frames);

    auto scores = add_tensor_values(ctx, matrix_ac, matrix_bd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    scores = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, scale),
        scores.shape,
        GGML_TYPE_F32);
    if (attention_mask.has_value()) {
        scores = add_tensor_values(ctx, scores, *attention_mask);
    }
    auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = batched_matmul(ctx, attn, v_heads);
    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = contiguous(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({1, frames, hidden}));
    return linear_lastdim(ctx, context, weights.out, writer);
}

engine::core::TensorValue flow_upsample_repeat_conv_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    int factor) {
    std::vector<float> repeat_weight(static_cast<size_t>(hidden * hidden * factor), 0.0f);
    for (int64_t c = 0; c < hidden; ++c) {
        const size_t base = static_cast<size_t>((c * hidden + c) * factor);
        std::fill_n(repeat_weight.begin() + base, factor, 1.0f);
    }
    auto input_bct = transpose_bct_btc(ctx, input_btc);
    const auto repeat_kernel = make_f32_graph_constant(
        ctx,
        engine::core::TensorShape::from_dims({hidden, hidden, factor}),
        repeat_weight,
        writer);
    auto repeated = engine::modules::ConvTranspose1dModule({
        hidden,
        hidden,
        factor,
        factor,
        0,
        1,
        false,
    }).build(ctx, input_bct, engine::modules::ConvTranspose1dWeights{repeat_kernel, std::nullopt});
    auto padded = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, repeated.tensor, weights.kernel - 1, 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames * factor + weights.kernel - 1}),
        GGML_TYPE_F32);
    auto y_bct = flow_conv1d_bct_backend(ctx, padded, weights, writer);
    return transpose_bct_btc(ctx, y_bct);
}

class FlowEncoderBackendRunner {
public:
    FlowEncoderBackendRunner(
        const S3FlowEncoderWeights & weights,
        int64_t frames,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config)
        : input_frames_(frames),
          hidden_size_(hidden_size),
          output_frames_(frames * weights.config.upsample_factor),
          embed_prelook_runner_(weights, frames, hidden_size, backend_config),
          upsample_runner_(weights, frames, hidden_size, backend_config),
          after_norm_runner_(weights.after_norm, *weights.execution_context, output_frames_, hidden_size, backend_config) {
        encoder_runners_.reserve(weights.encoders.size());
        for (const auto & layer : weights.encoders) {
            encoder_runners_.push_back(
                std::make_unique<FlowEncoderLayerBackendRunner>(
                    layer, *weights.execution_context, input_frames_, hidden_size_, backend_config));
        }
        up_encoder_runners_.reserve(weights.up_encoders.size());
        for (const auto & layer : weights.up_encoders) {
            up_encoder_runners_.push_back(
                std::make_unique<FlowEncoderLayerBackendRunner>(
                    layer, *weights.execution_context, output_frames_, hidden_size_, backend_config));
        }
    }

    S3FlowEncoderOutputs run(const std::vector<float> & input, int64_t valid_frames) {
        const auto attention_mask_values = make_attention_key_mask(1, input_frames_, input_frames_, valid_frames);
        embed_prelook_runner_.write_input(input);
        embed_prelook_runner_.compute();

        auto * previous_output = &embed_prelook_runner_.output_tensor();
        for (auto & runner : encoder_runners_) {
            runner->write_attention_mask(attention_mask_values);
            copy_backend_tensor(*previous_output, runner->input_tensor());
            runner->compute();
            previous_output = &runner->output_tensor();
        }

        copy_backend_tensor(*previous_output, upsample_runner_.input_tensor());
        upsample_runner_.compute();

        const int64_t valid_output_frames = valid_frames * (output_frames_ / input_frames_);
        const auto up_attention_mask_values = make_attention_key_mask(1, output_frames_, output_frames_, valid_output_frames);
        previous_output = &upsample_runner_.output_tensor();
        for (auto & runner : up_encoder_runners_) {
            runner->write_attention_mask(up_attention_mask_values);
            copy_backend_tensor(*previous_output, runner->input_tensor());
            runner->compute();
            previous_output = &runner->output_tensor();
        }

        copy_backend_tensor(*previous_output, after_norm_runner_.input_tensor());
        after_norm_runner_.compute();

        S3FlowEncoderOutputs outputs;
        outputs.hidden = after_norm_runner_.read_output();
        outputs.frames = valid_output_frames;
        outputs.storage_frames = output_frames_;
        outputs.hidden_size = hidden_size_;
        return outputs;
    }

private:
    class FlowEncoderEmbedPrelookBackendRunner {
    public:
        FlowEncoderEmbedPrelookBackendRunner(
            const S3FlowEncoderWeights & weights,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(weights.execution_context) {
            ggml_init_params params = {};
            params.mem_size = 192ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder embed/prelook");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_embed_prelook";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));

            auto x = linear_lastdim(ctx, input_tensor_, weights.embed_linear, writer_);
            x = layer_norm_lastdim(ctx, x, weights.embed_norm, writer_);
            x = engine::core::wrap_tensor(
                ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(hidden_size))),
                x.shape,
                GGML_TYPE_F32);
            output_ = flow_prelookahead_backend(
                ctx,
                x,
                frames,
                hidden_size,
                weights.prelook_conv1,
                weights.prelook_conv2,
                writer_);

            graph_ = ggml_new_graph_custom(ggml_, 16384, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            allocate_graph_resources(*execution_context_, graph_, gallocr_, "S3 flow encoder embed/prelook");
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderEmbedPrelookBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, gallocr_);
        }

        void write_input(const std::vector<float> & input) {
            engine::core::write_tensor_f32(input_tensor_, input);
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder embed/prelook");
            }
        }

        const engine::core::TensorValue & output_tensor() const {
            return output_;
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    class FlowEncoderLayerBackendRunner {
    public:
        FlowEncoderLayerBackendRunner(
            const S3FlowEncoderWeights::EncoderLayerWeights & weights,
            const engine::core::ExecutionContext & execution_context,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig & backend_config)
            : attention_runner_(weights, execution_context, frames, hidden_size, backend_config),
              ff_runner_(weights, execution_context, frames, hidden_size, backend_config) {}

        engine::core::TensorValue & input_tensor() {
            return attention_runner_.input_tensor();
        }

        void write_attention_mask(const std::vector<float> & values) {
            attention_runner_.write_attention_mask(values);
        }

        const engine::core::TensorValue & output_tensor() const {
            return ff_runner_.output_tensor();
        }

        void compute() {
            attention_runner_.compute();
            copy_backend_tensor(attention_runner_.output_tensor(), ff_runner_.input_tensor());
            ff_runner_.compute();
        }

    private:
        class FlowEncoderAttentionBackendRunner {
        public:
            FlowEncoderAttentionBackendRunner(
                const S3FlowEncoderWeights::EncoderLayerWeights & weights,
                const engine::core::ExecutionContext & execution_context,
                int64_t frames,
                int64_t hidden_size,
                const engine::core::BackendConfig &)
                : execution_context_(&execution_context) {
                ggml_init_params params = {};
                params.mem_size = 192ull * 1024ull * 1024ull;
                params.mem_buffer = nullptr;
                params.no_alloc = true;
                ggml_ = ggml_init(params);
                if (ggml_ == nullptr) {
                    throw std::runtime_error("failed to initialize ggml context for S3 flow encoder attention");
                }

                engine::core::ModuleBuildContext ctx = {};
                ctx.ggml = ggml_;
                ctx.module_instance_name = "s3_flow_encoder_attention";
                input_tensor_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, frames, hidden_size}));
                attention_mask_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, 1, frames, frames}));
                const auto pos_tensor = make_f32_graph_constant(
                    ctx,
                    engine::core::TensorShape::from_dims({1, (frames * 2) - 1, hidden_size}),
                    flow_relative_positional_encoding(frames, hidden_size),
                    writer_);
                auto x = layer_norm_lastdim(ctx, input_tensor_, weights.norm_mha, writer_, 1.0e-12f);
                auto attn = flow_relative_attention_backend(
                    ctx,
                    x,
                    pos_tensor,
                    frames,
                    hidden_size,
                    weights.attn,
                    writer_,
                    attention_mask_);
                output_ = add_tensor_values(ctx, input_tensor_, attn);

                graph_ = ggml_new_graph_custom(ggml_, 24576, false);
                ggml_build_forward_expand(graph_, output_.tensor);
                allocate_graph_resources(*execution_context_, graph_, gallocr_, "S3 flow encoder attention");
                writer_.flush();
                engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
            }

            ~FlowEncoderAttentionBackendRunner() {
                release_graph_resources(execution_context_, graph_, ggml_, gallocr_);
            }

            engine::core::TensorValue & input_tensor() {
                return input_tensor_;
            }

            void write_attention_mask(const std::vector<float> & values) {
                engine::core::write_tensor_f32(attention_mask_, values);
            }

            const engine::core::TensorValue & output_tensor() const {
                return output_;
            }

            void compute() {
                if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder attention");
                }
            }

        private:
            const engine::core::ExecutionContext * execution_context_ = nullptr;
            ggml_context * ggml_ = nullptr;
            ggml_gallocr_t gallocr_ = nullptr;
            ggml_cgraph * graph_ = nullptr;
            engine::core::TensorValue input_tensor_;
            engine::core::TensorValue attention_mask_;
            engine::core::TensorValue output_;
            engine::core::HostGraphPlan cpu_plan_;
            engine::core::DeferredTensorWriter writer_;
        };

        class FlowEncoderFeedForwardBackendRunner {
        public:
            FlowEncoderFeedForwardBackendRunner(
                const S3FlowEncoderWeights::EncoderLayerWeights & weights,
                const engine::core::ExecutionContext & execution_context,
                int64_t frames,
                int64_t hidden_size,
                const engine::core::BackendConfig &)
                : execution_context_(&execution_context) {
                ggml_init_params params = {};
                params.mem_size = 128ull * 1024ull * 1024ull;
                params.mem_buffer = nullptr;
                params.no_alloc = true;
                ggml_ = ggml_init(params);
                if (ggml_ == nullptr) {
                    throw std::runtime_error("failed to initialize ggml context for S3 flow encoder feed-forward");
                }

                engine::core::ModuleBuildContext ctx = {};
                ctx.ggml = ggml_;
                ctx.module_instance_name = "s3_flow_encoder_ff";
                input_tensor_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, frames, hidden_size}));
                auto x = layer_norm_lastdim(ctx, input_tensor_, weights.norm_ff, writer_, 1.0e-12f);
                auto ff = linear_lastdim(ctx, x, weights.ff.w1, writer_);
                ff = engine::core::wrap_tensor(ggml_silu(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
                ff = linear_lastdim(ctx, ff, weights.ff.w2, writer_);
                output_ = add_tensor_values(ctx, input_tensor_, ff);

                graph_ = ggml_new_graph_custom(ggml_, 16384, false);
                ggml_build_forward_expand(graph_, output_.tensor);
                allocate_graph_resources(*execution_context_, graph_, gallocr_, "S3 flow encoder feed-forward");
                writer_.flush();
                engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
            }

            ~FlowEncoderFeedForwardBackendRunner() {
                release_graph_resources(execution_context_, graph_, ggml_, gallocr_);
            }

            engine::core::TensorValue & input_tensor() {
                return input_tensor_;
            }

            const engine::core::TensorValue & output_tensor() const {
                return output_;
            }

            void compute() {
                if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder feed-forward");
                }
            }

        private:
            const engine::core::ExecutionContext * execution_context_ = nullptr;
            ggml_context * ggml_ = nullptr;
            ggml_gallocr_t gallocr_ = nullptr;
            ggml_cgraph * graph_ = nullptr;
            engine::core::TensorValue input_tensor_;
            engine::core::TensorValue output_;
            engine::core::HostGraphPlan cpu_plan_;
            engine::core::DeferredTensorWriter writer_;
        };

        FlowEncoderAttentionBackendRunner attention_runner_;
        FlowEncoderFeedForwardBackendRunner ff_runner_;
    };

    class FlowEncoderUpsampleBackendRunner {
    public:
        FlowEncoderUpsampleBackendRunner(
            const S3FlowEncoderWeights & weights,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(weights.execution_context) {
            ggml_init_params params = {};
            params.mem_size = 192ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder upsample");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_upsample";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));

            auto x = flow_upsample_repeat_conv_backend(
                ctx,
                input_tensor_,
                frames,
                hidden_size,
                weights.up_layer_conv,
                writer_, weights.config.upsample_factor);
            x = linear_lastdim(ctx, x, weights.up_embed_linear, writer_);
            x = layer_norm_lastdim(ctx, x, weights.up_embed_norm, writer_);
            output_ = engine::core::wrap_tensor(
                ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(hidden_size))),
                x.shape,
                GGML_TYPE_F32);

            graph_ = ggml_new_graph_custom(ggml_, 16384, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            allocate_graph_resources(*execution_context_, graph_, gallocr_, "S3 flow encoder upsample");
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderUpsampleBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, gallocr_);
        }

        engine::core::TensorValue & input_tensor() {
            return input_tensor_;
        }

        const engine::core::TensorValue & output_tensor() const {
            return output_;
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder upsample");
            }
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    class FlowEncoderAfterNormBackendRunner {
    public:
        FlowEncoderAfterNormBackendRunner(
            const S3FlowEncoderWeights::LayerNormWeights & weights,
            const engine::core::ExecutionContext & execution_context,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(&execution_context) {
            ggml_init_params params = {};
            params.mem_size = 64ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder after norm");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_after_norm";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));
            output_ = layer_norm_lastdim(ctx, input_tensor_, weights, writer_);

            graph_ = ggml_new_graph_custom(ggml_, 4096, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            allocate_graph_resources(*execution_context_, graph_, gallocr_, "S3 flow encoder after norm");
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderAfterNormBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, gallocr_);
        }

        engine::core::TensorValue & input_tensor() {
            return input_tensor_;
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder after norm");
            }
        }

        std::vector<float> read_output() const {
            return engine::core::read_tensor_f32(output_.tensor);
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    int64_t input_frames_ = 0;
    int64_t hidden_size_ = 0;
    int64_t output_frames_ = 0;
    FlowEncoderEmbedPrelookBackendRunner embed_prelook_runner_;
    std::vector<std::unique_ptr<FlowEncoderLayerBackendRunner>> encoder_runners_;
    FlowEncoderUpsampleBackendRunner upsample_runner_;
    std::vector<std::unique_ptr<FlowEncoderLayerBackendRunner>> up_encoder_runners_;
    FlowEncoderAfterNormBackendRunner after_norm_runner_;
};

class FlowDecoderBackendRunner {
public:
    FlowDecoderBackendRunner(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity_frames,
        const engine::core::BackendConfig &)
        : batch_(batch),
          capacity_frames_(capacity_frames),
          execution_context_(weights.execution_context) {
        if (weights.down_blocks.size() != 1 || weights.up_blocks.size() != 1) {
            throw std::runtime_error("S3 flow decoder runner expects exactly one down block and one up block");
        }
    }

    ~FlowDecoderBackendRunner() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        if (graph_ggml_ != nullptr) {
            ggml_free(graph_ggml_);
            graph_ggml_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    void ensure_active_graph(
        const S3FlowDecoderWeights & weights,
        int64_t frames,
        const engine::core::BackendConfig &) {
        const int64_t mel_channels = weights.final_proj.out_channels;
        const int64_t speaker_channels = mel_channels;
        if (frames <= 0 || frames > capacity_frames_) {
            throw std::runtime_error("S3 flow decoder active frames exceed capacity");
        }
        if (graph_ != nullptr && active_frames_ == frames) {
            return;
        }
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (graph_ggml_ != nullptr) {
            ggml_free(graph_ggml_);
            graph_ggml_ = nullptr;
        }
        output_tensor_ = {};
        attention_mask_ = {};
        skip_tensor_ = {};
        cpu_plan_.reset();
        active_frames_ = frames;
        cached_attention_mask_valid_frames_ = -1;
        cached_attention_mask_values_.clear();

        ggml_init_params params = {};
        params.mem_size = 512ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph_ggml_ = ggml_init(params);
        if (graph_ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml graph context for S3 flow decoder");
        }
        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = graph_ggml_;
        ctx.module_instance_name = "s3_flow_decoder";
        engine::core::DeferredTensorWriter writer;
        backend_ = execution_context_->backend();
        if (!backend_) {
            throw std::runtime_error("S3 flow decoder backend is not initialized");
        }

        x_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        mu_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        time_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, weights.config.time_embedding_size}));
        spks_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, speaker_channels}));
        cond_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        ggml_set_input(x_in_.tensor);
        ggml_set_input(mu_in_.tensor);
        ggml_set_input(time_in_.tensor);
        ggml_set_input(spks_in_.tensor);
        ggml_set_input(cond_in_.tensor);
        attention_mask_ =
            engine::core::make_tensor(ctx, GGML_TYPE_F16, engine::core::TensorShape::from_dims({batch_, weights.config.heads, frames, frames}));
        ggml_set_input(attention_mask_.tensor);
        ggml_set_output(x_in_.tensor);
        ggml_set_output(mu_in_.tensor);
        ggml_set_output(time_in_.tensor);
        ggml_set_output(spks_in_.tensor);
        ggml_set_output(cond_in_.tensor);
        ggml_set_output(attention_mask_.tensor);
        auto time_hidden = linear_lastdim(ctx, time_in_, weights.time_mlp_1, writer);
        time_hidden = engine::core::wrap_tensor(ggml_silu(ctx.ggml, time_hidden.tensor), time_hidden.shape, GGML_TYPE_F32);
        time_hidden = linear_lastdim(ctx, time_hidden, weights.time_mlp_2, writer);
        if (weights.meanflow) {
            r_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, weights.config.time_embedding_size}));
            ggml_set_input(r_in_.tensor);
            ggml_set_output(r_in_.tensor);
            auto r_hidden = linear_lastdim(ctx, r_in_, weights.time_mlp_1, writer);
            r_hidden = engine::core::wrap_tensor(ggml_silu(ctx.ggml, r_hidden.tensor), r_hidden.shape, GGML_TYPE_F32);
            r_hidden = linear_lastdim(ctx, r_hidden, weights.time_mlp_2, writer);
            auto concat_shape = time_hidden.shape;
            concat_shape.dims[concat_shape.rank - 1] += r_hidden.shape.dims[r_hidden.shape.rank - 1];
            auto concat_te = engine::core::wrap_tensor(
                ggml_concat(ctx.ggml, time_hidden.tensor, r_hidden.tensor, 0), concat_shape, GGML_TYPE_F32);
            time_hidden = linear_lastdim(ctx, concat_te, weights.time_embed_mixer, writer);
        }

        auto hidden_bct = cat_channels_bct(ctx, x_in_, mu_in_);
        auto repeat_spks = repeat_spks_bct(ctx, spks_in_, frames);
        hidden_bct = cat_channels_bct(ctx, hidden_bct, repeat_spks);
        hidden_bct = cat_channels_bct(ctx, hidden_bct, cond_in_);
        hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, weights.down_blocks[0].resnet, writer);
        auto hidden_btc = transpose_bct_btc(ctx, hidden_bct);
        for (const auto & transformer : weights.down_blocks[0].transformers) {
            hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
        }
        skip_tensor_ = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        hidden_bct = causal_conv1d_bct(ctx, skip_tensor_, weights.down_blocks[0].downsample, writer);

        for (const auto & mid : weights.mid_blocks) {
            hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, mid.resnet, writer);
            hidden_btc = transpose_bct_btc(ctx, hidden_bct);
            for (const auto & transformer : mid.transformers) {
                hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
            }
            hidden_bct = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        }

        hidden_bct = cat_channels_bct(ctx, hidden_bct, skip_tensor_);
        hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, weights.up_blocks[0].resnet, writer);
        hidden_btc = transpose_bct_btc(ctx, hidden_bct);
        for (const auto & transformer : weights.up_blocks[0].transformers) {
            hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
        }
        hidden_bct = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        hidden_bct = causal_conv1d_bct(ctx, hidden_bct, weights.up_blocks[0].upsample, writer);
        hidden_bct = causal_block_forward(ctx, hidden_bct, weights.final_block, writer);
        output_tensor_ = causal_conv1d_bct(ctx, hidden_bct, weights.final_proj, writer);
        ggml_set_output(output_tensor_.tensor);

        graph_ = ggml_new_graph_custom(graph_ggml_, 262144, false);
        ggml_build_forward_expand(graph_, output_tensor_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate backend graph tensors for S3 flow decoder");
        }
        if (execution_context_->uses_host_graph_plan()) {
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }
        writer.flush();
    }

    void set_conditioning(
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & mu,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t frames,
        const engine::core::BackendConfig & backend_config,
        S3FlowDecoderRunTiming * timing = nullptr) {
        ensure_active_graph(weights, frames, backend_config);
        const auto started = Clock::now();
        engine::core::write_tensor_f32(mu_in_, mu);
        engine::core::write_tensor_f32(spks_in_, spks);
        engine::core::write_tensor_f32(cond_in_, cond);
        if (timing != nullptr) {
            timing->conditioning_write_ms += engine::debug::elapsed_ms(started);
        }
    }

    S3FlowDecoderOutputs run(
        const std::vector<float> & x,
        const std::vector<float> & mask,
        const std::vector<float> & t,
        S3FlowDecoderRunTiming * timing = nullptr,
        const std::vector<float> * r = nullptr) {
        const int64_t mel_channels = x_in_.shape.dims[1];
        const int64_t time_dim = time_in_.shape.dims[1];
        if (timing != nullptr) {
            ++timing->calls;
        }
        auto started = Clock::now();
        auto time_emb_values = decoder_sinusoidal_pos_emb(t, batch_, time_dim);
        if (timing != nullptr) {
            timing->time_embedding_ms += engine::debug::elapsed_ms(started);
        }
        const int64_t valid_frames = valid_frames_from_mask(mask, batch_, active_frames_);
        started = Clock::now();
        engine::core::write_tensor_f32(x_in_, x);
        ensure_attention_mask(valid_frames);
        if (timing != nullptr) {
            timing->time_write_ms += engine::debug::elapsed_ms(started);
        }
        started = Clock::now();
        engine::core::write_tensor_f32(time_in_, time_emb_values);
        if (r != nullptr) {
            if (r_in_.tensor == nullptr) {
                throw std::runtime_error("S3 flow decoder runner was not built for meanflow but received r");
            }
            engine::core::write_tensor_f32(r_in_, decoder_sinusoidal_pos_emb(*r, batch_, time_dim));
        }
        if (timing != nullptr) {
            timing->input_write_ms += engine::debug::elapsed_ms(started);
        }
        started = Clock::now();
        const ggml_status status = engine::core::compute_graph(*execution_context_, graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow decoder");
        }
        if (timing != nullptr) {
            timing->graph_compute_ms += engine::debug::elapsed_ms(started);
        }
        S3FlowDecoderOutputs outputs;
        started = Clock::now();
        outputs.mel = engine::core::read_tensor_f32(output_tensor_.tensor);
        if (timing != nullptr) {
            timing->output_read_ms += engine::debug::elapsed_ms(started);
        }
        outputs.channels = mel_channels;
        outputs.frames = active_frames_;
        outputs.storage_frames = active_frames_;
        return outputs;
    }

private:
    void ensure_attention_mask(int64_t valid_frames) {
        if (cached_attention_mask_valid_frames_ == valid_frames) {
            return;
        }
        cached_attention_mask_values_ = make_attention_key_mask(
            batch_, active_frames_, active_frames_, valid_frames, attention_mask_.shape.dims[1]);
        engine::core::write_tensor_f16(attention_mask_, cached_attention_mask_values_);
        cached_attention_mask_valid_frames_ = valid_frames;
    }

    int64_t batch_ = 0;
    int64_t capacity_frames_ = 0;
    int64_t active_frames_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_context * graph_ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue x_in_;
    engine::core::TensorValue mu_in_;
    engine::core::TensorValue time_in_;
    engine::core::TensorValue r_in_;
    engine::core::TensorValue spks_in_;
    engine::core::TensorValue cond_in_;
    engine::core::TensorValue attention_mask_;
    engine::core::TensorValue skip_tensor_;
    engine::core::TensorValue output_tensor_;
    engine::core::HostGraphPlan cpu_plan_;
    int64_t cached_attention_mask_valid_frames_ = -1;
    std::vector<float> cached_attention_mask_values_;
};

}  // namespace

struct S3FlowSessionCache::State {
    explicit State(engine::core::BackendConfig) {}

    void prepare_encoder_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t capacity,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config) {
        if (!encoder_runner ||
            encoder_capacity_frames < capacity ||
            encoder_hidden_size != hidden_size ||
            !same_backend(encoder_backend, backend_config)) {
            encoder_runner.reset();
            encoder_runner = std::make_unique<FlowEncoderBackendRunner>(weights, capacity, hidden_size, backend_config);
            encoder_capacity_frames = capacity;
            encoder_hidden_size = hidden_size;
            encoder_backend = backend_config;
        }
    }

    void prepare_decoder_capacity(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity,
        const engine::core::BackendConfig & backend_config) {
        if (!decoder_runner ||
            decoder_batch != batch ||
            decoder_capacity_frames < capacity ||
            !same_backend(decoder_backend, backend_config)) {
            decoder_runner.reset();
            decoder_runner = std::make_unique<FlowDecoderBackendRunner>(weights, batch, capacity, backend_config);
            decoder_batch = batch;
            decoder_capacity_frames = capacity;
            decoder_backend = backend_config;
        }
    }

    FlowEncoderBackendRunner & encoder_runner_for_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t capacity_frames,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config) {
        prepare_encoder_capacity(weights, capacity_frames, hidden_size, backend_config);
        return *encoder_runner;
    }

    FlowDecoderBackendRunner & decoder_runner_for_capacity(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity_frames,
        const engine::core::BackendConfig & backend_config) {
        prepare_decoder_capacity(weights, batch, capacity_frames, backend_config);
        return *decoder_runner;
    }

    std::unique_ptr<FlowEncoderBackendRunner> encoder_runner;
    int64_t encoder_capacity_frames = 0;
    int64_t encoder_hidden_size = 0;
    engine::core::BackendConfig encoder_backend;
    std::unique_ptr<FlowDecoderBackendRunner> decoder_runner;
    int64_t decoder_batch = 0;
    int64_t decoder_capacity_frames = 0;
    engine::core::BackendConfig decoder_backend;
};

S3FlowSessionCache::S3FlowSessionCache(engine::core::BackendConfig backend)
    : state_(std::make_unique<State>(backend)) {}

S3FlowSessionCache::~S3FlowSessionCache() = default;

void S3FlowSessionCache::release_encoder_graphs() {
    state_->encoder_runner.reset();
    state_->encoder_capacity_frames = 0;
    state_->encoder_hidden_size = 0;
    state_->encoder_backend = {};
}

void S3FlowSessionCache::release_decoder_graphs() {
    state_->decoder_runner.reset();
    state_->decoder_batch = 0;
    state_->decoder_capacity_frames = 0;
    state_->decoder_backend = {};
}

std::shared_ptr<const S3FlowEncoderWeights> load_s3_flow_encoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    const S3GenConfig & config,
    const std::string & tensor_prefix,
    engine::assets::TensorStorageType weight_storage_type) {
    auto weights = std::make_shared<S3FlowEncoderWeights>();
    weights->config = config.encoder;
    const auto & c = config.encoder;
    const auto hidden = c.hidden_size;
    const auto mel = config.mel_channels;
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "s3gen.encoder.weights",
        1024ull * 1024ull * 1024ull);
    // ggml_get_rows (used by S3TokenEmbeddingGraph for this lookup table) only supports F32/F16
    // and legacy quant types on the CUDA backend, not K-quants -- pin this one small table
    // (~6.7 MB in F16) to F16 regardless of the requested weight_storage_type so a Q4_K/Q5_K/...
    // package (e.g. Chatterbox Turbo's chatterbox-turbo-s3gen-q4_k.gguf) doesn't crash on it.
    weights->input_embedding_tensor = weights->store->load_tensor(
        source,
        tensor_prefix + "input_embedding.weight",
        engine::assets::TensorStorageType::F16,
        {c.vocabulary_size, hidden});
    weights->speaker_affine = load_flow_linear(*weights->store, source, tensor_prefix + "spk_embed_affine_layer", mel, c.speaker_embedding_size, true, weight_storage_type);
    weights->encoder_proj = load_flow_linear(*weights->store, source, tensor_prefix + "encoder_proj", mel, hidden, true, weight_storage_type);
    weights->embed_linear = load_flow_linear(*weights->store, source, tensor_prefix + "encoder.embed.out.0", hidden, hidden, true, weight_storage_type);
    weights->embed_norm = load_flow_layer_norm(*weights->store, source, tensor_prefix + "encoder.embed.out.1", hidden);
    weights->prelook_conv1 =
        load_flow_conv1d(*weights->store, source, tensor_prefix + "encoder.pre_lookahead_layer.conv1", hidden, hidden, c.prelook_kernel_size, 1, 0, weight_storage_type);
    weights->prelook_conv2 =
        load_flow_conv1d(*weights->store, source, tensor_prefix + "encoder.pre_lookahead_layer.conv2", hidden, hidden, c.prelook_output_kernel_size, 1, 0, weight_storage_type);
    auto load_layer = [&](const std::string & prefix) {
        S3FlowEncoderWeights::EncoderLayerWeights layer;
        layer.norm_mha = load_flow_layer_norm(*weights->store, source, prefix + ".norm_mha", hidden);
        layer.attn.q = load_flow_linear(*weights->store, source, prefix + ".self_attn.linear_q", hidden, hidden, true, weight_storage_type);
        layer.attn.k = load_flow_linear(*weights->store, source, prefix + ".self_attn.linear_k", hidden, hidden, true, weight_storage_type);
        layer.attn.v = load_flow_linear(*weights->store, source, prefix + ".self_attn.linear_v", hidden, hidden, true, weight_storage_type);
        layer.attn.out = load_flow_linear(*weights->store, source, prefix + ".self_attn.linear_out", hidden, hidden, true, weight_storage_type);
        layer.attn.pos = load_flow_linear(*weights->store, source, prefix + ".self_attn.linear_pos", hidden, hidden, false, weight_storage_type);
        layer.attn.pos_bias_u_tensor = weights->store->load_f32_tensor(source, prefix + ".self_attn.pos_bias_u", {c.heads, hidden / c.heads});
        layer.attn.pos_bias_v_tensor = weights->store->load_f32_tensor(source, prefix + ".self_attn.pos_bias_v", {c.heads, hidden / c.heads});
        layer.norm_ff = load_flow_layer_norm(*weights->store, source, prefix + ".norm_ff", hidden);
        layer.ff.w1 = load_flow_linear(*weights->store, source, prefix + ".feed_forward.w_1", c.feed_forward_size, hidden, true, weight_storage_type);
        layer.ff.w2 = load_flow_linear(*weights->store, source, prefix + ".feed_forward.w_2", hidden, c.feed_forward_size, true, weight_storage_type);
        return layer;
    };
    for (int64_t i = 0; i < c.layers; ++i) {
        weights->encoders.push_back(load_layer(tensor_prefix + "encoder.encoders." + std::to_string(i)));
    }
    weights->up_layer_conv = load_flow_conv1d(*weights->store, source, tensor_prefix + "encoder.up_layer.conv", hidden, hidden, c.upsample_kernel_size, 1, 0, weight_storage_type);
    weights->up_embed_linear = load_flow_linear(*weights->store, source, tensor_prefix + "encoder.up_embed.out.0", hidden, hidden, true, weight_storage_type);
    weights->up_embed_norm = load_flow_layer_norm(*weights->store, source, tensor_prefix + "encoder.up_embed.out.1", hidden);
    for (int64_t i = 0; i < c.upsample_layers; ++i) {
        weights->up_encoders.push_back(load_layer(tensor_prefix + "encoder.up_encoders." + std::to_string(i)));
    }
    weights->after_norm = load_flow_layer_norm(*weights->store, source, tensor_prefix + "encoder.after_norm", hidden);
    weights->store->upload();
    return weights;
}

S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowEncoderWeights & weights,
    const std::vector<float> & input,
    int64_t frames,
    int64_t capacity_frames,
    int64_t hidden_size,
    engine::core::BackendConfig backend) {
    return cache.state_->encoder_runner_for_capacity(weights, capacity_frames, hidden_size, backend).run(input, frames);
}

std::shared_ptr<const S3FlowDecoderWeights> load_s3_flow_decoder_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    const S3GenConfig & config,
    const std::string & tensor_prefix,
    engine::assets::TensorStorageType weight_storage_type) {
    auto weights = std::make_shared<S3FlowDecoderWeights>();
    weights->config = config.decoder;
    const auto & c = config.decoder;
    const auto hidden = c.hidden_size;
    const auto inner = c.heads * c.head_dim;
    const auto mel = config.mel_channels;
    const auto estimator = tensor_prefix + "decoder.estimator.";
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "s3gen.decoder.weights",
        2048ull * 1024ull * 1024ull);

    auto load_causal_block = [&](const std::string & prefix, int64_t in_channels, int64_t out_channels) {
        S3FlowDecoderWeights::CausalBlockWeights block;
        block.conv = load_decoder_conv1d(*weights->store, source, prefix + ".block.0", out_channels, in_channels, 3, 1, true, weight_storage_type);
        block.norm = load_decoder_layer_norm(*weights->store, source, prefix + ".block.2", out_channels);
        return block;
    };

    auto load_resnet = [&](const std::string & prefix, int64_t in_channels, int64_t out_channels) {
        S3FlowDecoderWeights::ResnetBlockWeights block;
        block.time_mlp = load_decoder_linear(*weights->store, source, prefix + ".mlp.1", out_channels, c.time_hidden_size, true, weight_storage_type);
        block.block1 = load_causal_block(prefix + ".block1", in_channels, out_channels);
        block.block2 = load_causal_block(prefix + ".block2", out_channels, out_channels);
        block.res_conv = load_decoder_conv1d(*weights->store, source, prefix + ".res_conv", out_channels, in_channels, 1, 1, true, weight_storage_type);
        return block;
    };

    auto load_transformer = [&](const std::string & prefix) {
        S3FlowDecoderWeights::TransformerBlockWeights block;
        block.heads = c.heads;
        block.head_dim = c.head_dim;
        block.norm1 = load_decoder_layer_norm(*weights->store, source, prefix + ".norm1", hidden);
        block.attn_q = load_decoder_linear(*weights->store, source, prefix + ".attn1.to_q", inner, hidden, false, weight_storage_type);
        block.attn_k = load_decoder_linear(*weights->store, source, prefix + ".attn1.to_k", inner, hidden, false, weight_storage_type);
        block.attn_v = load_decoder_linear(*weights->store, source, prefix + ".attn1.to_v", inner, hidden, false, weight_storage_type);
        block.attn_out = load_decoder_linear(*weights->store, source, prefix + ".attn1.to_out.0", hidden, inner, true, weight_storage_type);
        block.norm3 = load_decoder_layer_norm(*weights->store, source, prefix + ".norm3", hidden);
        block.ff_proj_in = load_decoder_linear(*weights->store, source, prefix + ".ff.net.0.proj", c.feed_forward_size, hidden, true, weight_storage_type);
        block.ff_proj_out = load_decoder_linear(*weights->store, source, prefix + ".ff.net.2", hidden, c.feed_forward_size, true, weight_storage_type);
        return block;
    };

    weights->time_mlp_1 =
        load_decoder_linear(*weights->store, source, estimator + "time_mlp.linear_1", c.time_hidden_size, c.time_embedding_size, true, weight_storage_type);
    weights->time_mlp_2 =
        load_decoder_linear(*weights->store, source, estimator + "time_mlp.linear_2", c.time_hidden_size, c.time_hidden_size, true, weight_storage_type);

    weights->down_blocks.resize(1);
    weights->down_blocks[0].resnet = load_resnet(estimator + "down_blocks.0.0", 4 * mel, hidden);
    for (int64_t i = 0; i < c.attention_layers; ++i) {
        weights->down_blocks[0].transformers.push_back(load_transformer(estimator + "down_blocks.0.1." + std::to_string(i)));
    }
    weights->down_blocks[0].downsample =
        load_decoder_conv1d(*weights->store, source, estimator + "down_blocks.0.2", hidden, hidden, 3, 1, true, weight_storage_type);

    weights->mid_blocks.resize(c.middle_blocks);
    for (int64_t block_index = 0; block_index < c.middle_blocks; ++block_index) {
        weights->mid_blocks[static_cast<size_t>(block_index)].resnet =
            load_resnet(estimator + "mid_blocks." + std::to_string(block_index) + ".0", hidden, hidden);
        for (int64_t i = 0; i < c.attention_layers; ++i) {
            weights->mid_blocks[static_cast<size_t>(block_index)].transformers.push_back(
                load_transformer(estimator + "mid_blocks." + std::to_string(block_index) + ".1." + std::to_string(i)));
        }
    }

    weights->up_blocks.resize(1);
    weights->up_blocks[0].resnet = load_resnet(estimator + "up_blocks.0.0", 2 * hidden, hidden);
    for (int64_t i = 0; i < c.attention_layers; ++i) {
        weights->up_blocks[0].transformers.push_back(load_transformer(estimator + "up_blocks.0.1." + std::to_string(i)));
    }
    weights->up_blocks[0].upsample =
        load_decoder_conv1d(*weights->store, source, estimator + "up_blocks.0.2", hidden, hidden, 3, 1, true, weight_storage_type);

    weights->final_block = load_causal_block(estimator + "final_block", hidden, hidden);
    weights->final_proj = load_decoder_conv1d(*weights->store, source, estimator + "final_proj", mel, hidden, 1, 1, true, weight_storage_type);

    if (c.variant == S3FlowVariant::MeanFlow) {
        weights->meanflow = true;
        weights->time_embed_mixer = load_decoder_linear(
            *weights->store, source, estimator + "time_embed_mixer", c.time_hidden_size, 2 * c.time_hidden_size, false, weight_storage_type);
    }

    weights->store->upload();
    return weights;
}

S3FlowCFMOutputs compute_s3_flow_cfm_euler(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    engine::core::BackendConfig backend,
    S3FlowCFMTimingBreakdown * timing) {
    const int64_t mel_channels = weights.final_proj.out_channels;
    if (timing != nullptr) {
        *timing = {};
        timing->steps = num_steps;
    }
    auto started = Clock::now();
    std::vector<float> x = noise;
    if (timing != nullptr) {
        timing->initial_state_ms += engine::debug::elapsed_ms(started);
    }

    started = Clock::now();
    std::vector<float> t_span(static_cast<size_t>(num_steps + 1), 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int64_t i = 0; i <= num_steps; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(num_steps);
        if (cosine_schedule) {
            t = 1.0f - std::cos(t * 0.5f * kPi);
        }
        t_span[static_cast<size_t>(i)] = t;
    }
    if (timing != nullptr) {
        timing->schedule_ms += engine::debug::elapsed_ms(started);
    }

    const size_t batch_stride = static_cast<size_t>(mel_channels * frames);
    const int64_t valid_frames = valid_frames_from_mask(mask, batch, frames);
    if (timing != nullptr) {
        timing->decoder_calls = num_steps;
    }

    const int64_t cfg_batch = batch * 2;
    started = Clock::now();
    auto & runner = cache.state_->decoder_runner_for_capacity(weights, cfg_batch, capacity_frames, backend);
    if (timing != nullptr) {
        timing->runner_setup_ms += engine::debug::elapsed_ms(started);
    }
    started = Clock::now();
    std::vector<float> cfg_mu(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_cond(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_mask(static_cast<size_t>(cfg_batch * frames), 0.0f);
    std::vector<float> cfg_spks(static_cast<size_t>(cfg_batch * mel_channels), 0.0f);
    if (timing != nullptr) {
        timing->zero_conditioning_ms += engine::debug::elapsed_ms(started);
    }
    std::copy(mu.begin(), mu.end(), cfg_mu.begin());
    std::copy(cond.begin(), cond.end(), cfg_cond.begin());
    std::copy(mask.begin(), mask.end(), cfg_mask.begin());
    std::copy(mask.begin(), mask.end(), cfg_mask.begin() + static_cast<ptrdiff_t>(mask.size()));
    std::copy(spks.begin(), spks.end(), cfg_spks.begin());
    runner.set_conditioning(
        weights,
        cfg_mu,
        cfg_spks,
        cfg_cond,
        frames,
        backend,
        timing == nullptr ? nullptr : &timing->conditioned);
    std::vector<float> cfg_x(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_t(static_cast<size_t>(cfg_batch), 0.0f);
    std::vector<float> cfg_dxdt(batch_stride * static_cast<size_t>(batch), 0.0f);
    for (int64_t step = 0; step < num_steps; ++step) {
        const float t0 = t_span[static_cast<size_t>(step)];
        const float t1 = t_span[static_cast<size_t>(step + 1)];
        const float dt = t1 - t0;
        std::copy(x.begin(), x.end(), cfg_x.begin());
        std::copy(x.begin(), x.end(), cfg_x.begin() + static_cast<ptrdiff_t>(x.size()));
        std::fill(cfg_t.begin(), cfg_t.end(), t0);
        const auto cfg_outputs = runner.run(
            cfg_x,
            cfg_mask,
            cfg_t,
            timing == nullptr ? nullptr : &timing->conditioned);
        const float * conditioned_ptr = cfg_outputs.mel.data();
        const float * unconditioned_ptr = conditioned_ptr + static_cast<ptrdiff_t>(x.size());
        started = Clock::now();
        for (size_t i = 0; i < cfg_dxdt.size(); ++i) {
            cfg_dxdt[i] =
                (1.0f + cfg_rate) * conditioned_ptr[i] -
                cfg_rate * unconditioned_ptr[i];
            x[i] += dt * cfg_dxdt[i];
        }
        if (timing != nullptr) {
            timing->host_update_ms += engine::debug::elapsed_ms(started);
        }
    }

    S3FlowCFMOutputs outputs;
    outputs.mel = std::move(x);
    outputs.channels = mel_channels;
    outputs.frames = valid_frames;
    outputs.storage_frames = frames;
    return outputs;
}

// Meanflow-distilled decoders (Chatterbox Turbo) were trained with CFG already baked in, so
// inference is a plain (non-CFG, non-batch-doubled) Euler solve driven by two time inputs per
// step (t, r); mirrors ConditionalCFM.basic_euler in the upstream Python reference.
S3FlowCFMOutputs compute_s3_flow_cfm_meanflow(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps,
    engine::core::BackendConfig backend) {
    const int64_t mel_channels = weights.final_proj.out_channels;
    if (!weights.meanflow) {
        throw std::runtime_error("compute_s3_flow_cfm_meanflow requires meanflow-trained S3FlowDecoderWeights");
    }
    std::vector<float> x = noise;
    std::vector<float> t_span(static_cast<size_t>(num_steps + 1), 0.0f);
    for (int64_t i = 0; i <= num_steps; ++i) {
        t_span[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(num_steps);
    }
    const int64_t valid_frames = valid_frames_from_mask(mask, batch, frames);

    auto & runner = cache.state_->decoder_runner_for_capacity(weights, batch, capacity_frames, backend);
    runner.set_conditioning(weights, mu, spks, cond, frames, backend);
    std::vector<float> t_vec(static_cast<size_t>(batch), 0.0f);
    std::vector<float> r_vec(static_cast<size_t>(batch), 0.0f);
    for (int64_t step = 0; step < num_steps; ++step) {
        const float t0 = t_span[static_cast<size_t>(step)];
        const float t1 = t_span[static_cast<size_t>(step + 1)];
        const float dt = t1 - t0;
        std::fill(t_vec.begin(), t_vec.end(), t0);
        std::fill(r_vec.begin(), r_vec.end(), t1);
        const auto outputs = runner.run(x, mask, t_vec, nullptr, &r_vec);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += dt * outputs.mel[i];
        }
    }

    S3FlowCFMOutputs outputs;
    outputs.mel = std::move(x);
    outputs.channels = mel_channels;
    outputs.frames = valid_frames;
    outputs.storage_frames = frames;
    return outputs;
}

namespace {

uint64_t choose_seed(uint64_t seed) {
    if (seed != 0) return seed;
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());
}

int64_t token_capacity_quantum(const engine::core::BackendConfig & backend) {
    (void) backend;
    return 1;
}

int64_t round_up_capacity(int64_t value, int64_t quantum) {
    if (value <= 0 || quantum <= 0) {
        throw std::runtime_error("S3 capacity rounding requires positive inputs");
    }
    return ((value + quantum - 1) / quantum) * quantum;
}

std::vector<float> make_prefix_mask_bct(int64_t batch, int64_t frames, int64_t valid_frames) {
    std::vector<float> mask(static_cast<size_t>(batch * frames), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            mask[static_cast<size_t>(batch_index * frames + frame)] = frame < valid_frames ? 1.0f : 0.0f;
        }
    }
    return mask;
}

std::vector<int32_t> zero_pad_tokens(
    const std::vector<int32_t> & tokens,
    int64_t capacity) {
    std::vector<int32_t> padded(static_cast<size_t>(capacity), 0);
    std::copy(tokens.begin(), tokens.end(), padded.begin());
    return padded;
}

std::vector<float> zero_pad_btc(
    const std::vector<float> & values,
    int64_t valid_frames,
    int64_t capacity_frames,
    int64_t channels) {
    std::vector<float> padded(static_cast<size_t>(capacity_frames * channels), 0.0f);
    for (int64_t frame = 0; frame < valid_frames; ++frame) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            padded[static_cast<size_t>(frame * channels + channel)] =
                values[static_cast<size_t>(frame * channels + channel)];
        }
    }
    return padded;
}

std::vector<float> slice_bct(
    const std::vector<float> & values,
    int64_t source_frames,
    int64_t frames,
    int64_t channels) {
    std::vector<float> sliced(static_cast<size_t>(frames * channels), 0.0f);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            sliced[static_cast<size_t>(channel * frames + frame)] =
                values[static_cast<size_t>(channel * source_frames + frame)];
        }
    }
    return sliced;
}

void zero_tail_btc(
    std::vector<float> & values,
    int64_t valid_frames,
    int64_t capacity_frames,
    int64_t channels) {
    if (valid_frames >= capacity_frames) {
        return;
    }
    for (int64_t frame = valid_frames; frame < capacity_frames; ++frame) {
        std::fill_n(
            values.begin() + static_cast<ptrdiff_t>(frame * channels),
            static_cast<size_t>(channels),
            0.0f);
    }
}

std::vector<float> make_zero_padded_conditions_bct(
    const S3GenConditioning & ref_dict,
    int64_t total_frames,
    int64_t mel_channels) {
    std::vector<float> cond(static_cast<size_t>(mel_channels * total_frames), 0.0f);
    for (int64_t frame = 0; frame < ref_dict.prompt_feat_frames; ++frame) {
        for (int64_t dim = 0; dim < ref_dict.prompt_feat_dims; ++dim) {
            cond[static_cast<size_t>(dim * total_frames + frame)] =
                ref_dict.prompt_feat[static_cast<size_t>(frame * ref_dict.prompt_feat_dims + dim)];
        }
    }
    return cond;
}

engine::core::TensorValue linear_rows_graph(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowEncoderWeights::LinearWeights & weights) {
    auto projected = engine::core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, weights.weight_tensor.tensor, contiguous(ctx, input).tensor),
        engine::core::TensorShape::from_dims({input.shape.dims[0], weights.out_features}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        auto bias_view = engine::core::reshape_tensor(
            ctx,
            weights.bias_tensor,
            engine::core::TensorShape::from_dims({1, weights.out_features}));
        auto repeated = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, projected.tensor), projected.shape, GGML_TYPE_F32);
        projected = add_tensor_values(ctx, projected, repeated);
    }
    return projected;
}

class S3TokenEmbeddingGraph {
public:
    S3TokenEmbeddingGraph(
        const S3FlowEncoderWeights & weights,
        int64_t token_count,
        const engine::core::ExecutionContext & execution_context)
        : token_count_(token_count),
          execution_context_(&execution_context) {
        ggml_init_params params = {};
        params.mem_size = 32ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize S3 token embedding graph");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_token_embedding";
        ctx.backend_type = execution_context.backend_type();
        token_ids_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({token_count_}));
        embedding_out_ = engine::core::wrap_tensor(
            ggml_get_rows(ctx.ggml, weights.input_embedding_tensor.tensor, token_ids_.tensor),
            engine::core::TensorShape::from_dims({token_count_, weights.config.hidden_size}),
            GGML_TYPE_F32);

        graph_ = ggml_new_graph_custom(ggml_, 1024, false);
        ggml_build_forward_expand(graph_, embedding_out_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate S3 token embedding graph tensors");
        }
    }

    ~S3TokenEmbeddingGraph() {
        if (execution_context_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    std::vector<float> run(const std::vector<int32_t> & ids) {
        if (static_cast<int64_t>(ids.size()) != token_count_) {
            throw std::runtime_error("S3 token embedding graph token count mismatch");
        }
        engine::core::write_tensor_i32(token_ids_, ids);
        if (engine::core::compute_backend_graph(execution_context_->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("S3 token embedding graph compute failed");
        }
        return engine::core::read_tensor_f32(embedding_out_.tensor);
    }

private:
    int64_t token_count_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue token_ids_;
    engine::core::TensorValue embedding_out_;
};

class S3Token2MelPrepareGraph {
public:
    struct Outputs {
        std::vector<float> mu_bct;
        std::vector<float> speaker;
    };

    S3Token2MelPrepareGraph(
        const S3FlowEncoderWeights & weights,
        int64_t frames,
        const engine::core::ExecutionContext & execution_context)
        : frames_(frames),
          execution_context_(&execution_context) {
        ggml_init_params params = {};
        params.mem_size = 64ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize S3 token2mel prepare graph");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_token2mel_prepare";
        ctx.backend_type = execution_context.backend_type();
        encoded_in_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({frames_, weights.config.hidden_size}));
        speaker_in_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, weights.config.speaker_embedding_size}));

        auto mu_rows = linear_rows_graph(ctx, encoded_in_, weights.encoder_proj);
        mu_bct_ = engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, mu_rows.tensor, 1, 0, 2, 3)),
            engine::core::TensorShape::from_dims({weights.encoder_proj.out_features, frames_}),
            GGML_TYPE_F32);

        auto speaker_sq = engine::core::wrap_tensor(ggml_sqr(ctx.ggml, speaker_in_.tensor), speaker_in_.shape, GGML_TYPE_F32);
        auto norm_sq = engine::core::wrap_tensor(
            ggml_sum_rows(ctx.ggml, speaker_sq.tensor),
            engine::core::TensorShape::from_dims({1, 1}),
            GGML_TYPE_F32);
        auto norm = engine::core::wrap_tensor(ggml_sqrt(ctx.ggml, norm_sq.tensor), norm_sq.shape, GGML_TYPE_F32);
        auto norm_rep = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, norm.tensor, speaker_in_.tensor), speaker_in_.shape, GGML_TYPE_F32);
        auto speaker_norm = engine::core::wrap_tensor(
            ggml_div(ctx.ggml, speaker_in_.tensor, norm_rep.tensor),
            speaker_in_.shape,
            GGML_TYPE_F32);
        speaker_out_ = linear_rows_graph(ctx, speaker_norm, weights.speaker_affine);

        graph_ = ggml_new_graph_custom(ggml_, 4096, false);
        ggml_build_forward_expand(graph_, mu_bct_.tensor);
        ggml_build_forward_expand(graph_, speaker_out_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate S3 token2mel prepare graph tensors");
        }

    }

    ~S3Token2MelPrepareGraph() {
        if (execution_context_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    Outputs run(
        const std::vector<float> & encoded,
        const std::vector<float> & speaker_embedding) {
        engine::core::write_tensor_f32(encoded_in_, encoded);
        engine::core::write_tensor_f32(speaker_in_, speaker_embedding);
        if (engine::core::compute_backend_graph(execution_context_->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("S3 token2mel prepare graph compute failed");
        }
        Outputs outputs;
        outputs.mu_bct = engine::core::read_tensor_f32(mu_bct_.tensor);
        outputs.speaker = engine::core::read_tensor_f32(speaker_out_.tensor);
        return outputs;
    }

private:
    int64_t frames_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue encoded_in_;
    engine::core::TensorValue speaker_in_;
    engine::core::TensorValue mu_bct_;
    engine::core::TensorValue speaker_out_;
};

std::vector<float> make_gaussian_full_noise(int64_t channels, int64_t frames, uint64_t seed) {
    return engine::sampling::generate_torch_cuda_randn(
        static_cast<size_t>(channels * frames),
        choose_seed(seed),
        engine::sampling::TorchRandnPrecision::Float32);
}

}  // namespace

struct S3GenSessionCache::State {
    State(engine::core::BackendConfig backend_value, bool retain_graphs)
        : backend(std::move(backend_value)),
          token_capacity_controller(runtime::GraphCapacityMode::Grow),
          flow_cache(backend), retain_encoder_graphs(retain_graphs) {}

    struct PreparedCapacity {
        int64_t token_capacity = 0;
        int64_t frame_capacity = 0;
        std::unique_ptr<S3TokenEmbeddingGraph> token_embedding;
        std::unique_ptr<S3Token2MelPrepareGraph> token2mel_prepare;
    };

    runtime::MappedGraphCapacityAdapter make_token_capacity_adapter(
        const S3FlowEncoderWeights & weights,
        const engine::core::BackendConfig & backend_config) {
        const int64_t quantum = token_capacity_quantum(backend_config);
        return runtime::MappedGraphCapacityAdapter(
            quantum,
            quantum,
            [quantum](int64_t request_size) {
                return round_up_capacity(request_size, quantum);
            },
            [this]() {
                std::vector<int64_t> capacities;
                if (prepared_capacity.token_embedding && prepared_capacity.token2mel_prepare) {
                    capacities.push_back(prepared_capacity.token_capacity);
                }
                return capacities;
            },
            [this, &weights](int64_t capacity) {
                prepare_token_capacity(weights, capacity);
            });
    }

    void prepare_token_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t token_capacity) {
        if (token_capacity <= 0) {
            throw std::runtime_error("S3 token capacity must be positive");
        }
        if (prepared_capacity.token_embedding &&
            prepared_capacity.token2mel_prepare &&
            prepared_capacity.token_capacity == token_capacity) {
            return;
        }
        if (weights.execution_context == nullptr) {
            throw std::runtime_error("S3 token2mel prepare requires loaded backend weights");
        }
        prepared_capacity.token_capacity = token_capacity;
        prepared_capacity.frame_capacity = token_capacity * weights.config.upsample_factor;
        prepared_capacity.token_embedding =
            std::make_unique<S3TokenEmbeddingGraph>(weights, token_capacity, *weights.execution_context);
        prepared_capacity.token2mel_prepare =
            std::make_unique<S3Token2MelPrepareGraph>(weights, prepared_capacity.frame_capacity, *weights.execution_context);
    }

    PreparedCapacity & prepared_bundle(
        const S3FlowEncoderWeights & weights,
        int64_t token_count,
        const engine::core::BackendConfig & backend_config) {
        if (retain_encoder_graphs && prepared_capacity.token_capacity != token_count) {
            // Do not pad shorter requests to a previous long request's capacity.
            release_pre_cfm_graphs();
        }
        auto adapter = make_token_capacity_adapter(weights, backend_config);
        token_capacity_controller.ensure_prepared(adapter, token_count);
        const int64_t selected_capacity = token_capacity_controller.select_capacity_for_run(adapter, token_count);
        if (prepared_capacity.token_capacity != selected_capacity) {
            throw std::runtime_error("S3 token capacity controller selected an unprepared capacity");
        }
        return prepared_capacity;
    }

    void release_pre_cfm_graphs() {
        prepared_capacity.token_embedding.reset();
        prepared_capacity.token2mel_prepare.reset();
        flow_cache.release_encoder_graphs();
    }

    void release_cfm_decoder_graphs() {
        flow_cache.release_decoder_graphs();
    }

    engine::core::BackendConfig backend;
    runtime::GraphCapacityController token_capacity_controller;
    S3FlowSessionCache flow_cache;
    PreparedCapacity prepared_capacity;
    bool retain_encoder_graphs;
};

S3GenSessionCache::S3GenSessionCache(engine::core::BackendConfig backend, bool retain_encoder_graphs)
    : state_(std::make_unique<State>(backend, retain_encoder_graphs)) {}

S3GenSessionCache::~S3GenSessionCache() = default;

void S3GenSessionCache::release_runtime_graphs() {
    state_->release_pre_cfm_graphs();
    state_->release_cfm_decoder_graphs();
}

S3Token2MelOutputs compute_s3_token2mel_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const S3GenConditioning & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    const std::vector<float> & full_noise,
    uint64_t flow_seed,
    engine::core::BackendConfig backend,
    S3GenTimingBreakdown * timing) {
    constexpr int64_t batch = 1;
    const int64_t token_dim = encoder_weights.config.hidden_size;
    const int64_t mel_dim = encoder_weights.encoder_proj.out_features;
    const int64_t prompt_tokens = ref_dict.prompt_token_count;
    const int64_t total_tokens = prompt_tokens + speech_token_count;
    std::vector<int32_t> concat_tokens;
    concat_tokens.reserve(static_cast<size_t>(total_tokens));
    concat_tokens.insert(concat_tokens.end(), ref_dict.prompt_tokens.begin(), ref_dict.prompt_tokens.end());
    concat_tokens.insert(concat_tokens.end(), speech_tokens.begin(), speech_tokens.begin() + static_cast<ptrdiff_t>(speech_token_count));

    auto & capacity_bundle = cache.state_->prepared_bundle(encoder_weights, total_tokens, backend);
    const int64_t token_capacity = capacity_bundle.token_capacity;
    const int64_t frame_capacity = capacity_bundle.frame_capacity;
    const auto padded_tokens = zero_pad_tokens(concat_tokens, token_capacity);

    const auto embed_started = std::chrono::steady_clock::now();
    auto token_emb = capacity_bundle.token_embedding->run(padded_tokens);
    zero_tail_btc(token_emb, total_tokens, token_capacity, token_dim);
    if (timing != nullptr) {
        timing->token2mel_embed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embed_started).count();
    }

    const auto encoder_started = std::chrono::steady_clock::now();
    const auto encoded =
        compute_s3_flow_encoder_forward(
            cache.state_->flow_cache,
            encoder_weights,
            token_emb,
            total_tokens,
            token_capacity,
            token_dim,
            backend);
    std::vector<float> encoded_hidden = encoded.hidden;
    if (static_cast<int64_t>(encoded_hidden.size()) != encoded.storage_frames * token_dim) {
        encoded_hidden = zero_pad_btc(encoded_hidden, encoded.frames, encoded.storage_frames, token_dim);
    }
    if (timing != nullptr) {
        timing->token2mel_encoder_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - encoder_started).count();
    }

    const auto mu_started = std::chrono::steady_clock::now();
    const int64_t total_frames = encoded.frames;
    const auto prepared = capacity_bundle.token2mel_prepare->run(
        encoded_hidden,
        ref_dict.embedding);
    auto cond = make_zero_padded_conditions_bct(ref_dict, total_frames, mel_dim);
    auto mask = make_prefix_mask_bct(batch, total_frames, total_frames);
    auto mu = slice_bct(prepared.mu_bct, frame_capacity, total_frames, mel_dim);
    if (timing != nullptr) {
        timing->token2mel_mu_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mu_started).count();
    }

    std::vector<float> noise = full_noise.empty()
        ? make_gaussian_full_noise(mel_dim, total_frames, flow_seed)
        : full_noise;
    if (static_cast<int64_t>(noise.size()) != mel_dim * total_frames) {
        throw std::runtime_error(
            "S3Token2Mel full_noise size mismatch: got=" +
            std::to_string(noise.size()) +
            " expected=" +
            std::to_string(mel_dim * total_frames) +
            " total_frames=" +
            std::to_string(total_frames) +
            " prompt_tokens=" +
            std::to_string(prompt_tokens) +
            " speech_tokens=" +
            std::to_string(speech_token_count));
    }
    if (!cache.state_->retain_encoder_graphs) cache.state_->release_pre_cfm_graphs();
    const auto cfm_started = std::chrono::steady_clock::now();
    // Meanflow-distilled decoders (Chatterbox Turbo) use a plain, non-CFG Euler solve driven by
    // two time inputs per step; see S3FlowDecoderWeights::meanflow and
    // compute_s3_flow_cfm_meanflow. cfg_rate/cosine_schedule are meaningless for that path and
    // are ignored, matching upstream's ConditionalCFM.forward branching on `meanflow`.
    const auto mel_full = decoder_weights.meanflow
        ? compute_s3_flow_cfm_meanflow(
              cache.state_->flow_cache,
              decoder_weights,
              noise,
              mask,
              mu,
              prepared.speaker,
              cond,
              batch,
              total_frames,
              frame_capacity,
              num_steps,
              backend)
        : compute_s3_flow_cfm_euler(
              cache.state_->flow_cache,
              decoder_weights,
              noise,
              mask,
              mu,
              prepared.speaker,
              cond,
              batch,
              total_frames,
              frame_capacity,
              num_steps,
              cfg_rate,
              cosine_schedule,
              backend,
              timing == nullptr ? nullptr : &timing->token2mel_cfm);
    if (timing != nullptr) {
        timing->token2mel_cfm_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cfm_started).count();
    }

    const int64_t prompt_frames = ref_dict.prompt_feat_frames;
    const int64_t output_frames = total_frames - prompt_frames;
    S3Token2MelOutputs outputs;
    outputs.channels = mel_dim;
    outputs.frames = output_frames;
    outputs.mel.assign(static_cast<size_t>(mel_dim * output_frames), 0.0f);
    const int64_t mel_stride = mel_full.storage_frames;
    for (int64_t dim = 0; dim < mel_dim; ++dim) {
        for (int64_t frame = 0; frame < output_frames; ++frame) {
            outputs.mel[static_cast<size_t>(dim * output_frames + frame)] =
                mel_full.mel[static_cast<size_t>(dim * mel_stride + prompt_frames + frame)];
        }
    }
    cache.state_->release_cfm_decoder_graphs();
    return outputs;
}

}  // namespace

S3GenConfig::S3GenConfig() {
    auto & config = vocoder;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
}

struct S3GenRuntime::Impl {
    const core::ExecutionContext & execution;
    std::shared_ptr<const S3FlowEncoderWeights> encoder;
    std::shared_ptr<const S3FlowDecoderWeights> decoder;
    modules::HiftVocoderComponent vocoder;
    S3GenSessionCache cache;

    Impl(std::shared_ptr<const assets::TensorSource> source,
         const core::ExecutionContext & context, const S3GenConfig & config,
         const S3GenRuntimeOptions & options, const S3GenWeightBinding & binding)
        : execution(context), cache(context.config(), options.retain_encoder_graphs) {
        if (config.encoder.heads <= 0 || config.encoder.hidden_size <= 0 ||
            config.encoder.hidden_size % config.encoder.heads != 0 ||
            config.encoder.hidden_size % 2 != 0 || config.encoder.upsample_factor <= 0 ||
            config.encoder.vocabulary_size <= 0 || config.encoder.speaker_embedding_size <= 0 ||
            config.encoder.layers < 0 || config.encoder.upsample_layers < 0 ||
            config.encoder.feed_forward_size <= 0 || config.encoder.prelook_kernel_size <= 0 ||
            config.encoder.prelook_output_kernel_size <= 0 || config.encoder.upsample_kernel_size <= 0 ||
            config.decoder.heads <= 0 || config.decoder.head_dim <= 0 ||
            config.decoder.hidden_size <= 0 || config.decoder.feed_forward_size <= 0 ||
            config.decoder.time_embedding_size < 4 || config.decoder.time_embedding_size % 2 != 0 ||
            config.decoder.time_hidden_size <= 0 || config.decoder.middle_blocks < 0 ||
            config.decoder.attention_layers < 0 || config.mel_channels <= 0 ||
            config.vocoder.in_channels != config.mel_channels) {
            throw std::runtime_error("Invalid S3Gen architecture configuration");
        }
        encoder = load_s3_flow_encoder_weights(*source, context, config, binding.flow_prefix, options.weight_storage_type);
        decoder = load_s3_flow_decoder_weights(*source, context, config, binding.flow_prefix, options.weight_storage_type);
        auto vocoder_config = config.vocoder;
        vocoder_config.weight_storage_type = options.weight_storage_type;
        vocoder_config.tensor_prefix = binding.vocoder_prefix;
        vocoder_config.weight_layout = binding.vocoder_layout;
        vocoder = modules::HiftVocoderComponent::load_from_tensor_source(source, context.config(), vocoder_config);
    }
};

S3GenRuntime::S3GenRuntime(std::shared_ptr<const assets::TensorSource> source,
                         const core::ExecutionContext & execution, S3GenConfig config,
                         S3GenRuntimeOptions options, S3GenWeightBinding binding)
    : impl_(std::make_unique<Impl>(std::move(source), execution, config, options, binding)) {}
S3GenRuntime::~S3GenRuntime() = default;
S3GenRuntime::S3GenRuntime(S3GenRuntime &&) noexcept = default;
S3GenRuntime & S3GenRuntime::operator=(S3GenRuntime &&) noexcept = default;

bool S3GenRuntime::is_meanflow() const { return impl_->decoder->meanflow; }

S3FlowEncoderOutputs S3GenRuntime::encode(
    const std::vector<float> & embeddings, int64_t frames,
    int64_t capacity_frames, int64_t hidden_size) const {
    return compute_s3_flow_encoder_forward(
        impl_->cache.state_->flow_cache, *impl_->encoder, embeddings,
        frames, capacity_frames, hidden_size, impl_->execution.config());
}

S3Token2MelOutputs S3GenRuntime::token_to_mel(
    const S3GenConditioning & reference,
    const std::vector<int32_t> & speech_tokens, int64_t speech_token_count,
    int64_t num_steps, float cfg_rate, bool cosine_schedule,
    const std::vector<float> & full_noise, uint64_t flow_seed,
    S3GenTimingBreakdown * timing) const {
    return compute_s3_token2mel_inference(
        impl_->cache, *impl_->encoder, *impl_->decoder, reference,
        speech_tokens, speech_token_count, num_steps, cfg_rate, cosine_schedule,
        full_noise, flow_seed, impl_->execution.config(), timing);
}

modules::HiftVocoderOutput S3GenRuntime::decode_waveform(
    const std::vector<float> & mel, int64_t frames,
    uint64_t seed, uint64_t prior_noise_values) const {
    return impl_->vocoder.synthesize(mel, frames, seed, prior_noise_values);
}

const modules::HiftVocoderComponent & S3GenRuntime::vocoder() const {
    return impl_->vocoder;
}

void S3GenRuntime::release_flow_graphs() const {
    impl_->cache.release_runtime_graphs();
}

void S3GenRuntime::release_vocoder_graphs() const {
    impl_->vocoder.release_runtime_cache();
}

S3GenInferenceOutputs S3GenRuntime::synthesize(
    const S3GenConditioning & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    const std::vector<float> & full_noise,
    uint64_t flow_seed,
    uint64_t vocoder_seed,
    S3GenTimingBreakdown * timing) const {
    const auto token2mel_started = std::chrono::steady_clock::now();
    const auto mel = token_to_mel(
        ref_dict,
        speech_tokens,
        speech_token_count,
        num_steps,
        cfg_rate,
        cosine_schedule,
        full_noise,
        flow_seed,
        timing);
    if (timing != nullptr) {
        timing->token2mel_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - token2mel_started).count();
    }
    const uint64_t prior_noise_values = full_noise.empty()
        ? static_cast<uint64_t>(mel.channels * (ref_dict.prompt_feat_frames + mel.frames))
        : 0;
    const auto vocoder_started = std::chrono::steady_clock::now();
    const auto vocoder_outputs = decode_waveform(
        mel.mel,
        mel.frames,
        vocoder_seed,
        prior_noise_values);
    if (timing != nullptr) {
        timing->vocoder_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vocoder_started).count();
    }
    S3GenInferenceOutputs outputs;
    outputs.waveform = vocoder_outputs.waveform;
    outputs.samples = vocoder_outputs.samples;
    outputs.mel = mel.mel;
    outputs.mel_channels = mel.channels;
    outputs.mel_frames = mel.frames;
    return outputs;
}

}  // namespace engine::codecs::s3gen

#include "components/component_weights.h"
#include "components/audio_processing.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <mutex>

namespace engine::models::chatterbox::components {
namespace {

inline int torch_round_to_int(float value) {
    return static_cast<int>(std::nearbyint(value));
}

bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

S3TokenizerV2Weights::Conv1dWeights load_tokenizer_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    engine::assets::TensorStorageType weight_storage_type) {
    S3TokenizerV2Weights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return conv;
}

S3TokenizerV2Weights::LayerNormWeights load_tokenizer_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    S3TokenizerV2Weights::LayerNormWeights weights;
    weights.weight_tensor = store.load_f32_tensor(source, prefix + ".weight", {channels});
    weights.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {channels});
    return weights;
}

S3TokenizerV2Weights::LinearWeights load_tokenizer_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3TokenizerV2Weights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return linear;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::ensure_backend_addressable_layout(ctx, input);
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
    auto rhs_transposed = contiguous(ctx, transpose_last_two(ctx, rhs));
    auto output_shape = lhs.shape;
    output_shape.dims[lhs.shape.rank - 1] = rhs.shape.dims[rhs.shape.rank - 1];
    return engine::core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor),
        output_shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue linear_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3TokenizerV2Weights::LinearWeights & weights) {
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

engine::core::TensorValue layer_norm_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3TokenizerV2Weights::LayerNormWeights & weights,
    float eps = 1.0e-5f) {
    auto normalized = engine::core::wrap_tensor(
        ggml_norm(ctx.ggml, contiguous(ctx, input).tensor, eps),
        input.shape,
        GGML_TYPE_F32);
    auto scaled = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous(ctx, normalized).tensor, weights.weight_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, scaled).tensor, weights.bias_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue bct_to_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return permute_tensor(ctx, input, {0, 2, 1});
}

engine::core::TensorValue btc_to_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return permute_tensor(ctx, input, {0, 2, 1});
}

engine::core::TensorValue concat_along_axis(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs,
    int logical_axis) {
    auto output_shape = lhs.shape;
    output_shape.dims[logical_axis] += rhs.shape.dims[logical_axis];
    return engine::core::wrap_tensor(
        ggml_concat(
            ctx.ggml,
            lhs.tensor,
            rhs.tensor,
            engine::core::logical_axis_to_ggml_axis(lhs.shape.rank, logical_axis)),
        output_shape,
        lhs.type);
}

engine::core::TensorValue slice_last_dim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t length) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        length,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->ne[3],
        input.tensor->nb[1],
        input.tensor->nb[2],
        input.tensor->nb[3],
        static_cast<size_t>(start) * sizeof(float));
    auto output_shape = input.shape;
    output_shape.dims[input.shape.rank - 1] = length;
    return engine::core::wrap_tensor(view, output_shape, input.type);
}

engine::core::TensorValue view_btc_last_dim_chunk(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t length) {
    auto * view = ggml_view_3d(
        ctx.ggml,
        input.tensor,
        length,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->nb[1],
        input.tensor->nb[2],
        static_cast<size_t>(start) * sizeof(float));
    return engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], length}),
        input.type);
}

engine::core::TensorValue view_bthd_from_btc_chunk(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t time,
    int64_t heads,
    int64_t head_dim) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        head_dim,
        heads,
        time,
        input.shape.dims[0],
        static_cast<size_t>(head_dim) * sizeof(float),
        input.tensor->nb[1],
        input.tensor->nb[2],
        static_cast<size_t>(start) * sizeof(float));
    return engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], time, heads, head_dim}),
        input.type);
}

engine::core::TensorValue negate_tensor(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, contiguous(ctx, input).tensor, -1.0f),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue tokenizer_conv1d_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3TokenizerV2Weights::Conv1dWeights & weights) {
    const int64_t batch = input.shape.dims[0];
    const int64_t in_frames = input.shape.dims[2];
    const int64_t out_frames =
        (in_frames + 2 * weights.padding - weights.kernel) / weights.stride + 1;
    auto output = engine::core::wrap_tensor(
        ggml_conv_1d(
            ctx.ggml,
            contiguous(ctx, weights.weight_tensor).tensor,
            contiguous(ctx, input).tensor,
            static_cast<int>(weights.stride),
            static_cast<int>(weights.padding),
            1),
        engine::core::TensorShape::from_dims({batch, weights.out_channels, out_frames}),
        GGML_TYPE_F32);
    const auto bias_view = engine::core::reshape_tensor(
        ctx,
        weights.bias_tensor,
        engine::core::TensorShape::from_dims({1, weights.out_channels, 1}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, output.tensor, bias_view.tensor),
        output.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue tokenizer_fsmn_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const engine::core::TensorValue & depthwise_weight) {
    const int64_t channels = input_btc.shape.dims[2];
    const int64_t time = input_btc.shape.dims[1];
    auto input_bct = btc_to_bct(ctx, input_btc);
    auto memory = engine::core::wrap_tensor(
        ggml_conv_1d_dw(ctx.ggml, contiguous(ctx, depthwise_weight).tensor, contiguous(ctx, input_bct).tensor, 1, 15, 1),
        engine::core::TensorShape::from_dims({1, channels, time}),
        GGML_TYPE_F32);
    memory = engine::core::wrap_tensor(
        ggml_add(ctx.ggml, memory.tensor, input_bct.tensor),
        memory.shape,
        GGML_TYPE_F32);
    return bct_to_btc(ctx, memory);
}

engine::core::TensorValue apply_rotary_embedding_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cos_values,
    const engine::core::TensorValue & sin_values,
    int64_t head_dim) {
    const int64_t half = head_dim / 2;
    auto left = slice_last_dim(ctx, input, 0, half);
    auto right = slice_last_dim(ctx, input, half, half);
    auto rotated = concat_along_axis(ctx, negate_tensor(ctx, right), left, 3);

    auto scaled_input = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous(ctx, input).tensor, cos_values.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto scaled_rotated = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, contiguous(ctx, rotated).tensor, sin_values.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, scaled_input).tensor, contiguous(ctx, scaled_rotated).tensor),
        input.shape,
        GGML_TYPE_F32);
}

class S3TokenizerBackendRunner {
public:
    S3TokenizerBackendRunner(
        const S3TokenizerV2Weights & weights,
        int64_t input_frames)
        : time_(input_frames),
          execution_context_(*weights.execution_context) {
        constexpr int64_t batch = 1;
        constexpr int64_t input_mels = 128;
        constexpr int64_t channels = 1280;
        constexpr int64_t heads = 20;
        constexpr int64_t head_dim = 64;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        ggml_init_params params = {};
        params.mem_size = 512ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for S3 tokenizer");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "s3_tokenizer";
        ctx.backend_type = execution_context_.backend_type();

        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, input_mels, input_frames}));

        auto x = tokenizer_conv1d_bct(ctx, input_, weights.conv1);
        x = engine::core::wrap_tensor(ggml_gelu_erf(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        time_ = x.shape.dims[2];

        x = tokenizer_conv1d_bct(ctx, x, weights.conv2);
        x = engine::core::wrap_tensor(ggml_gelu_erf(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        time_ = x.shape.dims[2];

        cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, time_, 1, head_dim}));
        sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, time_, 1, head_dim}));

        auto seq = bct_to_btc(ctx, x);
        for (const auto & block : weights.blocks) {
            auto norm = layer_norm_lastdim(ctx, seq, block.attn_ln);
            auto qkv = contiguous(ctx, linear_lastdim(ctx, norm, block.attn_qkv_packed));
            auto q = view_bthd_from_btc_chunk(ctx, qkv, 0, time_, heads, head_dim);
            auto k = view_bthd_from_btc_chunk(ctx, qkv, channels, time_, heads, head_dim);
            auto v_flat = view_btc_last_dim_chunk(ctx, qkv, 2 * channels, channels);
            auto fsmn = tokenizer_fsmn_btc(ctx, v_flat, block.fsmn_weight_tensor);

            auto v = view_bthd_from_btc_chunk(ctx, qkv, 2 * channels, time_, heads, head_dim);

            q = apply_rotary_embedding_backend(ctx, q, cos_, sin_, head_dim);
            k = apply_rotary_embedding_backend(ctx, k, cos_, sin_, head_dim);

            q = permute_tensor(ctx, q, {0, 2, 1, 3});
            k = permute_tensor(ctx, k, {0, 2, 1, 3});
            v = permute_tensor(ctx, v, {0, 2, 1, 3});

            auto scores = batched_matmul(ctx, q, permute_tensor(ctx, k, {0, 1, 3, 2}));
            scores = engine::core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, attn_scale), scores.shape, GGML_TYPE_F32);
            scores = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);

            auto context = batched_matmul(ctx, scores, v);
            context = permute_tensor(ctx, context, {0, 2, 1, 3});
            context = contiguous(ctx, context);
            context = engine::core::reshape_tensor(
                ctx,
                context,
                engine::core::TensorShape::from_dims({batch, time_, channels}));
            auto attn_out = linear_lastdim(ctx, context, block.attn_out);

            auto sum = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, contiguous(ctx, seq).tensor, contiguous(ctx, attn_out).tensor),
                seq.shape,
                GGML_TYPE_F32);
            seq = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, contiguous(ctx, sum).tensor, contiguous(ctx, fsmn).tensor),
                sum.shape,
                GGML_TYPE_F32);

            auto mlp_in = layer_norm_lastdim(ctx, seq, block.mlp_ln);
            auto ff = linear_lastdim(ctx, mlp_in, block.mlp_fc1);
            ff = engine::core::wrap_tensor(ggml_gelu_erf(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
            ff = linear_lastdim(ctx, ff, block.mlp_fc2);
            seq = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, contiguous(ctx, seq).tensor, contiguous(ctx, ff).tensor),
                seq.shape,
                GGML_TYPE_F32);
        }

        quant_out_ = linear_lastdim(ctx, seq, weights.quantizer_project_down);
        ggml_set_name(quant_out_.tensor, "framework_s3tokenizer_quant_out");

        graph_ = ggml_new_graph_custom(ggml_, 65536, false);
        ggml_build_forward_expand(graph_, quant_out_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            ggml_free(ggml_);
            ggml_ = nullptr;
            throw std::runtime_error("failed to allocate backend tensors for S3 tokenizer");
        }

        std::vector<float> cos_values(static_cast<size_t>(time_ * head_dim), 0.0f);
        std::vector<float> sin_values(static_cast<size_t>(time_ * head_dim), 0.0f);
        const int64_t half = head_dim / 2;
        for (int64_t t = 0; t < time_; ++t) {
            for (int64_t i = 0; i < half; ++i) {
                const float inv_freq =
                    1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(head_dim));
                const float angle = static_cast<float>(t) * inv_freq;
                const float cs = std::cos(angle);
                const float sn = std::sin(angle);
                const size_t left = static_cast<size_t>(t * head_dim + i);
                const size_t right = static_cast<size_t>(t * head_dim + half + i);
                cos_values[left] = cs;
                cos_values[right] = cs;
                sin_values[left] = sn;
                sin_values[right] = sn;
            }
        }
        engine::core::write_tensor_f32(cos_, cos_values);
        engine::core::write_tensor_f32(sin_, sin_values);
    }

    ~S3TokenizerBackendRunner() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_context_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
    }

    TokenizerOutputs run(const std::vector<float> & mel_bct) {
        std::lock_guard<std::mutex> lock(run_mutex_);
        engine::core::write_tensor_f32(input_, mel_bct);
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for S3 tokenizer");
        }
        auto down = engine::core::read_tensor_f32(quant_out_.tensor);

        TokenizerOutputs outputs;
        outputs.token_count = time_;
        outputs.tokens.assign(static_cast<size_t>(time_), 0);
        const int powers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
        for (int64_t t = 0; t < time_; ++t) {
            int token = 0;
            for (int64_t d = 0; d < 8; ++d) {
                float xval = std::tanh(down[static_cast<size_t>(t * 8 + d)]);
                xval *= 0.9990000128746033f;
                const int level = torch_round_to_int(xval) + 1;
                token += level * powers[d];
            }
            outputs.tokens[static_cast<size_t>(t)] = token;
        }
        return outputs;
    }

private:
    int64_t time_ = 0;
    const engine::core::ExecutionContext & execution_context_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    engine::core::TensorValue cos_;
    engine::core::TensorValue sin_;
    engine::core::TensorValue quant_out_;
    std::mutex run_mutex_;
};

}  // namespace

class S3TokenizerInferenceSessionCache {
public:
    S3TokenizerInferenceSessionCache();
    ~S3TokenizerInferenceSessionCache();
    S3TokenizerInferenceSessionCache(S3TokenizerInferenceSessionCache &&) noexcept;
    S3TokenizerInferenceSessionCache & operator=(S3TokenizerInferenceSessionCache &&) noexcept;

    S3TokenizerInferenceSessionCache(const S3TokenizerInferenceSessionCache &) = delete;
    S3TokenizerInferenceSessionCache & operator=(const S3TokenizerInferenceSessionCache &) = delete;

    TokenizerOutputs run_backend(
        const S3TokenizerV2Weights & weights,
        const std::vector<float> & mel_bct,
        int64_t frames,
        const engine::core::BackendConfig & backend);

private:
    struct State;
    std::unique_ptr<State> state_;
};

struct S3TokenizerInferenceSessionCache::State {
    std::mutex mutex;
    std::shared_ptr<S3TokenizerBackendRunner> runner;
    const S3TokenizerV2Weights * weights = nullptr;
    int64_t frames = 0;
    engine::core::BackendConfig backend;
};

S3TokenizerInferenceSessionCache::S3TokenizerInferenceSessionCache()
    : state_(std::make_unique<State>()) {}

S3TokenizerInferenceSessionCache::~S3TokenizerInferenceSessionCache() = default;
S3TokenizerInferenceSessionCache::S3TokenizerInferenceSessionCache(S3TokenizerInferenceSessionCache &&) noexcept = default;
S3TokenizerInferenceSessionCache &
S3TokenizerInferenceSessionCache::operator=(S3TokenizerInferenceSessionCache &&) noexcept = default;

TokenizerOutputs S3TokenizerInferenceSessionCache::run_backend(
    const S3TokenizerV2Weights & weights,
    const std::vector<float> & mel_bct,
    int64_t frames,
    const engine::core::BackendConfig & backend) {
    std::shared_ptr<S3TokenizerBackendRunner> runner;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->runner ||
            state_->weights != &weights ||
            state_->frames != frames ||
            !same_backend(state_->backend, backend)) {
            if (!same_backend(weights.execution_context->config(), backend)) {
                throw std::runtime_error("S3 tokenizer backend does not match uploaded weight backend");
            }
            state_->runner = std::make_shared<S3TokenizerBackendRunner>(weights, frames);
            state_->weights = &weights;
            state_->frames = frames;
            state_->backend = backend;
        }
        runner = state_->runner;
    }
    return runner->run(mel_bct);
}

static std::shared_ptr<S3TokenizerInferenceSessionCache> make_s3_tokenizer_inference_session_cache() {
    return std::make_shared<S3TokenizerInferenceSessionCache>();
}

static std::shared_ptr<const S3TokenizerV2Weights> load_s3tokenizer_v2_weights(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto weights = std::make_shared<S3TokenizerV2Weights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.s3_tokenizer.weights",
        768ull * 1024ull * 1024ull);
    weights->conv1 = load_tokenizer_conv1d(
        *weights->store,
        source,
        "tokenizer.encoder.conv1",
        1280,
        128,
        3,
        2,
        1,
        weight_storage_type);
    weights->conv2 = load_tokenizer_conv1d(
        *weights->store,
        source,
        "tokenizer.encoder.conv2",
        1280,
        1280,
        3,
        2,
        1,
        weight_storage_type);
    weights->blocks.resize(6);
    for (int layer = 0; layer < 6; ++layer) {
        const std::string prefix = "tokenizer.encoder.blocks." + std::to_string(layer);
        auto & block = weights->blocks[static_cast<size_t>(layer)];
        block.attn_ln = load_tokenizer_layer_norm(*weights->store, source, prefix + ".attn_ln", 1280);
        block.attn_qkv_packed.out_features = 1280 * 3;
        block.attn_qkv_packed.in_features = 1280;
        block.attn_qkv_packed.use_bias = true;
        std::vector<float> qkv_weight(static_cast<size_t>(block.attn_qkv_packed.out_features * block.attn_qkv_packed.in_features));
        auto query_weight = read_f32_tensor(source, prefix + ".attn.query.weight", {1280, 1280});
        auto key_weight = read_f32_tensor(source, prefix + ".attn.key.weight", {1280, 1280});
        auto value_weight = read_f32_tensor(source, prefix + ".attn.value.weight", {1280, 1280});
        std::copy(
            query_weight.begin(),
            query_weight.end(),
            qkv_weight.begin());
        std::copy(
            key_weight.begin(),
            key_weight.end(),
            qkv_weight.begin() + static_cast<ptrdiff_t>(1280 * 1280));
        std::copy(
            value_weight.begin(),
            value_weight.end(),
            qkv_weight.begin() + static_cast<ptrdiff_t>(2 * 1280 * 1280));
        std::vector<float> qkv_bias(static_cast<size_t>(block.attn_qkv_packed.out_features), 0.0f);
        auto query_bias = read_f32_tensor(source, prefix + ".attn.query.bias", {1280});
        auto value_bias = read_f32_tensor(source, prefix + ".attn.value.bias", {1280});
        std::copy(
            query_bias.begin(),
            query_bias.end(),
            qkv_bias.begin());
        std::copy(
            value_bias.begin(),
            value_bias.end(),
            qkv_bias.begin() + static_cast<ptrdiff_t>(2 * 1280));
        block.attn_qkv_packed.weight_tensor = weights->store->make_from_f32(
            engine::core::TensorShape::from_dims({block.attn_qkv_packed.out_features, block.attn_qkv_packed.in_features}),
            weight_storage_type,
            std::move(qkv_weight));
        block.attn_qkv_packed.bias_tensor = weights->store->make_f32(
            engine::core::TensorShape::from_dims({block.attn_qkv_packed.out_features}),
            std::move(qkv_bias));
        block.attn_out = load_tokenizer_linear(
            *weights->store,
            source,
            prefix + ".attn.out",
            1280,
            1280,
            true,
            weight_storage_type);
        auto fsmn_weight = read_f32_tensor(source, prefix + ".attn.fsmn_block.weight", {1280, 1, 31});
        block.fsmn_weight_tensor = weights->store->make_from_f32(
            engine::core::TensorShape::from_dims({1280, 1, 31}),
            weight_storage_type,
            std::move(fsmn_weight));
        block.mlp_ln = load_tokenizer_layer_norm(*weights->store, source, prefix + ".mlp_ln", 1280);
        block.mlp_fc1 = load_tokenizer_linear(
            *weights->store,
            source,
            prefix + ".mlp.0",
            5120,
            1280,
            true,
            weight_storage_type);
        block.mlp_fc2 = load_tokenizer_linear(
            *weights->store,
            source,
            prefix + ".mlp.2",
            1280,
            5120,
            true,
            weight_storage_type);
    }
    weights->quantizer_project_down =
        load_tokenizer_linear(
            *weights->store,
            source,
            "tokenizer.quantizer._codebook.project_down",
            8,
            1280,
            true,
            weight_storage_type);
    weights->store->upload();
    return weights;
}

TokenizerOutputs compute_s3tokenizer_v2_codes(
    const S3TokenizerV2Weights & weights,
    const runtime::AudioBuffer & audio,
    std::optional<int64_t> max_len,
    S3TokenizerInferenceSessionCache * cache,
    engine::core::BackendConfig backend) {
    auto mel = compute_s3tokenizer_log_mel(audio);
    if (max_len && *max_len > 0) {
        const int64_t max_frames = *max_len * 4;
        if (mel.frames > max_frames) {
            std::vector<float> trimmed(static_cast<size_t>(mel.n_mels * max_frames), 0.0f);
            for (int64_t m = 0; m < mel.n_mels; ++m) {
                const float * src = mel.log_mel.data() + static_cast<ptrdiff_t>(m * mel.frames);
                float * dst = trimmed.data() + static_cast<ptrdiff_t>(m * max_frames);
                std::copy(src, src + max_frames, dst);
            }
            mel.log_mel = std::move(trimmed);
            mel.frames = max_frames;
        }
    }
    if (cache == nullptr) {
        S3TokenizerInferenceSessionCache local_cache;
        return local_cache.run_backend(weights, mel.log_mel, mel.frames, backend);
    }
    return cache->run_backend(weights, mel.log_mel, mel.frames, backend);
}

EmbedReferenceOutputs compute_s3_embed_ref_from_wavs(
    const S3TokenizerV2Weights & tokenizer_weights,
    const CAMPPlusEncoderComponent & speaker_encoder,
    const runtime::AudioBuffer & audio_24k,
    const runtime::AudioBuffer & audio_16k,
    S3TokenizerInferenceSessionCache * cache,
    engine::core::BackendConfig backend) {
    if (audio_24k.sample_rate != 24000) {
        throw std::runtime_error("compute_s3_embed_ref_from_wavs expects 24 kHz audio_24k");
    }
    if (audio_16k.sample_rate != 16000) {
        throw std::runtime_error("compute_s3_embed_ref_from_wavs expects 16 kHz audio_16k");
    }
    std::pair<S3PromptMelOutputs, double> prompt_mel_pair;
    std::pair<SpeakerEncoderOutputs, double> speaker_pair;
    std::pair<TokenizerOutputs, double> prompt_tokens_pair;
    {
        const auto started = std::chrono::steady_clock::now();
        auto value = compute_s3_prompt_mel(audio_24k);
        const auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        prompt_mel_pair = std::make_pair(std::move(value), ms);
    }
    {
        const auto started = std::chrono::steady_clock::now();
        auto value = speaker_encoder.embed_from_audio(audio_16k);
        const auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        speaker_pair = std::make_pair(std::move(value), ms);
    }
    {
        const auto started = std::chrono::steady_clock::now();
        auto value = compute_s3tokenizer_v2_codes(tokenizer_weights, audio_16k, std::nullopt, cache, backend);
        const auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        prompt_tokens_pair = std::make_pair(std::move(value), ms);
    }
    auto prompt_mel = std::move(prompt_mel_pair.first);
    auto speaker_embedding = std::move(speaker_pair.first);
    auto prompt_tokens = std::move(prompt_tokens_pair.first);

    const int64_t expected_token_count = prompt_mel.frames / 2;
    if (prompt_tokens.token_count != expected_token_count) {
        const int64_t keep = std::min(prompt_tokens.token_count, expected_token_count);
        prompt_tokens.tokens.resize(static_cast<size_t>(keep));
        prompt_tokens.token_count = keep;
    }

    EmbedReferenceOutputs outputs;
    outputs.prompt_tokens = std::move(prompt_tokens.tokens);
    outputs.prompt_token_count = prompt_tokens.token_count;
    outputs.prompt_feat.assign(static_cast<size_t>(prompt_mel.frames * prompt_mel.n_mels), 0.0f);
    for (int64_t mel_bin = 0; mel_bin < prompt_mel.n_mels; ++mel_bin) {
        for (int64_t frame = 0; frame < prompt_mel.frames; ++frame) {
            outputs.prompt_feat[static_cast<size_t>(frame * prompt_mel.n_mels + mel_bin)] =
                prompt_mel.mel[static_cast<size_t>(mel_bin * prompt_mel.frames + frame)];
        }
    }
    outputs.prompt_feat_frames = prompt_mel.frames;
    outputs.prompt_feat_dims = prompt_mel.n_mels;
    outputs.embedding = std::move(speaker_embedding.embedding);
    outputs.embedding_size = speaker_embedding.embedding_size;
    outputs.prompt_mel_ms = prompt_mel_pair.second;
    outputs.speaker_ms = speaker_pair.second;
    outputs.tokenizer_ms = prompt_tokens_pair.second;
    return outputs;
}

EmbedReferenceOutputs compute_s3_embed_ref(
    const S3TokenizerV2Weights & tokenizer_weights,
    const CAMPPlusEncoderComponent & speaker_encoder,
    const runtime::AudioBuffer & audio,
    S3TokenizerInferenceSessionCache * cache,
    engine::core::BackendConfig backend) {
    runtime::AudioBuffer audio_24k = audio;
    if (audio_24k.sample_rate != 24000) {
        audio_24k.sample_rate = 24000;
        audio_24k.channels = audio.channels;
        audio_24k.samples = resample_component_mono(audio.samples, audio.sample_rate, 24000);
    }
    runtime::AudioBuffer audio_16k = audio;
    if (audio_16k.sample_rate != 16000) {
        audio_16k.sample_rate = 16000;
        audio_16k.channels = audio.channels;
        audio_16k.samples = resample_component_mono(audio.samples, audio.sample_rate, 16000);
    }
    return compute_s3_embed_ref_from_wavs(tokenizer_weights, speaker_encoder, audio_24k, audio_16k, cache, backend);
}


}  // namespace engine::models::chatterbox::components

namespace engine::models::chatterbox {

struct S3TokenizerComponent::State {
    std::shared_ptr<components::S3TokenizerInferenceSessionCache> cache;
};

S3TokenizerComponent S3TokenizerComponent::load_from_source(
    const engine::assets::TensorSource & source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto runtime_weights = components::load_s3tokenizer_v2_weights(source, execution_context, weight_storage_type);
    auto weights = std::make_shared<S3TokenizerComponentWeights>();
    weights->runtime_weights = std::move(runtime_weights);
    S3TokenizerComponent component(std::move(weights), execution_context);
    component.state_ = std::make_shared<State>(State{
        components::make_s3_tokenizer_inference_session_cache(),
    });
    return component;
}

S3TokenizerComponent::S3TokenizerComponent(
    std::shared_ptr<const S3TokenizerComponentWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context) {}

const engine::core::BackendConfig & S3TokenizerComponent::backend() const noexcept {
    return execution_context_->config();
}

const std::shared_ptr<const S3TokenizerComponentWeights> & S3TokenizerComponent::weights() const noexcept {
    return weights_;
}

TokenizerOutputs S3TokenizerComponent::tokenize(
    const runtime::AudioBuffer & audio,
    std::optional<int64_t> max_len) const {
    return components::compute_s3tokenizer_v2_codes(
        *weights_->runtime_weights,
        audio,
        max_len,
        state_->cache.get(),
        execution_context_->config());
}

EmbedReferenceOutputs S3TokenizerComponent::embed_reference(
    const CAMPPlusEncoderComponent & speaker_encoder,
    const runtime::AudioBuffer & audio) const {
    return components::compute_s3_embed_ref(
        *weights_->runtime_weights,
        speaker_encoder,
        audio,
        state_->cache.get(),
        execution_context_->config());
}

EmbedReferenceOutputs S3TokenizerComponent::embed_reference_from_wavs(
    const CAMPPlusEncoderComponent & speaker_encoder,
    const runtime::AudioBuffer & audio_24k,
    const runtime::AudioBuffer & audio_16k) const {
    return components::compute_s3_embed_ref_from_wavs(
        *weights_->runtime_weights,
        speaker_encoder,
        audio_24k,
        audio_16k,
        state_->cache.get(),
        execution_context_->config());
}

}  // namespace engine::models::chatterbox

#include "engine/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace engine::modules {

namespace assets = engine::assets;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultAudioWeightContextBytes = 32ull * 1024ull * 1024ull;

int64_t qwen3_audio_encoder_token_count(int64_t input_frames) {
    if (input_frames <= 0) {
        throw std::runtime_error("Qwen3 requires positive feature frame count");
    }
    const auto floor_div = [](int64_t numerator, int64_t denominator) {
        int64_t quotient = numerator / denominator;
        const int64_t remainder = numerator % denominator;
        if (remainder != 0 && ((remainder < 0) != (denominator < 0))) --quotient;
        return quotient;
    };
    const int64_t input_lengths_leave = input_frames % 100;
    const int64_t feat_lengths = floor_div(input_lengths_leave - 1, 2) + 1;
    return floor_div(floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 +
        (input_frames / 100) * 13;
}

namespace qwen3_audio_encoder_detail {

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<int64_t> audio_chunk_lengths(int64_t input_frames, int64_t chunk_frames) {
    if (input_frames <= 0 || chunk_frames <= 0) {
        throw std::runtime_error("Qwen3 audio chunk lengths require positive sizes");
    }
    std::vector<int64_t> chunks;
    for (int64_t offset = 0; offset < input_frames; offset += chunk_frames) {
        chunks.push_back(std::min(chunk_frames, input_frames - offset));
    }
    return chunks;
}

int64_t max_value(const std::vector<int64_t> & values) {
    if (values.empty()) {
        throw std::runtime_error("Qwen3 expected a non-empty length list");
    }
    return *std::max_element(values.begin(), values.end());
}

int64_t sum_values(const std::vector<int64_t> & values) {
    int64_t sum = 0;
    for (const int64_t value : values) {
        sum += value;
    }
    return sum;
}

std::vector<int64_t> audio_attention_window_lengths(int64_t tokens, int64_t window_tokens) {
    if (tokens <= 0 || window_tokens <= 0) {
        throw std::runtime_error("Qwen3 audio attention window lengths require positive sizes");
    }
    std::vector<int64_t> lengths;
    for (int64_t offset = 0; offset < tokens; offset += window_tokens) {
        lengths.push_back(std::min(window_tokens, tokens - offset));
    }
    return lengths;
}

std::vector<float> audio_attention_mask(int64_t tokens, const std::vector<int64_t> & window_lengths) {
    std::vector<float> values(static_cast<size_t>(tokens * tokens), -INFINITY);
    int64_t offset = 0;
    for (const int64_t length : window_lengths) {
        for (int64_t row = 0; row < length; ++row) {
            for (int64_t col = 0; col < length; ++col) {
                values[static_cast<size_t>((offset + row) * tokens + offset + col)] = 0.0F;
            }
        }
        offset += length;
    }
    if (offset != tokens) {
        throw std::runtime_error("Qwen3 audio attention windows do not cover all tokens");
    }
    return values;
}

std::vector<float> sinusoidal_positions(int64_t length, int64_t channels) {
    if (channels % 2 != 0) {
        throw std::runtime_error("Qwen3 audio positional embedding requires even channel count");
    }
    std::vector<float> table(static_cast<size_t>(length * channels), 0.0F);
    const double increment = std::log(10000.0) / static_cast<double>(channels / 2 - 1);
    for (int64_t pos = 0; pos < length; ++pos) {
        for (int64_t dim = 0; dim < channels / 2; ++dim) {
            const double scaled = static_cast<double>(pos) * std::exp(-increment * static_cast<double>(dim));
            table[static_cast<size_t>(pos * channels + dim)] = static_cast<float>(std::sin(scaled));
            table[static_cast<size_t>(pos * channels + channels / 2 + dim)] = static_cast<float>(std::cos(scaled));
        }
    }
    return table;
}

core::TensorValue reshape_audio_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue audio_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::AttentionWeights & weights,
    int64_t hidden_size,
    int64_t heads,
    const core::TensorValue & attention_mask) {
    const int64_t dim = hidden_size / heads;
    const modules::LinearModule q_proj({hidden_size, hidden_size, true});
    const modules::LinearModule k_proj({hidden_size, hidden_size, true});
    const modules::LinearModule v_proj({hidden_size, hidden_size, true});
    const modules::LinearModule out_proj({hidden_size, hidden_size, true});

    auto q = reshape_audio_heads(ctx, q_proj.build(ctx, input, {weights.q_weight, weights.q_bias}), heads, dim);
    auto k = reshape_audio_heads(ctx, k_proj.build(ctx, input, {weights.k_weight, weights.k_bias}), heads, dim);
    auto v = reshape_audio_heads(ctx, v_proj.build(ctx, input, {weights.v_weight, weights.v_bias}), heads, dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    return out_proj.build(ctx, context, {weights.out_weight, weights.out_bias});
}

core::TensorValue audio_encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::TransformerEncoderBlockWeights & weights,
    const Qwen3AudioEncoderConfig & config,
    const core::TensorValue & attention_mask) {
    const modules::LayerNormModule norm({config.d_model, 1.0e-5F, true, true});
    auto attn_in = norm.build(ctx, input, weights.norm1);
    auto attn = audio_self_attention(
        ctx,
        attn_in,
        weights.self_attention,
        config.d_model,
        config.encoder_attention_heads,
        attention_mask);
    auto x = modules::AddModule().build(ctx, input, attn);
    auto ff_in = norm.build(ctx, x, weights.norm2);
    auto ff = modules::FeedForwardModule({config.d_model, config.encoder_ffn_dim, true, modules::GeluApproximation::ExactErf})
                  .build(ctx, ff_in, weights.feed_forward);
    return modules::AddModule().build(ctx, x, ff);
}

struct Conv2dWeightsData {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct LinearWeightsData {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct AudioLayerWeights {
    core::TensorValue self_attn_norm_weight;
    core::TensorValue self_attn_norm_bias;
    core::TensorValue q_proj_weight;
    core::TensorValue q_proj_bias;
    core::TensorValue k_proj_weight;
    core::TensorValue k_proj_bias;
    core::TensorValue v_proj_weight;
    core::TensorValue v_proj_bias;
    core::TensorValue out_proj_weight;
    core::TensorValue out_proj_bias;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
    core::TensorValue fc1_weight;
    core::TensorValue fc1_bias;
    core::TensorValue fc2_weight;
    core::TensorValue fc2_bias;
};

}  // namespace qwen3_audio_encoder_detail

using namespace qwen3_audio_encoder_detail;

struct Qwen3AudioEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    Conv2dWeightsData conv1;
    Conv2dWeightsData conv2;
    Conv2dWeightsData conv3;
    core::TensorValue conv_out_weight;
    std::vector<AudioLayerWeights> layers;
    core::TensorValue ln_post_weight;
    core::TensorValue ln_post_bias;
    LinearWeightsData proj1;
    LinearWeightsData proj2;
    core::TensorValue positional_embedding;
};

namespace qwen3_audio_encoder_detail {

Conv2dWeightsData load_conv2d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    std::initializer_list<int64_t> weight_shape,
    int64_t out_channels) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, weight_shape),
        store.load_f32_tensor(source, prefix + ".bias", {out_channels}),
    };
}

std::shared_ptr<const Qwen3AudioEncoderWeights> load_weights(
    const assets::TensorSource & source,
    const Qwen3AudioEncoderConfig & config,
    const Qwen3AudioEncoderWeightBinding & binding,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type,
    const std::string & trace_name) {
    const auto & audio_prefix = binding.tower_prefix;
    const auto & projector_prefix = binding.projector_prefix;
    auto weights = std::make_shared<Qwen3AudioEncoderWeights>();
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        trace_name + ".weights",
        kDefaultAudioWeightContextBytes);
    weights->store = store;
    weights->conv1 = load_conv2d(
        *store,
        source,
        audio_prefix + ".conv2d1",
        storage_type,
        {config.downsample_hidden_size, 1, 3, 3},
        config.downsample_hidden_size);
    weights->conv2 = load_conv2d(
        *store,
        source,
        audio_prefix + ".conv2d2",
        storage_type,
        {config.downsample_hidden_size, config.downsample_hidden_size, 3, 3},
        config.downsample_hidden_size);
    weights->conv3 = load_conv2d(
        *store,
        source,
        audio_prefix + ".conv2d3",
        storage_type,
        {config.downsample_hidden_size, config.downsample_hidden_size, 3, 3},
        config.downsample_hidden_size);
    const int64_t conv_freq = (((config.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
    weights->conv_out_weight = store->load_tensor(
        source,
        audio_prefix + ".conv_out.weight",
        storage_type,
        {config.d_model, config.downsample_hidden_size * conv_freq});
    weights->layers.reserve(static_cast<size_t>(config.encoder_layers));
    for (int64_t layer = 0; layer < config.encoder_layers; ++layer) {
        const std::string prefix = audio_prefix + ".layers." + std::to_string(layer);
        AudioLayerWeights w;
        w.self_attn_norm_weight = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.weight", {config.d_model});
        w.self_attn_norm_bias = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.bias", {config.d_model});
        w.q_proj_weight = store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.d_model, config.d_model});
        w.q_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.q_proj.bias", {config.d_model});
        w.k_proj_weight = store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.d_model, config.d_model});
        w.k_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.k_proj.bias", {config.d_model});
        w.v_proj_weight = store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.d_model, config.d_model});
        w.v_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.v_proj.bias", {config.d_model});
        w.out_proj_weight = store->load_tensor(source, prefix + ".self_attn.out_proj.weight", storage_type, {config.d_model, config.d_model});
        w.out_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.out_proj.bias", {config.d_model});
        w.final_norm_weight = store->load_f32_tensor(source, prefix + ".final_layer_norm.weight", {config.d_model});
        w.final_norm_bias = store->load_f32_tensor(source, prefix + ".final_layer_norm.bias", {config.d_model});
        w.fc1_weight = store->load_tensor(source, prefix + ".fc1.weight", storage_type, {config.encoder_ffn_dim, config.d_model});
        w.fc1_bias = store->load_f32_tensor(source, prefix + ".fc1.bias", {config.encoder_ffn_dim});
        w.fc2_weight = store->load_tensor(source, prefix + ".fc2.weight", storage_type, {config.d_model, config.encoder_ffn_dim});
        w.fc2_bias = store->load_f32_tensor(source, prefix + ".fc2.bias", {config.d_model});
        weights->layers.push_back(std::move(w));
    }
    weights->ln_post_weight = store->load_f32_tensor(source, audio_prefix + ".ln_post.weight", {config.d_model});
    weights->ln_post_bias = store->load_f32_tensor(source, audio_prefix + ".ln_post.bias", {config.d_model});
    const std::string proj1 = "." + binding.projection1;
    const std::string proj2 = "." + binding.projection2;
    weights->proj1 = {
        store->load_tensor(source, projector_prefix + proj1 + ".weight", storage_type, {config.d_model, config.d_model}),
        store->load_f32_tensor(source, projector_prefix + proj1 + ".bias", {config.d_model}),
    };
    weights->proj2 = {
        store->load_tensor(source, projector_prefix + proj2 + ".weight", storage_type, {config.output_dim, config.d_model}),
        store->load_f32_tensor(source, projector_prefix + proj2 + ".bias", {config.output_dim}),
    };
    weights->positional_embedding = store->make_f32(
        core::TensorShape::from_dims({config.max_source_positions, config.d_model}),
        sinusoidal_positions(config.max_source_positions, config.d_model));
    store->upload();
    return weights;
}

modules::TransformerEncoderBlockWeights bind_layer(const AudioLayerWeights & weights) {
    modules::TransformerEncoderBlockWeights block;
    block.norm1 = {weights.self_attn_norm_weight, weights.self_attn_norm_bias};
    block.self_attention.q_weight = weights.q_proj_weight;
    block.self_attention.q_bias = weights.q_proj_bias;
    block.self_attention.k_weight = weights.k_proj_weight;
    block.self_attention.k_bias = weights.k_proj_bias;
    block.self_attention.v_weight = weights.v_proj_weight;
    block.self_attention.v_bias = weights.v_proj_bias;
    block.self_attention.qkv_weight = std::nullopt;
    block.self_attention.qkv_bias = std::nullopt;
    block.self_attention.out_weight = weights.out_proj_weight;
    block.self_attention.out_bias = weights.out_proj_bias;
    block.layer_scale1 = std::nullopt;
    block.norm2 = {weights.final_norm_weight, weights.final_norm_bias};
    block.feed_forward.fc1_weight = weights.fc1_weight;
    block.feed_forward.fc1_bias = weights.fc1_bias;
    block.feed_forward.fc2_weight = weights.fc2_weight;
    block.feed_forward.fc2_bias = weights.fc2_bias;
    block.layer_scale2 = std::nullopt;
    return block;
}

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

}  // namespace qwen3_audio_encoder_detail

class Qwen3AudioEncoderGraph {
public:
    Qwen3AudioEncoderGraph(
        Qwen3AudioEncoderConfig encoder_config,
        std::shared_ptr<const Qwen3AudioEncoderWeights> weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t frames,
        std::string trace_name)
        : config_(std::move(encoder_config)),
          weights_(std::move(weights)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          frames_(frames),
          trace_name_(std::move(trace_name)) {
        if (weights_ == nullptr) {
            throw std::runtime_error("Qwen3 audio encoder graph requires weights");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 audio encoder backend is not initialized");
        }
        if (frames_ <= 0) {
            throw std::runtime_error("Qwen3 audio encoder graph requires positive frame count");
        }
        const auto build_start = Clock::now();
        const auto & config = config_;
        chunk_frame_limit_ = config.n_window * 2;
        chunk_lengths_ = audio_chunk_lengths(frames_, chunk_frame_limit_);
        chunk_frames_ = max_value(chunk_lengths_);
        chunk_count_ = static_cast<int64_t>(chunk_lengths_.size());
        chunk_token_lengths_.reserve(chunk_lengths_.size());
        for (const int64_t chunk_length : chunk_lengths_) {
            chunk_token_lengths_.push_back(qwen3_audio_encoder_token_count(chunk_length));
        }
        output_tokens_ = sum_values(chunk_token_lengths_);
        const int64_t max_chunk_tokens = max_value(chunk_token_lengths_);
        attention_window_tokens_ = max_chunk_tokens * (config.n_window_infer / chunk_frame_limit_);
        if (output_tokens_ > config.max_source_positions) {
            throw std::runtime_error("Qwen3 audio encoder token count exceeds max_source_positions");
        }
        attention_window_lengths_ = audio_attention_window_lengths(output_tokens_, attention_window_tokens_);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 audio encoder graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), trace_name_.c_str(), backend_type_};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({chunk_count_, config.num_mel_bins, chunk_frames_}));
        input_ = input.tensor;
        auto x = core::reshape_tensor(
            ctx,
            input,
            core::TensorShape::from_dims({chunk_count_, 1, config.num_mel_bins, chunk_frames_}));
        x = modules::Conv2dModule({1, config.downsample_hidden_size, 3, 3, 2, 2, 1, 1, 1, 1, true})
                .build(ctx, x, {weights_->conv1.weight, weights_->conv1.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::Conv2dModule(
                {config.downsample_hidden_size, config.downsample_hidden_size, 3, 3, 2, 2, 1, 1, 1, 1, true})
                .build(ctx, x, {weights_->conv2.weight, weights_->conv2.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::Conv2dModule(
                {config.downsample_hidden_size, config.downsample_hidden_size, 3, 3, 2, 2, 1, 1, 1, 1, true})
                .build(ctx, x, {weights_->conv3.weight, weights_->conv3.bias});
        x = modules::GeluModule().build(ctx, x);
        // Keep the ggml layout aligned with the Python conv output before flattening [B, T, C, F].
        const auto transposed_shape = core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[3], x.shape.dims[1], x.shape.dims[2]});
        x = core::wrap_tensor(ggml_permute(ctx.ggml, x.tensor, 2, 0, 1, 3), transposed_shape, x.type);
        const auto transposed_dims = core::to_ggml_dims(x.shape);
        x = core::wrap_tensor(
            ggml_cont_4d(
                ctx.ggml,
                x.tensor,
                transposed_dims[0],
                transposed_dims[1],
                transposed_dims[2],
                transposed_dims[3]),
            x.shape,
            x.type);
        x = core::reshape_tensor(
            ctx,
            x,
            core::TensorShape::from_dims({chunk_count_, x.shape.dims[1], x.shape.dims[2] * x.shape.dims[3]}));
        x = modules::LinearModule({x.shape.last_dim(), config.d_model, false}).build(
            ctx,
            x,
            {weights_->conv_out_weight, std::nullopt});
        const int64_t tokens = x.shape.dims[1];
        auto pos = weights_->positional_embedding;
        pos = modules::SliceModule({0, 0, tokens}).build(ctx, pos);
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, tokens, config.d_model}));
        if (chunk_count_ > 1) {
            pos = modules::RepeatModule({core::TensorShape::from_dims({chunk_count_, tokens, config.d_model})})
                      .build(ctx, pos);
        }
        x = modules::AddModule().build(ctx, x, pos);
        core::TensorValue compacted;
        for (int64_t chunk = 0; chunk < chunk_count_; ++chunk) {
            auto chunk_value = modules::SliceModule({0, chunk, 1}).build(ctx, x);
            const int64_t valid_tokens = chunk_token_lengths_[static_cast<size_t>(chunk)];
            if (valid_tokens < tokens) {
                chunk_value = modules::SliceModule({1, 0, valid_tokens}).build(ctx, chunk_value);
            }
            compacted = compacted.valid()
                ? modules::ConcatModule({1}).build(ctx, compacted, chunk_value)
                : chunk_value;
        }
        x = compacted;
        attention_mask_values_ = audio_attention_mask(output_tokens_, attention_window_lengths_);
        auto attention_mask = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, output_tokens_, output_tokens_}));
        attention_mask_ = attention_mask.tensor;
        for (const auto & layer : weights_->layers) {
            x = audio_encoder_layer(ctx, x, bind_layer(layer), config, attention_mask);
        }
        x = modules::LayerNormModule({config.d_model, 1.0e-5F, true, true})
                .build(ctx, x, {weights_->ln_post_weight, weights_->ln_post_bias});
        x = modules::LinearModule({config.d_model, config.d_model, true}).build(
            ctx,
            x,
            {weights_->proj1.weight, weights_->proj1.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::LinearModule({config.d_model, config.output_dim, true}).build(
            ctx,
            x,
            {weights_->proj2.weight, weights_->proj2.bias});
        output_ = x.tensor;
        output_tokens_ = x.shape.dims[1];
        output_dim_ = x.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        // unique_ptr so a throw below frees the partial reservation (a
        // throwing constructor runs no destructor)
        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
            return gallocr_ != nullptr && ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        if (!try_alloc() &&
            (engine::core::trim_backend_pools(backend_), !try_alloc())) {
            throw std::runtime_error("failed to allocate Qwen3 audio encoder graph");
        }
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(float));
        debug::timing_log_scalar(trace_name_ + ".graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar(trace_name_ + ".frames", frames_);
    }

    ~Qwen3AudioEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_, true);
    }

    bool matches(const Qwen3AudioEncoderWeights & weights, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && frames_ == frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    Qwen3AudioEmbeddings run(const Qwen3AudioFeatures & features) {
        const auto & config = config_;
        if (features.mel_bins != config.num_mel_bins || features.frames != frames_) {
            throw std::runtime_error("Qwen3 audio encoder feature shape mismatch");
        }
        if (static_cast<int64_t>(features.values.size()) != config.num_mel_bins * frames_) {
            throw std::runtime_error("Qwen3 audio encoder feature value count mismatch");
        }
        std::vector<float> padded_features(static_cast<size_t>(chunk_count_ * config.num_mel_bins * chunk_frames_), 0.0F);
        int64_t source_frame = 0;
        for (int64_t chunk = 0; chunk < chunk_count_; ++chunk) {
            const int64_t chunk_length = chunk_lengths_[static_cast<size_t>(chunk)];
            for (int64_t mel = 0; mel < config.num_mel_bins; ++mel) {
                const size_t dst = static_cast<size_t>((chunk * config.num_mel_bins + mel) * chunk_frames_);
                const size_t src = static_cast<size_t>(mel * frames_ + source_frame);
                std::copy_n(
                    features.values.begin() + static_cast<std::ptrdiff_t>(src),
                    static_cast<size_t>(chunk_length),
                    padded_features.begin() + static_cast<std::ptrdiff_t>(dst));
            }
            source_frame += chunk_length;
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, padded_features.data(), 0, padded_features.size() * sizeof(float));
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(float));
        debug::timing_log_scalar(trace_name_ + ".input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(backend_, compute_threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar(trace_name_ + ".graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 audio encoder graph compute failed");
        }
        Qwen3AudioEmbeddings out;
        out.tokens = output_tokens_;
        out.hidden_size = output_dim_;
        out.values.resize(static_cast<size_t>(out.tokens * out.hidden_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        debug::timing_log_scalar(trace_name_ + ".output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    Qwen3AudioEncoderConfig config_;
    std::shared_ptr<const Qwen3AudioEncoderWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t frames_ = 0;
    std::string trace_name_;
    int64_t chunk_frame_limit_ = 0;
    int64_t chunk_frames_ = 0;
    int64_t chunk_count_ = 0;
    std::vector<int64_t> chunk_lengths_;
    std::vector<int64_t> chunk_token_lengths_;
    std::vector<int64_t> attention_window_lengths_;
    std::vector<float> attention_mask_values_;
    int64_t attention_window_tokens_ = 0;
    int64_t output_tokens_ = 0;
    int64_t output_dim_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

Qwen3AudioEncoderRuntime::Qwen3AudioEncoderRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    Qwen3AudioEncoderConfig config,
    core::ExecutionContext & execution,
    Qwen3AudioEncoderRuntimeOptions options,
    Qwen3AudioEncoderWeightBinding binding)
    : config_(std::move(config)),
      execution_(&execution),
      options_(std::move(options)) {
    if (source == nullptr) {
        throw std::runtime_error("Qwen3 audio encoder requires a tensor source");
    }
    if (options_.graph_arena_bytes == 0) {
        throw std::runtime_error("Qwen3 audio encoder graph arena must be non-zero");
    }
    if (config_.activation_function != "gelu") {
        throw std::runtime_error("Qwen3 audio encoder requires GELU activation");
    }
    weights_ = load_weights(*source, config_, binding,
        execution.backend(), execution.backend_type(), options_.weight_storage_type, options_.trace_name);
}

Qwen3AudioEncoderRuntime::~Qwen3AudioEncoderRuntime() = default;

Qwen3AudioEmbeddings Qwen3AudioEncoderRuntime::encode(const Qwen3AudioFeatures & features) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Qwen3 audio encoder execution context is null");
    }
    if (features.encoder_tokens != qwen3_audio_encoder_token_count(features.frames)) {
        throw std::runtime_error("Qwen3 audio encoder token count mismatch");
    }
    const int threads = std::max(1, execution_->config().threads);
    if (graph_ == nullptr || !graph_->matches(*weights_, features.frames, execution_->backend(), threads)) {
        graph_.reset();
        graph_ = std::make_unique<Qwen3AudioEncoderGraph>(
            config_,
            weights_,
            *execution_,
            options_.graph_arena_bytes,
            features.frames, options_.trace_name);
    } else {
        debug::timing_log_scalar(options_.trace_name + ".graph.build_ms", 0.0);
        debug::trace_log_scalar(options_.trace_name + ".frames", features.frames);
    }
    auto out = graph_->run(features);
    if (out.tokens != features.encoder_tokens) {
        throw std::runtime_error("Qwen3 audio encoder output token count mismatch");
    }
    return out;
}

}  // namespace engine::modules

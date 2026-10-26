#include "engine/models/cosyvoice3/flow.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/projected_grouped_self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/cosyvoice3/rknn_flow.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::cosyvoice3 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

constexpr float kPi = 3.14159265358979323846F;
constexpr float kInferenceCfgRate = 0.7F;
constexpr int64_t kTimeEmbeddingSize = 256;
constexpr int64_t kConvPosGroups = 16;
constexpr int64_t kConvPosKernel = 31;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GraphMemory {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    ~GraphMemory() {
        reset(nullptr);
    }

    void reset(ggml_backend_t backend) {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        gallocr.reset();
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        input_ctx.reset();
        ctx.reset();
    }
};

std::vector<int32_t> position_ids(int64_t steps) {
    std::vector<int32_t> out(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

std::vector<float> cosine_time_schedule(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("CosyVoice3 num_inference_steps must be positive");
    }
    std::vector<float> out(static_cast<size_t>(steps + 1));
    for (int64_t i = 0; i <= steps; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(steps);
        out[static_cast<size_t>(i)] = 1.0F - std::cos(u * 0.5F * kPi);
    }
    return out;
}

std::vector<float> timestep_embedding(float timestep) {
    const int64_t half = kTimeEmbeddingSize / 2;
    const float step = std::log(10000.0F) / static_cast<float>(half - 1);
    std::vector<float> out(static_cast<size_t>(kTimeEmbeddingSize));
    for (int64_t i = 0; i < half; ++i) {
        const float freq = std::exp(static_cast<float>(i) * -step);
        const float arg = 1000.0F * timestep * freq;
        out[static_cast<size_t>(i)] = std::sin(arg);
        out[static_cast<size_t>(half + i)] = std::cos(arg);
    }
    return out;
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    return modules::RepeatModule({like.shape}).build(ctx, value);
}

core::TensorValue mul_broadcast(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & scale) {
    return modules::MulModule().build(ctx, x, repeat_like(ctx, scale, x));
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & shift,
    const core::TensorValue & scale) {
    auto one_plus = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, repeat_like(ctx, scale, x).tensor, 1.0F, 1.0F),
        x.shape,
        GGML_TYPE_F32);
    auto shifted = modules::AddModule().build(ctx, modules::MulModule().build(ctx, x, one_plus), repeat_like(ctx, shift, x));
    return shifted;
}

core::TensorValue mish(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    auto input = core::ensure_backend_addressable_layout(ctx, x);
    auto softplus = core::wrap_tensor(ggml_softplus(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    auto t = core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), input.shape, GGML_TYPE_F32);
    return modules::MulModule().build(ctx, input, t);
}

core::TensorValue grouped_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t groups) {
    const int64_t channels = input.shape.dims[1];
    const int64_t out_channels = weights.weight.shape.dims[0];
    const int64_t kernel = weights.weight.shape.dims[2];
    if (groups <= 0 || channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error("CosyVoice3 grouped Conv1d channel/group mismatch");
    }
    const int64_t channels_per_group = channels / groups;
    const int64_t out_channels_per_group = out_channels / groups;
    core::TensorValue output;
    const auto input_contiguous = core::ensure_backend_addressable_layout(ctx, input);
    for (int64_t group = 0; group < groups; ++group) {
        auto input_group = modules::SliceModule({1, group * channels_per_group, channels_per_group}).build(ctx, input_contiguous);
        auto weight_group = modules::SliceModule({0, group * out_channels_per_group, out_channels_per_group}).build(ctx, weights.weight);
        modules::Conv1dWeights group_weights{weight_group, std::nullopt};
        if (weights.bias.has_value()) {
            group_weights.bias =
                modules::SliceModule({0, group * out_channels_per_group, out_channels_per_group}).build(ctx, *weights.bias);
        }
        auto group_out = modules::Conv1dModule({
            channels_per_group,
            out_channels_per_group,
            kernel,
            1,
            0,
            1,
            weights.bias.has_value()}).build(ctx, input_group, group_weights);
        output = output.valid() ? modules::ConcatModule({1}).build(ctx, output, group_out) : group_out;
    }
    return output;
}

struct CosyDiTBlockWeights {
    modules::LinearWeights attn_norm;
    modules::ProjectedGroupedSelfAttentionWeights attention;
    modules::NormWeights ff_norm;
    modules::FeedForwardWeights ff;
};

struct CosyDiTFlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    modules::LinearWeights speaker_projection;
    modules::Conv1dWeights pre_lookahead_conv1;
    modules::Conv1dWeights pre_lookahead_conv2;
    modules::LinearWeights input_projection;
    modules::Conv1dWeights conv_pos_1;
    modules::Conv1dWeights conv_pos_2;
    modules::LinearWeights time_fc1;
    modules::LinearWeights time_fc2;
    std::vector<CosyDiTBlockWeights> blocks;
    modules::LinearWeights final_norm;
    modules::LinearWeights output_projection;
    core::TensorValue rand_noise;
    std::vector<float> rand_noise_host;
};

modules::ProjectedGroupedSelfAttentionWeights load_attention(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    engine::assets::TensorStorageType storage_type) {
    modules::ProjectedGroupedSelfAttentionWeights out;
    out.q_proj = binding::linear_from_source(store, source, prefix + ".to_q", storage_type, hidden, hidden, true);
    out.k_proj = binding::linear_from_source(store, source, prefix + ".to_k", storage_type, hidden, hidden, true);
    out.v_proj = binding::linear_from_source(store, source, prefix + ".to_v", storage_type, hidden, hidden, true);
    out.o_proj = binding::linear_from_source(store, source, prefix + ".to_out.0", storage_type, hidden, hidden, true);
    return out;
}

CosyDiTBlockWeights load_block(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    const CosyVoice3Config & config,
    engine::assets::TensorStorageType storage_type) {
    CosyDiTBlockWeights out;
    out.attn_norm = binding::linear_from_source(
        store, source, prefix + ".attn_norm.linear", storage_type, 6 * config.flow_hidden_size, config.flow_hidden_size, true);
    out.attention = load_attention(store, source, prefix + ".attn", config.flow_hidden_size, storage_type);
    out.ff.fc1_weight = store.load_tensor(
        source,
        prefix + ".ff.ff.0.0.weight",
        storage_type,
        {config.flow_hidden_size * config.flow_ff_mult, config.flow_hidden_size});
    out.ff.fc1_bias = store.load_f32_tensor(source, prefix + ".ff.ff.0.0.bias", {config.flow_hidden_size * config.flow_ff_mult});
    out.ff.fc2_weight = store.load_tensor(
        source,
        prefix + ".ff.ff.2.weight",
        storage_type,
        {config.flow_hidden_size, config.flow_hidden_size * config.flow_ff_mult});
    out.ff.fc2_bias = store.load_f32_tensor(source, prefix + ".ff.ff.2.bias", {config.flow_hidden_size});
    return out;
}

std::shared_ptr<CosyDiTFlowWeights> load_flow_weights(
    const CosyVoice3Assets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<CosyDiTFlowWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "cosyvoice3.flow.weights",
        weight_context_bytes);
    const auto & source = *assets.flow_weights;
    const auto & c = assets.config;
    weights->token_embedding = weights->store->load_tensor(source, "input_embedding.weight", storage_type, {c.speech_token_size, c.flow_mel_channels});
    weights->speaker_projection = binding::linear_from_source(
        *weights->store, source, "spk_embed_affine_layer", storage_type, c.flow_mel_channels, c.speaker_dim, true);
    weights->pre_lookahead_conv1 = binding::conv1d_from_source(
        *weights->store, source, "pre_lookahead_layer.conv1", storage_type, c.flow_hidden_size, c.flow_mel_channels, c.pre_lookahead_len + 1, true);
    weights->pre_lookahead_conv2 = binding::conv1d_from_source(
        *weights->store, source, "pre_lookahead_layer.conv2", storage_type, c.flow_mel_channels, c.flow_hidden_size, 3, true);
    // The ggml DiT weights are loaded in both paths, deliberately: the RKNN ladder covers a fixed
    // set of lengths, and an utterance longer than the largest engine runs the ggml graph, so the
    // copy is not spare memory. Skipping it would trade a fallback that works for one that throws
    // `weight is invalid` the first time the ladder runs out.
    weights->input_projection = binding::linear_from_source(
        *weights->store, source, "decoder.estimator.input_embed.proj", storage_type, c.flow_hidden_size, 320, true);
    weights->conv_pos_1.weight = weights->store->load_tensor(
        source,
        "decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight",
        storage_type,
        {c.flow_hidden_size, c.flow_hidden_size / kConvPosGroups, kConvPosKernel});
    weights->conv_pos_1.bias = weights->store->load_f32_tensor(
        source,
        "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias",
        {c.flow_hidden_size});
    weights->conv_pos_2.weight = weights->store->load_tensor(
        source,
        "decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight",
        storage_type,
        {c.flow_hidden_size, c.flow_hidden_size / kConvPosGroups, kConvPosKernel});
    weights->conv_pos_2.bias = weights->store->load_f32_tensor(
        source,
        "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias",
        {c.flow_hidden_size});
    weights->time_fc1 = binding::linear_from_source(
        *weights->store, source, "decoder.estimator.time_embed.time_mlp.0", storage_type, c.flow_hidden_size, kTimeEmbeddingSize, true);
    weights->time_fc2 = binding::linear_from_source(
        *weights->store, source, "decoder.estimator.time_embed.time_mlp.2", storage_type, c.flow_hidden_size, c.flow_hidden_size, true);
    weights->blocks.reserve(static_cast<size_t>(c.flow_layers));
    for (int64_t layer = 0; layer < c.flow_layers; ++layer) {
        weights->blocks.push_back(load_block(
            *weights->store,
            source,
            "decoder.estimator.transformer_blocks." + std::to_string(layer),
            c,
            storage_type));
    }
    weights->final_norm = binding::linear_from_source(
        *weights->store, source, "decoder.estimator.norm_out.linear", storage_type, 2 * c.flow_hidden_size, c.flow_hidden_size, true);
    weights->output_projection = binding::linear_from_source(
        *weights->store, source, "decoder.estimator.proj_out", storage_type, c.flow_mel_channels, c.flow_hidden_size, true);
    weights->rand_noise = weights->store->load_tensor(source, "decoder.rand_noise", engine::assets::TensorStorageType::F32, {1, c.flow_mel_channels, 50 * 300});
    weights->rand_noise_host = source.require_f32("decoder.rand_noise", {1, c.flow_mel_channels, 50 * 300});
    weights->store->upload();
    return weights;
}

modules::ProjectedGroupedSelfAttentionConfig attention_config(const CosyVoice3Config & config) {
    modules::ProjectedGroupedSelfAttentionConfig out;
    out.hidden_size = config.flow_hidden_size;
    out.attention_heads = config.flow_heads;
    out.kv_heads = config.flow_heads;
    out.head_dim = config.flow_head_dim;
    out.use_bias = true;
    out.use_rope = true;
    out.apply_rope_to_projected_prefix = true;
    out.rope_type = GGML_ROPE_TYPE_NORMAL;
    out.rope_theta = 10000.0F;
    out.local_rope_theta = 10000.0F;
    out.causality = modules::AttentionCausality::NonCausal;
    out.lowering = modules::GroupedQueryAttentionLowering::FlashGroupedViewKV;
    out.attention_precision = GGML_PREC_F32;
    return out;
}

core::TensorValue causal_conv_pos_embed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CosyDiTFlowWeights & weights) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, input.shape.rank}).build(ctx, input);
    x = core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, x.tensor, kConvPosKernel - 1, 0, 0, 0, 0, 0, 0, 0),
        core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], x.shape.dims[2] + kConvPosKernel - 1}),
        GGML_TYPE_F32);
    x = grouped_conv1d(ctx, x, weights.conv_pos_1, kConvPosGroups);
    x = mish(ctx, x);
    x = core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, x.tensor, kConvPosKernel - 1, 0, 0, 0, 0, 0, 0, 0),
        core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], x.shape.dims[2] + kConvPosKernel - 1}),
        GGML_TYPE_F32);
    x = grouped_conv1d(ctx, x, weights.conv_pos_2, kConvPosGroups);
    x = mish(ctx, x);
    x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    return x;
}

core::TensorValue dit_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & time,
    const core::TensorValue & positions,
    const CosyDiTBlockWeights & weights,
    const CosyVoice3Config & config,
    int64_t layer) {
    auto ada = modules::SiluModule().build(ctx, time);
    ada = modules::LinearModule({config.flow_hidden_size, 6 * config.flow_hidden_size, true}).build(ctx, ada, weights.attn_norm);
    auto shift_msa = modules::SliceModule({2, 0 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);
    auto scale_msa = modules::SliceModule({2, 1 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);
    auto gate_msa = modules::SliceModule({2, 2 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);
    auto shift_mlp = modules::SliceModule({2, 3 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);
    auto scale_mlp = modules::SliceModule({2, 4 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);
    auto gate_mlp = modules::SliceModule({2, 5 * config.flow_hidden_size, config.flow_hidden_size}).build(ctx, ada);

    auto x = modules::LayerNormModule({config.flow_hidden_size, 1.0e-6F, false, false}).build(ctx, input, {});
    x = modulate(ctx, x, shift_msa, scale_msa);
    x = modules::ProjectedGroupedSelfAttentionModule(attention_config(config)).build(ctx, x, positions, weights.attention, layer);
    x = mul_broadcast(ctx, x, gate_msa);
    auto out = modules::AddModule().build(ctx, input, x);

    x = modules::LayerNormModule({config.flow_hidden_size, 1.0e-6F, false, false}).build(ctx, out, {});
    x = modulate(ctx, x, shift_mlp, scale_mlp);
    x = modules::FeedForwardModule({
        config.flow_hidden_size,
        config.flow_hidden_size * config.flow_ff_mult,
        true,
        modules::GeluApproximation::Tanh,
    }).build(ctx, x, weights.ff);
    x = mul_broadcast(ctx, x, gate_mlp);
    return modules::AddModule().build(ctx, out, x);
}

class ConditionGraph {
public:
    ConditionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const CosyDiTFlowWeights> weights,
        CosyVoice3Config config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~ConditionGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<int32_t> & prompt_tokens, const std::vector<int32_t> & target_tokens) {
        const int64_t tokens = static_cast<int64_t>(prompt_tokens.size() + target_tokens.size());
        if (tokens <= 0) {
            throw std::runtime_error("CosyVoice3 flow requires speech tokens");
        }
        ensure(tokens);
        std::vector<int32_t> all_tokens;
        all_tokens.reserve(static_cast<size_t>(tokens));
        all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
        all_tokens.insert(all_tokens.end(), target_tokens.begin(), target_tokens.end());
        ggml_backend_tensor_set(token_ids_, all_tokens.data(), 0, all_tokens.size() * sizeof(int32_t));
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "cosyvoice3.flow.condition") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CosyVoice3 flow condition graph compute failed");
        }
        auto output = core::read_tensor_f32(output_.tensor);
        return output;
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        tokens_ = 0;
        token_ids_ = nullptr;
        output_ = {};
    }

private:
    void ensure(int64_t tokens) {
        if (mem_.graph != nullptr && tokens_ == tokens) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "cosyvoice3.flow.condition", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "cosyvoice3.flow.condition.inputs", execution_.backend_type()};

        auto ids = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens}));
        token_ids_ = ids.tensor;
        ggml_set_input(token_ids_);
        auto h = modules::EmbeddingModule({config_.speech_token_size, config_.flow_mel_channels})
                     .build(ctx, ids, weights_->token_embedding);
        const auto residual = h;
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        if (execution_.backend_type() == core::BackendType::Metal) {
            h = core::ensure_backend_addressable_layout(ctx, h);
        }
        h = core::wrap_tensor(
            ggml_pad_ext(ctx.ggml, h.tensor, 0, config_.pre_lookahead_len, 0, 0, 0, 0, 0, 0),
            core::TensorShape::from_dims({1, config_.flow_mel_channels, tokens + config_.pre_lookahead_len}),
            GGML_TYPE_F32);
        h = modules::Conv1dModule({config_.flow_mel_channels, config_.flow_hidden_size, config_.pre_lookahead_len + 1, 1, 0, 1, true})
                .build(ctx, h, weights_->pre_lookahead_conv1);
        h = modules::LeakyReluModule().build(ctx, h);
        h = core::wrap_tensor(
            ggml_pad_ext(ctx.ggml, h.tensor, 2, 0, 0, 0, 0, 0, 0, 0),
            core::TensorShape::from_dims({1, config_.flow_hidden_size, tokens + 2}),
            GGML_TYPE_F32);
        h = modules::Conv1dModule({config_.flow_hidden_size, config_.flow_mel_channels, 3, 1, 0, 1, true})
                .build(ctx, h, weights_->pre_lookahead_conv2);
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        h = modules::AddModule().build(ctx, h, residual);
        h = modules::Interpolate1dModule({tokens * config_.token_mel_ratio, modules::Interpolate1dMode::Nearest})
                .build(ctx, modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h));
        h = modules::TransposeModule({{0, 2, 1, 3}, h.shape.rank}).build(ctx, h);
        output_ = core::ensure_backend_addressable_layout(ctx, h);
        ggml_set_output(output_.tensor);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 20000, false);
        ggml_build_forward_expand(mem_.graph, output_.tensor);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate CosyVoice3 flow condition graph");
        }
        tokens_ = tokens;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const CosyDiTFlowWeights> weights_;
    CosyVoice3Config config_;
    size_t graph_arena_bytes_ = 0;
    GraphMemory mem_;
    int64_t tokens_ = 0;
    ggml_tensor * token_ids_ = nullptr;
    core::TensorValue output_;
};

class DiTGraph {
public:
    DiTGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const CosyDiTFlowWeights> weights,
        CosyVoice3Config config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~DiTGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(
        const std::vector<float> & x,
        const std::vector<float> & mu,
        const std::vector<float> & cond,
        const std::vector<float> & speaker,
        float t,
        int64_t frames) {
        if (frames <= 0) {
            throw std::runtime_error("CosyVoice3 DiT frames must be positive");
        }
        const int64_t batch = 2;
        const int64_t channels = config_.flow_mel_channels;
        if (static_cast<int64_t>(x.size()) != batch * channels * frames ||
            static_cast<int64_t>(mu.size()) != batch * channels * frames ||
            static_cast<int64_t>(cond.size()) != batch * channels * frames ||
            static_cast<int64_t>(speaker.size()) != batch * channels) {
            throw std::runtime_error("CosyVoice3 DiT input size mismatch");
        }
        // The graph takes the timestep embedding, not the timestep: the sinusoidal expansion is
        // host-side here and inside the graph in the RKNN engine. Both backends get `t`.
        const auto temb = timestep_embedding(t);
        std::vector<float> time_embedding(static_cast<size_t>(batch) * kTimeEmbeddingSize);
        std::copy(temb.begin(), temb.end(), time_embedding.begin());
        std::copy(temb.begin(), temb.end(), time_embedding.begin() + kTimeEmbeddingSize);
        ensure(frames);
        ggml_backend_tensor_set(x_, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(mu_, mu.data(), 0, mu.size() * sizeof(float));
        ggml_backend_tensor_set(cond_, cond.data(), 0, cond.size() * sizeof(float));
        ggml_backend_tensor_set(spks_, speaker.data(), 0, speaker.size() * sizeof(float));
        ggml_backend_tensor_set(time_, time_embedding.data(), 0, time_embedding.size() * sizeof(float));
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, "cosyvoice3.flow.dit") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CosyVoice3 DiT graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        frames_ = 0;
        x_ = nullptr;
        mu_ = nullptr;
        cond_ = nullptr;
        spks_ = nullptr;
        time_ = nullptr;
        positions_ = nullptr;
        output_ = {};
    }

private:
    void ensure(int64_t frames) {
        if (mem_.graph != nullptr && frames_ == frames) {
            return;
        }
        mem_.reset(execution_.backend());
        constexpr int64_t batch = 2;
        const int64_t channels = config_.flow_mel_channels;
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        core::ModuleBuildContext ctx{mem_.ctx.get(), "cosyvoice3.flow.dit", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), "cosyvoice3.flow.dit.inputs", execution_.backend_type()};

        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, channels, frames}));
        x_ = x.tensor;
        ggml_set_input(x_);
        auto mu = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, channels, frames}));
        mu_ = mu.tensor;
        ggml_set_input(mu_);
        auto cond = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, channels, frames}));
        cond_ = cond.tensor;
        ggml_set_input(cond_);
        auto spks = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, channels}));
        spks_ = spks.tensor;
        ggml_set_input(spks_);
        auto time = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, kTimeEmbeddingSize}));
        time_ = time.tensor;
        ggml_set_input(time_);
        auto positions = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        positions_ = positions.tensor;

        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        mu = modules::TransposeModule({{0, 2, 1, 3}, mu.shape.rank}).build(ctx, mu);
        cond = modules::TransposeModule({{0, 2, 1, 3}, cond.shape.rank}).build(ctx, cond);
        x = core::ensure_backend_addressable_layout(ctx, x);
        mu = core::ensure_backend_addressable_layout(ctx, mu);
        cond = core::ensure_backend_addressable_layout(ctx, cond);
        auto spks_btf = core::reshape_tensor(ctx, spks, core::TensorShape::from_dims({batch, 1, channels}));
        spks_btf = modules::RepeatModule({core::TensorShape::from_dims({batch, frames, channels})}).build(ctx, spks_btf);
        spks_btf = core::ensure_backend_addressable_layout(ctx, spks_btf);
        auto input = modules::ConcatModule({2}).build(ctx, x, cond);
        input = modules::ConcatModule({2}).build(ctx, input, mu);
        input = modules::ConcatModule({2}).build(ctx, input, spks_btf);
        input = core::ensure_backend_addressable_layout(ctx, input);
        input = modules::LinearModule({320, config_.flow_hidden_size, true}).build(ctx, input, weights_->input_projection);
        input = core::ensure_backend_addressable_layout(ctx, input);
        auto pos = causal_conv_pos_embed(ctx, input, *weights_);
        pos = core::ensure_backend_addressable_layout(ctx, pos);
        input = modules::AddModule().build(ctx, input, pos);
        input = core::ensure_backend_addressable_layout(ctx, input);

        time = modules::LinearModule({kTimeEmbeddingSize, config_.flow_hidden_size, true}).build(ctx, time, weights_->time_fc1);
        time = modules::SiluModule().build(ctx, time);
        time = modules::LinearModule({config_.flow_hidden_size, config_.flow_hidden_size, true}).build(ctx, time, weights_->time_fc2);
        time = core::reshape_tensor(ctx, time, core::TensorShape::from_dims({batch, 1, config_.flow_hidden_size}));

        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            input = dit_block(ctx, input, time, positions, weights_->blocks[layer], config_, static_cast<int64_t>(layer));
        }
        auto norm_mod = modules::SiluModule().build(ctx, time);
        norm_mod = modules::LinearModule({config_.flow_hidden_size, 2 * config_.flow_hidden_size, true}).build(ctx, norm_mod, weights_->final_norm);
        auto scale = modules::SliceModule({2, 0, config_.flow_hidden_size}).build(ctx, norm_mod);
        auto shift = modules::SliceModule({2, config_.flow_hidden_size, config_.flow_hidden_size}).build(ctx, norm_mod);
        input = modules::LayerNormModule({config_.flow_hidden_size, 1.0e-6F, false, false}).build(ctx, input, {});
        input = modulate(ctx, input, shift, scale);
        input = modules::LinearModule({config_.flow_hidden_size, channels, true}).build(ctx, input, weights_->output_projection);
        input = modules::TransposeModule({{0, 2, 1, 3}, input.shape.rank}).build(ctx, input);
        output_ = core::ensure_backend_addressable_layout(ctx, input);
        ggml_set_output(output_.tensor);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 200000, false);
        ggml_build_forward_expand(mem_.graph, output_.tensor);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate CosyVoice3 DiT graph");
        }
        const auto pos_ids = position_ids(frames);
        ggml_backend_tensor_set(positions_, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));
        frames_ = frames;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const CosyDiTFlowWeights> weights_;
    CosyVoice3Config config_;
    size_t graph_arena_bytes_ = 0;
    GraphMemory mem_;
    int64_t frames_ = 0;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * mu_ = nullptr;
    ggml_tensor * cond_ = nullptr;
    ggml_tensor * spks_ = nullptr;
    ggml_tensor * time_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    core::TensorValue output_;
};

std::vector<float> normalize_and_project_speaker(
    const CosyVoice3Config & config,
    const CosyDiTFlowWeights & weights,
    core::ExecutionContext & execution,
    const std::vector<float> & speaker_embedding,
    size_t graph_arena_bytes) {
    if (static_cast<int64_t>(speaker_embedding.size()) != config.speaker_dim) {
        throw std::runtime_error("CosyVoice3 speaker embedding size mismatch");
    }
    float norm_sq = 0.0F;
    for (float value : speaker_embedding) {
        norm_sq += value * value;
    }
    const float inv_norm = 1.0F / std::sqrt(std::max(norm_sq, 1.0e-12F));
    std::vector<float> normalized(speaker_embedding.size());
    for (size_t i = 0; i < speaker_embedding.size(); ++i) {
        normalized[i] = speaker_embedding[i] * inv_norm;
    }

    GraphMemory mem;
    ggml_init_params params{graph_arena_bytes, nullptr, true};
    mem.ctx.reset(ggml_init(params));
    ggml_init_params input_params{4ull * 1024ull * 1024ull, nullptr, true};
    mem.input_ctx.reset(ggml_init(input_params));
    core::ModuleBuildContext ctx{mem.ctx.get(), "cosyvoice3.flow.speaker", execution.backend_type()};
    core::ModuleBuildContext input_ctx{mem.input_ctx.get(), "cosyvoice3.flow.speaker.inputs", execution.backend_type()};
    auto input = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.speaker_dim}));
    ggml_set_input(input.tensor);
    auto output = modules::LinearModule({config.speaker_dim, config.flow_mel_channels, true}).build(ctx, input, weights.speaker_projection);
    output = core::ensure_backend_addressable_layout(ctx, output);
    ggml_set_output(output.tensor);
    mem.graph = ggml_new_graph_custom(mem.ctx.get(), 4096, false);
    ggml_build_forward_expand(mem.graph, output.tensor);
    mem.input_buffer = ggml_backend_alloc_ctx_tensors(mem.input_ctx.get(), execution.backend());
    mem.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    if (mem.input_buffer == nullptr || mem.gallocr == nullptr ||
        !ggml_gallocr_reserve(mem.gallocr.get(), mem.graph) ||
        !ggml_gallocr_alloc_graph(mem.gallocr.get(), mem.graph)) {
        throw std::runtime_error("CosyVoice3 speaker projection graph failed");
    }
    ggml_backend_tensor_set(input.tensor, normalized.data(), 0, normalized.size() * sizeof(float));
    if (core::compute_backend_graph(execution.backend(), mem.graph, nullptr, "cosyvoice3.flow.speaker") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("CosyVoice3 speaker projection graph failed");
    }
    auto out = core::read_tensor_f32(output.tensor);
    mem.reset(execution.backend());
    return out;
}

std::vector<float> read_noise_prefix(
    const CosyVoice3Config & config,
    const CosyDiTFlowWeights & weights,
    int64_t frames) {
    if (frames > 50 * 300) {
        throw std::runtime_error("CosyVoice3 requested mel frames exceed fixed flow noise capacity");
    }
    const auto & full = weights.rand_noise_host;
    std::vector<float> out(static_cast<size_t>(config.flow_mel_channels * frames));
    for (int64_t ch = 0; ch < config.flow_mel_channels; ++ch) {
        const auto src = full.begin() + static_cast<std::ptrdiff_t>(ch * 50 * 300);
        const auto dst = out.begin() + static_cast<std::ptrdiff_t>(ch * frames);
        std::copy(src, src + frames, dst);
    }
    return out;
}

}  // namespace

class CosyVoice3DiTFlowRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        const std::vector<std::string> & rknn_engine_paths)
        : assets_(std::move(assets)),
          execution_(execution),
          weights_(load_flow_weights(
              *assets_,
              execution_,
              weight_context_bytes,
              storage_type)),
          condition_(execution_, weights_, assets_->config, graph_arena_bytes),
          dit_(execution_, weights_, assets_->config, graph_arena_bytes),
          graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 flow runtime requires assets");
        }
        if (!rknn_engine_paths.empty()) {
            // The engines' own channel count is checked against the checkpoint when an engine
            // loads, so nothing to compare here: registering the ladder loads none of it.
            rknn_ = std::make_unique<CosyVoice3RknnFlow>(
                rknn_engine_paths, assets_->config.flow_mel_channels);
            debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.engines", rknn_->engine_count());
            debug::trace_log_scalar("cosyvoice3.flow.dit.rknn.max_frames", rknn_->max_frames());
        }
    }

    CosyVoice3FlowOutput generate(const CosyVoice3FlowRequest & request) {
        const auto start = std::chrono::steady_clock::now();
        const auto & c = assets_->config;
        if (request.speech_tokens.empty()) {
            throw std::runtime_error("CosyVoice3 flow requires generated speech tokens");
        }
        if (request.prompt_mel_frames <= 0 ||
            static_cast<int64_t>(request.prompt_mel.size()) != request.prompt_mel_frames * c.flow_mel_channels) {
            throw std::runtime_error("CosyVoice3 flow prompt mel size mismatch");
        }
        const auto cond_start = std::chrono::steady_clock::now();
        auto mu_btc = condition_.run(request.prompt_speech_tokens, request.speech_tokens);
        const int64_t total_frames = static_cast<int64_t>(mu_btc.size()) / c.flow_mel_channels;
        const int64_t target_frames = total_frames - request.prompt_mel_frames;
        if (target_frames <= 0) {
            throw std::runtime_error("CosyVoice3 flow target mel frames must be positive");
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.condition_ms", engine::debug::elapsed_ms(cond_start));

        std::vector<float> mu(static_cast<size_t>(2 * c.flow_mel_channels * total_frames), 0.0F);
        for (int64_t frame = 0; frame < total_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                mu[static_cast<size_t>(ch * total_frames + frame)] =
                    mu_btc[static_cast<size_t>(frame * c.flow_mel_channels + ch)];
            }
        }
        std::vector<float> cond(static_cast<size_t>(2 * c.flow_mel_channels * total_frames), 0.0F);
        for (int64_t frame = 0; frame < request.prompt_mel_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                cond[static_cast<size_t>(ch * total_frames + frame)] =
                    request.prompt_mel[static_cast<size_t>(frame * c.flow_mel_channels + ch)];
            }
        }
        auto x = read_noise_prefix(c, *weights_, total_frames);
        std::vector<float> x_batched(static_cast<size_t>(2 * c.flow_mel_channels * total_frames));
        std::copy(x.begin(), x.end(), x_batched.begin());
        std::copy(x.begin(), x.end(), x_batched.begin() + static_cast<std::ptrdiff_t>(x.size()));

        auto speaker_cond = normalize_and_project_speaker(
            c,
            *weights_,
            execution_,
            request.speaker_embedding,
            graph_arena_bytes_);
        std::vector<float> spks(static_cast<size_t>(2 * c.flow_mel_channels), 0.0F);
        std::copy(speaker_cond.begin(), speaker_cond.end(), spks.begin());

        const auto schedule = cosine_time_schedule(request.num_inference_steps);
        // The NPU ladder covers a fixed range of lengths; past its largest engine the ggml graph
        // is the only backend that has a shape for the utterance, so the fallback is per call.
        const bool on_npu = rknn_ != nullptr && total_frames <= rknn_->max_frames();
        if (rknn_ != nullptr && !on_npu) {
            debug::timing_log_scalar(
                "cosyvoice3.flow.dit.rknn.fallback_frames", static_cast<int64_t>(total_frames));
        }
        const auto dit_start = std::chrono::steady_clock::now();
        for (int64_t step = 1; step < static_cast<int64_t>(schedule.size()); ++step) {
            const float t = schedule[static_cast<size_t>(step - 1)];
            const float dt = schedule[static_cast<size_t>(step)] - t;
            auto pred = on_npu
                ? rknn_->run(x_batched.data(), mu.data(), cond.data(), spks.data(), t, total_frames)
                : dit_.run(x_batched, mu, cond, spks, t, total_frames);
            const size_t branch = static_cast<size_t>(c.flow_mel_channels * total_frames);
            for (size_t i = 0; i < branch; ++i) {
                const float guided = (1.0F + kInferenceCfgRate) * pred[i] - kInferenceCfgRate * pred[branch + i];
                x_batched[i] += dt * guided;
            }
            std::copy(x_batched.begin(), x_batched.begin() + static_cast<std::ptrdiff_t>(branch), x_batched.begin() + static_cast<std::ptrdiff_t>(branch));
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.dit_ms", engine::debug::elapsed_ms(dit_start));

        CosyVoice3FlowOutput out;
        out.frames = target_frames;
        out.mel.resize(static_cast<size_t>(target_frames * c.flow_mel_channels));
        for (int64_t frame = 0; frame < target_frames; ++frame) {
            for (int64_t ch = 0; ch < c.flow_mel_channels; ++ch) {
                out.mel[static_cast<size_t>(frame * c.flow_mel_channels + ch)] =
                    x_batched[static_cast<size_t>(ch * total_frames + request.prompt_mel_frames + frame)];
            }
        }
        engine::debug::timing_log_scalar("cosyvoice3.flow.total_ms", engine::debug::elapsed_ms(start));
        engine::debug::trace_log_scalar("cosyvoice3.flow.total_frames", static_cast<double>(total_frames));
        engine::debug::trace_log_scalar("cosyvoice3.flow.target_frames", static_cast<double>(target_frames));
        return out;
    }

    void release_graphs() {
        condition_.release_graph();
        dit_.release_graph();
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    core::ExecutionContext & execution_;
    std::shared_ptr<CosyDiTFlowWeights> weights_;
    ConditionGraph condition_;
    DiTGraph dit_;
    std::unique_ptr<CosyVoice3RknnFlow> rknn_;
    size_t graph_arena_bytes_ = 0;
};

CosyVoice3DiTFlowRuntime::CosyVoice3DiTFlowRuntime(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type,
    const std::vector<std::string> & rknn_engine_paths)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          rknn_engine_paths)) {}

CosyVoice3DiTFlowRuntime::~CosyVoice3DiTFlowRuntime() = default;

CosyVoice3FlowOutput CosyVoice3DiTFlowRuntime::generate(const CosyVoice3FlowRequest & request) {
    return impl_->generate(request);
}

void CosyVoice3DiTFlowRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3

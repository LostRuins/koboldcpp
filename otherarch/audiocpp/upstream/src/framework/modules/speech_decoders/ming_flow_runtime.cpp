#include "engine/framework/modules/speech_decoders/ming_flow_runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

class MingPatchNetwork {
public:
    MingPatchNetwork(const assets::TensorSource & source, const MingPatchNetworkConfig & config,
        const std::string & prefix, int64_t latent_dim, int64_t llm_dim, bool conditioned,
        core::ExecutionContext & execution)
        : execution_(execution), store_(execution.backend(), execution.backend_type(), prefix + ".weights", 4 * 1024 * 1024),
          conditioned_(conditioned), latent_dim_(latent_dim), llm_dim_(llm_dim) {
        namespace binding = modules::binding;
        hidden_ = config.hidden_size;
        heads_ = config.heads;
        const int64_t depth = config.depth;
        intermediate_ = config.intermediate_size;
        const auto storage = assets::TensorStorageType::Native;
        input_projection_ = binding::linear_from_source(store_, source, prefix + ".x_embedder", storage,
            hidden_, latent_dim_, true);
        final_norm_ = binding::norm_weight_from_source(store_, source, prefix + ".final_layer.norm_final", hidden_);
        output_dim_ = conditioned ? latent_dim_ : llm_dim_;
        output_projection_ = binding::linear_from_source(store_, source, prefix + ".final_layer.linear", storage,
            output_dim_, hidden_, true);
        if (conditioned_) {
            condition_projection_ = binding::linear_from_source(store_, source, prefix + ".c_embedder.cond_embedder",
                storage, hidden_, llm_dim_, true);
            time_in_ = binding::linear_from_source(store_, source, prefix + ".t_embedder.time_mlp.0",
                storage, hidden_, 256, true);
            time_out_ = binding::linear_from_source(store_, source, prefix + ".t_embedder.time_mlp.2",
                storage, hidden_, hidden_, true);
        } else {
            cls_ = store_.load_tensor(source, prefix + ".word_embedder.weight", storage, {1, hidden_});
        }
        auto frequencies = source.require_f32(prefix + ".rotary_embed.inv_freq");
        core::round_f32_to_bf16_in_place(frequencies);
        for (size_t i = 0; i < frequencies.size(); ++i) {
            frequencies[i] = std::pow(10000.0f, -static_cast<float>(2 * i) / (hidden_ / heads_)) / frequencies[i];
        }
        rope_factors_ = store_.make_f32(core::TensorShape::from_dims({hidden_ / heads_ / 2}), frequencies);
        for (int64_t i = 0; i < depth; ++i) {
            const auto p = prefix + ".blocks." + std::to_string(i);
            Block block;
            block.norm1 = binding::norm_weight_from_source(store_, source, p + ".norm1", hidden_);
            block.norm2 = binding::norm_weight_from_source(store_, source, p + ".norm2", hidden_);
            block.q = binding::linear_from_source(store_, source, p + ".attn.to_q", storage, hidden_, hidden_, true);
            block.k = binding::linear_from_source(store_, source, p + ".attn.to_k", storage, hidden_, hidden_, true);
            block.v = binding::linear_from_source(store_, source, p + ".attn.to_v", storage, hidden_, hidden_, true);
            block.attn_out = binding::linear_from_source(store_, source, p + ".attn.to_out.0", storage, hidden_, hidden_, true);
            block.ff_in = binding::linear_from_source(store_, source, p + ".mlp.ff.0.0", storage, intermediate_, hidden_, true);
            block.ff_out = binding::linear_from_source(store_, source, p + ".mlp.ff.2", storage, hidden_, intermediate_, true);
            blocks_.push_back(std::move(block));
        }
        store_.upload();
        graph_ = build(conditioned_ ? 2 : 1);
    }

    std::vector<float> run(const std::vector<float> & latent, const std::vector<float> & condition,
        const std::vector<float> & history, float time) {
        if (latent.empty() || latent.size() % latent_dim_ != 0 ||
            (conditioned_ && (latent.size() != static_cast<size_t>(latent_dim_) ||
                condition.size() != static_cast<size_t>(llm_dim_) || history.size() != latent.size()))) {
            throw std::runtime_error("Ming patch network input dimensions do not match the checkpoint");
        }
        const int64_t batch = conditioned_ ? 2 : latent.size() / latent_dim_;
        if (!conditioned_ && batch > 1 && (!reference_graph_ || reference_graph_->batch != batch)) {
            reference_graph_.reset();
            reference_graph_ = build(batch);
        }
        auto & graph = !conditioned_ && batch > 1 ? *reference_graph_ : *graph_;
        auto input = latent;
        if (conditioned_) input.insert(input.end(), latent.begin(), latent.end());
        core::write_tensor_f32(graph.input, input);
        if (conditioned_) {
            auto history_batch = history;
            history_batch.insert(history_batch.end(), history.begin(), history.end());
            auto condition_batch = condition;
            condition_batch.resize(2 * condition.size(), 0.0f);
            core::write_tensor_f32(graph.history, history_batch);
            core::write_tensor_f32(graph.condition, condition_batch);
            std::vector<float> scaled_time{1000.0f * time};
            core::round_f32_to_bf16_in_place(scaled_time);
            std::vector<float> embedding(512);
            const float frequency_step = static_cast<float>(std::log(10000.0) / 127.0);
            for (size_t i = 0; i < 128; ++i) {
                const float phase = scaled_time[0] * std::exp(-frequency_step * static_cast<float>(i));
                embedding[i] = embedding[256 + i] = std::sin(phase);
                embedding[128 + i] = embedding[384 + i] = std::cos(phase);
            }
            core::round_f32_to_bf16_in_place(embedding);
            core::write_tensor_f32(graph.time, embedding);
        }
        if (core::compute_graph(execution_, graph.graph, graph.plan, "ming3_tts.patch") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Ming patch graph execution failed");
        }
        return core::read_tensor_f32(graph.output.tensor);
    }

private:
    struct Block {
        modules::NormWeights norm1, norm2;
        modules::LinearWeights q, k, v, attn_out, ff_in, ff_out;
    };

    struct Graph {
        Graph(core::ExecutionContext & execution, int64_t batch) : execution(execution), batch(batch) {}
        ~Graph() {
            core::release_backend_graph_resources(execution.backend(), graph, true);
            plan.reset();
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }
        core::ExecutionContext & execution;
        int64_t batch;
        core::TensorValue input, history, condition, time, positions, output;
        ggml_context * context = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t allocator = nullptr;
        core::HostGraphPlan plan;
    };

    std::unique_ptr<Graph> build(int64_t batch) {
        auto graph = std::make_unique<Graph>(execution_, batch);
        const int64_t steps = conditioned_ ? 3 : 2;
        const int64_t dim = hidden_ / heads_;
        graph->context = ggml_init({4 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{graph->context, "ming3_tts.patch", execution_.backend_type()};
        graph->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, latent_dim_}));
        ggml_set_input(graph->input.tensor);
        auto x = modules::LinearModule({latent_dim_, hidden_, true}).build(ctx, graph->input, input_projection_);
        x = modules::Bf16RoundModule{}.build(ctx, x);
        if (conditioned_) {
            graph->history = core::make_tensor(ctx, GGML_TYPE_F32, graph->input.shape);
            graph->condition = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, llm_dim_}));
            graph->time = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, 256}));
            ggml_set_input(graph->history.tensor); ggml_set_input(graph->condition.tensor); ggml_set_input(graph->time.tensor);
            auto history = modules::LinearModule({latent_dim_, hidden_, true}).build(ctx, graph->history, input_projection_);
            history = modules::Bf16RoundModule{}.build(ctx, history);
            auto condition = modules::LinearModule({llm_dim_, hidden_, true}).build(ctx, graph->condition, condition_projection_);
            condition = modules::Bf16RoundModule{}.build(ctx, condition);
            auto t = modules::LinearModule({256, hidden_, true}).build(ctx, graph->time, time_in_);
            t = modules::Bf16RoundModule{}.build(ctx, t);
            t = modules::SiluModule{}.build(ctx, t);
            t = modules::Bf16RoundModule{}.build(ctx, t);
            t = modules::LinearModule({hidden_, hidden_, true}).build(ctx, t, time_out_);
            t = modules::Bf16RoundModule{}.build(ctx, t);
            t = modules::AddModule{}.build(ctx, t, condition);
            t = modules::Bf16RoundModule{}.build(ctx, t);
            x = modules::ConcatModule({1}).build(ctx, history, x);
            x = modules::ConcatModule({1}).build(ctx, t, x);
        } else {
            const auto cls = core::wrap_tensor(ggml_cast(ctx.ggml, cls_.tensor, GGML_TYPE_F32), cls_.shape, GGML_TYPE_F32);
            auto cls_batch = ggml_repeat(ctx.ggml, ggml_reshape_3d(ctx.ggml, cls.tensor, hidden_, 1, 1), x.tensor);
            x = modules::ConcatModule({1}).build(ctx,
                core::wrap_tensor(cls_batch, core::TensorShape::from_dims({batch, 1, hidden_}), GGML_TYPE_F32), x);
        }
        graph->positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(graph->positions.tensor);
        ggml_set_output(graph->positions.tensor);
        modules::NormConfig norm{hidden_, 1.0e-6f, true, false};
        norm.normalized_round_type = GGML_TYPE_BF16;
        modules::RoPEConfig rope{dim, GGML_ROPE_TYPE_NEOX, 10000.0f};
        rope.intermediate_round_type = GGML_TYPE_BF16;
        for (const auto & block : blocks_) {
            auto normalized = modules::RMSNormModule(norm).build(ctx, x, block.norm1);
            normalized = modules::Bf16RoundModule{}.build(ctx, normalized);
            auto q = modules::LinearModule({hidden_, hidden_, true}).build(ctx, normalized, block.q);
            auto k = modules::LinearModule({hidden_, hidden_, true}).build(ctx, normalized, block.k);
            auto v = modules::LinearModule({hidden_, hidden_, true}).build(ctx, normalized, block.v);
            q = modules::Bf16RoundModule{}.build(ctx, q);
            k = modules::Bf16RoundModule{}.build(ctx, k);
            v = modules::Bf16RoundModule{}.build(ctx, v);
            for (auto * projected : {&q, &k}) {
                auto * pairs = ggml_reshape_4d(ctx.ggml, projected->tensor, 2, dim / 2, heads_, batch * steps);
                pairs = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, pairs, 1, 0, 2, 3));
                *projected = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, pairs, dim, heads_, steps, batch),
                    core::TensorShape::from_dims({batch, steps, heads_, dim}), GGML_TYPE_F32);
                *projected = modules::RoPEModule(rope).build(ctx, *projected, graph->positions, &rope_factors_);
                *projected = modules::Bf16RoundModule{}.build(ctx, *projected);
                pairs = ggml_reshape_4d(ctx.ggml, projected->tensor, dim / 2, 2, heads_, batch * steps);
                pairs = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, pairs, 1, 0, 2, 3));
                *projected = core::wrap_tensor(ggml_reshape_4d(ctx.ggml, pairs, dim, heads_, steps, batch),
                    core::TensorShape::from_dims({batch, steps, heads_, dim}), GGML_TYPE_F32);
            }
            v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({batch, steps, heads_, dim}));
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            q = core::ensure_backend_addressable_layout(ctx, q);
            k = core::ensure_backend_addressable_layout(ctx, k);
            v = core::ensure_backend_addressable_layout(ctx, v);
            auto attention = modules::ScaledDotProductAttentionModule({dim}).build(ctx, q, k, v);
            attention = modules::Bf16RoundModule{}.build(ctx, attention);
            attention = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attention),
                core::TensorShape::from_dims({batch, steps, hidden_}));
            attention = modules::LinearModule({hidden_, hidden_, true}).build(ctx, attention, block.attn_out);
            attention = modules::Bf16RoundModule{}.build(ctx, attention);
            x = modules::AddModule{}.build(ctx, x, attention);
            x = modules::Bf16RoundModule{}.build(ctx, x);
            normalized = modules::RMSNormModule(norm).build(ctx, x, block.norm2);
            normalized = modules::Bf16RoundModule{}.build(ctx, normalized);
            auto ff = modules::LinearModule({hidden_, intermediate_, true}).build(ctx, normalized, block.ff_in);
            ff = modules::Bf16RoundModule{}.build(ctx, ff);
            ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
            ff = modules::Bf16RoundModule{}.build(ctx, ff);
            ff = modules::LinearModule({intermediate_, hidden_, true}).build(ctx, ff, block.ff_out);
            ff = modules::Bf16RoundModule{}.build(ctx, ff);
            x = modules::AddModule{}.build(ctx, x, ff);
            x = modules::Bf16RoundModule{}.build(ctx, x);
        }
        x = modules::RMSNormModule(norm).build(ctx, x, final_norm_);
        x = modules::Bf16RoundModule{}.build(ctx, x);
        x = modules::LinearModule({hidden_, output_dim_, true}).build(ctx, x, output_projection_);
        x = modules::Bf16RoundModule{}.build(ctx, x);
        graph->output = modules::SliceModule({1, conditioned_ ? steps - 1 : 0, 1}).build(ctx, x);
        graph->output = core::wrap_tensor(ggml_cont(ctx.ggml, graph->output.tensor), graph->output.shape, graph->output.type);
        ggml_set_output(graph->output.tensor);
        graph->graph = ggml_new_graph_custom(graph->context, 8192, false);
        ggml_build_forward_expand(graph->graph, graph->output.tensor);
        graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_reserve(graph->allocator, graph->graph) || !ggml_gallocr_alloc_graph(graph->allocator, graph->graph)) {
            throw std::runtime_error("Ming patch graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph->graph, graph->plan);
        const std::array<int32_t, 3> positions{0, 1, 2};
        ggml_backend_tensor_set(graph->positions.tensor, positions.data(), 0, steps * sizeof(int32_t));
        debug::timing_log_context_reservation("ming3_tts.patch", graph->context);
        return graph;
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    bool conditioned_;
    int64_t latent_dim_, llm_dim_, hidden_, heads_, intermediate_, output_dim_;
    std::vector<Block> blocks_;
    modules::LinearWeights input_projection_, output_projection_, condition_projection_, time_in_, time_out_;
    modules::NormWeights final_norm_;
    core::TensorValue cls_, rope_factors_;
    std::unique_ptr<Graph> graph_, reference_graph_;
};

}  // namespace

struct MingFlowRuntime::Impl {
    Impl(const assets::TensorSource & source, const MingFlowConfig & config, core::ExecutionContext & execution)
        : flow(source, config.network, config.network_prefix,
              config.latent_dim, config.conditioning_dim, true, execution),
          aggregator(source, config.aggregator, config.aggregator_prefix,
              config.latent_dim, config.conditioning_dim, false, execution) {
        time = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
        for (auto & t : time) t /= 32.0f;
        auto phase = time;
        for (auto & t : phase) t *= 1.5707963267948966f;
        core::round_f32_to_bf16_in_place(phase);
        for (auto & t : phase) t = std::cos(t);
        core::round_f32_to_bf16_in_place(phase);
        for (auto & t : phase) t -= 1.0f;
        core::round_f32_to_bf16_in_place(phase);
        for (size_t i = 0; i < time.size(); ++i) phase[i] += time[i];
        core::round_f32_to_bf16_in_place(phase);
        for (size_t i = 0; i < time.size(); ++i) time[i] -= phase[i];
        core::round_f32_to_bf16_in_place(time);
        debug::trace_log_f32("ming3_tts.flow.timesteps", {11}, time);
    }
    MingPatchNetwork flow, aggregator;
    std::vector<float> time;
};

MingFlowRuntime::MingFlowRuntime(std::shared_ptr<const assets::TensorSource> source,
    const MingFlowConfig & config, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(*source, config, execution)) {}
MingFlowRuntime::~MingFlowRuntime() = default;

std::vector<float> MingFlowRuntime::predict_velocity(const std::vector<float> & latent,
    const std::vector<float> & conditioning, const std::vector<float> & history, float time) {
    return impl_->flow.run(latent, conditioning, history, time);
}

std::vector<float> MingFlowRuntime::aggregate(const std::vector<float> & latent) {
    return impl_->aggregator.run(latent, {}, {}, 0.0f);
}

std::vector<float> MingFlowRuntime::sample(const std::vector<float> & conditioning,
    const std::vector<float> & history, const std::vector<float> & noise, const std::vector<float> & sde_noise,
    float cfg, float sigma, float temperature) {
    if (sde_noise.size() != noise.size() * 10 || temperature < 0) {
        throw std::runtime_error("Ming flow requires ten SDE-noise frames and a nonnegative temperature");
    }
    const auto & time = impl_->time;
    auto y = noise;
    for (size_t step = 0; step < 10; ++step) {
        auto prediction = impl_->flow.run(y, conditioning, history, time[step]);
        std::vector<float> guided(noise.size());
        for (size_t i = 0; i < guided.size(); ++i) guided[i] = prediction[i] - prediction[guided.size() + i];
        core::round_f32_to_bf16_in_place(guided);
        for (auto & value : guided) value *= cfg;
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < guided.size(); ++i) guided[i] += prediction[i];
        core::round_f32_to_bf16_in_place(guided);
        std::vector<float> dt{time[step + 1] - time[step]};
        core::round_f32_to_bf16_in_place(dt);
        for (auto & value : guided) value *= dt[0];
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < y.size(); ++i) y[i] += guided[i];
        core::round_f32_to_bf16_in_place(y);
        for (size_t i = 0; i < y.size(); ++i) guided[i] = sigma * std::sqrt(temperature) *
            std::sqrt(std::abs(dt[0])) * sde_noise[step * y.size() + i];
        core::round_f32_to_bf16_in_place(guided);
        for (size_t i = 0; i < y.size(); ++i) y[i] += guided[i];
        core::round_f32_to_bf16_in_place(y);
    }
    return y;
}

}  // namespace engine::modules

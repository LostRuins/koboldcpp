#include "engine/community_models/audio8_asr/projector.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::community_models::audio8_asr {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

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

struct TowerLayerWeights {
    core::TensorValue norm_weight;
    core::TensorValue norm_bias;
    core::TensorValue fc1_weight;
    core::TensorValue fc1_bias;
    core::TensorValue fc2_weight;
    core::TensorValue fc2_bias;
};

struct ProjectorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<TowerLayerWeights> layers;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
    core::TensorValue projector_norm_weight;
    core::TensorValue projector_norm_bias;
    core::TensorValue projector_weight;
    core::TensorValue projector_bias;
};

ProjectorWeights load_weights(
    const assets::TensorSource & source,
    const Audio8TowerConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    ProjectorWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audio8_asr.projector.weights",
        weight_context_bytes);
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const std::string prefix =
            "audio_mlp_tower.layers." + std::to_string(layer);
        TowerLayerWeights w;
        w.norm_weight = weights.store->load_f32_tensor(
            source, prefix + ".norm.weight", {config.hidden_size});
        w.norm_bias = weights.store->load_f32_tensor(
            source, prefix + ".norm.bias", {config.hidden_size});
        w.fc1_weight = weights.store->load_tensor(
            source, prefix + ".fc1.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.fc1_bias = weights.store->load_f32_tensor(
            source, prefix + ".fc1.bias", {config.intermediate_size});
        w.fc2_weight = weights.store->load_tensor(
            source, prefix + ".fc2.weight", storage_type, {config.hidden_size, config.intermediate_size});
        w.fc2_bias = weights.store->load_f32_tensor(
            source, prefix + ".fc2.bias", {config.hidden_size});
        weights.layers.push_back(std::move(w));
    }
    weights.final_norm_weight = weights.store->load_f32_tensor(
        source, "audio_mlp_tower.final_norm.weight", {config.hidden_size});
    weights.final_norm_bias = weights.store->load_f32_tensor(
        source, "audio_mlp_tower.final_norm.bias", {config.hidden_size});
    weights.projector_norm_weight = weights.store->load_f32_tensor(
        source, "audio_projector.0.weight", {config.hidden_size});
    weights.projector_norm_bias = weights.store->load_f32_tensor(
        source, "audio_projector.0.bias", {config.hidden_size});
    weights.projector_weight = weights.store->load_tensor(
        source, "audio_projector.1.weight", storage_type, {config.output_size, config.hidden_size});
    weights.projector_bias = weights.store->load_f32_tensor(
        source, "audio_projector.1.bias", {config.output_size});
    weights.store->upload();
    return weights;
}

// torch adaptive_avg_pool1d output i averages input positions
// [floor(i * T / N), ceil((i + 1) * T / N)). The matrix is laid out for
// ggml_mul_mat(A, B) with A = transposed hidden [T, hidden] and
// B = this matrix [T, N]: element (t, n) lives at t + n * T.
std::vector<float> adaptive_pool_matrix(int64_t input_tokens, int64_t output_tokens) {
    std::vector<float> matrix(
        static_cast<size_t>(input_tokens * output_tokens), 0.0f);
    for (int64_t out = 0; out < output_tokens; ++out) {
        const int64_t start = (out * input_tokens) / output_tokens;
        const int64_t end = ((out + 1) * input_tokens + output_tokens - 1) / output_tokens;
        const float scale = 1.0f / static_cast<float>(end - start);
        for (int64_t in = start; in < end; ++in) {
            matrix[static_cast<size_t>(in + out * input_tokens)] = scale;
        }
    }
    return matrix;
}

class ProjectorGraph {
public:
    ProjectorGraph(
        std::shared_ptr<const Audio8TowerConfig> config,
        std::shared_ptr<ProjectorWeights> weights,
        core::BackendType backend_type,
        ggml_backend_t backend,
        int64_t input_tokens,
        int64_t output_tokens,
        size_t graph_arena_bytes)
        : config_(std::move(config)),
          weights_(std::move(weights)),
          backend_(backend),
          input_tokens_(input_tokens),
          output_tokens_(output_tokens) {
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Audio8 ASR projector graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "audio8_asr.projector", backend_type};

        input_ = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32, config_->hidden_size, input_tokens_);
        auto x = core::wrap_tensor(
            input_,
            core::TensorShape::from_dims({1, input_tokens_, config_->hidden_size}),
            GGML_TYPE_F32);
        for (const auto & layer : weights_->layers) {
            auto normed = modules::LayerNormModule(
                              {config_->hidden_size, config_->norm_eps, true, true})
                              .build(ctx, x, {layer.norm_weight, layer.norm_bias});
            auto fc1 = modules::LinearModule(
                           {config_->hidden_size, config_->intermediate_size, true, GGML_PREC_F32})
                           .build(ctx, normed, {layer.fc1_weight, layer.fc1_bias});
            auto act = core::wrap_tensor(
                ggml_gelu_erf(ctx.ggml, fc1.tensor),
                core::TensorShape::from_dims({1, input_tokens_, config_->intermediate_size}),
                GGML_TYPE_F32);
            auto fc2 = modules::LinearModule(
                           {config_->intermediate_size, config_->hidden_size, true, GGML_PREC_F32})
                           .build(ctx, act, {layer.fc2_weight, layer.fc2_bias});
            x = core::wrap_tensor(
                ggml_add(ctx.ggml, x.tensor, fc2.tensor),
                core::TensorShape::from_dims({1, input_tokens_, config_->hidden_size}),
                GGML_TYPE_F32);
        }
        auto final_norm = modules::LayerNormModule(
                              {config_->hidden_size, config_->norm_eps, true, true})
                              .build(ctx, x, {weights_->final_norm_weight, weights_->final_norm_bias});

        pool_matrix_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, input_tokens_, output_tokens_);
        auto pool = core::wrap_tensor(
            pool_matrix_,
            core::TensorShape::from_dims({output_tokens_, input_tokens_}),
            GGML_TYPE_F32);
        // Materialized tower features [T, hidden]: mul_mat's "weight" operand
        // must be contiguous, so the transpose view is copied eagerly.
        auto hidden_transposed = core::wrap_tensor(
            ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, final_norm.tensor)),
            core::TensorShape::from_dims({config_->hidden_size, input_tokens_}),
            GGML_TYPE_F32);
        // mul_mat([T, hidden], [T, N]) -> [hidden, N]: pooled rows stay
        // token-major for the projector that follows.
        auto pooled = core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, hidden_transposed.tensor, pool.tensor),
            core::TensorShape::from_dims({1, output_tokens_, config_->hidden_size}),
            GGML_TYPE_F32);

        auto projector_norm = modules::LayerNormModule(
                                  {config_->hidden_size, config_->norm_eps, true, true})
                                  .build(ctx, pooled, {weights_->projector_norm_weight, weights_->projector_norm_bias});
        auto projected = modules::LinearModule(
                             {config_->hidden_size, config_->output_size, true, GGML_PREC_F32})
                             .build(ctx, projector_norm, {weights_->projector_weight, weights_->projector_bias});
        output_ = projected.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);

        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
            return gallocr_ != nullptr &&
                ggml_gallocr_reserve(gallocr_.get(), graph_) &&
                ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        if (!try_alloc() &&
            (engine::core::trim_backend_pools(backend_), !try_alloc())) {
            throw engine::runtime::CapacityError(
                "Audio8 ASR projector graph does not fit in device memory at this size ("
                + std::to_string(input_tokens_) + " encoder tokens, "
                + std::to_string(output_tokens_) + " audio tokens)");
        }
        pool_values_ = adaptive_pool_matrix(input_tokens_, output_tokens_);
        debug::timing_log_scalar("audio8_asr.projector.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("audio8_asr.projector.encoder_tokens", input_tokens_);
        debug::trace_log_scalar("audio8_asr.projector.audio_tokens", output_tokens_);
    }

    ~ProjectorGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_, true);
    }

    bool matches(int64_t input_tokens, int64_t output_tokens) const {
        return input_tokens_ == input_tokens && output_tokens_ == output_tokens;
    }

    std::vector<float> run(const std::vector<float> & encoder_output) {
        if (static_cast<int64_t>(encoder_output.size()) != input_tokens_ * config_->hidden_size) {
            throw std::runtime_error("Audio8 ASR projector encoder output size mismatch");
        }
        const auto upload_start = Clock::now();
        // Re-uploaded every run: leaves are not pinned by the graph allocator
        // (the gallocr may hand a leaf's buffer to a later node once consumed),
        // so both the input and the constant pool matrix are rewritten here.
        ggml_backend_tensor_set(input_, encoder_output.data(), 0, encoder_output.size() * sizeof(float));
        ggml_backend_tensor_set(
            pool_matrix_, pool_values_.data(), 0, pool_values_.size() * sizeof(float));
        debug::timing_log_scalar("audio8_asr.projector.input_upload_ms", engine::debug::elapsed_ms(upload_start, Clock::now()));
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Audio8 ASR projector graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(output_tokens_ * config_->output_size));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        return output;
    }

private:
    std::shared_ptr<const Audio8TowerConfig> config_;
    std::shared_ptr<ProjectorWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    int64_t input_tokens_ = 0;
    int64_t output_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * pool_matrix_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<float> pool_values_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

}  // namespace

struct Audio8ProjectorRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        Audio8TowerConfig config,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type)
        : config_(std::make_shared<const Audio8TowerConfig>(std::move(config))),
          weights_(std::make_shared<ProjectorWeights>(load_weights(
              *weights_source,
              *config_,
              execution.backend(),
              execution.backend_type(),
              weight_context_bytes,
              weight_storage_type))),
          execution_(execution),
          graph_arena_bytes_(graph_arena_bytes) {}

    std::shared_ptr<const Audio8TowerConfig> config_;
    std::shared_ptr<ProjectorWeights> weights_;
    core::ExecutionContext & execution_;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<ProjectorGraph> graph_;
};

Audio8ProjectorRuntime::Audio8ProjectorRuntime(
    std::shared_ptr<const assets::TensorSource> weights_source,
    const Audio8TowerConfig & config,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          config,
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

Audio8ProjectorRuntime::~Audio8ProjectorRuntime() = default;

Audio8ASRAudioEmbeddings Audio8ProjectorRuntime::project(
    const std::vector<float> & encoder_output,
    int64_t encoder_tokens,
    int64_t audio_tokens) {
    if (encoder_tokens <= 0 || audio_tokens <= 0 || audio_tokens > encoder_tokens) {
        throw std::runtime_error("Audio8 ASR projector token counts are invalid");
    }
    if (impl_->graph_ == nullptr || !impl_->graph_->matches(encoder_tokens, audio_tokens)) {
        impl_->graph_.reset();
        impl_->graph_ = std::make_unique<ProjectorGraph>(
            impl_->config_,
            impl_->weights_,
            impl_->execution_.backend_type(),
            impl_->execution_.backend(),
            encoder_tokens,
            audio_tokens,
            impl_->graph_arena_bytes_);
    }
    Audio8ASRAudioEmbeddings out;
    out.values = impl_->graph_->run(encoder_output);
    out.tokens = audio_tokens;
    out.hidden_size = impl_->config_->output_size;
    return out;
}

}  // namespace engine::community_models::audio8_asr

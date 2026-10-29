#pragma once

#include "engine/models/chatterbox/t3_component.h"

#include "component_weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace engine::models::chatterbox {
namespace {

size_t & t3_host_thread_count_setting() {
    static size_t thread_count = 1;
    return thread_count;
}

void set_t3_host_thread_count(size_t thread_count) {
    t3_host_thread_count_setting() = std::max<size_t>(1, thread_count);
}

class T3HostThreadPool {
public:
    explicit T3HostThreadPool(size_t thread_count)
        : thread_count_(std::max<size_t>(1, thread_count)) {
        const size_t worker_count = thread_count_ - 1;
        workers_.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    ~T3HostThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            generation_ += 1;
        }
        work_cv_.notify_all();
        for (auto & worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    size_t thread_count() const {
        return thread_count_;
    }

    template <typename Fn>
    void parallel_for(size_t total, size_t grain, Fn && fn) {
        if (total == 0) {
            return;
        }
        if (workers_.empty() || total <= grain) {
            fn(0, total);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = std::forward<Fn>(fn);
            total_ = total;
            grain_ = std::max<size_t>(1, grain);
            next_ = 0;
            active_workers_ = workers_.size();
            done_ = false;
            generation_ += 1;
        }
        work_cv_.notify_all();
        size_t begin = 0;
        size_t end = 0;
        while (claim_chunk(begin, end)) {
            fn(begin, end);
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() {
                return done_;
            });
            fn_ = {};
        }
    }

private:
    bool claim_chunk(size_t & begin, size_t & end) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_ >= total_) {
            return false;
        }
        begin = next_;
        end = std::min(total_, begin + grain_);
        next_ = end;
        return true;
    }

    void finish_worker() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_workers_ == 0) {
            return;
        }
        active_workers_ -= 1;
        if (active_workers_ == 0) {
            done_ = true;
            done_cv_.notify_one();
        }
    }

    void worker_loop() {
        size_t seen_generation = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait(lock, [&]() {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = generation_;
            }
            size_t begin = 0;
            size_t end = 0;
            while (claim_chunk(begin, end)) {
                fn_(begin, end);
            }
            finish_worker();
        }
    }

    std::vector<std::thread> workers_;
    size_t thread_count_ = 1;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::function<void(size_t, size_t)> fn_;
    size_t total_ = 0;
    size_t grain_ = 1;
    size_t next_ = 0;
    size_t active_workers_ = 0;
    size_t generation_ = 0;
    bool done_ = true;
    bool stop_ = false;
};

T3HostThreadPool & t3_host_thread_pool() {
    static std::mutex pool_mutex;
    static std::unique_ptr<T3HostThreadPool> pool;
    const size_t requested_threads = t3_host_thread_count_setting();
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!pool || pool->thread_count() != requested_threads) {
        pool = std::make_unique<T3HostThreadPool>(requested_threads);
    }
    return *pool;
}

void softmax_into(const std::vector<float> & logits, std::vector<float> & probs) {
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : logits) {
        max_value = std::max(max_value, value);
    }
    probs.assign(logits.size(), 0.0f);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_value);
        sum += static_cast<double>(probs[i]);
    }
    for (float & value : probs) {
        value = static_cast<float>(static_cast<double>(value) / sum);
    }
}

std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs;
    softmax_into(logits, probs);
    return probs;
}

std::vector<float> gather_rows(
    const std::vector<float> & table,
    int64_t rows,
    int64_t cols,
    const std::vector<int32_t> & indices) {
    std::vector<float> out(static_cast<size_t>(indices.size() * cols), 0.0f);
    for (size_t i = 0; i < indices.size(); ++i) {
        const int32_t index = indices[i];
        if (index < 0 || index >= rows) {
            throw std::runtime_error("embedding index out of range");
        }
        const float * src = table.data() + static_cast<ptrdiff_t>(index * cols);
        float * dst = out.data() + static_cast<ptrdiff_t>(i * cols);
        std::copy(src, src + cols, dst);
    }
    return out;
}

std::vector<float> linear_rows(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_features,
    const std::vector<float> & weight,
    int64_t out_features,
    const std::vector<float> * bias = nullptr) {
    std::vector<float> out(static_cast<size_t>(rows * out_features), 0.0f);
    const size_t total_outputs = static_cast<size_t>(rows * out_features);
    auto compute_range = [&](size_t begin, size_t end) {
        for (size_t linear_index = begin; linear_index < end; ++linear_index) {
            const int64_t row = static_cast<int64_t>(linear_index / static_cast<size_t>(out_features));
            const int64_t out_index = static_cast<int64_t>(linear_index % static_cast<size_t>(out_features));
            const float * src = input.data() + static_cast<ptrdiff_t>(row * in_features);
            double value = bias == nullptr ? 0.0 : static_cast<double>((*bias)[static_cast<size_t>(out_index)]);
            const float * w = weight.data() + static_cast<ptrdiff_t>(out_index * in_features);
            for (int64_t in_index = 0; in_index < in_features; ++in_index) {
                value += static_cast<double>(w[in_index]) * static_cast<double>(src[in_index]);
            }
            out[static_cast<size_t>(row * out_features + out_index)] = static_cast<float>(value);
        }
    };
    if (total_outputs * static_cast<size_t>(in_features) > 262144) {
        t3_host_thread_pool().parallel_for(total_outputs, 32, compute_range);
    } else {
        compute_range(0, total_outputs);
    }
    return out;
}

std::vector<float> layer_norm_rows(
    const std::vector<float> & input,
    int64_t rows,
    int64_t hidden_size,
    const std::vector<float> & weight,
    const std::vector<float> & bias,
    float eps) {
    std::vector<float> out(static_cast<size_t>(rows * hidden_size), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const float * src = input.data() + static_cast<ptrdiff_t>(row * hidden_size);
        double mean = 0.0;
        for (int64_t i = 0; i < hidden_size; ++i) {
            mean += static_cast<double>(src[i]);
        }
        mean /= static_cast<double>(hidden_size);
        double var = 0.0;
        for (int64_t i = 0; i < hidden_size; ++i) {
            const double delta = static_cast<double>(src[i]) - mean;
            var += delta * delta;
        }
        var /= static_cast<double>(hidden_size);
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        float * dst = out.data() + static_cast<ptrdiff_t>(row * hidden_size);
        for (int64_t i = 0; i < hidden_size; ++i) {
            dst[i] = static_cast<float>((static_cast<double>(src[i]) - mean) * static_cast<double>(inv)) *
                weight[static_cast<size_t>(i)] + bias[static_cast<size_t>(i)];
        }
    }
    return out;
}

std::vector<float> attention_context_from_projected_qkv_batched(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int64_t batch,
    int64_t q_len,
    int64_t kv_len,
    int64_t hidden_size,
    int64_t num_heads);

std::vector<float> attention_qkv_batched(
    const std::vector<float> & q_input,
    int64_t batch,
    int64_t q_len,
    const std::vector<float> & kv_input,
    int64_t kv_len,
    int64_t hidden_size,
    int64_t num_heads,
    const std::vector<float> & q_weight,
    const std::vector<float> & q_bias,
    const std::vector<float> & k_weight,
    const std::vector<float> & k_bias,
    const std::vector<float> & v_weight,
    const std::vector<float> & v_bias,
    const std::vector<float> & out_weight,
    const std::vector<float> & out_bias) {
    const int64_t head_dim = hidden_size / num_heads;
    const auto q = linear_rows(q_input, batch * q_len, hidden_size, q_weight, hidden_size, &q_bias);
    const auto k = linear_rows(kv_input, batch * kv_len, hidden_size, k_weight, hidden_size, &k_bias);
    const auto v = linear_rows(kv_input, batch * kv_len, hidden_size, v_weight, hidden_size, &v_bias);
    (void)head_dim;
    const auto attended = attention_context_from_projected_qkv_batched(
        q, k, v, batch, q_len, kv_len, hidden_size, num_heads);
    return linear_rows(attended, batch * q_len, hidden_size, out_weight, hidden_size, &out_bias);
}

std::vector<float> attention_context_from_projected_qkv_batched(
    const std::vector<float> & q,
    const std::vector<float> & k,
    const std::vector<float> & v,
    int64_t batch,
    int64_t q_len,
    int64_t kv_len,
    int64_t hidden_size,
    int64_t num_heads) {
    const int64_t head_dim = hidden_size / num_heads;
    std::vector<float> attended(static_cast<size_t>(batch * q_len * hidden_size), 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (batch * q_len * num_heads > 32)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < q_len; ++row) {
            for (int64_t head = 0; head < num_heads; ++head) {
                std::vector<float> scores(static_cast<size_t>(kv_len), 0.0f);
                for (int64_t key_row = 0; key_row < kv_len; ++key_row) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const int64_t idx = head * head_dim + d;
                        dot += static_cast<double>(q[static_cast<size_t>(((b * q_len + row) * hidden_size) + idx)]) *
                               static_cast<double>(k[static_cast<size_t>(((b * kv_len + key_row) * hidden_size) + idx)]);
                    }
                    scores[static_cast<size_t>(key_row)] = static_cast<float>(dot * static_cast<double>(scale));
                }
                const auto probs = softmax(scores);
                for (int64_t d = 0; d < head_dim; ++d) {
                    double value = 0.0;
                    const int64_t idx = head * head_dim + d;
                    for (int64_t key_row = 0; key_row < kv_len; ++key_row) {
                        value += static_cast<double>(probs[static_cast<size_t>(key_row)]) *
                                 static_cast<double>(v[static_cast<size_t>(((b * kv_len + key_row) * hidden_size) + idx)]);
                    }
                    attended[static_cast<size_t>(((b * q_len + row) * hidden_size) + idx)] = static_cast<float>(value);
                }
            }
        }
    }
    return attended;
}

core::TensorValue contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue concat_along_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs,
    int logical_axis) {
    auto output_shape = lhs.shape;
    output_shape.dims[logical_axis] += rhs.shape.dims[logical_axis];
    return core::wrap_tensor(
        ggml_concat(
            ctx.ggml,
            lhs.tensor,
            rhs.tensor,
            core::logical_axis_to_ggml_axis(lhs.shape.rank, logical_axis)),
        output_shape,
        lhs.type);
}

core::TensorValue add_contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, lhs).tensor, contiguous(ctx, rhs).tensor),
        lhs.shape,
        lhs.type);
}

class T3DecodeBackendOwner {
public:
    T3DecodeBackendOwner(
        const T3InferenceWeights & weights,
        const engine::core::BackendConfig & config)
        : config_(config),
          execution_context_(weights.execution_context) {
        if (!execution_context_) {
            throw std::runtime_error("T3 backend owner requires loaded backend weights");
        }
    }

    ggml_backend_t backend() const noexcept { return execution_context_->backend(); }
    const engine::core::ExecutionContext & execution_context() const noexcept { return *execution_context_; }
    const engine::core::BackendConfig & config() const noexcept { return config_; }
    bool uses_host_graph_plan() const noexcept { return execution_context_->uses_host_graph_plan(); }

private:
    engine::core::BackendConfig config_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
};

core::TensorValue make_graph_param_tensor(
    const T3GraphWeight & weight);

core::TensorValue apply_llama3_rope(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & rope_factors,
    int64_t head_dim) {
    return core::wrap_tensor(
        ggml_rope_ext(
            ctx.ggml,
            input.tensor,
            positions.tensor,
            rope_factors.tensor,
            static_cast<int>(head_dim),
            GGML_ROPE_TYPE_NEOX,
            0,
            500000.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue make_graph_param_tensor(
    const T3GraphWeight & weight) {
    return weight.tensor;
}

core::TensorValue swiglu_backend(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const T3GraphWeight & gate_weight,
    const T3GraphWeight & up_weight,
    const T3GraphWeight & down_weight) {
    const int64_t hidden_size = input.shape.last_dim();
    const int64_t intermediate_size = gate_weight.tensor.shape.dims[0];

    const auto gate_w = make_graph_param_tensor(gate_weight);
    const auto up_w = make_graph_param_tensor(up_weight);
    const auto down_w = make_graph_param_tensor(down_weight);

    const auto gate = modules::LinearModule({hidden_size, intermediate_size, false}).build(
        ctx,
        input,
        {gate_w, std::nullopt});
    const auto up = modules::LinearModule({hidden_size, intermediate_size, false}).build(
        ctx,
        input,
        {up_w, std::nullopt});
    const auto activated = modules::SiluModule().build(ctx, gate);
    const auto fused = modules::MulModule().build(ctx, activated, up);
    return modules::LinearModule({intermediate_size, hidden_size, false}).build(
        ctx,
        fused,
        {down_w, std::nullopt});
}

struct T3LayerBackendOutputs {
    core::TensorValue hidden;
    core::TensorValue key;
    core::TensorValue value;
};

struct T3BackendLayerState {
    std::vector<float> key;
    std::vector<float> value;
};

struct T3BackendCacheState {
    int64_t batch = 0;
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t steps = 0;
    std::vector<T3BackendLayerState> layers;
};

T3LayerBackendOutputs build_t3_layer_backend(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & rope_factors,
    const T3InferenceWeights::TransformerLayer & layer,
    int64_t num_heads,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t hidden_size = input.shape.last_dim();
    const int64_t head_dim = hidden_size / num_heads;

    auto hidden = modules::RMSNormModule({hidden_size, 1.0e-5f, true, false}).build(
        ctx,
        input,
        {layer.input_layernorm_tensor, std::nullopt});

    const auto q_w = make_graph_param_tensor(layer.q_proj_weight);
    const auto k_w = make_graph_param_tensor(layer.k_proj_weight);
    const auto v_w = make_graph_param_tensor(layer.v_proj_weight);
    const auto o_w = make_graph_param_tensor(layer.o_proj_weight);

    auto q = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {q_w, std::nullopt});
    auto k = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {k_w, std::nullopt});
    auto v = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {v_w, std::nullopt});

    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));

    q = apply_llama3_rope(ctx, q, positions, rope_factors, head_dim);
    k = apply_llama3_rope(ctx, k, positions, rope_factors, head_dim);

    const auto key_out = k;
    const auto value_out = v;

    const auto prefix_key_ready = prefix_key.has_value() ? std::optional<core::TensorValue>(contiguous(ctx, *prefix_key)) : std::nullopt;
    const auto prefix_value_ready = prefix_value.has_value() ? std::optional<core::TensorValue>(contiguous(ctx, *prefix_value)) : std::nullopt;

    auto all_k = prefix_key_ready.has_value() ? concat_along_axis(ctx, *prefix_key_ready, k, 1) : k;
    auto all_v = prefix_value_ready.has_value() ? concat_along_axis(ctx, *prefix_value_ready, v, 1) : v;

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    auto all_k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_k);
    auto all_v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_v);
    auto k_transposed = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, all_k_heads);
    auto scores = modules::MatMulModule().build(ctx, q_heads, k_transposed);
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = add_contiguous(ctx, scores, *attention_mask);
        attn = core::wrap_tensor(
            ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))), scores.shape, GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, prefix_key.has_value() ? static_cast<int>(prefix_key->shape.dims[1]) : 0), scores.shape, GGML_TYPE_F32);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    }

    auto context = modules::MatMulModule().build(ctx, attn, all_v_heads);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = contiguous(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    context = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, context, {o_w, std::nullopt});
    hidden = add_contiguous(ctx, context, input);

    const auto ff_in = modules::RMSNormModule({hidden_size, 1.0e-5f, true, false}).build(
        ctx,
        hidden,
        {layer.post_attention_layernorm_tensor, std::nullopt});
    const auto ff_out = swiglu_backend(
        ctx,
        ff_in,
        layer.gate_proj_weight,
        layer.up_proj_weight,
        layer.down_proj_weight);
    return {add_contiguous(ctx, ff_out, hidden), key_out, value_out};
}

T3LayerBackendOutputs build_t3_layer_backend_cached(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & rope_factors,
    const T3InferenceWeights::TransformerLayer & layer,
    int64_t num_heads,
    const std::vector<core::TensorValue> & batch_key_cache,
    const std::vector<core::TensorValue> & batch_value_cache,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t hidden_size = input.shape.last_dim();
    const int64_t head_dim = hidden_size / num_heads;
    if (static_cast<int64_t>(batch_key_cache.size()) != input.shape.dims[0] ||
        batch_value_cache.size() != batch_key_cache.size()) {
        throw std::runtime_error("T3 cached decode cache batch count mismatch");
    }

    auto hidden = modules::RMSNormModule({hidden_size, 1.0e-5f, true, false}).build(
        ctx,
        input,
        {layer.input_layernorm_tensor, std::nullopt});

    const auto q_w = make_graph_param_tensor(layer.q_proj_weight);
    const auto k_w = make_graph_param_tensor(layer.k_proj_weight);
    const auto v_w = make_graph_param_tensor(layer.v_proj_weight);
    const auto o_w = make_graph_param_tensor(layer.o_proj_weight);

    auto q = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {q_w, std::nullopt});
    auto k = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {k_w, std::nullopt});
    auto v = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, hidden, {v_w, std::nullopt});

    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], num_heads, head_dim}));

    q = apply_llama3_rope(ctx, q, positions, rope_factors, head_dim);
    k = apply_llama3_rope(ctx, k, positions, rope_factors, head_dim);

    const auto key_out = k;
    const auto value_out = v;

    const modules::FastKVSetRowsModule set_rows;
    std::vector<core::TensorValue> updated_keys;
    std::vector<core::TensorValue> updated_values;
    updated_keys.reserve(batch_key_cache.size());
    updated_values.reserve(batch_value_cache.size());
    for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
        auto key_row = modules::SliceModule({0, batch_index, 1}).build(ctx, k);
        auto value_row = modules::SliceModule({0, batch_index, 1}).build(ctx, v);
        updated_keys.push_back(set_rows.build(
            ctx,
            batch_key_cache[static_cast<size_t>(batch_index)],
            key_row,
            cache_slot));
        updated_values.push_back(set_rows.build(
            ctx,
            batch_value_cache[static_cast<size_t>(batch_index)],
            value_row,
            cache_slot));
    }

    auto all_k = updated_keys.front();
    auto all_v = updated_values.front();
    for (size_t batch_index = 1; batch_index < updated_keys.size(); ++batch_index) {
        all_k = modules::ConcatModule({0}).build(ctx, all_k, updated_keys[batch_index]);
        all_v = modules::ConcatModule({0}).build(ctx, all_v, updated_values[batch_index]);
    }

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    auto all_k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_k);
    auto all_v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_v);
    q_heads = contiguous(ctx, q_heads);
    all_k_heads = contiguous(ctx, all_k_heads);
    all_v_heads = contiguous(ctx, all_v_heads);

    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        all_k_heads.tensor,
        all_v_heads.tensor,
        attention_mask.tensor,
        1.0f / std::sqrt(static_cast<float>(head_dim)),
        0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    auto context = core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], head_dim}),
        GGML_TYPE_F32);
    context = contiguous(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    context = modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, context, {o_w, std::nullopt});
    hidden = add_contiguous(ctx, context, input);

    const auto ff_in = modules::RMSNormModule({hidden_size, 1.0e-5f, true, false}).build(
        ctx,
        hidden,
        {layer.post_attention_layernorm_tensor, std::nullopt});
    const auto ff_out = swiglu_backend(
        ctx,
        ff_in,
        layer.gate_proj_weight,
        layer.up_proj_weight,
        layer.down_proj_weight);
    return {add_contiguous(ctx, ff_out, hidden), key_out, value_out};
}

inline bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type &&
        lhs.device == rhs.device &&
        lhs.threads == rhs.threads;
}

inline void fill_t3_decode_attention_mask(
    std::vector<float> & values,
    int64_t batch,
    int64_t cache_steps,
    int64_t valid_steps) {
    const int64_t masked_prefix_begin = std::clamp<int64_t>(valid_steps + 1, 0, cache_steps);
    values.assign(static_cast<size_t>(batch * cache_steps), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        float * row = values.data() + static_cast<ptrdiff_t>(batch_index * cache_steps);
        for (int64_t step = masked_prefix_begin; step < cache_steps; ++step) {
            row[step] = -10000.0f;
        }
    }
}

class T3DecodeBackendRunner {
public:
    T3DecodeBackendRunner(
        const T3InferenceWeights & weights,
        int64_t batch,
        int64_t cache_steps,
        std::shared_ptr<T3DecodeBackendOwner> owner)
        : owner_(std::move(owner)),
          backend_config_(owner_->config()),
          batch_(batch),
          cache_steps_(cache_steps),
          hidden_size_(weights.hidden_size),
          num_heads_(weights.num_heads),
          head_dim_(weights.hidden_size / weights.num_heads),
          step_elems_(batch * weights.hidden_size) {
        if (batch_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("T3DecodeBackendRunner requires positive batch and cache_steps");
        }

        ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for T3 decode runner");
        }

        core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "t3_decode_runner";

        try {
            input_hidden_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_, 1, hidden_size_}));
            positions_ = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({1}));
            attention_mask_ = core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({batch_, 1, 1, cache_steps_}));

            key_cache_tensors_.reserve(weights.layers.size());
            value_cache_tensors_.reserve(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                std::vector<core::TensorValue> layer_keys;
                std::vector<core::TensorValue> layer_values;
                layer_keys.reserve(static_cast<size_t>(batch_));
                layer_values.reserve(static_cast<size_t>(batch_));
                for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
                    layer_keys.push_back(core::make_tensor(
                        ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({1, cache_steps_, num_heads_, head_dim_})));
                    layer_values.push_back(core::make_tensor(
                        ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({1, cache_steps_, num_heads_, head_dim_})));
                }
                key_cache_tensors_.push_back(std::move(layer_keys));
                value_cache_tensors_.push_back(std::move(layer_values));
            }

            auto hidden = input_hidden_;
            cache_slot_ = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({1}));
            current_keys_.reserve(weights.layers.size());
            current_values_.reserve(weights.layers.size());
            current_key_scratch_.resize(weights.layers.size());
            current_value_scratch_.resize(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                auto layer_outputs = build_t3_layer_backend_cached(
                    ctx,
                    hidden,
                    positions_,
                    weights.rope_factors_tensor,
                    weights.layers[layer_index],
                    num_heads_,
                    key_cache_tensors_[layer_index],
                    value_cache_tensors_[layer_index],
                    cache_slot_,
                    attention_mask_);
                hidden = layer_outputs.hidden;
                current_keys_.push_back(layer_outputs.key);
                current_values_.push_back(layer_outputs.value);
            }

            hidden_out_ = modules::RMSNormModule({hidden_size_, 1.0e-5f, true, false}).build(
                ctx,
                hidden,
                {weights.tfmr_norm_tensor, std::nullopt});
            const auto speech_head_w = make_graph_param_tensor(weights.speech_head_weight);
            logits_out_ = modules::LinearModule({hidden_size_, weights.speech_vocab, false}).build(
                ctx,
                hidden_out_,
                {speech_head_w, std::nullopt});

            graph_ = ggml_new_graph_custom(ggml_, 65536, false);
            ggml_build_forward_expand(graph_, hidden_out_.tensor);
            ggml_build_forward_expand(graph_, logits_out_.tensor);
            for (const auto & key : current_keys_) {
                ggml_build_forward_expand(graph_, key.tensor);
            }
            for (const auto & value : current_values_) {
                ggml_build_forward_expand(graph_, value.tensor);
            }
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, owner_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for T3 decode runner");
            }
            engine::core::prepare_host_graph_plan(owner_->execution_context(), graph_, cpu_plan_);
        } catch (...) {
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
                buffer_ = nullptr;
            }
            if (ggml_ != nullptr) {
                ggml_free(ggml_);
                ggml_ = nullptr;
            }
            throw;
        }

        cache_state_.batch = batch_;
        cache_state_.hidden_size = hidden_size_;
        cache_state_.num_heads = num_heads_;
        cache_state_.head_dim = head_dim_;
        cache_state_.layers.resize(weights.layers.size());
        attention_mask_scratch_.resize(static_cast<size_t>(batch_ * cache_steps_), 0.0f);
        import_state(cache_state_);
    }

    ~T3DecodeBackendRunner() {
        if (owner_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(owner_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
    }

    bool matches(
        int64_t batch,
        int64_t cache_steps,
        const engine::core::BackendConfig & backend_config) const {
        return batch_ == batch &&
            cache_steps_ == cache_steps &&
            same_backend(backend_config_, backend_config);
    }

    void import_state(const T3BackendCacheState & state) {
        if (state.batch != 0 && state.batch != batch_) {
            throw std::runtime_error("T3DecodeBackendRunner state batch mismatch");
        }
        if (state.hidden_size != 0 && state.hidden_size != hidden_size_) {
            throw std::runtime_error("T3DecodeBackendRunner state hidden size mismatch");
        }
        if (state.layers.size() != cache_state_.layers.size()) {
            throw std::runtime_error("T3DecodeBackendRunner state layer count mismatch");
        }
        if (state.steps > cache_steps_) {
            throw std::runtime_error("T3DecodeBackendRunner state exceeds cache capacity");
        }
        cache_state_ = state;
        cache_state_.batch = batch_;
        cache_state_.hidden_size = hidden_size_;
        cache_state_.num_heads = num_heads_;
        cache_state_.head_dim = head_dim_;
        for (size_t layer_index = 0; layer_index < cache_state_.layers.size(); ++layer_index) {
            const auto & layer = cache_state_.layers[layer_index];
            if (cache_state_.steps == 0 || layer.key.empty() || layer.value.empty()) {
                continue;
            }
            std::vector<float> batch_key(static_cast<size_t>(cache_state_.steps * hidden_size_), 0.0f);
            std::vector<float> batch_value(static_cast<size_t>(cache_state_.steps * hidden_size_), 0.0f);
            for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
                for (int64_t step = 0; step < cache_state_.steps; ++step) {
                    const size_t state_offset = static_cast<size_t>((step * batch_ + batch_index) * hidden_size_);
                    const size_t batch_offset = static_cast<size_t>(step * hidden_size_);
                    std::copy_n(
                        layer.key.data() + static_cast<std::ptrdiff_t>(state_offset),
                        hidden_size_,
                        batch_key.data() + static_cast<std::ptrdiff_t>(batch_offset));
                    std::copy_n(
                        layer.value.data() + static_cast<std::ptrdiff_t>(state_offset),
                        hidden_size_,
                        batch_value.data() + static_cast<std::ptrdiff_t>(batch_offset));
                }
                core::write_tensor_f32_slice(
                    key_cache_tensors_[layer_index][static_cast<size_t>(batch_index)],
                    0,
                    batch_key.data(),
                    batch_key.size());
                core::write_tensor_f32_slice(
                    value_cache_tensors_[layer_index][static_cast<size_t>(batch_index)],
                    0,
                    batch_value.data(),
                    batch_value.size());
            }
        }
    }

    T3BackendCacheState export_state() const {
        return cache_state_;
    }

    T3BackendCacheState export_state_from_device() const {
        T3BackendCacheState state = cache_state_;
        state.layers.assign(cache_state_.layers.size(), {});
        if (cache_state_.steps <= 0) {
            return state;
        }
        for (size_t layer_index = 0; layer_index < key_cache_tensors_.size(); ++layer_index) {
            auto & layer = state.layers[layer_index];
            layer.key.assign(static_cast<size_t>(cache_state_.steps * batch_ * hidden_size_), 0.0f);
            layer.value.assign(static_cast<size_t>(cache_state_.steps * batch_ * hidden_size_), 0.0f);
            for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
                const auto batch_key = core::read_tensor_f32(
                    key_cache_tensors_[layer_index][static_cast<size_t>(batch_index)].tensor);
                const auto batch_value = core::read_tensor_f32(
                    value_cache_tensors_[layer_index][static_cast<size_t>(batch_index)].tensor);
                for (int64_t step = 0; step < cache_state_.steps; ++step) {
                    const size_t src_offset = static_cast<size_t>(step * hidden_size_);
                    const size_t dst_offset = static_cast<size_t>((step * batch_ + batch_index) * hidden_size_);
                    std::copy_n(
                        batch_key.data() + static_cast<std::ptrdiff_t>(src_offset),
                        hidden_size_,
                        layer.key.data() + static_cast<std::ptrdiff_t>(dst_offset));
                    std::copy_n(
                        batch_value.data() + static_cast<std::ptrdiff_t>(src_offset),
                        hidden_size_,
                        layer.value.data() + static_cast<std::ptrdiff_t>(dst_offset));
                }
            }
        }
        return state;
    }

    int64_t valid_steps() const noexcept {
        return cache_state_.steps;
    }

    int64_t cache_capacity_steps() const noexcept {
        return cache_steps_;
    }

    int64_t current_end() const noexcept {
        return cache_state_.steps;
    }

    void set_capture_cache_state(bool capture) noexcept {
        capture_cache_state_ = capture;
    }

    std::vector<float> step(const std::vector<float> & input_hidden, int64_t position) {
        if (static_cast<int64_t>(input_hidden.size()) != batch_ * hidden_size_) {
            throw std::runtime_error("T3DecodeBackendRunner step input size mismatch");
        }
        fill_t3_decode_attention_mask(attention_mask_scratch_, batch_, cache_steps_, cache_state_.steps);
        core::write_tensor_f32(input_hidden_, input_hidden);
        core::write_tensor_i32(positions_, std::vector<int32_t>{static_cast<int32_t>(position)});
        core::write_tensor_i32(cache_slot_, std::vector<int32_t>{static_cast<int32_t>(cache_state_.steps)});
        core::write_tensor_f16(attention_mask_, attention_mask_scratch_);

        const ggml_status status = engine::core::compute_graph(owner_->execution_context(), graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml compute failed for T3 decode runner");
        }

        for (size_t layer_index = 0; layer_index < current_keys_.size(); ++layer_index) {
            if (capture_cache_state_) {
                core::read_tensor_f32_into(current_keys_[layer_index].tensor, current_key_scratch_[layer_index]);
                core::read_tensor_f32_into(current_values_[layer_index].tensor, current_value_scratch_[layer_index]);
                auto & layer = cache_state_.layers[layer_index];
                layer.key.insert(layer.key.end(), current_key_scratch_[layer_index].begin(), current_key_scratch_[layer_index].end());
                layer.value.insert(layer.value.end(), current_value_scratch_[layer_index].begin(), current_value_scratch_[layer_index].end());
            }
        }
        cache_state_.steps += 1;
        return core::read_tensor_f32(logits_out_.tensor);
    }

    std::vector<float> read_hidden() const {
        return core::read_tensor_f32(hidden_out_.tensor);
    }

private:
    std::shared_ptr<T3DecodeBackendOwner> owner_;
    engine::core::BackendConfig backend_config_;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::HostGraphPlan cpu_plan_;
    int64_t batch_ = 0;
    int64_t cache_steps_ = 0;
    int64_t hidden_size_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    int64_t step_elems_ = 0;
    core::TensorValue input_hidden_;
    core::TensorValue positions_;
    core::TensorValue cache_slot_;
    core::TensorValue attention_mask_;
    core::TensorValue hidden_out_;
    core::TensorValue logits_out_;
    std::vector<std::vector<core::TensorValue>> key_cache_tensors_;
    std::vector<std::vector<core::TensorValue>> value_cache_tensors_;
    std::vector<core::TensorValue> current_keys_;
    std::vector<core::TensorValue> current_values_;
    T3BackendCacheState cache_state_;
    bool capture_cache_state_ = false;
    std::vector<float> attention_mask_scratch_;
    std::vector<std::vector<float>> current_key_scratch_;
    std::vector<std::vector<float>> current_value_scratch_;
};

struct T3PrefillBackendOutput {
    std::vector<float> logits;
    T3BackendCacheState cache;
};

class T3PrefillBackendRunner {
public:
    T3PrefillBackendRunner(
        const T3InferenceWeights & weights,
        int64_t batch,
        int64_t prefix_steps,
        int64_t seq_len,
        std::shared_ptr<T3DecodeBackendOwner> owner)
        : owner_(std::move(owner)),
          backend_config_(owner_->config()),
          batch_(batch),
          prefix_steps_(prefix_steps),
          seq_len_(seq_len),
          hidden_size_(weights.hidden_size),
          speech_vocab_(weights.speech_vocab),
          num_heads_(weights.num_heads),
          head_dim_(weights.hidden_size / weights.num_heads) {
        if (batch_ <= 0 || prefix_steps_ < 0 || seq_len_ <= 0) {
            throw std::runtime_error("T3 prefill runner requires positive batch/seq_len and non-negative prefix");
        }

        ggml_init_params params = {};
        params.mem_size = 768ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for T3 prefill runner");
        }

        core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "t3_prefill_runner";

        try {
            input_hidden_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_, seq_len_, hidden_size_}));
            positions_ = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({seq_len_}));

            if (prefix_steps_ > 0) {
                prefix_key_tensors_.reserve(weights.layers.size());
                prefix_value_tensors_.reserve(weights.layers.size());
                for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                    prefix_key_tensors_.push_back(core::make_tensor(
                        ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({batch_, prefix_steps_, num_heads_, head_dim_})));
                    prefix_value_tensors_.push_back(core::make_tensor(
                        ctx,
                        GGML_TYPE_F32,
                        core::TensorShape::from_dims({batch_, prefix_steps_, num_heads_, head_dim_})));
                }
            }

            auto hidden = input_hidden_;
            current_keys_.reserve(weights.layers.size());
            current_values_.reserve(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                const std::optional<core::TensorValue> prefix_key = prefix_steps_ > 0
                    ? std::optional<core::TensorValue>(prefix_key_tensors_[layer_index])
                    : std::nullopt;
                const std::optional<core::TensorValue> prefix_value = prefix_steps_ > 0
                    ? std::optional<core::TensorValue>(prefix_value_tensors_[layer_index])
                    : std::nullopt;
                auto layer_outputs = build_t3_layer_backend(
                    ctx,
                    hidden,
                    positions_,
                    weights.rope_factors_tensor,
                    weights.layers[layer_index],
                    num_heads_,
                    prefix_key,
                    prefix_value);
                hidden = layer_outputs.hidden;
                current_keys_.push_back(layer_outputs.key);
                current_values_.push_back(layer_outputs.value);
            }

            hidden_out_ = modules::RMSNormModule({hidden_size_, 1.0e-5f, true, false}).build(
                ctx,
                hidden,
                {weights.tfmr_norm_tensor, std::nullopt});
            const auto speech_head_w = make_graph_param_tensor(weights.speech_head_weight);
            logits_out_ = modules::LinearModule({hidden_size_, speech_vocab_, false}).build(
                ctx,
                hidden_out_,
                {speech_head_w, std::nullopt});

            graph_ = ggml_new_graph_custom(ggml_, 131072, false);
            ggml_build_forward_expand(graph_, logits_out_.tensor);
            for (const auto & key : current_keys_) {
                ggml_build_forward_expand(graph_, key.tensor);
            }
            for (const auto & value : current_values_) {
                ggml_build_forward_expand(graph_, value.tensor);
            }
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, owner_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for T3 prefill runner");
            }
            engine::core::prepare_host_graph_plan(owner_->execution_context(), graph_, cpu_plan_);
        } catch (...) {
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
                buffer_ = nullptr;
            }
            if (ggml_ != nullptr) {
                ggml_free(ggml_);
                ggml_ = nullptr;
            }
            throw;
        }

        std::vector<int32_t> positions(static_cast<size_t>(seq_len_), 0);
        for (int64_t i = 0; i < seq_len_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(prefix_steps_ + i);
        }
        core::write_tensor_i32(positions_, positions);
        prefix_scratch_.resize(static_cast<size_t>(batch_ * prefix_steps_ * hidden_size_), 0.0f);
    }

    ~T3PrefillBackendRunner() {
        if (owner_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(owner_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
    }

    bool matches(
        int64_t batch,
        int64_t prefix_steps,
        int64_t seq_len,
        const engine::core::BackendConfig & backend_config) const {
        return batch_ == batch &&
            prefix_steps_ == prefix_steps &&
            seq_len_ == seq_len &&
            same_backend(backend_config_, backend_config);
    }

    T3PrefillBackendOutput run(const std::vector<float> & input_hidden, const T3BackendCacheState & prefix_state) {
        if (static_cast<int64_t>(input_hidden.size()) != batch_ * seq_len_ * hidden_size_) {
            throw std::runtime_error("T3 prefill runner input size mismatch");
        }
        if (prefix_state.batch != batch_ ||
            prefix_state.hidden_size != hidden_size_ ||
            prefix_state.num_heads != num_heads_ ||
            prefix_state.head_dim != head_dim_ ||
            prefix_state.steps != prefix_steps_) {
            throw std::runtime_error("T3 prefill runner prefix cache shape mismatch");
        }
        if (prefix_state.layers.size() != current_keys_.size()) {
            throw std::runtime_error("T3 prefill runner prefix layer count mismatch");
        }

        core::write_tensor_f32(input_hidden_, input_hidden);
        if (prefix_steps_ > 0) {
            for (size_t layer_index = 0; layer_index < prefix_state.layers.size(); ++layer_index) {
                write_prefix_tensor(prefix_key_tensors_[layer_index], prefix_state.layers[layer_index].key);
                write_prefix_tensor(prefix_value_tensors_[layer_index], prefix_state.layers[layer_index].value);
            }
        }

        const ggml_status status = engine::core::compute_graph(owner_->execution_context(), graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml compute failed for T3 prefill runner");
        }

        T3PrefillBackendOutput out;
        const auto logits = core::read_tensor_f32(logits_out_.tensor);
        out.logits.resize(static_cast<size_t>(batch_ * speech_vocab_), 0.0f);
        for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
            const size_t src = static_cast<size_t>((batch_index * seq_len_ + (seq_len_ - 1)) * speech_vocab_);
            const size_t dst = static_cast<size_t>(batch_index * speech_vocab_);
            std::copy_n(
                logits.data() + static_cast<std::ptrdiff_t>(src),
                speech_vocab_,
                out.logits.data() + static_cast<std::ptrdiff_t>(dst));
        }

        out.cache.batch = batch_;
        out.cache.hidden_size = hidden_size_;
        out.cache.num_heads = num_heads_;
        out.cache.head_dim = head_dim_;
        out.cache.steps = prefix_steps_ + seq_len_;
        out.cache.layers.resize(current_keys_.size());
        for (size_t layer_index = 0; layer_index < current_keys_.size(); ++layer_index) {
            const auto dynamic_key = core::read_tensor_f32(current_keys_[layer_index].tensor);
            const auto dynamic_value = core::read_tensor_f32(current_values_[layer_index].tensor);
            merge_layer_state(
                prefix_state.layers[layer_index].key,
                dynamic_key,
                out.cache.layers[layer_index].key);
            merge_layer_state(
                prefix_state.layers[layer_index].value,
                dynamic_value,
                out.cache.layers[layer_index].value);
        }
        return out;
    }

private:
    void write_prefix_tensor(const core::TensorValue & tensor, const std::vector<float> & state_values) {
        const size_t expected = static_cast<size_t>(prefix_steps_ * batch_ * hidden_size_);
        if (state_values.size() != expected) {
            throw std::runtime_error("T3 prefill runner prefix state size mismatch");
        }
        for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
            for (int64_t step = 0; step < prefix_steps_; ++step) {
                const size_t state_offset = static_cast<size_t>((step * batch_ + batch_index) * hidden_size_);
                const size_t dst_offset = static_cast<size_t>((batch_index * prefix_steps_ + step) * hidden_size_);
                std::copy_n(
                    state_values.data() + static_cast<std::ptrdiff_t>(state_offset),
                    hidden_size_,
                    prefix_scratch_.data() + static_cast<std::ptrdiff_t>(dst_offset));
            }
        }
        core::write_tensor_f32(tensor, prefix_scratch_);
    }

    void merge_layer_state(
        const std::vector<float> & prefix_values,
        const std::vector<float> & dynamic_values,
        std::vector<float> & merged) const {
        const int64_t total_steps = prefix_steps_ + seq_len_;
        merged.assign(static_cast<size_t>(total_steps * batch_ * hidden_size_), 0.0f);
        if (prefix_steps_ > 0) {
            const size_t expected_prefix = static_cast<size_t>(prefix_steps_ * batch_ * hidden_size_);
            if (prefix_values.size() != expected_prefix) {
                throw std::runtime_error("T3 prefill runner prefix merge size mismatch");
            }
            std::copy(prefix_values.begin(), prefix_values.end(), merged.begin());
        }
        const size_t expected_dynamic = static_cast<size_t>(batch_ * seq_len_ * hidden_size_);
        if (dynamic_values.size() != expected_dynamic) {
            throw std::runtime_error("T3 prefill runner dynamic merge size mismatch");
        }
        for (int64_t batch_index = 0; batch_index < batch_; ++batch_index) {
            for (int64_t step = 0; step < seq_len_; ++step) {
                const size_t src_offset = static_cast<size_t>((batch_index * seq_len_ + step) * hidden_size_);
                const size_t dst_offset = static_cast<size_t>(((prefix_steps_ + step) * batch_ + batch_index) * hidden_size_);
                std::copy_n(
                    dynamic_values.data() + static_cast<std::ptrdiff_t>(src_offset),
                    hidden_size_,
                    merged.data() + static_cast<std::ptrdiff_t>(dst_offset));
            }
        }
    }

    std::shared_ptr<T3DecodeBackendOwner> owner_;
    engine::core::BackendConfig backend_config_;
    int64_t batch_ = 0;
    int64_t prefix_steps_ = 0;
    int64_t seq_len_ = 0;
    int64_t hidden_size_ = 0;
    int64_t speech_vocab_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::HostGraphPlan cpu_plan_;
    core::TensorValue input_hidden_;
    core::TensorValue positions_;
    core::TensorValue hidden_out_;
    core::TensorValue logits_out_;
    std::vector<core::TensorValue> prefix_key_tensors_;
    std::vector<core::TensorValue> prefix_value_tensors_;
    std::vector<core::TensorValue> current_keys_;
    std::vector<core::TensorValue> current_values_;
    std::vector<float> prefix_scratch_;
};

void apply_repetition_penalty_in_place(
    std::vector<float> & logits,
    const std::vector<int32_t> & generated_ids,
    float penalty,
    std::vector<uint8_t> & seen) {
    if (penalty == 1.0f) {
        return;
    }
    seen.assign(logits.size(), 0);
    for (int32_t token : generated_ids) {
        if (token < 0 || token >= static_cast<int32_t>(logits.size())) {
            continue;
        }
        if (seen[static_cast<size_t>(token)] != 0) {
            continue;
        }
        seen[static_cast<size_t>(token)] = 1;
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0f ? value * penalty : value / penalty;
    }
}

void apply_min_p_in_place(
    std::vector<float> & logits,
    float min_p,
    std::vector<float> & probs) {
    if (min_p <= 0.0f) {
        return;
    }
    softmax_into(logits, probs);
    size_t max_index = 0;
    float max_prob = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_index = i;
        }
    }
    const float cutoff = max_prob * min_p;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (probs[i] < cutoff && i != max_index) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

void apply_top_p_in_place(
    std::vector<float> & logits,
    float top_p,
    std::vector<float> & probs,
    std::vector<size_t> & order,
    std::vector<uint8_t> & remove) {
    if (top_p >= 1.0f) {
        return;
    }
    softmax_into(logits, probs);
    order.resize(probs.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return probs[a] > probs[b];
    });
    remove.assign(probs.size(), 0);
    float cumulative = 0.0f;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const size_t index = *it;
        cumulative += probs[index];
        if (cumulative <= (1.0f - top_p)) {
            remove[index] = 1;
        }
    }
    if (!order.empty()) {
        remove[order[0]] = 0;
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        if (remove[i] != 0) {
            logits[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

class TorchMt19937 {
public:
    explicit TorchMt19937(uint64_t seed)
        : state_{},
          left_(1),
          next_(0) {
        seed = components::choose_seed(seed);
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffU);
        for (size_t index = 1; index < state_.size(); ++index) {
            state_[index] = static_cast<uint32_t>(
                1812433253U * (state_[index - 1] ^ (state_[index - 1] >> 30U)) + static_cast<uint32_t>(index));
        }
    }

    uint32_t random() {
        if (--left_ == 0) {
            next_state();
        }
        uint32_t value = state_[next_++];
        value ^= (value >> 11U);
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= (value >> 18U);
        return value;
    }

    uint64_t random64() {
        const uint64_t high = static_cast<uint64_t>(random());
        const uint64_t low = static_cast<uint64_t>(random());
        return (high << 32U) | low;
    }

private:
    static constexpr size_t kStateSize = 624;
    static constexpr size_t kStateM = 397;
    static constexpr uint32_t kMatrixA = 0x9908b0dfU;
    static constexpr uint32_t kUpperMask = 0x80000000U;
    static constexpr uint32_t kLowerMask = 0x7fffffffU;

    static uint32_t mix_bits(uint32_t first, uint32_t second) {
        return (first & kUpperMask) | (second & kLowerMask);
    }

    static uint32_t twist(uint32_t first, uint32_t second) {
        return (mix_bits(first, second) >> 1U) ^ ((second & 1U) ? kMatrixA : 0U);
    }

    void next_state() {
        size_t offset = 0;
        left_ = static_cast<int>(kStateSize);
        next_ = 0;

        for (size_t count = 0; count < (kStateSize - kStateM); ++count, ++offset) {
            state_[offset] = state_[offset + kStateM] ^ twist(state_[offset], state_[offset + 1]);
        }
        for (size_t count = 0; count < (kStateM - 1); ++count, ++offset) {
            state_[offset] = state_[offset + kStateM - kStateSize] ^ twist(state_[offset], state_[offset + 1]);
        }
        state_[offset] = state_[offset + kStateM - kStateSize] ^ twist(state_[offset], state_[0]);
    }

    std::array<uint32_t, kStateSize> state_;
    int left_;
    size_t next_;
};

double torch_uniform_double(TorchMt19937 & rng) {
    constexpr uint64_t kMask = (static_cast<uint64_t>(1) << std::numeric_limits<double>::digits) - 1U;
    constexpr double kDivisor =
        1.0 / static_cast<double>(static_cast<uint64_t>(1) << std::numeric_limits<double>::digits);
    return static_cast<double>(rng.random64() & kMask) * kDivisor;
}

int32_t sample_from_logits(
    const std::vector<float> & logits,
    bool do_sample,
    TorchMt19937 & rng) {
    if (!do_sample) {
        return static_cast<int32_t>(std::distance(
            logits.begin(),
            std::max_element(logits.begin(), logits.end())));
    }
    const auto probs = softmax(logits);
    const double draw = torch_uniform_double(rng);
    double cumulative = 0.0;
    for (size_t index = 0; index < probs.size(); ++index) {
        cumulative += static_cast<double>(probs[index]);
        if (draw < cumulative) {
            return static_cast<int32_t>(index);
        }
    }
    return static_cast<int32_t>(probs.empty() ? 0 : (probs.size() - 1));
}

}  // namespace

}  // namespace engine::models::chatterbox

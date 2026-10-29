#pragma once

#include "engine/community_models/chatterbox_turbo/t3_turbo_component.h"

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
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace engine::community_models::chatterbox_turbo {
namespace {

constexpr int32_t kTurboStartSpeechToken = 6561;
constexpr int32_t kTurboStopSpeechToken = 6562;
constexpr int32_t kTurboMaxSpeechToken = 6560;  // tokens >= kTurboStartSpeechToken are control tokens

std::vector<float> gather_rows(
    const std::vector<float> & table,
    int64_t rows,
    int64_t cols,
    const std::vector<int32_t> & indices) {
    std::vector<float> out(static_cast<size_t>(indices.size() * static_cast<size_t>(cols)), 0.0f);
    for (size_t i = 0; i < indices.size(); ++i) {
        const int32_t index = indices[i];
        if (index < 0 || index >= rows) {
            throw std::runtime_error("chatterbox_turbo embedding index out of range");
        }
        const float * src = table.data() + static_cast<ptrdiff_t>(static_cast<int64_t>(index) * cols);
        float * dst = out.data() + static_cast<ptrdiff_t>(i * static_cast<size_t>(cols));
        std::copy(src, src + cols, dst);
    }
    return out;
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

void apply_top_k_in_place(std::vector<float> & logits, int64_t top_k) {
    if (top_k <= 0 || top_k >= static_cast<int64_t>(logits.size())) {
        return;
    }
    std::vector<float> sorted = logits;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<ptrdiff_t>(top_k - 1), sorted.end(), std::greater<float>());
    const float threshold = sorted[static_cast<size_t>(top_k - 1)];
    for (float & value : logits) {
        if (value < threshold) {
            value = -std::numeric_limits<float>::infinity();
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

class TurboMt19937 {
public:
    explicit TurboMt19937(uint32_t seed)
        : state_{}, left_(1), next_(0) {
        state_[0] = seed;
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

double uniform_double(TurboMt19937 & rng) {
    constexpr uint64_t kMask = (static_cast<uint64_t>(1) << std::numeric_limits<double>::digits) - 1U;
    constexpr double kDivisor =
        1.0 / static_cast<double>(static_cast<uint64_t>(1) << std::numeric_limits<double>::digits);
    return static_cast<double>(rng.random64() & kMask) * kDivisor;
}

int32_t sample_from_logits(const std::vector<float> & logits, bool do_sample, TurboMt19937 & rng) {
    if (!do_sample) {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    const auto probs = softmax(logits);
    const double draw = uniform_double(rng);
    double cumulative = 0.0;
    for (size_t index = 0; index < probs.size(); ++index) {
        cumulative += static_cast<double>(probs[index]);
        if (draw < cumulative) {
            return static_cast<int32_t>(index);
        }
    }
    return static_cast<int32_t>(probs.empty() ? 0 : (probs.size() - 1));
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue make_graph_param_tensor(const T3TurboGraphWeight & weight) {
    return weight.tensor;
}

// GPT2 pre-LN transformer block. hidden_size/num_heads/head_dim/mlp are read from the layer's
// tensor shapes at call sites; no RoPE, no position tensor (positions are summed into the
// input embeddings host-side before this graph runs, matching how GPT2's own wpe is applied
// once to combined inputs_embeds rather than per attention layer).
struct T3TurboLayerOutput {
    core::TensorValue hidden;
    core::TensorValue key;
    core::TensorValue value;
};

core::TensorValue turbo_gpt_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t hidden_size,
    int64_t intermediate_size,
    const T3TurboInferenceWeights::TransformerLayer & layer) {
    auto hidden = modules::LinearModule({hidden_size, intermediate_size, true}).build(
        ctx, input, {make_graph_param_tensor(layer.ffn_fc_weight), layer.ffn_fc_bias_tensor});
    hidden = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, hidden);
    return modules::LinearModule({intermediate_size, hidden_size, true}).build(
        ctx, hidden, {make_graph_param_tensor(layer.ffn_proj_weight), layer.ffn_proj_bias_tensor});
}

core::TensorValue concat_along_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs,
    int logical_axis) {
    auto output_shape = lhs.shape;
    output_shape.dims[logical_axis] += rhs.shape.dims[logical_axis];
    return core::wrap_tensor(
        ggml_concat(ctx.ggml, lhs.tensor, rhs.tensor, core::logical_axis_to_ggml_axis(lhs.shape.rank, logical_axis)),
        output_shape,
        lhs.type);
}

// prefix_key/prefix_value (if present) hold the already-computed cond-prefix K/V for this layer,
// so the dynamic [text; speech] window here can attend over the full [cond; text; speech]
// history even though it is a fresh graph with no KV cache of its own.
T3TurboLayerOutput build_t3_turbo_layer_full(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const T3TurboInferenceWeights::TransformerLayer & layer,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_heads,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt) {
    const int64_t head_dim = hidden_size / num_heads;
    auto normed = modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(
        ctx, input, {layer.ln1_weight_tensor, layer.ln1_bias_tensor});
    auto qkv = modules::LinearModule({hidden_size, 3 * hidden_size, true}).build(
        ctx, normed, {make_graph_param_tensor(layer.attn_qkv_weight), layer.attn_qkv_bias_tensor});
    auto q = modules::SliceModule({2, 0, hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({2, hidden_size, hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * hidden_size, hidden_size}).build(ctx, qkv);

    auto reshape_heads = [&](const core::TensorValue & value) {
        return core::reshape_tensor(
            ctx, contiguous(ctx, value), core::TensorShape::from_dims({value.shape.dims[0], value.shape.dims[1], num_heads, head_dim}));
    };
    auto k_cache = reshape_heads(k);
    auto v_cache = reshape_heads(v);

    auto all_k = prefix_key.has_value() ? concat_along_axis(ctx, contiguous(ctx, *prefix_key), k_cache, 1) : k_cache;
    auto all_v = prefix_value.has_value() ? concat_along_axis(ctx, contiguous(ctx, *prefix_value), v_cache, 1) : v_cache;

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(q));
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_v);

    auto scores = modules::MatMulModule{}.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k_heads));
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0f / std::sqrt(static_cast<float>(head_dim))), scores.shape, GGML_TYPE_F32);
    const int n_past = prefix_key.has_value() ? static_cast<int>(prefix_key->shape.dims[1]) : 0;
    scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, n_past), scores.shape, GGML_TYPE_F32);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = modules::MatMulModule{}.build(ctx, attn, v_heads);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(ctx, contiguous(ctx, context), input.shape);
    context = modules::LinearModule({hidden_size, hidden_size, true}).build(
        ctx, context, {make_graph_param_tensor(layer.attn_output_weight), layer.attn_output_bias_tensor});
    auto hidden = core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, context).tensor, contiguous(ctx, input).tensor), input.shape, GGML_TYPE_F32);

    auto mlp_in = modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(
        ctx, hidden, {layer.ln2_weight_tensor, layer.ln2_bias_tensor});
    auto mlp_out = turbo_gpt_mlp(ctx, mlp_in, hidden_size, intermediate_size, layer);
    auto output = core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, mlp_out).tensor, contiguous(ctx, hidden).tensor), hidden.shape, GGML_TYPE_F32);
    return {output, k_cache, v_cache};
}

T3TurboLayerOutput build_t3_turbo_layer_cached(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const T3TurboInferenceWeights::TransformerLayer & layer,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_heads,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t head_dim = hidden_size / num_heads;
    auto normed = modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(
        ctx, input, {layer.ln1_weight_tensor, layer.ln1_bias_tensor});
    auto qkv = modules::LinearModule({hidden_size, 3 * hidden_size, true}).build(
        ctx, normed, {make_graph_param_tensor(layer.attn_qkv_weight), layer.attn_qkv_bias_tensor});
    auto q = modules::SliceModule({2, 0, hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({2, hidden_size, hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * hidden_size, hidden_size}).build(ctx, qkv);
    auto reshape_heads = [&](const core::TensorValue & value) {
        return core::reshape_tensor(
            ctx, contiguous(ctx, value), core::TensorShape::from_dims({value.shape.dims[0], value.shape.dims[1], num_heads, head_dim}));
    };
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, reshape_heads(q));
    k = reshape_heads(k);
    v = reshape_heads(v);

    const modules::FastKVSetRowsModule set_rows;
    auto updated_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_value = set_rows.build(ctx, cache_value, v, cache_slot);

    q_heads = contiguous(ctx, q_heads);
    auto k_heads = contiguous(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_key));
    auto v_heads = contiguous(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, updated_value));

    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        k_heads.tensor,
        v_heads.tensor,
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
    context = core::reshape_tensor(ctx, context, input.shape);
    context = modules::LinearModule({hidden_size, hidden_size, true}).build(
        ctx, context, {make_graph_param_tensor(layer.attn_output_weight), layer.attn_output_bias_tensor});
    auto hidden = core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, context).tensor, contiguous(ctx, input).tensor), input.shape, GGML_TYPE_F32);

    auto mlp_in = modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(
        ctx, hidden, {layer.ln2_weight_tensor, layer.ln2_bias_tensor});
    auto mlp_out = turbo_gpt_mlp(ctx, mlp_in, hidden_size, intermediate_size, layer);
    auto output = core::wrap_tensor(
        ggml_add(ctx.ggml, contiguous(ctx, mlp_out).tensor, contiguous(ctx, hidden).tensor), hidden.shape, GGML_TYPE_F32);
    return {output, k, v};
}

inline bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

class T3TurboBackendOwner {
public:
    T3TurboBackendOwner(const T3TurboInferenceWeights & weights, const engine::core::BackendConfig & config)
        : config_(config), execution_context_(weights.execution_context) {
        if (!execution_context_) {
            throw std::runtime_error("T3 Turbo backend owner requires loaded backend weights");
        }
    }
    ggml_backend_t backend() const noexcept { return execution_context_->backend(); }
    const engine::core::ExecutionContext & execution_context() const noexcept { return *execution_context_; }
    const engine::core::BackendConfig & config() const noexcept { return config_; }

private:
    engine::core::BackendConfig config_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
};

struct T3TurboLayerCacheState {
    std::vector<float> key;
    std::vector<float> value;
};

struct T3TurboCacheState {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t steps = 0;
    std::vector<T3TurboLayerCacheState> layers;
};

// Batch is always 1: Chatterbox Turbo never uses classifier-free guidance, so there is no
// unconditional duplicate branch the way base Chatterbox's T3 decode runners need.
class T3TurboDecodeBackendRunner {
public:
    T3TurboDecodeBackendRunner(
        const T3TurboInferenceWeights & weights,
        int64_t cache_steps,
        std::shared_ptr<T3TurboBackendOwner> owner)
        : owner_(std::move(owner)),
          backend_config_(owner_->config()),
          cache_steps_(cache_steps),
          hidden_size_(weights.hidden_size),
          num_heads_(weights.num_heads),
          head_dim_(weights.hidden_size / weights.num_heads) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("T3TurboDecodeBackendRunner requires positive cache_steps");
        }
        ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for T3 Turbo decode runner");
        }
        core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "t3_turbo_decode_runner";
        try {
            input_hidden_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, hidden_size_}));
            attention_mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps_}));
            cache_slot_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));

            key_cache_tensors_.reserve(weights.layers.size());
            value_cache_tensors_.reserve(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                key_cache_tensors_.push_back(core::make_tensor(
                    ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, cache_steps_, num_heads_, head_dim_})));
                value_cache_tensors_.push_back(core::make_tensor(
                    ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, cache_steps_, num_heads_, head_dim_})));
            }

            auto hidden = input_hidden_;
            current_keys_.reserve(weights.layers.size());
            current_values_.reserve(weights.layers.size());
            current_key_scratch_.resize(weights.layers.size());
            current_value_scratch_.resize(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                auto out = build_t3_turbo_layer_cached(
                    ctx,
                    hidden,
                    weights.layers[layer_index],
                    hidden_size_,
                    weights.mlp_intermediate_size,
                    num_heads_,
                    key_cache_tensors_[layer_index],
                    value_cache_tensors_[layer_index],
                    cache_slot_,
                    attention_mask_);
                hidden = out.hidden;
                current_keys_.push_back(out.key);
                current_values_.push_back(out.value);
            }
            hidden_out_ = modules::LayerNormModule({hidden_size_, 1.0e-5f, true, true}).build(
                ctx, hidden, {weights.ln_f_weight_tensor, weights.ln_f_bias_tensor});
            logits_out_ = modules::LinearModule({hidden_size_, weights.speech_vocab, true}).build(
                ctx, hidden_out_, {make_graph_param_tensor(weights.speech_head_weight), weights.speech_head_bias_tensor});

            graph_ = ggml_new_graph_custom(ggml_, 65536, false);
            ggml_build_forward_expand(graph_, logits_out_.tensor);
            for (const auto & key : current_keys_) {
                ggml_build_forward_expand(graph_, key.tensor);
            }
            for (const auto & value : current_values_) {
                ggml_build_forward_expand(graph_, value.tensor);
            }
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, owner_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for T3 Turbo decode runner");
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
        cache_state_.hidden_size = hidden_size_;
        cache_state_.num_heads = num_heads_;
        cache_state_.head_dim = head_dim_;
        cache_state_.layers.resize(weights.layers.size());
        attention_mask_scratch_.resize(static_cast<size_t>(cache_steps_), 0.0f);
        import_state(cache_state_);
    }

    ~T3TurboDecodeBackendRunner() {
        if (owner_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(owner_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    bool matches(int64_t cache_steps, const engine::core::BackendConfig & backend_config) const {
        return cache_steps_ == cache_steps && same_backend(backend_config_, backend_config);
    }

    void import_state(const T3TurboCacheState & state) {
        if (state.layers.size() != cache_state_.layers.size()) {
            throw std::runtime_error("T3TurboDecodeBackendRunner state layer count mismatch");
        }
        if (state.steps > cache_steps_) {
            throw std::runtime_error("T3TurboDecodeBackendRunner state exceeds cache capacity");
        }
        cache_state_ = state;
        cache_state_.hidden_size = hidden_size_;
        cache_state_.num_heads = num_heads_;
        cache_state_.head_dim = head_dim_;
        for (size_t layer_index = 0; layer_index < cache_state_.layers.size(); ++layer_index) {
            const auto & layer = cache_state_.layers[layer_index];
            if (cache_state_.steps == 0 || layer.key.empty() || layer.value.empty()) {
                continue;
            }
            core::write_tensor_f32_slice(key_cache_tensors_[layer_index], 0, layer.key.data(), layer.key.size());
            core::write_tensor_f32_slice(value_cache_tensors_[layer_index], 0, layer.value.data(), layer.value.size());
        }
    }

    T3TurboCacheState export_state() const { return cache_state_; }

    T3TurboCacheState export_state_from_device() const {
        T3TurboCacheState state = cache_state_;
        state.layers.assign(cache_state_.layers.size(), {});
        if (cache_state_.steps <= 0) {
            return state;
        }
        for (size_t layer_index = 0; layer_index < key_cache_tensors_.size(); ++layer_index) {
            state.layers[layer_index].key = core::read_tensor_f32(key_cache_tensors_[layer_index].tensor);
            state.layers[layer_index].value = core::read_tensor_f32(value_cache_tensors_[layer_index].tensor);
        }
        return state;
    }

    int64_t valid_steps() const noexcept { return cache_state_.steps; }
    int64_t cache_capacity_steps() const noexcept { return cache_steps_; }

    void set_capture_cache_state(bool capture) noexcept { capture_cache_state_ = capture; }

    std::vector<float> step(const std::vector<float> & input_hidden, int64_t /*position*/) {
        if (static_cast<int64_t>(input_hidden.size()) != hidden_size_) {
            throw std::runtime_error("T3TurboDecodeBackendRunner step input size mismatch");
        }
        const int64_t masked_prefix_begin = std::clamp<int64_t>(cache_state_.steps + 1, 0, cache_steps_);
        attention_mask_scratch_.assign(static_cast<size_t>(cache_steps_), 0.0f);
        for (int64_t step = masked_prefix_begin; step < cache_steps_; ++step) {
            attention_mask_scratch_[static_cast<size_t>(step)] = -10000.0f;
        }
        core::write_tensor_f32(input_hidden_, input_hidden);
        core::write_tensor_i32(cache_slot_, std::vector<int32_t>{static_cast<int32_t>(cache_state_.steps)});
        core::write_tensor_f16(attention_mask_, attention_mask_scratch_);

        const ggml_status status = engine::core::compute_graph(owner_->execution_context(), graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml compute failed for T3 Turbo decode runner");
        }
        if (capture_cache_state_) {
            for (size_t layer_index = 0; layer_index < current_keys_.size(); ++layer_index) {
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

private:
    std::shared_ptr<T3TurboBackendOwner> owner_;
    engine::core::BackendConfig backend_config_;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::HostGraphPlan cpu_plan_;
    int64_t cache_steps_ = 0;
    int64_t hidden_size_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    core::TensorValue input_hidden_;
    core::TensorValue cache_slot_;
    core::TensorValue attention_mask_;
    core::TensorValue hidden_out_;
    core::TensorValue logits_out_;
    std::vector<core::TensorValue> key_cache_tensors_;
    std::vector<core::TensorValue> value_cache_tensors_;
    std::vector<core::TensorValue> current_keys_;
    std::vector<core::TensorValue> current_values_;
    T3TurboCacheState cache_state_;
    bool capture_cache_state_ = false;
    std::vector<float> attention_mask_scratch_;
    std::vector<std::vector<float>> current_key_scratch_;
    std::vector<std::vector<float>> current_value_scratch_;
};

struct T3TurboPrefillOutput {
    std::vector<float> logits;
    T3TurboCacheState cache;
};

class T3TurboPrefillBackendRunner {
public:
    T3TurboPrefillBackendRunner(
        const T3TurboInferenceWeights & weights,
        int64_t prefix_steps,
        int64_t seq_len,
        std::shared_ptr<T3TurboBackendOwner> owner)
        : owner_(std::move(owner)),
          backend_config_(owner_->config()),
          prefix_steps_(prefix_steps),
          seq_len_(seq_len),
          hidden_size_(weights.hidden_size),
          speech_vocab_(weights.speech_vocab),
          num_heads_(weights.num_heads),
          head_dim_(weights.hidden_size / weights.num_heads) {
        if (seq_len_ <= 0 || prefix_steps_ < 0) {
            throw std::runtime_error("T3 Turbo prefill runner requires positive seq_len and non-negative prefix");
        }
        ggml_init_params params = {};
        params.mem_size = 768ull * 1024ull * 1024ull;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for T3 Turbo prefill runner");
        }
        core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "t3_turbo_prefill_runner";
        try {
            input_hidden_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, seq_len_, hidden_size_}));
            if (prefix_steps_ > 0) {
                prefix_key_tensors_.reserve(weights.layers.size());
                prefix_value_tensors_.reserve(weights.layers.size());
                for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                    prefix_key_tensors_.push_back(core::make_tensor(
                        ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, prefix_steps_, num_heads_, head_dim_})));
                    prefix_value_tensors_.push_back(core::make_tensor(
                        ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, prefix_steps_, num_heads_, head_dim_})));
                }
            }
            auto hidden = input_hidden_;
            current_keys_.reserve(weights.layers.size());
            current_values_.reserve(weights.layers.size());
            for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
                const std::optional<core::TensorValue> prefix_key =
                    prefix_steps_ > 0 ? std::optional<core::TensorValue>(prefix_key_tensors_[layer_index]) : std::nullopt;
                const std::optional<core::TensorValue> prefix_value =
                    prefix_steps_ > 0 ? std::optional<core::TensorValue>(prefix_value_tensors_[layer_index]) : std::nullopt;
                auto out = build_t3_turbo_layer_full(
                    ctx,
                    hidden,
                    weights.layers[layer_index],
                    hidden_size_,
                    weights.mlp_intermediate_size,
                    num_heads_,
                    prefix_key,
                    prefix_value);
                hidden = out.hidden;
                current_keys_.push_back(out.key);
                current_values_.push_back(out.value);
            }
            hidden_out_ = modules::LayerNormModule({hidden_size_, 1.0e-5f, true, true}).build(
                ctx, hidden, {weights.ln_f_weight_tensor, weights.ln_f_bias_tensor});
            logits_out_ = modules::LinearModule({hidden_size_, speech_vocab_, true}).build(
                ctx, hidden_out_, {make_graph_param_tensor(weights.speech_head_weight), weights.speech_head_bias_tensor});

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
                throw std::runtime_error("failed to allocate backend tensors for T3 Turbo prefill runner");
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
    }

    ~T3TurboPrefillBackendRunner() {
        if (owner_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(owner_->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    bool matches(int64_t prefix_steps, int64_t seq_len, const engine::core::BackendConfig & backend_config) const {
        return prefix_steps_ == prefix_steps && seq_len_ == seq_len && same_backend(backend_config_, backend_config);
    }

    T3TurboPrefillOutput run(const std::vector<float> & input_hidden, const T3TurboCacheState & prefix_state) {
        if (static_cast<int64_t>(input_hidden.size()) != seq_len_ * hidden_size_) {
            throw std::runtime_error("T3 Turbo prefill runner input size mismatch");
        }
        if (prefix_state.steps != prefix_steps_ || prefix_state.layers.size() != current_keys_.size()) {
            throw std::runtime_error("T3 Turbo prefill runner prefix cache shape mismatch");
        }
        core::write_tensor_f32(input_hidden_, input_hidden);
        for (size_t layer_index = 0; layer_index < prefix_key_tensors_.size(); ++layer_index) {
            core::write_tensor_f32(prefix_key_tensors_[layer_index], prefix_state.layers[layer_index].key);
            core::write_tensor_f32(prefix_value_tensors_[layer_index], prefix_state.layers[layer_index].value);
        }
        const ggml_status status = engine::core::compute_graph(owner_->execution_context(), graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml compute failed for T3 Turbo prefill runner");
        }
        T3TurboPrefillOutput out;
        const auto logits = core::read_tensor_f32(logits_out_.tensor);
        out.logits.assign(logits.end() - speech_vocab_, logits.end());
        out.cache.hidden_size = hidden_size_;
        out.cache.num_heads = num_heads_;
        out.cache.head_dim = head_dim_;
        out.cache.steps = prefix_steps_ + seq_len_;
        out.cache.layers.resize(current_keys_.size());
        for (size_t layer_index = 0; layer_index < current_keys_.size(); ++layer_index) {
            auto & layer = out.cache.layers[layer_index];
            if (prefix_steps_ > 0) {
                layer.key = prefix_state.layers[layer_index].key;
                layer.value = prefix_state.layers[layer_index].value;
            }
            const auto dyn_key = core::read_tensor_f32(current_keys_[layer_index].tensor);
            const auto dyn_value = core::read_tensor_f32(current_values_[layer_index].tensor);
            layer.key.insert(layer.key.end(), dyn_key.begin(), dyn_key.end());
            layer.value.insert(layer.value.end(), dyn_value.begin(), dyn_value.end());
        }
        return out;
    }

private:
    std::shared_ptr<T3TurboBackendOwner> owner_;
    engine::core::BackendConfig backend_config_;
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
    core::TensorValue hidden_out_;
    core::TensorValue logits_out_;
    std::vector<core::TensorValue> prefix_key_tensors_;
    std::vector<core::TensorValue> prefix_value_tensors_;
    std::vector<core::TensorValue> current_keys_;
    std::vector<core::TensorValue> current_values_;
};

}  // namespace
}  // namespace engine::community_models::chatterbox_turbo

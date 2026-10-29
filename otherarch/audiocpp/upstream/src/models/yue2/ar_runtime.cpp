#include "engine/models/yue2/ar_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace engine::models::yue2 {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

constexpr int64_t kArDecodeChunkTokens = 5120;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct Yue2SamplerRange {
    int64_t begin = 0;
    int64_t end = 0;
};

struct Yue2SamplerScratch {
    std::vector<int32_t> candidates;
    std::vector<int32_t> kept;
    std::vector<double> weights;
};

engine::modules::DecoderLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Yue2ModelConfig & config,
    assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    engine::modules::DecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.attention_heads * config.head_dim, config.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.kv_heads * config.head_dim, config.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.attention_heads * config.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    return out;
}

engine::modules::CausalDecoderRuntimeWeights load_prefix_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Yue2ModelConfig & config,
    assets::TensorStorageType storage_type) {
    engine::modules::CausalDecoderRuntimeWeights weights;
    weights.token_embedding = store.load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.stack.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.stack.layers.push_back(load_layer(store, source, config, storage_type, layer));
    }
    return weights;
}

void load_generation_weights(
    engine::modules::CausalDecoderRuntimeWeights & weights,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Yue2ModelConfig & config,
    assets::TensorStorageType storage_type) {
    weights.final_norm = binding::norm_weight_from_source(store, source, "model.norm", config.hidden_size);
    weights.lm_head = binding::linear_from_source(
        store,
        source,
        "lm_head",
        storage_type,
        config.vocab_size,
        config.hidden_size,
        false);
}

engine::modules::CausalDecoderRuntimeConfig make_runtime_config(
    const Yue2ModelConfig & config,
    core::BackendType backend_type,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    int64_t logits_size = 0) {
    engine::modules::CausalDecoderRuntimeConfig out;
    out.trace_name = "yue2.ar";
    out.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
    out.decode_graph_arena_bytes = decode_graph_arena_bytes;
    out.decoder.stack.hidden_size = config.hidden_size;
    out.decoder.stack.layers = config.layers;
    out.decoder.stack.num_attention_heads = config.attention_heads;
    out.decoder.stack.num_key_value_heads = config.kv_heads;
    out.decoder.stack.head_dim = config.head_dim;
    out.decoder.stack.intermediate_size = config.intermediate_size;
    out.decoder.stack.rms_norm_eps = config.rms_norm_eps;
    out.decoder.stack.rope_theta = config.rope_theta;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = true;
    out.decoder.stack.runtime.attention.prefill_mode = engine::modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = engine::modules::DecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = engine::modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode =
        engine::modules::DecoderStaticCacheSetRowsMode::BackendViewOptimized;
    if (backend_type == core::BackendType::Cuda || backend_type == core::BackendType::Hip ||
        backend_type == core::BackendType::Vulkan) {
        out.decoder.static_cache_type = GGML_TYPE_F16;
    }
    out.decoder.logits_size = logits_size > 0 ? logits_size : config.vocab_size;
    out.decoder.logits_mode = engine::modules::CausalDecoderLogitsMode::LastStep;
    out.readback_round_type = GGML_TYPE_BF16;
    return out;
}

runtime::TransformerBatchedKVState make_cfg_batched_state(
    const runtime::TransformerKVState & positive,
    const runtime::TransformerKVState & negative) {
    if (positive.layers.size() != negative.layers.size()) {
        throw std::runtime_error("Yue2 CFG batched state layer count mismatch");
    }
    const int64_t positive_steps = positive.layers.empty() ? 0 : positive.layers.front().valid_steps;
    const int64_t negative_steps = negative.layers.empty() ? 0 : negative.layers.front().valid_steps;
    const int64_t max_steps = std::max(positive_steps, negative_steps);
    if (positive_steps <= 0 || negative_steps <= 0 || max_steps <= 0) {
        throw std::runtime_error("Yue2 CFG batched state requires non-empty prefix states");
    }
    runtime::TransformerBatchedKVState out;
    out.batch_size = 2;
    out.current_end = std::max(positive.current_end, negative.current_end);
    out.current_end_by_batch = {positive.current_end, negative.current_end};
    out.valid_steps_by_batch = {positive_steps, negative_steps};
    out.layers.resize(positive.layers.size());
    for (size_t layer = 0; layer < positive.layers.size(); ++layer) {
        const auto & pos = positive.layers[layer];
        const auto & neg = negative.layers[layer];
        if (pos.valid_steps != positive_steps || neg.valid_steps != negative_steps ||
            pos.key.size() != pos.value.size() || neg.key.size() != neg.value.size()) {
            throw std::runtime_error("Yue2 CFG batched state source shape mismatch");
        }
        if (pos.key.size() % static_cast<size_t>(positive_steps) != 0 ||
            neg.key.size() % static_cast<size_t>(negative_steps) != 0) {
            throw std::runtime_error("Yue2 CFG batched state source step shape mismatch");
        }
        const size_t row_elems = pos.key.size() / static_cast<size_t>(positive_steps);
        if (neg.key.size() / static_cast<size_t>(negative_steps) != row_elems) {
            throw std::runtime_error("Yue2 CFG batched state row size mismatch");
        }
        auto & dst = out.layers[layer];
        dst.valid_steps = max_steps;
        dst.key.resize((static_cast<size_t>(positive_steps) + static_cast<size_t>(negative_steps)) * row_elems);
        dst.value.assign(dst.key.size(), 0.0F);
        std::copy(pos.key.begin(), pos.key.end(), dst.key.begin());
        std::copy(pos.value.begin(), pos.value.end(), dst.value.begin());
        std::copy(neg.key.begin(), neg.key.end(), dst.key.begin() + static_cast<std::ptrdiff_t>(pos.key.size()));
        std::copy(neg.value.begin(), neg.value.end(), dst.value.begin() + static_cast<std::ptrdiff_t>(pos.value.size()));
    }
    return out;
}

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & emitted,
    const Yue2ARSamplingWindow & window) {
    const float penalty = window.sampling.repetition_penalty;
    if (penalty == 1.0F || emitted.empty()) {
        return;
    }
    const int64_t begin = std::max<int64_t>(0, static_cast<int64_t>(emitted.size()) - window.sampling.penalty_window);
    for (int64_t i = begin; i < static_cast<int64_t>(emitted.size()); ++i) {
        const int32_t token = emitted[static_cast<size_t>(i)];
        if (token < 0 || token >= static_cast<int32_t>(logits.size())) {
            continue;
        }
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0F ? value * penalty : value / penalty;
    }
}

int32_t sample_from_allowed_ranges(
    std::vector<float> & logits,
    const std::array<Yue2SamplerRange, 2> & ranges,
    const Yue2SamplingConfig & sampling,
    std::mt19937 & rng,
    Yue2SamplerScratch & scratch,
    const char * context) {
    if (!(sampling.temperature > 0.0F) || !std::isfinite(sampling.temperature)) {
        throw std::runtime_error(std::string(context) + " temperature must be finite and positive");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F || !std::isfinite(sampling.top_p)) {
        throw std::runtime_error(std::string(context) + " top_p must be finite and in [0, 1]");
    }
    scratch.kept.clear();
    scratch.candidates.clear();
    const bool bounded_top_k = sampling.top_k > 0;
    if (bounded_top_k) {
        const size_t keep = static_cast<size_t>(std::max<int64_t>(sampling.top_k, 1));
        auto worse_than = [&](int32_t lhs, int32_t rhs) {
            const float lhs_score = logits[static_cast<size_t>(lhs)];
            const float rhs_score = logits[static_cast<size_t>(rhs)];
            if (lhs_score == rhs_score) {
                return lhs > rhs;
            }
            return lhs_score > rhs_score;
        };
        for (const auto & range : ranges) {
            const int64_t begin = std::max<int64_t>(0, range.begin);
            const int64_t end = std::min<int64_t>(range.end, static_cast<int64_t>(logits.size()));
            for (int64_t token = begin; token < end; ++token) {
                const float score = logits[static_cast<size_t>(token)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const auto token_i32 = static_cast<int32_t>(token);
                if (scratch.candidates.size() < keep) {
                    scratch.candidates.push_back(token_i32);
                    std::push_heap(scratch.candidates.begin(), scratch.candidates.end(), worse_than);
                } else {
                    const int32_t worst = scratch.candidates.front();
                    const float worst_score = logits[static_cast<size_t>(worst)];
                    if (score > worst_score || (score == worst_score && token_i32 < worst)) {
                        std::pop_heap(scratch.candidates.begin(), scratch.candidates.end(), worse_than);
                        scratch.candidates.back() = token_i32;
                        std::push_heap(scratch.candidates.begin(), scratch.candidates.end(), worse_than);
                    }
                }
            }
        }
        scratch.kept = scratch.candidates;
    } else {
        for (const auto & range : ranges) {
            const int64_t begin = std::max<int64_t>(0, range.begin);
            const int64_t end = std::min<int64_t>(range.end, static_cast<int64_t>(logits.size()));
            for (int64_t token = begin; token < end; ++token) {
                if (std::isfinite(logits[static_cast<size_t>(token)])) {
                    scratch.kept.push_back(static_cast<int32_t>(token));
                }
            }
        }
    }
    if (scratch.kept.empty()) {
        throw std::runtime_error(std::string(context) + " sampler has no finite logits");
    }

    auto score_at = [&](int32_t token) {
        return logits[static_cast<size_t>(token)] / sampling.temperature;
    };
    if (sampling.top_p < 1.0F) {
        std::sort(scratch.kept.begin(), scratch.kept.end(), [&](int32_t lhs, int32_t rhs) {
            const float lhs_score = score_at(lhs);
            const float rhs_score = score_at(rhs);
            if (lhs_score == rhs_score) {
                return lhs < rhs;
            }
            return lhs_score < rhs_score;
        });
    } else if (bounded_top_k) {
        std::sort(scratch.kept.begin(), scratch.kept.end());
    }
    float max_score = -std::numeric_limits<float>::infinity();
    for (const int32_t token : scratch.kept) {
        max_score = std::max(max_score, score_at(token));
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error(std::string(context) + " sampler max score is invalid");
    }
    scratch.weights.resize(scratch.kept.size());
    double total = 0.0;
    for (size_t index = 0; index < scratch.kept.size(); ++index) {
        scratch.weights[index] = std::exp(static_cast<double>(score_at(scratch.kept[index]) - max_score));
        total += scratch.weights[index];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error(std::string(context) + " probability mass is invalid");
    }

    if (sampling.top_p < 1.0F) {
        double cumulative = 0.0;
        size_t kept_count = 0;
        const double remove_mass = 1.0 - static_cast<double>(sampling.top_p);
        const size_t protected_from = scratch.kept.size() - 1;
        for (size_t index = 0; index < scratch.kept.size(); ++index) {
            cumulative += scratch.weights[index] / total;
            if (index < protected_from && cumulative <= remove_mass) {
                continue;
            }
            scratch.kept[kept_count] = scratch.kept[index];
            scratch.weights[kept_count] = scratch.weights[index];
            ++kept_count;
        }
        scratch.kept.resize(kept_count);
        scratch.weights.resize(kept_count);
    }

    std::discrete_distribution<size_t> distribution(scratch.weights.begin(), scratch.weights.end());
    return scratch.kept[distribution(rng)];
}

int32_t sample_token(
    std::vector<float> & logits,
    const std::vector<int32_t> & emitted,
    const Yue2ARSamplingWindow & window,
    std::mt19937 & rng,
    Yue2SamplerScratch & scratch) {
    if (window.begin < 0 || window.end <= window.begin ||
        window.stop_token < 0 || static_cast<int64_t>(logits.size()) <= window.stop_token ||
        static_cast<int64_t>(logits.size()) < window.end) {
        throw std::runtime_error("Yue2 AR sampling window is invalid");
    }
    apply_repetition_penalty(logits, emitted, window);
    return sample_from_allowed_ranges(
        logits,
        {
            Yue2SamplerRange{window.begin, window.end},
            Yue2SamplerRange{
                static_cast<int64_t>(emitted.size()) >= window.min_tokens ? window.stop_token : 0,
                static_cast<int64_t>(emitted.size()) >= window.min_tokens ? window.stop_token + 1 : 0,
            },
        },
        window.sampling,
        rng,
        scratch,
        "Yue2 AR");
}

int32_t compact_semantic_index(int32_t token) {
    if (token == kMusicEndToken) {
        return 0;
    }
    if (token >= kCodecOffset && token < kCodecOffset + kCodecSize) {
        return token - kCodecOffset + 1;
    }
    return -1;
}

int32_t sample_semantic_token(
    std::vector<float> & logits,
    const std::vector<int32_t> & emitted,
    const Yue2ARSamplingWindow & window,
    std::mt19937 & rng,
    Yue2SamplerScratch & scratch) {
    if (static_cast<int64_t>(logits.size()) != kCodecSize + 1 ||
        window.begin != kCodecOffset ||
        window.end != kCodecOffset + kCodecSize ||
        window.stop_token != kMusicEndToken) {
        throw std::runtime_error("Yue2 compact semantic logits shape mismatch");
    }
    const float penalty = window.sampling.repetition_penalty;
    if (penalty != 1.0F && !emitted.empty()) {
        const int64_t begin = std::max<int64_t>(0, static_cast<int64_t>(emitted.size()) - window.sampling.penalty_window);
        for (int64_t i = begin; i < static_cast<int64_t>(emitted.size()); ++i) {
            const int32_t index = compact_semantic_index(emitted[static_cast<size_t>(i)]);
            if (index < 0 || index >= static_cast<int32_t>(logits.size())) {
                continue;
            }
            float & value = logits[static_cast<size_t>(index)];
            value = value < 0.0F ? value * penalty : value / penalty;
        }
    }
    const int32_t index = sample_from_allowed_ranges(
        logits,
        {
            Yue2SamplerRange{1, static_cast<int64_t>(logits.size())},
            Yue2SamplerRange{
                static_cast<int64_t>(emitted.size()) >= window.min_tokens ? 0 : 0,
                static_cast<int64_t>(emitted.size()) >= window.min_tokens ? 1 : 0,
            },
        },
        window.sampling,
        rng,
        scratch,
        "Yue2 semantic AR");
    return index == 0 ? kMusicEndToken : kCodecOffset + index - 1;
}

core::TensorValue view_linear_rows(
    ggml_context * ctx,
    const core::TensorValue & weight,
    int64_t row_offset,
    int64_t rows,
    int64_t cols,
    const char * label) {
    if (weight.shape.rank != 2 ||
        weight.shape.dims[0] < row_offset + rows ||
        weight.shape.dims[1] != cols) {
        throw std::runtime_error(std::string("Yue2 ") + label + " weight view is invalid");
    }
    const size_t row_stride = weight.tensor->nb[1];
    const size_t byte_offset = static_cast<size_t>(row_offset) * row_stride;
    return core::wrap_tensor(
        ggml_view_2d(ctx, weight.tensor, cols, rows, row_stride, byte_offset),
        core::TensorShape::from_dims({rows, cols}),
        weight.type);
}

}  // namespace

struct Yue2ARRuntime::Impl {
    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes)
        : execution(execution),
          assets(std::move(assets)),
          weight_type(weight_type) {
        if (!this->assets) {
            throw std::runtime_error("Yue2 AR runtime requires assets");
        }
        store = std::make_shared<core::BackendWeightStore>(
            execution.backend(),
            execution.backend_type(),
            "yue2.ar.weights",
            weight_context_bytes);
        const auto & config = this->assets->config.model;
        const auto & source = *this->assets->model_weights;
        runtime_weights = load_prefix_weights(*store, source, config, weight_type);
        store->upload();
        runtime_config = make_runtime_config(
            config,
            execution.backend_type(),
            prefill_graph_arena_bytes,
            decode_graph_arena_bytes);
        abc_runtime_config = make_runtime_config(
            config,
            execution.backend_type(),
            prefill_graph_arena_bytes,
            decode_graph_arena_bytes,
            kAbcEndToken + 1);
        semantic_runtime_config = make_runtime_config(
            config,
            execution.backend_type(),
            prefill_graph_arena_bytes,
            decode_graph_arena_bytes,
            kCodecSize + 1);
    }

    struct PrefixStateGraph {
        PrefixStateGraph(Impl & owner, int64_t steps)
            : owner(&owner),
              steps(steps) {
            const auto & config = owner.assets->config.model;
            ggml_init_params params{owner.runtime_config.prefill_graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            ggml_init_params state_params{
                ggml_tensor_overhead() * static_cast<size_t>(config.layers * 2),
                nullptr,
                true};
            state_ctx.reset(ggml_init(state_params));
            if (ctx == nullptr || state_ctx == nullptr) {
                throw std::runtime_error("failed to initialize Yue2 AR prefix-state graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "yue2.ar.prefix_state", owner.execution.backend_type()};
            core::ModuleBuildContext state_build{state_ctx.get(), "yue2.ar.prefix_state.cache", owner.execution.backend_type()};
            const auto build_start = Clock::now();
            input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, steps);
            positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, steps);
            attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
            auto ids = core::wrap_tensor(input, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
            auto pos = core::wrap_tensor(positions, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
            auto mask = core::wrap_tensor(attention_mask, core::TensorShape::from_dims({1, 1, steps, steps}), GGML_TYPE_F16);
            auto x = engine::modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                         .build(build, ids, owner.runtime_weights.token_embedding);
            x = core::reshape_tensor(build, x, core::TensorShape::from_dims({1, steps, config.hidden_size}));
            auto stack = engine::modules::DecoderStackModule(owner.runtime_config.decoder.stack)
                             .build(build, x, pos, owner.runtime_weights.stack, std::nullopt, mask);
            keys.reserve(stack.state.layers.size());
            values.reserve(stack.state.layers.size());
            key_values.reserve(stack.state.layers.size());
            value_values.reserve(stack.state.layers.size());
            for (const auto & layer : stack.state.layers) {
                if (!layer.key.has_value() || !layer.value.has_value()) {
                    throw std::runtime_error("Yue2 AR prefix-state graph did not return K/V state");
                }
                const bool skip_bf16_round = owner.execution.backend_type() == core::BackendType::Metal;
                auto key_value = core::wrap_tensor(
                    skip_bf16_round ? layer.key->tensor
                        : ggml_cast(ctx.get(), ggml_cast(ctx.get(), layer.key->tensor, GGML_TYPE_BF16), GGML_TYPE_F32),
                    layer.key->shape,
                    GGML_TYPE_F32);
                auto value_value = core::wrap_tensor(
                    skip_bf16_round ? layer.value->tensor
                        : ggml_cast(ctx.get(), ggml_cast(ctx.get(), layer.value->tensor, GGML_TYPE_BF16), GGML_TYPE_F32),
                    layer.value->shape,
                    GGML_TYPE_F32);
                key_value = core::wrap_tensor(
                    ggml_cast(ctx.get(), key_value.tensor, GGML_TYPE_F16),
                    key_value.shape,
                    GGML_TYPE_F16);
                value_value = core::wrap_tensor(
                    ggml_cast(ctx.get(), value_value.tensor, GGML_TYPE_F16),
                    value_value.shape,
                    GGML_TYPE_F16);
                auto * key = ggml_cpy(ctx.get(), key_value.tensor, ggml_dup_tensor(ctx.get(), key_value.tensor));
                auto * value = ggml_cpy(ctx.get(), value_value.tensor, ggml_dup_tensor(ctx.get(), value_value.tensor));
                ggml_set_output(key);
                ggml_set_output(value);
                keys.push_back(key);
                values.push_back(value);
                key_values.push_back(core::make_tensor(state_build, GGML_TYPE_F16, layer.key->shape));
                value_values.push_back(core::make_tensor(state_build, GGML_TYPE_F16, layer.value->shape));
            }
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            for (auto * key : keys) {
                ggml_build_forward_expand(graph, key);
            }
            for (auto * value : values) {
                ggml_build_forward_expand(graph, value);
            }
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.execution.backend()));
            if (gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("failed to allocate Yue2 AR prefix-state graph");
            }
            state_buffer = ggml_backend_alloc_ctx_tensors(state_ctx.get(), owner.execution.backend());
            if (state_buffer == nullptr) {
                throw std::runtime_error("failed to allocate Yue2 AR prefix-state cache");
            }
            position_values = engine::modules::decoder_position_ids(steps);
            mask_values = engine::modules::causal_prefill_mask_values(1, steps);
            ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            ggml_backend_tensor_set(attention_mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
            engine::debug::timing_log_scalar("yue2.ar.prefix_state.graph.build_ms", engine::debug::elapsed_ms(build_start));
            engine::debug::timing_log_context_reservation("yue2.ar.prefix_state.graph", ctx.get());
        }

        ~PrefixStateGraph() {
            core::release_backend_graph_resources(owner->execution.backend(), graph);
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
            }
            if (state_buffer != nullptr) {
                ggml_backend_buffer_free(state_buffer);
            }
        }

        bool matches(int64_t s) const noexcept {
            return steps == s;
        }

        runtime::TransformerKVState run(const std::vector<int32_t> & tokens) {
            compute(tokens);
            runtime::TransformerKVState out;
            out.current_end = steps;
            out.layers.resize(keys.size());
            for (size_t layer = 0; layer < keys.size(); ++layer) {
                auto & state = out.layers[layer];
                state.valid_steps = steps;
                core::read_tensor_f32_into(keys[layer], state.key);
                core::read_tensor_f32_into(values[layer], state.value);
                core::round_f32_to_bf16_in_place(state.key);
                core::round_f32_to_bf16_in_place(state.value);
            }
            return out;
        }

        Yue2ARDevicePrefixState run_device(const std::vector<int32_t> & tokens) {
            compute(tokens);
            for (size_t layer = 0; layer < keys.size(); ++layer) {
                ggml_backend_tensor_copy(keys[layer], key_values[layer].tensor);
                ggml_backend_tensor_copy(values[layer], value_values[layer].tensor);
            }
            Yue2ARDevicePrefixState out;
            out.current_end = steps;
            out.keys = key_values;
            out.values = value_values;
            return out;
        }

        void compute(const std::vector<int32_t> & tokens) {
            if (static_cast<int64_t>(tokens.size()) != steps) {
                throw std::runtime_error("Yue2 AR prefix-state token size mismatch");
            }
            ggml_backend_tensor_set(input, tokens.data(), 0, tokens.size() * sizeof(int32_t));
            ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            ggml_backend_tensor_set(attention_mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
            const auto status = core::compute_backend_graph(owner->execution.backend(), graph, nullptr, "yue2.ar.prefix_state");
            ggml_backend_synchronize(owner->execution.backend());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Yue2 AR prefix-state graph compute failed");
            }
        }

        Impl * owner = nullptr;
        int64_t steps = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t state_buffer = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * attention_mask = nullptr;
        std::vector<ggml_tensor *> keys;
        std::vector<ggml_tensor *> values;
        std::vector<core::TensorValue> key_values;
        std::vector<core::TensorValue> value_values;
        std::vector<int32_t> position_values;
        std::vector<ggml_fp16_t> mask_values;
    };

    std::vector<int32_t> generate(
        const std::vector<int32_t> & prefix,
        const Yue2ARSamplingWindow & window,
        uint64_t seed,
        const std::vector<int32_t> & forced) {
        if (prefix.empty()) {
            throw std::runtime_error("Yue2 AR prefix must not be empty");
        }
        if (static_cast<int64_t>(forced.size()) > window.max_tokens) {
            throw std::runtime_error("Yue2 AR forced tokens exceed the sampling window");
        }
        const bool compact_semantic = is_semantic_window(window);
        const bool compact_abc = is_abc_window(window);
        ensure_generation_runtime(false, compact_semantic, compact_abc);
        auto & active_runtime = compact_semantic ? semantic_runtime : (compact_abc ? abc_runtime : runtime);
        const auto total_start = Clock::now();
        engine::debug::timing_log_scalar("yue2.ar.generate.prefix_tokens", prefix.size());
        auto cache_steps_for = [](int64_t prefix_tokens, int64_t remaining_tokens) {
            return prefix_tokens + std::min<int64_t>(remaining_tokens, kArDecodeChunkTokens);
        };
        std::vector<int32_t> emitted;
        emitted.reserve(static_cast<size_t>(window.max_tokens));
        emitted.insert(emitted.end(), forced.begin(), forced.end());
        std::vector<int32_t> forced_prefix;
        if (!forced.empty()) {
            forced_prefix.reserve(prefix.size() + forced.size());
            forced_prefix.insert(forced_prefix.end(), prefix.begin(), prefix.end());
            forced_prefix.insert(forced_prefix.end(), forced.begin(), forced.end());
            engine::debug::timing_log_scalar("yue2.ar.generate.forced_tokens", forced.size());
        }
        const auto & start_prefix = forced.empty() ? prefix : forced_prefix;
        const auto prefill_start = Clock::now();
        auto prefill = active_runtime->prefill_tokens_into_decode_cache(
            start_prefix,
            cache_steps_for(
                static_cast<int64_t>(start_prefix.size()),
                window.max_tokens - static_cast<int64_t>(emitted.size())));
        engine::debug::timing_log_scalar("yue2.ar.generate.prefill_ms", engine::debug::elapsed_ms(prefill_start));
        std::mt19937 rng(static_cast<uint32_t>(seed));
        Yue2SamplerScratch scratch;
        engine::modules::CausalDecoderStepResult decode_result;
        decode_result.logits = std::move(prefill.logits);
        decode_result.hidden = std::move(prefill.hidden);
        double sample_ms = 0.0;
        double decode_ms = 0.0;
        double refill_prefill_ms = 0.0;
        int64_t refill_count = 0;
        for (int64_t step = static_cast<int64_t>(emitted.size()); step < window.max_tokens; ++step) {
            const auto sample_start = Clock::now();
            const int32_t token = compact_semantic ?
                sample_semantic_token(decode_result.logits, emitted, window, rng, scratch) :
                sample_token(decode_result.logits, emitted, window, rng, scratch);
            sample_ms += engine::debug::elapsed_ms(sample_start);
            if (token == window.stop_token) {
                engine::debug::timing_log_scalar("yue2.ar.generate.sample_ms", sample_ms);
                engine::debug::timing_log_scalar("yue2.ar.generate.decode_ms", decode_ms);
                engine::debug::timing_log_scalar("yue2.ar.generate.refill_prefill_ms", refill_prefill_ms);
                engine::debug::timing_log_scalar("yue2.ar.generate.refill_count", refill_count);
                engine::debug::timing_log_scalar("yue2.ar.generate.emitted_tokens", emitted.size());
                engine::debug::timing_log_scalar("yue2.ar.generate.total_ms", engine::debug::elapsed_ms(total_start));
                return emitted;
            }
            emitted.push_back(token);
            if (static_cast<int64_t>(emitted.size()) >= window.max_tokens) {
                break;
            }
            if (active_runtime->decode_valid_steps() >= active_runtime->decode_cache_steps()) {
                std::vector<int32_t> refill_prefix;
                refill_prefix.reserve(prefix.size() + emitted.size());
                refill_prefix.insert(refill_prefix.end(), prefix.begin(), prefix.end());
                refill_prefix.insert(refill_prefix.end(), emitted.begin(), emitted.end());
                const auto refill_prefill_start = Clock::now();
                const int64_t remaining = window.max_tokens - static_cast<int64_t>(emitted.size());
                auto refill = active_runtime->prefill_tokens_into_decode_cache(
                    refill_prefix,
                    cache_steps_for(static_cast<int64_t>(refill_prefix.size()), remaining));
                refill_prefill_ms += engine::debug::elapsed_ms(refill_prefill_start);
                decode_result.logits = std::move(refill.logits);
                decode_result.hidden = std::move(refill.hidden);
                ++refill_count;
                continue;
            }
            const auto decode_start = Clock::now();
            active_runtime->decode_token_into(token, decode_result);
            decode_ms += engine::debug::elapsed_ms(decode_start);
        }
        engine::debug::timing_log_scalar("yue2.ar.generate.sample_ms", sample_ms);
        engine::debug::timing_log_scalar("yue2.ar.generate.decode_ms", decode_ms);
        engine::debug::timing_log_scalar("yue2.ar.generate.refill_prefill_ms", refill_prefill_ms);
        engine::debug::timing_log_scalar("yue2.ar.generate.refill_count", refill_count);
        engine::debug::timing_log_scalar("yue2.ar.generate.emitted_tokens", emitted.size());
        engine::debug::timing_log_scalar("yue2.ar.generate.total_ms", engine::debug::elapsed_ms(total_start));
        return emitted;
    }

    std::vector<int32_t> generate_cfg(
        const std::vector<int32_t> & positive_prefix,
        const std::vector<int32_t> & negative_prefix,
        const Yue2ARSamplingWindow & window,
        float guidance_scale,
        uint64_t seed,
        const std::vector<int32_t> & forced) {
        if (guidance_scale == 1.0F) {
            return generate(positive_prefix, window, seed, forced);
        }
        if (static_cast<int64_t>(forced.size()) > window.max_tokens) {
            throw std::runtime_error("Yue2 AR forced tokens exceed the sampling window");
        }
        const bool compact_semantic = is_semantic_window(window);
        const bool compact_abc = is_abc_window(window);
        ensure_generation_runtime(false, compact_semantic, compact_abc);
        auto & positive_runtime = compact_semantic ? semantic_runtime : (compact_abc ? abc_runtime : runtime);
        const auto total_start = Clock::now();
        auto append_forced = [&forced](const std::vector<int32_t> & tokens) {
            std::vector<int32_t> out;
            out.reserve(tokens.size() + forced.size());
            out.insert(out.end(), tokens.begin(), tokens.end());
            out.insert(out.end(), forced.begin(), forced.end());
            return out;
        };
        const auto positive_tokens = append_forced(positive_prefix);
        const auto negative_tokens = append_forced(negative_prefix);
        engine::debug::timing_log_scalar("yue2.ar.cfg.positive_prefix_tokens", positive_tokens.size());
        engine::debug::timing_log_scalar("yue2.ar.cfg.negative_prefix_tokens", negative_tokens.size());
        if (!forced.empty()) {
            engine::debug::timing_log_scalar("yue2.ar.cfg.forced_tokens", forced.size());
        }
        const auto positive_prefill_start = Clock::now();
        auto positive = positive_runtime->prefill_tokens(positive_tokens);
        engine::debug::timing_log_scalar("yue2.ar.cfg.prefill_positive_ms", engine::debug::elapsed_ms(positive_prefill_start));
        const auto negative_prefill_start = Clock::now();
        auto negative = positive_runtime->prefill_tokens(negative_tokens);
        engine::debug::timing_log_scalar("yue2.ar.cfg.prefill_negative_ms", engine::debug::elapsed_ms(negative_prefill_start));
        const auto start_decode_start = Clock::now();
        const int64_t cache_steps =
            std::max<int64_t>(
                static_cast<int64_t>(positive_tokens.size()),
                static_cast<int64_t>(negative_tokens.size())) +
            window.max_tokens - static_cast<int64_t>(forced.size());
        positive_runtime->start_decode_tokens_batched(
            make_cfg_batched_state(positive.state, negative.state),
            cache_steps);
        engine::debug::timing_log_scalar("yue2.ar.cfg.start_decode_ms", engine::debug::elapsed_ms(start_decode_start));
        std::vector<int32_t> emitted;
        emitted.reserve(static_cast<size_t>(window.max_tokens));
        emitted.insert(emitted.end(), forced.begin(), forced.end());
        std::mt19937 rng(static_cast<uint32_t>(seed));
        Yue2SamplerScratch scratch;
        std::vector<float> logits(positive.logits.size(), 0.0F);
        double sample_ms = 0.0;
        double decode_batched_ms = 0.0;
        for (int64_t step = static_cast<int64_t>(emitted.size()); step < window.max_tokens; ++step) {
            if (positive.logits.size() != negative.logits.size()) {
                throw std::runtime_error("Yue2 CFG logits size mismatch");
            }
            for (size_t i = 0; i < logits.size(); ++i) {
                logits[i] = negative.logits[i] + (positive.logits[i] - negative.logits[i]) * guidance_scale;
            }
            const auto sample_start = Clock::now();
            const int32_t token = compact_semantic ?
                sample_semantic_token(logits, emitted, window, rng, scratch) :
                sample_token(logits, emitted, window, rng, scratch);
            sample_ms += engine::debug::elapsed_ms(sample_start);
            if (token == window.stop_token) {
                engine::debug::timing_log_scalar("yue2.ar.cfg.sample_ms", sample_ms);
                engine::debug::timing_log_scalar("yue2.ar.cfg.decode_batched_ms", decode_batched_ms);
                engine::debug::timing_log_scalar("yue2.ar.cfg.emitted_tokens", emitted.size());
                engine::debug::timing_log_scalar("yue2.ar.cfg.total_ms", engine::debug::elapsed_ms(total_start));
                return emitted;
            }
            emitted.push_back(token);
            if (static_cast<int64_t>(emitted.size()) >= window.max_tokens) {
                break;
            }
            const auto decode_start = Clock::now();
            auto batched = positive_runtime->decode_tokens_batched({token, token});
            decode_batched_ms += engine::debug::elapsed_ms(decode_start);
            if (batched.logits.size() % 2 != 0) {
                throw std::runtime_error("Yue2 CFG batched logits shape mismatch");
            }
            const size_t row = batched.logits.size() / 2;
            positive.logits.assign(batched.logits.begin(), batched.logits.begin() + static_cast<std::ptrdiff_t>(row));
            negative.logits.assign(batched.logits.begin() + static_cast<std::ptrdiff_t>(row), batched.logits.end());
        }
        engine::debug::timing_log_scalar("yue2.ar.cfg.sample_ms", sample_ms);
        engine::debug::timing_log_scalar("yue2.ar.cfg.decode_batched_ms", decode_batched_ms);
        engine::debug::timing_log_scalar("yue2.ar.cfg.emitted_tokens", emitted.size());
        engine::debug::timing_log_scalar("yue2.ar.cfg.total_ms", engine::debug::elapsed_ms(total_start));
        return emitted;
    }

    runtime::TransformerKVState prefill_state(const std::vector<int32_t> & tokens) {
        const int64_t steps = static_cast<int64_t>(tokens.size());
        if (!prefix_state_graph || !prefix_state_graph->matches(steps)) {
            prefix_state_graph = std::make_unique<PrefixStateGraph>(*this, steps);
        }
        auto state = prefix_state_graph->run(tokens);
        return state;
    }

    Yue2ARDevicePrefixState prefill_device_state(const std::vector<int32_t> & tokens) {
        const int64_t steps = static_cast<int64_t>(tokens.size());
        if (!prefix_state_graph || !prefix_state_graph->matches(steps)) {
            prefix_state_graph = std::make_unique<PrefixStateGraph>(*this, steps);
        }
        auto state = prefix_state_graph->run_device(tokens);
        return state;
    }

    static bool is_semantic_window(const Yue2ARSamplingWindow & window) noexcept {
        return window.begin == kCodecOffset &&
            window.end == kCodecOffset + kCodecSize &&
            window.stop_token == kMusicEndToken;
    }

    static bool is_abc_window(const Yue2ARSamplingWindow & window) noexcept {
        return window.begin == 0 &&
            window.end == kEodToken &&
            window.stop_token == kAbcEndToken;
    }

    void ensure_generation_runtime(bool require_negative, bool compact_semantic, bool compact_abc) {
        auto & active_runtime = compact_semantic ? semantic_runtime : (compact_abc ? abc_runtime : runtime);
        auto & active_negative_runtime =
            compact_semantic ? semantic_negative_runtime : (compact_abc ? abc_negative_runtime : negative_runtime);
        const auto & active_config =
            compact_semantic ? semantic_runtime_config : (compact_abc ? abc_runtime_config : runtime_config);
        if (active_runtime && (!require_negative || active_negative_runtime)) {
            return;
        }
        const auto total_start = Clock::now();
        if (!generation_store) {
            const auto & config = assets->config.model;
            const auto & source = *assets->model_weights;
            generation_store = std::make_shared<core::BackendWeightStore>(
                execution.backend(),
                execution.backend_type(),
                "yue2.ar.generation.weights",
                64ull * 1024ull * 1024ull);
            load_generation_weights(runtime_weights, *generation_store, source, config, weight_type);
            generation_store->upload();
            ggml_init_params view_params{ggml_tensor_overhead() * 8, nullptr, true};
            generation_view_ctx.reset(ggml_init(view_params));
            if (generation_view_ctx == nullptr) {
                throw std::runtime_error("failed to initialize Yue2 AR generation weight views");
            }
            abc_runtime_weights = runtime_weights;
            semantic_runtime_weights = runtime_weights;
            if (!runtime_weights.lm_head.has_value()) {
                throw std::runtime_error("Yue2 AR generation runtime requires lm_head");
            }
            abc_runtime_weights.lm_head = engine::modules::LinearWeights{
                view_linear_rows(
                    generation_view_ctx.get(),
                    runtime_weights.lm_head->weight,
                    0,
                    kAbcEndToken + 1,
                    config.hidden_size,
                    "ABC lm_head"),
                std::nullopt,
            };
            semantic_runtime_weights.lm_head = engine::modules::LinearWeights{
                view_linear_rows(
                    generation_view_ctx.get(),
                    runtime_weights.lm_head->weight,
                    kMusicEndToken,
                    kCodecSize + 1,
                    config.hidden_size,
                    "semantic lm_head"),
                std::nullopt,
            };
        }
        if (!active_runtime) {
            active_runtime = std::make_unique<engine::modules::CausalDecoderRuntime>(
                execution,
                active_config,
                compact_semantic ? semantic_runtime_weights : (compact_abc ? abc_runtime_weights : runtime_weights));
        }
        if (require_negative && !active_negative_runtime) {
            active_negative_runtime = std::make_unique<engine::modules::CausalDecoderRuntime>(
                execution,
                active_config,
                compact_semantic ? semantic_runtime_weights : (compact_abc ? abc_runtime_weights : runtime_weights));
        }
        engine::debug::timing_log_scalar("yue2.ar.ensure_generation_ms", engine::debug::elapsed_ms(total_start));
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const Yue2Assets> assets;
    std::shared_ptr<core::BackendWeightStore> store;
    std::shared_ptr<core::BackendWeightStore> generation_store;
    std::unique_ptr<ggml_context, GgmlContextDeleter> generation_view_ctx;
    assets::TensorStorageType weight_type;
    engine::modules::CausalDecoderRuntimeConfig runtime_config;
    engine::modules::CausalDecoderRuntimeConfig abc_runtime_config;
    engine::modules::CausalDecoderRuntimeConfig semantic_runtime_config;
    engine::modules::CausalDecoderRuntimeWeights runtime_weights;
    engine::modules::CausalDecoderRuntimeWeights abc_runtime_weights;
    engine::modules::CausalDecoderRuntimeWeights semantic_runtime_weights;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> runtime;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> negative_runtime;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> abc_runtime;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> abc_negative_runtime;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> semantic_runtime;
    std::unique_ptr<engine::modules::CausalDecoderRuntime> semantic_negative_runtime;
    std::unique_ptr<PrefixStateGraph> prefix_state_graph;
};

Yue2ARRuntime::Yue2ARRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const Yue2Assets> assets,
    assets::TensorStorageType weight_type,
    size_t weight_context_bytes,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          weight_type,
          weight_context_bytes,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes)) {}

Yue2ARRuntime::~Yue2ARRuntime() = default;

std::vector<int32_t> Yue2ARRuntime::generate(
    const std::vector<int32_t> & prefix,
    const Yue2ARSamplingWindow & window,
    uint64_t seed,
    const std::vector<int32_t> & forced) {
    return impl_->generate(prefix, window, seed, forced);
}

std::vector<int32_t> Yue2ARRuntime::generate_cfg(
    const std::vector<int32_t> & positive_prefix,
    const std::vector<int32_t> & negative_prefix,
    const Yue2ARSamplingWindow & window,
    float guidance_scale,
    uint64_t seed,
    const std::vector<int32_t> & forced) {
    return impl_->generate_cfg(positive_prefix, negative_prefix, window, guidance_scale, seed, forced);
}

runtime::TransformerKVState Yue2ARRuntime::prefill_state(const std::vector<int32_t> & tokens) {
    return impl_->prefill_state(tokens);
}

Yue2ARDevicePrefixState Yue2ARRuntime::prefill_device_state(const std::vector<int32_t> & tokens) {
    return impl_->prefill_device_state(tokens);
}

void Yue2ARRuntime::release_runtime_graphs() {
    impl_->prefix_state_graph.reset();
    if (impl_->runtime) {
        impl_->runtime->release_runtime_graphs();
    }
    if (impl_->negative_runtime) {
        impl_->negative_runtime->release_runtime_graphs();
    }
    if (impl_->abc_runtime) {
        impl_->abc_runtime->release_runtime_graphs();
    }
    if (impl_->abc_negative_runtime) {
        impl_->abc_negative_runtime->release_runtime_graphs();
    }
    if (impl_->semantic_runtime) {
        impl_->semantic_runtime->release_runtime_graphs();
    }
    if (impl_->semantic_negative_runtime) {
        impl_->semantic_negative_runtime->release_runtime_graphs();
    }
}

}  // namespace engine::models::yue2

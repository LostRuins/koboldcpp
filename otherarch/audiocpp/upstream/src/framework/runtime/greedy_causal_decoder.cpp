#include "engine/framework/runtime/greedy_causal_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/sampling/greedy_decode.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::runtime {
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

struct DecoderLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue q_bias;
    core::TensorValue k_proj;
    core::TensorValue k_bias;
    core::TensorValue v_proj;
    core::TensorValue v_bias;
    core::TensorValue qkv_weight;
    core::TensorValue qkv_bias;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct DecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<DecoderLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

modules::DecoderLayerWeights bind_layer_weights(
    const DecoderLayerWeights & weights,
    const GreedyCausalDecoderSpec & spec) {
    modules::DecoderLayerWeights out;
    out.input_norm = {weights.input_norm, std::nullopt};
    out.self_attention.q_weight = weights.q_proj;
    if (spec.attention_bias) {
        out.self_attention.q_bias = weights.q_bias;
        out.self_attention.k_bias = weights.k_bias;
        out.self_attention.v_bias = weights.v_bias;
    }
    out.self_attention.k_weight = weights.k_proj;
    out.self_attention.v_weight = weights.v_proj;
    if (spec.packed_qkv) {
        out.self_attention.qkv_weight = weights.qkv_weight;
        if (spec.attention_bias) {
            out.self_attention.qkv_bias = weights.qkv_bias;
        }
    }
    out.self_attention.out_weight = weights.o_proj;
    if (spec.decoder.stack.use_qk_norm) {
        out.q_norm = {weights.q_norm, std::nullopt};
        out.k_norm = {weights.k_norm, std::nullopt};
    }
    out.post_norm = {weights.post_norm, std::nullopt};
    out.mlp.gate_proj = {weights.gate_proj, std::nullopt};
    out.mlp.up_proj = {weights.up_proj, std::nullopt};
    out.mlp.down_proj = {weights.down_proj, std::nullopt};
    return out;
}

modules::CausalDecoderWeights bind_decoder_weights(
    const DecoderWeights & weights,
    const GreedyCausalDecoderSpec & spec) {
    modules::CausalDecoderWeights out;
    out.stack.layers.reserve(weights.layers.size());
    for (const auto & layer : weights.layers) {
        out.stack.layers.push_back(bind_layer_weights(layer, spec));
    }
    out.final_norm = {weights.norm, std::nullopt};
    out.lm_head = {weights.lm_head, std::nullopt};
    return out;
}

core::TensorValue prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const DecoderWeights & weights,
    const GreedyCausalDecoderSpec & spec,
    ggml_tensor * token_ids,
    int64_t prompt_steps,
    const std::vector<float> & injection_values,
    int64_t injection_tokens,
    const std::vector<int32_t> & injection_positions,
    ggml_tensor * injection_values_tensor,
    ggml_tensor * injection_positions_tensor) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({spec.vocab_size, spec.decoder.stack.hidden_size})
                 .build(ctx, ids, weights.token_embedding);
    if (injection_tokens > 0) {
        auto injection = core::wrap_tensor(
            injection_values_tensor,
            core::TensorShape::from_dims({injection_tokens, spec.decoder.stack.hidden_size}),
            GGML_TYPE_F32);
        auto positions = core::wrap_tensor(
            injection_positions_tensor,
            core::TensorShape::from_dims({injection_tokens}),
            GGML_TYPE_I64);
        x = core::wrap_tensor(
            ggml_set_rows(ctx.ggml, x.tensor, injection.tensor, positions.tensor),
            x.shape,
            GGML_TYPE_F32);
    }
    (void)injection_values;
    (void)injection_positions;
    return core::reshape_tensor(
        ctx, x, core::TensorShape::from_dims({1, prompt_steps, spec.decoder.stack.hidden_size}));
}

DecoderWeights load_weights(
    const assets::TensorSource & source,
    const GreedyCausalDecoderSpec & spec,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & stack = spec.decoder.stack;
    if (stack.hidden_size <= 0 || stack.num_attention_heads <= 0 || stack.head_dim <= 0 ||
        stack.layers <= 0 || spec.vocab_size <= 0) {
        throw std::runtime_error("Greedy Qwen decoder spec is invalid");
    }
    DecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "greedy_qwen_decoder.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        spec.token_embedding_tensor,
        storage_type,
        {spec.vocab_size, stack.hidden_size});
    const int64_t dim = stack.head_dim;
    weights.layers.reserve(static_cast<size_t>(stack.layers));
    for (int64_t layer = 0; layer < stack.layers; ++layer) {
        const std::string prefix = spec.layer_prefix + "." + std::to_string(layer);
        DecoderLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {stack.hidden_size});
        if (spec.packed_qkv) {
            const int64_t qkv_rows =
                (stack.num_attention_heads + 2 * stack.num_key_value_heads) * dim;
            w.qkv_weight = weights.store->load_tensor(
                source, prefix + ".self_attn.qkv_proj.weight", storage_type, {qkv_rows, stack.hidden_size});
            if (spec.attention_bias) {
                w.qkv_bias = weights.store->load_f32_tensor(source, prefix + ".self_attn.qkv_proj.bias", {qkv_rows});
            }
        } else {
            w.q_proj = weights.store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {stack.num_attention_heads * dim, stack.hidden_size});
            w.k_proj = weights.store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {stack.num_key_value_heads * dim, stack.hidden_size});
            w.v_proj = weights.store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {stack.num_key_value_heads * dim, stack.hidden_size});
            if (spec.attention_bias) {
                w.q_bias = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_proj.bias", {stack.num_attention_heads * dim});
                w.k_bias = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_proj.bias", {stack.num_key_value_heads * dim});
                w.v_bias = weights.store->load_f32_tensor(source, prefix + ".self_attn.v_proj.bias", {stack.num_key_value_heads * dim});
            }
        }
        w.o_proj = weights.store->load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {stack.hidden_size, stack.num_attention_heads * dim});
        if (stack.use_qk_norm) {
            w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
            w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        }
        w.post_norm = weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {stack.hidden_size});
        w.gate_proj = weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {stack.intermediate_size, stack.hidden_size});
        w.up_proj = weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {stack.intermediate_size, stack.hidden_size});
        w.down_proj = weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {stack.hidden_size, stack.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, spec.final_norm_tensor, {stack.hidden_size});
    if (spec.tie_word_embeddings) {
        if (spec.decoder.logits_size != 0 && spec.decoder.logits_size != spec.vocab_size) {
            throw std::runtime_error("tied output embedding requires logits_size == vocab_size");
        }
        weights.lm_head = weights.token_embedding;
    } else {
        weights.lm_head = weights.store->load_tensor(
            source,
            spec.lm_head_tensor,
            storage_type,
            {spec.decoder.logits_size != 0 ? spec.decoder.logits_size : spec.vocab_size, stack.hidden_size});
    }
    weights.store->upload();
    return weights;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("Greedy Qwen decoder cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

bool is_eos(const GreedyCausalDecoderSpec & spec, int32_t token) {
    return std::find(spec.eos_token_ids.begin(), spec.eos_token_ids.end(), static_cast<int64_t>(token)) !=
        spec.eos_token_ids.end();
}

class ThinkerWeightsRuntime {
public:
    ThinkerWeightsRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        GreedyCausalDecoderSpec spec,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : spec_(std::make_shared<const GreedyCausalDecoderSpec>(std::move(spec))),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<DecoderWeights>(load_weights(
              *source,
              *spec_,
              backend_,
              backend_type_,
              weight_context_bytes,
              storage_type))) {}

    const GreedyCausalDecoderSpec & spec() const noexcept {
        return *spec_;
    }

    const DecoderWeights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const GreedyCausalDecoderSpec> spec_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DecoderWeights> weights_;
};

// A small fixed-width lookup graph also works for quantized embedding weights.
// Padding is read back only for the last block, then discarded before injection.
class PromptEmbeddingGraph {
public:
    static constexpr int64_t kSteps = 64;
    explicit PromptEmbeddingGraph(std::shared_ptr<ThinkerWeightsRuntime> runtime)
        : runtime_(std::move(runtime)) {
        ctx_.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!ctx_) { throw std::runtime_error("failed to initialize prompt embedding graph"); }
        core::ModuleBuildContext ctx{ctx_.get(), "greedy_qwen_decoder.embedding", runtime_->backend_type()};
        ids_ = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, kSteps);
        auto ids = core::wrap_tensor(ids_, core::TensorShape::from_dims({kSteps}), GGML_TYPE_I32);
        const auto & spec = runtime_->spec();
        output_ = modules::EmbeddingModule({spec.vocab_size, spec.decoder.stack.hidden_size})
                      .build(ctx, ids, runtime_->weights().token_embedding).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx.ggml, 64, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend())));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("failed to allocate prompt embedding graph");
        }
    }
    ~PromptEmbeddingGraph() {
        core::release_backend_graph_resources(runtime_->backend(), graph_, true);
    }
    std::vector<float> run(const std::vector<int32_t> & ids) {
        const int64_t width = runtime_->spec().decoder.stack.hidden_size;
        std::vector<float> result(ids.size() * static_cast<size_t>(width));
        std::vector<int32_t> block(kSteps, 0);
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        for (size_t offset = 0; offset < ids.size(); offset += kSteps) {
            const size_t count = std::min<size_t>(kSteps, ids.size() - offset);
            std::fill(block.begin(), block.end(), 0);
            std::copy_n(ids.begin() + offset, count, block.begin());
            ggml_backend_tensor_set(ids_, block.data(), 0, block.size() * sizeof(int32_t));
            if (core::compute_backend_graph(runtime_->backend(), graph_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("prompt embedding graph compute failed");
            }
            ggml_backend_synchronize(runtime_->backend());
            ggml_backend_tensor_get(output_, result.data() + offset * width, 0, count * width * sizeof(float));
        }
        return result;
    }
private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> allocator_;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<ThinkerWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t injection_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          injection_tokens_(injection_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Greedy Qwen decoder prefill requires positive prompt length");
        }
        if (injection_tokens_ < 0 || injection_tokens_ > prompt_steps_) {
            throw std::runtime_error("Greedy Qwen decoder prefill injection token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize greedy Qwen decoder prefill graph context");
        }
        const auto & spec = runtime_->spec();
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "greedy_qwen_decoder.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        injection_values_ = ggml_new_tensor_2d(
            ctx_.get(), GGML_TYPE_F32, spec.decoder.stack.hidden_size, std::max<int64_t>(injection_tokens_, 1));
        injection_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(injection_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            spec,
            token_ids_,
            prompt_steps_,
            {},
            injection_tokens_,
            {},
            injection_values_,
            injection_positions_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        auto decoder_out = modules::CausalDecoderModule(spec.decoder)
                               .build(ctx, x, positions, bind_decoder_weights(weights, spec));
        for (const auto & layer : decoder_out.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("greedy Qwen decoder prefill did not return K/V state");
            }
            // The graph allocator recycles intermediates; copy K/V into their
            // own tensors and mark them as outputs so run() can read them back.
            auto * key = ggml_cpy(
                ctx_.get(),
                layer.key->tensor,
                ggml_dup_tensor(ctx_.get(), layer.key->tensor));
            auto * value = ggml_cpy(
                ctx_.get(),
                layer.value->tensor,
                ggml_dup_tensor(ctx_.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            keys_.push_back(key);
            values_.push_back(value);
        }
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        for (auto * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (auto * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        const auto try_alloc = [&]() {
            gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend())));
            return gallocr_ != nullptr &&
                ggml_gallocr_reserve(gallocr_.get(), graph_) &&
                ggml_gallocr_alloc_graph(gallocr_.get(), graph_);
        };
        if (!try_alloc() &&
            (engine::core::trim_backend_pools(runtime_->backend()), !try_alloc())) {
            throw engine::runtime::CapacityError(
                "greedy Qwen decoder prefill graph does not fit in device memory at this size ("
                + std::to_string(prompt_steps_) + " prompt steps, of which "
                + std::to_string(injection_tokens_) + " are injected tokens)");
        }
        position_ids_ = modules::decoder_position_ids(prompt_steps_);
        debug::timing_log_scalar("greedy_qwen_decoder.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::timing_log_context_reservation("greedy_qwen_decoder.prefill.graph", ctx_.get());
        debug::trace_log_scalar("greedy_qwen_decoder.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
    }

    bool matches(int64_t prompt_steps, int64_t injection_tokens) const {
        return prompt_steps_ == prompt_steps && injection_tokens_ == injection_tokens;
    }

    PrefillOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & injection_values,
        const std::vector<int32_t> & injection_positions) {
        const auto & spec = runtime_->spec();
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("greedy Qwen decoder prefill token id count mismatch");
        }
        if (static_cast<int64_t>(injection_values.size()) != injection_tokens_ * spec.decoder.stack.hidden_size) {
            throw std::runtime_error("greedy Qwen decoder prefill injection value size mismatch");
        }
        if (static_cast<int64_t>(injection_positions.size()) != injection_tokens_) {
            throw std::runtime_error("greedy Qwen decoder prefill injection position count mismatch");
        }
        // Re-uploaded on every run: leaves are not pinned by the graph allocator.
        ggml_backend_tensor_set(positions_, position_ids_.data(), 0, position_ids_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (injection_tokens_ > 0) {
            std::vector<int64_t> positions(injection_positions.begin(), injection_positions.end());
            ggml_backend_tensor_set(
                injection_values_, injection_values.data(), 0, injection_values.size() * sizeof(float));
            ggml_backend_tensor_set(
                injection_positions_, positions.data(), 0, positions.size() * sizeof(int64_t));
        }
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("greedy Qwen decoder prefill graph compute failed");
        }
        PrefillOutput out;
        const int64_t logits_size = spec.decoder.logits_size != 0
            ? spec.decoder.logits_size
            : spec.vocab_size;
        out.logits.resize(static_cast<size_t>(logits_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(
            prompt_steps_ * spec.decoder.stack.num_key_value_heads * spec.decoder.stack.head_dim);
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        return out;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t injection_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * injection_values_ = nullptr;
    ggml_tensor * injection_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<int32_t> position_ids_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
};

class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<ThinkerWeightsRuntime> runtime, int64_t cache_steps, size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("greedy Qwen decoder decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize greedy Qwen decoder decode graph context");
        }
        const auto & spec = runtime_->spec();
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "greedy_qwen_decoder.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token_id = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({spec.vocab_size, spec.decoder.stack.hidden_size})
                     .build(ctx, token_id, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, spec.decoder.stack.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto decoder_out = modules::CausalDecoderModule(spec.decoder)
                               .build_static_cache_tail(
                                   ctx,
                                   graph_,
                                   x,
                                   positions,
                                   bind_decoder_weights(weights, spec),
                                   cache_steps_,
                                   attention_mask,
                                   cache_slot);
        step_cache_ = std::move(decoder_out.cache);
        logits_ = decoder_out.logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            engine::core::trim_backend_pools(runtime_->backend());
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        }
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate greedy Qwen decoder decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("greedy_qwen_decoder.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::timing_log_context_reservation("greedy_qwen_decoder.decode.graph", ctx_.get());
        debug::trace_log_scalar("greedy_qwen_decoder.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(int64_t required_steps) const {
        return cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    std::vector<float> run_step(int32_t token) {
        const auto & spec = runtime_->spec();
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("greedy Qwen decoder decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        modules::write_decoder_cached_step_mask(
            attention_mask_,
            attention_mask_values_,
            cache_steps_,
            step_cache_.valid_steps(),
            step_cache_.valid_steps());
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("greedy Qwen decoder decode graph compute failed");
        }
        const int64_t logits_size = spec.decoder.logits_size != 0
            ? spec.decoder.logits_size
            : spec.vocab_size;
        logits_buffer_.resize(static_cast<size_t>(logits_size));
        ggml_backend_tensor_get(logits_, logits_buffer_.data(), 0, logits_buffer_.size() * sizeof(float));
        step_cache_.advance_after_direct_append(1);
        // The caller moves out of this buffer before the next step.
        return std::move(logits_buffer_);
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<float> logits_buffer_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct GreedyCausalDecoderRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> weights_source,
        GreedyCausalDecoderSpec spec,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<ThinkerWeightsRuntime>(
              std::move(weights_source),
              std::move(spec),
              execution,
              weight_context_bytes,
              storage_type)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes),
          execution(&execution) {}

    std::shared_ptr<ThinkerWeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
    core::ExecutionContext * execution;
    std::unique_ptr<PromptEmbeddingGraph> embedding_graph;
    std::unique_ptr<modules::CausalDecoderRuntime> reusable_decoder;

    std::vector<int32_t> generate_reusing_graphs(
        const GreedyCausalDecoderRuntime::Prompt & prompt,
        int64_t max_new_tokens,
        int64_t cached_prefix_steps,
        int64_t capacity_bucket) {
        const auto & spec = weights->spec();
        if (!reusable_decoder) {
            modules::CausalDecoderRuntimeConfig config;
            config.trace_name = "greedy_qwen_decoder.reusable";
            config.decoder = spec.decoder;
            config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
            config.decode_graph_arena_bytes = decode_graph_arena_bytes;
            config.evict_cuda_graph_cache_on_release = true;
            const auto bound = bind_decoder_weights(weights->weights(), spec);
            modules::CausalDecoderRuntimeWeights bound_weights;
            bound_weights.token_embedding = weights->weights().token_embedding;
            bound_weights.stack = bound.stack;
            bound_weights.final_norm = bound.final_norm;
            bound_weights.lm_head = bound.lm_head;
            reusable_decoder = std::make_unique<modules::CausalDecoderRuntime>(
                *execution, config, bound_weights);
        }
        if (!embedding_graph) {
            embedding_graph = std::make_unique<PromptEmbeddingGraph>(weights);
        }
        const int64_t width = spec.decoder.stack.hidden_size;
        const int64_t steps = static_cast<int64_t>(prompt.input_ids.size());
        const int64_t required = steps + max_new_tokens;
        // Grow in bounded capacity buckets, keeping only one decode and one
        // block-prefill graph. A new prompt clears KV on device, not via a
        // host export/import of every layer.
        const int64_t capacity = std::min(
            spec.max_position_embeddings, (required + capacity_bucket - 1) / capacity_bucket * capacity_bucket);
        // Incremental prompts size the block to the recomputed suffix (at most
        // 256 rows, split evenly) so no padded rows are computed; the legacy
        // path keeps its fixed 64-step block.
        const auto block_steps_for = [&](int64_t rows) {
            if (cached_prefix_steps <= 0) {
                return int64_t{64};
            }
            const int64_t blocks = (rows + 255) / 256;
            return std::min<int64_t>(256, ((rows + blocks - 1) / blocks + 7) / 8 * 8);
        };
        const int64_t kept = reusable_decoder->retainable_prefix_steps(
            steps, capacity, block_steps_for(steps - cached_prefix_steps), cached_prefix_steps);
        // Only rows past the kept prefix are uploaded; the rest stay zero.
        std::vector<float> embeddings(static_cast<size_t>(steps * width), 0.0F);
        const auto suffix = embedding_graph->run(
            std::vector<int32_t>(prompt.input_ids.begin() + kept, prompt.input_ids.end()));
        std::copy(suffix.begin(), suffix.end(), embeddings.begin() + kept * width);
        for (size_t i = 0; i < prompt.injection.positions.size(); ++i) {
            if (prompt.injection.positions[i] >= kept) {
                std::copy_n(prompt.injection.values.data() + i * width, width,
                            embeddings.data() + prompt.injection.positions[i] * width);
            }
        }
        const auto start = Clock::now();
        auto logits = reusable_decoder->prefill_embeddings_into_cache(
            embeddings, steps, capacity, block_steps_for(steps - kept), kept).logits;
        debug::timing_log_scalar("greedy_qwen_decoder.reusable.prefill_ms", engine::debug::elapsed_ms(start));
        std::vector<int32_t> out;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            const int32_t token = argmax_index(logits);
            if (is_eos(spec, token)) { break; }
            out.push_back(token);
            if (step + 1 < max_new_tokens) {
                logits = reusable_decoder->decode_token(token).logits;
            }
        }
        return out;
    }

};

GreedyCausalDecoderRuntime::GreedyCausalDecoderRuntime(
    std::shared_ptr<const assets::TensorSource> weights_source,
    const GreedyCausalDecoderSpec & spec,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(weights_source),
          spec,
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

GreedyCausalDecoderRuntime::~GreedyCausalDecoderRuntime() = default;

namespace {

void validate_generate_request(
    const GreedyCausalDecoderSpec & spec,
    const GreedyCausalDecoderRuntime::Prompt & prompt,
    int64_t max_new_tokens) {
    if (prompt.input_ids.empty()) {
        throw std::runtime_error("greedy Qwen decoder prompt is empty");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("greedy Qwen decoder max_new_tokens must be positive");
    }
    const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
    if (prompt_steps + max_new_tokens > spec.max_position_embeddings) {
        throw std::runtime_error("greedy Qwen decoder request exceeds max_position_embeddings");
    }
    const auto & injection = prompt.injection;
    if (injection.tokens < 0 || injection.tokens > prompt_steps ||
        static_cast<int64_t>(injection.positions.size()) != injection.tokens ||
        static_cast<int64_t>(injection.values.size()) != injection.tokens * spec.decoder.stack.hidden_size) {
        throw std::runtime_error("greedy Qwen decoder injection shape does not match the prompt");
    }
    for (const auto position : injection.positions) {
        if (position < 0 || position >= prompt_steps) {
            throw std::runtime_error("greedy Qwen decoder injection position is out of range");
        }
    }
}

}  // namespace

std::vector<int32_t> GreedyCausalDecoderRuntime::generate_incremental(
    const Prompt & prompt,
    int64_t max_new_tokens,
    int64_t cached_prefix_steps) {
    validate_generate_request(impl_->weights->spec(), prompt, max_new_tokens);
    if (cached_prefix_steps < 0 || cached_prefix_steps > static_cast<int64_t>(prompt.input_ids.size())) {
        throw std::runtime_error("greedy Qwen decoder cached prefix exceeds the prompt");
    }
    return impl_->generate_reusing_graphs(prompt, max_new_tokens, cached_prefix_steps, /*capacity_bucket=*/512);
}

std::vector<int32_t> GreedyCausalDecoderRuntime::generate(const Prompt & prompt, int64_t max_new_tokens, bool reuse_graphs) {
    const auto & spec = impl_->weights->spec();
    validate_generate_request(spec, prompt, max_new_tokens);
    const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
    const auto & injection = prompt.injection;
    if (reuse_graphs) {
        return impl_->generate_reusing_graphs(prompt, max_new_tokens, /*cached_prefix_steps=*/0, /*capacity_bucket=*/128);
    }
    if (impl_->prefill_graph == nullptr || !impl_->prefill_graph->matches(prompt_steps, injection.tokens)) {
        impl_->prefill_graph.reset();
        impl_->prefill_graph = std::make_unique<PrefillGraph>(
            impl_->weights,
            prompt_steps,
            injection.tokens,
            impl_->prefill_graph_arena_bytes);
    }
    auto prefill = impl_->prefill_graph->run(
        prompt.input_ids,
        injection.values,
        injection.positions);
    const int64_t required_cache_steps = prompt_steps + max_new_tokens;
    if (impl_->decode_graph == nullptr || !impl_->decode_graph->can_run(required_cache_steps)) {
        impl_->decode_graph.reset();
        impl_->decode_graph = std::make_unique<DecodeGraph>(
            impl_->weights,
            required_cache_steps,
            impl_->decode_graph_arena_bytes);
    }
    impl_->decode_graph->import_state(prefill.kv_state);

    std::vector<int32_t> out;
    std::vector<float> logits = std::move(prefill.logits);
    for (int64_t step = 0; step < max_new_tokens; ++step) {
        const int32_t token = argmax_index(logits);
        if (is_eos(spec, token)) {
            break;
        }
        out.push_back(token);
        logits = impl_->decode_graph->run_step(token);
    }
    return out;
}

}  // namespace engine::runtime

#include "engine/models/vibevoice/decoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace {

namespace binding = modules::binding;

constexpr int64_t kScratchTailCachedAttentionMinSteps = 32768;
constexpr int64_t kLargeCacheGrowthStep = 2048;
constexpr int64_t kLayerwisePrefillMinSteps = 2048;
constexpr ggml_type kDecoderCacheType = GGML_TYPE_F16;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t require_head_dim(const VibeVoiceQwen2Config & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("VibeVoice decoder hidden sizes must be positive");
    }
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("VibeVoice decoder attention dimensions must be positive");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("VibeVoice decoder attention heads must be divisible by key/value heads");
    }
    if (config.num_attention_heads * config.head_dim != config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder hidden_size must equal attention heads times head_dim");
    }
    return config.head_dim;
}

core::TensorValue reshape_heads(
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

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue flash_attention_from_grouped_heads_view_kv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_contiguous.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

VibeVoiceQwen2LayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const VibeVoiceQwen2Config & config,
    int64_t layer,
    assets::TensorStorageType weight_storage_type) {
    const int64_t dim = require_head_dim(config);
    const std::string prefix = "model.language_model.layers." + std::to_string(layer);
    VibeVoiceQwen2LayerWeights weights;
    weights.input_norm = source.require_f32_tensor(prefix + ".input_layernorm.weight", {config.hidden_size});
    weights.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        weight_storage_type,
        {config.num_attention_heads * dim, config.hidden_size});
    weights.self_attention.q_bias = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_attention_heads * dim});
    weights.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        weight_storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    weights.self_attention.k_bias = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_key_value_heads * dim});
    weights.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        weight_storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    weights.self_attention.v_bias = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.bias",
        assets::TensorStorageType::F32,
        {config.num_key_value_heads * dim});
    weights.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.num_attention_heads * dim});
    weights.post_norm = source.require_f32_tensor(prefix + ".post_attention_layernorm.weight", {config.hidden_size});
    weights.mlp.gate_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.gate_proj.weight",
        weight_storage_type,
        {config.intermediate_size, config.hidden_size});
    weights.mlp.up_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.up_proj.weight",
        weight_storage_type,
        {config.intermediate_size, config.hidden_size});
    weights.mlp.down_proj.weight = store.load_tensor(
        source,
        prefix + ".mlp.down_proj.weight",
        weight_storage_type,
        {config.hidden_size, config.intermediate_size});
    return weights;
}

int64_t cache_graph_capacity(int64_t required_capacity, int64_t model_capacity) {
    if (required_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cache capacity must be positive");
    }
    if (model_capacity > 0 && required_capacity > model_capacity) {
        throw std::runtime_error("VibeVoice decoder cache requirement exceeds model position capacity");
    }
    int64_t capacity = required_capacity;
    if (required_capacity < kScratchTailCachedAttentionMinSteps) {
        capacity = 1;
        while (capacity < required_capacity) {
            if (capacity > std::numeric_limits<int64_t>::max() / 2) {
                throw std::runtime_error("VibeVoice decoder cache capacity overflow");
            }
            capacity *= 2;
        }
    } else {
        capacity = ((required_capacity + kLargeCacheGrowthStep - 1) / kLargeCacheGrowthStep) *
            kLargeCacheGrowthStep;
    }
    return model_capacity > 0 ? std::min(capacity, model_capacity) : capacity;
}

runtime::TransformerKVState empty_decoder_state(size_t layers) {
    runtime::TransformerKVState state;
    state.layers.resize(layers);
    return state;
}

std::vector<ggml_fp16_t> build_causal_prefill_mask_values(int64_t steps) {
    const ggml_fp16_t masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const ggml_fp16_t visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> values(static_cast<size_t>(steps * steps), visible);
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = q + 1; k < steps; ++k) {
            values[static_cast<size_t>(q * steps + k)] = masked;
        }
    }
    return values;
}

std::vector<int32_t> build_position_values(int64_t steps) {
    std::vector<int32_t> positions(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return positions;
}

void write_causal_prefill_mask(ggml_tensor * tensor, int64_t batch_size, int64_t steps) {
    const auto mask_values = build_causal_prefill_mask_values(steps);
    if (batch_size == 1) {
        ggml_backend_tensor_set(
            tensor,
            mask_values.data(),
            0,
            mask_values.size() * sizeof(ggml_fp16_t));
        return;
    }
    std::vector<ggml_fp16_t> batch_mask;
    batch_mask.reserve(static_cast<size_t>(batch_size) * mask_values.size());
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        batch_mask.insert(batch_mask.end(), mask_values.begin(), mask_values.end());
    }
    ggml_backend_tensor_set(
        tensor,
        batch_mask.data(),
        0,
        batch_mask.size() * sizeof(ggml_fp16_t));
}

}  // namespace

VibeVoiceQwen2Weights load_vibevoice_decoder_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("VibeVoice decoder requires model weights");
    }
    const auto & config = assets.config.decoder;
    require_head_dim(config);
    VibeVoiceQwen2Weights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibevoice.decoder.weights",
        weight_context_bytes);
    const auto queue_started = std::chrono::steady_clock::now();
    weights.token_embedding = weights.store->load_tensor(
        *assets.model_weights,
        "model.language_model.embed_tokens.weight",
        weight_storage_type,
        {config.vocab_size, config.hidden_size});
    if (config.tie_word_embeddings) {
        weights.lm_head = weights.token_embedding;
    } else {
        weights.lm_head = weights.store->load_tensor(
            *assets.model_weights,
            assets.model_weights->require_tensor_name({"lm_head.weight", "model.lm_head.weight"}),
            weight_storage_type,
            {config.vocab_size, config.hidden_size});
    }
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        weights.layers.push_back(load_layer_weights(
            *weights.store,
            *assets.model_weights,
            config,
            layer,
            weight_storage_type));
    }
    weights.norm = assets.model_weights->require_f32_tensor("model.language_model.norm.weight", {config.hidden_size});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weight_queue_ms",
        engine::debug::elapsed_ms(queue_started));
    const auto upload_started = std::chrono::steady_clock::now();
    weights.store->upload();
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weight_upload_ms",
        engine::debug::elapsed_ms(upload_started));
    return weights;
}

class VibeVoiceQwen2EmbeddingGraph {
public:
    VibeVoiceQwen2EmbeddingGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t steps,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          steps_(steps) {
        if (steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder embedding graph requires positive steps");
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder embedding graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.embed_tokens"};
        input_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, steps_, 1);
        auto ids = core::wrap_tensor(
            input_ids_,
            core::TensorShape::from_dims({1, steps_}),
            GGML_TYPE_I32);
        auto output = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                          .build(ctx, ids, runtime_->weights().token_embedding);
        output = core::ensure_backend_addressable_layout(ctx, output);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder embedding graph");
        }
    }

    ~VibeVoiceQwen2EmbeddingGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t steps) const {
        return runtime_ == &runtime && steps_ == steps;
    }

    VibeVoiceTokenEmbeddings run(const std::vector<int32_t> & input_ids) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(input_ids.size()) != steps_) {
            throw std::runtime_error("VibeVoice decoder embedding input size mismatch");
        }
        ggml_backend_tensor_set(input_ids_, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder embedding graph compute failed");
        }

        VibeVoiceTokenEmbeddings out;
        out.steps = steps_;
        out.hidden_size = config.hidden_size;
        out.values.resize(static_cast<size_t>(steps_ * config.hidden_size));
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

private:
    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ids_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class VibeVoiceQwen2PrefillGraph {
public:
    VibeVoiceQwen2PrefillGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t batch_size,
        int64_t prompt_steps,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          batch_size_(batch_size),
          prompt_steps_(prompt_steps),
          layerwise_(prompt_steps >= kLayerwisePrefillMinSteps) {
        if (batch_size_ <= 0) {
            throw std::runtime_error("VibeVoice decoder prefill graph requires positive batch size");
        }
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder prefill graph requires positive prompt steps");
        }
        if (layerwise_) {
            return;
        }
        const auto & config = runtime_->assets().config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder prefill graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.prefill"};
        auto x = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_size_, prompt_steps_, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions_value = core::wrap_tensor(
            positions_,
            core::TensorShape::from_dims({prompt_steps_}),
            GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, prompt_steps_, prompt_steps_, 1, batch_size_);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({batch_size_, 1, prompt_steps_, prompt_steps_}),
            GGML_TYPE_F16);

        auto & constants = runtime_->constants();
        constants.begin_graph();
        for (const auto & layer : runtime_->weights().layers) {
            auto layer_out = build_vibevoice_decoder_layer(
                ctx,
                x,
                positions_value,
                layer,
                config,
                constants,
                std::nullopt,
                std::nullopt,
                attention_mask);
            x = layer_out.output;
            keys_.push_back(layer_out.key.tensor);
            values_.push_back(layer_out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, runtime_->weights().norm));
        hidden_output_ = x.tensor;
        auto logits = modules::LinearModule(
                          binding::linear_config(config.hidden_size, config.vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, runtime_->weights().lm_head));
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder prefill graph");
        }

        const auto positions = build_position_values(prompt_steps_);
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        write_causal_prefill_mask(attention_mask_, batch_size_, prompt_steps_);
    }

    ~VibeVoiceQwen2PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t batch_size, int64_t prompt_steps) const {
        return runtime_ == &runtime && batch_size_ == batch_size && prompt_steps_ == prompt_steps &&
            layerwise_ == (prompt_steps >= kLayerwisePrefillMinSteps);
    }

    VibeVoiceDecoderPrefillOutput run(const std::vector<float> & embeddings) {
        auto results = run_batch({embeddings});
        if (results.size() != 1) {
            throw std::runtime_error("VibeVoice decoder prefill graph returned unexpected batch size");
        }
        return std::move(results.front());
    }

    std::vector<VibeVoiceDecoderPrefillOutput> run_batch(const std::vector<std::vector<float>> & embeddings) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embeddings.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice decoder prefill embedding batch size mismatch");
        }
        const size_t per_sample = static_cast<size_t>(prompt_steps_ * config.hidden_size);
        for (const auto & sample : embeddings) {
            if (sample.size() != per_sample) {
                throw std::runtime_error("VibeVoice decoder prefill embedding size mismatch");
            }
        }
        if (layerwise_) {
            return run_batch_layerwise(embeddings);
        }
        std::vector<float> input(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & sample = embeddings[static_cast<size_t>(batch)];
            std::copy(
                sample.begin(),
                sample.end(),
                input.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(per_sample)));
        }
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder prefill graph compute failed");
        }

        const size_t logits_per_sample = static_cast<size_t>(config.vocab_size);
        const size_t hidden_per_sample = static_cast<size_t>(config.hidden_size);
        std::vector<float> logits(static_cast<size_t>(batch_size_) * logits_per_sample, 0.0F);
        std::vector<float> hidden(static_cast<size_t>(batch_size_) * hidden_per_sample, 0.0F);
        ggml_backend_tensor_get(logits_output_, logits.data(), 0, logits.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, hidden.data(), 0, hidden.size() * sizeof(float));

        std::vector<VibeVoiceDecoderPrefillOutput> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & sample = out[static_cast<size_t>(batch)];
            sample.result.logits.vocab_size = config.vocab_size;
            sample.result.logits.values.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(logits_per_sample)),
                logits.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(logits_per_sample)));
            sample.result.last_hidden.dims = config.hidden_size;
            sample.result.last_hidden.values.assign(
                hidden.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(hidden_per_sample)),
                hidden.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(hidden_per_sample)));
            sample.state.current_end = prompt_steps_;
            sample.state.layers.resize(keys_.size());
        }

        const size_t layer_values = static_cast<size_t>(
            prompt_steps_ * config.num_key_value_heads * require_head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            std::vector<float> key_values(static_cast<size_t>(batch_size_) * layer_values, 0.0F);
            std::vector<float> value_values(static_cast<size_t>(batch_size_) * layer_values, 0.0F);
            ggml_backend_tensor_get(keys_[layer], key_values.data(), 0, key_values.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], value_values.data(), 0, value_values.size() * sizeof(float));
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                auto & state_layer = out[static_cast<size_t>(batch)].state.layers[layer];
                state_layer.valid_steps = prompt_steps_;
                const auto begin = static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(layer_values));
                const auto end = static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(layer_values));
                state_layer.key.assign(key_values.begin() + begin, key_values.begin() + end);
                state_layer.value.assign(value_values.begin() + begin, value_values.begin() + end);
            }
        }
        return out;
    }

private:
    struct LayerOutput {
        std::vector<float> hidden;
        std::vector<float> key;
        std::vector<float> value;
    };

    class LayerGraph {
    public:
        LayerGraph(
            const VibeVoiceQwen2WeightsRuntime & runtime,
            const VibeVoiceQwen2LayerWeights & layer,
            int64_t batch_size,
            int64_t prompt_steps,
            size_t graph_arena_bytes)
            : runtime_(&runtime),
              batch_size_(batch_size),
              prompt_steps_(prompt_steps),
              constants_(
                  runtime.backend(),
                  runtime.threads(),
                  "vibevoice.decoder.prefill.layer.constants",
                  8ull * 1024ull * 1024ull) {
            const auto & config = runtime_->assets().config.decoder;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize VibeVoice decoder layer prefill graph context");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.prefill.layer"};
            auto x = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, prompt_steps_, config.hidden_size}));
            input_ = x.tensor;
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
            auto positions = core::wrap_tensor(
                positions_,
                core::TensorShape::from_dims({prompt_steps_}),
                GGML_TYPE_I32);
            attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, prompt_steps_, prompt_steps_, 1, batch_size_);
            auto attention_mask = core::wrap_tensor(
                attention_mask_,
                core::TensorShape::from_dims({batch_size_, 1, prompt_steps_, prompt_steps_}),
                GGML_TYPE_F16);

            constants_.begin_graph();
            auto out = build_vibevoice_decoder_layer(
                ctx,
                x,
                positions,
                layer,
                config,
                constants_,
                std::nullopt,
                std::nullopt,
                attention_mask);
            output_ = out.output.tensor;
            key_ = out.key.tensor;
            value_ = out.value.tensor;
            graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
            ggml_set_output(output_);
            ggml_build_forward_expand(graph_, output_);
            constants_.finish_graph();
            constants_.ensure_uploaded();
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate VibeVoice decoder layer prefill graph");
            }

            const auto position_values = build_position_values(prompt_steps_);
            ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            write_causal_prefill_mask(attention_mask_, batch_size_, prompt_steps_);
        }

        ~LayerGraph() {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        LayerOutput run(const std::vector<float> & input) {
            const auto & config = runtime_->assets().config.decoder;
            const size_t hidden_values = static_cast<size_t>(batch_size_ * prompt_steps_ * config.hidden_size);
            if (input.size() != hidden_values) {
                throw std::runtime_error("VibeVoice decoder layer prefill input size mismatch");
            }
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
            ggml_backend_synchronize(runtime_->backend());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("VibeVoice decoder layer prefill graph compute failed");
            }

            LayerOutput out;
            out.hidden.resize(hidden_values);
            ggml_backend_tensor_get(output_, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            const size_t layer_values = static_cast<size_t>(
                batch_size_ * prompt_steps_ * config.num_key_value_heads * require_head_dim(config));
            out.key.resize(layer_values);
            out.value.resize(layer_values);
            ggml_backend_tensor_get(key_, out.key.data(), 0, out.key.size() * sizeof(float));
            ggml_backend_tensor_get(value_, out.value.data(), 0, out.value.size() * sizeof(float));
            return out;
        }

    private:
        const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
        int64_t batch_size_ = 0;
        int64_t prompt_steps_ = 0;
        core::ConstantTensorCache constants_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * attention_mask_ = nullptr;
        ggml_tensor * output_ = nullptr;
        ggml_tensor * key_ = nullptr;
        ggml_tensor * value_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    class FinalGraph {
    public:
        FinalGraph(
            const VibeVoiceQwen2WeightsRuntime & runtime,
            int64_t batch_size,
            size_t graph_arena_bytes)
            : runtime_(&runtime),
              batch_size_(batch_size),
              constants_(
                  runtime.backend(),
                  runtime.threads(),
                  "vibevoice.decoder.prefill.final.constants",
                  4ull * 1024ull * 1024ull) {
            const auto & config = runtime_->assets().config.decoder;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("failed to initialize VibeVoice decoder final prefill graph context");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.prefill.final"};
            auto x = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1, config.hidden_size}));
            input_ = x.tensor;
            constants_.begin_graph();
            x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                    .build(ctx, x, binding::norm_data(constants_, runtime_->weights().norm));
            hidden_output_ = x.tensor;
            auto logits = modules::LinearModule(
                              binding::linear_config(config.hidden_size, config.vocab_size, false))
                              .build(ctx, x, binding::linear_data(constants_, runtime_->weights().lm_head));
            logits_output_ = logits.tensor;
            graph_ = ggml_new_graph_custom(ctx_.get(), 8192, false);
            ggml_set_output(logits_output_);
            ggml_build_forward_expand(graph_, logits_output_);
            constants_.finish_graph();
            constants_.ensure_uploaded();
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate VibeVoice decoder final prefill graph");
            }
        }

        ~FinalGraph() {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        std::vector<VibeVoiceDecoderResult> run(const std::vector<float> & input) {
            const auto & config = runtime_->assets().config.decoder;
            const size_t expected = static_cast<size_t>(batch_size_ * config.hidden_size);
            if (input.size() != expected) {
                throw std::runtime_error("VibeVoice decoder final prefill input size mismatch");
            }
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            core::set_backend_threads(runtime_->backend(), runtime_->threads());
            const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
            ggml_backend_synchronize(runtime_->backend());
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("VibeVoice decoder final prefill graph compute failed");
            }

            const size_t logits_per_sample = static_cast<size_t>(config.vocab_size);
            const size_t hidden_per_sample = static_cast<size_t>(config.hidden_size);
            std::vector<float> logits(static_cast<size_t>(batch_size_) * logits_per_sample, 0.0F);
            std::vector<float> hidden(static_cast<size_t>(batch_size_) * hidden_per_sample, 0.0F);
            ggml_backend_tensor_get(logits_output_, logits.data(), 0, logits.size() * sizeof(float));
            ggml_backend_tensor_get(hidden_output_, hidden.data(), 0, hidden.size() * sizeof(float));

            std::vector<VibeVoiceDecoderResult> results(static_cast<size_t>(batch_size_));
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                auto & result = results[static_cast<size_t>(batch)];
                result.logits.vocab_size = config.vocab_size;
                result.logits.values.assign(
                    logits.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(logits_per_sample)),
                    logits.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(logits_per_sample)));
                result.last_hidden.dims = config.hidden_size;
                result.last_hidden.values.assign(
                    hidden.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(hidden_per_sample)),
                    hidden.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(hidden_per_sample)));
            }
            return results;
        }

    private:
        const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
        int64_t batch_size_ = 0;
        core::ConstantTensorCache constants_;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * hidden_output_ = nullptr;
        ggml_tensor * logits_output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    std::vector<VibeVoiceDecoderPrefillOutput> run_batch_layerwise(
        const std::vector<std::vector<float>> & embeddings) const {
        const auto & config = runtime_->assets().config.decoder;
        const size_t per_sample = static_cast<size_t>(prompt_steps_ * config.hidden_size);
        std::vector<float> hidden(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & sample = embeddings[static_cast<size_t>(batch)];
            std::copy(
                sample.begin(),
                sample.end(),
                hidden.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(per_sample)));
        }

        std::vector<VibeVoiceDecoderPrefillOutput> out(static_cast<size_t>(batch_size_));
        for (auto & sample : out) {
            sample.state.current_end = prompt_steps_;
            sample.state.layers.resize(runtime_->weights().layers.size());
        }
        const size_t layer_values = static_cast<size_t>(
            prompt_steps_ * config.num_key_value_heads * require_head_dim(config));

        for (size_t layer = 0; layer < runtime_->weights().layers.size(); ++layer) {
            LayerGraph graph(
                *runtime_,
                runtime_->weights().layers[layer],
                batch_size_,
                prompt_steps_,
                512ull * 1024ull * 1024ull);
            auto layer_out = graph.run(hidden);
            hidden = std::move(layer_out.hidden);
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                auto & state_layer = out[static_cast<size_t>(batch)].state.layers[layer];
                state_layer.valid_steps = prompt_steps_;
                const auto begin = static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(layer_values));
                const auto end = static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(layer_values));
                state_layer.key.assign(layer_out.key.begin() + begin, layer_out.key.begin() + end);
                state_layer.value.assign(layer_out.value.begin() + begin, layer_out.value.begin() + end);
            }
        }

        std::vector<float> last_hidden(static_cast<size_t>(batch_size_ * config.hidden_size), 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto begin = hidden.begin() +
                static_cast<std::ptrdiff_t>((batch * prompt_steps_ + prompt_steps_ - 1) * config.hidden_size);
            std::copy(
                begin,
                begin + static_cast<std::ptrdiff_t>(config.hidden_size),
                last_hidden.begin() + static_cast<std::ptrdiff_t>(batch * config.hidden_size));
        }
        FinalGraph final_graph(*runtime_, batch_size_, 64ull * 1024ull * 1024ull);
        auto results = final_graph.run(last_hidden);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            out[static_cast<size_t>(batch)].result = std::move(results[static_cast<size_t>(batch)]);
        }
        return out;
    }

    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t prompt_steps_ = 0;
    bool layerwise_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer_scratch_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask);

class VibeVoiceQwen2CachedStepGraph {
public:
    VibeVoiceQwen2CachedStepGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder cached step graph requires positive cache steps");
        }
        const auto & config = runtime_->assets().config.decoder;
        const int64_t head_dim = require_head_dim(config);
        scratch_tail_ = cache_steps_ >= kScratchTailCachedAttentionMinSteps;
        const int64_t cache_tensor_steps = cache_steps_ + (scratch_tail_ ? 1 : 0);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder cached step graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.cached_step"};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_tensor_steps, 1, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_tensor_steps}),
            GGML_TYPE_F16);

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(runtime_->weights().layers.size());
        cache_values.reserve(runtime_->weights().layers.size());
        auto & constants = runtime_->constants();
        constants.begin_graph();
        for (const auto & layer : runtime_->weights().layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                runtime_->cache_type(),
                core::TensorShape::from_dims({1, cache_tensor_steps, config.num_key_value_heads, head_dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                runtime_->cache_type(),
                core::TensorShape::from_dims({1, cache_tensor_steps, config.num_key_value_heads, head_dim})));
            auto layer_out = scratch_tail_
                ? build_vibevoice_decoder_layer_scratch_tail(
                      ctx,
                      graph_,
                      x,
                      positions_value,
                      layer,
                      config,
                      constants,
                      cache_keys.back(),
                      cache_values.back(),
                      attention_mask_value)
                : build_vibevoice_decoder_layer_static_tail(
                      ctx,
                      x,
                      positions_value,
                      layer,
                      config,
                      constants,
                      cache_keys.back(),
                      cache_values.back(),
                      cache_slot_value,
                      attention_mask_value);
            x = layer_out.output;
            if (scratch_tail_) {
                key_sources_.push_back(ggml_view_1d(ctx_.get(), layer_out.key.tensor, config.num_key_value_heads * head_dim, 0));
                value_sources_.push_back(ggml_view_1d(ctx_.get(), layer_out.value.tensor, config.num_key_value_heads * head_dim, 0));
            }
        }
        runtime::TransformerKVCacheOptions cache_options;
        cache_options.allow_f16_storage = runtime_->cache_type() == GGML_TYPE_F16;
        cache_options.allow_bf16_storage = runtime_->cache_type() == GGML_TYPE_BF16;
        cache_ = runtime::TransformerKVCache(
            cache_tensor_steps,
            config.num_key_value_heads * head_dim,
            std::move(cache_keys),
            std::move(cache_values),
            cache_options);
        if (scratch_tail_) {
            build_transfer_views(config.num_key_value_heads * head_dim);
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, runtime_->weights().norm));
        hidden_output_ = x.tensor;
        auto logits = modules::LinearModule(
                          binding::linear_config(config.hidden_size, config.vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, runtime_->weights().lm_head));
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder cached step graph");
        }
        attention_mask_buffer_.assign(static_cast<size_t>(cache_tensor_steps), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~VibeVoiceQwen2CachedStepGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_decode(const VibeVoiceQwen2WeightsRuntime & runtime, int64_t required_capacity) const {
        return runtime_ == &runtime && cache_steps_ >= required_capacity;
    }

    int64_t current_end() const noexcept {
        return cache_.current_end();
    }

    void import_state(const runtime::TransformerKVState & state) {
        cache_.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return cache_.export_state();
    }

    VibeVoiceDecoderResult run_step(const std::vector<float> & embedding) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
            throw std::runtime_error("VibeVoice decoder cached step embedding size mismatch");
        }
        if (cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("VibeVoice decoder cached step exceeds cache capacity");
        }
        ggml_backend_tensor_set(input_, embedding.data(), 0, embedding.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(position));
        const int32_t cache_slot = static_cast<int32_t>(cache_.valid_steps());
        if (!scratch_tail_) {
            ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));
        }
        std::fill(
            attention_mask_buffer_.begin(),
            attention_mask_buffer_.end(),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t i = 0; i < cache_.valid_steps(); ++i) {
            attention_mask_buffer_[static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
        }
        const size_t current_slot = scratch_tail_
            ? static_cast<size_t>(cache_steps_)
            : static_cast<size_t>(cache_slot);
        attention_mask_buffer_[current_slot] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder cached step graph compute failed");
        }

        VibeVoiceDecoderResult out;
        out.logits.vocab_size = config.vocab_size;
        out.logits.values.resize(static_cast<size_t>(config.vocab_size));
        out.last_hidden.dims = config.hidden_size;
        out.last_hidden.values.resize(static_cast<size_t>(config.hidden_size));
        ggml_backend_tensor_get(logits_output_, out.logits.values.data(), 0, out.logits.values.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, out.last_hidden.values.data(), 0, out.last_hidden.values.size() * sizeof(float));
        if (scratch_tail_) {
            const size_t dst_slot = static_cast<size_t>(cache_.valid_steps());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
                ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
            }
        }
        cache_.advance_after_direct_append(1);
        return out;
    }

private:
    void build_transfer_views(int64_t step_elems) {
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * ggml_type_size(cache_.key_tensor(0).type);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx_.get(), cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
    }

    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t cache_steps_ = 0;
    bool scratch_tail_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    runtime::TransformerKVCache cache_;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class VibeVoiceQwen2CachedBatchStepGraph {
public:
    VibeVoiceQwen2CachedBatchStepGraph(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        int64_t batch_size,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : runtime_(&runtime),
          batch_size_(batch_size),
          cache_steps_(cache_steps) {
        if (batch_size_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("VibeVoice decoder cached batch graph requires positive dimensions");
        }
        const auto & config = runtime_->assets().config.decoder;
        const int64_t head_dim = require_head_dim(config);
        step_elems_ = config.num_key_value_heads * head_dim;

        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice decoder cached batch graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "vibevoice.decoder.cached_batch_step"};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size_, 1, config.hidden_size}));
        input_ = x.tensor;
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions_value = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot_value = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto attention_mask = core::make_tensor(
            ctx,
            GGML_TYPE_F16,
            core::TensorShape::from_dims({batch_size_, 1, 1, cache_steps_}));
        attention_mask_ = attention_mask.tensor;

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        auto & constants = runtime_->constants();
        constants.begin_graph();
        for (const auto & layer : runtime_->weights().layers) {
            cache_keys_.push_back(core::make_tensor(
                ctx,
                runtime_->cache_type(),
                core::TensorShape::from_dims({batch_size_, cache_steps_, config.num_key_value_heads, head_dim})));
            cache_values_.push_back(core::make_tensor(
                ctx,
                runtime_->cache_type(),
                core::TensorShape::from_dims({batch_size_, cache_steps_, config.num_key_value_heads, head_dim})));
            auto layer_out = build_vibevoice_decoder_layer_static_tail(
                ctx,
                x,
                positions_value,
                layer,
                config,
                constants,
                cache_keys_.back(),
                cache_values_.back(),
                cache_slot_value,
                attention_mask);
            x = layer_out.output;
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, binding::norm_data(constants, runtime_->weights().norm));
        hidden_output_ = x.tensor;
        auto logits = modules::LinearModule(
                          binding::linear_config(config.hidden_size, config.vocab_size, false))
                          .build(ctx, x, binding::linear_data(constants, runtime_->weights().lm_head));
        logits_output_ = logits.tensor;
        ggml_set_output(logits_output_);
        ggml_build_forward_expand(graph_, logits_output_);
        constants.finish_graph();
        constants.ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VibeVoice decoder cached batch graph");
        }
        attention_mask_buffer_.assign(
            static_cast<size_t>(batch_size_ * cache_steps_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~VibeVoiceQwen2CachedBatchStepGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(
        const VibeVoiceQwen2WeightsRuntime & runtime,
        const std::vector<VibeVoiceQwen2CachedState *> & states,
        int64_t required_capacity) const {
        if (runtime_ != &runtime || cache_steps_ < required_capacity ||
            static_cast<int64_t>(states.size()) != batch_size_ || states.size() != states_.size()) {
            return false;
        }
        for (size_t i = 0; i < states.size(); ++i) {
            if (states_[i] != states[i]) {
                return false;
            }
        }
        return true;
    }

    int64_t current_end() const noexcept {
        return current_end_;
    }

    void import_states(const std::vector<VibeVoiceQwen2CachedState *> & states) {
        if (static_cast<int64_t>(states.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice decoder cached batch state count mismatch");
        }
        states_ = states;
        std::vector<runtime::TransformerKVState> source_states;
        source_states.reserve(states.size());
        for (auto * state : states) {
            if (state == nullptr) {
                throw std::runtime_error("VibeVoice decoder cached batch received null state");
            }
            if (state->batch_owner_ != nullptr && state->batch_owner_ != this) {
                throw std::runtime_error("VibeVoice decoder cached state belongs to another batch graph");
            }
            if (state->graph_has_state_ && state->graph_ != nullptr) {
                source_states.push_back(state->graph_->export_state());
            } else {
                source_states.push_back(state->pending_state_);
            }
        }
        current_end_ = source_states.front().current_end;
        valid_steps_ = source_states.front().layers.empty() ? 0 : source_states.front().layers.front().valid_steps;
        for (const auto & state : source_states) {
            if (state.current_end != current_end_) {
                throw std::runtime_error("VibeVoice decoder cached batch requires matching current_end");
            }
            if (state.layers.size() != cache_keys_.size()) {
                throw std::runtime_error("VibeVoice decoder cached batch layer count mismatch");
            }
            const int64_t state_steps = state.layers.empty() ? 0 : state.layers.front().valid_steps;
            if (state_steps != valid_steps_) {
                throw std::runtime_error("VibeVoice decoder cached batch requires matching valid_steps");
            }
        }
        if (valid_steps_ > cache_steps_) {
            throw std::runtime_error("VibeVoice decoder cached batch state exceeds cache capacity");
        }

        const size_t sample_cache_elems = static_cast<size_t>(cache_steps_ * step_elems_);
        const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
        for (size_t layer = 0; layer < cache_keys_.size(); ++layer) {
            std::vector<float> key_values(static_cast<size_t>(batch_size_) * sample_cache_elems, 0.0F);
            std::vector<float> value_values(static_cast<size_t>(batch_size_) * sample_cache_elems, 0.0F);
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                const auto & layer_state = source_states[static_cast<size_t>(batch)].layers[layer];
                if (layer_state.valid_steps != valid_steps_ ||
                    layer_state.key.size() != keep_elems ||
                    layer_state.value.size() != keep_elems) {
                    throw std::runtime_error("VibeVoice decoder cached batch state payload size mismatch");
                }
                const auto offset = static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(sample_cache_elems));
                std::copy(layer_state.key.begin(), layer_state.key.end(), key_values.begin() + offset);
                std::copy(layer_state.value.begin(), layer_state.value.end(), value_values.begin() + offset);
            }
            if (runtime_->cache_type() == GGML_TYPE_F16) {
                core::write_tensor_f16(cache_keys_[layer], key_values);
                core::write_tensor_f16(cache_values_[layer], value_values);
            } else if (runtime_->cache_type() == GGML_TYPE_BF16) {
                core::write_tensor_bf16(cache_keys_[layer], key_values);
                core::write_tensor_bf16(cache_values_[layer], value_values);
            } else {
                core::write_tensor_f32(cache_keys_[layer], key_values);
                core::write_tensor_f32(cache_values_[layer], value_values);
            }
        }

        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto * state = states_[static_cast<size_t>(batch)];
            state->graph_.reset();
            state->pending_state_ = {};
            state->graph_has_state_ = false;
            state->batch_owner_ = this;
        }
    }

    std::vector<VibeVoiceDecoderResult> run_step(const std::vector<std::vector<float>> & embeddings) {
        const auto & config = runtime_->assets().config.decoder;
        if (static_cast<int64_t>(embeddings.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice decoder cached batch embedding count mismatch");
        }
        if (valid_steps_ >= cache_steps_) {
            throw std::runtime_error("VibeVoice decoder cached batch step exceeds cache capacity");
        }
        std::vector<float> input(static_cast<size_t>(batch_size_ * config.hidden_size), 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & embedding = embeddings[static_cast<size_t>(batch)];
            if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
                throw std::runtime_error("VibeVoice decoder cached batch embedding size mismatch");
            }
            std::copy(
                embedding.begin(),
                embedding.end(),
                input.begin() + static_cast<std::ptrdiff_t>(batch * config.hidden_size));
        }
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        const int32_t position = static_cast<int32_t>(current_end_);
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(position));
        const int32_t cache_slot = static_cast<int32_t>(valid_steps_);
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(cache_slot));

        std::fill(
            attention_mask_buffer_.begin(),
            attention_mask_buffer_.end(),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const size_t base = static_cast<size_t>(batch * cache_steps_);
            for (int64_t i = 0; i < valid_steps_; ++i) {
                attention_mask_buffer_[base + static_cast<size_t>(i)] = ggml_fp32_to_fp16(0.0F);
            }
            attention_mask_buffer_[base + static_cast<size_t>(cache_slot)] = ggml_fp32_to_fp16(0.0F);
        }
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_buffer_.data(),
            0,
            attention_mask_buffer_.size() * sizeof(ggml_fp16_t));

        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice decoder cached batch step graph compute failed");
        }

        const size_t logits_per_sample = static_cast<size_t>(config.vocab_size);
        const size_t hidden_per_sample = static_cast<size_t>(config.hidden_size);
        std::vector<float> logits(static_cast<size_t>(batch_size_) * logits_per_sample, 0.0F);
        std::vector<float> hidden(static_cast<size_t>(batch_size_) * hidden_per_sample, 0.0F);
        ggml_backend_tensor_get(logits_output_, logits.data(), 0, logits.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_output_, hidden.data(), 0, hidden.size() * sizeof(float));

        ++valid_steps_;
        ++current_end_;

        std::vector<VibeVoiceDecoderResult> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & result = out[static_cast<size_t>(batch)];
            result.logits.vocab_size = config.vocab_size;
            result.logits.values.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(logits_per_sample)),
                logits.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(logits_per_sample)));
            result.last_hidden.dims = config.hidden_size;
            result.last_hidden.values.assign(
                hidden.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(hidden_per_sample)),
                hidden.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(hidden_per_sample)));
        }
        return out;
    }

    void export_to_states() {
        if (states_.empty()) {
            return;
        }
        const size_t sample_cache_elems = static_cast<size_t>(cache_steps_ * step_elems_);
        const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
        std::vector<runtime::TransformerKVState> outputs(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & state = outputs[static_cast<size_t>(batch)];
            state.current_end = current_end_;
            state.layers.resize(cache_keys_.size());
            for (auto & layer : state.layers) {
                layer.valid_steps = valid_steps_;
            }
        }
        for (size_t layer = 0; layer < cache_keys_.size(); ++layer) {
            std::vector<float> key_values;
            std::vector<float> value_values;
            if (runtime_->cache_type() == GGML_TYPE_F16) {
                key_values = core::read_tensor_f16(cache_keys_[layer].tensor);
                value_values = core::read_tensor_f16(cache_values_[layer].tensor);
            } else if (runtime_->cache_type() == GGML_TYPE_BF16) {
                key_values = core::read_tensor_bf16(cache_keys_[layer].tensor);
                value_values = core::read_tensor_bf16(cache_values_[layer].tensor);
            } else {
                key_values = core::read_tensor_f32(cache_keys_[layer].tensor);
                value_values = core::read_tensor_f32(cache_values_[layer].tensor);
            }
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                const size_t offset = static_cast<size_t>(batch) * sample_cache_elems;
                auto & layer_state = outputs[static_cast<size_t>(batch)].layers[layer];
                layer_state.key.assign(
                    key_values.begin() + static_cast<std::ptrdiff_t>(offset),
                    key_values.begin() + static_cast<std::ptrdiff_t>(offset + keep_elems));
                layer_state.value.assign(
                    value_values.begin() + static_cast<std::ptrdiff_t>(offset),
                    value_values.begin() + static_cast<std::ptrdiff_t>(offset + keep_elems));
            }
        }
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto * state = states_[static_cast<size_t>(batch)];
            state->pending_state_ = std::move(outputs[static_cast<size_t>(batch)]);
            state->graph_has_state_ = false;
            state->batch_owner_ = nullptr;
        }
        states_.clear();
    }

    const VibeVoiceQwen2WeightsRuntime * runtime_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t cache_steps_ = 0;
    int64_t step_elems_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * logits_output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_buffer_;
    std::vector<core::TensorValue> cache_keys_;
    std::vector<core::TensorValue> cache_values_;
    std::vector<VibeVoiceQwen2CachedState *> states_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

VibeVoiceQwen2CachedState::VibeVoiceQwen2CachedState() = default;
VibeVoiceQwen2CachedState::~VibeVoiceQwen2CachedState() = default;
VibeVoiceQwen2CachedState::VibeVoiceQwen2CachedState(VibeVoiceQwen2CachedState &&) noexcept = default;
VibeVoiceQwen2CachedState & VibeVoiceQwen2CachedState::operator=(VibeVoiceQwen2CachedState &&) noexcept = default;

VibeVoiceQwen2WeightsRuntime::VibeVoiceQwen2WeightsRuntime(
    std::shared_ptr<const VibeVoiceAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      cache_type_(kDecoderCacheType),
      threads_(threads) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice decoder weights runtime requires assets");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("VibeVoice decoder weights runtime requires positive thread count");
    }
    const auto backend_started = std::chrono::steady_clock::now();
    backend_ = core::init_backend({backend_type, device, threads_});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_backend_init_ms",
        engine::debug::elapsed_ms(backend_started));
    const auto weights_started = std::chrono::steady_clock::now();
    weights_ = std::make_shared<VibeVoiceQwen2Weights>(
        load_vibevoice_decoder_weights(
            *assets_,
            backend_,
            backend_type,
            weight_context_bytes,
            weight_storage_type));
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.decoder_weights_load_ms",
        engine::debug::elapsed_ms(weights_started));
    constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.decoder.constants",
        constant_context_bytes);
}

VibeVoiceQwen2WeightsRuntime::~VibeVoiceQwen2WeightsRuntime() {
    cached_batch_graphs_.clear();
    prefill_graph_.reset();
    embedding_graph_.reset();
    constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

const VibeVoiceAssets & VibeVoiceQwen2WeightsRuntime::assets() const noexcept {
    return *assets_;
}

const VibeVoiceQwen2Weights & VibeVoiceQwen2WeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t VibeVoiceQwen2WeightsRuntime::backend() const noexcept {
    return backend_;
}

core::ConstantTensorCache & VibeVoiceQwen2WeightsRuntime::constants() const noexcept {
    return *constants_;
}

ggml_type VibeVoiceQwen2WeightsRuntime::cache_type() const noexcept {
    return cache_type_;
}

int VibeVoiceQwen2WeightsRuntime::threads() const noexcept {
    return threads_;
}

VibeVoiceTokenEmbeddings VibeVoiceQwen2WeightsRuntime::embed_tokens(
    const std::vector<int32_t> & input_ids) const {
    if (input_ids.empty()) {
        throw std::runtime_error("VibeVoice decoder embedding requires at least one token");
    }
    const int64_t steps = static_cast<int64_t>(input_ids.size());
    if (embedding_graph_ == nullptr || !embedding_graph_->matches(*this, steps)) {
        embedding_graph_.reset();
        embedding_graph_ = std::make_unique<VibeVoiceQwen2EmbeddingGraph>(
            *this,
            steps,
            64ull * 1024ull * 1024ull);
    }
    return embedding_graph_->run(input_ids);
}

VibeVoiceDecoderPrefillOutput VibeVoiceQwen2WeightsRuntime::prefill_embeddings(
    const std::vector<float> & embeddings,
    int64_t steps) const {
    const auto & config = assets_->config.decoder;
    if (steps <= 0) {
        throw std::runtime_error("VibeVoice decoder prefill requires positive steps");
    }
    if (static_cast<int64_t>(embeddings.size()) != steps * config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder prefill embedding payload size mismatch");
    }
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*this, 1, steps)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<VibeVoiceQwen2PrefillGraph>(
            *this,
            1,
            steps,
            1024ull * 1024ull * 1024ull);
    }
    return prefill_graph_->run(embeddings);
}

std::vector<VibeVoiceDecoderPrefillOutput> VibeVoiceQwen2WeightsRuntime::prefill_embeddings_batch(
    const std::vector<std::vector<float>> & embeddings,
    int64_t steps) const {
    const auto & config = assets_->config.decoder;
    if (embeddings.empty()) {
        throw std::runtime_error("VibeVoice decoder batch prefill requires at least one sample");
    }
    if (steps <= 0) {
        throw std::runtime_error("VibeVoice decoder batch prefill requires positive steps");
    }
    const size_t expected_size = static_cast<size_t>(steps * config.hidden_size);
    for (const auto & sample : embeddings) {
        if (sample.size() != expected_size) {
            throw std::runtime_error("VibeVoice decoder batch prefill embedding payload size mismatch");
        }
    }
    const int64_t batch_size = static_cast<int64_t>(embeddings.size());
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*this, batch_size, steps)) {
        prefill_graph_.reset();
        prefill_graph_ = std::make_unique<VibeVoiceQwen2PrefillGraph>(
            *this,
            batch_size,
            steps,
            1024ull * 1024ull * 1024ull);
    }
    return prefill_graph_->run_batch(embeddings);
}

void VibeVoiceQwen2WeightsRuntime::release_prompt_graphs() const {
    prefill_graph_.reset();
    embedding_graph_.reset();
}

VibeVoiceQwen2CachedBatchStepGraph * VibeVoiceQwen2WeightsRuntime::find_cached_batch_graph(
    const VibeVoiceQwen2CachedState & state) const {
    if (state.batch_owner_ == nullptr) {
        return nullptr;
    }
    for (const auto & graph : cached_batch_graphs_) {
        if (graph.get() == state.batch_owner_) {
            return graph.get();
        }
    }
    throw std::runtime_error("VibeVoice decoder cached state belongs to an unknown batch graph");
}

void VibeVoiceQwen2WeightsRuntime::export_and_drop_cached_batch_graph(
    const VibeVoiceQwen2CachedBatchStepGraph * owner) const {
    if (owner == nullptr) {
        return;
    }
    for (auto it = cached_batch_graphs_.begin(); it != cached_batch_graphs_.end(); ++it) {
        if (it->get() == owner) {
            (*it)->export_to_states();
            cached_batch_graphs_.erase(it);
            return;
        }
    }
    throw std::runtime_error("VibeVoice decoder cached state belongs to an unknown batch graph");
}

void VibeVoiceQwen2WeightsRuntime::reset_cached_state(
    VibeVoiceQwen2CachedState & state,
    runtime::TransformerKVState prefill_state) const {
    if (state.batch_owner_ != nullptr) {
        export_and_drop_cached_batch_graph(state.batch_owner_);
    }
    state.pending_state_ = std::move(prefill_state);
    state.graph_has_state_ = false;
    state.batch_owner_ = nullptr;
}

VibeVoiceDecoderResult VibeVoiceQwen2WeightsRuntime::cached_step(
    const std::vector<float> & embedding,
    VibeVoiceQwen2CachedState & state,
    int64_t cache_capacity) const {
    const auto & config = assets_->config.decoder;
    if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
        throw std::runtime_error("VibeVoice decoder cached step embedding payload size mismatch");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached step requires positive cache capacity");
    }
    if (state.batch_owner_ != nullptr) {
        export_and_drop_cached_batch_graph(state.batch_owner_);
    }
    const int64_t current_end = state.graph_has_state_ && state.graph_ != nullptr
        ? state.graph_->current_end()
        : state.pending_state_.current_end;
    const int64_t required_capacity = cache_graph_capacity(
        std::max<int64_t>(cache_capacity, current_end + 1),
        config.max_position_embeddings);
    if (state.graph_ != nullptr && state.graph_has_state_ && !state.graph_->can_decode(*this, required_capacity)) {
        state.pending_state_ = state.graph_->export_state();
        state.graph_has_state_ = false;
    }
    if (state.graph_ == nullptr || !state.graph_->can_decode(*this, required_capacity)) {
        state.graph_.reset();
        const size_t graph_arena_bytes = required_capacity >= kScratchTailCachedAttentionMinSteps
            ? 1536ull * 1024ull * 1024ull
            : 1024ull * 1024ull * 1024ull;
        state.graph_ = std::make_unique<VibeVoiceQwen2CachedStepGraph>(
            *this,
            required_capacity,
            graph_arena_bytes);
    }
    if (!state.graph_has_state_) {
        if (state.pending_state_.layers.empty() && state.pending_state_.current_end == 0) {
            state.pending_state_ = empty_decoder_state(weights_->layers.size());
        }
        state.graph_->import_state(state.pending_state_);
        state.pending_state_ = {};
        state.graph_has_state_ = true;
    }
    return state.graph_->run_step(embedding);
}

std::vector<VibeVoiceDecoderResult> VibeVoiceQwen2WeightsRuntime::cached_step_batch(
    const std::vector<std::vector<float>> & embeddings,
    const std::vector<VibeVoiceQwen2CachedState *> & states,
    int64_t cache_capacity) const {
    const auto & config = assets_->config.decoder;
    if (embeddings.empty()) {
        throw std::runtime_error("VibeVoice decoder cached batch step requires at least one sample");
    }
    if (embeddings.size() != states.size()) {
        throw std::runtime_error("VibeVoice decoder cached batch step state count mismatch");
    }
    if (cache_capacity <= 0) {
        throw std::runtime_error("VibeVoice decoder cached batch step requires positive cache capacity");
    }
    for (const auto & embedding : embeddings) {
        if (static_cast<int64_t>(embedding.size()) != config.hidden_size) {
            throw std::runtime_error("VibeVoice decoder cached batch step embedding payload size mismatch");
        }
    }
    for (auto * state : states) {
        if (state == nullptr) {
            throw std::runtime_error("VibeVoice decoder cached batch step received null state");
        }
    }
    if (embeddings.size() == 1) {
        return {cached_step(embeddings.front(), *states.front(), cache_capacity)};
    }

    auto state_current_end = [&](const VibeVoiceQwen2CachedState & state) -> int64_t {
        if (state.batch_owner_ != nullptr) {
            return find_cached_batch_graph(state)->current_end();
        }
        return state.graph_has_state_ && state.graph_ != nullptr
            ? state.graph_->current_end()
            : state.pending_state_.current_end;
    };

    const int64_t current_end = state_current_end(*states.front());
    for (auto * state : states) {
        if (state_current_end(*state) != current_end) {
            throw std::runtime_error("VibeVoice decoder cached batch step requires matching current_end");
        }
    }
    const int64_t required_capacity = cache_graph_capacity(
        std::max<int64_t>(cache_capacity, current_end + 1),
        config.max_position_embeddings);

    VibeVoiceQwen2CachedBatchStepGraph * graph = nullptr;
    for (const auto & candidate : cached_batch_graphs_) {
        if (candidate->matches(*this, states, required_capacity)) {
            graph = candidate.get();
            break;
        }
    }

    if (graph == nullptr) {
        std::vector<const VibeVoiceQwen2CachedBatchStepGraph *> owners;
        for (auto * state : states) {
            if (state->batch_owner_ == nullptr) {
                continue;
            }
            if (std::find(owners.begin(), owners.end(), state->batch_owner_) == owners.end()) {
                owners.push_back(state->batch_owner_);
            }
        }
        for (const auto * owner : owners) {
            export_and_drop_cached_batch_graph(owner);
        }

        cached_batch_graphs_.push_back(std::make_unique<VibeVoiceQwen2CachedBatchStepGraph>(
            *this,
            static_cast<int64_t>(states.size()),
            required_capacity,
            1024ull * 1024ull * 1024ull));
        graph = cached_batch_graphs_.back().get();
        graph->import_states(states);
    }
    return graph->run_step(embeddings);
}

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) {
    if (prefix_key.has_value() != prefix_value.has_value()) {
        throw std::runtime_error("VibeVoice decoder layer requires both prefix key and value or neither");
    }
    const int64_t dim = require_head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, true));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.input_norm));
    auto q = q_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.q_weight, weights.self_attention.q_bias));
    auto k = k_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.k_weight, weights.self_attention.k_bias));
    auto v = v_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.v_weight, weights.self_attention.v_bias));
    q = reshape_heads(ctx, q, config.num_attention_heads, dim);
    k = reshape_heads(ctx, k, config.num_key_value_heads, dim);
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    core::TensorValue context;
    if (!prefix_key.has_value() && attention_mask.has_value()) {
        q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
        auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k);
        auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v);
        context = flash_attention_from_grouped_heads_view_kv(ctx, q_heads, k_heads, v_heads, dim, *attention_mask);
    } else {
        auto k_heads = repeat_kv_heads(
            ctx,
            modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k),
            kv_repeats);
        auto v_heads = repeat_kv_heads(
            ctx,
            modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v),
            kv_repeats);
        context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
        context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto attn_out = o_proj.build(ctx, context, binding::linear_data(constants, weights.self_attention.out_weight));
    auto x = add.build(
        ctx,
        input,
        attn_out);

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.post_norm));
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, weights.mlp.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, weights.mlp.up_proj);
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), weights.mlp.down_proj);
    auto output = add.build(ctx, x, ff);
    return {output, k, v};
}

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer_static_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t dim = require_head_dim(config);
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, true));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.input_norm));
    auto q = q_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.q_weight, weights.self_attention.q_bias));
    auto k = k_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.k_weight, weights.self_attention.k_bias));
    auto v = v_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.v_weight, weights.self_attention.v_bias));
    q = reshape_heads(ctx, q, config.num_attention_heads, dim);
    k = reshape_heads(ctx, k, config.num_key_value_heads, dim);
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    const modules::FastKVSetRowsModule set_rows(
        cache_key.type == GGML_TYPE_F32
            ? modules::FastKVSetRowsConfig{}
            : modules::FastKVSetRowsConfig{modules::FastKVSetRowsMode::BackendViewOptimized});
    auto updated_cache_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_cache_value = set_rows.build(ctx, cache_value, v, cache_slot);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_key.shape.rank}).build(ctx, updated_cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_value.shape.rank}).build(ctx, updated_cache_value);
    auto context = flash_attention_from_grouped_heads_view_kv(
        ctx,
        q_heads,
        k_heads,
        v_heads,
        dim,
        attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], 1, config.num_attention_heads * dim}));
    auto attn_out = o_proj.build(ctx, context, binding::linear_data(constants, weights.self_attention.out_weight));
    auto x = add.build(
        ctx,
        input,
        attn_out);

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.post_norm));
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, weights.mlp.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, weights.mlp.up_proj);
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), weights.mlp.down_proj);
    auto output = add.build(ctx, x, ff);
    return {output, k, v};
}

VibeVoiceDecoderLayerOutputs build_vibevoice_decoder_layer_scratch_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const VibeVoiceQwen2LayerWeights & weights,
    const VibeVoiceQwen2Config & config,
    core::ConstantTensorCache & constants,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask) {
    if (graph == nullptr) {
        throw std::runtime_error("VibeVoice decoder scratch-tail layer requires a graph");
    }
    const int64_t dim = require_head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
    const modules::LinearModule q_proj(
        binding::linear_config(config.hidden_size, config.num_attention_heads * dim, true));
    const modules::LinearModule k_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule v_proj(
        binding::linear_config(config.hidden_size, config.num_key_value_heads * dim, true));
    const modules::LinearModule o_proj(
        binding::linear_config(config.num_attention_heads * dim, config.hidden_size, false));
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(constants, weights.input_norm));
    auto q = q_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.q_weight, weights.self_attention.q_bias));
    auto k = k_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.k_weight, weights.self_attention.k_bias));
    auto v = v_proj.build(
        ctx,
        attn_in,
        binding::linear_data(constants, weights.self_attention.v_weight, weights.self_attention.v_bias));
    q = reshape_heads(ctx, q, config.num_attention_heads, dim);
    k = reshape_heads(ctx, k, config.num_key_value_heads, dim);
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);

    const size_t scratch_offset = static_cast<size_t>(
        scratch_slot * config.num_key_value_heads * dim * static_cast<int64_t>(sizeof(float)));
    auto * key_tail = ggml_view_1d(
        ctx.ggml,
        cache_key.tensor,
        config.num_key_value_heads * dim,
        scratch_offset);
    auto * value_tail = ggml_view_1d(
        ctx.ggml,
        cache_value.tensor,
        config.num_key_value_heads * dim,
        scratch_offset);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_value.shape.rank}).build(ctx, cache_value);
    auto context = flash_attention_from_grouped_heads_view_kv(
        ctx,
        q_heads,
        k_heads,
        v_heads,
        dim,
        attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], 1, config.num_attention_heads * dim}));
    auto attn_out = o_proj.build(ctx, context, binding::linear_data(constants, weights.self_attention.out_weight));
    auto x = add.build(ctx, input, attn_out);

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(constants, weights.post_norm));
    auto gate = modules::LinearModule(
                    binding::linear_config(config.hidden_size, config.intermediate_size, false))
                    .build(ctx, ff_in, weights.mlp.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule(
                  binding::linear_config(config.hidden_size, config.intermediate_size, false))
                  .build(ctx, ff_in, weights.mlp.up_proj);
    auto ff = modules::LinearModule(
                  binding::linear_config(config.intermediate_size, config.hidden_size, false))
                  .build(ctx, modules::MulModule{}.build(ctx, gate, up), weights.mlp.down_proj);
    auto output = add.build(ctx, x, ff);
    return {output, k, v};
}

}  // namespace engine::models::vibevoice

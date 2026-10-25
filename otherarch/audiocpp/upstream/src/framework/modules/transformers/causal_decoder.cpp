#include "engine/framework/modules/transformers/causal_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

void validate_config(const CausalDecoderConfig & config) {
    if (config.stack.hidden_size <= 0 || config.logits_size <= 0) {
        throw std::runtime_error("QwenCausalDecoderConfig requires positive hidden and logits sizes");
    }
    if (config.stack.layers <= 0) {
        throw std::runtime_error("QwenCausalDecoderConfig requires a positive layer count");
    }
    if (config.static_cache_type != GGML_TYPE_F32 && config.static_cache_type != GGML_TYPE_F16 &&
        config.static_cache_type != GGML_TYPE_BF16) {
        throw std::runtime_error("QwenCausalDecoderConfig static cache type must be f32, f16, or bf16");
    }
}

void validate_hidden_config(const DecoderHiddenConfig & config) {
    if (config.stack.hidden_size <= 0) {
        throw std::runtime_error("QwenDecoderHiddenConfig requires positive hidden size");
    }
    if (config.stack.layers <= 0) {
        throw std::runtime_error("QwenDecoderHiddenConfig requires a positive layer count");
    }
    if (config.static_cache_type != GGML_TYPE_F32 && config.static_cache_type != GGML_TYPE_F16 &&
        config.static_cache_type != GGML_TYPE_BF16) {
        throw std::runtime_error("QwenDecoderHiddenConfig static cache type must be f32, f16, or bf16");
    }
}

void validate_steps(int64_t steps, const char * label) {
    if (steps <= 0) {
        throw std::runtime_error(std::string(label) + " requires positive step count");
    }
}

runtime::TransformerKVCacheOptions transformer_cache_options(ggml_type type) {
    runtime::TransformerKVCacheOptions out;
    out.allow_f16_storage = type == GGML_TYPE_F16;
    out.allow_bf16_storage = type == GGML_TYPE_BF16;
    return out;
}

core::TensorValue select_hidden_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_sequence,
    CausalDecoderLogitsMode mode) {
    if (mode == CausalDecoderLogitsMode::AllSteps) {
        return hidden_sequence;
    }
    const int64_t steps = hidden_sequence.shape.dims[1];
    return SliceModule({1, steps - 1, 1}).build(ctx, hidden_sequence);
}

DecoderHiddenConfig hidden_config_from_causal(const CausalDecoderConfig & config) {
    DecoderHiddenConfig out;
    out.stack = config.stack;
    out.hidden_mode = config.logits_mode;
    out.static_cache_type = config.static_cache_type;
    return out;
}

DecoderHiddenWeights hidden_weights_from_causal(const CausalDecoderWeights & weights) {
    DecoderHiddenWeights out;
    out.stack = weights.stack;
    out.final_norm = weights.final_norm;
    return out;
}

}  // namespace

DecoderHiddenModule::DecoderHiddenModule(DecoderHiddenConfig config)
    : config_(std::move(config)) {
    validate_hidden_config(config_);
}

const DecoderHiddenConfig & DecoderHiddenModule::config() const noexcept {
    return config_;
}

DecoderHiddenOutputs DecoderHiddenModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderHiddenWeights & weights,
    const std::optional<DecoderStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenDecoderHiddenModule input shape must be [batch, steps, hidden]");
    }

    auto stack = DecoderStackModule(config_.stack)
                     .build(ctx, input, positions, weights.stack, prefix_state, attention_mask);
    auto hidden_sequence = RMSNormModule({config_.stack.hidden_size, config_.stack.rms_norm_eps, true, false, false, false,
                                         config_.stack.activation_cast.norm_intermediate_type})
                               .build(ctx, stack.output, weights.final_norm);
    auto hidden = select_hidden_steps(ctx, hidden_sequence, config_.hidden_mode);
    return {std::move(stack.output), hidden, std::move(stack.state)};
}

DecoderHiddenStaticCacheOutputs DecoderHiddenModule::build_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderHiddenWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const std::optional<core::TensorValue> & cache_slot) const {
    if (graph == nullptr) {
        throw std::runtime_error("QwenDecoderHiddenModule static-cache build requires a graph");
    }
    validate_steps(cache_steps, "QwenDecoderHiddenModule static-cache build");
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenDecoderHiddenModule static-cache input shape must be [batch, steps, hidden]");
    }
    if (input.shape.dims[0] != 1 || input.shape.dims[1] != 1) {
        throw std::runtime_error("QwenDecoderHiddenModule static-cache build currently supports single-token decode");
    }
    if (static_cast<int64_t>(weights.stack.layers.size()) != config_.stack.layers) {
        throw std::runtime_error("QwenDecoderHiddenWeights layer count does not match config");
    }

    const int64_t step_elems = config_.stack.num_key_value_heads * config_.stack.head_dim;
    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
    cache_keys.reserve(weights.stack.layers.size());
    cache_values.reserve(weights.stack.layers.size());

    auto x = input;
    const DecoderLayerModule layer_module(decoder_layer_config_from_stack(config_.stack));
    for (const auto & layer : weights.stack.layers) {
        cache_keys.push_back(core::make_tensor(
            ctx,
            config_.static_cache_type,
            core::TensorShape::from_dims({1, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        cache_values.push_back(core::make_tensor(
            ctx,
            config_.static_cache_type,
            core::TensorShape::from_dims({1, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        auto out = layer_module.build_with_static_cache_tail(
            ctx,
            graph,
            x,
            positions,
            layer,
            cache_keys.back(),
            cache_values.back(),
            cache_slot,
            attention_mask);
        x = out.output;
    }

    auto hidden = RMSNormModule({config_.stack.hidden_size, config_.stack.rms_norm_eps, true, false, false, false,
                                config_.stack.activation_cast.norm_intermediate_type})
                      .build(ctx, x, weights.final_norm);
    return {
        std::move(x),
        hidden,
        runtime::TransformerKVCache(
            cache_steps,
            step_elems,
            std::move(cache_keys),
            std::move(cache_values),
            transformer_cache_options(config_.static_cache_type)),
    };
}

DecoderHiddenBatchedStaticCacheOutputs DecoderHiddenModule::build_static_cache_tail_batched(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DecoderHiddenWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot) const {
    if (graph == nullptr) {
        throw std::runtime_error("QwenDecoderHiddenModule batched static-cache build requires a graph");
    }
    validate_steps(cache_steps, "QwenDecoderHiddenModule batched static-cache build");
    if (input.shape.rank != 3 || input.shape.dims[0] <= 0 || input.shape.dims[1] != 1 ||
        input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenDecoderHiddenModule batched static-cache input shape must be [batch, 1, hidden]");
    }
    if (static_cast<int64_t>(weights.stack.layers.size()) != config_.stack.layers) {
        throw std::runtime_error("QwenDecoderHiddenWeights layer count does not match config");
    }

    const int64_t batch_size = input.shape.dims[0];
    const int64_t row_elems = config_.stack.num_key_value_heads * config_.stack.head_dim;
    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
    cache_keys.reserve(weights.stack.layers.size());
    cache_values.reserve(weights.stack.layers.size());

    auto x = input;
    const DecoderLayerModule layer_module(decoder_layer_config_from_stack(config_.stack));
    for (const auto & layer : weights.stack.layers) {
        cache_keys.push_back(core::make_tensor(
            ctx,
            config_.static_cache_type,
            core::TensorShape::from_dims(
                {batch_size, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        cache_values.push_back(core::make_tensor(
            ctx,
            config_.static_cache_type,
            core::TensorShape::from_dims(
                {batch_size, cache_steps, config_.stack.num_key_value_heads, config_.stack.head_dim})));
        auto out = layer_module.build_with_static_cache_tail_batched(
            ctx,
            graph,
            x,
            positions,
            layer,
            cache_keys.back(),
            cache_values.back(),
            cache_slot,
            attention_mask);
        x = out.output;
    }

    auto hidden = RMSNormModule({config_.stack.hidden_size, config_.stack.rms_norm_eps, true, false, false, false,
                                config_.stack.activation_cast.norm_intermediate_type})
                      .build(ctx, x, weights.final_norm);
    return {
        std::move(x),
        hidden,
        runtime::TransformerBatchedKVCache(
            cache_steps,
            batch_size,
            row_elems,
            std::move(cache_keys),
            std::move(cache_values),
            transformer_cache_options(config_.static_cache_type)),
    };
}

CausalDecoderModule::CausalDecoderModule(CausalDecoderConfig config)
    : config_(std::move(config)) {
    validate_config(config_);
}

const CausalDecoderConfig & CausalDecoderModule::config() const noexcept {
    return config_;
}

CausalDecoderOutputs CausalDecoderModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const CausalDecoderWeights & weights,
    const std::optional<DecoderStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenCausalDecoderModule input shape must be [batch, steps, hidden]");
    }

    auto hidden_out = DecoderHiddenModule(hidden_config_from_causal(config_))
                          .build(
                              ctx,
                              input,
                              positions,
                              hidden_weights_from_causal(weights),
                              prefix_state,
                              attention_mask);

    auto logits_input = hidden_out.hidden;
    if (config_.lm_head_input_type.has_value() && logits_input.type != *config_.lm_head_input_type) {
        logits_input = core::wrap_tensor(
            ggml_cast(ctx.ggml, logits_input.tensor, *config_.lm_head_input_type),
            logits_input.shape,
            *config_.lm_head_input_type);
    }
    const auto logits = LinearModule({
                            config_.stack.hidden_size,
                            config_.logits_size,
                            config_.use_lm_head_bias,
                            config_.lm_head_precision,
                        })
                            .build(ctx, logits_input, weights.lm_head);
    return {std::move(hidden_out.sequence), hidden_out.hidden, logits, std::move(hidden_out.state)};
}

CausalDecoderStaticCacheOutputs CausalDecoderModule::build_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const CausalDecoderWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const std::optional<core::TensorValue> & cache_slot) const {
    if (graph == nullptr) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache build requires a graph");
    }
    validate_steps(cache_steps, "QwenCausalDecoderModule static-cache build");
    if (input.shape.rank != 3 || input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache input shape must be [batch, steps, hidden]");
    }
    if (input.shape.dims[0] != 1 || input.shape.dims[1] != 1) {
        throw std::runtime_error("QwenCausalDecoderModule static-cache build currently supports single-token decode");
    }

    auto hidden_out = DecoderHiddenModule(hidden_config_from_causal(config_))
                          .build_static_cache_tail(
                              ctx,
                              graph,
                              input,
                              positions,
                              hidden_weights_from_causal(weights),
                              cache_steps,
                              attention_mask,
                              cache_slot);
    auto logits_input = hidden_out.hidden;
    if (config_.lm_head_input_type.has_value() && logits_input.type != *config_.lm_head_input_type) {
        logits_input = core::wrap_tensor(
            ggml_cast(ctx.ggml, logits_input.tensor, *config_.lm_head_input_type),
            logits_input.shape,
            *config_.lm_head_input_type);
    }
    const auto logits = LinearModule({
                            config_.stack.hidden_size,
                            config_.logits_size,
                            config_.use_lm_head_bias,
                            config_.lm_head_precision,
                        })
                            .build(ctx, logits_input, weights.lm_head);
    return {
        std::move(hidden_out.sequence),
        hidden_out.hidden,
        logits,
        std::move(hidden_out.cache),
    };
}

CausalDecoderBatchedStaticCacheOutputs CausalDecoderModule::build_static_cache_tail_batched(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const CausalDecoderWeights & weights,
    int64_t cache_steps,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slot) const {
    if (graph == nullptr) {
        throw std::runtime_error("QwenCausalDecoderModule batched static-cache build requires a graph");
    }
    validate_steps(cache_steps, "QwenCausalDecoderModule batched static-cache build");
    if (input.shape.rank != 3 || input.shape.dims[0] <= 0 || input.shape.dims[1] != 1 ||
        input.shape.dims[2] != config_.stack.hidden_size) {
        throw std::runtime_error("QwenCausalDecoderModule batched static-cache input shape must be [batch, 1, hidden]");
    }

    auto hidden_out = DecoderHiddenModule(hidden_config_from_causal(config_))
                          .build_static_cache_tail_batched(
                              ctx,
                              graph,
                              input,
                              positions,
                              hidden_weights_from_causal(weights),
                              cache_steps,
                              attention_mask,
                              cache_slot);
    auto logits_input = hidden_out.hidden;
    if (config_.lm_head_input_type.has_value() && logits_input.type != *config_.lm_head_input_type) {
        logits_input = core::wrap_tensor(
            ggml_cast(ctx.ggml, logits_input.tensor, *config_.lm_head_input_type),
            logits_input.shape,
            *config_.lm_head_input_type);
    }
    const auto logits = LinearModule({
                            config_.stack.hidden_size,
                            config_.logits_size,
                            config_.use_lm_head_bias,
                            config_.lm_head_precision,
                        })
                            .build(ctx, logits_input, weights.lm_head);
    return {
        std::move(hidden_out.sequence),
        hidden_out.hidden,
        logits,
        std::move(hidden_out.cache),
    };
}

std::vector<int32_t> decoder_position_ids(int64_t steps, int64_t offset) {
    validate_steps(steps, "qwen_position_ids");
    std::vector<int32_t> out(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(offset + i);
    }
    return out;
}

std::vector<ggml_fp16_t> causal_prefill_mask_values(int64_t batch_size, int64_t steps) {
    if (batch_size <= 0) {
        throw std::runtime_error("qwen_causal_prefill_mask_values requires positive batch size");
    }
    validate_steps(steps, "qwen_causal_prefill_mask_values");
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> one(static_cast<size_t>(steps * steps), masked);
    for (int64_t row = 0; row < steps; ++row) {
        const size_t row_offset = static_cast<size_t>(row * steps);
        for (int64_t col = 0; col <= row; ++col) {
            one[row_offset + static_cast<size_t>(col)] = visible;
        }
    }
    if (batch_size == 1) {
        return one;
    }
    std::vector<ggml_fp16_t> out;
    out.reserve(static_cast<size_t>(batch_size) * one.size());
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        out.insert(out.end(), one.begin(), one.end());
    }
    return out;
}

std::vector<ggml_fp16_t> causal_suffix_mask_values(
    int64_t batch_size,
    int64_t query_steps,
    int64_t prefix_steps) {
    if (batch_size <= 0) {
        throw std::runtime_error("qwen_causal_suffix_mask_values requires positive batch size");
    }
    validate_steps(query_steps, "qwen_causal_suffix_mask_values");
    if (prefix_steps < 0) {
        throw std::runtime_error("qwen_causal_suffix_mask_values requires non-negative prefix steps");
    }
    const int64_t key_steps = prefix_steps + query_steps;
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> one(static_cast<size_t>(query_steps * key_steps), masked);
    for (int64_t row = 0; row < query_steps; ++row) {
        const size_t row_offset = static_cast<size_t>(row * key_steps);
        std::fill_n(
            one.begin() + static_cast<std::ptrdiff_t>(row_offset),
            prefix_steps + row + 1,
            visible);
    }
    std::vector<ggml_fp16_t> out;
    out.reserve(static_cast<size_t>(batch_size) * one.size());
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        out.insert(out.end(), one.begin(), one.end());
    }
    return out;
}

void write_causal_prefill_mask(
    ggml_tensor * tensor,
    int64_t batch_size,
    int64_t steps) {
    if (tensor == nullptr) {
        throw std::runtime_error("write_qwen_causal_prefill_mask requires a tensor");
    }
    auto values = causal_prefill_mask_values(batch_size, steps);
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
}

void write_decoder_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot) {
    if (tensor == nullptr) {
        throw std::runtime_error("write_qwen_cached_step_mask requires a tensor");
    }
    validate_steps(mask_steps, "write_qwen_cached_step_mask");
    if (visible_prefix_steps < 0 || visible_prefix_steps > mask_steps) {
        throw std::runtime_error("write_qwen_cached_step_mask visible prefix is out of range");
    }
    if (current_slot < 0 || current_slot >= mask_steps) {
        throw std::runtime_error("write_qwen_cached_step_mask current slot is out of range");
    }
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    if (scratch.size() != static_cast<size_t>(mask_steps)) {
        scratch.resize(static_cast<size_t>(mask_steps));
    }
    std::fill(scratch.begin(), scratch.end(), masked);
    for (int64_t i = 0; i < visible_prefix_steps; ++i) {
        scratch[static_cast<size_t>(i)] = visible;
    }
    scratch[static_cast<size_t>(current_slot)] = visible;
    ggml_backend_tensor_set(tensor, scratch.data(), 0, scratch.size() * sizeof(ggml_fp16_t));
}

void write_decoder_batched_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t batch_size,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot) {
    if (tensor == nullptr) {
        throw std::runtime_error("write_qwen_batched_cached_step_mask requires a tensor");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("write_qwen_batched_cached_step_mask requires positive batch size");
    }
    validate_steps(mask_steps, "write_qwen_batched_cached_step_mask");
    if (visible_prefix_steps < 0 || visible_prefix_steps > mask_steps) {
        throw std::runtime_error("write_qwen_batched_cached_step_mask visible prefix is out of range");
    }
    if (current_slot < 0 || current_slot >= mask_steps) {
        throw std::runtime_error("write_qwen_batched_cached_step_mask current slot is out of range");
    }
    const auto masked = ggml_fp32_to_fp16(-INFINITY);
    const auto visible = ggml_fp32_to_fp16(0.0F);
    const size_t row_size = static_cast<size_t>(mask_steps);
    const size_t total_size = static_cast<size_t>(batch_size) * row_size;
    if (scratch.size() != total_size) {
        scratch.resize(total_size);
    }
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * row_size;
        std::fill(
            scratch.begin() + static_cast<std::ptrdiff_t>(offset),
            scratch.begin() + static_cast<std::ptrdiff_t>(offset + row_size),
            masked);
        for (int64_t i = 0; i < visible_prefix_steps; ++i) {
            scratch[offset + static_cast<size_t>(i)] = visible;
        }
        scratch[offset + static_cast<size_t>(current_slot)] = visible;
    }
    ggml_backend_tensor_set(tensor, scratch.data(), 0, scratch.size() * sizeof(ggml_fp16_t));
}

}  // namespace engine::modules

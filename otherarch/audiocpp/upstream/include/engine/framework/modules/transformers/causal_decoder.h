#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <ggml.h>

namespace engine::modules {

enum class CausalDecoderLogitsMode {
    LastStep,
    AllSteps,
};

struct DecoderHiddenConfig {
    DecoderStackConfig stack;
    CausalDecoderLogitsMode hidden_mode = CausalDecoderLogitsMode::LastStep;
    ggml_type static_cache_type = GGML_TYPE_F32;
};

struct CausalDecoderConfig {
    DecoderStackConfig stack;
    int64_t logits_size = 0;
    CausalDecoderLogitsMode logits_mode = CausalDecoderLogitsMode::LastStep;
    bool use_lm_head_bias = false;
    ggml_prec lm_head_precision = GGML_PREC_DEFAULT;
    std::optional<ggml_type> lm_head_input_type;
    ggml_type static_cache_type = GGML_TYPE_F32;
};

struct CausalDecoderWeights {
    DecoderStackWeights stack;
    NormWeights final_norm;
    LinearWeights lm_head;
};

struct DecoderHiddenWeights {
    DecoderStackWeights stack;
    NormWeights final_norm;
};

struct DecoderHiddenOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    DecoderStackState state;
};

struct DecoderHiddenStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    runtime::TransformerKVCache cache;
};

struct DecoderHiddenBatchedStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    runtime::TransformerBatchedKVCache cache;
};

struct CausalDecoderOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    DecoderStackState state;
};

struct CausalDecoderStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerKVCache cache;
};

struct CausalDecoderBatchedStaticCacheOutputs {
    core::TensorValue sequence;
    core::TensorValue hidden;
    core::TensorValue logits;
    runtime::TransformerBatchedKVCache cache;
};

class DecoderHiddenModule {
public:
    explicit DecoderHiddenModule(DecoderHiddenConfig config);

    const DecoderHiddenConfig & config() const noexcept;

    DecoderHiddenOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderHiddenWeights & weights,
        const std::optional<DecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    DecoderHiddenStaticCacheOutputs build_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderHiddenWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & cache_slot = std::nullopt) const;

    DecoderHiddenBatchedStaticCacheOutputs build_static_cache_tail_batched(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderHiddenWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const core::TensorValue & cache_slot) const;

private:
    DecoderHiddenConfig config_;
};

class CausalDecoderModule {
public:
    explicit CausalDecoderModule(CausalDecoderConfig config);

    const CausalDecoderConfig & config() const noexcept;

    CausalDecoderOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const CausalDecoderWeights & weights,
        const std::optional<DecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    CausalDecoderStaticCacheOutputs build_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const CausalDecoderWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & cache_slot = std::nullopt) const;

    CausalDecoderBatchedStaticCacheOutputs build_static_cache_tail_batched(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const CausalDecoderWeights & weights,
        int64_t cache_steps,
        const core::TensorValue & attention_mask,
        const core::TensorValue & cache_slot) const;

private:
    CausalDecoderConfig config_;
};

std::vector<int32_t> decoder_position_ids(int64_t steps, int64_t offset = 0);

std::vector<ggml_fp16_t> causal_prefill_mask_values(int64_t batch_size, int64_t steps);

std::vector<ggml_fp16_t> causal_suffix_mask_values(
    int64_t batch_size,
    int64_t query_steps,
    int64_t prefix_steps);

void write_causal_prefill_mask(
    ggml_tensor * tensor,
    int64_t batch_size,
    int64_t steps);

void write_decoder_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot);

void write_decoder_batched_cached_step_mask(
    ggml_tensor * tensor,
    std::vector<ggml_fp16_t> & scratch,
    int64_t batch_size,
    int64_t mask_steps,
    int64_t visible_prefix_steps,
    int64_t current_slot);

}  // namespace engine::modules

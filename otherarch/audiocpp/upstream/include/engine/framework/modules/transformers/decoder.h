#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

struct ggml_cgraph;

namespace engine::modules {

enum class DecoderAttentionMode {
    ManualRepeat,
    FlashGrouped,
    FlashGroupedViewKV,
    ManualRepeatThenGroupedQuery,
};

enum class DecoderStaticCacheUpdateMode {
    ScratchTail,
    DirectSetRows,
};

enum class DecoderStaticCacheSetRowsMode {
    Exact,
    BackendViewOptimized,
};

enum class DecoderQKVLayout {
    Separate,
    PackedQKV,
};

enum class DecoderMLPMode {
    Exact,
    FusedSwiGLU,
    PackedGateUp,
};

enum class DecoderPrefixAttentionMode {
    Exact,
    FlashWithPrefix,
};

enum class DecoderPositionEncoding {
    Rotary,
    None,
};

struct DecoderActivationCastPolicy {
    bool enabled = false;
    ggml_type type = GGML_TYPE_BF16;
    // Use the fused single-kernel round-to-bf16 op instead of a
    // cast -> bf16 -> cast -> f32 round trip. Only valid on backends that
    // implement GGML_UNARY_OP_ROUND_BF16 (CUDA/HIP, CPU fallback).
    bool fused_round = false;
    bool after_input_norm = false;
    bool after_qkv_projection = false;
    bool after_qk_norm = false;
    bool after_rope = false;
    bool after_static_cache_update = false;
    bool after_attention = false;
    bool after_attention_output = false;
    bool after_residual = false;
    bool after_ffn_norm = false;
    bool after_mlp_projection = false;
    bool after_mlp_silu = false;
    bool after_mlp_mul = false;
    bool after_output = false;
    // Qwen rounds normalized activations before applying the norm weight.
    std::optional<ggml_type> norm_intermediate_type = std::nullopt;
    std::optional<ggml_type> rope_intermediate_type = std::nullopt;
};

struct DecoderAttentionPolicy {
    DecoderAttentionMode prefill_mode = DecoderAttentionMode::ManualRepeat;
    DecoderAttentionMode static_mode = DecoderAttentionMode::FlashGrouped;
    DecoderPrefixAttentionMode prefix_mode = DecoderPrefixAttentionMode::Exact;
    int64_t grouped_query_min_steps = 0;
    // False routes flash branches through repeat-KV + matmul/softmax for GPUs
    // without a flash kernel (e.g. CUDA sm70). True preserves historical behavior.
    bool allow_flash_attention = true;
};

struct DecoderStaticCachePolicy {
    DecoderStaticCacheUpdateMode update_mode = DecoderStaticCacheUpdateMode::ScratchTail;
    DecoderStaticCacheSetRowsMode set_rows_mode = DecoderStaticCacheSetRowsMode::Exact;
};

struct DecoderMLPPolicy {
    DecoderMLPMode mode = DecoderMLPMode::Exact;
};

struct DecoderRuntimePolicy {
    DecoderAttentionPolicy attention;
    DecoderStaticCachePolicy static_cache;
    DecoderMLPPolicy mlp;
};

struct DecoderLayerConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int rope_type = GGML_ROPE_TYPE_NEOX;
    DecoderPositionEncoding position_encoding = DecoderPositionEncoding::Rotary;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    DecoderQKVLayout qkv_layout = DecoderQKVLayout::Separate;
    bool use_qk_norm = true;
    DecoderActivationCastPolicy activation_cast;
    DecoderRuntimePolicy runtime;
};

struct DecoderMLPWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    std::optional<LinearWeights> gate_up_proj;
    LinearWeights down_proj;
};

struct DecoderLayerWeights {
    NormWeights input_norm;
    AttentionWeights self_attention;
    NormWeights q_norm;
    NormWeights k_norm;
    NormWeights post_norm;
    DecoderMLPWeights mlp;
    // Optional per-frequency RoPE divisors (head_dim / 2), used by Llama-3
    // scaling and compatible checkpoints.
    std::optional<core::TensorValue> rope_frequency_factors;
};

struct DecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class DecoderLayerModule {
public:
    explicit DecoderLayerModule(DecoderLayerConfig config);

    const DecoderLayerConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    DecoderLayerOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderLayerWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    DecoderLayerOutputs build_with_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const std::optional<core::TensorValue> & cache_slot,
        const core::TensorValue & attention_mask) const;

    DecoderLayerOutputs build_with_static_cache_block(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const std::optional<core::TensorValue> & cache_slot,
        const core::TensorValue & attention_mask) const;

    DecoderLayerOutputs build_with_static_cache_tail_batched(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & cache_slot,
        const core::TensorValue & attention_mask) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    DecoderLayerOutputs build_static_cache_impl(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const std::optional<core::TensorValue> & cache_slot,
        const core::TensorValue & attention_mask,
        bool block) const;
    DecoderLayerConfig config_;
};

struct DecoderStackConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int rope_type = GGML_ROPE_TYPE_NEOX;
    DecoderPositionEncoding position_encoding = DecoderPositionEncoding::Rotary;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    DecoderQKVLayout qkv_layout = DecoderQKVLayout::Separate;
    bool use_qk_norm = true;
    DecoderActivationCastPolicy activation_cast;
    DecoderRuntimePolicy runtime;
};

DecoderLayerConfig decoder_layer_config_from_stack(const DecoderStackConfig & config);

struct DecoderStackWeights {
    std::vector<DecoderLayerWeights> layers;
};

struct DecoderStackLayerState {
    std::optional<core::TensorValue> key;
    std::optional<core::TensorValue> value;
};

struct DecoderStackState {
    std::vector<DecoderStackLayerState> layers;
};

struct DecoderStackOutputs {
    core::TensorValue output;
    DecoderStackState state;
};

class DecoderStackModule {
public:
    explicit DecoderStackModule(DecoderStackConfig config);

    const DecoderStackConfig & config() const noexcept;

    DecoderStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const DecoderStackWeights & weights,
        const std::optional<DecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    DecoderStackConfig config_;
};

}  // namespace engine::modules

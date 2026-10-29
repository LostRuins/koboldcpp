#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

namespace engine::modules {

struct TransformerEncoderBlockConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    AttentionPrefixCacheLayout prefix_cache_layout = AttentionPrefixCacheLayout::SequenceHeads;
    bool use_packed_qkv = false;
    bool use_flash_attention = false;
};

struct TransformerEncoderBlockWeights {
    NormWeights norm1;
    AttentionWeights self_attention;
    std::optional<LayerScaleWeights> layer_scale1;
    NormWeights norm2;
    FeedForwardWeights feed_forward;
    std::optional<LayerScaleWeights> layer_scale2;
};

struct StreamingTransformerEncoderBlockOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class TransformerEncoderBlockModule {
public:
    explicit TransformerEncoderBlockModule(TransformerEncoderBlockConfig config);

    const TransformerEncoderBlockConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const TransformerEncoderBlockWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    TransformerEncoderBlockConfig config_;
};

class StreamingTransformerEncoderBlockModule {
public:
    explicit StreamingTransformerEncoderBlockModule(TransformerEncoderBlockConfig config);

    const TransformerEncoderBlockConfig & config() const noexcept;

    StreamingTransformerEncoderBlockOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const TransformerEncoderBlockWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    TransformerEncoderBlockConfig config_;
};

struct StreamingTransformerStackConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    AttentionPrefixCacheLayout prefix_cache_layout = AttentionPrefixCacheLayout::SequenceHeads;
};

struct StreamingTransformerStackWeights {
    std::vector<TransformerEncoderBlockWeights> layers;
};

struct StreamingTransformerLayerState {
    std::optional<core::TensorValue> key;
    std::optional<core::TensorValue> value;
};

struct StreamingTransformerStackState {
    std::vector<StreamingTransformerLayerState> layers;
};

struct StreamingTransformerStackOutputs {
    core::TensorValue output;
    StreamingTransformerStackState state;
};

class StreamingTransformerStackModule {
public:
    explicit StreamingTransformerStackModule(StreamingTransformerStackConfig config);

    const StreamingTransformerStackConfig & config() const noexcept;

    StreamingTransformerStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const StreamingTransformerStackWeights & weights,
        const std::optional<StreamingTransformerStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    StreamingTransformerStackConfig config_;
};

struct ProjectedTransformerConfig {
    int64_t input_dimension = 0;
    int64_t output_dimension = 0;
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    AttentionPrefixCacheLayout prefix_cache_layout = AttentionPrefixCacheLayout::SequenceHeads;
};

struct ProjectedTransformerWeights {
    std::optional<LinearWeights> input_projection;
    std::vector<TransformerEncoderBlockWeights> transformer_layers;
    std::optional<LinearWeights> output_projection;
};

struct StreamingProjectedTransformerWeights {
    std::optional<LinearWeights> input_projection;
    std::vector<TransformerEncoderBlockWeights> transformer_layers;
    std::optional<LinearWeights> output_projection;
};

struct StreamingProjectedTransformerOutputs {
    core::TensorValue output;
    StreamingTransformerStackState state;
};

class ProjectedTransformerModule {
public:
    explicit ProjectedTransformerModule(ProjectedTransformerConfig config);

    const ProjectedTransformerConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_bct,
        const ProjectedTransformerWeights & weights) const;

private:
    ProjectedTransformerConfig config_;
};

class StreamingProjectedTransformerModule {
public:
    explicit StreamingProjectedTransformerModule(ProjectedTransformerConfig config);

    const ProjectedTransformerConfig & config() const noexcept;

    StreamingProjectedTransformerOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input_bct,
        const core::TensorValue & positions,
        const StreamingProjectedTransformerWeights & weights,
        const std::optional<StreamingTransformerStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    ProjectedTransformerConfig config_;
};

struct TransformerStackConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_DEFAULT;
    AttentionPrefixCacheLayout prefix_cache_layout = AttentionPrefixCacheLayout::SequenceHeads;
};

struct TransformerStackWeights {
    std::vector<TransformerEncoderBlockWeights> layers;
};

class TransformerStackModule {
public:
    explicit TransformerStackModule(TransformerStackConfig config);

    const TransformerStackConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const TransformerStackWeights & weights) const;

private:
    TransformerStackConfig config_;
};

struct TransformerDecoderBlockConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    FeedForwardActivation activation = FeedForwardActivation::Gelu;
    bool use_packed_qkv = false;
    bool use_packed_kv = false;
    bool use_flash_cross_attention = false;
};

struct TransformerDecoderBlockWeights {
    NormWeights norm1;
    AttentionWeights self_attention;
    NormWeights norm2;
    AttentionWeights cross_attention;
    NormWeights norm3;
    FeedForwardWeights feed_forward;
};

class TransformerDecoderBlockModule {
public:
    explicit TransformerDecoderBlockModule(TransformerDecoderBlockConfig config);

    const TransformerDecoderBlockConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & memory,
        const TransformerDecoderBlockWeights & weights) const;

    // Requires packed QKV/KV opt-ins. Caches and masks are caller-owned, using
    // SelfAttentionModule::build_cached_tail and CrossAttentionModule::build_cached layouts.
    // When use_flash_cross_attention is enabled, memory_mask instead follows
    // CrossAttentionModule::build_cached_flash's additive F16 mask contract.
    core::TensorValue build_cached_tail(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const TransformerDecoderBlockWeights & weights,
        const core::TensorValue & self_key_cache,
        const core::TensorValue & self_value_cache,
        const core::TensorValue & cache_slot,
        const core::TensorValue & causal_mask,
        const CrossAttentionKeyValue & memory_key_value,
        const core::TensorValue & memory_mask) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    TransformerDecoderBlockConfig config_;
};

}  // namespace engine::modules

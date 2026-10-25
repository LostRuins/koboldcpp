#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

namespace engine::modules {

struct ConformerConvModuleConfig {
    int64_t hidden_size = 0;
    int64_t kernel_size = 0;
    bool use_bias = true;
    float eps = 1e-5f;
    int64_t cache_drop_size = 0;
    bool contiguous_glu_gate = false;
};

struct ConvSubsamplingConfig {
    int64_t input_features = 0;
    int64_t output_features = 0;
    int64_t conv_channels = 0;
    int64_t subsampling_factor = 4;
    int64_t kernel_size = 3;
    int stride = 2;
    bool use_bias = true;
};

struct ConvSubsamplingWeights {
    Conv2dWeights conv0;
    Conv2dWeights conv1;
    LinearWeights linear;
};

struct ConvSubsamplingOutputs {
    core::TensorValue output;
    core::TensorValue lengths;
};

struct EspnetConv2dSubsampling8Config {
    int64_t input_features = 0;
    int64_t hidden_size = 0;
    int64_t time_row_alignment = 1;
};

struct EspnetConv2dSubsampling8Weights {
    Conv2dWeights conv0;
    Conv2dWeights conv1;
    Conv2dWeights conv2;
    LinearWeights projection;
};

class EspnetConv2dSubsampling8Module {
public:
    explicit EspnetConv2dSubsampling8Module(EspnetConv2dSubsampling8Config config);
    // Input/output are [batch, time, features]. Three unpadded stride-2 convolutions.
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const EspnetConv2dSubsampling8Weights & weights) const;

private:
    EspnetConv2dSubsampling8Config config_;
};

struct DepthwiseConvSubsamplingConfig {
    int64_t input_features = 0;
    int64_t output_features = 0;
    int64_t conv_channels = 0;
    int kernel_size = 3;
    int stride = 2;
    int padding = 1;
    bool use_bias = true;
};

struct DepthwiseConvSubsamplingStageWeights {
    Conv2dWeights depthwise;
    Conv2dWeights pointwise;
};

struct DepthwiseConvSubsamplingWeights {
    Conv2dWeights input_conv;
    std::vector<DepthwiseConvSubsamplingStageWeights> stages;
    LinearWeights projection;
};

class DepthwiseConvSubsamplingModule {
public:
    explicit DepthwiseConvSubsamplingModule(DepthwiseConvSubsamplingConfig config);
    // Input is [batch, time, features]; optional masks cover each downsampling stage.
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const DepthwiseConvSubsamplingWeights & weights,
        const std::vector<core::TensorValue> & stage_keep_masks = {}) const;

private:
    DepthwiseConvSubsamplingConfig config_;
};

class ConvSubsamplingModule {
public:
    explicit ConvSubsamplingModule(ConvSubsamplingConfig config);
    ConvSubsamplingOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & lengths,
        const ConvSubsamplingWeights & weights) const;

private:
    ConvSubsamplingConfig config_;
};

struct ConformerConvModuleWeights {
    NormWeights norm;
    LinearWeights pointwise_in;
    DepthwiseConv1dWeights depthwise;
    ChannelAffineWeights depthwise_norm;
    LinearWeights pointwise_out;
};

struct StreamingConformerConvOutputs {
    core::TensorValue output;
    core::TensorValue next_cache;
};

class ConformerConvModule {
public:
    explicit ConformerConvModule(ConformerConvModuleConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConformerConvModuleWeights & weights,
        const std::optional<core::TensorValue> & keep_mask = std::nullopt) const;

private:
    ConformerConvModuleConfig config_;
};

class StreamingConformerConvModule {
public:
    explicit StreamingConformerConvModule(ConformerConvModuleConfig config);
    StreamingConformerConvOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConformerConvModuleWeights & weights,
        const std::optional<core::TensorValue> & prefix_cache = std::nullopt,
        const std::optional<core::TensorValue> & keep_mask = std::nullopt) const;

private:
    ConformerConvModuleConfig config_;
};

struct ConformerBlockConfig {
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    int64_t kernel_size = 0;
    float eps = 1e-5f;
    bool use_bias = true;
    int64_t left_context = -1;
    int64_t right_context = -1;
    int64_t cache_drop_size = 0;
    bool contiguous_glu_gate = false;
};

struct ConformerBlockWeights {
    NormWeights ffn1_norm;
    LinearWeights ffn1_fc1;
    LinearWeights ffn1_fc2;
    NormWeights norm1;
    AttentionWeights self_attention;
    ConformerConvModuleWeights conv;
    NormWeights norm2;
    LinearWeights ffn2_fc1;
    LinearWeights ffn2_fc2;
    NormWeights final_norm;
};

struct RelativeConformerBlockWeights {
    NormWeights ffn1_norm;
    LinearWeights ffn1_fc1;
    LinearWeights ffn1_fc2;
    NormWeights norm1;
    RelativeAttentionWeights self_attention;
    ConformerConvModuleWeights conv;
    NormWeights norm2;
    LinearWeights ffn2_fc1;
    LinearWeights ffn2_fc2;
    NormWeights final_norm;
};

struct StreamingConformerBlockOutputs {
    core::TensorValue output;
    core::TensorValue next_channel_cache;
    core::TensorValue next_time_cache;
};

class ConformerBlockModule {
public:
    explicit ConformerBlockModule(ConformerBlockConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConformerBlockWeights & weights) const;

private:
    ConformerBlockConfig config_;
};

class RelativeConformerBlockModule {
public:
    explicit RelativeConformerBlockModule(ConformerBlockConfig config);
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const std::optional<core::TensorValue> & pos_emb,
        const RelativeConformerBlockWeights & weights,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt,
        const std::optional<core::TensorValue> & keep_mask = std::nullopt,
        const std::optional<core::TensorValue> & query_keep_mask = std::nullopt,
        const std::optional<core::TensorValue> & projected_pos_emb = std::nullopt) const;

private:
    ConformerBlockConfig config_;
};

class StreamingConformerBlockModule {
public:
    explicit StreamingConformerBlockModule(ConformerBlockConfig config);
    StreamingConformerBlockOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & pos_emb,
        const ConformerBlockWeights & weights,
        const std::optional<core::TensorValue> & prefix_channel_cache = std::nullopt,
        const std::optional<core::TensorValue> & prefix_time_cache = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    ConformerBlockConfig config_;
};

class StreamingRelativeConformerBlockModule {
public:
    explicit StreamingRelativeConformerBlockModule(ConformerBlockConfig config);
    StreamingConformerBlockOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & pos_emb,
        const RelativeConformerBlockWeights & weights,
        const std::optional<core::TensorValue> & prefix_channel_cache = std::nullopt,
        const std::optional<core::TensorValue> & prefix_time_cache = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    ConformerBlockConfig config_;
};

struct CacheAwareStreamingConfig {
    int64_t chunk_size = 0;
    int64_t shift_size = 0;
    int64_t valid_out_len = 0;
    int64_t cache_drop_size = 0;
    int64_t last_channel_cache_size = 0;
    int64_t pre_encode_cache_size = 0;
    int64_t drop_extra_pre_encoded = 0;
};

CacheAwareStreamingConfig make_cache_aware_streaming_config(
    int64_t subsampling_factor,
    int64_t left_context,
    int64_t right_context,
    int64_t chunk_size,
    int64_t shift_size);

}  // namespace engine::modules

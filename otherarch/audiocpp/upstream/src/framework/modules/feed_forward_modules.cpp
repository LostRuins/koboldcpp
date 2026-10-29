#include "attention/attention_internal.h"
#include "engine/framework/modules/streaming_conv_modules.h"

namespace engine::modules {

using namespace attention::internal;

FeedForwardModule::FeedForwardModule(FeedForwardConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "FeedForwardConfig.hidden_size");
    validate_hidden_positive(config_.intermediate_size, "FeedForwardConfig.intermediate_size");
}

const FeedForwardConfig & FeedForwardModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & FeedForwardModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue FeedForwardModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FeedForwardWeights & weights) const {
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    return build_feed_forward_impl(ctx, input, config_, require_feed_forward_weights(weights, config_.use_bias));
}

const core::ModuleSchema & FeedForwardModule::static_schema() noexcept {
    return kFeedForwardSchema;
}

FeedForwardGeluModule::FeedForwardGeluModule(FeedForwardConfig config) : config_(config) {
    config_.gelu_approximation = GeluApproximation::Tanh;
    validate_hidden_positive(config_.hidden_size, "FeedForwardGeluConfig.hidden_size");
    validate_hidden_positive(config_.intermediate_size, "FeedForwardGeluConfig.intermediate_size");
}

const FeedForwardConfig & FeedForwardGeluModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & FeedForwardGeluModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue FeedForwardGeluModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FeedForwardWeights & weights) const {
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    return build_feed_forward_impl(ctx, input, config_, require_feed_forward_weights(weights, config_.use_bias));
}

const core::ModuleSchema & FeedForwardGeluModule::static_schema() noexcept {
    return kFeedForwardGeluSchema;
}

GatedFeedForwardModule::GatedFeedForwardModule(GatedFeedForwardConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "GatedFeedForwardConfig.hidden_size");
    validate_hidden_positive(config_.intermediate_size, "GatedFeedForwardConfig.intermediate_size");
}

const GatedFeedForwardConfig & GatedFeedForwardModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & GatedFeedForwardModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GatedFeedForwardModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const GatedFeedForwardWeights & weights) const {
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    return build_gated_feed_forward_impl(ctx, input, config_, require_gated_feed_forward_weights(weights, config_.use_bias));
}

const core::ModuleSchema & GatedFeedForwardModule::static_schema() noexcept {
    return kGatedFeedForwardSchema;
}

namespace {

core::TensorValue build_conv_feed_forward_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool causal,
    bool explicit_symmetric_padding,
    bool use_bias) {
    if (causal) {
        return StreamingConv1dModule({
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            use_bias,
            StreamingPadMode::Constant,
            StreamingConv1dPaddingMode::StrictCausal,
        }).build(ctx, input, weights);
    }
    if (explicit_symmetric_padding) {
        const int64_t pad = kernel_size / 2;
        return StreamingConv1dModule({
            in_channels,
            out_channels,
            kernel_size,
            1,
            1,
            use_bias,
            StreamingPadMode::Constant,
            StreamingConv1dPaddingMode::Explicit,
            pad,
            pad,
        }).build(ctx, input, weights);
    }
    return Conv1dModule({in_channels, out_channels, kernel_size, 1, static_cast<int>(kernel_size / 2), 1, use_bias})
        .build(ctx, input, weights);
}

}  // namespace

ConvFeedForwardModule::ConvFeedForwardModule(ConvFeedForwardConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "ConvFeedForwardConfig.hidden_size");
    validate_hidden_positive(config_.intermediate_size, "ConvFeedForwardConfig.intermediate_size");
    if (config_.kernel_size <= 0) {
        throw std::runtime_error("ConvFeedForwardConfig.kernel_size must be positive");
    }
}

const ConvFeedForwardConfig & ConvFeedForwardModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ConvFeedForwardModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ConvFeedForwardModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvFeedForwardWeights & weights) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");

    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = build_conv_feed_forward_conv(
        ctx,
        x,
        weights.proj,
        config_.hidden_size,
        config_.intermediate_size,
        config_.kernel_size,
        config_.causal,
        false,
        config_.use_bias);
    x = GeluModule({config_.gelu_approximation}).build(ctx, x);
    x = build_conv_feed_forward_conv(
        ctx,
        x,
        weights.out,
        config_.intermediate_size,
        config_.hidden_size,
        config_.kernel_size,
        config_.causal,
        false,
        config_.use_bias);
    return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

const core::ModuleSchema & ConvFeedForwardModule::static_schema() noexcept {
    return kConvFeedForwardSchema;
}

GatedConvFeedForwardModule::GatedConvFeedForwardModule(GatedConvFeedForwardConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "GatedConvFeedForwardConfig.hidden_size");
    validate_hidden_positive(config_.intermediate_size, "GatedConvFeedForwardConfig.intermediate_size");
    if (config_.kernel_size <= 0) {
        throw std::runtime_error("GatedConvFeedForwardConfig.kernel_size must be positive");
    }
}

const GatedConvFeedForwardConfig & GatedConvFeedForwardModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & GatedConvFeedForwardModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GatedConvFeedForwardModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const GatedConvFeedForwardWeights & weights) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");

    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    auto gate = build_conv_feed_forward_conv(
        ctx,
        x,
        weights.gate_proj,
        config_.hidden_size,
        config_.intermediate_size,
        config_.kernel_size,
        config_.causal,
        config_.explicit_symmetric_padding,
        config_.use_bias);
    switch (config_.activation) {
        case GatedFeedForwardActivation::Gelu:
            gate = GeluModule({GeluApproximation::Tanh}).build(ctx, gate);
            break;
        case GatedFeedForwardActivation::Silu:
            gate = SiluModule{}.build(ctx, gate);
            break;
    }
    auto up = build_conv_feed_forward_conv(
        ctx,
        x,
        weights.up_proj,
        config_.hidden_size,
        config_.intermediate_size,
        config_.kernel_size,
        config_.causal,
        config_.explicit_symmetric_padding,
        config_.use_bias);
    auto hidden = MulModule{}.build(ctx, gate, up);
    hidden = build_conv_feed_forward_conv(
        ctx,
        hidden,
        weights.down_proj,
        config_.intermediate_size,
        config_.hidden_size,
        config_.kernel_size,
        config_.causal,
        config_.explicit_symmetric_padding,
        config_.use_bias);
    return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
}

const core::ModuleSchema & GatedConvFeedForwardModule::static_schema() noexcept {
    return kGatedConvFeedForwardSchema;
}

}  // namespace engine::modules

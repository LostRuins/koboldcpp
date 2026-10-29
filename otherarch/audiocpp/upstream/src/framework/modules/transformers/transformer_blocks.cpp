#include "../attention/attention_internal.h"

namespace engine::modules {

using namespace attention::internal;

TransformerEncoderBlockModule::TransformerEncoderBlockModule(TransformerEncoderBlockConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "TransformerEncoderBlockConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "TransformerEncoderBlockConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "TransformerEncoderBlockConfig.intermediate_size");
    if (config_.hidden_size % config_.num_heads != 0) {
        throw std::runtime_error("TransformerEncoderBlockConfig.hidden_size must be divisible by num_heads");
    }
}

const TransformerEncoderBlockConfig & TransformerEncoderBlockModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & TransformerEncoderBlockModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TransformerEncoderBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerEncoderBlockWeights & weights) const {
    validate_sequence_input(input, config_.hidden_size, "input");

    const LayerNormModule norm1(make_norm_config(config_.hidden_size, config_.eps));
    AttentionConfig attention_config{
        config_.hidden_size,
        config_.num_heads,
        config_.use_bias,
        config_.projection_precision,
        config_.attention_precision,
        config_.prefix_cache_layout,
    };
    attention_config.use_packed_qkv = config_.use_packed_qkv;
    attention_config.use_flash_attention = config_.use_flash_attention;
    const SelfAttentionModule self_attention(attention_config);
    const LayerNormModule norm2(make_norm_config(config_.hidden_size, config_.eps));
    const FeedForwardModule feed_forward({
        config_.hidden_size,
        config_.intermediate_size,
        config_.use_bias,
        GeluApproximation::ExactErf,
        config_.projection_precision,
    });
    const ResidualAddModule add;

    auto cur = norm1.build(ctx, input, weights.norm1);
    cur = self_attention.build(ctx, cur, weights.self_attention);
    if (weights.layer_scale1.has_value()) {
        cur = LayerScaleModule().build(ctx, cur, *weights.layer_scale1);
    }
    cur = add.build(ctx, input, cur);
    auto ff_in = norm2.build(ctx, cur, weights.norm2);
    auto ff_out = feed_forward.build(ctx, ff_in, weights.feed_forward);
    if (weights.layer_scale2.has_value()) {
        ff_out = LayerScaleModule().build(ctx, ff_out, *weights.layer_scale2);
    }
    return add.build(ctx, cur, ff_out);
}

const core::ModuleSchema & TransformerEncoderBlockModule::static_schema() noexcept {
    return kTransformerEncoderBlockSchema;
}

StreamingTransformerEncoderBlockModule::StreamingTransformerEncoderBlockModule(TransformerEncoderBlockConfig config)
    : config_(config) {
    validate_hidden_positive(config_.hidden_size, "StreamingTransformerEncoderBlockConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "StreamingTransformerEncoderBlockConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "StreamingTransformerEncoderBlockConfig.intermediate_size");
    if (config_.hidden_size % config_.num_heads != 0) {
        throw std::runtime_error("StreamingTransformerEncoderBlockConfig.hidden_size must be divisible by num_heads");
    }
}

const TransformerEncoderBlockConfig & StreamingTransformerEncoderBlockModule::config() const noexcept {
    return config_;
}

StreamingTransformerEncoderBlockOutputs StreamingTransformerEncoderBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TransformerEncoderBlockWeights & weights,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");

    const LayerNormModule norm1(make_norm_config(config_.hidden_size, config_.eps));
    const StreamingSelfAttentionModule self_attention({
        config_.hidden_size,
        config_.num_heads,
        config_.use_bias,
        config_.projection_precision,
        config_.attention_precision,
        config_.prefix_cache_layout,
    });
    const LayerNormModule norm2(make_norm_config(config_.hidden_size, config_.eps));
    const FeedForwardModule feed_forward({
        config_.hidden_size,
        config_.intermediate_size,
        config_.use_bias,
        GeluApproximation::ExactErf,
        config_.projection_precision,
    });
    const ResidualAddModule add;

    auto attn_in = norm1.build(ctx, input, weights.norm1);
    auto attn = self_attention.build(ctx, attn_in, positions, weights.self_attention, prefix_key, prefix_value, attention_mask);
    auto attn_out = attn.output;
    if (weights.layer_scale1.has_value()) {
        attn_out = LayerScaleModule().build(ctx, attn_out, *weights.layer_scale1);
    }
    auto cur = add.build(ctx, input, attn_out);
    auto ff_in = norm2.build(ctx, cur, weights.norm2);
    auto ff_out = feed_forward.build(ctx, ff_in, weights.feed_forward);
    if (weights.layer_scale2.has_value()) {
        ff_out = LayerScaleModule().build(ctx, ff_out, *weights.layer_scale2);
    }
    return {add.build(ctx, cur, ff_out), attn.key, attn.value};
}

StreamingTransformerStackModule::StreamingTransformerStackModule(StreamingTransformerStackConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "StreamingTransformerStackConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "StreamingTransformerStackConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "StreamingTransformerStackConfig.intermediate_size");
    validate_hidden_positive(config_.layers, "StreamingTransformerStackConfig.layers");
}

const StreamingTransformerStackConfig & StreamingTransformerStackModule::config() const noexcept {
    return config_;
}

StreamingTransformerStackOutputs StreamingTransformerStackModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const StreamingTransformerStackWeights & weights,
    const std::optional<StreamingTransformerStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("StreamingTransformerStackWeights layer count does not match config.layers");
    }
    if (prefix_state.has_value() && static_cast<int64_t>(prefix_state->layers.size()) != config_.layers) {
        throw std::runtime_error("StreamingTransformerStackState layer count does not match config.layers");
    }
    auto output = input;
    StreamingTransformerStackState state;
    state.layers.reserve(weights.layers.size());
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        const auto & layer_weights = weights.layers[layer_index];
        const auto * layer_prefix = prefix_state.has_value() ? &prefix_state->layers[layer_index] : nullptr;
        auto layer_outputs = StreamingTransformerEncoderBlockModule({
            config_.hidden_size,
            config_.num_heads,
            config_.intermediate_size,
            config_.eps,
            config_.use_bias,
            config_.projection_precision,
            config_.attention_precision,
            config_.prefix_cache_layout,
        }).build(
            ctx,
            output,
            positions,
            layer_weights,
            layer_prefix != nullptr ? layer_prefix->key : std::nullopt,
            layer_prefix != nullptr ? layer_prefix->value : std::nullopt,
            attention_mask);
        output = layer_outputs.output;
        state.layers.push_back({layer_outputs.key, layer_outputs.value});
    }
    return {output, std::move(state)};
}

ProjectedTransformerModule::ProjectedTransformerModule(ProjectedTransformerConfig config) : config_(config) {
    validate_hidden_positive(config_.input_dimension, "ProjectedTransformerConfig.input_dimension");
    validate_hidden_positive(config_.output_dimension, "ProjectedTransformerConfig.output_dimension");
    validate_hidden_positive(config_.hidden_size, "ProjectedTransformerConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "ProjectedTransformerConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "ProjectedTransformerConfig.intermediate_size");
    validate_hidden_positive(config_.layers, "ProjectedTransformerConfig.layers");
}

const ProjectedTransformerConfig & ProjectedTransformerModule::config() const noexcept {
    return config_;
}

core::TensorValue ProjectedTransformerModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const ProjectedTransformerWeights & weights) const {
    core::validate_rank_between(input_bct, 3, 3, "input_bct");
    if (input_bct.shape.dims[1] != config_.input_dimension) {
        throw std::runtime_error("ProjectedTransformerModule input channel dimension mismatch");
    }

    auto x = permute_tensor(ctx, input_bct, {0, 2, 1});
    x = ensure_contiguous_layout(ctx, x);
    if (weights.input_projection.has_value()) {
        x = LinearModule({config_.input_dimension, config_.hidden_size, config_.use_bias, config_.projection_precision})
                .build(ctx, x, *weights.input_projection);
    }
    TransformerStackWeights transformer_weights{weights.transformer_layers};
    x = TransformerStackModule({
        config_.hidden_size,
        config_.num_heads,
        config_.intermediate_size,
        config_.layers,
        config_.eps,
        config_.use_bias,
        config_.projection_precision,
        config_.attention_precision,
        config_.prefix_cache_layout,
    }).build(ctx, x, transformer_weights);
    if (weights.output_projection.has_value()) {
        x = LinearModule({config_.hidden_size, config_.output_dimension, config_.use_bias, config_.projection_precision})
                .build(ctx, x, *weights.output_projection);
    }
    return permute_tensor(ctx, x, {0, 2, 1});
}

StreamingProjectedTransformerModule::StreamingProjectedTransformerModule(ProjectedTransformerConfig config) : config_(config) {
    validate_hidden_positive(config_.input_dimension, "StreamingProjectedTransformerConfig.input_dimension");
    validate_hidden_positive(config_.output_dimension, "StreamingProjectedTransformerConfig.output_dimension");
    validate_hidden_positive(config_.hidden_size, "StreamingProjectedTransformerConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "StreamingProjectedTransformerConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "StreamingProjectedTransformerConfig.intermediate_size");
    validate_hidden_positive(config_.layers, "StreamingProjectedTransformerConfig.layers");
}

const ProjectedTransformerConfig & StreamingProjectedTransformerModule::config() const noexcept {
    return config_;
}

StreamingProjectedTransformerOutputs StreamingProjectedTransformerModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const core::TensorValue & positions,
    const StreamingProjectedTransformerWeights & weights,
    const std::optional<StreamingTransformerStackState> & prefix_state,
    const std::optional<core::TensorValue> & attention_mask) const {
    core::validate_rank_between(input_bct, 3, 3, "input_bct");
    if (input_bct.shape.dims[1] != config_.input_dimension) {
        throw std::runtime_error("StreamingProjectedTransformerModule input channel dimension mismatch");
    }

    auto x = permute_tensor(ctx, input_bct, {0, 2, 1});
    x = ensure_contiguous_layout(ctx, x);
    if (weights.input_projection.has_value()) {
        x = LinearModule({config_.input_dimension, config_.hidden_size, config_.use_bias, config_.projection_precision})
                .build(ctx, x, *weights.input_projection);
    }
    StreamingTransformerStackWeights transformer_weights{weights.transformer_layers};
    auto stack_outputs = StreamingTransformerStackModule({
        config_.hidden_size,
        config_.num_heads,
        config_.intermediate_size,
        config_.layers,
        config_.eps,
        config_.use_bias,
        config_.projection_precision,
        config_.attention_precision,
        config_.prefix_cache_layout,
    }).build(ctx, x, positions, transformer_weights, prefix_state, attention_mask);
    x = stack_outputs.output;
    if (weights.output_projection.has_value()) {
        x = LinearModule({config_.hidden_size, config_.output_dimension, config_.use_bias, config_.projection_precision})
                .build(ctx, x, *weights.output_projection);
    }
    return {permute_tensor(ctx, x, {0, 2, 1}), std::move(stack_outputs.state)};
}

TransformerStackModule::TransformerStackModule(TransformerStackConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "TransformerStackConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "TransformerStackConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "TransformerStackConfig.intermediate_size");
    validate_hidden_positive(config_.layers, "TransformerStackConfig.layers");
}

const TransformerStackConfig & TransformerStackModule::config() const noexcept {
    return config_;
}

core::TensorValue TransformerStackModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerStackWeights & weights) const {
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("TransformerStackWeights layer count does not match TransformerStackConfig.layers");
    }
    auto output = input;
    for (const auto & layer_weights : weights.layers) {
        output = TransformerEncoderBlockModule({
            config_.hidden_size,
            config_.num_heads,
            config_.intermediate_size,
            config_.eps,
            config_.use_bias,
            config_.projection_precision,
            config_.attention_precision,
            config_.prefix_cache_layout,
        }).build(ctx, output, layer_weights);
    }
    return output;
}

TransformerDecoderBlockModule::TransformerDecoderBlockModule(TransformerDecoderBlockConfig config) : config_(config) {
    validate_hidden_positive(config_.hidden_size, "TransformerDecoderBlockConfig.hidden_size");
    validate_hidden_positive(config_.num_heads, "TransformerDecoderBlockConfig.num_heads");
    validate_hidden_positive(config_.intermediate_size, "TransformerDecoderBlockConfig.intermediate_size");
    if (config_.hidden_size % config_.num_heads != 0) {
        throw std::runtime_error("TransformerDecoderBlockConfig.hidden_size must be divisible by num_heads");
    }
}

const TransformerDecoderBlockConfig & TransformerDecoderBlockModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & TransformerDecoderBlockModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TransformerDecoderBlockModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & memory,
    const TransformerDecoderBlockWeights & weights) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    validate_sequence_input(memory, config_.hidden_size, "memory");

    const LayerNormModule norm1(make_norm_config(config_.hidden_size, config_.eps));
    AttentionConfig self_config{config_.hidden_size, config_.num_heads, config_.use_bias};
    self_config.use_packed_qkv = config_.use_packed_qkv;
    const SelfAttentionModule self_attention(self_config);
    const LayerNormModule norm2(make_norm_config(config_.hidden_size, config_.eps));
    AttentionConfig cross_config{config_.hidden_size, config_.num_heads, config_.use_bias};
    cross_config.use_packed_kv = config_.use_packed_kv;
    const CrossAttentionModule cross_attention(cross_config);
    const LayerNormModule norm3(make_norm_config(config_.hidden_size, config_.eps));
    const FeedForwardModule feed_forward(
        {config_.hidden_size, config_.intermediate_size, config_.use_bias, GeluApproximation::ExactErf,
            GGML_PREC_DEFAULT, config_.activation});
    const ResidualAddModule add;

    auto cur = norm1.build(ctx, input, weights.norm1);
    cur = self_attention.build(ctx, cur, weights.self_attention);
    cur = add.build(ctx, input, cur);

    auto cross_in = norm2.build(ctx, cur, weights.norm2);
    auto cross_out = cross_attention.build(ctx, cross_in, memory, weights.cross_attention);
    cur = add.build(ctx, cur, cross_out);

    auto ff_in = norm3.build(ctx, cur, weights.norm3);
    auto ff_out = feed_forward.build(ctx, ff_in, weights.feed_forward);
    return add.build(ctx, cur, ff_out);
}

core::TensorValue TransformerDecoderBlockModule::build_cached_tail(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerDecoderBlockWeights & weights,
    const core::TensorValue & self_key_cache,
    const core::TensorValue & self_value_cache,
    const core::TensorValue & cache_slot,
    const core::TensorValue & causal_mask,
    const CrossAttentionKeyValue & memory_key_value,
    const core::TensorValue & memory_mask) const {
    validate_sequence_input(input, config_.hidden_size, "input");
    AttentionConfig self_config{config_.hidden_size, config_.num_heads, config_.use_bias};
    self_config.use_packed_qkv = config_.use_packed_qkv;
    self_config.causal = true;
    AttentionConfig cross_config{config_.hidden_size, config_.num_heads, config_.use_bias};
    cross_config.use_packed_kv = config_.use_packed_kv;
    const LayerNormModule norm(make_norm_config(config_.hidden_size, config_.eps));
    const ResidualAddModule add;
    auto x = norm.build(ctx, input, weights.norm1);
    x = SelfAttentionModule(self_config).build_cached_tail(ctx, x, weights.self_attention,
        self_key_cache, self_value_cache, cache_slot, causal_mask).output;
    auto cur = add.build(ctx, input, x);
    x = norm.build(ctx, cur, weights.norm2);
    x = config_.use_flash_cross_attention
        ? CrossAttentionModule(cross_config).build_cached_flash(ctx, x, memory_key_value, weights.cross_attention, memory_mask)
        : CrossAttentionModule(cross_config).build_cached(ctx, x, memory_key_value, weights.cross_attention, memory_mask);
    cur = add.build(ctx, cur, x);
    x = norm.build(ctx, cur, weights.norm3);
    x = FeedForwardModule({config_.hidden_size, config_.intermediate_size, config_.use_bias,
        GeluApproximation::ExactErf, GGML_PREC_DEFAULT, config_.activation}).build(ctx, x, weights.feed_forward);
    return add.build(ctx, cur, x);
}

const core::ModuleSchema & TransformerDecoderBlockModule::static_schema() noexcept {
    return kTransformerDecoderBlockSchema;
}

}  // namespace engine::modules

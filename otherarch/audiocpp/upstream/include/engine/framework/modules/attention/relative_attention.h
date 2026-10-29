#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"

#include <optional>

namespace engine::modules {

class RelativeSelfAttentionModule {
public:
    explicit RelativeSelfAttentionModule(RelativeAttentionConfig config);

    const RelativeAttentionConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const std::optional<core::TensorValue> & pos_emb,
        const RelativeAttentionWeights & weights,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt,
        const std::optional<core::TensorValue> & query_keep_mask = std::nullopt,
        const std::optional<core::TensorValue> & projected_pos_emb = std::nullopt) const;

private:
    RelativeAttentionConfig config_;
};

class StreamingRelativeSelfAttentionModule {
public:
    explicit StreamingRelativeSelfAttentionModule(RelativeAttentionConfig config);

    const RelativeAttentionConfig & config() const noexcept;

    StreamingAttentionOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & pos_emb,
        const RelativeAttentionWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    RelativeAttentionConfig config_;
};

}  // namespace engine::modules

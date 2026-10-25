#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"

#include <optional>

namespace engine::modules {

struct CrossAttentionKeyValue {
    core::TensorValue key;
    core::TensorValue value;
};

class CrossAttentionModule {
public:
    explicit CrossAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const core::TensorValue & memory,
        const AttentionWeights & weights) const;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const core::TensorValue & memory,
        const AttentionWeights & weights,
        const core::TensorValue & memory_mask,
        const core::TensorValue * attention_prior = nullptr,
        core::TensorValue * last_attention = nullptr) const;

    core::TensorValue build_cached(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const CrossAttentionKeyValue & key_value,
        const AttentionWeights & weights,
        const core::TensorValue & memory_mask,
        const core::TensorValue * attention_prior = nullptr,
        core::TensorValue * last_attention = nullptr) const;

    // Opt-in flash path. KV is [B,H,K,D]; mask is contiguous F16 [B|1,H|1,Q,K],
    // with additive scores (0 for allowed, -infinity for excluded positions).
    // Every query must have at least one allowed key. No attention prior/output.
    core::TensorValue build_cached_flash(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & query,
        const CrossAttentionKeyValue & key_value,
        const AttentionWeights & weights,
        const core::TensorValue & attention_mask) const;

    CrossAttentionKeyValue build_key_value(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & memory,
        const AttentionWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    AttentionConfig config_;
};

}  // namespace engine::modules

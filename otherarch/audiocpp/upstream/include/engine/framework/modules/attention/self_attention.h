#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"

#include <optional>

namespace engine::modules {

class SelfAttentionModule {
public:
    explicit SelfAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AttentionWeights & weights) const;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AttentionWeights & weights,
        const std::optional<core::TensorValue> & attention_mask) const;

    StreamingAttentionOutputs build_cached_tail(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AttentionWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & cache_slot,
        const core::TensorValue & attention_mask,
        FastKVSetRowsMode set_rows_mode = FastKVSetRowsMode::BackendViewOptimized) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    AttentionConfig config_;
};

class StreamingSelfAttentionModule {
public:
    explicit StreamingSelfAttentionModule(AttentionConfig config);

    const AttentionConfig & config() const noexcept;

    StreamingAttentionOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const AttentionWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    AttentionConfig config_;
};

}  // namespace engine::modules

#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"

#include <ggml.h>

#include <optional>

namespace engine::modules {

enum class GroupedQueryAttentionLowering {
    ManualRepeat,
    FlashGrouped,
    FlashGroupedViewKV,
};

struct GroupedQueryAttentionConfig {
    int64_t head_dim = 0;
    GroupedQueryAttentionLowering lowering = GroupedQueryAttentionLowering::FlashGrouped;
    ggml_prec precision = GGML_PREC_F32;
    AttentionCausality causality = AttentionCausality::NonCausal;
};

class GroupedQueryAttentionModule {
public:
    explicit GroupedQueryAttentionModule(GroupedQueryAttentionConfig config);

    const GroupedQueryAttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & q_heads,
        const core::TensorValue & k_heads,
        const core::TensorValue & v_heads,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    GroupedQueryAttentionConfig config_;
};

}  // namespace engine::modules

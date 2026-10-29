#pragma once

#include "engine/framework/core/module.h"

#include <ggml.h>

#include <optional>

namespace engine::modules {

enum class ScaledDotProductAttentionLowering {
    Explicit,
    Flash,
    FlashPreserveViews,
    ExplicitCpuPerHead,
    CudaFlashF16Head40,
};

enum class AttentionCausality {
    NonCausal,
    Causal,
};

struct ScaledDotProductAttentionConfig {
    int64_t head_dim = 0;
    ScaledDotProductAttentionLowering lowering = ScaledDotProductAttentionLowering::Explicit;
    ggml_prec precision = GGML_PREC_F32;
    AttentionCausality causality = AttentionCausality::NonCausal;
};

class ScaledDotProductAttentionModule {
public:
    explicit ScaledDotProductAttentionModule(ScaledDotProductAttentionConfig config);

    const ScaledDotProductAttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & q_heads,
        const core::TensorValue & k_heads,
        const core::TensorValue & v_heads,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt,
        // Explicit lowering only: multiply scores by this scalar tensor instead
        // of the default 1/sqrt(head_dim) scale, preserving tensor arithmetic.
        const std::optional<core::TensorValue> & score_scale = std::nullopt) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    ScaledDotProductAttentionConfig config_;
};

}  // namespace engine::modules

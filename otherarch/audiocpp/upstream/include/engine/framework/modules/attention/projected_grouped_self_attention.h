#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>

namespace engine::modules {

enum class ProjectedGroupedSelfAttentionQKNorm {
    None,
    GemmaRMSNorm,
};

struct ProjectedGroupedSelfAttentionConfig {
    int64_t hidden_size = 0;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    bool use_bias = false;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
    ggml_prec attention_precision = GGML_PREC_F32;
    bool use_weight_type_projection_precision = false;
    bool use_fast_cuda_projection = false;
    GroupedQueryAttentionLowering lowering = GroupedQueryAttentionLowering::FlashGrouped;
    AttentionCausality causality = AttentionCausality::NonCausal;
    ProjectedGroupedSelfAttentionQKNorm qk_norm = ProjectedGroupedSelfAttentionQKNorm::None;
    float qk_norm_eps = 1.0e-6F;
    bool use_rope = true;
    bool apply_rope_to_projected_prefix = false;
    int rope_type = GGML_ROPE_TYPE_NEOX;
    float rope_theta = 1000000.0F;
    float local_rope_theta = 10000.0F;
    float rope_freq_scale = 1.0F;
    int64_t sliding_window_pattern = 0;
    float query_pre_attn_scalar = 0.0F;
};

struct ProjectedGroupedSelfAttentionWeights {
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
    NormWeights q_norm;
    NormWeights k_norm;
};

class ProjectedGroupedSelfAttentionModule {
public:
    explicit ProjectedGroupedSelfAttentionModule(ProjectedGroupedSelfAttentionConfig config);

    const ProjectedGroupedSelfAttentionConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const ProjectedGroupedSelfAttentionWeights & weights,
        int64_t layer_index,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    ProjectedGroupedSelfAttentionConfig config_;
};

}  // namespace engine::modules

#include "engine/framework/modules/attention/grouped_query_attention.h"

#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

inline const core::ModulePortSpec kGqaInputs[] = {
    {"q_heads", core::PortKind::Activation, false},
    {"k_heads", core::PortKind::Activation, false},
    {"v_heads", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kGqaOutputs[] = {
    {"context", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kGroupedQueryAttentionSchema = {
    "GroupedQueryAttention",
    "nn.attention",
    kGqaInputs,
    3,
    kGqaOutputs,
    1,
    "Applies grouped-query attention from [batch, heads, steps, dim] Q/K/V tensors and returns [batch, steps, query_heads, dim].",
};

int64_t validate_grouped_heads(
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    int64_t dim) {
    core::validate_rank_between(q, 4, 4, "q_heads");
    core::validate_rank_between(k, 4, 4, "k_heads");
    core::validate_rank_between(v, 4, 4, "v_heads");
    if (q.shape.dims[0] != k.shape.dims[0] || q.shape.dims[0] != v.shape.dims[0]) {
        throw std::runtime_error("GroupedQueryAttention batch size mismatch");
    }
    if (k.shape.dims[1] != v.shape.dims[1]) {
        throw std::runtime_error("GroupedQueryAttention key/value head count mismatch");
    }
    if (q.shape.dims[1] <= 0 || k.shape.dims[1] <= 0 || q.shape.dims[1] % k.shape.dims[1] != 0) {
        throw std::runtime_error("GroupedQueryAttention query heads must be divisible by key/value heads");
    }
    if (k.shape.dims[2] != v.shape.dims[2]) {
        throw std::runtime_error("GroupedQueryAttention key/value step mismatch");
    }
    if (q.shape.dims[3] != dim || k.shape.dims[3] != dim || v.shape.dims[3] != dim) {
        throw std::runtime_error("GroupedQueryAttention head dimension mismatch");
    }
    return q.shape.dims[1] / k.shape.dims[1];
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = core::ensure_backend_addressable_layout(ctx, expanded);
    return core::reshape_tensor(
        ctx,
        expanded,
        core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue build_flash_grouped(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision,
    AttentionCausality causality,
    bool view_kv) {
    if (!attention_mask.has_value() && causality == AttentionCausality::Causal) {
        throw std::runtime_error("GroupedQueryAttention flash lowering requires an explicit causal mask");
    }
    const auto q = core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k = view_kv ? k_heads : core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v = view_kv ? v_heads : core::ensure_backend_addressable_layout(ctx, v_heads);
    ggml_tensor * mask = attention_mask.has_value() ? attention_mask->tensor : nullptr;
    auto * flash = ggml_flash_attn_ext(ctx.ggml, q.tensor, k.tensor, v.tensor, mask, scale, 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], q.shape.dims[3]}),
        GGML_TYPE_F32);
}

}  // namespace

GroupedQueryAttentionModule::GroupedQueryAttentionModule(GroupedQueryAttentionConfig config)
    : config_(config) {
    if (config_.head_dim <= 0) {
        throw std::runtime_error("GroupedQueryAttentionConfig.head_dim must be positive");
    }
}

const GroupedQueryAttentionConfig & GroupedQueryAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & GroupedQueryAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue GroupedQueryAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask) const {
    const int64_t repeats = validate_grouped_heads(q_heads, k_heads, v_heads, config_.head_dim);
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.head_dim));
    switch (config_.lowering) {
        case GroupedQueryAttentionLowering::ManualRepeat: {
            auto k_repeated = repeat_kv_heads(ctx, k_heads, repeats);
            auto v_repeated = repeat_kv_heads(ctx, v_heads, repeats);
            return ScaledDotProductAttentionModule({
                config_.head_dim,
                ScaledDotProductAttentionLowering::Explicit,
                config_.precision,
                config_.causality,
            }).build(ctx, q_heads, k_repeated, v_repeated, attention_mask);
        }
        case GroupedQueryAttentionLowering::FlashGrouped:
            return build_flash_grouped(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality, false);
        case GroupedQueryAttentionLowering::FlashGroupedViewKV:
            return build_flash_grouped(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality, true);
    }
    throw std::runtime_error("Unsupported grouped-query attention lowering");
}

const core::ModuleSchema & GroupedQueryAttentionModule::static_schema() noexcept {
    return kGroupedQueryAttentionSchema;
}

}  // namespace engine::modules

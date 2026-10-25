#include "engine/framework/modules/attention/scaled_dot_product_attention.h"

#include "attention_internal.h"

#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace engine::modules {
namespace {

inline const core::ModulePortSpec kSdpaInputs[] = {
    {"q_heads", core::PortKind::Activation, false},
    {"k_heads", core::PortKind::Activation, false},
    {"v_heads", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kSdpaOutputs[] = {
    {"context", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kScaledDotProductAttentionSchema = {
    "ScaledDotProductAttention",
    "nn.attention",
    kSdpaInputs,
    3,
    kSdpaOutputs,
    1,
    "Applies scaled dot-product attention from pre-split [batch, heads, steps, dim] Q/K/V tensors and returns [batch, steps, heads, dim].",
};

void validate_heads(const core::TensorValue & q, const core::TensorValue & k, const core::TensorValue & v, int64_t dim) {
    core::validate_rank_between(q, 4, 4, "q_heads");
    core::validate_rank_between(k, 4, 4, "k_heads");
    core::validate_rank_between(v, 4, 4, "v_heads");
    if (q.shape.dims[0] != k.shape.dims[0] || q.shape.dims[0] != v.shape.dims[0]) {
        throw std::runtime_error("ScaledDotProductAttention batch size mismatch");
    }
    if (q.shape.dims[1] != k.shape.dims[1] || q.shape.dims[1] != v.shape.dims[1]) {
        throw std::runtime_error("ScaledDotProductAttention head count mismatch");
    }
    if (k.shape.dims[2] != v.shape.dims[2]) {
        throw std::runtime_error("ScaledDotProductAttention key/value step mismatch");
    }
    if (q.shape.dims[3] != dim || k.shape.dims[3] != dim || v.shape.dims[3] <= 0) {
        throw std::runtime_error("ScaledDotProductAttention head dimension mismatch");
    }
}

core::TensorValue build_explicit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision,
    AttentionCausality causality,
    const std::optional<core::TensorValue> & score_scale) {
    const MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    ggml_mul_mat_set_prec(scores.tensor, precision);
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    if (score_scale.has_value()) {
        auto scalar_shape = scores.shape;
        for (size_t axis = 0; axis < scalar_shape.rank; ++axis) scalar_shape.dims[axis] = 1;
        auto multiplier = core::reshape_tensor(ctx, *score_scale, scalar_shape);
        scores = MulModule().build(ctx, scores, RepeatModule({scores.shape}).build(ctx, multiplier));
    }
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        attn = core::wrap_tensor(
            ggml_soft_max_ext(ctx.ggml, scores.tensor, attention_mask->tensor, score_scale ? 1.0F : scale, 0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        if (!score_scale) {
            scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        }
        if (causality == AttentionCausality::Causal) {
            scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        }
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    auto context = matmul.build(ctx, attn, v_heads);
    ggml_mul_mat_set_prec(context.tensor, precision);
    return TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

std::array<int, core::kMaxTensorRank> transpose_last_two_axes(size_t rank) {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    if (rank < 2) {
        throw std::runtime_error("ScaledDotProductAttention CPU matmul requires rank >= 2");
    }
    std::swap(axes[rank - 2], axes[rank - 1]);
    return axes;
}

core::TensorValue cpu_matmul_f32(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs,
    ggml_prec precision) {
    const size_t rank = lhs.shape.rank;
    auto rhs_t = TransposeModule({transpose_last_two_axes(rank), rank}).build(ctx, rhs);
    rhs_t = core::wrap_tensor(ggml_cont(ctx.ggml, rhs_t.tensor), rhs_t.shape, rhs_t.type);
    ggml_tensor * out = ggml_mul_mat(ctx.ggml, rhs_t.tensor, lhs.tensor);
    ggml_mul_mat_set_prec(out, precision);
    core::TensorShape out_shape = lhs.shape;
    out_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    return core::wrap_tensor(out, out_shape, GGML_TYPE_F32);
}

core::TensorValue build_explicit_cpu_per_head_one(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q,
    const core::TensorValue & k,
    const core::TensorValue & v,
    int64_t head_dim,
    float scale,
    ggml_prec precision,
    AttentionCausality causality) {
    if (q.shape.dims[0] != 1 || q.shape.dims[1] != 1 ||
        k.shape.dims[0] != 1 || k.shape.dims[1] != 1 ||
        v.shape.dims[0] != 1 || v.shape.dims[1] != 1) {
        throw std::runtime_error("ScaledDotProductAttention ExplicitCpuPerHead expects one sliced batch/head");
    }
    const int64_t query_steps = q.shape.dims[2];
    const int64_t key_steps = k.shape.dims[2];
    auto q_2d = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({query_steps, head_dim}));
    auto k_2d = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({key_steps, head_dim}));
    auto v_2d = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({key_steps, v.shape.dims[3]}));
    ggml_tensor * scores_raw = ggml_mul_mat(ctx.ggml, k_2d.tensor, q_2d.tensor);
    ggml_mul_mat_set_prec(scores_raw, precision);
    auto scores = core::wrap_tensor(
        scores_raw,
        core::TensorShape::from_dims({query_steps, key_steps}),
        GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
    if (causality == AttentionCausality::Causal) {
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
    }
    scores = core::wrap_tensor(ggml_cont(ctx.ggml, scores.tensor), scores.shape, scores.type);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    auto context = cpu_matmul_f32(ctx, attn, v_2d, precision);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({1, query_steps, 1, v.shape.dims[3]}));
}

core::TensorValue build_explicit_cpu_per_head(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision,
    AttentionCausality causality) {
    if (ctx.backend_type != core::BackendType::Cpu) {
        throw std::runtime_error("ScaledDotProductAttention ExplicitCpuPerHead requires CPU backend");
    }
    if (attention_mask.has_value()) {
        throw std::runtime_error("ScaledDotProductAttention ExplicitCpuPerHead does not support attention masks");
    }
    std::vector<core::TensorValue> batches;
    batches.reserve(static_cast<size_t>(q_heads.shape.dims[0]));
    for (int64_t batch = 0; batch < q_heads.shape.dims[0]; ++batch) {
        const auto q_batch = SliceModule({0, batch, 1}).build(ctx, q_heads);
        const auto k_batch = SliceModule({0, batch, 1}).build(ctx, k_heads);
        const auto v_batch = SliceModule({0, batch, 1}).build(ctx, v_heads);
        std::vector<core::TensorValue> heads;
        heads.reserve(static_cast<size_t>(q_heads.shape.dims[1]));
        for (int64_t head = 0; head < q_heads.shape.dims[1]; ++head) {
            const auto q_head = SliceModule({1, head, 1}).build(ctx, q_batch);
            const auto k_head = SliceModule({1, head, 1}).build(ctx, k_batch);
            const auto v_head = SliceModule({1, head, 1}).build(ctx, v_batch);
            heads.push_back(build_explicit_cpu_per_head_one(
                ctx,
                q_head,
                k_head,
                v_head,
                q_heads.shape.dims[3],
                scale,
                precision,
                causality));
        }
        auto batch_context = heads.front();
        for (size_t i = 1; i < heads.size(); ++i) {
            batch_context = ConcatModule({2}).build(ctx, batch_context, heads[i]);
        }
        batches.push_back(batch_context);
    }
    auto context = batches.front();
    for (size_t i = 1; i < batches.size(); ++i) {
        context = ConcatModule({0}).build(ctx, context, batches[i]);
    }
    return core::ensure_backend_addressable_layout(ctx, context);
}

core::TensorValue build_flash(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision,
    AttentionCausality causality) {
    if (!attention_mask.has_value() && causality == AttentionCausality::Causal) {
        throw std::runtime_error("ScaledDotProductAttention flash lowering requires an explicit causal mask");
    }
    const auto q = core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k = core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v = core::ensure_backend_addressable_layout(ctx, v_heads);
    ggml_tensor * mask = attention_mask.has_value() ? attention_mask->tensor : nullptr;
    auto * flash = ggml_flash_attn_ext(ctx.ggml, q.tensor, k.tensor, v.tensor, mask, scale, 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], q.shape.dims[3]}),
        GGML_TYPE_F32);
}

core::TensorValue build_flash_preserve_views(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision,
    AttentionCausality causality) {
    if (!attention_mask.has_value() && causality == AttentionCausality::Causal) {
        throw std::runtime_error("ScaledDotProductAttention view-preserving flash lowering requires an explicit causal mask");
    }
    ggml_tensor * mask = attention_mask.has_value() ? attention_mask->tensor : nullptr;
    auto * flash = ggml_flash_attn_ext(ctx.ggml, q_heads.tensor, k_heads.tensor, v_heads.tensor, mask, scale, 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], q_heads.shape.dims[3]}),
        GGML_TYPE_F32);
}

core::TensorValue build_cuda_flash_f16_head40(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale) {
    if (ctx.backend_type != core::BackendType::Cuda ||
        q_heads.type != GGML_TYPE_F16 || k_heads.type != GGML_TYPE_F16 ||
        v_heads.type != GGML_TYPE_F16 || q_heads.shape.dims[3] != 40 ||
        attention_mask.has_value()) {
        throw std::runtime_error("CUDA FP16 head-40 attention requires FP16 Q/K/V and no mask");
    }
    // kcpp: Audio.cpp carries a dedicated CUDA head-40 kernel. The standard
    // flash-attention op supports the same shape; cast its F32 output to retain
    // the requested F16 contract.
    auto portable = build_flash(
        ctx, q_heads, k_heads, v_heads, attention_mask, scale,
        GGML_PREC_DEFAULT, AttentionCausality::NonCausal);
    return core::wrap_tensor(
        ggml_cast(ctx.ggml, portable.tensor, GGML_TYPE_F16),
        portable.shape,
        GGML_TYPE_F16);
}

}  // namespace

ScaledDotProductAttentionModule::ScaledDotProductAttentionModule(ScaledDotProductAttentionConfig config)
    : config_(config) {
    if (config_.head_dim <= 0) {
        throw std::runtime_error("ScaledDotProductAttentionConfig.head_dim must be positive");
    }
}

const ScaledDotProductAttentionConfig & ScaledDotProductAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ScaledDotProductAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ScaledDotProductAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & score_scale) const {
    validate_heads(q_heads, k_heads, v_heads, config_.head_dim);
    if (score_scale && (config_.lowering != ScaledDotProductAttentionLowering::Explicit ||
                       score_scale->shape.num_elements() != 1)) {
        throw std::runtime_error("ScaledDotProductAttention tensor score scale requires explicit lowering and one element");
    }
    if ((config_.lowering == ScaledDotProductAttentionLowering::Flash ||
         config_.lowering == ScaledDotProductAttentionLowering::FlashPreserveViews) &&
        v_heads.shape.dims[3] != config_.head_dim) {
        throw std::runtime_error("ScaledDotProductAttention flash lowering requires equal Q/K/V widths");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.head_dim));
    switch (config_.lowering) {
        case ScaledDotProductAttentionLowering::Explicit:
            return build_explicit(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality, score_scale);
        case ScaledDotProductAttentionLowering::ExplicitCpuPerHead:
            return build_explicit_cpu_per_head(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality);
        case ScaledDotProductAttentionLowering::Flash:
            return build_flash(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality);
        case ScaledDotProductAttentionLowering::FlashPreserveViews:
            return build_flash_preserve_views(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision, config_.causality);
        case ScaledDotProductAttentionLowering::CudaFlashF16Head40:
            if (config_.causality != AttentionCausality::NonCausal) {
                throw std::runtime_error("CUDA FP16 head-40 attention does not support causal attention");
            }
            return build_cuda_flash_f16_head40(ctx, q_heads, k_heads, v_heads, attention_mask, scale);
    }
    throw std::runtime_error("Unsupported scaled dot-product attention lowering");
}

const core::ModuleSchema & ScaledDotProductAttentionModule::static_schema() noexcept {
    return kScaledDotProductAttentionSchema;
}

}  // namespace engine::modules

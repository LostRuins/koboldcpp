#include "engine/framework/modules/positional_modules.h"

#include "engine/framework/modules/structural_modules.h"

#include <array>
#include <stdexcept>
#include <string>

namespace engine::modules {
namespace {

void require_positive(int64_t value, const char * name) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void validate_split_rope_table(
    const core::TensorValue & input,
    const core::TensorValue & table,
    int64_t half_dim,
    const char * label) {
    core::validate_rank_between(table, 4, 4, label);
    if (!((table.shape.dims[0] == 1 || table.shape.dims[0] == input.shape.dims[0]) &&
          table.shape.dims[1] == input.shape.dims[1] &&
          table.shape.dims[2] == input.shape.dims[2] &&
          table.shape.dims[3] == half_dim)) {
        throw std::runtime_error(std::string("SplitRoPE ") + label + " shape mismatch");
    }
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    if (value.shape.rank == like.shape.rank) {
        bool same = true;
        for (size_t axis = 0; axis < value.shape.rank; ++axis) {
            same = same && value.shape.dims[axis] == like.shape.dims[axis];
        }
        if (same) {
            return value;
        }
    }
    return core::wrap_tensor(ggml_repeat(ctx.ggml, value.tensor, like.tensor), like.shape, value.type);
}

}  // namespace

RoPEModule::RoPEModule(RoPEConfig config) : config_(config) {
    require_positive(config_.dimensions, "RoPEConfig.dimensions");
}

core::TensorValue RoPEModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue * frequency_factors) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 2, core::kMaxTensorRank, "input");
    core::validate_shape(positions, core::TensorShape::from_dims({input.shape.dims[1]}), "positions");
    if (positions.type != GGML_TYPE_I32) {
        throw std::runtime_error("RoPE positions must be GGML_TYPE_I32");
    }
    if (config_.dimensions > input.shape.last_dim()) {
        throw std::runtime_error("RoPE dimensions exceed input last dimension");
    }
    if (frequency_factors != nullptr) {
        core::validate_shape(
            *frequency_factors,
            core::TensorShape::from_dims({config_.dimensions / 2}),
            "RoPE frequency factors");
    }
    if (config_.intermediate_round_type.has_value()) {
        if (config_.mode != GGML_ROPE_TYPE_NEOX || config_.dimensions != input.tensor->ne[0]) {
            throw std::runtime_error("Rounded RoPE requires full-dimension NEOX rotation");
        }
        const int64_t half = config_.dimensions / 2;
        auto * basis = ggml_new_tensor_4d(ctx.ggml, GGML_TYPE_F32, half, 1, input.tensor->ne[2], 1);
        auto * identity = ggml_concat(ctx.ggml, ggml_fill(ctx.ggml, basis, 1.0f),
            ggml_fill(ctx.ggml, basis, 0.0f), 0);
        auto * table = ggml_rope_ext(ctx.ggml, identity, positions.tensor,
            frequency_factors ? frequency_factors->tensor : nullptr, config_.dimensions,
            config_.mode, 0, config_.theta, config_.freq_scale, config_.ext_factor,
            config_.attn_factor, config_.beta_fast, config_.beta_slow);
        table = *config_.intermediate_round_type == GGML_TYPE_BF16
            ? ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, table, GGML_TYPE_BF16), GGML_TYPE_F32)
            : ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, table, *config_.intermediate_round_type), GGML_TYPE_F32);
        auto * cos = ggml_view_4d(ctx.ggml, table, half, 1, table->ne[2], 1,
            table->nb[1], table->nb[2], table->nb[3], 0);
        auto * sin = ggml_view_4d(ctx.ggml, table, half, 1, table->ne[2], 1,
            table->nb[1], table->nb[2], table->nb[3], half * sizeof(float));
        const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
        auto * lower = ggml_view_4d(ctx.ggml, contiguous.tensor, half, input.tensor->ne[1],
            input.tensor->ne[2], input.tensor->ne[3], contiguous.tensor->nb[1],
            contiguous.tensor->nb[2], contiguous.tensor->nb[3], 0);
        auto * upper = ggml_view_4d(ctx.ggml, contiguous.tensor, half, input.tensor->ne[1],
            input.tensor->ne[2], input.tensor->ne[3], contiguous.tensor->nb[1],
            contiguous.tensor->nb[2], contiguous.tensor->nb[3], half * sizeof(float));
        std::array<ggml_tensor *, 4> products{
            ggml_mul(ctx.ggml, lower, cos), ggml_mul(ctx.ggml, upper, sin),
            ggml_mul(ctx.ggml, upper, cos), ggml_mul(ctx.ggml, lower, sin)};
        for (auto *& product : products) {
            product = *config_.intermediate_round_type == GGML_TYPE_BF16
                ? ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, product, GGML_TYPE_BF16), GGML_TYPE_F32)
                : ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, product, *config_.intermediate_round_type), GGML_TYPE_F32);
        }
        return core::wrap_tensor(ggml_concat(ctx.ggml,
            ggml_sub(ctx.ggml, products[0], products[1]), ggml_add(ctx.ggml, products[2], products[3]), 0),
            input.shape, GGML_TYPE_F32);
    }
    return core::wrap_tensor(
        ggml_rope_ext(
            ctx.ggml,
            input.tensor,
            positions.tensor,
            frequency_factors != nullptr ? frequency_factors->tensor : nullptr,
            static_cast<int>(config_.dimensions),
            config_.mode,
            0,
            config_.theta,
            config_.freq_scale,
            config_.ext_factor,
            config_.attn_factor,
            config_.beta_fast,
            config_.beta_slow),
        input.shape,
        input.type);
}

SplitRoPEModule::SplitRoPEModule(SplitRoPEConfig config) : config_(config) {
    require_positive(config_.dimensions, "SplitRoPEConfig.dimensions");
    if (config_.dimensions % 2 != 0) {
        throw std::runtime_error("SplitRoPEConfig.dimensions must be even");
    }
}

const SplitRoPEConfig & SplitRoPEModule::config() const noexcept {
    return config_;
}

core::TensorValue SplitRoPEModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_last_dim(input, config_.dimensions, "input");
    const int64_t half_dim = config_.dimensions / 2;
    validate_split_rope_table(input, cos, half_dim, "cos");
    validate_split_rope_table(input, sin, half_dim, "sin");

    const int last_axis = static_cast<int>(input.shape.rank - 1);
    const auto x1 = SliceModule({last_axis, 0, half_dim}).build(ctx, input);
    const auto x2 = SliceModule({last_axis, half_dim, half_dim}).build(ctx, input);
    const auto cos_full = repeat_like(ctx, cos, x1);
    const auto sin_full = repeat_like(ctx, sin, x1);
    auto first = core::wrap_tensor(
        ggml_sub(
            ctx.ggml,
            ggml_mul(ctx.ggml, x1.tensor, cos_full.tensor),
            ggml_mul(ctx.ggml, x2.tensor, sin_full.tensor)),
        x1.shape,
        GGML_TYPE_F32);
    auto second = core::wrap_tensor(
        ggml_add(
            ctx.ggml,
            ggml_mul(ctx.ggml, x2.tensor, cos_full.tensor),
            ggml_mul(ctx.ggml, x1.tensor, sin_full.tensor)),
        x2.shape,
        GGML_TYPE_F32);
    return ConcatModule({last_axis}).build(ctx, first, second);
}

SplitRoPEAttentionModule::SplitRoPEAttentionModule(SplitRoPEAttentionConfig config) : config_(config) {
    require_positive(config_.heads, "SplitRoPEAttentionConfig.heads");
    require_positive(config_.head_dim, "SplitRoPEAttentionConfig.head_dim");
    if (config_.head_dim % 2 != 0) {
        throw std::runtime_error("SplitRoPEAttentionConfig.head_dim must be even");
    }
}

const SplitRoPEAttentionConfig & SplitRoPEAttentionModule::config() const noexcept {
    return config_;
}

core::TensorValue SplitRoPEAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cos,
    const core::TensorValue & sin) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_last_dim(input, config_.heads * config_.head_dim, "input");
    auto x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config_.heads, config_.head_dim}));
    x = TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    return SplitRoPEModule({config_.head_dim}).build(ctx, x, cos, sin);
}

}  // namespace engine::modules

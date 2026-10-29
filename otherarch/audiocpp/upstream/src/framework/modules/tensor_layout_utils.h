#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::modules::tensor_layout {

inline core::TensorValue ensure_contiguous_layout_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

inline core::TensorValue ensure_contiguous_nontransposed_layout_if_needed(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor) && !ggml_is_transposed(value.tensor)) {
        return value;
    }
    return core::wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
}

inline core::TensorValue swap_channel_time_axes_3d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

inline core::TensorValue swap_channel_time_axes_4d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, input);
}

}  // namespace engine::modules::tensor_layout

#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"

namespace engine::modules {

enum class FastConv1dKind {
    // Uses the explicit ggml_conv_1d_fast_1d_im2col entrypoint.
    MinittsFast1dIm2col,
};

class FastConv1dModule {
public:
    FastConv1dModule(Conv1dConfig config, FastConv1dKind kind);

    const Conv1dConfig & config() const noexcept;
    FastConv1dKind kind() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv1dConfig config_;
    FastConv1dKind kind_;
};

}  // namespace engine::modules

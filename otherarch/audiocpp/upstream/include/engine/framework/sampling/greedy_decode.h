#pragma once

#include "engine/framework/core/module.h"

namespace engine::sampling {

class GreedyDecodeModule {
public:
    const engine::core::ModuleSchema & schema() const noexcept;
    engine::core::TensorValue build(engine::core::ModuleBuildContext & ctx, const engine::core::TensorValue & logits) const;
    static const engine::core::ModuleSchema & static_schema() noexcept;
};

}  // namespace engine::sampling

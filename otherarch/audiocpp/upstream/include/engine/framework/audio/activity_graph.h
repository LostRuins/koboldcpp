#pragma once

#include "engine/framework/core/module.h"

namespace engine::sampling {

struct VADGateConfig {
    float threshold = 0.0f;
};

class VADGateModule {
public:
    explicit VADGateModule(VADGateConfig config);

    const VADGateConfig & config() const noexcept;
    const engine::core::ModuleSchema & schema() const noexcept;
    engine::core::TensorValue build(engine::core::ModuleBuildContext & ctx, const engine::core::TensorValue & energy) const;
    static const engine::core::ModuleSchema & static_schema() noexcept;

private:
    VADGateConfig config_;
};

}  // namespace engine::sampling

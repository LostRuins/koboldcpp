#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

struct PositionalEncodingConfig {
    int64_t hidden_size = 0;
};

struct PositionalEncodingWeights {
    core::TensorValue encoding;
};

class PositionalEncodingModule {
public:
    explicit PositionalEncodingModule(PositionalEncodingConfig config);

    const PositionalEncodingConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PositionalEncodingWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    PositionalEncodingConfig config_;
};

}  // namespace engine::modules

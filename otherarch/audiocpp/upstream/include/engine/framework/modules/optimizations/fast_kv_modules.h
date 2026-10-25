#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

enum class FastKVSetRowsMode {
    Exact,
    BackendViewOptimized,
};

struct FastKVSetRowsConfig {
    FastKVSetRowsMode mode = FastKVSetRowsMode::Exact;
};

class FastKVSetRowsModule {
public:
    explicit FastKVSetRowsModule(FastKVSetRowsConfig config = {});

    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & cache,
        const core::TensorValue & row,
        const core::TensorValue & row_index) const;

    // Explicit single-sequence, multi-token cache update.
    core::TensorValue build_block(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & cache,
        const core::TensorValue & rows,
        const core::TensorValue & indices) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    FastKVSetRowsConfig config_;
};

}  // namespace engine::modules

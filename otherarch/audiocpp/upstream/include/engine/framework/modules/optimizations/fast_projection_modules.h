#pragma once

#include "engine/framework/modules/linear_module.h"

namespace engine::modules {

struct FastPackedProjection4Config {
    int64_t in_features = 0;
    int64_t out_features = 0;
    ggml_prec precision = GGML_PREC_DEFAULT;
};

class FastPackedProjection4Module {
public:
    explicit FastPackedProjection4Module(FastPackedProjection4Config config);

    const FastPackedProjection4Config & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const LinearWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    FastPackedProjection4Config config_;
};

}  // namespace engine::modules

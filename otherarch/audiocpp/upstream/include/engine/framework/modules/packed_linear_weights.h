#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

struct PackedLinearSource {
    std::string weight;
    std::optional<std::string> bias;
    int64_t out_features = 0;
};

struct PackedLinearWeightsConfig {
    int64_t in_features = 0;
    std::vector<PackedLinearSource> sources;
    bool use_bias = false;
};

class PackedLinearWeightsBuilder {
public:
    explicit PackedLinearWeightsBuilder(PackedLinearWeightsConfig config);

    const PackedLinearWeightsConfig & config() const noexcept;

    LinearWeights build(
        core::BackendWeightStore & store,
        const assets::TensorSource & source,
        assets::TensorStorageType storage_type) const;

private:
    PackedLinearWeightsConfig config_;
};

}  // namespace engine::modules

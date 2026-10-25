#pragma once

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <string>
#include <vector>

namespace engine::rknn {

// Selected engine artifacts may each contain a comma-separated list of paths. Resolve and
// deduplicate them, price through the driver (file size when unavailable), and sum residency.
// Component/artifact selection and eviction policy belong to the caller.
runtime::ModelResourcePlan engine_resources(
    const std::vector<std::string> & artifacts,
    const std::filesystem::path & model_path);

}  // namespace engine::rknn

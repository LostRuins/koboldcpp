#include "engine/framework/rknn/engine_resources.h"

#include "engine/framework/rknn/engine.h"

#include <set>
#include <sstream>

namespace engine::rknn {

runtime::ModelResourcePlan engine_resources(
    const std::vector<std::string> & artifacts,
    const std::filesystem::path & model_path) {
    runtime::ModelResourcePlan out;
    std::error_code ec;
    const auto root = std::filesystem::is_directory(model_path, ec)
        ? model_path : model_path.parent_path();
    std::set<std::filesystem::path> counted;
    for (const auto & artifact : artifacts) {
        std::istringstream paths(artifact);
        std::string declared;
        while (std::getline(paths, declared, ',')) {
            if (declared.empty()) {
                continue;
            }
            const std::filesystem::path path = resolve_engine_path(declared, root);
            // Missing files are diagnosed by the real engine load, not the memory guard.
            if (!std::filesystem::is_regular_file(path, ec)) {
                continue;
            }
            const auto canonical = std::filesystem::weakly_canonical(path);
            if (!counted.insert(canonical).second) {
                continue;
            }
            const auto mib = engine_memory_mib(path.string(), "model resources");
            const auto bytes = mib > 0
                ? static_cast<size_t>(mib) * (1024ull * 1024ull)
                : static_cast<size_t>(std::filesystem::file_size(path));
            out.engines.push_back({canonical, bytes});
            out.resident_bytes += bytes;
        }
    }
    return out;
}

}  // namespace engine::rknn

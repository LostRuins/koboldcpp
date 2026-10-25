#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace engine::model_spec {

enum class ResourceKind {
    Files,
    Tensors,
};

class ScopedSpecOverride {
public:
    explicit ScopedSpecOverride(const std::optional<std::filesystem::path> & path,
        const std::filesystem::path & model_path = {});
    ~ScopedSpecOverride();

    ScopedSpecOverride(const ScopedSpecOverride &) = delete;
    ScopedSpecOverride & operator=(const ScopedSpecOverride &) = delete;

private:
    std::optional<std::filesystem::path> previous_;
    std::optional<std::filesystem::path> previous_model_path_;
};

[[nodiscard]] std::filesystem::path default_spec_path(std::string_view family);
[[nodiscard]] std::filesystem::path default_package_spec_path(std::string_view family);
[[nodiscard]] std::filesystem::path default_contract_spec_path(std::string_view family);
// Like default_contract_spec_path, but returns nullopt instead of throwing when the active
// GGUF embeds a legacy (pre-v1) spec and no schema-v1 contract is available anywhere.
[[nodiscard]] std::optional<std::filesystem::path> find_contract_spec_path(std::string_view family);

[[nodiscard]] engine::io::json::Value load_spec(const std::filesystem::path & spec_path);
[[nodiscard]] engine::io::json::Value load_contract_spec(const std::filesystem::path & spec_path);

[[nodiscard]] assets::ResourceBundle load_resource_bundle(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path);
[[nodiscard]] assets::ResourceBundle load_resource_bundle_for_family(
    const std::filesystem::path & model_path,
    std::string_view family);

[[nodiscard]] std::vector<assets::ResourceFile> discover_resources(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ResourceKind kind);

}  // namespace engine::model_spec

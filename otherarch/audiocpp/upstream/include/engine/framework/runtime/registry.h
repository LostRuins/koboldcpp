#pragma once

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::runtime {

struct RegistryConfig {
    std::vector<std::string> enabled_families;
};

class ModelRegistry {
public:
    void register_loader(std::shared_ptr<IVoiceModelLoader> loader);

    bool empty() const noexcept;
    size_t size() const noexcept;
    std::vector<std::string> families() const;
    bool supports_family(const std::string & family) const noexcept;
    /** Path-free loader catalog for ``--list-loaders --json``. */
    std::vector<LoaderAdvertisement> advertise_loaders() const;
    ModelInspection inspect(const ModelLoadRequest & request) const;
    ModelInspection inspect(const std::filesystem::path & model_path) const;
    std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const;
    std::unique_ptr<ILoadedVoiceModel> load(const std::filesystem::path & model_path) const;

    /**
     * What the named family's session options ask for beyond weight files, from the loader that
     * owns the family (see IVoiceModelLoader::model_resources). A family with no loader registered
     * in this registry -- or one that reports nothing -- yields an empty plan.
     */
    ModelResourcePlan model_resources(
        const std::string & family,
        const std::filesystem::path & model_path,
        const std::unordered_map<std::string, std::string> & session_options) const;

private:
    void validate_request(const ModelLoadRequest & request) const;
    const IVoiceModelLoader * find_loader(const ModelLoadRequest & request) const;

    std::vector<std::shared_ptr<IVoiceModelLoader>> loaders_;
};

RegistryConfig load_registry_config(const std::filesystem::path & path);
ModelRegistry make_registry_from_config(
    const RegistryConfig & config,
    const std::vector<std::shared_ptr<IVoiceModelLoader>> & available_loaders);
ModelRegistry make_default_registry(const std::optional<std::filesystem::path> & config_path = std::nullopt);

/**
 * Close the load-progress bracket opened by ModelRegistry::load: emit
 * ``runtime.load.phase loaded`` and the final upload_progress curve point.
 * Hosts call this once the task session is built — that is when the weight
 * stores have actually uploaded — not when the registry load returns.
 * Safe (no-op) when the load was never traced or was already closed.
 */
void finish_model_load_trace();

}  // namespace engine::runtime

#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::model_spec {
enum class ResourceKind;
}

namespace engine::runtime {

struct NamedAsset {
    std::string id;
    std::filesystem::path path;
};

struct TaskCapability {
    VoiceTaskKind task = VoiceTaskKind::Vad;
    std::vector<RunMode> modes;
};

struct CapabilitySet {
    std::vector<TaskCapability> supported_tasks;
    std::vector<std::string> languages;
    bool supports_speaker_reference = false;
    bool supports_style_condition = false;
    bool supports_timestamps = false;
};

struct ModelMetadata {
    std::string family;
    std::string variant;
    std::string description;
    std::vector<std::string> config_candidates;
    std::vector<std::string> weight_candidates;
};

struct CliOptionInfo {
    std::string name;
    std::string value_name;
    std::string description;
    bool required = false;
    std::optional<std::string> default_value = std::nullopt;
    std::optional<std::string> min_value = std::nullopt;
    std::optional<std::string> max_value = std::nullopt;
};

struct ModelCliInterface {
    std::vector<CliOptionInfo> request_options;
    std::vector<CliOptionInfo> session_options;
    std::vector<CliOptionInfo> load_options;
};

struct ModelInspection {
    ModelMetadata metadata;
    CapabilitySet capabilities;
    ModelCliInterface cli;
    std::filesystem::path model_root;
    std::vector<NamedAsset> discovered_configs;
    std::vector<NamedAsset> discovered_weights;
};

struct ModelLoadRequest {
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> model_spec_override = std::nullopt;
    std::optional<std::string> family_hint = std::nullopt;
    std::optional<std::string> config_id = std::nullopt;
    std::optional<std::string> weight_id = std::nullopt;
    std::unordered_map<std::string, std::string> options;
};

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<NamedAsset> discover_named_assets_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    engine::model_spec::ResourceKind kind);

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept;

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role);

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) noexcept(false);

class ILoadedVoiceModel {
public:
    virtual ~ILoadedVoiceModel() = default;

    virtual const ModelMetadata & metadata() const noexcept = 0;
    virtual const CapabilitySet & capabilities() const noexcept = 0;
    virtual std::unique_ptr<IVoiceTaskSession> create_task_session(
        const TaskSpec & task,
        const SessionOptions & options) const = 0;
};

struct LoaderAdvertisement {
    std::string family;
    CapabilitySet capabilities;
    std::string instructions_policy;
    std::vector<std::string> api_endpoints;
};

/**
 * One selected resource, with its model-reported resident cost in bytes.
 */
struct ModelEngineRequirement {
    std::filesystem::path path;
    size_t resident_bytes = 0;
};

/**
 * Resources beyond weight files and their aggregate residency. The model applies its own
 * residency policy; consumers use the paths for deduplication and add resident_bytes as given.
 */
struct ModelResourcePlan {
    std::vector<ModelEngineRequirement> engines;
    size_t resident_bytes = 0;
};

/** Map advertised capabilities to the HTTP surfaces they normally use. */
std::vector<std::string> default_api_endpoints_for_capabilities(const CapabilitySet & capabilities);

class IVoiceModelLoader {
public:
    virtual ~IVoiceModelLoader() = default;

    virtual std::string family() const = 0;
    /** Alternate family names accepted for family hints (e.g. "habibi" for "f5_tts"). */
    virtual std::vector<std::string> family_aliases() const { return {}; }
    virtual bool can_load(const ModelLoadRequest & request) const = 0;
    virtual ModelInspection inspect(const ModelLoadRequest & request) const = 0;
    virtual std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const = 0;

    /**
     * Path-free loader catalog for ``--list-loaders --json``.
     * Override ``advertised_capabilities`` (and policy when non-default) on each loader.
     */
    virtual CapabilitySet advertised_capabilities() const;
    virtual std::string advertised_instructions_policy() const;
    virtual std::vector<std::string> advertised_api_endpoints() const;
    LoaderAdvertisement advertise() const;

    /**
     * The artifacts this family will hold resident for one load of `model_path` under these
     * session options, priced by the family. A family whose stages are all ggml graphs reports
     * nothing: its host-consumer estimate is the weight files plus the caller's runtime factor.
     */
    virtual ModelResourcePlan model_resources(
        const std::filesystem::path & model_path,
        const std::unordered_map<std::string, std::string> & session_options) const {
        (void)model_path;
        (void)session_options;
        return {};
    }
};

}  // namespace engine::runtime

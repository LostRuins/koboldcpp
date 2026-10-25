#include "engine/framework/runtime/model.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/filesystem.h"

#include <unordered_map>
#include <stdexcept>

namespace engine::runtime {

namespace {

std::string normalized_candidate_key(const std::string & relative_candidate) {
    return std::filesystem::path(relative_candidate).generic_string();
}

std::string candidate_stem_or_filename(const std::string & relative_candidate) {
    const auto path = std::filesystem::path(relative_candidate);
    if (!path.stem().empty()) {
        return path.stem().string();
    }
    return path.filename().string();
}

}

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates) {
    std::unordered_map<std::string, int> stem_counts;
    for (const auto & candidate : relative_candidates) {
        ++stem_counts[candidate_stem_or_filename(candidate)];
    }

    std::vector<NamedAsset> assets;
    assets.reserve(relative_candidates.size());
    for (const auto & candidate : relative_candidates) {
        const auto path = root / candidate;
        if (engine::io::is_existing_file(path)) {
            const auto base_id = candidate_stem_or_filename(candidate);
            const auto id = stem_counts[base_id] == 1 ? base_id : normalized_candidate_key(candidate);
            assets.push_back({id, std::filesystem::weakly_canonical(path)});
        }
    }
    return assets;
}

std::vector<NamedAsset> discover_named_assets_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    engine::model_spec::ResourceKind kind) {
    const auto resources = engine::model_spec::discover_resources(model_path, spec_path, kind);
    std::vector<NamedAsset> assets;
    assets.reserve(resources.size());
    for (const auto & resource : resources) {
        assets.push_back({resource.id, resource.path});
    }
    return assets;
}

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept {
    for (const auto & asset : assets) {
        if (asset.id == id) {
            return &asset;
        }
    }
    return nullptr;
}

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role) {
    const auto * asset = find_named_asset(assets, id);
    if (asset == nullptr) {
        throw std::runtime_error("unknown " + role + " id: " + id);
    }
    return *asset;
}

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) {
    if (id.has_value()) {
        return &require_named_asset(assets, *id, role);
    }
    if (assets.empty()) {
        return nullptr;
    }
    if (assets.size() > 1) {
        throw std::runtime_error("multiple " + role + " candidates found; specify --" + role);
    }
    return &assets.front();
}

std::vector<std::string> default_api_endpoints_for_capabilities(const CapabilitySet & capabilities) {
    bool has_asr = false;
    bool has_speech = false;
    bool has_other = false;
    for (const auto & task : capabilities.supported_tasks) {
        switch (task.task) {
        case VoiceTaskKind::Asr:
            has_asr = true;
            break;
        case VoiceTaskKind::Tts:
        case VoiceTaskKind::VoiceCloning:
        case VoiceTaskKind::VoiceDesign:
            has_speech = true;
            break;
        default:
            has_other = true;
            break;
        }
    }
    if (has_asr && !has_speech && !has_other) {
        return {"/v1/audio/transcriptions"};
    }
    if (has_speech && !has_other) {
        return {"/v1/audio/speech"};
    }
    if (has_speech && has_other) {
        return {"/v1/tasks/run", "/v1/audio/speech"};
    }
    return {"/v1/tasks/run"};
}

CapabilitySet IVoiceModelLoader::advertised_capabilities() const {
    // Path-free catalog entry. Each loader overrides with the same task/mode
    // contract it exposes via inspect/load (without requiring a model path).
    return {};
}

std::string IVoiceModelLoader::advertised_instructions_policy() const {
    // Generic default from advertised tasks. Loaders with a different contract override.
    for (const auto & task : advertised_capabilities().supported_tasks) {
        if (task.task == VoiceTaskKind::Tts || task.task == VoiceTaskKind::VoiceDesign
            || task.task == VoiceTaskKind::VoiceCloning) {
            return "openai_instruct";
        }
    }
    return "none";
}

std::vector<std::string> IVoiceModelLoader::advertised_api_endpoints() const {
    return default_api_endpoints_for_capabilities(advertised_capabilities());
}

LoaderAdvertisement IVoiceModelLoader::advertise() const {
    LoaderAdvertisement row;
    row.family = family();
    row.capabilities = advertised_capabilities();
    row.instructions_policy = advertised_instructions_policy();
    row.api_endpoints = advertised_api_endpoints();
    return row;
}

}  // namespace engine::runtime

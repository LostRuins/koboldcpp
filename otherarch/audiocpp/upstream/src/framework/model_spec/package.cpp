#include "engine/framework/model_spec/package.h"

#include "engine/framework/model_spec/schema.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::model_spec {
namespace {

thread_local std::optional<std::filesystem::path> active_model_spec_override;
thread_local std::optional<std::filesystem::path> active_model_path;
thread_local bool active_embedded_spec_checked = false;
thread_local std::optional<assets::GgufEmbeddedModelSpec> active_embedded_spec;

const std::unordered_map<std::string, std::string_view> & builtin_model_specs() {
    static const std::unordered_map<std::string, std::string_view> specs = {
#include "model_specs.inc"
    };
    return specs;
}

std::optional<std::string_view> builtin_model_spec(const std::filesystem::path & spec_path) {
    if (spec_path.parent_path() != "@builtin")
        return std::nullopt;
    const auto it = builtin_model_specs().find(spec_path.stem().string());
    if (it == builtin_model_specs().end())
        return std::nullopt;
    return it->second;
}

std::string model_spec_description(const std::filesystem::path & spec_path) {
    if (spec_path.parent_path() == "@gguf") {
        return "embedded GGUF model spec for family '" + spec_path.stem().string() + "'";
    }
    if (spec_path.parent_path() == "@builtin") {
        return "builtin model spec for family '" + spec_path.stem().string() + "'";
    }
    return "model spec '" + spec_path.string() + "'";
}

bool is_gguf_file(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path))
        return false;
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".gguf";
}

std::string gguf_file_list(const std::vector<std::filesystem::path> & files) {
    std::string names;
    for (size_t i = 0; i < files.size(); ++i) {
        if (i != 0) {
            names += ", ";
        }
        names += files[i].filename().string();
    }
    return names;
}

// Only fires for a directory holding several GGUFs and no model.gguf to pick between them
// (e.g. both vevo2-q8_0.gguf and vevo2-f16.gguf installed side by side).
std::string ambiguous_directory_gguf_message(const std::filesystem::path & directory,
    const std::vector<std::filesystem::path> & files) {
    return "model directory contains " + std::to_string(files.size()) + " GGUF files: " + directory.string() +
           "; found: " + gguf_file_list(files) +
           "; pass one of them directly with --model, or keep a single GGUF in the directory";
}

std::string directory_gguf_hint(std::string_view family) {
    if (!active_model_path.has_value()) {
        return {};
    }
    const auto files = assets::directory_gguf_files(*active_model_path);
    if (files.empty()) {
        return {};
    }
    if (files.size() > 1) {
        return ambiguous_directory_gguf_message(*active_model_path, files);
    }
    return "GGUF has no embedded model spec for family '" + std::string(family) + "': " +
           files.front().string() + "; install model_specs/" + std::string(family) +
           ".json next to it, or pass --model-spec-override";
}

std::optional<std::filesystem::path> active_gguf_path() {
    if (!active_model_path.has_value())
        return std::nullopt;
    const auto & path = *active_model_path;
    if (is_gguf_file(path)) {
        return engine::io::normalize_path(path);
    }
    if (const auto found = assets::find_directory_gguf(path)) {
        return engine::io::normalize_path(*found);
    }
    return std::nullopt;
}

const std::optional<assets::GgufEmbeddedModelSpec> & embedded_model_spec() {
    if (!active_embedded_spec_checked) {
        active_embedded_spec_checked = true;
        if (const auto gguf = active_gguf_path()) {
            active_embedded_spec = assets::read_gguf_embedded_model_spec(*gguf);
        }
    }
    return active_embedded_spec;
}

bool embedded_model_spec_has_v1_contract(
    const assets::GgufEmbeddedModelSpec & embedded,
    std::string_view family) {
    if (embedded.family != family) {
        throw std::runtime_error("GGUF embeds model spec for family '" + embedded.family + "', not '" +
                                 std::string(family) + "'");
    }
    const auto root = engine::io::json::parse(embedded.json);
    return root.find("schema_version") != nullptr;
}

bool external_spec_matches_family(const std::filesystem::path & path, std::string_view family) {
    if (!engine::io::is_existing_file(path))
        return false;
    try {
        const auto root = engine::io::json::parse_file(path);
        return engine::io::json::require_string(root, "family") == family;
    } catch (const std::exception & error) {
        throw std::runtime_error("invalid candidate model spec '" + path.string() +
                                 "' while resolving family '" + std::string(family) + "': " + error.what());
    }
}

std::optional<std::filesystem::path> discover_external_model_spec(std::string_view family) {
    std::vector<std::filesystem::path> candidates;
    if (active_model_path.has_value()) {
        const auto root = engine::io::is_existing_directory(*active_model_path) ? *active_model_path
                                                                                : active_model_path->parent_path();
        candidates.push_back(root / "model_specs" / (std::string(family) + ".json"));
        candidates.push_back(root / (std::string(family) + ".json"));
        candidates.push_back(root / "model_spec.json");
        candidates.push_back(root.parent_path() / "model_specs" / (std::string(family) + ".json"));
    }
    auto cursor = std::filesystem::current_path();
    while (true) {
        candidates.push_back(cursor / "model_specs" / (std::string(family) + ".json"));
        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty())
            break;
        cursor = parent;
    }
    for (const auto & candidate : candidates) {
        if (external_spec_matches_family(candidate, family)) {
            return engine::io::normalize_path(candidate);
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> discover_workspace_model_spec(std::string_view family) {
    std::vector<std::filesystem::path> candidates;
    auto cursor = std::filesystem::current_path();
    while (true) {
        candidates.push_back(cursor / "model_specs" / (std::string(family) + ".json"));
        const auto parent = cursor.parent_path();
        if (parent == cursor || parent.empty())
            break;
        cursor = parent;
    }
    for (const auto & candidate : candidates) {
        if (external_spec_matches_family(candidate, family)) {
            return engine::io::normalize_path(candidate);
        }
    }
    return std::nullopt;
}

engine::io::json::Value parse_model_spec(const std::filesystem::path & spec_path, bool validate) {
    engine::io::json::Value root;
    if (spec_path.parent_path() == "@gguf") {
        const auto & spec = embedded_model_spec();
        if (!spec.has_value() || spec->family != spec_path.stem().string()) {
            throw std::runtime_error("embedded GGUF model spec is not available for: " +
                                     spec_path.stem().string());
        }
        root = engine::io::json::parse(spec->json);
        if (engine::io::json::require_string(root, "family") != spec->family) {
            throw std::runtime_error("embedded GGUF model spec family metadata does not match its JSON");
        }
    } else if (const auto text = builtin_model_spec(spec_path)) {
        root = engine::io::json::parse(*text);
    } else {
        root = engine::io::json::parse_file(spec_path);
    }
    if (validate) {
        validate_spec(root, model_spec_description(spec_path));
    }
    return root;
}

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return engine::io::normalize_path(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return engine::io::normalize_path(model_path.parent_path());
    }
    throw std::runtime_error("model path does not exist: " + model_path.string());
}

std::string require_source_format(const engine::io::json::Value & source) {
    return engine::io::json::require_string(source, "format");
}

using ResourceRoots = std::unordered_map<std::string, std::filesystem::path>;

ResourceRoots resolve_source_roots(const std::filesystem::path & model_root, const engine::io::json::Value & source,
    const std::optional<std::filesystem::path> & standalone_gguf) {
    ResourceRoots roots;
    const auto & root_object = source.require("roots").as_object();
    for (const auto & [id, value] : root_object) {
        const auto root_value = value.as_string();
        if (root_value == "$gguf" && !standalone_gguf.has_value()) {
            throw std::runtime_error("model source requires a GGUF root: " + id);
        }
        const auto root_path = root_value == "$gguf" ? *standalone_gguf : model_root / root_value;
        if (!engine::io::is_existing_directory(root_path) && !engine::io::is_existing_file(root_path)) {
            throw std::runtime_error("missing model root: " + id + "=" + root_path.string());
        }
        roots.emplace(id, engine::io::normalize_path(root_path));
    }
    return roots;
}

std::filesystem::path resolve_resource_ref(const std::unordered_map<std::string, std::filesystem::path> & roots,
    const std::string & ref) {
    const auto split = ref.find(':');
    if (split == std::string::npos || split == 0) {
        throw std::runtime_error("invalid model resource reference: " + ref);
    }
    const auto root_id = ref.substr(0, split);
    const auto relative_path = ref.substr(split + 1);
    const auto root_it = roots.find(root_id);
    if (root_it == roots.end()) {
        throw std::runtime_error("unknown model resource root: " + root_id);
    }
    if (relative_path.empty()) {
        return root_it->second;
    }
    return root_it->second / relative_path;
}

void add_resource_map(assets::ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        const auto path = resolve_resource_ref(roots, ref.as_string());
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing model package file '" + id + "': " + path.string());
        }
        bundle.add_file(id, path);
    }
}

std::filesystem::path resolve_tensor_source_ref(const ResourceRoots & roots, const engine::io::json::Value & value,
    std::string & prefix) {
    if (value.is_string()) {
        prefix.clear();
        return resolve_resource_ref(roots, value.as_string());
    }
    const auto & object = value.as_object();
    const auto source_it = object.find("source");
    if (source_it == object.end()) {
        throw std::runtime_error("model tensor source object requires source");
    }
    const auto prefix_it = object.find("prefix");
    prefix = prefix_it == object.end() ? "" : prefix_it->second.as_string();
    return resolve_resource_ref(roots, source_it->second.as_string());
}

void add_tensor_map(assets::ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        std::string prefix;
        const auto path = resolve_tensor_source_ref(roots, ref, prefix);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("missing model tensor source '" + id + "': " + path.string());
        }
        bundle.add_tensor_source(id, path, std::move(prefix));
    }
}

// Tensor sources a package may or may not ship. The required `tensors` map is
// checked eagerly, which is what a package wants for the weights it cannot run
// without; a family whose variants are separate multi-gigabyte downloads needs
// the other answer, or installing one variant means downloading all of them.
// A model that selects a missing variant reports it itself, where it can name
// the variant instead of a resource id.
void add_optional_tensor_map(assets::ResourceBundle & bundle,
    const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        std::string prefix;
        const auto path = resolve_tensor_source_ref(roots, ref, prefix);
        if (engine::io::is_existing_file(path)) {
            bundle.add_tensor_source(id, path, std::move(prefix));
        }
    }
}

void add_optional_resource_map(assets::ResourceBundle & bundle, const ResourceRoots & roots,
    const engine::io::json::Value * map_value) {
    if (map_value == nullptr || map_value->is_null()) {
        return;
    }
    for (const auto & [id, ref] : map_value->as_object()) {
        const auto path = resolve_resource_ref(roots, ref.as_string());
        if (engine::io::is_existing_file(path)) {
            bundle.add_file(id, path);
        }
    }
}

std::vector<assets::ResourceFile> resources_from_resource_map(const ResourceRoots & roots,
    const engine::io::json::Value * map_value,
    bool required) {
    std::vector<assets::ResourceFile> out;
    if (map_value == nullptr || map_value->is_null()) {
        return out;
    }
    const auto & map = map_value->as_object();
    out.reserve(map.size());
    for (const auto & [id, ref] : map) {
        std::string prefix;
        const auto path = resolve_tensor_source_ref(roots, ref, prefix);
        if (required || engine::io::is_existing_file(path)) {
            out.push_back({id, engine::io::normalize_path(path)});
        }
    }
    return out;
}

assets::ResourceBundle load_source(const std::filesystem::path & model_root, const engine::io::json::Value & source,
    const ResourceRoots & roots) {
    assets::ResourceBundle bundle(model_root);
    add_resource_map(bundle, roots, source.find("files"));
    add_optional_resource_map(bundle, roots, source.find("optional_files"));
    add_tensor_map(bundle, roots, source.find("tensors"));
    add_optional_tensor_map(bundle, roots, source.find("optional_tensors"));
    return bundle;
}

std::vector<assets::ResourceFile> discover_safetensors_source_resources(const engine::io::json::Value & source,
    ResourceKind kind,
    const ResourceRoots & roots) {
    auto resources = resources_from_resource_map(
        roots, source.find(kind == ResourceKind::Files ? "files" : "tensors"), true);
    auto optional = resources_from_resource_map(
        roots, source.find(kind == ResourceKind::Files ? "optional_files" : "optional_tensors"), false);
    resources.insert(resources.end(), optional.begin(), optional.end());
    return resources;
}

struct SelectedSource {
    std::filesystem::path model_root;
    engine::io::json::Value source;
    ResourceRoots roots;
    std::string spec_description;
    std::string source_format;
};

SelectedSource select_source(const std::filesystem::path & model_path, const std::filesystem::path & model_root,
    const std::string & spec_description, const engine::io::json::Value & source) {
    const auto format = require_source_format(source);
    if (format == "gguf") {
        const auto prepared = assets::prepare_model_directory(model_path);
        auto roots = resolve_source_roots(prepared.model_root, source, prepared.standalone_gguf);
        return SelectedSource{prepared.model_root, source, std::move(roots), spec_description, format};
    }
    if (format == "safetensors") {
        auto roots = resolve_source_roots(model_root, source, std::nullopt);
        return SelectedSource{model_root, source, std::move(roots), spec_description, format};
    }
    throw std::runtime_error("unsupported model source format: " + format);
}

SelectedSource require_selected_source(const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    const auto model_root = resolve_model_root(model_path);
    const auto spec_description = model_spec_description(spec_path);
    engine::io::json::Value spec;
    try {
        spec = parse_model_spec(spec_path, true);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to parse " + spec_description + ": " + error.what());
    }
    const auto & sources = spec.require("sources").as_array();
    const bool explicit_gguf_path = is_gguf_file(model_path);
    const bool use_gguf = explicit_gguf_path || assets::find_directory_gguf(model_path).has_value();
    if (!use_gguf) {
        // A directory whose GGUFs cannot be narrowed down to one would otherwise fall through to
        // the safetensors source and fail much later on a config file the GGUF package never ships.
        if (const auto files = assets::directory_gguf_files(model_path); files.size() > 1) {
            throw std::runtime_error(ambiguous_directory_gguf_message(model_path, files));
        }
    }
    for (const auto & source : sources) {
        const auto format = require_source_format(source);
        if ((use_gguf && format == "gguf") || (!use_gguf && format == "safetensors")) {
            try {
                return select_source(model_path, model_root, spec_description, source);
            } catch (const std::exception & error) {
                throw std::runtime_error("failed to select " + std::string(use_gguf ? "GGUF" : "safetensors") +
                                         " source from " + spec_description + ": " + error.what());
            }
        }
    }
    throw std::runtime_error(std::string("no ") + (use_gguf ? "gguf" : "safetensors") + " model source in " +
                             spec_description);
}

}  // namespace

ScopedSpecOverride::ScopedSpecOverride(const std::optional<std::filesystem::path> & path,
    const std::filesystem::path & model_path)
    : previous_(active_model_spec_override), previous_model_path_(active_model_path) {
    active_model_spec_override = path;
    active_model_path = model_path.empty() ? std::nullopt : std::make_optional(model_path);
    active_embedded_spec_checked = false;
    active_embedded_spec.reset();
}

ScopedSpecOverride::~ScopedSpecOverride() {
    active_model_spec_override = std::move(previous_);
    active_model_path = std::move(previous_model_path_);
    active_embedded_spec_checked = false;
    active_embedded_spec.reset();
}

std::filesystem::path default_package_spec_path(std::string_view family) {
    if (active_model_spec_override.has_value() && *active_model_spec_override != "embedded") {
        auto path = *active_model_spec_override;
        if (engine::io::is_existing_directory(path)) {
            path /= std::string(family) + ".json";
        }
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("model spec override not found: " + path.string());
        }
        return engine::io::normalize_path(path);
    }
    if (const auto & embedded = embedded_model_spec(); embedded.has_value()) {
        if (embedded->family != family) {
            throw std::runtime_error("GGUF embeds model spec for family '" + embedded->family + "', not '" +
                                     std::string(family) + "'");
        }
        return std::filesystem::path("@gguf") / (std::string(family) + ".json");
    }
    if (active_model_spec_override.has_value() && *active_model_spec_override == "embedded") {
        throw std::runtime_error("model spec override 'embedded' requires a GGUF with an embedded model spec "
                                 "for family '" + std::string(family) + "'");
    }
    if (builtin_model_specs().find(std::string(family)) != builtin_model_specs().end()) {
        return std::filesystem::path("@builtin") / (std::string(family) + ".json");
    }
    if (const auto external = discover_external_model_spec(family)) {
        return *external;
    }
    if (const auto hint = directory_gguf_hint(family); !hint.empty()) {
        throw std::runtime_error(hint);
    }
    throw std::runtime_error("model spec not found for family '" + std::string(family) +
                             "' (provide --model-spec-override, embed it in the GGUF, enable "
                             "AUDIOCPP_DEPLOYMENT_BUILD, or install model_specs/" +
                             std::string(family) + ".json)");
}

std::filesystem::path default_contract_spec_path(std::string_view family) {
    if (const auto path = find_contract_spec_path(family)) {
        engine::debug::trace_log_scalar("runtime.model.contract_spec", path->string());
        return *path;
    }
    throw std::runtime_error("GGUF for family '" + std::string(family) +
                             "' embeds a legacy model spec, but no current schema-v1 model contract was found. "
                             "Install model_specs/" + std::string(family) +
                             ".json, enable AUDIOCPP_DEPLOYMENT_BUILD, regenerate the GGUF, "
                             "or pass --model-spec-override.");
}

std::optional<std::filesystem::path> find_contract_spec_path(std::string_view family) {
    if (active_model_spec_override.has_value()) {
        if (*active_model_spec_override == "embedded") {
            const auto path = default_package_spec_path(family);
            if (!embedded_model_spec_has_v1_contract(*embedded_model_spec(), family)) {
                throw std::runtime_error("model spec override 'embedded' requires a schema-v1 embedded contract "
                                         "for family '" + std::string(family) + "'");
            }
            return path;
        }
        auto path = *active_model_spec_override;
        if (engine::io::is_existing_directory(path)) {
            path /= std::string(family) + ".json";
        }
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("model spec override not found: " + path.string());
        }
        return engine::io::normalize_path(path);
    }
    if (const auto external = discover_workspace_model_spec(family)) {
        return *external;
    }
    if (builtin_model_specs().find(std::string(family)) != builtin_model_specs().end()) {
        return std::filesystem::path("@builtin") / (std::string(family) + ".json");
    }
    bool active_gguf_has_legacy_spec = false;
    if (const auto gguf = active_gguf_path()) {
        const auto & embedded = embedded_model_spec();
        if (!embedded.has_value()) {
            throw std::runtime_error("GGUF for family '" + std::string(family) +
                                     "' does not embed an audio.cpp model spec: " + gguf->string() +
                                     ". Published GGUF packages must embed a model spec; regenerate the GGUF "
                                     "or pass --model-spec-override.");
        }
        if (embedded_model_spec_has_v1_contract(*embedded, family)) {
            return std::filesystem::path("@gguf") / (std::string(family) + ".json");
        }
        active_gguf_has_legacy_spec = true;
    }
    if (const auto hint = directory_gguf_hint(family); !hint.empty()) {
        throw std::runtime_error(hint);
    }
    if (active_gguf_has_legacy_spec) {
        return std::nullopt;
    }
    throw std::runtime_error("model contract spec not found for family '" + std::string(family) +
                             "' (provide --model-spec-override, install model_specs/" +
                             std::string(family) + ".json, enable AUDIOCPP_DEPLOYMENT_BUILD, or embed it in the GGUF)");
}

std::filesystem::path default_spec_path(std::string_view family) {
    return default_package_spec_path(family);
}

assets::ResourceBundle load_resource_bundle(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path) {
    auto selected = require_selected_source(model_path, spec_path);
    try {
        auto bundle = load_source(selected.model_root, selected.source, selected.roots);
        // A GGUF that embeds its sidecars gets a materialized root, so the bundle root is not
        // always the directory the model lives in. Keep that one too: it is what an operator
        // names an installed file by.
        bundle.set_model_directory(resolve_model_root(model_path));
        return bundle;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to load model resources using " + selected.spec_description +
                                 " source '" + selected.source_format + "': " + error.what());
    }
}

assets::ResourceBundle load_resource_bundle_for_family(
    const std::filesystem::path & model_path,
    std::string_view family) {
    ScopedSpecOverride scoped(active_model_spec_override, model_path);
    return load_resource_bundle(model_path, default_spec_path(family));
}

engine::io::json::Value load_spec(const std::filesystem::path & spec_path) {
    return parse_model_spec(spec_path, true);
}

engine::io::json::Value load_contract_spec(const std::filesystem::path & spec_path) {
    return parse_model_spec(spec_path, false);
}

std::vector<assets::ResourceFile> discover_resources(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    ResourceKind kind) {
    auto selected = require_selected_source(model_path, spec_path);
    try {
        return discover_safetensors_source_resources(selected.source, kind, selected.roots);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to discover model resources using " + selected.spec_description +
                                 " source '" + selected.source_format + "': " + error.what());
    }
}

}  // namespace engine::model_spec

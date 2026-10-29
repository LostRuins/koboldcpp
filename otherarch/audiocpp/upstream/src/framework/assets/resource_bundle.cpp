#include "engine/framework/assets/resource_bundle.h"

#include "engine/framework/io/filesystem.h"

#include <algorithm>
#include <stdexcept>

namespace engine::assets {
namespace {

std::filesystem::path require_model_root(const std::filesystem::path & model_root) {
    if (model_root.empty()) {
        throw std::runtime_error("asset model_root is required for model-relative resources");
    }
    return model_root;
}

std::filesystem::path resolve_model_file(
    const std::filesystem::path & model_root,
    const std::filesystem::path & relative_path) {
    return require_model_root(model_root) / relative_path;
}

}  // namespace

ResourceBundle::ResourceBundle(std::filesystem::path model_root)
    : ResourceBundle(std::move(model_root), ResourceBundleOptions{}) {}

ResourceBundle::ResourceBundle(std::filesystem::path model_root, ResourceBundleOptions options)
    : model_root_(std::move(model_root)),
      cache_mutex_(options.synchronized_cache_access ? std::make_shared<std::mutex>() : nullptr) {}

void ResourceBundle::add_file(std::string id, const std::filesystem::path & path) {
    if (id.empty()) {
        throw std::runtime_error("asset resource id must not be empty");
    }
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("asset resource file does not exist: " + path.string());
    }
    auto [it, inserted] = files_.emplace(std::move(id), engine::io::normalize_path(path));
    if (!inserted) {
        throw std::runtime_error("duplicate asset resource id: " + it->first);
    }
}

void ResourceBundle::add_tensor_source(std::string id, const std::filesystem::path & path, std::string tensor_prefix) {
    const auto key = id;
    add_file(std::move(id), path);
    auto [it, inserted] = tensor_resources_.emplace(
        key,
        TensorResource{
            engine::io::normalize_path(path),
            std::move(tensor_prefix),
        });
    if (!inserted) {
        throw std::runtime_error("duplicate asset tensor resource id: " + it->first);
    }
}

void ResourceBundle::add_model_file(std::string id, const std::filesystem::path & relative_path) {
    add_file(std::move(id), resolve_model_file(model_root_, relative_path));
}

bool ResourceBundle::add_optional_model_file(std::string id, const std::filesystem::path & relative_path) {
    const auto path = resolve_model_file(model_root_, relative_path);
    if (!engine::io::is_existing_file(path)) {
        return false;
    }
    add_file(std::move(id), path);
    return true;
}

void ResourceBundle::add_model_files(std::initializer_list<ResourceSpec> specs) {
    for (const auto & spec : specs) {
        if (spec.required) {
            add_model_file(spec.id, spec.relative_path);
            continue;
        }
        (void) add_optional_model_file(spec.id, spec.relative_path);
    }
}

const std::filesystem::path & ResourceBundle::model_root() const noexcept {
    return model_root_;
}

void ResourceBundle::set_model_directory(std::filesystem::path path) {
    model_directory_ = std::move(path);
}

const std::filesystem::path & ResourceBundle::model_directory() const noexcept {
    return model_directory_.empty() ? model_root_ : model_directory_;
}

bool ResourceBundle::has_file(std::string_view id) const noexcept {
    return files_.find(std::string(id)) != files_.end();
}

const std::filesystem::path * ResourceBundle::find_file(std::string_view id) const noexcept {
    const auto it = files_.find(std::string(id));
    if (it == files_.end()) {
        return nullptr;
    }
    return &it->second;
}

const std::filesystem::path & ResourceBundle::require_file(std::string_view id) const {
    const auto * path = find_file(id);
    if (path == nullptr) {
        throw std::runtime_error("missing asset resource: " + std::string(id));
    }
    return *path;
}

std::vector<ResourceFile> ResourceBundle::files() const {
    std::vector<ResourceFile> out;
    out.reserve(files_.size());
    for (const auto & [id, path] : files_) {
        out.push_back({id, path});
    }
    std::sort(out.begin(), out.end(), [](const ResourceFile & lhs, const ResourceFile & rhs) {
        return lhs.id < rhs.id;
    });
    return out;
}

std::string ResourceBundle::read_text(std::string_view id) const {
    return engine::io::read_text_file(require_file(id));
}

engine::io::json::Value ResourceBundle::parse_json(std::string_view id) const {
    return engine::io::json::parse_file(require_file(id));
}

engine::io::json::Value ResourceBundle::parse_jsonc(std::string_view id) const {
    return engine::io::json::parse_jsonc_file(require_file(id));
}

engine::io::yaml::FlattenedDocument ResourceBundle::parse_flattened_yaml(std::string_view id) const {
    // kcpp: libyaml is outside the selective integration. Keep the API so
    // upstream headers stay intact, but reject YAML packages explicitly.
    throw std::runtime_error("YAML resources are not enabled in the KoboldCpp Audio.cpp integration: " +
                             std::string(id));
}

std::shared_ptr<const TensorSource> ResourceBundle::open_tensor_source(std::string_view id) const {
    return open_tensor_source(id, TensorSourceOptions{});
}

std::shared_ptr<const TensorSource> ResourceBundle::open_tensor_source(
    std::string_view id, TensorSourceOptions options) const {
    std::unique_lock<std::mutex> lock;
    if (cache_mutex_) { lock = std::unique_lock<std::mutex>(*cache_mutex_); }
    auto & sources = options.synchronized_access ? synchronized_tensor_sources_ : tensor_sources_;
    auto & paths = options.synchronized_access ? synchronized_tensor_sources_by_path_ : tensor_sources_by_path_;
    const auto key = std::string(id);
    const auto it = sources.find(key);
    if (it != sources.end()) {
        return it->second;
    }
    const auto resource = tensor_resources_.find(key);
    const auto path = resource == tensor_resources_.end()
        ? require_file(id)
        : resource->second.path;
    const auto path_key = engine::io::normalize_path(path).generic_string();
    auto base = paths.find(path_key);
    if (base == paths.end()) {
        base = paths.emplace(
            path_key, engine::assets::open_tensor_source(path, options)).first;
    }
    auto source = resource == tensor_resources_.end()
        ? base->second
        : engine::assets::make_prefixed_tensor_source(
              base->second, resource->second.prefix);
    sources.emplace(key, source);
    return source;
}

}  // namespace engine::assets

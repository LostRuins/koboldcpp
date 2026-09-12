#include "model_loader.h"

#include <algorithm>
#include <atomic>
#include <limits>

#include "core/util.h"
#include "name_conversion.h"

static uint64_t next_source_revision() {
    static std::atomic<uint64_t> revision{0};
    return revision.fetch_add(1, std::memory_order_relaxed) + 1;
}

bool ModelLoader::read_file_stamp(const std::string& path, FileStamp& stamp) {
    std::error_code error;
    const auto file_path = std::filesystem::u8path(path);
    stamp.path           = path;
    stamp.size           = 0;
    stamp.modified       = std::filesystem::last_write_time(file_path, error);
    if (!error && std::filesystem::is_regular_file(file_path, error)) {
        stamp.size = std::filesystem::file_size(file_path, error);
    }
    if (error) {
        LOG_ERROR("cannot inspect model source '%s': %s", path.c_str(), error.message().c_str());
        return false;
    }
    return true;
}

bool ModelLoader::file_unchanged(const FileStamp& stamp) {
    std::error_code error;
    if (!std::filesystem::exists(std::filesystem::u8path(stamp.path), error)) {
        return false;
    }
    FileStamp current;
    return read_file_stamp(stamp.path, current) &&
           current.size == stamp.size && current.modified == stamp.modified;
}

void ModelLoader::invalidate_file_data() {
    file_data.clear();
    model_files_processed = false;
}

void ModelLoader::rebuild_catalog() {
    tensor_storage_map.clear();
    metadata_.clear();
    for (const auto& file : files_) {
        if (file.scope == FileScope::Isolated)
            continue;
        for (const auto& entry : file.tensors) {
            tensor_storage_map[entry.first] = entry.second;
        }
        for (const auto& entry : file.metadata) {
            metadata_[entry.first] = entry.second;
        }
    }
    if (names_converted_) {
        const SDVersion version = version_ == VERSION_COUNT ? get_sd_version() : version_;
        tensor_storage_map.clear();
        for (const auto& file : files_) {
            if (file.scope == FileScope::Isolated)
                continue;
            for (const auto& entry : file.tensors) {
                TensorStorage tensor            = entry.second;
                tensor.name                     = convert_tensor_name(tensor.name, version);
                tensor_storage_map[tensor.name] = std::move(tensor);
            }
        }
    }
    std::set<size_t> used_files;
    for (const auto& file : files_) {
        for (const auto& entry : file.tensors) {
            used_files.insert(entry.second.file_index);
        }
    }
    for (size_t i = 0; i < file_paths_.size(); ++i) {
        if (used_files.count(i) == 0) {
            file_paths_[i].clear();
        }
    }
    set_wtype_override(wtype_override_, tensor_type_rules_);
}

bool ModelLoader::add_file_impl(const std::string& path, const std::string& prefix, FileId* id, bool force, FileScope scope) {
    FileStamp root;
    if (!read_file_stamp(path, root)) {
        return false;
    }
    auto existing = std::find_if(files_.begin(), files_.end(), [&](const FileRecord& file) {
        return file.path == root.path && file.prefix == prefix && file.scope == scope;
    });
    if (existing != files_.end() && !force &&
        std::all_of(existing->dependencies.begin(), existing->dependencies.end(), file_unchanged)) {
        if (id != nullptr) {
            *id = existing->id;
        }
        return true;
    }

    ModelLoader parsed;
    try {
        if (!parsed.parse_file(root.path, prefix)) {
            return false;
        }
    } catch (const std::exception& error) {
        LOG_ERROR("invalid model source '%s': %s", path.c_str(), error.what());
        return false;
    }

    std::vector<size_t> file_indices;
    std::vector<FileStamp> physical_files;
    for (const auto& physical_path : parsed.file_paths_) {
        FileStamp stamp;
        if (!read_file_stamp(physical_path, stamp)) {
            return false;
        }
        parsed.parsed_dependencies_.push_back(stamp);
        file_indices.push_back(add_file_path(stamp.path));
        physical_files.push_back(std::move(stamp));
    }
    for (auto& entry : parsed.tensor_storage_map) {
        auto& tensor = entry.second;
        // Pickle preserves rank-zero scalars; GGML uses a one-element dimension.
        if (tensor.n_dims == 0) {
            tensor.n_dims = 1;
        }
        if (tensor.n_dims < 1 || tensor.n_dims > SD_MAX_DIMS || tensor.type < 0 ||
            tensor.type >= GGML_TYPE_COUNT || tensor.file_index >= parsed.file_paths_.size()) {
            LOG_ERROR("invalid tensor metadata for '%s'", tensor.name.c_str());
            return false;
        }
        uint64_t elements = 1;
        for (int i = 0; i < tensor.n_dims; ++i) {
            if (tensor.ne[i] < 0 || (elements != 0 && static_cast<uint64_t>(tensor.ne[i]) > INT64_MAX / elements)) {
                LOG_ERROR("invalid tensor dimensions for '%s'", tensor.name.c_str());
                return false;
            }
            elements *= tensor.ne[i];
        }
        const uint64_t block_size = ggml_blck_size(tensor.type);
        const uint64_t type_size  = ggml_type_size(tensor.type) * ((tensor.is_f64 || tensor.is_i64) ? 2 : 1);
        if (block_size == 0 || type_size == 0 || elements % block_size != 0 || elements / block_size > INT64_MAX / type_size) {
            LOG_ERROR("invalid tensor storage size for '%s'", tensor.name.c_str());
            return false;
        }
        if (tensor.index_in_zip < 0) {
            const auto& stamp = physical_files[tensor.file_index];
            if (tensor.offset > stamp.size || elements / block_size * type_size > stamp.size - tensor.offset) {
                LOG_ERROR("tensor '%s' extends beyond its model file", tensor.name.c_str());
                return false;
            }
        }
    }
    if (!std::all_of(parsed.parsed_dependencies_.begin(), parsed.parsed_dependencies_.end(), file_unchanged)) {
        LOG_ERROR("model source changed while reading metadata: '%s'", path.c_str());
        return false;
    }

    FileRecord record;
    // Snapshots and independently created loaders must never alias different versions.
    record.revision = next_source_revision();
    record.id       = existing == files_.end() ? record.revision : existing->id;
    ++revision_;
    record.path   = root.path;
    record.prefix = prefix;
    record.scope  = scope;
    std::set<std::string> seen_dependencies;
    for (auto& stamp : parsed.parsed_dependencies_) {
        if (seen_dependencies.insert(stamp.path).second) {
            record.dependencies.push_back(std::move(stamp));
        }
    }
    record.metadata = std::move(parsed.metadata_);
    record.tensors  = std::move(parsed.tensor_storage_map);
    for (auto& entry : record.tensors) {
        entry.second.file_index    = file_indices[entry.second.file_index];
        entry.second.file_id       = record.id;
        entry.second.file_revision = record.revision;
    }
    if (id != nullptr) {
        *id = record.id;
    }
    if (existing == files_.end()) {
        files_.push_back(std::move(record));
    } else {
        *existing = std::move(record);
    }
    rebuild_catalog();
    return true;
}

bool ModelLoader::add_file(const std::string& path, const std::string& prefix, FileId* id, bool force, FileScope scope) {
    ModelLoader candidate = *this;
    FileId added_id       = 0;
    if (!candidate.add_file_impl(path, prefix, &added_id, force, scope)) {
        return false;
    }
    *this = std::move(candidate);
    if (id != nullptr) {
        *id = added_id;
    }
    return true;
}

bool ModelLoader::del_file(FileId id) {
    auto it = std::find_if(files_.begin(), files_.end(), [id](const FileRecord& file) { return file.id == id; });
    if (it == files_.end()) {
        return false;
    }
    files_.erase(it);
    ++revision_;
    rebuild_catalog();
    return true;
}

bool ModelLoader::files_changed(bool& changed, bool include_isolated) const {
    changed = false;
    for (const auto& file : files_) {
        if (!include_isolated && file.scope == FileScope::Isolated)
            continue;
        for (const auto& stamp : file.dependencies) {
            std::error_code error;
            if (!std::filesystem::exists(std::filesystem::u8path(stamp.path), error) && !error) {
                // An updated index may no longer reference this dependency.
                changed = true;
                continue;
            }
            FileStamp current;
            if (!read_file_stamp(stamp.path, current)) {
                return false;
            }
            changed |= current.size != stamp.size || current.modified != stamp.modified;
        }
    }
    return true;
}

bool ModelLoader::refresh_files(bool include_isolated) {
    bool changed;
    if (!files_changed(changed, include_isolated)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    ModelLoader candidate = *this;
    for (const auto& file : files_) {
        if (!include_isolated && file.scope == FileScope::Isolated)
            continue;
        if (!candidate.add_file_impl(file.path, file.prefix, nullptr, false, file.scope)) {
            return false;
        }
    }
    *this = std::move(candidate);
    return true;
}

bool ModelLoader::validate_sources(const std::set<std::string>* tensor_names) const {
    std::set<FileId> required;
    if (tensor_names != nullptr) {
        for (const auto& name : *tensor_names) {
            auto it = tensor_storage_map.find(name);
            if (it != tensor_storage_map.end()) {
                required.insert(it->second.file_id);
            }
        }
    }
    for (const auto& file : files_) {
        if (tensor_names != nullptr && required.count(file.id) == 0) {
            continue;
        }
        if (!std::all_of(file.dependencies.begin(), file.dependencies.end(), file_unchanged)) {
            LOG_ERROR("model source changed; refresh it before execution: '%s'", file.path.c_str());
            return false;
        }
    }
    return true;
}

ModelLoader::FileVersions ModelLoader::file_versions(const std::vector<std::string>& prefixes) const {
    FileVersions versions;
    for (const auto& entry : tensor_storage_map) {
        if (prefixes.empty() || std::any_of(prefixes.begin(), prefixes.end(), [&](const std::string& prefix) {
                return starts_with(entry.first, prefix);
            })) {
            versions[entry.second.file_id] = entry.second.file_revision;
        }
    }
    return versions;
}

uint64_t ModelLoader::file_revision(FileId id) const {
    for (const auto& file : files_) {
        if (file.id == id)
            return file.revision;
    }
    return 0;
}

std::string ModelLoader::file_path(FileId id) const {
    for (const auto& file : files_) {
        if (file.id == id)
            return file.path;
    }
    return {};
}

ModelLoader ModelLoader::file_reader(FileId id, SDVersion version) const {
    ModelLoader reader;
    reader.file_paths_      = file_paths_;
    reader.n_threads_       = n_threads_;
    reader.version_         = version;
    reader.names_converted_ = true;
    for (const auto& file : files_) {
        if (file.id == id) {
            reader.files_.push_back(file);
            reader.files_.back().scope = FileScope::Catalog;
            break;
        }
    }
    reader.rebuild_catalog();
    return reader;
}

String2TensorStorage ModelLoader::file_tensors(FileId id, SDVersion version) const {
    return file_reader(id, version).tensor_storage_map;
}

bool ModelLoader::load_file_tensors(FileId id, SDVersion version, on_new_tensor_cb_t callback, const std::set<std::string>& names, bool use_mmap) const {
    if (file_revision(id) == 0)
        return false;
    auto reader = file_reader(id, version);
    return reader.load_tensors(callback, use_mmap, &names, false);
}

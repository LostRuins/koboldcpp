#include "model_manager.h"

#include <algorithm>

#include "core/util.h"

static bool same_tensor_source(const TensorStorage& a, const TensorStorage& b) {
    return a.file_id == b.file_id && a.file_revision == b.file_revision &&
           a.file_index == b.file_index && a.offset == b.offset && a.index_in_zip == b.index_in_zip &&
           a.storage_key == b.storage_key && a.type == b.type && a.expected_type == b.expected_type &&
           a.n_dims == b.n_dims && std::equal(a.ne, a.ne + SD_MAX_DIMS, b.ne) &&
           a.is_f8_e4m3 == b.is_f8_e4m3 && a.is_f8_e5m2 == b.is_f8_e5m2 &&
           a.is_f64 == b.is_f64 && a.is_i64 == b.is_i64 &&
           a.is_int8_tensorwise == b.is_int8_tensorwise && a.int8_convrot == b.int8_convrot &&
           a.int8_convrot_group_size == b.int8_convrot_group_size;
}

void ModelManager::invalidate_sources(const std::unordered_set<TensorState*>& states) {
    auto affected = states;
    for (const auto& block : params_storage_blocks_) {
        if (std::any_of(block->states.begin(), block->states.end(), [&](TensorState* state) { return states.count(state) != 0; })) {
            affected.insert(block->states.begin(), block->states.end());
        }
    }
    for (auto it = prefetch_blocks_.begin(); it != prefetch_blocks_.end();) {
        if (std::any_of(it->second->states.begin(), it->second->states.end(), [&](TensorState* state) { return affected.count(state) != 0; })) {
            free_prefetch_block(*it->second);
            it = prefetch_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        if (std::any_of((*it)->staged_tensors.begin(), (*it)->staged_tensors.end(), [&](const auto& entry) { return affected.count(entry.first) != 0; })) {
            ggml_backend_synchronize((*it)->compute_backend);
            free_compute_staging_block(**it);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end();) {
        if (std::any_of((*it)->states.begin(), (*it)->states.end(), [&](TensorState* state) { return affected.count(state) != 0; })) {
            free_params_storage_block(**it);
            it = params_storage_blocks_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto* state : affected) {
        state->metadata_validated = false;
        state->applied_lora_epoch = UINT64_MAX;
    }
}

bool ModelManager::set_loader(ModelLoader loader) {
    if (!workspace_reclaimers_.empty() || std::any_of(tensor_states_.begin(), tensor_states_.end(), [](const auto& state) {
            return state->pin_count != 0;
        })) {
        LOG_ERROR("cannot update model sources during execution");
        return false;
    }
    std::map<std::pair<ModelLoader::FileId, SDVersion>, String2TensorStorage> scoped;
    auto sources_for = [&](const TensorState& state) -> const String2TensorStorage& {
        if (state.source_file == 0)
            return loader.get_tensor_storage_map();
        auto key   = std::make_pair(state.source_file, state.source_version);
        auto found = scoped.find(key);
        if (found == scoped.end())
            found = scoped.emplace(key, loader.file_tensors(key.first, key.second)).first;
        return found->second;
    };
    bool lora_changed = false;
    for (const auto& spec : loras_) {
        lora_changed |= loader.file_revision(spec.file_id) != spec.file_revision;
    }
    std::unordered_set<TensorState*> changed;
    for (const auto& state : tensor_states_) {
        const auto& sources = sources_for(*state);
        auto source         = sources.find(state->name);
        const bool found    = source != sources.end();
        if (found != state->has_source || (found && !same_tensor_source(state->source, source->second)) ||
            (lora_changed && state->component != ModelComponent::LoRA && state->applied_lora_epoch != UINT64_MAX)) {
            changed.insert(state.get());
        }
    }
    invalidate_sources(changed);
    for (auto* state : changed) {
        const auto& sources = sources_for(*state);
        auto source         = sources.find(state->name);
        state->has_source   = source != sources.end();
        state->source       = state->has_source ? source->second : TensorStorage{};
    }
    if (lora_changed) {
        ++current_lora_epoch_;
        for (auto& spec : loras_)
            spec.file_revision = loader.file_revision(spec.file_id);
    }
    model_loader_ = std::move(loader);
    model_loader_.set_n_threads(n_threads_);
    return true;
}

bool ModelManager::add_file(const std::string& path, const std::string& prefix, ModelLoader::FileId* id, bool force) {
    ModelLoader candidate = model_loader_;
    ModelLoader::FileId added_id;
    if (!candidate.add_file(path, prefix, &added_id, force) || !set_loader(std::move(candidate))) {
        return false;
    }
    if (id != nullptr) {
        *id = added_id;
    }
    return true;
}

bool ModelManager::del_file(ModelLoader::FileId id) {
    ModelLoader candidate = model_loader_;
    return candidate.del_file(id) && set_loader(std::move(candidate));
}

bool ModelManager::refresh_files() {
    ModelLoader candidate = model_loader_;
    return candidate.refresh_files() && set_loader(std::move(candidate));
}

ModelLoader::FileVersions ModelManager::source_versions(const std::set<ModelComponent>& components, const ModelLoader& loader) const {
    ModelLoader::FileVersions versions;
    const auto& sources = loader.get_tensor_storage_map();
    for (const auto& state : tensor_states_) {
        if (components.count(state->component) == 0) {
            continue;
        }
        if (state->source_file != 0) {
            versions[state->source_file] = loader.file_revision(state->source_file);
            continue;
        }
        auto source = sources.find(state->name);
        if (source != sources.end()) {
            versions[source->second.file_id] = source->second.file_revision;
        }
    }
    return versions;
}

size_t ModelManager::registered_params_size(const std::set<ModelComponent>& components) const {
    size_t bytes = 0;
    std::unordered_set<const ggml_tensor*> seen;
    for (const auto& state : tensor_states_) {
        if (components.count(state->component) != 0 && state->tensor != nullptr && seen.insert(state->tensor).second) {
            bytes += ggml_nbytes(state->tensor);
        }
    }
    return bytes;
}

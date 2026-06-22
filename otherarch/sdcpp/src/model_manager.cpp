#include "model_manager.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "model/adapter/lora.hpp"

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
    GGML_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
    size_t align = (alignment - ((reinterpret_cast<uintptr_t>(buffer) + offset) % alignment)) % alignment;
    return offset + align;
}

static bool lora_specs_equal(const std::vector<ModelManager::LoraSpec>& lhs,
                             const std::vector<ModelManager::LoraSpec>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].path != rhs[i].path ||
            lhs[i].multiplier != rhs[i].multiplier ||
            lhs[i].is_high_noise != rhs[i].is_high_noise ||
            lhs[i].tensor_name_prefix_filter != rhs[i].tensor_name_prefix_filter ||
            lhs[i].required != rhs[i].required) {
            return false;
        }
    }
    return true;
}

static std::string lora_id(const ModelManager::LoraSpec& lora) {
    return lora.is_high_noise ? "|high_noise|" + lora.path : lora.path;
}

static bool backend_supports_host_buffer(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (sd_backend_is_cpu(backend)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.buffer_from_host_ptr;
}

ModelManager::~ModelManager() {
    release_all();
}

void ModelManager::set_common_ignore_tensors(std::set<std::string> ignore_tensors) {
    common_ignore_tensors_ = std::move(ignore_tensors);
}

void ModelManager::set_loras(std::vector<LoraSpec> loras, SDVersion version) {
    if (loras.empty() && loras_.empty()) {
        lora_version_ = version;
        return;
    }
    if (lora_version_ == version && lora_specs_equal(loras_, loras)) {
        return;
    }

    loras_        = std::move(loras);
    lora_version_ = version;
    current_lora_epoch_++;
    reset_lora_applied_params();
}

std::set<std::string> ModelManager::tensor_names() const {
    std::set<std::string> names;
    for (const auto& state : tensor_states_) {
        if (state != nullptr) {
            names.insert(state->name);
        }
    }
    return names;
}

size_t estimate_tensors_size(const std::map<std::string, ggml_tensor*>& tensors) {
    size_t size = 0;
    std::unordered_set<ggml_tensor*> seen;
    for (const auto& pair : tensors) {
        ggml_tensor* tensor = pair.second;
        if (tensor == nullptr || seen.find(tensor) != seen.end()) {
            continue;
        }
        seen.insert(tensor);
        size += ggml_nbytes(tensor);
    }
    return size;
}

bool ModelManager::register_param_tensors(const std::string& desc,
                                          std::map<std::string, ggml_tensor*> tensors,
                                          ResidencyMode residency_mode,
                                          ggml_backend_t compute_backend,
                                          ggml_backend_t params_backend,
                                          size_t* registered_tensor_size) {
    if (desc.empty()) {
        LOG_ERROR("model manager tensor desc is empty");
        return false;
    }
    if (registered_tensor_size != nullptr) {
        *registered_tensor_size += estimate_tensors_size(tensors);
    }

    std::vector<std::unique_ptr<TensorState>> new_states;
    new_states.reserve(tensors.size());

    for (const auto& pair : tensors) {
        const std::string& name = pair.first;
        ggml_tensor* tensor     = pair.second;
        if (tensor == nullptr) {
            continue;
        }
        if (tensor_states_by_name_.find(name) != tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor name '%s' is already registered", name.c_str());
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        auto state             = std::make_unique<TensorState>();
        state->name            = name;
        state->tensor          = tensor;
        state->desc            = desc;
        state->residency_mode  = residency_mode;
        state->compute_backend = compute_backend;
        state->params_backend  = params_backend;
        new_states.push_back(std::move(state));
    }

    for (auto& state : new_states) {
        TensorState* registered_state                  = state.get();
        tensor_states_by_name_[registered_state->name] = registered_state;
        tensor_states_.push_back(std::move(state));
    }
    return true;
}

bool ModelManager::load_all_params_eagerly() {
    std::vector<TensorState*> all_states;
    all_states.reserve(tensor_states_.size());
    for (const auto& s : tensor_states_) {
        if (s != nullptr) {
            all_states.push_back(s.get());
        }
    }
    return load_tensors_to_params_backend(all_states);
}

bool ModelManager::validate_registered_tensors() {
    bool ok = true;
    for (const auto& state : tensor_states_) {
        if (state == nullptr) {
            ok = false;
            continue;
        }
        bool state_ok = validate_tensor(*state);
        if (state_ok) {
            state->metadata_validated = true;
        }
        ok = state_ok && ok;
    }
    return ok;
}

bool ModelManager::load_tensors_to_params_backend(const std::vector<TensorState*>& states) {
    std::vector<TensorState*> need_load;
    need_load.reserve(states.size());
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (!state->metadata_validated) {
            if (!validate_tensor(*state)) {
                return false;
            }
            state->metadata_validated = true;
        }
        if (!state->loaded_to_params_backend) {
            need_load.push_back(state);
        }
    }
    if (need_load.empty()) {
        return true;
    }

    std::vector<ParamsStorageBlock*> created_storage_blocks;
    if (!mmap_params(need_load, created_storage_blocks)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }

    std::vector<TensorState*> need_alloc;
    need_alloc.reserve(need_load.size());
    for (TensorState* state : need_load) {
        if (state->tensor != nullptr && state->tensor->data == nullptr && state->tensor->view_src == nullptr) {
            need_alloc.push_back(state);
        }
    }

    if (!alloc_params_buffers(need_alloc, created_storage_blocks) ||
        !load_tensors(need_load)) {
        for (ParamsStorageBlock* block : created_storage_blocks) {
            if (block != nullptr) {
                free_params_storage_block(*block);
                erase_params_storage_block(block);
            }
        }
        return false;
    }
    for (ParamsStorageBlock* block : created_storage_blocks) {
        if (block != nullptr && block->buffer != nullptr) {
            LOG_DEBUG("model manager prepared params backend buffer (%6.2f MB, %zu tensors, %s)",
                      ggml_backend_buffer_get_size(block->buffer) / (1024.f * 1024.f),
                      block->states.size(),
                      ggml_backend_buffer_is_host(block->buffer) ? "RAM" : "VRAM");
        }
    }

    return true;
}

bool ModelManager::stage_tensors_to_compute_backend(const std::vector<TensorState*>& states) {
    std::map<ggml_backend_t, std::vector<TensorState*>> states_by_compute_backend;
    for (TensorState* state : states) {
        if (state == nullptr || should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->params_backend == nullptr) {
            LOG_ERROR("model manager params backend is null for tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->compute_backend == state->params_backend || state->staged_to_compute_backend) {
            continue;
        }
        if (!state->loaded_to_params_backend || state->tensor == nullptr || state->tensor->data == nullptr) {
            LOG_ERROR("model manager tensor '%s' is not loaded to params backend", state->name.c_str());
            return false;
        }
        states_by_compute_backend[state->compute_backend].push_back(state);
    }

    for (const auto& pair : states_by_compute_backend) {
        ggml_backend_t compute_backend          = pair.first;
        const std::vector<TensorState*>& states = pair.second;
        if (states.empty()) {
            continue;
        }

        int64_t t0 = ggml_time_ms();

        ggml_init_params init_params;
        init_params.mem_size   = std::max<size_t>(1, states.size()) * ggml_tensor_overhead();
        init_params.mem_buffer = nullptr;
        init_params.no_alloc   = true;

        ggml_context* staging_ctx = ggml_init(init_params);
        GGML_ASSERT(staging_ctx != nullptr);

        std::vector<std::pair<TensorState*, ggml_tensor*>> staged_tensors;
        staged_tensors.reserve(states.size());
        for (TensorState* state : states) {
            ggml_tensor* staging_tensor = ggml_dup_tensor(staging_ctx, state->tensor);
            ggml_set_name(staging_tensor, state->tensor->name);
            staged_tensors.push_back({state, staging_tensor});
        }

        ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors(staging_ctx, compute_backend);
        if (compute_buffer == nullptr) {
            LOG_ERROR("model manager alloc compute params backend buffer failed, num_tensors = %zu",
                      staged_tensors.size());
            ggml_free(staging_ctx);
            return false;
        }
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& staged_tensor : staged_tensors) {
            TensorState* state          = staged_tensor.first;
            ggml_tensor* managed_tensor = state->tensor;
            ggml_tensor* staging_tensor = staged_tensor.second;
            ggml_backend_tensor_copy(managed_tensor, staging_tensor);
            std::swap(managed_tensor->buffer, staging_tensor->buffer);
            std::swap(managed_tensor->data, staging_tensor->data);
            std::swap(managed_tensor->extra, staging_tensor->extra);
        }
        ggml_backend_synchronize(compute_backend);

        auto block             = std::make_unique<ComputeStagingBlock>();
        block->compute_backend = compute_backend;
        block->buffer          = compute_buffer;
        block->staging_ctx     = staging_ctx;
        block->staged_tensors  = std::move(staged_tensors);
        for (auto& staged_tensor : block->staged_tensors) {
            TensorState* state               = staged_tensor.first;
            state->staged_to_compute_backend = true;
        }
        compute_staging_blocks_.push_back(std::move(block));

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("model manager staged compute params (%6.2f MB, %zu tensors) to %s, taking %.2fs",
                  ggml_backend_buffer_get_size(compute_buffer) / (1024.f * 1024.f),
                  states.size(),
                  ggml_backend_name(compute_backend),
                  (t1 - t0) * 1.0f / 1000);
    }

    return true;
}

bool ModelManager::apply_loras_to_params(const std::vector<TensorState*>& states) {
    if (loras_.empty()) {
        return true;
    }

    struct LoraApplyGroup {
        std::map<std::string, ggml_tensor*> model_tensors;
        std::vector<TensorState*> states;
    };

    std::map<ggml_backend_t, LoraApplyGroup> groups;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr ||
            should_ignore(*state) || is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->applied_lora_epoch == current_lora_epoch_) {
            continue;
        }
        if (state->compute_backend == nullptr) {
            LOG_ERROR("model manager compute backend is null for lora target tensor '%s'", state->name.c_str());
            return false;
        }
        if (state->tensor->data == nullptr) {
            LOG_ERROR("model manager lora target tensor '%s' is not prepared", state->name.c_str());
            return false;
        }
        LoraApplyGroup& group            = groups[state->compute_backend];
        group.model_tensors[state->name] = state->tensor;
        group.states.push_back(state);
    }

    if (groups.empty()) {
        return true;
    }

    std::set<std::string> all_tensor_names = tensor_names();
    for (auto& group_pair : groups) {
        ggml_backend_t compute_backend = group_pair.first;
        LoraApplyGroup& group          = group_pair.second;
        for (const LoraSpec& lora_spec : loras_) {
            if (group.model_tensors.empty()) {
                continue;
            }

            std::string id = lora_id(lora_spec);
            auto lora      = std::make_shared<LoraModel>(id,
                                                    compute_backend,
                                                    compute_backend,
                                                    lora_spec.path,
                                                    lora_spec.is_high_noise ? "model.high_noise_" : "",
                                                    lora_version_);

            LoraModel::filter_t lora_tensor_filter = nullptr;
            if (!lora_spec.tensor_name_prefix_filter.empty()) {
                lora_tensor_filter = [&](const std::string& tensor_name) {
                    return starts_with(tensor_name, lora_spec.tensor_name_prefix_filter);
                };
            }
            if (!lora->load_from_file(n_threads_, lora_tensor_filter)) {
                LOG_WARN("load lora tensors from %s failed", lora_spec.path.c_str());
                if (lora_spec.required) {
                    return false;
                }
                continue;
            }
            if (lora->lora_tensors.empty()) {
                if (lora_spec.required) {
                    LOG_ERROR("required lora has no tensors: %s", lora_spec.path.c_str());
                    return false;
                }
                continue;
            }
            lora->multiplier = lora_spec.multiplier;
            lora->apply(group.model_tensors, all_tensor_names, lora_version_, n_threads_, false);
            lora->release_loaded_tensors();
        }

        for (TensorState* state : group.states) {
            if (state != nullptr) {
                state->applied_lora_epoch = current_lora_epoch_;
            }
        }
    }
    return true;
}

void ModelManager::reset_lora_applied_params() {
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
    for (auto& state : tensor_states_) {
        state->applied_lora_epoch = UINT64_MAX;
    }
}

bool ModelManager::should_ignore(const TensorState& state) const {
    for (const auto& ignore_prefix : common_ignore_tensors_) {
        if (starts_with(state.name, ignore_prefix)) {
            return true;
        }
    }
    return false;
}

bool ModelManager::is_optional_missing_tensor(const std::string& name) const {
    return name.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos ||
           name.find("alphas_cumprod") != std::string::npos;
}

bool ModelManager::validate_tensor(const TensorState& state) const {
    if (state.tensor == nullptr || should_ignore(state) || is_optional_missing_tensor(state.name)) {
        return true;
    }

    const auto& tensor_storage_map = model_loader_.get_tensor_storage_map();
    auto ts_it                     = tensor_storage_map.find(state.name);
    if (ts_it == tensor_storage_map.end()) {
        LOG_ERROR("%s tensor '%s' not in model metadata", state.desc.c_str(), state.name.c_str());
        return false;
    }

    const TensorStorage& tensor_storage = ts_it->second;
    if (state.tensor->ne[0] != tensor_storage.ne[0] ||
        state.tensor->ne[1] != tensor_storage.ne[1] ||
        state.tensor->ne[2] != tensor_storage.ne[2] ||
        state.tensor->ne[3] != tensor_storage.ne[3]) {
        LOG_ERROR(
            "%s tensor '%s' has wrong shape in model metadata: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
            state.desc.c_str(),
            state.name.c_str(),
            (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
            (int)state.tensor->ne[0], (int)state.tensor->ne[1], (int)state.tensor->ne[2], (int)state.tensor->ne[3]);
        return false;
    }
    return true;
}

bool ModelManager::mmap_params(const std::vector<TensorState*>& states,
                               std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::string, ggml_tensor*> mmap_candidates;
    std::map<std::string, TensorState*> mmap_states;
    for (TensorState* state : states) {
        if (state == nullptr || !can_mmap_storage(*state) || state->tensor == nullptr ||
            state->tensor->data != nullptr || state->tensor->view_src != nullptr) {
            continue;
        }
        mmap_candidates[state->name] = state->tensor;
        mmap_states[state->name]     = state;
    }
    if (mmap_candidates.empty()) {
        return true;
    }

    auto mmap_store = model_loader_.mmap_tensors(mmap_candidates, {}, writable_mmap_);
    if (mmap_store.empty()) {
        return true;
    }

    auto block                = std::make_unique<ParamsStorageBlock>();
    block->mmap_tensor_stores = std::move(mmap_store);
    ParamsStorageBlock* raw   = block.get();
    for (const auto& pair : mmap_states) {
        TensorState* state = pair.second;
        if (state != nullptr && state->tensor != nullptr && state->tensor->data != nullptr) {
            block->states.push_back(state);
        }
    }

    if (!block->states.empty()) {
        params_storage_blocks_.push_back(std::move(block));
        created_storage_blocks.push_back(raw);
    }
    return true;
}

bool ModelManager::can_mmap_storage(const TensorState& state) const {
    if (!enable_mmap_ || state.residency_mode != ResidencyMode::ParamBackend) {
        return false;
    }
    if (state.compute_backend == nullptr || state.params_backend == nullptr) {
        return false;
    }
    return sd_backend_is_cpu(state.compute_backend) ||
           sd_backend_is_cpu(state.params_backend) ||
           backend_supports_host_buffer(state.compute_backend);
}

bool ModelManager::alloc_params_buffers(const std::vector<TensorState*>& states,
                                        std::vector<ParamsStorageBlock*>& created_storage_blocks) {
    std::map<std::pair<ggml_backend_buffer_type_t, int>, std::vector<TensorState*>> states_by_buffer_type;
    for (TensorState* state : states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        ggml_backend_buffer_type_t params_buft = params_buffer_type_for(*state);
        if (params_buft == nullptr) {
            return false;
        }
        states_by_buffer_type[{params_buft, static_cast<int>(state->residency_mode)}].push_back(state);
    }

    for (const auto& pair : states_by_buffer_type) {
        ggml_backend_buffer_type_t params_buft  = pair.first.first;
        const std::vector<TensorState*>& states = pair.second;
        size_t alignment                        = ggml_backend_buft_get_alignment(params_buft);
        size_t max_size                         = ggml_backend_buft_get_max_size(params_buft);

        auto alloc_chunk = [&](const std::vector<TensorState*>& chunk, size_t chunk_size) -> bool {
            if (chunk.empty() || chunk_size == 0) {
                return true;
            }

            ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(params_buft, chunk_size);
            if (buffer == nullptr) {
                LOG_ERROR("model manager alloc params backend buffer failed, size = %.2fMB",
                          chunk_size / (1024.0 * 1024.0));
                return false;
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            std::vector<ggml_tensor*> initialized_tensors;
            void* base    = ggml_backend_buffer_get_base(buffer);
            size_t offset = aligned_offset(base, 0, ggml_backend_buffer_get_alignment(buffer));
            for (TensorState* state : chunk) {
                ggml_tensor* tensor     = state->tensor;
                size_t tensor_size      = GGML_PAD(ggml_backend_buffer_get_alloc_size(buffer, tensor),
                                                   ggml_backend_buffer_get_alignment(buffer));
                enum ggml_status status = ggml_backend_tensor_alloc(buffer, tensor, static_cast<char*>(base) + offset);
                if (status != GGML_STATUS_SUCCESS) {
                    LOG_ERROR("model manager failed to initialize params tensor '%s'", ggml_get_name(tensor));
                    for (ggml_tensor* initialized : initialized_tensors) {
                        initialized->buffer = nullptr;
                        initialized->data   = nullptr;
                        initialized->extra  = nullptr;
                    }
                    LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                              ggml_backend_buffer_get_size(buffer) / (1024.f * 1024.f),
                              initialized_tensors.size(),
                              ggml_backend_buffer_is_host(buffer) ? "RAM" : "VRAM");
                    ggml_backend_buffer_free(buffer);
                    return false;
                }
                initialized_tensors.push_back(tensor);
                offset += tensor_size;
            }

            auto block              = std::make_unique<ParamsStorageBlock>();
            block->buffer           = buffer;
            block->states           = chunk;
            ParamsStorageBlock* raw = block.get();
            params_storage_blocks_.push_back(std::move(block));
            created_storage_blocks.push_back(raw);

            return true;
        };

        std::vector<TensorState*> chunk;
        size_t chunk_size = 0;
        for (TensorState* state : states) {
            ggml_tensor* tensor = state->tensor;
            size_t tensor_size  = GGML_PAD(ggml_backend_buft_get_alloc_size(params_buft, tensor), alignment);
            // Some backends, e.g. Vulkan, report a preferred chunk size here rather than a
            // hard per-tensor allocation limit. Oversized tensors are allocated alone.
            if (!chunk.empty() && max_size > 0 && chunk_size + tensor_size > max_size) {
                if (!alloc_chunk(chunk, chunk_size)) {
                    return false;
                }
                chunk.clear();
                chunk_size = 0;
            }
            chunk.push_back(state);
            chunk_size += tensor_size;
        }

        if (!alloc_chunk(chunk, chunk_size)) {
            return false;
        }
    }

    return true;
}

bool ModelManager::load_tensors(const std::vector<TensorState*>& states) {
    std::map<std::string, TensorState*> states_by_name;
    std::set<std::string> target_tensor_names;
    for (TensorState* state : states) {
        if (state == nullptr) {
            continue;
        }
        states_by_name[state->name] = state;
        target_tensor_names.insert(state->name);
    }
    if (states_by_name.empty()) {
        return true;
    }

    std::set<std::string> loaded_names;
    std::mutex loaded_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        *dst_tensor             = nullptr;

        auto state_it = states_by_name.find(name);
        if (state_it == states_by_name.end()) {
            return true;
        }

        TensorState* state = state_it->second;
        if (state == nullptr || state->tensor == nullptr) {
            LOG_ERROR("model manager tensor '%s' is null", name.c_str());
            return false;
        }

        if (state->tensor->ne[0] != tensor_storage.ne[0] ||
            state->tensor->ne[1] != tensor_storage.ne[1] ||
            state->tensor->ne[2] != tensor_storage.ne[2] ||
            state->tensor->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "model manager tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)state->tensor->ne[0], (int)state->tensor->ne[1], (int)state->tensor->ne[2], (int)state->tensor->ne[3]);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(loaded_names_mutex);
            loaded_names.insert(name);
        }
        *dst_tensor = state->tensor;
        return true;
    };

    if (!model_loader_.load_tensors(on_new_tensor_cb, enable_mmap_, &target_tensor_names)) {
        LOG_ERROR("model manager load tensors failed");
        return false;
    }

    bool missing = false;
    for (const auto& pair : states_by_name) {
        const std::string& name = pair.first;
        if (loaded_names.find(name) == loaded_names.end()) {
            LOG_ERROR("model manager tensor '%s' was not loaded", name.c_str());
            missing = true;
        }
    }
    if (missing) {
        return false;
    }

    for (const auto& pair : states_by_name) {
        pair.second->loaded_to_params_backend = true;
    }
    return true;
}

ggml_backend_buffer_type_t ModelManager::params_buffer_type_for(const TensorState& state) const {
    if (state.params_backend == nullptr) {
        LOG_ERROR("model manager params backend is null for tensor '%s'", state.name.c_str());
        return nullptr;
    }
    ggml_backend_buffer_type_t params_buft = nullptr;
    if (state.compute_backend != nullptr && state.params_backend != state.compute_backend) {
        ggml_backend_dev_t compute_dev = ggml_backend_get_device(state.compute_backend);
        if (compute_dev != nullptr) {
            params_buft = ggml_backend_dev_host_buffer_type(compute_dev);
        }
    }
    if (params_buft == nullptr) {
        params_buft = ggml_backend_get_default_buffer_type(state.params_backend);
    }
    return params_buft;
}

void ModelManager::free_compute_staging_block(ComputeStagingBlock& block) {
    for (auto& staged_tensor : block.staged_tensors) {
        TensorState* state          = staged_tensor.first;
        ggml_tensor* staging_tensor = staged_tensor.second;
        if (state == nullptr || state->tensor == nullptr || staging_tensor == nullptr) {
            continue;
        }
        ggml_tensor* managed_tensor = state->tensor;
        managed_tensor->buffer      = staging_tensor->buffer;
        managed_tensor->data        = staging_tensor->data;
        managed_tensor->extra       = staging_tensor->extra;
        staging_tensor->buffer      = nullptr;
        staging_tensor->data        = nullptr;
        staging_tensor->extra       = nullptr;

        state->staged_to_compute_backend = false;
        state->applied_lora_epoch        = UINT64_MAX;
    }

    if (block.buffer != nullptr) {
        LOG_DEBUG("model manager releasing compute params (%6.2f MB, %zu tensors) from %s",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.staged_tensors.size(),
                  block.compute_backend != nullptr ? ggml_backend_name(block.compute_backend) : "unknown");
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    if (block.staging_ctx != nullptr) {
        ggml_free(block.staging_ctx);
        block.staging_ctx = nullptr;
    }
    block.staged_tensors.clear();
}

void ModelManager::release_compute_staging_blocks(bool force,
                                                  const std::unordered_set<TensorState*>* target_states) {
    for (auto it = compute_staging_blocks_.begin(); it != compute_staging_blocks_.end();) {
        ComputeStagingBlock* block = it->get();
        bool can_release           = force;
        if (!can_release) {
            can_release = std::all_of(block->staged_tensors.begin(),
                                      block->staged_tensors.end(),
                                      [target_states](const std::pair<TensorState*, ggml_tensor*>& pair) {
                                          TensorState* state = pair.first;
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->active_prepare_count == 0;
                                      });
        }

        if (can_release) {
            free_compute_staging_block(*block);
            it = compute_staging_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void ModelManager::free_params_storage_block(ParamsStorageBlock& block) {
    if (block.buffer != nullptr) {
        LOG_DEBUG("model manager releasing params backend buffer (%6.2f MB, %zu tensors, %s)",
                  ggml_backend_buffer_get_size(block.buffer) / (1024.f * 1024.f),
                  block.states.size(),
                  ggml_backend_buffer_is_host(block.buffer) ? "RAM" : "VRAM");
        ggml_backend_buffer_free(block.buffer);
        block.buffer = nullptr;
    }
    block.mmap_tensor_stores.clear();

    for (TensorState* state : block.states) {
        if (state == nullptr || state->tensor == nullptr) {
            continue;
        }
        state->tensor->buffer = nullptr;
        state->tensor->data   = nullptr;
        state->tensor->extra  = nullptr;

        state->loaded_to_params_backend = false;
        state->applied_lora_epoch       = UINT64_MAX;
    }
    block.states.clear();
}

void ModelManager::release_params_storage_blocks(bool force,
                                                 const std::unordered_set<TensorState*>* target_states) {
    for (auto it = params_storage_blocks_.begin(); it != params_storage_blocks_.end();) {
        ParamsStorageBlock* block = it->get();
        bool can_release          = force;
        if (!can_release) {
            can_release = std::all_of(block->states.begin(),
                                      block->states.end(),
                                      [target_states](TensorState* state) {
                                          if (state == nullptr) {
                                              return true;
                                          }
                                          if (target_states != nullptr &&
                                              target_states->find(state) == target_states->end()) {
                                              return false;
                                          }
                                          return state->active_prepare_count == 0 &&
                                                 !state->staged_to_compute_backend &&
                                                 state->residency_mode == ResidencyMode::Disk;
                                      });
        }

        if (can_release) {
            free_params_storage_block(*block);
            it = params_storage_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void ModelManager::erase_params_storage_block(ParamsStorageBlock* block) {
    auto it = std::find_if(params_storage_blocks_.begin(),
                           params_storage_blocks_.end(),
                           [block](const std::unique_ptr<ParamsStorageBlock>& item) {
                               return item.get() == block;
                           });
    if (it != params_storage_blocks_.end()) {
        params_storage_blocks_.erase(it);
    }
}

void ModelManager::release_all() {
    for (auto& state : tensor_states_) {
        state->active_prepare_count = 0;
        state->applied_lora_epoch   = UINT64_MAX;
    }
    release_compute_staging_blocks(true);
    release_params_storage_blocks(true);
}

bool ModelManager::resolve_required_tensor_states(const std::vector<ggml_tensor*>& tensors,
                                                  std::vector<TensorState*>& required_states) const {
    required_states.clear();
    std::unordered_set<TensorState*> seen;
    for (ggml_tensor* tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        const char* raw_name = ggml_get_name(tensor);
        if (raw_name == nullptr || raw_name[0] == '\0') {
            LOG_ERROR("model manager unnamed tensor is not registered");
            return false;
        }
        auto state_it = tensor_states_by_name_.find(raw_name);
        if (state_it == tensor_states_by_name_.end()) {
            LOG_ERROR("model manager tensor '%s' is not registered", raw_name);
            return false;
        }
        TensorState* state = state_it->second;
        if (state == nullptr) {
            LOG_ERROR("model manager tensor '%s' has no tensor state", raw_name);
            return false;
        }
        if (seen.insert(state).second) {
            required_states.push_back(state);
        }
    }
    return true;
}

bool ModelManager::prepare_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return true;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    if (!load_tensors_to_params_backend(required_states)) {
        return false;
    }

    if (!stage_tensors_to_compute_backend(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    if (!apply_loras_to_params(required_states)) {
        release_compute_staging_blocks(false);
        release_params_storage_blocks(false);
        return false;
    }

    for (TensorState* state : required_states) {
        if (state == nullptr) {
            continue;
        }
        state->active_prepare_count++;
    }
    return true;
}

void ModelManager::finish_compute_backend_usage(const std::vector<TensorState*>& states) {
    if (states.empty()) {
        return;
    }

    std::unordered_set<TensorState*> target_states;
    for (TensorState* state : states) {
        if (state == nullptr || !target_states.insert(state).second) {
            continue;
        }
        if (state->active_prepare_count > 0) {
            state->active_prepare_count--;
        }
    }
    release_compute_staging_blocks(false, &target_states);
}

void ModelManager::release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    finish_compute_backend_usage(required_states);
}

void ModelManager::release_params_backend_params(const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return;
    }
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return;
    }
    if (required_states.empty()) {
        return;
    }
    std::unordered_set<TensorState*> target_states(required_states.begin(), required_states.end());
    release_params_storage_blocks(false, &target_states);
}

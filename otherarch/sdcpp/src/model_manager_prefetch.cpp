#include "model_manager.h"

#include <algorithm>
#include <utility>

#include "core/ggml_extend_backend.h"
#include "core/util.h"

ggml_backend_t ModelManager::prefetch_backend_for(ggml_backend_t compute_backend) {
    auto existing = prefetch_backends_.find(compute_backend);
    if (existing != prefetch_backends_.end()) {
        return existing->second;
    }
    if (compute_backend == nullptr) {
        return nullptr;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(compute_backend);
    if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    ggml_backend_t transfer_backend = ggml_backend_dev_init(device, nullptr);
    if (transfer_backend == nullptr) {
        LOG_WARN("model manager failed to create a prefetch backend for %s",
                 ggml_backend_name(compute_backend));
    }
    prefetch_backends_[compute_backend] = transfer_backend;
    return transfer_backend;
}

void ModelManager::synchronize_prefetch_block(PrefetchBlock& block) {
    if (block.event != nullptr) {
        ggml_backend_event_synchronize(block.event);
        ggml_backend_event_free(block.event);
        block.event = nullptr;
    } else if (block.transfer_backend != nullptr) {
        ggml_backend_synchronize(block.transfer_backend);
    }
    block.transfer_backend = nullptr;
}

void ModelManager::free_prefetch_block(PrefetchBlock& block) {
    synchronize_prefetch_block(block);
    for (auto& staging_block : block.staging_blocks) {
        if (staging_block == nullptr) {
            continue;
        }
        staging_block->staged_tensors.clear();
        if (staging_block->buffer != nullptr) {
            ggml_backend_buffer_free(staging_block->buffer);
            staging_block->buffer = nullptr;
        }
        if (staging_block->staging_ctx != nullptr) {
            ggml_free(staging_block->staging_ctx);
            staging_block->staging_ctx = nullptr;
        }
    }
    block.staging_blocks.clear();
}

bool ModelManager::populate_prefetch_block(PrefetchBlock& block) {
    if (block.states.empty() || block.compute_backend == nullptr) {
        return false;
    }

    block.transfer_backend = prefetch_backend_for(block.compute_backend);
    if (block.transfer_backend == nullptr) {
        return false;
    }

    for (TensorState* state : block.states) {
        if (state == nullptr || state->tensor == nullptr ||
            state->tensor->buffer == nullptr || state->tensor->data == nullptr ||
            state->params_backend == nullptr || state->staged_to_compute_backend ||
            state->pin_count > 0) {
            return false;
        }
    }

    ggml_backend_buffer_type_t buffer_type =
        ggml_backend_get_default_buffer_type(block.compute_backend);
    if (buffer_type == nullptr) {
        return false;
    }
    const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
    size_t backend_limit   = ggml_backend_buft_get_max_size(buffer_type);
    if (!ggml_backend_buft_is_host(buffer_type) &&
        (backend_limit == 0 || backend_limit > MAX_RESIDENCY_BLOCK_BYTES)) {
        backend_limit = MAX_RESIDENCY_BLOCK_BYTES;
    }

    auto enqueue_chunk = [&](const std::vector<TensorState*>& chunk) -> bool {
        if (chunk.empty()) {
            return true;
        }
        ggml_init_params init_params;
        init_params.mem_size      = chunk.size() * ggml_tensor_overhead();
        init_params.mem_buffer    = nullptr;
        init_params.no_alloc      = true;
        ggml_context* staging_ctx = ggml_init(init_params);
        if (staging_ctx == nullptr) {
            return false;
        }

        auto staging_block             = std::make_unique<ComputeStagingBlock>();
        staging_block->compute_backend = block.compute_backend;
        staging_block->staging_ctx     = staging_ctx;
        staging_block->staged_tensors.reserve(chunk.size());
        for (TensorState* state : chunk) {
            ggml_tensor* staging_tensor = ggml_dup_tensor(staging_ctx, state->tensor);
            ggml_set_name(staging_tensor, state->tensor->name);
            if (ggml_backend_buffer_is_host(state->tensor->buffer) &&
                (!ggml_is_contiguous(state->tensor) || !ggml_is_contiguous(staging_tensor) ||
                 ggml_nbytes(state->tensor) != ggml_nbytes(staging_tensor))) {
                ggml_free(staging_ctx);
                staging_block->staging_ctx = nullptr;
                return false;
            }
            staging_block->staged_tensors.push_back({state, staging_tensor});
        }
        staging_block->buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(staging_ctx, buffer_type);
        if (staging_block->buffer == nullptr) {
            ggml_free(staging_ctx);
            staging_block->staging_ctx = nullptr;
            return false;
        }
        ggml_backend_buffer_set_usage(staging_block->buffer,
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const auto& pair : staging_block->staged_tensors) {
            TensorState* state          = pair.first;
            ggml_tensor* staging_tensor = pair.second;
            const bool host_source      = ggml_backend_buffer_is_host(state->tensor->buffer);
            if (host_source) {
                ggml_backend_tensor_set_async(block.transfer_backend,
                                              staging_tensor,
                                              state->tensor->data,
                                              0,
                                              ggml_nbytes(state->tensor));
            } else {
                ggml_backend_tensor_copy_async(state->params_backend,
                                               block.transfer_backend,
                                               state->tensor,
                                               staging_tensor);
            }
        }
        block.staging_blocks.push_back(std::move(staging_block));
        return true;
    };

    std::vector<TensorState*> chunk;
    size_t chunk_size = 0;
    for (TensorState* state : block.states) {
        const size_t tensor_size = GGML_PAD(
            ggml_backend_buft_get_alloc_size(buffer_type, state->tensor), alignment);
        if (!chunk.empty() && backend_limit > 0 &&
            tensor_size > backend_limit - std::min(chunk_size, backend_limit)) {
            if (!enqueue_chunk(chunk)) {
                return false;
            }
            chunk.clear();
            chunk_size = 0;
        }
        chunk.push_back(state);
        chunk_size = tensor_size > SIZE_MAX - chunk_size ? SIZE_MAX : chunk_size + tensor_size;
    }
    if (!enqueue_chunk(chunk)) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(block.transfer_backend);
    block.event               = ggml_backend_event_new(device);
    if (block.event != nullptr) {
        ggml_backend_event_record(block.event, block.transfer_backend);
    }

    size_t total_size = 0;
    for (const auto& staging_block : block.staging_blocks) {
        if (staging_block != nullptr && staging_block->buffer != nullptr) {
            const size_t buffer_size = ggml_backend_buffer_get_size(staging_block->buffer);
            total_size               = buffer_size > SIZE_MAX - total_size ? SIZE_MAX : total_size + buffer_size;
        }
    }
    LOG_DEBUG("model manager queued segment prefetch (%6.2f MB, %zu tensors) to %s",
              total_size / (1024.f * 1024.f),
              block.states.size(),
              ggml_backend_name(block.compute_backend));
    return true;
}

WeightPrefetchResult ModelManager::prefetch_params(
    uintptr_t owner_id,
    const std::vector<ggml_tensor*>& tensors) {
    if (tensors.empty()) {
        return WeightPrefetchResult::AlreadyResident;
    }

    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return WeightPrefetchResult::Failed;
    }

    std::vector<TensorState*> states;
    states.reserve(required_states.size());
    ggml_backend_t compute_backend = nullptr;
    bool needs_synchronous_load    = false;
    for (TensorState* state : required_states) {
        if (state == nullptr || should_ignore(*state) ||
            is_optional_missing_tensor(state->name)) {
            continue;
        }
        if (state->compute_backend == state->params_backend) {
            needs_synchronous_load = needs_synchronous_load ||
                                     !state->loaded_to_params_backend;
            continue;
        }
        if (state->staged_to_compute_backend || state->pin_count > 0) {
            continue;
        }
        // Split buffers cannot use the primary device's asynchronous upload path.
        if (split_buffer_type_for(*state) != nullptr) {
            return WeightPrefetchResult::Unsupported;
        }
        if (compute_backend == nullptr) {
            compute_backend = state->compute_backend;
        } else if (compute_backend != state->compute_backend) {
            return WeightPrefetchResult::Failed;
        }
        states.push_back(state);
    }
    if (states.empty()) {
        return needs_synchronous_load ? WeightPrefetchResult::Unsupported
                                      : WeightPrefetchResult::AlreadyResident;
    }
    if (compute_backend == nullptr || sd_backend_is_cpu(compute_backend)) {
        return WeightPrefetchResult::Unsupported;
    }
    ggml_backend_dev_t compute_device = ggml_backend_get_device(compute_backend);
    ggml_backend_dev_props compute_props{};
    if (compute_device == nullptr) {
        return WeightPrefetchResult::Unsupported;
    }
    ggml_backend_dev_get_props(compute_device, &compute_props);
    if (!compute_props.caps.async) {
        return WeightPrefetchResult::Unsupported;
    }
    clear_prefetched_params(owner_id);
    if (!load_tensors_to_params_backend(states)) {
        return WeightPrefetchResult::Failed;
    }

    auto block             = std::make_unique<PrefetchBlock>();
    block->states          = std::move(states);
    block->compute_backend = compute_backend;
    if (!populate_prefetch_block(*block)) {
        free_prefetch_block(*block);
        return WeightPrefetchResult::Failed;
    }
    prefetch_blocks_[owner_id] = std::move(block);
    return WeightPrefetchResult::Scheduled;
}

bool ModelManager::activate_prefetched_params(
    uintptr_t owner_id,
    const std::vector<ggml_tensor*>& tensors) {
    std::vector<TensorState*> required_states;
    if (!resolve_required_tensor_states(tensors, required_states)) {
        return false;
    }

    const bool already_staged = std::all_of(
        required_states.begin(),
        required_states.end(),
        [&](TensorState* state) {
            return state == nullptr || should_ignore(*state) ||
                   is_optional_missing_tensor(state->name) ||
                   state->compute_backend == state->params_backend ||
                   state->staged_to_compute_backend;
        });
    if (already_staged) {
        clear_prefetched_params(owner_id);
        return true;
    }

    auto existing = prefetch_blocks_.find(owner_id);
    if (existing == prefetch_blocks_.end()) {
        return false;
    }
    std::unique_ptr<PrefetchBlock> block = std::move(existing->second);
    prefetch_blocks_.erase(existing);
    synchronize_prefetch_block(*block);

    for (const auto& staging_block : block->staging_blocks) {
        if (staging_block == nullptr) {
            continue;
        }
        for (const auto& pair : staging_block->staged_tensors) {
            TensorState* state          = pair.first;
            ggml_tensor* staging_tensor = pair.second;
            if (state == nullptr || state->tensor == nullptr || staging_tensor == nullptr ||
                state->staged_to_compute_backend || state->pin_count > 0) {
                free_prefetch_block(*block);
                return false;
            }
        }
    }
    const uint64_t use_epoch = ++residency_epoch_;
    for (auto& staging_block : block->staging_blocks) {
        if (staging_block == nullptr) {
            continue;
        }
        for (auto& pair : staging_block->staged_tensors) {
            TensorState* state          = pair.first;
            ggml_tensor* staging_tensor = pair.second;
            std::swap(state->tensor->buffer, staging_tensor->buffer);
            std::swap(state->tensor->data, staging_tensor->data);
            std::swap(state->tensor->extra, staging_tensor->extra);
            state->staged_to_compute_backend = true;
            state->last_use_epoch            = use_epoch;
        }
        compute_staging_blocks_.push_back(std::move(staging_block));
    }
    block->staging_blocks.clear();
    return true;
}

void ModelManager::clear_prefetched_params(uintptr_t owner_id) {
    auto existing = prefetch_blocks_.find(owner_id);
    if (existing == prefetch_blocks_.end()) {
        return;
    }
    std::unique_ptr<PrefetchBlock> block = std::move(existing->second);
    prefetch_blocks_.erase(existing);
    free_prefetch_block(*block);
}

void ModelManager::clear_all_prefetched_params() {
    for (auto& entry : prefetch_blocks_) {
        free_prefetch_block(*entry.second);
    }
    prefetch_blocks_.clear();
}

void ModelManager::release_prefetch() {
    clear_all_prefetched_params();
    for (auto& entry : prefetch_backends_) {
        if (entry.second != nullptr) {
            ggml_backend_free(entry.second);
        }
    }
    prefetch_backends_.clear();
}

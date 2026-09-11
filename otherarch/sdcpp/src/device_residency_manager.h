#ifndef __DEVICE_RESIDENCY_MANAGER_H__
#define __DEVICE_RESIDENCY_MANAGER_H__

#include <cstdint>
#include <functional>
#include <vector>

#include "ggml-backend.h"

struct ggml_tensor;

enum class WeightPrefetchResult {
    Scheduled,
    AlreadyResident,
    Unsupported,
    Failed,
};

struct WeightResidencyInfo {
    bool async_prefetch_supported = false;
    size_t missing_bytes          = 0;
};

struct DeviceMemoryRequest {
    ggml_backend_t compute_backend  = nullptr;
    uintptr_t owner_id              = 0;
    size_t pending_allocation_bytes = 0;
    size_t runtime_resident_bytes   = 0;
    size_t max_backend_bytes        = 0;

    // Runtime buffers only; the manager accounts for weights separately.
    size_t runtime_peak_bytes() const {
        return pending_allocation_bytes > SIZE_MAX - runtime_resident_bytes
                   ? SIZE_MAX
                   : runtime_resident_bytes + pending_allocation_bytes;
    }
};

struct DeviceResidencyManager {
    virtual ~DeviceResidencyManager() = default;

    virtual bool segmented_compute_enabled() const                                          = 0;
    virtual bool prefetch_enabled() const                                                   = 0;
    virtual void set_workspace_reclaimer(uintptr_t owner_id, std::function<bool()> reclaim) = 0;
    virtual void remove_runtime_owner(uintptr_t owner_id)                                   = 0;
    // Capacity requests select their backend's weights; protection spans all backends.
    virtual bool fits_compute_backend_capacity(const DeviceMemoryRequest& request,
                                               const std::vector<ggml_tensor*>& required_params) const = 0;
    virtual bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                        ggml_backend_t compute_backend)                                = 0;
    virtual bool prepare_params(const std::vector<ggml_tensor*>& tensors)                              = 0;
    virtual void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors)              = 0;
    virtual void evict_compute_backend_params(const std::vector<ggml_tensor*>& tensors)                = 0;
    virtual WeightResidencyInfo inspect_compute_backend_params(
        const std::vector<ggml_tensor*>& tensors) const          = 0;
    virtual void update_runtime_residency(uintptr_t owner_id,
                                          ggml_backend_t compute_backend,
                                          size_t resident_bytes) = 0;
    virtual bool ensure_compute_backend_capacity(
        const DeviceMemoryRequest& request,
        const std::vector<ggml_tensor*>& required_params,
        const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
        const std::vector<ggml_tensor*>& protected_params) = 0;
    virtual WeightPrefetchResult prefetch_params(
        uintptr_t owner_id,
        const std::vector<ggml_tensor*>& tensors)                                     = 0;
    virtual bool activate_prefetched_params(uintptr_t owner_id,
                                            const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void clear_prefetched_params(uintptr_t owner_id)                          = 0;
};

// Transitional alias for model constructors that have not yet adopted the
// residency-oriented name. It does not introduce a second implementation.
using RunnerWeightManager = DeviceResidencyManager;

#endif  // __DEVICE_RESIDENCY_MANAGER_H__

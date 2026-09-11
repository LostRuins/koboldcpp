#ifndef __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__
#define __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "ggml-backend.h"

struct DeviceMemoryRequest;
struct DeviceResidencyManager;
struct ggml_cgraph;
struct ggml_tensor;

namespace sd::ggml_graph_cut {
    struct Plan;
}

namespace sd {
    class SegmentWeightPipeline {
    private:
        std::weak_ptr<DeviceResidencyManager> residency_manager_;
        ggml_backend_t compute_backend_ = nullptr;
        uintptr_t owner_id_             = 0;
        std::vector<std::vector<ggml_tensor*>> segment_params_;
        std::vector<ggml_tensor*> queued_params_;
        std::vector<ggml_tensor*> pinned_params_;
        size_t queued_segment_ = SIZE_MAX;
        bool enabled_          = true;

        size_t next_parameter_segment(size_t segment_index) const;
        std::vector<std::vector<ggml_tensor*>> preferred_eviction_order() const;
        void disable();
        void activate(size_t segment_index);
        void clear();

    public:
        SegmentWeightPipeline(
            const std::shared_ptr<DeviceResidencyManager>& residency_manager,
            ggml_backend_t compute_backend,
            uintptr_t owner_id,
            ggml_cgraph* graph,
            const ggml_graph_cut::Plan& plan,
            const std::unordered_set<const ggml_tensor*>& params,
            bool enabled = true);
        ~SegmentWeightPipeline();

        const std::vector<ggml_tensor*>& params(size_t index) const { return segment_params_[index]; }
        bool ensure_segment_capacity(size_t segment_index,
                                     const std::vector<DeviceMemoryRequest>& requests);
        bool segment_start(size_t segment_index, const std::function<bool()>& ensure_capacity);
        void segment_end();
        // Prefetch is best effort; segment_start falls back to synchronous loading.
        void enqueue_next(size_t segment_index,
                          const DeviceMemoryRequest& request);
    };
}

#endif  // __SD_CORE_SEGMENT_WEIGHT_PIPELINE_H__

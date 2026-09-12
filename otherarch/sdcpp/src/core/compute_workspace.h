#ifndef __SD_CORE_COMPUTE_WORKSPACE_H__
#define __SD_CORE_COMPUTE_WORKSPACE_H__

#include <functional>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace sd {
    struct BackendBufferSize {
        ggml_backend_t backend = nullptr;
        size_t bytes           = 0;
    };

    class ComputeWorkspace {
        ggml_backend_t backend_;
        std::vector<ggml_backend_t> extra_backends_;
        ggml_backend_t cpu_backend_     = nullptr;
        ggml_gallocr_t allocator_       = nullptr;
        ggml_backend_sched_t scheduler_ = nullptr;
        size_t scheduler_capacity_      = 0;
        std::vector<uint64_t> reservation_;
        bool active_         = false;
        size_t reservations_ = 0;

        ggml_backend_sched_t make_scheduler(size_t graph_size);
        bool needs_scheduler(ggml_cgraph* graph) const;

    public:
        struct Measurement {
            std::vector<BackendBufferSize> buffers;
            bool scheduler = false;
        };
        using AssignNodes = std::function<void(ggml_backend_sched_t, ggml_cgraph*)>;

        explicit ComputeWorkspace(ggml_backend_t backend)
            : backend_(backend) {}
        ~ComputeWorkspace();
        ComputeWorkspace(const ComputeWorkspace&)            = delete;
        ComputeWorkspace& operator=(const ComputeWorkspace&) = delete;

        void set_extra_backends(const std::vector<ggml_backend_t>& backends);
        bool measurement_matches(ggml_cgraph* graph, const Measurement& measurement) const;
        bool prepare(const Measurement& measurement);
        bool release_excess(const Measurement& measurement);
        bool allocate(ggml_cgraph* graph, const AssignNodes& assign_nodes);
        Measurement measure(
            ggml_cgraph* graph,
            size_t direct_bytes,
            const std::function<ggml_backend_t(const ggml_tensor*)>& external_backend,
            const AssignNodes& assign_nodes);
        void synchronize() const;
        void segment_end();
        bool release();
        bool active() const { return active_; }
        ggml_backend_sched_t scheduler() const { return scheduler_; }
        ggml_backend_t cpu_backend() const { return cpu_backend_; }
        size_t bytes(ggml_backend_t backend) const;
        size_t reservation_count() const { return reservations_; }
    };
}

#endif  // __SD_CORE_COMPUTE_WORKSPACE_H__

#ifndef __SD_CORE_EXECUTION_ENGINE_H__
#define __SD_CORE_EXECUTION_ENGINE_H__

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/compute_workspace.h"
#include "core/ggml_graph_cut.h"
#include "core/runner_cache.h"
#include "core/tensor.hpp"
#include "device_residency_manager.h"

namespace sd {
    struct ComputeTask {
        std::string description;
        ggml_cgraph* graph  = nullptr;
        ggml_tensor* output = nullptr;
        std::vector<ggml_tensor*> weights;
        std::map<ggml_tensor*, const void*> inputs;
        std::vector<std::pair<ggml_tensor*, std::string>> debug_tensors;
        int n_threads  = 1;
        bool no_return = false;
        std::function<bool()> read_outputs;
    };

    struct ComputeResult {
        ggml_status status = GGML_STATUS_FAILED;
        Tensor<float> output;
    };

    class ExecutionSession {
        friend class ExecutionEngine;
        friend class GraphExecution;

        ggml_backend_t backend_;
        std::weak_ptr<DeviceResidencyManager> residency_manager_;
        RunnerCache cache_;
        GraphCutTensorCache cut_cache_;
        ComputeWorkspace workspace_;
        std::unordered_set<ggml_tensor*> params_;
        std::vector<ggml_backend_t> extra_backends_;
        size_t max_backend_bytes_ = 0;
        bool layer_split_enabled_ = false;
        std::vector<size_t> backend_limits_;
        ggml_graph_cut::PlanCache plan_cache_;
        std::unordered_map<const ggml_tensor*, ggml_backend_t> weight_assignments_;
        std::unordered_map<const ggml_tensor*, ggml_backend_t> node_assignments_;
        bool primary_notice_logged_ = false;
        std::map<ggml_backend_t, size_t> logged_compute_bytes_;
        size_t logged_segment_count_ = 0;
        bool started_                = false;
        bool executing_              = false;

        size_t resident_bytes(ggml_backend_t backend) const;
        uintptr_t owner_id() const { return reinterpret_cast<uintptr_t>(this); }

    public:
        ExecutionSession(ggml_backend_t backend, const std::shared_ptr<DeviceResidencyManager>& manager);
        ExecutionSession(const ExecutionSession&)            = delete;
        ExecutionSession& operator=(const ExecutionSession&) = delete;

        RunnerCache& cache() { return cache_; }
        bool started() const { return started_; }
        size_t compute_bytes() const { return workspace_.bytes(backend_); }
        void clear_plan_cache() { plan_cache_.graph_cut_plans.clear(); }
        void set_max_backend_bytes(size_t bytes) { max_backend_bytes_ = bytes; }
        void set_layer_split_enabled(bool enabled);
        void set_backend_limits(const std::vector<size_t>& limits);
        void set_runtime_backends(const std::vector<ggml_backend_t>& backends);
    };

    class ExecutionEngine {
        friend class GraphExecution;

        std::map<uintptr_t, std::unique_ptr<ExecutionSession>> sessions_;
        DeviceResidencyManager* manager_;

        void sync_residency(ExecutionSession& session);

    public:
        explicit ExecutionEngine(DeviceResidencyManager* manager = nullptr)
            : manager_(manager) {}
        ~ExecutionEngine();
        ExecutionSession& create_session(ggml_backend_t backend, const std::shared_ptr<DeviceResidencyManager>& manager);
        void release_session(ExecutionSession& session);
        bool start(ExecutionSession& session);
        void end(ExecutionSession& session);
        void clear_cache(ExecutionSession& session);
        void discard_graph(ExecutionSession& session);
        bool ensure_capacity(DeviceResidencyManager& manager,
                             const DeviceMemoryRequest& request,
                             const std::vector<ggml_tensor*>& required_params,
                             const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
                             const std::vector<ggml_tensor*>& protected_params);
        // Graphs, input data and output callbacks remain borrowed until execute returns.
        ComputeResult execute(ExecutionSession& session, const ComputeTask& task);
    };
}

#endif  // __SD_CORE_EXECUTION_ENGINE_H__

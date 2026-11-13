#include "core/execution_engine.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <thread>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/ggml_tensor_utils.h"
#include "core/layer_split_partition.h"
#include "core/segment_graph_bindings.h"
#include "core/segment_weight_pipeline.h"
#include "core/tensor_ggml.hpp"
#include "core/util.h"

std::shared_ptr<sd::ExecutionEngine> DeviceResidencyManager::execution_engine() {
    if (execution_engine_ == nullptr) {
        execution_engine_ = std::make_shared<sd::ExecutionEngine>(this);
    }
    return execution_engine_;
}

namespace sd {
    class GraphExecution {
        ExecutionEngine& engine_;
        ExecutionSession& session_;
        const ComputeTask& task_;
        std::shared_ptr<DeviceResidencyManager> manager_;
        std::unordered_set<const ggml_tensor*> params_tensor_set_;
        ggml_status last_compute_status_ = GGML_STATUS_FAILED;
        Tensor<float> output_;

        using GraphCutPlan = ggml_graph_cut::Plan;
        ggml_tensor* canonical_param_tensor(ggml_tensor* tensor);
        std::vector<ggml_tensor*> collect_used_param_tensors(ggml_cgraph* graph);
        ggml_backend_t backend_for_weight(const ggml_tensor* tensor) const;
        // Weightless ops follow the most recent weight device. Views must stay
        // unpinned so the scheduler preserves cross-device copies for consumers.
        void pin_multi_device_nodes(ggml_backend_sched_t sched, ggml_cgraph* graph, ggml_cgraph* original_graph = nullptr);
        Tensor<float> read_graph_tensor(ggml_tensor* tensor, const char* label);
        void copy_data_to_backend_tensor(ggml_cgraph* graph);
        const GraphCutPlan& resolve_graph_cut_plan(ggml_cgraph* graph);
        bool assign_graph_cut_layer_split_backends(ggml_cgraph* graph);
        ComputeWorkspace::Measurement measure(ggml_cgraph* graph, const std::unordered_set<const ggml_tensor*>* external_outputs = nullptr);
        std::vector<DeviceMemoryRequest> memory_requests(const std::vector<BackendBufferSize>& sizes, size_t pending_cache_bytes) const;
        bool fits(const std::vector<DeviceMemoryRequest>& requests, const std::vector<ggml_tensor*>& params) const;
        bool execute_segment(ggml_cgraph* graph, int n_threads, const std::function<void()>& on_before_compute);
        bool execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs);
        bool is_multi_device() const { return !session_.extra_backends_.empty(); }
        size_t retained_runtime_buffer_bytes(ggml_backend_t backend) const { return session_.resident_bytes(backend); }
        void sync_runtime_residency() { engine_.sync_residency(session_); }

    public:
        GraphExecution(ExecutionEngine& engine, ExecutionSession& session, const ComputeTask& task)
            : engine_(engine), session_(session), task_(task), manager_(session.residency_manager_.lock()) {
            params_tensor_set_.insert(task.weights.begin(), task.weights.end());
            if (manager_ != nullptr) {
                for (int i = 0; i < ggml_graph_cut::leaf_count(task.graph); ++i) {
                    auto* parameter = manager_->resolve_param_tensor(ggml_graph_cut::leaf_tensor(task.graph, i));
                    if (parameter != nullptr) {
                        params_tensor_set_.insert(parameter);
                    }
                }
                for (auto* tensor : params_tensor_set_) {
                    auto* parameter = manager_->resolve_param_tensor(const_cast<ggml_tensor*>(tensor));
                    if (parameter != nullptr) {
                        session_.params_.insert(parameter);
                    }
                }
            }
        }

        ComputeResult run() {
            if (!execute_graph(task_.graph, task_.n_threads, task_.no_return, task_.read_outputs)) {
                return {last_compute_status_, {}};
            }
            return {GGML_STATUS_SUCCESS, std::move(output_)};
        }
    };

    static bool backend_can_allocate_op(ggml_backend_t backend, const ggml_tensor* node) {
        // Vulkan reports a preferred allocation chunk size, not a hard tensor limit.
        if (sd_backend_is(backend, "Vulkan")) {
            return true;
        }
        auto buft        = ggml_backend_get_default_buffer_type(backend);
        size_t alignment = ggml_backend_buft_get_alignment(buft);
        size_t max_size  = ggml_backend_buft_get_max_size(buft);
        max_size -= max_size % alignment;
        if (ggml_backend_buft_get_alloc_size(buft, node) > max_size) {
            return false;
        }
        // Cross-backend copies must also fit the destination's allocation limit.
        for (auto src : node->src) {
            if (src != nullptr && ggml_backend_buft_get_alloc_size(buft, src) > max_size) {
                return false;
            }
        }
        return true;
    }

    static size_t add_bytes(size_t a, size_t b) {
        return b > SIZE_MAX - a ? SIZE_MAX : a + b;
    }

    ggml_tensor* GraphExecution::canonical_param_tensor(ggml_tensor* tensor) {
        for (auto* current = tensor; current != nullptr; current = current->view_src) {
            if (params_tensor_set_.count(current) != 0)
                return current;
        }
        return nullptr;
    }

    std::vector<ggml_tensor*> GraphExecution::collect_used_param_tensors(ggml_cgraph* gf) {
        std::vector<ggml_tensor*> used_params;
        if (gf == nullptr || params_tensor_set_.empty()) {
            return used_params;
        }

        std::unordered_set<const ggml_tensor*> seen_params;
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        seen_params.reserve(static_cast<size_t>(n_leafs));
        for (int i = 0; i < n_leafs; ++i) {
            ggml_tensor* leaf       = sd::ggml_graph_cut::leaf_tensor(gf, i);
            ggml_tensor* param_leaf = canonical_param_tensor(leaf);
            if (param_leaf != nullptr &&
                seen_params.insert(param_leaf).second) {
                used_params.push_back(param_leaf);
            }
        }
        return used_params;
    }

    ggml_backend_t GraphExecution::backend_for_weight(const ggml_tensor* tensor) const {
        if (tensor == nullptr || tensor->buffer == nullptr) {
            return nullptr;
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            ggml_backend_buffer_is_host(tensor->buffer)) {
            return nullptr;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        if (dev == nullptr) {
            return nullptr;
        }
        if (ggml_backend_get_device(session_.backend_) == dev) {
            return session_.backend_;
        }
        for (ggml_backend_t backend : session_.extra_backends_) {
            if (ggml_backend_get_device(backend) == dev) {
                return backend;
            }
        }
        return nullptr;
    }

    void GraphExecution::pin_multi_device_nodes(ggml_backend_sched_t sched, ggml_cgraph* gf, ggml_cgraph* original_graph) {
        if (sched == nullptr || gf == nullptr) {
            return;
        }
        ggml_backend_t current = session_.backend_;
        const int n_nodes      = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; i++) {
            ggml_tensor* node    = ggml_graph_node(gf, i);
            auto node_assignment = session_.node_assignments_.find(original_graph == nullptr ? node : ggml_graph_node(original_graph, i));
            if (node_assignment != session_.node_assignments_.end()) {
                current = node_assignment->second;
            }
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                ggml_backend_t weight_backend = backend_for_weight(node->src[s]);
                if (weight_backend != nullptr) {
                    if (node_assignment == session_.node_assignments_.end()) {
                        current = weight_backend;
                    }
                }
            }
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
                continue;
            }
            if (!backend_can_allocate_op(current, node)) {
                auto cpu = session_.workspace_.cpu_backend();
                if (cpu != nullptr && ggml_backend_supports_op(cpu, node)) {
                    ggml_backend_sched_set_tensor_backend(sched, node, cpu);
                }
            } else if (ggml_backend_supports_op(current, node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, current);
            }
        }
    }

    Tensor<float> GraphExecution::read_graph_tensor(ggml_tensor* tensor, const char* label) {
        if (tensor == nullptr) {
            LOG_ERROR("%s %s tensor is null", task_.description.c_str(), label);
            return {};
        }
        if (tensor->type != GGML_TYPE_F32) {
            LOG_ERROR("%s %s tensor type mismatch: got %s",
                      task_.description.c_str(),
                      label,
                      ggml_type_name(tensor->type));
            return {};
        }
        ggml_backend_buffer_t buf = sd::ggml_graph_cut::tensor_buffer(tensor);
        if (buf == nullptr) {
            LOG_ERROR("%s %s tensor buffer missing: name=%s op=%s buffer=%p view_src=%p view_src_buffer=%p data=%p",
                      task_.description.c_str(),
                      label,
                      tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                      ggml_op_name(tensor->op),
                      tensor->buffer,
                      tensor->view_src,
                      tensor->view_src ? tensor->view_src->buffer : nullptr,
                      tensor->data);
            return {};
        }

        return sd::make_sd_tensor_from_ggml<float>(tensor);
    }

    void GraphExecution::copy_data_to_backend_tensor(ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);
        std::unordered_set<const ggml_tensor*> graph_tensor_set;
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_nodes = ggml_graph_n_nodes(gf);
        graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
        for (int i = 0; i < n_leafs; ++i) {
            graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_nodes; ++i) {
            graph_tensor_set.insert(ggml_graph_node(gf, i));
        }

        for (auto& kv : task_.inputs) {
            auto tensor = kv.first;
            auto data   = kv.second;
            if (tensor == nullptr || data == nullptr) {
                continue;
            }
            const char* name = ggml_get_name(tensor);
            if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) {
                continue;
            }
            if (tensor->buffer == nullptr) {
                LOG_WARN("%s skip backend tensor copy: tensor buffer not set, name='%s', ne=[%lld,%lld,%lld,%lld], type=%s",
                         task_.description.c_str(),
                         name != nullptr ? name : "",
                         (long long)tensor->ne[0],
                         (long long)tensor->ne[1],
                         (long long)tensor->ne[2],
                         (long long)tensor->ne[3],
                         ggml_type_name(tensor->type));
                continue;
            }

            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (buf == nullptr) {
                LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                         task_.description.c_str(),
                         tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                         tensor ? ggml_op_name(tensor->op) : "<null>",
                         data,
                         tensor ? tensor->view_src : nullptr,
                         (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
                continue;
            }

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }
    }

    const GraphExecution::GraphCutPlan& GraphExecution::resolve_graph_cut_plan(ggml_cgraph* gf) {
        GGML_ASSERT(gf != nullptr);
        return sd::ggml_graph_cut::resolve_plan(session_.backend_,
                                                gf,
                                                &session_.plan_cache_,
                                                params_tensor_set_,
                                                task_.description.c_str());
    }

    bool GraphExecution::assign_graph_cut_layer_split_backends(ggml_cgraph* gf) {
        session_.node_assignments_.clear();
        if (!session_.layer_split_enabled_) {
            return true;
        }
        if (!is_multi_device()) {
            LOG_ERROR("%s graph-cut layer split requires multiple runtime backends", task_.description.c_str());
            return false;
        }

        const auto& plan = resolve_graph_cut_plan(gf);
        if (!plan.valid || !plan.has_cuts || plan.segments.size() <= 1) {
            auto manager = manager_;
            if (manager == nullptr) {
                LOG_ERROR("%s weight manager is not set for graph-cut layer split", task_.description.c_str());
                return false;
            }
            std::vector<ggml_tensor*> graph_params = collect_used_param_tensors(gf);
            if (!graph_params.empty() &&
                !manager->assign_compute_backend(graph_params, session_.backend_)) {
                LOG_ERROR("%s graph-cut layer split failed to assign unmarked graph params to %s",
                          task_.description.c_str(),
                          sd::layer_split_backend_device_display_name(session_.backend_).c_str());
                return false;
            }
            for (ggml_tensor* param : graph_params) {
                if (param != nullptr) {
                    session_.weight_assignments_[param] = session_.backend_;
                }
            }
            const int n_nodes = ggml_graph_n_nodes(gf);
            for (int i = 0; i < n_nodes; i++) {
                ggml_tensor* node = ggml_graph_node(gf, i);
                if (node != nullptr) {
                    session_.node_assignments_[node] = session_.backend_;
                }
            }
            if (!session_.primary_notice_logged_) {
                LOG_WARN("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                         task_.description.c_str(),
                         sd::layer_split_backend_device_display_name(session_.backend_).c_str(),
                         graph_params.size());
                session_.primary_notice_logged_ = true;
            } else {
                LOG_VERBOSE("%s graph-cut layer split: graph has no mark_graph_cut segments; using primary backend %s for %zu graph params",
                            task_.description.c_str(),
                            sd::layer_split_backend_device_display_name(session_.backend_).c_str(),
                            graph_params.size());
            }
            return true;
        }

        std::vector<ggml_backend_t> split_backends;
        split_backends.reserve(session_.extra_backends_.size() + 1);
        split_backends.push_back(session_.backend_);
        for (ggml_backend_t backend : session_.extra_backends_) {
            if (backend != nullptr) {
                split_backends.push_back(backend);
            }
        }

        auto manager = manager_;
        if (manager == nullptr) {
            LOG_ERROR("%s weight manager is not set for graph-cut layer split", task_.description.c_str());
            return false;
        }

        sd::GraphCutLayerSplitAssignment assignment;
        auto canonicalize_param = [this](ggml_tensor* tensor) {
            return canonical_param_tensor(tensor);
        };
        if (!sd::partition_graph_cut_layer_split(task_.description.c_str(),
                                                 gf,
                                                 plan,
                                                 split_backends,
                                                 session_.backend_limits_,
                                                 session_.max_backend_bytes_,
                                                 session_.weight_assignments_,
                                                 canonicalize_param,
                                                 &assignment)) {
            return false;
        }

        for (size_t i = 0; i < split_backends.size(); i++) {
            if (assignment.tensors_by_backend[i].empty()) {
                continue;
            }
            if (!manager->assign_compute_backend(assignment.tensors_by_backend[i], split_backends[i])) {
                LOG_ERROR("%s graph-cut layer split failed to assign params to %s",
                          task_.description.c_str(),
                          sd::layer_split_backend_device_display_name(split_backends[i]).c_str());
                return false;
            }
        }

        session_.node_assignments_ = std::move(assignment.node_assignments);
        sd::log_graph_cut_layer_split_assignment(task_.description.c_str(), split_backends, assignment);

        return true;
    }

    ComputeWorkspace::Measurement GraphExecution::measure(ggml_cgraph* graph, const std::unordered_set<const ggml_tensor*>* external_outputs) {
        auto external_backend = [&](const ggml_tensor* tensor) -> ggml_backend_t {
            if (external_outputs != nullptr && external_outputs->count(tensor)) {
                return session_.backend_;
            }
            if (!params_tensor_set_.count(tensor)) {
                return nullptr;
            }
            auto placement = session_.weight_assignments_.find(tensor);
            return placement == session_.weight_assignments_.end() ? session_.backend_ : placement->second;
        };
        auto assign_nodes = [&](ggml_backend_sched_t scheduler, ggml_cgraph* copy) {
            pin_multi_device_nodes(scheduler, copy, graph);
        };
        return session_.workspace_.measure(graph, external_backend, assign_nodes, !is_multi_device());
    }

    std::vector<DeviceMemoryRequest> GraphExecution::memory_requests(
        const std::vector<BackendBufferSize>& sizes,
        size_t pending_cache_bytes) const {
        std::vector<DeviceMemoryRequest> requests;
        for (const auto& size : sizes) {
            const size_t retained    = retained_runtime_buffer_bytes(size.backend);
            const size_t reusable    = session_.workspace_.bytes(size.backend);
            const size_t cache_bytes = size.backend == session_.backend_ ? pending_cache_bytes : 0;
            const size_t pending     = add_bytes(size.bytes > reusable ? size.bytes - reusable : 0, cache_bytes);
            size_t limit             = session_.max_backend_bytes_;
            if (is_multi_device()) {
                size_t index = 0;
                if (size.backend != session_.backend_) {
                    auto position = std::find(session_.extra_backends_.begin(), session_.extra_backends_.end(), size.backend);
                    index         = static_cast<size_t>(position - session_.extra_backends_.begin()) + 1;
                }
                if (index < session_.backend_limits_.size()) {
                    limit = session_.backend_limits_[index];
                }
            }
            requests.push_back({size.backend, session_.owner_id(), pending,
                                retained, limit});
        }
        return requests;
    }

    bool GraphExecution::fits(const std::vector<DeviceMemoryRequest>& requests,
                              const std::vector<ggml_tensor*>& params) const {
        auto manager = manager_;
        if (manager == nullptr) {
            return params.empty();
        }
        for (const auto& request : requests) {
            if (!manager->fits_compute_backend_capacity(request, params)) {
                return false;
            }
        }
        return true;
    }

    bool GraphExecution::execute_segment(ggml_cgraph* graph, int n_threads, const std::function<void()>& on_before_compute) {
        if (sd_backend_is_cpu(session_.backend_)) {
            sd_backend_cpu_set_n_threads(session_.backend_, n_threads);
        }
        if (session_.workspace_.cpu_backend() != nullptr) {
            sd_backend_cpu_set_n_threads(session_.workspace_.cpu_backend(), n_threads);
        }
        auto scheduler = session_.workspace_.scheduler();
        GGML_ASSERT(scheduler != nullptr);
        ggml_backend_sched_set_eval_callback(scheduler, sd_get_backend_eval_callback(), sd_get_backend_eval_callback_data());
#ifndef SD_USE_UPSTREAM_GGML
        struct BeforeComputeCallback {
            ggml_backend_t backend;
            const std::function<void()>& callback;
            bool invoked = false;
        } before_compute{session_.backend_, on_before_compute};
        ggml_backend_sched_set_compute_callback(
            scheduler, [](ggml_backend_t backend, void* data) {
                auto& hook = *static_cast<BeforeComputeCallback*>(data);
                if (backend == hook.backend && !hook.invoked) {
                    hook.invoked = true;
                    hook.callback();
                }
            },
            &before_compute);
        const auto status = ggml_backend_sched_graph_compute_async(scheduler, graph);
        ggml_backend_sched_set_compute_callback(scheduler, nullptr, nullptr);
#else
        on_before_compute();
        const auto status = ggml_backend_sched_graph_compute_async(scheduler, graph);
#endif
        session_.workspace_.synchronize();
        if (status != GGML_STATUS_SUCCESS) {
            last_compute_status_ = status;
            LOG_ERROR("%s compute failed: %s", task_.description.c_str(), ggml_status_to_string(status));
            return false;
        }
        const std::string description = task_.description;
        if (!task_.debug_tensors.empty()) {
            std::unordered_set<const ggml_tensor*> graph_tensors;
            const int leaf_count = ggml_graph_cut::leaf_count(graph);
            const int node_count = ggml_graph_n_nodes(graph);
            graph_tensors.reserve(static_cast<size_t>(leaf_count + node_count));
            for (int index = 0; index < leaf_count; ++index) {
                graph_tensors.insert(ggml_graph_cut::leaf_tensor(graph, index));
            }
            for (int index = 0; index < node_count; ++index) {
                graph_tensors.insert(ggml_graph_node(graph, index));
            }

            for (const auto& entry : task_.debug_tensors) {
                ggml_tensor* tensor = entry.first;
                if (tensor == nullptr || graph_tensors.find(tensor) == graph_tensors.end()) {
                    continue;
                }
                ggml_backend_buffer_t buffer =
                    tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
                if (buffer == nullptr) {
                    LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                             description.c_str(),
                             entry.second.c_str());
                    continue;
                }
                if (tensor->type != GGML_TYPE_F32) {
                    LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                             description.c_str(),
                             entry.second.c_str(),
                             ggml_type_name(tensor->type));
                    continue;
                }
                auto debug_tensor = make_sd_tensor_from_ggml<float>(tensor);
                print_sd_tensor(debug_tensor, false, entry.second.c_str());
            }
        }

        return true;
    }

    bool GraphExecution::execute_graph(ggml_cgraph* graph, int n_threads, bool no_return, const std::function<bool()>& read_outputs) {
        if (!assign_graph_cut_layer_split_backends(graph)) {
            return false;
        }
        const auto params       = collect_used_param_tensors(graph);
        const auto& cached_plan = resolve_graph_cut_plan(graph);
        auto manager            = manager_;
        const bool can_segment  = !is_multi_device() && !sd_backend_is_cpu(session_.backend_) &&
                                  manager != nullptr && manager->segmented_compute_enabled() &&
                                  cached_plan.valid && cached_plan.has_cuts && cached_plan.segments.size() > 1;
        // An empty workspace request is a lower bound on the full graph's memory demand.
        bool segmented = can_segment && !fits(memory_requests({{session_.backend_, 0}}, 0), params);
        ComputeWorkspace::Measurement full_measurement;
        if (!segmented) {
            full_measurement = measure(graph);
            if (full_measurement.buffers.empty()) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return false;
            }
            if (can_segment) {
                // Planning headroom absorbs allocation estimate drift; execution keeps the normal limits.
                constexpr size_t planning_headroom = 128ULL * 1024ULL * 1024ULL;
                auto requests                      = memory_requests(full_measurement.buffers, session_.cache_.pending_bytes(graph));
                for (auto& request : requests) {
                    request.pending_allocation_bytes = add_bytes(request.pending_allocation_bytes, planning_headroom);
                }
                segmented = !fits(requests, params);
            }
        }
        ggml_graph_cut::Plan monolithic_plan;
        if (!segmented) {
            monolithic_plan.segments.emplace_back();
            auto& segment               = monolithic_plan.segments.back();
            segment.group_name          = "graph";
            segment.compute_buffer_size = cached_plan.compute_buffer_size;
            segment.internal_node_indices.reserve(ggml_graph_n_nodes(graph));
            segment.input_refs.reserve(ggml_graph_cut::leaf_count(graph));
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                segment.internal_node_indices.push_back(i);
            }
            for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
                auto tensor = ggml_graph_cut::leaf_tensor(graph, i);
                ggml_graph_cut::Segment::InputRef input;
                input.leaf_index = i;
                input.type       = canonical_param_tensor(tensor) != nullptr
                                       ? ggml_graph_cut::Segment::INPUT_PARAM
                                       : ggml_graph_cut::Segment::INPUT_EXTERNAL;
                segment.input_refs.push_back(input);
            }
        }
        const auto& plan            = segmented ? cached_plan : monolithic_plan;
        const bool segments_changed = plan.segments.size() != session_.logged_segment_count_;
        if (segments_changed && (segmented || session_.logged_segment_count_ > 1)) {
            LOG_VERBOSE("%s using %zu segment%s", task_.description.c_str(),
                        plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
        }
        SegmentGraphBindings bindings(session_.cut_cache_, plan, graph);
        SegmentWeightPipeline weights(manager, session_.backend_, session_.owner_id(),
                                      graph, plan, params_tensor_set_,
                                      segmented && manager != nullptr && manager->prefetch_enabled());

        std::map<ggml_backend_t, size_t> peak_compute_bytes;
        auto track_compute_buffer = [&](ggml_backend_t backend) {
            if (backend != nullptr) {
                auto& peak = peak_compute_bytes[backend];
                peak       = std::max(peak, session_.workspace_.bytes(backend));
            }
        };
        for (size_t index = 0; index < plan.segments.size(); ++index) {
            const auto& segment = plan.segments[index];
            const bool last     = index + 1 == plan.segments.size();
            auto fail_segment   = [&](const char* phase) {
                LOG_ERROR("%s segment %zu/%zu (%s) failed during %s", task_.description.c_str(),
                          index + 1, plan.segments.size(), segment.group_name.c_str(), phase);
                return false;
            };
            session_.cut_cache_.prune(segment.live_cut_names);
            bindings.reset(segment);
            if (!bindings.bind_cached_inputs(segment, task_.description.c_str())) {
                return fail_segment("input binding");
            }
            ggml_context* segment_context = nullptr;
            auto segment_graph            = segmented
                                                ? ggml_graph_cut::build_segment_graph(graph, segment, &segment_context)
                                                : graph;
            struct SegmentCleanup {
                GraphExecution& runner;
                SegmentWeightPipeline& weights;
                SegmentGraphBindings& bindings;
                ggml_context* context;
                ~SegmentCleanup() {
                    if (!runner.session_.workspace_.segment_end()) {
                        runner.last_compute_status_ = GGML_STATUS_FAILED;
                    }
                    bindings.restore();
                    weights.segment_end();
                    ggml_free(context);
                    runner.sync_runtime_residency();
                }
            } segment_cleanup{*this, weights, bindings, segment_context};

            std::unordered_set<const ggml_tensor*> cache_outputs;
            if (segmented) {
                for (int output_index : segment.output_node_indices) {
                    auto tensor = ggml_graph_node(graph, output_index);
                    if (ggml_graph_cut::is_graph_cut_tensor(tensor) && segment.future_cut_names.count(tensor->name) &&
                        GraphCutTensorCache::can_bind_output(session_.backend_, tensor)) {
                        cache_outputs.insert(tensor);
                    }
                }
            }
            auto measurement = segmented ? measure(segment_graph, &cache_outputs) : full_measurement;
            if (!session_.workspace_.prepare(measurement)) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return fail_segment("workspace preparation");
            }
            const size_t cut_bytes = last ? 0 : session_.cut_cache_.estimate_output_bytes(graph, segment);
            size_t new_cache_bytes = add_bytes(cut_bytes, session_.cache_.pending_bytes(segment_graph));
            auto ensure_capacity   = [&]() {
                sync_runtime_residency();
                auto requests = memory_requests(measurement.buffers, new_cache_bytes);
                if (fits(requests, weights.params(index))) {
                    return true;
                }
                if (session_.workspace_.release_excess(measurement)) {
                    sync_runtime_residency();
                    requests = memory_requests(measurement.buffers, new_cache_bytes);
                }
                const bool ready = weights.ensure_segment_capacity(index, requests);
                if (!ready && manager != nullptr) {
                    last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                }
                return ready;
            };
            if (!weights.segment_start(index, ensure_capacity)) {
                return fail_segment("weight preparation");
            }
            // Preparing weights can execute LoRA graphs and reclaim an idle workspace.
            if (!session_.workspace_.prepare(measurement)) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return fail_segment("workspace preparation");
            }
            if (!ensure_capacity()) {
                return fail_segment("workspace capacity check");
            }
            if (!session_.cut_cache_.bind_outputs(cache_outputs)) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return fail_segment("output cache allocation");
            }
            new_cache_bytes = add_bytes(last ? 0 : session_.cut_cache_.estimate_output_bytes(graph, segment), session_.cache_.pending_bytes(segment_graph));
            if (!session_.workspace_.allocate(segment_graph, [&](ggml_backend_sched_t scheduler, ggml_cgraph* current) {
                    pin_multi_device_nodes(scheduler, current);
                })) {
                last_compute_status_ = GGML_STATUS_ALLOC_FAILED;
                return fail_segment("workspace allocation");
            }
            for (const auto& size : measurement.buffers) {
                track_compute_buffer(size.backend);
            }
            if (session_.workspace_.scheduler() != nullptr) {
                track_compute_buffer(session_.workspace_.cpu_backend());
            }
            if (!ensure_capacity()) {
                return fail_segment("allocated capacity check");
            }
            copy_data_to_backend_tensor(segment_graph);
            auto prefetch_requests = memory_requests(measurement.buffers, new_cache_bytes);
            if (!prefetch_requests.empty()) {
                weights.enqueue_next(index, prefetch_requests.front());
            }
            LOG_DEBUG("%s executing segment %zu/%zu: %s", task_.description.c_str(),
                      index + 1, plan.segments.size(), segment.group_name.c_str());
            if (!execute_segment(segment_graph, n_threads, [&]() { weights.submit_next(); })) {
                return fail_segment("execution");
            }
            auto cache_status = session_.cache_.capture(segment_graph);
            if (cache_status == GGML_STATUS_SUCCESS) {
                cache_status = session_.cut_cache_.capture(graph, segment, task_.description.c_str());
            }
            if (cache_status != GGML_STATUS_SUCCESS) {
                last_compute_status_ = cache_status;
                return fail_segment("output caching");
            }
            sync_runtime_residency();
            if (last) {
                if (read_outputs && !read_outputs()) {
                    return fail_segment("output finalization");
                }
                if (!no_return) {
                    output_ = read_graph_tensor(task_.output, "output");
                    if (output_.empty()) {
                        return fail_segment("output readback");
                    }
                }
            }
            if (!session_.workspace_.segment_end()) {
                last_compute_status_ = GGML_STATUS_FAILED;
                return fail_segment("workspace synchronization");
            }
            // Final outputs and their callbacks may still be views of consumed cuts.
            session_.cut_cache_.prune(segment.future_cut_names);
        }
        if (segments_changed || peak_compute_bytes != session_.logged_compute_bytes_) {
            for (const auto& entry : peak_compute_bytes) {
                LOG_VERBOSE("%s compute buffer size: %.2f MB(%s) on %s (peak across %zu segment%s)",
                            task_.description.c_str(), entry.second / (1024.0 * 1024.0),
                            sd_backend_is_cpu(entry.first) ? "RAM" : "VRAM", ggml_backend_name(entry.first),
                            plan.segments.size(), plan.segments.size() == 1 ? "" : "s");
            }
            session_.logged_compute_bytes_ = std::move(peak_compute_bytes);
            session_.logged_segment_count_ = plan.segments.size();
        }
        return true;
    }

    ExecutionSession::ExecutionSession(ggml_backend_t backend, const std::shared_ptr<DeviceResidencyManager>& manager)
        : backend_(backend), residency_manager_(manager), cache_(backend), cut_cache_(backend), workspace_(backend) {
        GGML_ASSERT(backend_ != nullptr);
    }

    size_t ExecutionSession::resident_bytes(ggml_backend_t backend) const {
        size_t bytes = workspace_.bytes(backend);
        if (backend == backend_) {
            const auto add = [](size_t a, size_t b) { return b > SIZE_MAX - a ? SIZE_MAX : a + b; };
            bytes          = add(bytes, cache_.resident_bytes(ggml_backend_get_device(backend)));
            bytes          = add(bytes, cut_cache_.resident_bytes(ggml_backend_get_device(backend)));
        }
        return bytes;
    }

    void ExecutionSession::set_layer_split_enabled(bool enabled) {
        layer_split_enabled_ = enabled;
        if (!enabled) {
            weight_assignments_.clear();
            node_assignments_.clear();
            primary_notice_logged_ = false;
        }
    }

    void ExecutionSession::set_backend_limits(const std::vector<size_t>& limits) {
        backend_limits_ = limits;
        weight_assignments_.clear();
        node_assignments_.clear();
        primary_notice_logged_ = false;
    }

    void ExecutionSession::set_runtime_backends(const std::vector<ggml_backend_t>& backends) {
        extra_backends_.clear();
        for (auto backend : backends) {
            if (backend != nullptr && backend != backend_ &&
                std::find(extra_backends_.begin(), extra_backends_.end(), backend) == extra_backends_.end()) {
                extra_backends_.push_back(backend);
            }
        }
        workspace_.set_extra_backends(extra_backends_);
        weight_assignments_.clear();
        node_assignments_.clear();
        primary_notice_logged_ = false;
    }

    ExecutionEngine::~ExecutionEngine() {
        for (auto& entry : sessions_) {
            end(*entry.second);
        }
    }

    ExecutionSession& ExecutionEngine::create_session(ggml_backend_t backend, const std::shared_ptr<DeviceResidencyManager>& manager) {
        GGML_ASSERT(manager.get() == manager_);
        auto session = std::make_unique<ExecutionSession>(backend, manager);
        auto& result = *session;
        sessions_.emplace(session->owner_id(), std::move(session));
        return result;
    }

    void ExecutionEngine::release_session(ExecutionSession& session) {
        end(session);
        sessions_.erase(session.owner_id());
    }

    bool ExecutionEngine::start(ExecutionSession& session) {
        if (session.started_) {
            return true;
        }
        sd_backend_set_buffer_cache_enabled(session.backend_, true);
        for (auto backend : session.extra_backends_) {
            sd_backend_set_buffer_cache_enabled(backend, true);
        }
        session.workspace_.set_extra_backends(session.extra_backends_);
        if (auto manager = session.residency_manager_.lock()) {
            manager->register_runtime_owner(session.owner_id());
        }
        session.started_ = true;
        return true;
    }

    void ExecutionEngine::end(ExecutionSession& session) {
        GGML_ASSERT(!session.executing_);
        if (!session.started_) {
            return;
        }
        session.workspace_.release();
        session.cache_.clear();
        session.cut_cache_.clear();
        session.logged_compute_bytes_.clear();
        session.logged_segment_count_ = 0;
        if (auto manager = session.residency_manager_.lock()) {
            manager->clear_prefetched_params(session.owner_id());
            std::vector<ggml_tensor*> params(session.params_.begin(), session.params_.end());
            manager->evict_compute_backend_params(params);
            manager->remove_runtime_owner(session.owner_id());
        }
        session.params_.clear();
        sd_backend_clear_buffer_cache(session.backend_);
        for (auto backend : session.extra_backends_) {
            sd_backend_clear_buffer_cache(backend);
        }
        session.started_ = false;
    }

    void ExecutionEngine::sync_residency(ExecutionSession& session) {
        if (auto manager = session.residency_manager_.lock()) {
            manager->update_runtime_residency(session.owner_id(), session.backend_, session.resident_bytes(session.backend_));
            for (auto backend : session.extra_backends_) {
                manager->update_runtime_residency(session.owner_id(), backend, session.resident_bytes(backend));
            }
        }
    }

    bool ExecutionEngine::ensure_capacity(
        DeviceResidencyManager& manager,
        const DeviceMemoryRequest& request,
        const std::vector<ggml_tensor*>& required_params,
        const std::vector<std::vector<ggml_tensor*>>& preferred_eviction_order,
        const std::vector<ggml_tensor*>& protected_params) {
        GGML_ASSERT(&manager == manager_);
        auto capacity      = [&]() { return manager.inspect_compute_backend_capacity(request, required_params); };
        const auto initial = capacity();
        if (!initial.valid) {
            return false;
        }
        if (initial.fits()) {
            return true;
        }
        for (auto& entry : sessions_) {
            auto& session = *entry.second;
            if (session.owner_id() != request.owner_id && session.workspace_.release()) {
                sync_residency(session);
                if (capacity().fits(true)) {
                    return true;
                }
            }
        }
        auto protected_weights = protected_params;
        protected_weights.insert(protected_weights.end(), required_params.begin(), required_params.end());
        auto candidates = manager.evictable_compute_backend_blocks(request.compute_backend, protected_weights);
        std::unordered_map<const ggml_tensor*, size_t> preferred_positions;
        for (size_t index = 0; index < preferred_eviction_order.size(); ++index) {
            for (auto* tensor : preferred_eviction_order[index]) {
                auto* weight = manager.resolve_param_tensor(tensor);
                preferred_positions.emplace(weight != nullptr ? weight : tensor, index);
            }
        }
        auto preferred_position = [&](const WeightResidencyBlock& block) {
            size_t position = 0;
            // A shared allocation follows the last of its members in the eviction order.
            for (auto* tensor : block.tensors) {
                const auto found = preferred_positions.find(tensor);
                if (found == preferred_positions.end()) {
                    return SIZE_MAX;
                }
                position = std::max(position, found->second);
            }
            return position;
        };
        std::stable_sort(candidates.begin(), candidates.end(), [&](const auto& lhs, const auto& rhs) {
            if (lhs.requires_demotion != rhs.requires_demotion) {
                return !lhs.requires_demotion;
            }
            if (!lhs.requires_demotion) {
                const auto left = preferred_position(lhs), right = preferred_position(rhs);
                if (left != right) {
                    return left < right;
                }
                if (left != SIZE_MAX) {
                    return false;
                }
            }
            return lhs.last_use_epoch < rhs.last_use_epoch;
        });
        for (const auto& block : candidates) {
            if (manager.evict_compute_backend_block(block) && capacity().fits(true)) {
                return true;
            }
        }
        auto available = capacity();
        if (available.fits(true)) {
            return true;
        }
        if (sd_backend_is(request.compute_backend, "Vulkan") &&
            available.required_budget_bytes <= available.available_budget_bytes) {
            ggml_backend_synchronize(request.compute_backend);
            sd_backend_clear_buffer_cache(request.compute_backend);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
            do {
                available = capacity();
                if (available.fits(true)) {
                    return true;
                }
                if (available.required_budget_bytes > available.available_budget_bytes) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now() < deadline);
        }
        available               = manager.inspect_compute_backend_capacity(request, required_params, true);
        const auto device_bytes = available.available_device_bytes == SIZE_MAX
                                      ? std::string("unknown")
                                      : sd_format("%.2f MB", available.available_device_bytes / (1024.0 * 1024.0));
        const auto budget_bytes = available.available_budget_bytes == SIZE_MAX
                                      ? std::string("unlimited")
                                      : sd_format("%.2f MB", available.available_budget_bytes / (1024.0 * 1024.0));
        LOG_WARN("execution engine cannot make enough memory available on %s: need %.2f MB device / %.2f MB budget, available %s device / %s budget",
                 ggml_backend_name(request.compute_backend),
                 available.required_device_bytes / (1024.0 * 1024.0),
                 available.required_budget_bytes / (1024.0 * 1024.0), device_bytes.c_str(), budget_bytes.c_str());
        return false;
    }

    void ExecutionEngine::clear_cache(ExecutionSession& session) {
        session.cache_.clear();
        sync_residency(session);
    }

    void ExecutionEngine::discard_graph(ExecutionSession& session) {
        GGML_ASSERT(!session.executing_);
        session.cache_.graph_end(false);
        session.cut_cache_.clear();
        session.workspace_.release();
        sync_residency(session);
    }

    ComputeResult ExecutionEngine::execute(ExecutionSession& session, const ComputeTask& task) {
        if (session.executing_ || task.graph == nullptr || !start(session)) {
            return {};
        }
        session.executing_ = true;
        ComputeResult result;
        try {
            GraphExecution execution(*this, session, task);
            result = execution.run();
        } catch (const std::bad_alloc&) {
            result.status = GGML_STATUS_ALLOC_FAILED;
            LOG_ERROR("%s graph allocation failed", task.description.c_str());
        } catch (const std::exception& error) {
            LOG_ERROR("%s graph execution failed on %s: %s", task.description.c_str(),
                      ggml_backend_name(session.backend_), error.what());
        } catch (...) {
            LOG_ERROR("%s graph execution failed on %s: unknown exception", task.description.c_str(),
                      ggml_backend_name(session.backend_));
        }
        if (!session.workspace_.segment_end()) {
            result.status = GGML_STATUS_FAILED;
        }
        const bool success = result.status == GGML_STATUS_SUCCESS;
        session.cache_.graph_end(success);
        session.cut_cache_.clear();
        if (!success) {
            result.output = {};
            session.workspace_.release();
        }
        session.executing_ = false;
        sync_residency(session);
        return result;
    }
}

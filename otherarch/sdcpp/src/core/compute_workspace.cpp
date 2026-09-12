#include "core/compute_workspace.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/ggml_graph_cut.h"
#include "ggml-cpu.h"
#include "ggml/src/ggml-impl.h"

namespace sd {
    ComputeWorkspace::~ComputeWorkspace() {
        segment_end();
        release();
        ggml_backend_free(cpu_backend_);
    }

    void ComputeWorkspace::set_extra_backends(const std::vector<ggml_backend_t>& backends) {
        if (extra_backends_ != backends) {
            GGML_ASSERT(!active_);
            release();
            extra_backends_ = backends;
        }
    }

    bool ComputeWorkspace::needs_scheduler(ggml_cgraph* graph) const {
        if (!extra_backends_.empty()) {
            return true;
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend_, ggml_graph_node(graph, i))) {
                return true;
            }
        }
        return false;
    }

    ggml_backend_sched_t ComputeWorkspace::make_scheduler(size_t graph_size) {
        std::vector<ggml_backend_t> backends{backend_};
        backends.insert(backends.end(), extra_backends_.begin(), extra_backends_.end());
        if (!sd_backend_is_cpu(backend_)) {
            if (cpu_backend_ == nullptr) {
                cpu_backend_ = sd_backend_cpu_init();
            }
            if (cpu_backend_ == nullptr) {
                return nullptr;
            }
            backends.push_back(cpu_backend_);
        }
        std::vector<ggml_backend_buffer_type_t> bufts;
        for (auto backend : backends) {
            auto buft = backend == cpu_backend_
                            ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_))
                            : nullptr;
            bufts.push_back(buft != nullptr ? buft : ggml_backend_get_default_buffer_type(backend));
        }
        return ggml_backend_sched_new(backends.data(), bufts.data(), static_cast<int>(backends.size()),
                                      graph_size, false, false);
    }

    bool ComputeWorkspace::measurement_matches(ggml_cgraph* graph, const Measurement& measurement) const {
        return measurement.scheduler == needs_scheduler(graph);
    }

    bool ComputeWorkspace::prepare(const Measurement& measurement) {
        GGML_ASSERT(!active_);
        if (measurement.buffers.empty()) {
            return false;
        }
        const bool grows = std::any_of(measurement.buffers.begin(), measurement.buffers.end(),
                                       [&](const BackendBufferSize& size) { return size.bytes > bytes(size.backend); });
        if (measurement.scheduler != (scheduler_ != nullptr) || grows) {
            release();
        }
        return true;
    }

    bool ComputeWorkspace::release_excess(const Measurement& measurement) {
        return std::any_of(measurement.buffers.begin(), measurement.buffers.end(),
                           [&](const BackendBufferSize& size) { return bytes(size.backend) > size.bytes; }) &&
               release();
    }

    bool ComputeWorkspace::allocate(ggml_cgraph* graph, const AssignNodes& assign_nodes) {
        GGML_ASSERT(!active_);
        const bool use_scheduler = needs_scheduler(graph);
        if (use_scheduler) {
            if (allocator_ != nullptr) {
                release();
            }
            const size_t capacity = static_cast<size_t>(graph->n_nodes + graph->n_leafs) + 8;
            if (scheduler_ == nullptr || capacity > scheduler_capacity_) {
                release();
                scheduler_          = make_scheduler(capacity);
                scheduler_capacity_ = capacity;
            }
            if (scheduler_ == nullptr) {
                return false;
            }
            ggml_backend_sched_reset(scheduler_);
            assign_nodes(scheduler_, graph);
            // Scheduler allocation rewrites sources. Split the execution graph only once.
            if (!ggml_backend_sched_alloc_graph(scheduler_, graph)) {
                release();
                return false;
            }
        } else {
            if (scheduler_ != nullptr) {
                release();
            }
            if (allocator_ == nullptr) {
                allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            }
            auto signature = ggml_graph_cut::graph_layout(graph, true);
            if (signature != reservation_) {
                if (!ggml_gallocr_reserve(allocator_, graph)) {
                    release();
                    return false;
                }
                reservation_ = std::move(signature);
                ++reservations_;
            }
            if (!ggml_gallocr_alloc_graph(allocator_, graph)) {
                release();
                return false;
            }
        }
        active_ = true;
        return true;
    }

    ComputeWorkspace::Measurement ComputeWorkspace::measure(
        ggml_cgraph* graph,
        size_t direct_bytes,
        const std::function<ggml_backend_t(const ggml_tensor*)>& external_backend,
        const AssignNodes& assign_nodes) {
        if (!needs_scheduler(graph)) {
            return {{{backend_, direct_bytes}}, false};
        }
        std::vector<const ggml_tensor*> tensors;
        std::unordered_set<const ggml_tensor*> seen;
        auto visit = [&](const ggml_tensor* tensor) {
            if (tensor != nullptr && seen.insert(tensor).second) {
                tensors.push_back(tensor);
            }
        };
        for (int i = 0; i < graph->n_nodes; ++i) {
            visit(graph->nodes[i]);
        }
        for (int i = 0; i < graph->n_leafs; ++i) {
            visit(graph->leafs[i]);
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            visit(tensors[i]->view_src);
            for (auto source : tensors[i]->src) {
                visit(source);
            }
        }
        const size_t graph_size = tensors.size() + 8;
        auto context            = ggml_init({tensors.size() * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
        if (context == nullptr) {
            return {};
        }
        std::unordered_map<const ggml_tensor*, ggml_tensor*> copies;
        std::map<ggml_backend_t, ggml_backend_buffer_t> external_buffers;
        for (auto tensor : tensors) {
            auto copy      = ggml_dup_tensor(context, tensor);
            *copy          = *tensor;
            copies[tensor] = copy;
        }
        for (const auto& entry : copies) {
            auto source    = entry.first;
            auto copy      = entry.second;
            copy->view_src = source->view_src == nullptr ? nullptr : copies.at(source->view_src);
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                copy->src[i] = source->src[i] == nullptr ? nullptr : copies.at(source->src[i]);
            }
            auto external = external_backend(source);
            if (external != nullptr && source->view_src == nullptr) {
                auto& buffer = external_buffers[external];
                if (buffer == nullptr) {
                    buffer = ggml_backend_alloc_buffer(external, 0);
                    GGML_ASSERT(buffer != nullptr);
                    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                }
                copy->buffer = buffer;
                copy->data   = reinterpret_cast<void*>(static_cast<uintptr_t>(1));
                copy->extra  = nullptr;
            }
        }
        auto copy_graph     = ggml_new_graph_custom(context, graph_size, false);
        copy_graph->n_nodes = graph->n_nodes;
        copy_graph->n_leafs = graph->n_leafs;
        for (int i = 0; i < graph->n_nodes; ++i) {
            copy_graph->nodes[i] = copies.at(graph->nodes[i]);
        }
        for (int i = 0; i < graph->n_leafs; ++i) {
            copy_graph->leafs[i] = copies.at(graph->leafs[i]);
        }
        Measurement result;
        result.scheduler = true;
        auto scheduler   = make_scheduler(graph_size);
        if (scheduler != nullptr) {
            assign_nodes(scheduler, copy_graph);
            std::vector<size_t> sizes(extra_backends_.size() + 2);
            ggml_backend_sched_reserve_size(scheduler, copy_graph, sizes.data());
            result.buffers.push_back({backend_, sizes[0]});
            for (size_t i = 0; i < extra_backends_.size(); ++i) {
                result.buffers.push_back({extra_backends_[i], sizes[i + 1]});
            }
            ggml_backend_sched_free(scheduler);
        }
        for (const auto& entry : external_buffers) {
            ggml_backend_buffer_free(entry.second);
        }
        ggml_free(context);
        return result;
    }

    void ComputeWorkspace::synchronize() const {
        if (scheduler_ != nullptr) {
            ggml_backend_sched_synchronize(scheduler_);
        } else {
            ggml_backend_synchronize(backend_);
        }
    }

    void ComputeWorkspace::segment_end() {
        if (active_) {
            synchronize();
            active_ = false;
        }
    }

    bool ComputeWorkspace::release() {
        if (active_) {
            return false;
        }
        ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
        ggml_backend_sched_free(scheduler_);
        scheduler_          = nullptr;
        scheduler_capacity_ = 0;
        reservation_.clear();
        return true;
    }

    size_t ComputeWorkspace::bytes(ggml_backend_t backend) const {
        if (scheduler_ != nullptr) {
            return ggml_backend_sched_get_buffer_size(scheduler_, backend);
        }
        return allocator_ != nullptr && backend == backend_ ? ggml_gallocr_get_buffer_size(allocator_, 0) : 0;
    }
}

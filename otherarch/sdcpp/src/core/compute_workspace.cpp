#include "core/compute_workspace.h"

#include <algorithm>
#include <exception>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "core/ggml_extend_backend.h"
#include "core/util.h"
#include "ggml-impl.h"

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

    bool ComputeWorkspace::prepare(const Measurement& measurement) {
        GGML_ASSERT(!active_);
        if (measurement.buffers.empty()) {
            return false;
        }
        const bool grows = std::any_of(measurement.buffers.begin(), measurement.buffers.end(),
                                       [&](const BackendBufferSize& size) { return size.bytes > bytes(size.backend); });
        if (grows) {
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
        active_ = true;
        return true;
    }

    ComputeWorkspace::Measurement ComputeWorkspace::measure(
        ggml_cgraph* graph,
        const std::function<ggml_backend_t(const ggml_tensor*)>& external_backend,
        const AssignNodes& assign_nodes,
        bool cache_measurement) {
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
        std::vector<uint8_t> key;
        if (cache_measurement) {
            key.reserve(tensors.size() * sizeof(ggml_tensor));
            auto append = [&](const auto& value) {
                const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
                key.insert(key.end(), bytes, bytes + sizeof(value));
            };
            append(graph->n_nodes);
            append(graph->n_leafs);
            append(graph->order);
            for (auto tensor : tensors) {
                append(tensor);
                append(tensor->type);
                append(tensor->ne);
                append(tensor->nb);
                append(tensor->op);
                append(tensor->op_params);
                append(tensor->flags);
                append(tensor->src);
                append(tensor->view_src);
                append(tensor->view_offs);
                append(tensor->name);
                const auto external = external_backend(tensor);
                append(external);
                if (external == nullptr || tensor->view_src != nullptr) {
                    append(tensor->buffer);
                    append(tensor->data);
                    append(tensor->extra);
                    if (tensor->buffer != nullptr) {
                        const auto usage = ggml_backend_buffer_get_usage(tensor->buffer);
                        const auto type  = ggml_backend_buffer_get_type(tensor->buffer);
                        append(usage);
                        append(type);
                    }
                }
                const size_t index = ggml_hash_find(&graph->visited_hash_set, tensor);
                const int32_t uses = index != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, index)
                                         ? graph->use_counts[index]
                                         : 0;
                append(uses);
            }
            // Match the full allocation metadata, including topology and external placement.
            if (key == measurement_key_) {
                return {measured_buffers_};
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
        copy_graph->order = graph->order;
        // Preserve fusion metadata, including uses outside a graph view, without rebuilding its node order.
        for (const auto& entry : copies) {
            const size_t source_index = ggml_hash_find(&graph->visited_hash_set, entry.first);
            if (source_index != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, source_index)) {
                const size_t copy_index            = ggml_hash_insert(&copy_graph->visited_hash_set, entry.second);
                copy_graph->use_counts[copy_index] = graph->use_counts[source_index];
            }
        }
        Measurement result;
        auto scheduler = make_scheduler(graph_size);
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
        if (cache_measurement && !result.buffers.empty()) {
            measurement_key_  = std::move(key);
            measured_buffers_ = result.buffers;
        }
        return result;
    }

    void ComputeWorkspace::synchronize() const {
        if (scheduler_ != nullptr) {
            ggml_backend_sched_synchronize(scheduler_);
        } else {
            ggml_backend_synchronize(backend_);
        }
    }

    bool ComputeWorkspace::segment_end() noexcept {
        if (!active_) {
            return true;
        }
        // Outer cleanup guards must not retry a failed backend submission.
        active_ = false;
        try {
            synchronize();
            return true;
        } catch (const std::exception& error) {
            LOG_ERROR("%s workspace synchronization failed during segment cleanup: %s",
                      ggml_backend_name(backend_), error.what());
        } catch (...) {
            LOG_ERROR("%s workspace synchronization failed during segment cleanup: unknown exception",
                      ggml_backend_name(backend_));
        }
        return false;
    }

    bool ComputeWorkspace::release() {
        if (active_) {
            return false;
        }
        measurement_key_.clear();
        measured_buffers_.clear();
        ggml_backend_sched_free(scheduler_);
        scheduler_          = nullptr;
        scheduler_capacity_ = 0;
        return true;
    }

    size_t ComputeWorkspace::bytes(ggml_backend_t backend) const {
        if (scheduler_ != nullptr) {
            return ggml_backend_sched_get_buffer_size(scheduler_, backend);
        }
        return 0;
    }
}

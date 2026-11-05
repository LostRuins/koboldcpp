#include "core/ggml_compat.h"
#include "core/ggml_extend_backend.h"

#ifdef SD_GGML_GRAPH_COMPAT
// Upstream ggml still requires its private graph layout for these operations.
#include "ggml-impl.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

int ggml_graph_n_leafs(const ggml_cgraph* graph) {
    return graph->n_leafs;
}

ggml_tensor* ggml_graph_leaf(const ggml_cgraph* graph, int index) {
    if (index < 0) {
        index += graph->n_leafs;
    }
    GGML_ASSERT(index >= 0 && index < graph->n_leafs);
    return graph->leafs[index];
}

void ggml_graph_add_leaf(ggml_cgraph* graph, ggml_tensor* tensor) {
    GGML_ASSERT(tensor && graph->n_leafs < graph->size);
    graph->leafs[graph->n_leafs++] = tensor;
}

int32_t ggml_graph_use_count(const ggml_cgraph* graph, const ggml_tensor* tensor) {
    if (!graph || !tensor || !graph->visited_hash_set.size) {
        return 0;
    }
    const size_t index = ggml_hash_find(&graph->visited_hash_set, tensor);
    return index != GGML_HASHSET_FULL && ggml_bitset_get(graph->visited_hash_set.used, index)
               ? graph->use_counts[index]
               : 0;
}

bool ggml_graph_cpy_mapped(const ggml_cgraph* src, ggml_cgraph* dst, ggml_graph_tensor_map_fn map, void* user_data) {
    if (!src || !dst || src == dst || !map || dst->grads || dst->grad_accs ||
        src->n_nodes > dst->size || src->n_leafs > dst->size) {
        return false;
    }
    std::unordered_map<const ggml_tensor*, ggml_tensor*> mapping;
    std::unordered_set<ggml_tensor*> mapped;
    std::vector<const ggml_tensor*> sources;
    bool valid = true;
    auto remap = [&](const ggml_tensor* source) {
        const auto found = mapping.find(source);
        if (found != mapping.end()) {
            return found->second;
        }
        auto* tensor = map(source, user_data);
        if (tensor && !mapped.insert(tensor).second) {
            valid = false;
        }
        mapping.emplace(source, tensor);
        sources.push_back(source);
        return tensor;
    };
    std::vector<ggml_tensor*> nodes(src->n_nodes), leafs(src->n_leafs);
    for (int i = 0; i < src->n_nodes; ++i) {
        if (!(nodes[i] = remap(src->nodes[i])) || !valid) {
            return false;
        }
    }
    for (int i = 0; i < src->n_leafs; ++i) {
        if (!(leafs[i] = remap(src->leafs[i])) || !valid) {
            return false;
        }
    }
    std::vector<std::pair<ggml_tensor*, int32_t>> counts;
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto* source = sources[i];
        if (source->view_src) {
            remap(source->view_src);
        }
        for (const auto* input : source->src) {
            if (input) {
                remap(input);
            }
        }
        if (!valid) {
            return false;
        }
        auto* tensor = mapping.at(source);
        if (!tensor || !src->visited_hash_set.size) {
            continue;
        }
        const size_t index = ggml_hash_find(&src->visited_hash_set, source);
        if (index == GGML_HASHSET_FULL || !ggml_bitset_get(src->visited_hash_set.used, index)) {
            continue;
        }
        if (counts.size() >= dst->visited_hash_set.size) {
            return false;
        }
        counts.emplace_back(tensor, src->use_counts[index]);
    }
    ggml_graph_clear(dst);
    dst->order = src->order;
    dst->uid   = 0;
    for (auto* tensor : nodes) {
        ggml_graph_add_node(dst, tensor);
    }
    for (auto* tensor : leafs) {
        ggml_graph_add_leaf(dst, tensor);
    }
    for (const auto& entry : counts) {
        const size_t index     = ggml_hash_insert(&dst->visited_hash_set, entry.first);
        dst->use_counts[index] = entry.second;
    }
    return true;
}

bool ggml_graph_cpy_use_counts(const ggml_cgraph* src, ggml_cgraph* dst) {
    if (!src || !dst || !src->visited_hash_set.size || !dst->visited_hash_set.size) {
        return false;
    }
    std::unordered_set<ggml_tensor*> represented;
    for (int i = 0; i < dst->n_nodes; ++i) {
        represented.insert(dst->nodes[i]);
    }
    for (int i = 0; i < dst->n_leafs; ++i) {
        represented.insert(dst->leafs[i]);
    }
    size_t occupied = 0;
    for (size_t i = 0; i < dst->visited_hash_set.size; ++i) {
        occupied += ggml_bitset_get(dst->visited_hash_set.used, i);
    }
    std::vector<std::pair<ggml_tensor*, int32_t>> counts;
    for (auto* tensor : represented) {
        if (!tensor) {
            return false;
        }
        const size_t source_index = ggml_hash_find(&src->visited_hash_set, tensor);
        if (source_index == GGML_HASHSET_FULL || !ggml_bitset_get(src->visited_hash_set.used, source_index)) {
            continue;
        }
        const size_t index = ggml_hash_find(&dst->visited_hash_set, tensor);
        if (index == GGML_HASHSET_FULL || !ggml_bitset_get(dst->visited_hash_set.used, index)) {
            if (++occupied > dst->visited_hash_set.size) {
                return false;
            }
        }
        counts.emplace_back(tensor, src->use_counts[source_index]);
    }
    for (const auto& entry : counts) {
        const size_t index     = ggml_hash_find_or_insert(&dst->visited_hash_set, entry.first);
        dst->use_counts[index] = entry.second;
    }
    return true;
}

ggml_status ggml_backend_graph_compute_async_range(ggml_backend_t backend, ggml_cgraph* graph, int first, int last) {
    if (!backend || !graph || first < 0 || last < first || last > graph->n_nodes) {
        return GGML_STATUS_FAILED;
    }
    if (first == last) {
        return GGML_STATUS_SUCCESS;
    }
    auto view = ggml_graph_view(graph, first, last);
    return ggml_backend_graph_compute_async(backend, &view);
}
#endif

#ifdef SD_USE_UPSTREAM_GGML
#include "ggml-backend-impl.h"
#endif

ggml_backend_buffer_t sd_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void* ptr, size_t size, size_t max_tensor_size) {
    auto buffer = ggml_backend_dev_buffer_from_host_ptr(device, ptr, size, max_tensor_size);
#ifdef SD_USE_UPSTREAM_GGML
    if (buffer && !buffer->context) {
        auto reg = ggml_backend_dev_backend_reg(device);
        if (reg && std::strcmp(ggml_backend_reg_name(reg), "Metal") == 0) {
            // Older Metal backends wrap failed mappings and cannot free their null context.
            buffer->iface.free_buffer = nullptr;
            ggml_backend_buffer_free(buffer);
            return nullptr;
        }
    }
#endif
    return buffer;
}

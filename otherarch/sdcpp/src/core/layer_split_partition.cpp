#include "core/layer_split_partition.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>

#include "core/util.h"

namespace sd {

    static bool layer_split_path_segment_starts_at(const std::string& name, size_t pos) {
        return pos == 0 || name[pos - 1] == '.';
    }

    static bool layer_split_has_path_segment(const std::string& name, const char* segment) {
        size_t pos = name.find(segment);
        while (pos != std::string::npos) {
            if (layer_split_path_segment_starts_at(name, pos)) {
                return true;
            }
            pos = name.find(segment, pos + 1);
        }
        return false;
    }

    int layer_split_tensor_block_index(const std::string& name) {
        static const char* unet_block_segments[] = {"input_blocks.", "output_blocks.", "middle_block.",
                                                    "down_blocks.", "up_blocks.", "mid_block."};
        for (const char* segment : unet_block_segments) {
            if (layer_split_has_path_segment(name, segment)) {
                return -1;
            }
        }

        static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                               "single_blocks.", "blocks.", "block.", "layers."};
        for (const char* keyword : block_keywords) {
            size_t pos = name.find(keyword);
            while (pos != std::string::npos) {
                if (!layer_split_path_segment_starts_at(name, pos)) {
                    pos = name.find(keyword, pos + 1);
                    continue;
                }
                pos += std::strlen(keyword);
                size_t end = pos;
                while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
                    end++;
                }
                if (end > pos && (end == name.size() || name[end] == '.')) {
                    return std::atoi(name.substr(pos, end - pos).c_str());
                }
                break;
            }
        }
        return -1;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "unknown";
    }

    static size_t graph_cut_layer_split_backend_vram_limit(const std::vector<size_t>& backend_vram_limits,
                                                           size_t backend_index,
                                                           size_t primary_backend_vram_limit) {
        if (backend_index < backend_vram_limits.size()) {
            return backend_vram_limits[backend_index];
        }
        return backend_index == 0 ? primary_backend_vram_limit : 0;
    }

    static std::vector<int64_t> graph_cut_layer_split_backend_capacities(const std::vector<ggml_backend_t>& backends,
                                                                         const std::vector<size_t>& backend_vram_limits,
                                                                         size_t primary_backend_vram_limit) {
        std::vector<int64_t> capacities(backends.size(), std::numeric_limits<int64_t>::max() / 4);
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        for (size_t i = 0; i < backends.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
            size_t free_bytes = 0, total_bytes = 0;
            if (dev != nullptr) {
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            }
            if (free_bytes > 0) {
                capacities[i] = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes, 0);
            }
            size_t limit_bytes = graph_cut_layer_split_backend_vram_limit(backend_vram_limits,
                                                                          i,
                                                                          primary_backend_vram_limit);
            if (limit_bytes > 0) {
                capacities[i] = std::min<int64_t>(capacities[i], (int64_t)limit_bytes);
            }
        }
        return capacities;
    }

    bool partition_graph_cut_layer_split(const char* desc,
                                         ggml_cgraph* gf,
                                         const sd::ggml_graph_cut::Plan& plan,
                                         const std::vector<ggml_backend_t>& split_backends,
                                         const std::vector<size_t>& backend_vram_limits,
                                         size_t primary_backend_vram_limit,
                                         std::unordered_map<const ggml_tensor*, ggml_backend_t>& param_assignments,
                                         const std::function<ggml_tensor*(ggml_tensor*)>& canonical_param_tensor,
                                         GraphCutLayerSplitAssignment* assignment_out) {
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(assignment_out != nullptr);
        GGML_ASSERT(canonical_param_tensor != nullptr);
        GGML_ASSERT(!split_backends.empty());

        GraphCutLayerSplitAssignment assignment;
        assignment.segment_count = plan.segments.size();
        assignment.tensors_by_backend.resize(split_backends.size());
        assignment.bytes_by_backend.resize(split_backends.size(), 0);
        assignment.first_segment_by_backend.resize(split_backends.size(), plan.segments.size());
        assignment.last_segment_by_backend.resize(split_backends.size(), 0);

        std::vector<std::vector<ggml_tensor*>> segment_params(plan.segments.size());
        std::vector<int64_t> segment_param_bytes(plan.segments.size(), 0);
        std::unordered_set<ggml_tensor*> seen_params;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            std::vector<ggml_tensor*> params = sd::ggml_graph_cut::param_tensors(gf, plan.segments[seg_idx]);
            for (ggml_tensor* raw_param : params) {
                ggml_tensor* param = canonical_param_tensor(raw_param);
                if (param == nullptr || !seen_params.insert(param).second) {
                    continue;
                }
                segment_params[seg_idx].push_back(param);
                segment_param_bytes[seg_idx] += (int64_t)ggml_nbytes(param);
            }
        }

        int64_t total_param_bytes = 0;
        for (int64_t bytes : segment_param_bytes) {
            total_param_bytes += bytes;
        }
        if (total_param_bytes <= 0) {
            LOG_ERROR("%s graph-cut layer split found no graph params to assign", desc);
            return false;
        }

        std::vector<int64_t> backend_capacities = graph_cut_layer_split_backend_capacities(split_backends,
                                                                                           backend_vram_limits,
                                                                                           primary_backend_vram_limit);

        std::vector<ggml_backend_t> backend_by_segment(plan.segments.size(), split_backends[0]);
        size_t current_backend = 0;
        int64_t current_used   = 0;
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            int64_t bytes = segment_param_bytes[seg_idx];
            while (current_backend + 1 < split_backends.size() &&
                   bytes > 0 &&
                   current_used + bytes > backend_capacities[current_backend]) {
                current_backend++;
                current_used = 0;
            }
            if (bytes > 0 && current_used + bytes > backend_capacities[current_backend]) {
                LOG_ERROR("%s graph-cut layer split: segment %zu needs %.1f MB on %s, but only %.1f MB is available under current VRAM limits",
                          desc,
                          seg_idx,
                          (current_used + bytes) / (1024.0 * 1024.0),
                          layer_split_backend_device_display_name(split_backends[current_backend]).c_str(),
                          backend_capacities[current_backend] / (1024.0 * 1024.0));
                return false;
            }
            current_used += bytes;
            backend_by_segment[seg_idx] = split_backends[current_backend];

            for (ggml_tensor* param : segment_params[seg_idx]) {
                ggml_backend_t target_backend = split_backends[current_backend];
                auto assigned_it              = param_assignments.find(param);
                if (assigned_it == param_assignments.end()) {
                    param_assignments[param]            = target_backend;
                    assignment.has_new_param_assignment = true;
                } else {
                    target_backend = assigned_it->second;
                }

                auto backend_it = std::find(split_backends.begin(), split_backends.end(), target_backend);
                if (backend_it == split_backends.end()) {
                    LOG_ERROR("%s graph-cut layer split tensor '%s' is assigned to an unavailable backend",
                              desc,
                              ggml_get_name(param));
                    return false;
                }
                size_t backend_idx                               = (size_t)std::distance(split_backends.begin(), backend_it);
                assignment.first_segment_by_backend[backend_idx] = std::min(assignment.first_segment_by_backend[backend_idx], seg_idx);
                assignment.last_segment_by_backend[backend_idx]  = std::max(assignment.last_segment_by_backend[backend_idx], seg_idx + 1);
                assignment.tensors_by_backend[backend_idx].push_back(param);
                assignment.bytes_by_backend[backend_idx] += (int64_t)ggml_nbytes(param);
            }
        }

        const int n_nodes = ggml_graph_n_nodes(gf);
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); seg_idx++) {
            ggml_backend_t backend = backend_by_segment[seg_idx];
            const auto& segment    = plan.segments[seg_idx];
            for (int node_index : segment.internal_node_indices) {
                if (node_index < 0 || node_index >= n_nodes) {
                    continue;
                }
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (node != nullptr) {
                    assignment.node_assignments[node] = backend;
                }
            }
            for (int node_index : segment.output_node_indices) {
                if (node_index < 0 || node_index >= n_nodes) {
                    continue;
                }
                ggml_tensor* node = ggml_graph_node(gf, node_index);
                if (node != nullptr) {
                    assignment.node_assignments[node] = backend;
                }
            }
        }

        *assignment_out = std::move(assignment);
        return true;
    }

    void log_graph_cut_layer_split_assignment(const char* desc,
                                              const std::vector<ggml_backend_t>& split_backends,
                                              const GraphCutLayerSplitAssignment& assignment) {
        for (size_t i = 0; i < split_backends.size(); i++) {
            if (i >= assignment.tensors_by_backend.size() ||
                assignment.tensors_by_backend[i].empty()) {
                continue;
            }
            size_t first_segment = assignment.first_segment_by_backend[i] == assignment.segment_count
                                       ? 0
                                       : assignment.first_segment_by_backend[i];
            size_t last_segment  = assignment.last_segment_by_backend[i];
            if (assignment.has_new_param_assignment) {
                LOG_INFO("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                         desc,
                         layer_split_backend_device_display_name(split_backends[i]).c_str(),
                         first_segment,
                         last_segment,
                         assignment.tensors_by_backend[i].size(),
                         assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            } else {
                LOG_DEBUG("%s graph-cut layer split: %s <- segments [%zu, %zu), %zu tensors, %.1f MB",
                          desc,
                          layer_split_backend_device_display_name(split_backends[i]).c_str(),
                          first_segment,
                          last_segment,
                          assignment.tensors_by_backend[i].size(),
                          assignment.bytes_by_backend[i] / (1024.0 * 1024.0));
            }
        }
    }

}  // namespace sd

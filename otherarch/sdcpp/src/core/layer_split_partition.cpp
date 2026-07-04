#include "core/layer_split_partition.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "core/util.h"

namespace sd {

    int layer_split_tensor_block_index(const std::string& name) {
        static const char* block_keywords[] = {"transformer_blocks.", "joint_blocks.", "double_blocks.",
                                               "single_blocks.", "blocks.", "block.", "layers."};
        for (const char* keyword : block_keywords) {
            size_t pos = name.find(keyword);
            if (pos == std::string::npos) {
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
        }
        return -1;
    }

    std::string layer_split_backend_device_display_name(ggml_backend_t backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const char* name       = dev != nullptr ? ggml_backend_dev_name(dev) : ggml_backend_name(backend);
        return name != nullptr ? name : "unknown";
    }

    static bool layer_split_backend_supports_tensor(ggml_backend_t backend, const ggml_tensor* tensor) {
        return backend != nullptr && tensor != nullptr && ggml_backend_supports_op(backend, tensor);
    }

    static size_t layer_split_supported_target(const std::string& desc,
                                               const std::string& tensor_name,
                                               const ggml_tensor* tensor,
                                               const std::vector<ggml_backend_t>& backends,
                                               size_t preferred) {
        if (tensor == nullptr || backends.empty()) {
            return preferred;
        }
        size_t preferred_safe = std::min(preferred, backends.size() - 1);
        if (layer_split_backend_supports_tensor(backends[preferred_safe], tensor)) {
            return preferred_safe;
        }
        for (size_t i = 0; i < backends.size(); i++) {
            if (layer_split_backend_supports_tensor(backends[i], tensor)) {
                LOG_WARN("%s layer split: moving tensor '%s' from %s to %s because the preferred backend cannot run op=%s type=%s nbytes=%.2f MB",
                         desc.c_str(),
                         tensor_name.c_str(),
                         layer_split_backend_device_display_name(backends[preferred_safe]).c_str(),
                         layer_split_backend_device_display_name(backends[i]).c_str(),
                         ggml_op_name(tensor->op),
                         ggml_type_name(tensor->type),
                         ggml_nbytes(tensor) / (1024.0 * 1024.0));
                return i;
            }
        }
        LOG_WARN("%s layer split: tensor '%s' is not supported by any split backend: op=%s type=%s nbytes=%.2f MB",
                 desc.c_str(),
                 tensor_name.c_str(),
                 ggml_op_name(tensor->op),
                 ggml_type_name(tensor->type),
                 ggml_nbytes(tensor) / (1024.0 * 1024.0));
        return preferred_safe;
    }

    std::vector<std::map<std::string, ggml_tensor*>> partition_layer_split_tensors(
        const std::string& desc,
        const std::map<std::string, ggml_tensor*>& tensors,
        const std::map<std::string, ggml_tensor*>& split_tensors,
        const std::vector<ggml_backend_t>& backends) {
        std::vector<std::map<std::string, ggml_tensor*>> partitions(backends.size());
        if (backends.empty()) {
            LOG_WARN("%s: no backend available for a layer split", desc.c_str());
            return partitions;
        }

        std::map<int, int64_t> block_bytes;
        std::map<std::string, size_t> non_block_targets;
        std::vector<int64_t> other_bytes_by_backend(backends.size(), 0);
        int64_t total_block_bytes = 0;
        int64_t total_other_bytes = 0;
        int n_blocks              = 0;
        for (const auto& kv : tensors) {
            int64_t bytes = (int64_t)ggml_nbytes(kv.second);
            int idx       = split_tensors.count(kv.first) != 0 ? layer_split_tensor_block_index(kv.first) : -1;
            if (idx >= 0) {
                block_bytes[idx] += bytes;
                total_block_bytes += bytes;
                n_blocks = std::max(n_blocks, idx + 1);
            } else {
                size_t target               = layer_split_supported_target(desc, kv.first, kv.second, backends, 0);
                non_block_targets[kv.first] = target;
                other_bytes_by_backend[target] += bytes;
                total_other_bytes += bytes;
            }
        }
        if (n_blocks == 0) {
            LOG_WARN("%s: no transformer blocks found for a layer split; keeping tensors on compatible backends starting from %s",
                     desc.c_str(),
                     layer_split_backend_device_display_name(backends[0]).c_str());
            for (const auto& kv : tensors) {
                size_t target  = 0;
                auto target_it = non_block_targets.find(kv.first);
                if (target_it != non_block_targets.end()) {
                    target = target_it->second;
                }
                partitions[target][kv.first] = kv.second;
            }
            return partitions;
        }

        // Reserve compute headroom and subtract each device's actual non-block
        // bytes from its block budget.
        constexpr int64_t compute_headroom_bytes = 2ll * 1024 * 1024 * 1024;
        std::vector<double> device_weights(backends.size(), 1.0);
        double weight_sum = 0.0;
        for (size_t i = 0; i < backends.size(); i++) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backends[i]);
            size_t free_bytes = 0, total_bytes = 0;
            if (dev != nullptr) {
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            }
            // Keep a small share even for tight devices instead of dropping them.
            int64_t usable_bytes = std::max<int64_t>((int64_t)free_bytes - compute_headroom_bytes,
                                                     (int64_t)free_bytes / 8);
            device_weights[i]    = usable_bytes > 0 ? (double)usable_bytes : 1.0;
            weight_sum += device_weights[i];
        }

        std::vector<int64_t> block_budgets(backends.size(), 0);
        const int64_t total_bytes = total_block_bytes + total_other_bytes;
        for (size_t i = 0; i < backends.size(); i++) {
            int64_t budget   = (int64_t)((double)total_bytes * device_weights[i] / weight_sum);
            budget           = std::max<int64_t>(budget - other_bytes_by_backend[i], 0);
            block_budgets[i] = budget;
        }

        std::vector<int> boundaries(backends.size(), n_blocks);
        size_t current = 0;
        int64_t used   = 0;
        for (int b = 0; b < n_blocks; b++) {
            int64_t bytes = block_bytes.count(b) != 0 ? block_bytes[b] : 0;
            if (current + 1 < backends.size() && used > 0 && used + bytes > block_budgets[current]) {
                boundaries[current] = b;
                current++;
                used = 0;
            }
            used += bytes;
        }

        for (const auto& kv : tensors) {
            size_t target = 0;
            int idx       = split_tensors.count(kv.first) != 0 ? layer_split_tensor_block_index(kv.first) : -1;
            if (idx >= 0) {
                while (target < boundaries.size() && idx >= boundaries[target]) {
                    target++;
                }
                target = std::min(target, backends.size() - 1);
                target = layer_split_supported_target(desc, kv.first, kv.second, backends, target);
            } else {
                auto target_it = non_block_targets.find(kv.first);
                if (target_it != non_block_targets.end()) {
                    target = target_it->second;
                }
            }
            partitions[target][kv.first] = kv.second;
        }

        int range_start = 0;
        for (size_t i = 0; i < backends.size(); i++) {
            int range_end                = boundaries[i];
            const char* non_block_suffix = other_bytes_by_backend[i] > 0 ? " + non-block tensors" : "";
            LOG_INFO("%s layer split: %s <- blocks [%d, %d)%s",
                     desc.c_str(),
                     layer_split_backend_device_display_name(backends[i]).c_str(),
                     range_start,
                     range_end,
                     non_block_suffix);
            range_start = range_end;
        }
        return partitions;
    }

}  // namespace sd

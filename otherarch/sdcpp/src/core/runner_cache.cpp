#include "core/runner_cache.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "core/ggml_graph_cut.h"
#include "core/util.h"

namespace sd {
    static std::unordered_set<const ggml_tensor*> cache_graph_tensors(ggml_cgraph* graph) {
        std::unordered_set<const ggml_tensor*> tensors;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            tensors.insert(ggml_graph_node(graph, i));
        }
        for (int i = 0; i < ggml_graph_cut::leaf_count(graph); ++i) {
            tensors.insert(ggml_graph_cut::leaf_tensor(graph, i));
        }
        return tensors;
    }

    CachedTensor::~CachedTensor() {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
    }

    std::unique_ptr<CachedTensor> CachedTensor::copy(ggml_backend_t backend,
                                                     const std::string& name,
                                                     ggml_tensor* source) {
        if (ggml_graph_cut::tensor_buffer(source) == nullptr) {
            return nullptr;
        }
        auto entry     = std::make_unique<CachedTensor>();
        entry->context = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        if (entry->context == nullptr) {
            return nullptr;
        }
        entry->tensor = ggml_dup_tensor(entry->context, source);
        // Cut views are rebound with their original strides and offsets.
        std::copy(std::begin(source->nb), std::end(source->nb), std::begin(entry->tensor->nb));
        ggml_set_name(entry->tensor, name.c_str());
        entry->buffer = ggml_backend_alloc_ctx_tensors(entry->context, backend);
        if (entry->buffer == nullptr) {
            return nullptr;
        }
        if (source->view_src != nullptr || !ggml_is_contiguous(source) || source->buffer == nullptr) {
            std::vector<uint8_t> data(ggml_nbytes(source));
            ggml_backend_tensor_get(source, data.data(), 0, data.size());
            ggml_backend_tensor_set(entry->tensor, data.data(), 0, data.size());
        } else {
            ggml_backend_tensor_copy(source, entry->tensor);
        }
        return entry;
    }

    static ggml_tensor* cached_tensor(const CachedTensors& tensors, const std::string& name) {
        auto entry = tensors.find(name);
        return entry == tensors.end() ? nullptr : entry->second->tensor;
    }

    static size_t resident_bytes(const CachedTensors& tensors, ggml_backend_dev_t device) {
        size_t bytes = 0;
        for (const auto& entry : tensors) {
            auto buffer = entry.second->buffer;
            if (!ggml_backend_buffer_is_host(buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) == device) {
                const size_t size = ggml_backend_buffer_get_size(buffer);
                bytes             = size > SIZE_MAX - bytes ? SIZE_MAX : bytes + size;
            }
        }
        return bytes;
    }

    ggml_tensor* RunnerCache::get(const std::string& name) const {
        return cached_tensor(committed_, name);
    }

    void RunnerCache::stage(const std::string& name, ggml_tensor* tensor) {
        if (tensor != nullptr) {
            ggml_set_output(tensor);
            outputs_[name] = tensor;
        }
    }

    size_t RunnerCache::pending_bytes(ggml_cgraph* graph) const {
        if (outputs_.empty()) {
            return 0;
        }
        auto tensors = cache_graph_tensors(graph);
        auto buft    = ggml_backend_get_default_buffer_type(backend_);
        size_t bytes = 0;
        for (const auto& output : outputs_) {
            if (pending_.count(output.first) || !tensors.count(output.second)) {
                continue;
            }
            const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, output.second),
                                         ggml_backend_buft_get_alignment(buft));
            bytes             = size > SIZE_MAX - bytes ? SIZE_MAX : bytes + size;
        }
        return bytes;
    }

    size_t RunnerCache::resident_bytes(ggml_backend_dev_t device) const {
        const size_t committed = sd::resident_bytes(committed_, device);
        const size_t pending   = sd::resident_bytes(pending_, device);
        return pending > SIZE_MAX - committed ? SIZE_MAX : committed + pending;
    }

    bool RunnerCache::capture(ggml_cgraph* graph) {
        if (outputs_.empty()) {
            return true;
        }
        const auto tensors = cache_graph_tensors(graph);
        for (const auto& output : outputs_) {
            if (pending_.count(output.first) || !tensors.count(output.second)) {
                continue;
            }
            GGML_ASSERT(ggml_is_contiguous(output.second));
            auto entry = CachedTensor::copy(backend_, output.first, output.second);
            if (entry == nullptr) {
                return false;
            }
            pending_[output.first] = std::move(entry);
        }
        ggml_backend_synchronize(backend_);
        return true;
    }

    void RunnerCache::graph_end(bool success) {
        // Graph inputs can still reference the previous generation until graph end.
        if (success) {
            for (auto& entry : pending_) {
                committed_[entry.first] = std::move(entry.second);
            }
        }
        pending_.clear();
        outputs_.clear();
    }

    void RunnerCache::clear() {
        graph_end(false);
        committed_.clear();
    }

    ggml_tensor* GraphCutTensorCache::get(const std::string& name) const {
        return cached_tensor(tensors_, name);
    }

    size_t GraphCutTensorCache::resident_bytes(ggml_backend_dev_t device) const {
        return sd::resident_bytes(tensors_, device);
    }

    size_t GraphCutTensorCache::estimate_output_bytes(
        ggml_cgraph* graph,
        const ggml_graph_cut::Segment& segment) const {
        ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend_);
        if (buffer_type == nullptr) {
            return SIZE_MAX;
        }
        const size_t alignment = ggml_backend_buft_get_alignment(buffer_type);
        size_t total_size      = 0;
        for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
            ggml_tensor* output = ggml_graph_cut::output_tensor(graph, segment, output_idx);
            if (output == nullptr || !ggml_graph_cut::is_graph_cut_tensor(output) ||
                !segment.future_cut_names.count(output->name)) {
                continue;
            }
            ggml_tensor* source      = ggml_graph_cut::cache_source_tensor(output);
            const size_t tensor_size = GGML_PAD(
                ggml_backend_buft_get_alloc_size(buffer_type, source), alignment);
            total_size = tensor_size > SIZE_MAX - total_size ? SIZE_MAX : total_size + tensor_size;
        }
        return total_size;
    }

    void GraphCutTensorCache::prune(const std::unordered_set<std::string>& keep_names) {
        for (auto it = tensors_.begin(); it != tensors_.end();) {
            it = keep_names.count(it->first) ? std::next(it) : tensors_.erase(it);
        }
    }

    bool GraphCutTensorCache::capture(ggml_cgraph* graph,
                                      const ggml_graph_cut::Segment& segment,
                                      const char* log_desc) {
        size_t copied_bytes = 0;
        size_t copied_count = 0;
        for (int index : segment.output_node_indices) {
            auto output = ggml_graph_node(graph, index);
            if (!ggml_graph_cut::is_graph_cut_tensor(output) ||
                !segment.future_cut_names.count(output->name)) {
                continue;
            }
            auto entry = CachedTensor::copy(backend_, output->name, ggml_graph_cut::cache_source_tensor(output));
            if (entry == nullptr) {
                LOG_ERROR("%s failed to capture graph cut tensor: %s", log_desc, output->name);
                return false;
            }
            const size_t size = ggml_backend_buffer_get_size(entry->buffer);
            copied_bytes      = size > SIZE_MAX - copied_bytes ? SIZE_MAX : copied_bytes + size;
            ++copied_count;
            tensors_[output->name] = std::move(entry);
        }
        ggml_backend_synchronize(backend_);
        if (copied_count > 0) {
            LOG_DEBUG("%s graph cut cache added %6.2f MB (%zu tensors)",
                      log_desc, copied_bytes / (1024.f * 1024.f), copied_count);
        }
        return true;
    }
}

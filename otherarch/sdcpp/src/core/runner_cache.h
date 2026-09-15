#ifndef __SD_CORE_RUNNER_CACHE_H__
#define __SD_CORE_RUNNER_CACHE_H__

#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "ggml-backend.h"

namespace sd::ggml_graph_cut {
    struct Segment;
}

namespace sd {
    struct CachedTensor {
        ggml_context* context        = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor* tensor          = nullptr;
        ~CachedTensor();
        static std::unique_ptr<CachedTensor> copy(ggml_backend_t backend,
                                                  const std::string& name,
                                                  ggml_tensor* source);
    };
    using CachedTensors = std::map<std::string, std::unique_ptr<CachedTensor>>;

    class RunnerCache {
        ggml_backend_t backend_;
        CachedTensors committed_;
        CachedTensors pending_;
        std::map<std::string, ggml_tensor*> outputs_;

    public:
        explicit RunnerCache(ggml_backend_t backend)
            : backend_(backend) {}
        RunnerCache(const RunnerCache&)            = delete;
        RunnerCache& operator=(const RunnerCache&) = delete;

        ggml_tensor* get(const std::string& name) const;
        void stage(const std::string& name, ggml_tensor* tensor);
        const std::map<std::string, ggml_tensor*>& outputs() const { return outputs_; }
        size_t pending_bytes(ggml_cgraph* graph) const;
        size_t resident_bytes(ggml_backend_dev_t device) const;
        bool capture(ggml_cgraph* graph);
        void graph_end(bool success);
        void clear();
    };

    class GraphCutTensorCache {
        ggml_backend_t backend_;
        CachedTensors tensors_;

    public:
        explicit GraphCutTensorCache(ggml_backend_t backend)
            : backend_(backend) {}
        ggml_tensor* get(const std::string& name) const;
        size_t resident_bytes(ggml_backend_dev_t device) const;
        size_t estimate_output_bytes(ggml_cgraph* graph,
                                     const ggml_graph_cut::Segment& segment) const;
        bool capture(ggml_cgraph* graph, const ggml_graph_cut::Segment& segment, const char* log_desc);
        void prune(const std::unordered_set<std::string>& keep_names);
        void clear() { tensors_.clear(); }
    };
}

#endif  // __SD_CORE_RUNNER_CACHE_H__

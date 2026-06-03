#ifndef __LAYER_REGISTRY_H__
#define __LAYER_REGISTRY_H__

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace sd::layer_registry {

    struct LayerInfo {
        std::vector<ggml_tensor*> tensors;
        std::vector<ggml_tensor*> gpu_twins;
        ggml_context* twin_ctx           = nullptr;
        ggml_backend_buffer_t gpu_buffer = nullptr;
        bool on_gpu                      = false;
        size_t bytes                     = 0;
    };

    class LayerRegistry {
    public:
        LayerRegistry() = default;
        LayerRegistry(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend)
            : gpu_backend_(gpu_backend), cpu_backend_(cpu_backend) {}

        void set_backends(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend) {
            gpu_backend_ = gpu_backend;
            cpu_backend_ = cpu_backend;
        }
        void register_layer(const std::string& name, ggml_tensor* tensor);
        bool move_layer_to_gpu(const std::string& name);
        bool move_layer_to_cpu(const std::string& name);
        bool is_layer_on_gpu(const std::string& name) const;
        size_t get_layer_size(const std::string& name) const;
        size_t get_layer_count() const { return layers_.size(); }

        const std::map<std::string, LayerInfo>& layers() const { return layers_; }

    private:
        ggml_backend_t gpu_backend_ = nullptr;
        ggml_backend_t cpu_backend_ = nullptr;
        std::map<std::string, LayerInfo> layers_;
    };

}  // namespace sd::layer_registry

#endif

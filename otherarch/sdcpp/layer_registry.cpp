#include "layer_registry.h"

#include <utility>

#include "util.h"

namespace sd::layer_registry {

    void LayerRegistry::register_layer(const std::string& name, ggml_tensor* tensor) {
        auto& info = layers_[name];
        info.tensors.push_back(tensor);
        info.bytes += ggml_nbytes(tensor);
    }

    bool LayerRegistry::move_layer_to_gpu(const std::string& name) {
        auto it = layers_.find(name);
        if (it == layers_.end())
            return false;

        LayerInfo& info = it->second;
        if (info.on_gpu)
            return true;
        if (gpu_backend_ == nullptr || cpu_backend_ == nullptr) {
            LOG_ERROR("layer_registry: backends not set; cannot move '%s' to GPU",
                      name.c_str());
            return false;
        }
        if (info.tensors.empty()) {
            info.on_gpu = true;
            return true;
        }

        // 1. Build a no_alloc context big enough to hold one twin tensor per CPU
        //    tensor, plus a little overhead.
        const size_t ctx_size = info.tensors.size() * ggml_tensor_overhead() + 1024;
        ggml_init_params ctx_params{ctx_size, /*mem_buffer=*/nullptr, /*no_alloc=*/true};
        ggml_context* twin_ctx = ggml_init(ctx_params);
        if (twin_ctx == nullptr) {
            LOG_ERROR("layer_registry: failed to allocate twin context for '%s'",
                      name.c_str());
            return false;
        }

        // 2. Create one GPU twin per CPU tensor. The twin shares the original
        //    name so any name-based lookup keeps working.
        std::vector<ggml_tensor*> gpu_twins;
        gpu_twins.reserve(info.tensors.size());
        for (ggml_tensor* cpu_t : info.tensors) {
            ggml_tensor* twin = ggml_dup_tensor(twin_ctx, cpu_t);
            if (cpu_t->name[0] != '\0') {
                ggml_set_name(twin, cpu_t->name);
            }
            gpu_twins.push_back(twin);
        }

        // 3. Back the twins with a GPU buffer in one alloc call.
        ggml_backend_buffer_t gpu_buffer = ggml_backend_alloc_ctx_tensors(twin_ctx, gpu_backend_);
        if (gpu_buffer == nullptr) {
            LOG_ERROR("layer_registry: failed to allocate GPU buffer for '%s'",
                      name.c_str());
            ggml_free(twin_ctx);
            return false;
        }

        // 4. H2D copy + sync.
        for (size_t i = 0; i < info.tensors.size(); ++i) {
            ggml_backend_tensor_copy(info.tensors[i], gpu_twins[i]);
        }
        ggml_backend_synchronize(gpu_backend_);

        // 5. Swap buffer/data/extra so the originals now point at GPU memory.
        for (size_t i = 0; i < info.tensors.size(); ++i) {
            std::swap(info.tensors[i]->buffer, gpu_twins[i]->buffer);
            std::swap(info.tensors[i]->data, gpu_twins[i]->data);
            std::swap(info.tensors[i]->extra, gpu_twins[i]->extra);
        }

        info.gpu_twins  = std::move(gpu_twins);
        info.twin_ctx   = twin_ctx;
        info.gpu_buffer = gpu_buffer;
        info.on_gpu     = true;
        return true;
    }

    bool LayerRegistry::move_layer_to_cpu(const std::string& name) {
        auto it = layers_.find(name);
        if (it == layers_.end())
            return false;

        LayerInfo& info = it->second;
        if (!info.on_gpu)
            return true;
        if (info.tensors.size() != info.gpu_twins.size()) {
            LOG_ERROR("layer_registry: twin/tensor count mismatch for '%s'",
                      name.c_str());
            return false;
        }

        // 1. Swap back: originals point at CPU memory again.
        for (size_t i = 0; i < info.tensors.size(); ++i) {
            if (info.gpu_twins[i] == nullptr)
                continue;
            std::swap(info.tensors[i]->buffer, info.gpu_twins[i]->buffer);
            std::swap(info.tensors[i]->data, info.gpu_twins[i]->data);
            std::swap(info.tensors[i]->extra, info.gpu_twins[i]->extra);
        }

        // 2. Free the GPU buffer + twin context.
        if (info.gpu_buffer != nullptr) {
            ggml_backend_buffer_free(info.gpu_buffer);
            info.gpu_buffer = nullptr;
        }
        if (info.twin_ctx != nullptr) {
            ggml_free(info.twin_ctx);
            info.twin_ctx = nullptr;
        }
        info.gpu_twins.clear();
        info.on_gpu = false;
        return true;
    }

    bool LayerRegistry::is_layer_on_gpu(const std::string& name) const {
        auto it = layers_.find(name);
        return it != layers_.end() && it->second.on_gpu;
    }

    size_t LayerRegistry::get_layer_size(const std::string& name) const {
        auto it = layers_.find(name);
        return it != layers_.end() ? it->second.bytes : 0;
    }

}  // namespace sd::layer_registry

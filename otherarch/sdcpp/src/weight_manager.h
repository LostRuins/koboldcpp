#ifndef __WEIGHT_MANAGER_H__
#define __WEIGHT_MANAGER_H__

#include <vector>

struct ggml_tensor;

struct RunnerWeightManager {
    virtual ~RunnerWeightManager()                                                        = default;
    virtual bool prepare_params(const std::vector<ggml_tensor*>& tensors)                 = 0;
    virtual void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void release_params_backend_params(const std::vector<ggml_tensor*>& tensors)  = 0;
};

#endif  // __WEIGHT_MANAGER_H__

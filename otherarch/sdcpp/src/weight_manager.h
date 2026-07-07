#ifndef __WEIGHT_MANAGER_H__
#define __WEIGHT_MANAGER_H__

#include <vector>

#include "ggml-backend.h"

struct ggml_tensor;

struct RunnerWeightManager {
    virtual ~RunnerWeightManager()                                                        = default;
    virtual bool assign_compute_backend(const std::vector<ggml_tensor*>& tensors,
                                        ggml_backend_t compute_backend)                   = 0;
    virtual bool prepare_params(const std::vector<ggml_tensor*>& tensors)                 = 0;
    virtual void release_compute_backend_params(const std::vector<ggml_tensor*>& tensors) = 0;
    virtual void release_params_backend_params(const std::vector<ggml_tensor*>& tensors)  = 0;
};

#endif  // __WEIGHT_MANAGER_H__

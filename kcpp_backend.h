#pragma once

#include "ggml-backend.h"
#include "ggml_v2.h"
#include "ggml_v3.h"

// backends with GGML_USE_CUDA
#define KCPP_BACKENDS_USE_CUDA "cuda|rocm"

// backends that support tensor split
#define KCPP_BACKENDS_TENSOR_SPLIT "cuda|rocm|vulkan"

// backends that support blas
#define KCPP_BACKENDS_BLAS "blas|cuda|rocm|vulkan|sycl"

// checks if the provided backend (or the first one) matches a |-separated name list
int kcpp_backend_check(const char* name_list, ggml_backend_t backend = nullptr);

ggml_backend_dev_t kcpp_backend_get_device(int index);

// per-backend aux functions

void kcpp_backend_cuda_ggmlv2_transform_tensor(ggml_v2_tensor * tensor);

void kcpp_backend_cuda_ggmlv3_assign_buffers(struct ggml_v3_tensor * tensor);
void kcpp_backend_cuda_ggmlv3_assign_buffers_force_inplace(struct ggml_v3_tensor * tensor);
void kcpp_backend_cuda_ggmlv3_assign_buffers_no_scratch(struct ggml_v3_tensor * tensor);
void kcpp_backend_cuda_ggmlv3_free_data(struct ggml_v3_tensor * tensor);
void kcpp_backend_cuda_ggmlv3_free_scratch(void);
void kcpp_backend_cuda_ggmlv3_set_main_device(int device);
void kcpp_backend_cuda_ggmlv3_set_mul_mat_q(bool mul_mat_q);
void kcpp_backend_cuda_ggmlv3_set_scratch_size(size_t scratch_size);
void kcpp_backend_cuda_ggmlv3_set_tensor_split(const float * tensor_split);
void kcpp_backend_cuda_ggmlv3_transform_tensor(void * data, struct ggml_v3_tensor * tensor);

void kcpp_backend_cuda_set_mul_mat_q(int use_mmq);

void kcpp_backend_hip_initialize();

bool kcpp_backend_metal_supports_family(ggml_backend_t backend, int family);


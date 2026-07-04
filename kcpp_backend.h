#pragma once

#include "ggml-backend.h"

// backends with GGML_USE_CUDA
#define KCPP_BACKENDS_USE_CUDA "cuda|rocm"

// backends that support tensor split
#define KCPP_BACKENDS_TENSOR_SPLIT "cuda|rocm|vulkan"

// backends that support blas
#define KCPP_BACKENDS_BLAS "blas|cuda|rocm|vulkan|sycl"

// checks if the provided backend (or the first one) matches a |-separated name list
int kcpp_backend_check(const char* name_list, ggml_backend_t backend = nullptr);


// per-backend aux functions

void kcpp_backend_cuda_ggmlv3_set_main_device(int device);

void kcpp_backend_cuda_set_mul_mat_q(int use_mmq);

void kcpp_backend_hip_initialize();


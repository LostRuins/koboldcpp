#pragma once

#include "ggml_v3.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_V3_CUDA_NAME "ROCm"
#define GGML_V3_CUBLAS_NAME "hipBLAS"
#else
#define GGML_V3_CUDA_NAME "CUDA"
#define GGML_V3_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_V3_CUDA_MAX_DEVICES       16

// Always success. To check if CUDA is actually loaded, use `ggml_v3_cublas_loaded`.
GGML_V3_API void   ggml_v3_init_cublas(void);

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
GGML_V3_API bool   ggml_v3_cublas_loaded(void);

GGML_V3_API void * ggml_v3_cuda_host_malloc(size_t size);
GGML_V3_API void   ggml_v3_cuda_host_free(void * ptr);

GGML_V3_API bool   ggml_v3_cuda_can_mul_mat(const struct ggml_v3_tensor * src0, const struct ggml_v3_tensor * src1, struct ggml_v3_tensor * dst);
GGML_V3_API void   ggml_v3_cuda_set_tensor_split(const float * tensor_split);
GGML_V3_API void   ggml_v3_cuda_transform_tensor(void * data, struct ggml_v3_tensor * tensor);
GGML_V3_API void   ggml_v3_cuda_free_data(struct ggml_v3_tensor * tensor);

GGML_V3_API void   ggml_v3_cuda_assign_buffers(struct ggml_v3_tensor * tensor);
GGML_V3_API void   ggml_v3_cuda_assign_buffers_no_scratch(struct ggml_v3_tensor * tensor);
GGML_V3_API void   ggml_v3_cuda_assign_buffers_force_inplace(struct ggml_v3_tensor * tensor);

GGML_V3_API void   ggml_v3_cuda_assign_buffers_no_alloc(struct ggml_v3_tensor * tensor);
GGML_V3_API void   ggml_v3_cuda_assign_scratch_offset(struct ggml_v3_tensor * tensor, size_t offset);
GGML_V3_API void   ggml_v3_cuda_copy_to_device(struct ggml_v3_tensor * tensor);

GGML_V3_API void   ggml_v3_cuda_set_main_device(int main_device);
GGML_V3_API void   ggml_v3_cuda_set_mul_mat_q(bool mul_mat_q);
GGML_V3_API void   ggml_v3_cuda_set_scratch_size(size_t scratch_size);
GGML_V3_API void   ggml_v3_cuda_free_scratch(void);
GGML_V3_API bool   ggml_v3_cuda_compute_forward(struct ggml_v3_compute_params * params, struct ggml_v3_tensor * tensor);

GGML_V3_API int    ggml_v3_cuda_get_device_count(void);
GGML_V3_API void   ggml_v3_cuda_get_device_description(int device, char * description, size_t description_size);


#ifdef  __cplusplus
}
#endif

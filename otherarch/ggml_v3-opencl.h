#pragma once

#include "ggml_v3.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_V3_API void ggml_v3_cl_init(void);

GGML_V3_API void   ggml_v3_cl_mul(const struct ggml_v3_tensor * src0, const struct ggml_v3_tensor * src1, struct ggml_v3_tensor * dst);
GGML_V3_API bool   ggml_v3_cl_can_mul_mat(const struct ggml_v3_tensor * src0, const struct ggml_v3_tensor * src1, struct ggml_v3_tensor * dst);
GGML_V3_API size_t ggml_v3_cl_mul_mat_get_wsize(const struct ggml_v3_tensor * src0, const struct ggml_v3_tensor * src1, struct ggml_v3_tensor * dst);
GGML_V3_API void   ggml_v3_cl_mul_mat(const struct ggml_v3_tensor * src0, const struct ggml_v3_tensor * src1, struct ggml_v3_tensor * dst, void * wdata, size_t wsize);

GGML_V3_API void * ggml_v3_cl_host_malloc(size_t size);
GGML_V3_API void   ggml_v3_cl_host_free(void * ptr);

GGML_V3_API void ggml_v3_cl_free_data(const struct ggml_v3_tensor* tensor);

GGML_V3_API void ggml_v3_cl_transform_tensor(void * data, struct ggml_v3_tensor * tensor);

#ifdef  __cplusplus
}
#endif
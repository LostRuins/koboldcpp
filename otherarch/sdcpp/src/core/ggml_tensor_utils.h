#ifndef __SD_CORE_GGML_TENSOR_UTILS_H__
#define __SD_CORE_GGML_TENSOR_UTILS_H__

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "core/tensor.hpp"
#include "core/util.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "stable-diffusion.h"

class RNG;

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ int align_up(int n, int multiple) {
    return n + align_up_offset(n, multiple);
}

void ggml_ext_im_set_randn_f32(ggml_tensor* tensor, std::shared_ptr<RNG> rng);

__STATIC_INLINE__ void ggml_ext_tensor_set_f32(ggml_tensor* tensor, float value, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_ext_tensor_get_f32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_ext_tensor_get_i32(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    if (tensor->buffer != nullptr) {
        int value;
        ggml_backend_tensor_get(tensor, &value, i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_ext_tensor_get_f16(const ggml_tensor* tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 * tensor->nb[0]);
}

__STATIC_INLINE__ float sd_image_get_f32(sd_image_t image, int64_t iw, int64_t ih, int64_t ic, bool scale = true) {
    float value = *(image.data + ih * image.width * image.channel + iw * image.channel + ic);
    if (scale) {
        value /= 255.f;
    }
    return value;
}

void print_ggml_tensor(ggml_tensor* tensor, bool shape_only = false, const char* mark = "");

template <typename T>
__STATIC_INLINE__ void print_sd_tensor(const sd::Tensor<T>& tensor, bool shape_only = false, const char* mark = "") {
    printf("%s: shape(", mark);
    for (size_t i = 0; i < static_cast<size_t>(tensor.dim()); ++i) {
        printf("%s%lld", i == 0 ? "" : ", ", static_cast<long long>(tensor.shape()[i]));
    }
    printf(")\n");
    fflush(stdout);
    if (shape_only) {
        return;
    }
    if (tensor.empty()) {
        return;
    }
    int range                  = 3;
    std::vector<int64_t> shape = tensor.shape();
    while (shape.size() < 4) {
        shape.push_back(1);
    }
    for (int64_t i3 = 0; i3 < shape[3]; i3++) {
        if (i3 >= range && i3 + range < shape[3]) {
            continue;
        }
        for (int64_t i2 = 0; i2 < shape[2]; i2++) {
            if (i2 >= range && i2 + range < shape[2]) {
                continue;
            }
            for (int64_t i1 = 0; i1 < shape[1]; i1++) {
                if (i1 >= range && i1 + range < shape[1]) {
                    continue;
                }
                for (int64_t i0 = 0; i0 < shape[0]; i0++) {
                    if (i0 >= range && i0 + range < shape[0]) {
                        continue;
                    }
                    size_t offset = static_cast<size_t>(i0 + shape[0] * (i1 + shape[1] * (i2 + shape[2] * i3)));
                    printf("  [%lld, %lld, %lld, %lld] = ", static_cast<long long>(i3), static_cast<long long>(i2), static_cast<long long>(i1), static_cast<long long>(i0));
                    if constexpr (std::is_same_v<T, float>) {
                        printf("%f\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, ggml_fp16_t>) {
                        printf("%f\n", ggml_fp16_to_fp32(tensor[static_cast<int64_t>(offset)]));
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        printf("%d\n", tensor[static_cast<int64_t>(offset)]);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        printf("%lld\n", static_cast<long long>(tensor[static_cast<int64_t>(offset)]));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t, int64_t, int64_t, int64_t)>& fn);

void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t)>& fn);

void ggml_ext_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap = 0.1f);

ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path);

__STATIC_INLINE__ float sigmoid(float x) {
    return 1 / (1.0f + expf(-x));
}

// SPECIAL OPERATIONS WITH TENSORS

uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, uint8_t* image_data = nullptr);

uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, int idx, bool video = false);

void sd_image_to_ggml_tensor(sd_image_t image,
                             ggml_tensor* tensor,
                             bool scale = true);

void ggml_ext_tensor_apply_mask(ggml_tensor* image_data,
                                ggml_tensor* mask,
                                ggml_tensor* output,
                                float masked_value = 0.5f);

float ggml_ext_tensor_mean(ggml_tensor* src);

// a = a+b
void ggml_ext_tensor_add_inplace(ggml_tensor* a, ggml_tensor* b);

void ggml_ext_tensor_scale_inplace(ggml_tensor* src, float scale);

void ggml_ext_tensor_clamp_inplace(ggml_tensor* src, float min, float max);

ggml_tensor* ggml_ext_tensor_concat(ggml_context* ctx,
                                    ggml_tensor* a,
                                    ggml_tensor* b,
                                    int dim);

// convert values from [0, 1] to [-1, 1]
void scale_to_minus1_1(ggml_tensor* src);

// convert values from [-1, 1] to [0, 1]
void scale_to_0_1(ggml_tensor* src);

ggml_tensor* vector_to_ggml_tensor(ggml_context* ctx,
                                   const std::vector<float>& vec);

ggml_tensor* vector_to_ggml_tensor_i32(ggml_context* ctx,
                                       const std::vector<int>& vec);

std::vector<float> arange(float start, float end, float step = 1.f);

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
std::vector<float> timestep_embedding(std::vector<float> timesteps,
                                      int dim,
                                      int max_period       = 10000,
                                      bool flip_sin_to_cos = true,
                                      float scale          = 1.f);

void set_timestep_embedding(std::vector<float> timesteps,
                            ggml_tensor* embedding,
                            int dim,
                            int max_period = 10000);

void set_timestep_embedding(std::vector<float> timesteps,
                            sd::Tensor<float>* embedding,
                            int dim,
                            int max_period = 10000);

ggml_tensor* new_timestep_embedding(ggml_context* ctx,
                                    std::vector<float> timesteps,
                                    int dim,
                                    int max_period = 10000);

size_t ggml_tensor_num(ggml_context* ctx);

#endif  // __SD_CORE_GGML_TENSOR_UTILS_H__

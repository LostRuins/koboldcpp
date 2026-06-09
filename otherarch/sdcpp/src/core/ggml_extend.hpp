#ifndef __SD_CORE_GGML_EXTEND_HPP__
#define __SD_CORE_GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_graph_cut.h"
#include "core/layer_registry.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "core/tensor.hpp"
#include "model.h"

#include "core/rng.hpp"
#include "core/tensor_ggml.hpp"
#include "core/util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

#ifndef SD_UNUSED
#define SD_UNUSED(x) (void)(x)
#endif

__STATIC_INLINE__ int align_up_offset(int n, int multiple) {
    return (multiple - n % multiple) % multiple;
}

__STATIC_INLINE__ int align_up(int n, int multiple) {
    return n + align_up_offset(n, multiple);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void*) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
            LOG_DEBUG(text);
            break;
        case GGML_LOG_LEVEL_INFO:
            LOG_INFO(text);
            break;
        case GGML_LOG_LEVEL_WARN:
            LOG_WARN(text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            LOG_ERROR(text);
            break;
        default:
            LOG_DEBUG(text);
    }
}

static_assert(GGML_MAX_NAME >= 128, "GGML_MAX_NAME must be at least 128");

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
__STATIC_INLINE__ ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int mode = 0) {
    // reshape A
    // swap 0th and nth axis
    a           = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0));
    int64_t ne1 = a->ne[1];
    int64_t ne2 = a->ne[2];
    int64_t ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshape_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));

    // reshape output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0);
    return result;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_merge_lora(ggml_context* ctx,
                                                   ggml_tensor* lora_down,
                                                   ggml_tensor* lora_up,
                                                   ggml_tensor* lora_mid = nullptr) {
    ggml_tensor* updown;
    // flat lora tensors to multiply it
    int64_t lora_up_rows  = lora_up->ne[ggml_n_dims(lora_up) - 1];
    lora_up               = ggml_reshape_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
    auto lora_down_n_dims = ggml_n_dims(lora_down);
    // assume n_dims should always be a multiple of 2 (otherwise rank 1 doesn't work)
    lora_down_n_dims       = (lora_down_n_dims + lora_down_n_dims % 2);
    int64_t lora_down_rows = lora_down->ne[lora_down_n_dims - 1];
    lora_down              = ggml_reshape_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

    // ggml_mul_mat requires tensor b transposed
    lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));
    if (lora_mid == nullptr) {
        updown = ggml_mul_mat(ctx, lora_up, lora_down);
        updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
    } else {
        // undoing tucker decomposition for conv layers.
        // lora_mid  has shape (3,    3,   Rank, Rank)
        // lora_down has shape (Rank, In,  1,    1)
        // lora_up   has shape (Rank, Out, 1,    1)
        // conv layer shape is (3,    3,   Out,  In)
        updown = ggml_ext_mul_n_mode(ctx, ggml_ext_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
        updown = ggml_cont(ctx, updown);
    }
    return updown;
}

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
__STATIC_INLINE__ ggml_tensor* ggml_ext_kronecker(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul(ctx,
                    ggml_interpolate(ctx,
                                     a,
                                     a->ne[0] * b->ne[0],
                                     a->ne[1] * b->ne[1],
                                     a->ne[2] * b->ne[2],
                                     a->ne[3] * b->ne[3],
                                     GGML_SCALE_MODE_NEAREST),
                    b);
}

__STATIC_INLINE__ void ggml_ext_im_set_randn_f32(ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_ext_im_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

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

__STATIC_INLINE__ void print_ggml_tensor(ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%zu, %zu, %zu, %zu)\n", mark, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        if (i3 >= range && i3 + range < tensor->ne[3]) {
            continue;
        }
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            if (i2 >= range && i2 + range < tensor->ne[2]) {
                continue;
            }
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                if (i1 >= range && i1 + range < tensor->ne[1]) {
                    continue;
                }
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    if (i0 >= range && i0 + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_ext_tensor_get_f32(tensor, i0, i1, i2, i3));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %f\n", i3, i2, i1, i0, ggml_fp16_to_fp32(ggml_ext_tensor_get_f16(tensor, i0, i1, i2, i3)));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i3\n", i3, i2, i1, i0, ggml_ext_tensor_get_i32(tensor, i0, i1, i2, i3));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

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

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t, int64_t, int64_t, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    fn(tensor, i0, i1, i2, i3);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_iter(
    ggml_tensor* tensor,
    const std::function<void(ggml_tensor*, int64_t)>& fn) {
    int64_t n0 = tensor->ne[0];
    int64_t n1 = tensor->ne[1];
    int64_t n2 = tensor->ne[2];
    int64_t n3 = tensor->ne[3];

    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        fn(tensor, i);
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap = 0.1f) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    ggml_ext_tensor_iter(a, [&](ggml_tensor* a, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float a_value = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
        float b_value = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
        if (abs(a_value - b_value) > gap) {
            LOG_WARN("[%ld, %ld, %ld, %ld] %f %f", i3, i2, i1, i0, a_value, b_value);
        }
    });
}

__STATIC_INLINE__ ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    std::ifstream file(sd_get_u8path(file_path), std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return nullptr;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;

    file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

    LOG_DEBUG("load_tensor_from_file %d %d %d", n_dims, length, ttype);

    if (file.eof()) {
        LOG_ERROR("incomplete file '%s'", file_path.c_str());
        return nullptr;
    }

    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        file.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }
    std::string name(length, 0);
    file.read(&name[0], length);
    ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (ggml_type)ttype, ne[0], ne[1], ne[2], ne[3]);
    const size_t bpe    = ggml_type_size(ggml_type(ttype));
    file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));
    return tensor;
}

// __STATIC_INLINE__ void save_tensor_to_file(const std::string& file_name, ggml_tensor* tensor, const std::string & name) {
//     std::string file_name_ = file_name + ".tensor";
//     std::string name_ = name;
//     std::ofstream file("./" + file_name_, std::ios::binary);
//     file.write(reinterpret_cast<char*>(&tensor->n_dims), sizeof(tensor->n_dims));
//     int len = (int)name_.size();
//     file.write(reinterpret_cast<char*>(&len), sizeof(len));
//     int ttype = (int)tensor->type;
//     file.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
//     for (int i = 0; i < tensor->n_dims; ++i) {
//         int ne_ = (int) tensor->ne[i];
//         file.write(reinterpret_cast<char*>(&ne_), sizeof(ne_));
//     }
//     file.write(&name_[0], len);
//     char* data = nullptr;
//     file.write((char*)tensor->data, ggml_nbytes(tensor));
//     file.close();
// }

__STATIC_INLINE__ float sigmoid(float x) {
    return 1 / (1.0f + expf(-x));
}

// SPECIAL OPERATIONS WITH TENSORS

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, uint8_t* image_data = nullptr) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    if (image_data == nullptr) {
        image_data = (uint8_t*)malloc(width * height * channels);
    }
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_ext_tensor_get_f32(input, ix, iy, k);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, int idx, bool video = false) {
    int64_t width  = input->ne[0];
    int64_t height = input->ne[1];
    int64_t channels;
    if (video) {
        channels = input->ne[3];
    } else {
        channels = input->ne[2];
    }
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int ih = 0; ih < height; ih++) {
        for (int iw = 0; iw < width; iw++) {
            for (int ic = 0; ic < channels; ic++) {
                float value;
                if (video) {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, idx, ic);
                } else {
                    value = ggml_ext_tensor_get_f32(input, iw, ih, ic, idx);
                }
                *(image_data + ih * width * channels + iw * channels + ic) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void sd_image_to_ggml_tensor(sd_image_t image,
                                               ggml_tensor* tensor,
                                               bool scale = true) {
    GGML_ASSERT(image.width == tensor->ne[0]);
    GGML_ASSERT(image.height == tensor->ne[1]);
    GGML_ASSERT(image.channel == tensor->ne[2]);
    GGML_ASSERT(1 == tensor->ne[3]);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    ggml_ext_tensor_iter(tensor, [&](ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float value = sd_image_get_f32(image, i0, i1, i2, scale);
        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2, i3);
    });
}

__STATIC_INLINE__ void ggml_ext_tensor_apply_mask(ggml_tensor* image_data,
                                                  ggml_tensor* mask,
                                                  ggml_tensor* output,
                                                  float masked_value = 0.5f) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    float rescale_mx = 1.f * mask->ne[0] / output->ne[0];
    float rescale_my = 1.f * mask->ne[1] / output->ne[1];
    GGML_ASSERT(output->type == GGML_TYPE_F32);
    for (int ix = 0; ix < width; ix++) {
        for (int iy = 0; iy < height; iy++) {
            int mx  = (int)(ix * rescale_mx);
            int my  = (int)(iy * rescale_my);
            float m = ggml_ext_tensor_get_f32(mask, mx, my);
            m       = round(m);  // inpaint models need binary masks
            ggml_ext_tensor_set_f32(mask, m, mx, my);
            for (int k = 0; k < channels; k++) {
                float value = ggml_ext_tensor_get_f32(image_data, ix, iy, k);
                value       = (1 - m) * (value - masked_value) + masked_value;
                ggml_ext_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ float ggml_ext_tensor_mean(ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_ext_tensor_add_inplace(ggml_tensor* a, ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_scale_inplace(ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_ext_tensor_clamp_inplace(ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_tensor_concat(ggml_context* ctx,
                                                      ggml_tensor* a,
                                                      ggml_tensor* b,
                                                      int dim) {
    int64_t ne[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            continue;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }
    ggml_tensor* result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);
    int64_t o[4]        = {0, 0, 0, 0};
    o[dim]              = a->ne[dim];

    float v;
    for (int i3 = 0; i3 < result->ne[3]; i3++) {
        for (int i2 = 0; i2 < result->ne[2]; i2++) {
            for (int i1 = 0; i1 < result->ne[1]; i1++) {
                for (int i0 = 0; i0 < result->ne[0]; i0++) {
                    if (i0 < a->ne[0] && i1 < a->ne[1] && i2 < a->ne[2] && i3 < a->ne[3]) {
                        v = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
                    } else {
                        v = ggml_ext_tensor_get_f32(b, i0 - o[0], i1 - o[1], i2 - o[2], i3 - o[3]);
                    }

                    ggml_ext_tensor_set_f32(result, v, i0, i1, i2, i3);
                }
            }
        }
    }
    return result;
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void scale_to_minus1_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void scale_to_0_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                                             ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

// torch like permute
__STATIC_INLINE__ ggml_tensor* ggml_ext_torch_permute(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      int axis0,
                                                      int axis1,
                                                      int axis2,
                                                      int axis3) {
    int torch_axes[4] = {axis0, axis1, axis2, axis3};

    int ggml_axes[4] = {0};
    for (int i = 0; i < 4; ++i) {
        int found = 0;
        for (int j = 0; j < 4; ++j) {
            if (torch_axes[j] == i) {
                ggml_axes[i] = j;
                found        = 1;
                break;
            }
        }
        GGML_ASSERT(found && "Invalid permute input: must be a permutation of 0-3");
    }

    return ggml_permute(ctx, x, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_slice(ggml_context* ctx,
                                              ggml_tensor* x,
                                              int dim,
                                              int64_t start,
                                              int64_t end,
                                              bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    if (x->ne[dim] == 1) {
        return x;
    }
    while (start < 0) {
        start = x->ne[dim] + start;
    }
    while (end < 0) {
        end = x->ne[dim] + end;
    }
    GGML_ASSERT(end > start);
    GGML_ASSERT(start >= 0 && start < x->ne[dim]);
    GGML_ASSERT(end > start && end <= x->ne[dim]);

    int64_t slice_size  = end - start;
    int64_t slice_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    slice_ne[dim]       = slice_size;

    x = ggml_view_4d(ctx, x,
                     slice_ne[0], slice_ne[1], slice_ne[2], slice_ne[3],
                     x->nb[1], x->nb[2], x->nb[3], start * x->nb[dim]);

    if (cont) {
        x = ggml_cont(ctx, x);
    }

    return x;
}

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> ggml_ext_chunk(ggml_context* ctx,
                                                           ggml_tensor* x,
                                                           int num,
                                                           int64_t dim,
                                                           bool cont = true) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    std::vector<ggml_tensor*> chunks;
    int64_t chunk_size  = x->ne[dim] / num;
    int64_t stride      = chunk_size * x->nb[dim];
    int64_t chunk_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    chunk_ne[dim]       = chunk_size;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            chunk_ne[0], chunk_ne[1], chunk_ne[2], chunk_ne[3],
            x->nb[1], x->nb[2], x->nb[3], stride * i);
        if (cont) {
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first = true) {
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    auto x_vec = ggml_ext_chunk(ctx, x, 2, 0, false);
    ggml_tensor* gate;
    if (gate_first) {
        gate = x_vec[0];
        x    = x_vec[1];
    } else {
        x    = x_vec[0];
        gate = x_vec[1];
    }
    gate = ggml_cont(ctx, gate);
    gate = ggml_silu_inplace(ctx, gate);

    x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, ne0/2]

    return x;
}

typedef std::function<bool(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void sd_tiling_calc_tiles(int& num_tiles_dim,
                                            float& tile_overlap_factor_dim,
                                            int small_dim,
                                            int tile_size,
                                            const float tile_overlap_factor,
                                            bool circular) {
    int tile_overlap     = static_cast<int>(tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    if (circular) {
        // circular means the last and first tile are overlapping (wraping around)
        num_tiles_dim = small_dim / non_tile_overlap;

        if (num_tiles_dim < 1) {
            num_tiles_dim = 1;
        }

        tile_overlap_factor_dim = (tile_size - small_dim / num_tiles_dim) / (float)tile_size;

        // if single tile and tile_overlap_factor is not 0, add one to ensure we have at least two overlapping tiles
        if (num_tiles_dim == 1 && tile_overlap_factor_dim > 0) {
            num_tiles_dim++;
            tile_overlap_factor_dim = 0.5;
        }

        return;
    }
    // else, non-circular means the last and first tile are not overlapping

    num_tiles_dim     = (small_dim - tile_overlap) / non_tile_overlap;
    int overshoot_dim = ((num_tiles_dim + 1) * non_tile_overlap + tile_overlap) % small_dim;

    if ((overshoot_dim != non_tile_overlap) && (overshoot_dim <= num_tiles_dim * (tile_size / 2 - tile_overlap))) {
        // if tiles don't fit perfectly using the desired overlap
        // and there is enough room to squeeze an extra tile without overlap becoming >0.5
        num_tiles_dim++;
    }

    tile_overlap_factor_dim = (float)(tile_size * num_tiles_dim - small_dim) / (float)(tile_size * (num_tiles_dim - 1));
    if (num_tiles_dim <= 2) {
        if (small_dim <= tile_size) {
            num_tiles_dim           = 1;
            tile_overlap_factor_dim = 0;
        } else {
            num_tiles_dim           = 2;
            tile_overlap_factor_dim = (2 * tile_size - small_dim) / (float)tile_size;
        }
    }
}

// Tiling

__STATIC_INLINE__ int64_t sd_tensor_plane_size(const sd::Tensor<float>& tensor) {
    GGML_ASSERT(tensor.dim() >= 2);
    return tensor.shape()[0] * tensor.shape()[1];
}

__STATIC_INLINE__ sd::Tensor<float> sd_tensor_split_2d(const sd::Tensor<float>& input, int width, int height, int x, int y) {
    GGML_ASSERT(input.dim() >= 4);
    std::vector<int64_t> output_shape = input.shape();
    output_shape[0]                   = width;
    output_shape[1]                   = height;
    sd::Tensor<float> output(std::move(output_shape));
    int64_t input_width  = input.shape()[0];
    int64_t input_height = input.shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(output);
    int64_t plane_count  = input.numel() / input_plane;
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            int64_t src_xy = (ix + x) % input_width + input_width * ((iy + y) % input_height);
            int64_t dst_xy = ix + width * iy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                output[plane * output_plane + dst_xy] = input[plane * input_plane + src_xy];
            }
        }
    }
    return output;
}

__STATIC_INLINE__ void sd_tensor_merge_2d(const sd::Tensor<float>& input,
                                          sd::Tensor<float>* output,
                                          int x,
                                          int y,
                                          int overlap_x,
                                          int overlap_y,
                                          bool circular_x,
                                          bool circular_y,
                                          int x_skip = 0,
                                          int y_skip = 0) {
    GGML_ASSERT(output != nullptr);
    int64_t width        = input.shape()[0];
    int64_t height       = input.shape()[1];
    int64_t img_width    = output->shape()[0];
    int64_t img_height   = output->shape()[1];
    int64_t input_plane  = sd_tensor_plane_size(input);
    int64_t output_plane = sd_tensor_plane_size(*output);
    int64_t plane_count  = input.numel() / input_plane;
    GGML_ASSERT(output->numel() / output_plane == plane_count);

    // unclamped -> expects x in the range [0-1]
    auto smootherstep_f32 = [](const float x) -> float {
        GGML_ASSERT(x >= 0.f && x <= 1.f);
        return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
    };

    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            int64_t src_xy = ix + width * iy;
            int64_t ox     = (x + ix) % img_width;
            int64_t oy     = (y + iy) % img_height;
            int64_t dst_xy = ox + img_width * oy;
            for (int64_t plane = 0; plane < plane_count; ++plane) {
                float new_value = input[plane * input_plane + src_xy];
                if (overlap_x > 0 || overlap_y > 0) {
                    float old_value   = (*output)[plane * output_plane + dst_xy];
                    const float x_f_0 = (circular_x || (overlap_x > 0 && x > 0)) ? (ix - x_skip) / float(overlap_x) : 1.f;
                    const float x_f_1 = (circular_x || (overlap_x > 0 && x < (img_width - width))) ? (width - ix) / float(overlap_x) : 1.f;
                    const float y_f_0 = (circular_y || (overlap_y > 0 && y > 0)) ? (iy - y_skip) / float(overlap_y) : 1.f;
                    const float y_f_1 = (circular_y || (overlap_y > 0 && y < (img_height - height))) ? (height - iy) / float(overlap_y) : 1.f;
                    const float x_f   = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f   = std::min(std::min(y_f_0, y_f_1), 1.f);
                    (*output)[plane * output_plane + dst_xy] =
                        old_value + new_value * smootherstep_f32(y_f) * smootherstep_f32(x_f);
                } else {
                    (*output)[plane * output_plane + dst_xy] = new_value;
                }
            }
        }
    }
}

template <typename Fn>
__STATIC_INLINE__ sd::Tensor<float> process_tiles_2d(const sd::Tensor<float>& input,
                                                     int output_width,
                                                     int output_height,
                                                     int scale,
                                                     int p_tile_size_x,
                                                     int p_tile_size_y,
                                                     float tile_overlap_factor,
                                                     bool circular_x,
                                                     bool circular_y,
                                                     Fn&& on_processing,
                                                     bool silent = false) {
    sd::Tensor<float> output;
    int input_width  = static_cast<int>(input.shape()[0]);
    int input_height = static_cast<int>(input.shape()[1]);

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;
    bool decode      = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor, circular_x);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor, circular_y);

    int tile_overlap_x     = static_cast<int32_t>(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;
    int tile_overlap_y     = static_cast<int32_t>(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;
    int tile_size_x        = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y        = p_tile_size_y < small_height ? p_tile_size_y : small_height;
    int input_tile_size_x  = tile_size_x;
    int input_tile_size_y  = tile_size_y;
    int output_tile_size_x = tile_size_x;
    int output_tile_size_y = tile_size_y;
    if (decode) {
        output_tile_size_x *= scale;
        output_tile_size_y *= scale;
    } else {
        input_tile_size_x *= scale;
        input_tile_size_y *= scale;
    }

    int num_tiles   = num_tiles_x * num_tiles_y;
    int tile_count  = 1;
    bool last_y     = false;
    bool last_x     = false;
    float last_time = 0.0f;
    if (!silent) {
        LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
        LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);
        LOG_DEBUG("processing %i tiles", num_tiles);
        pretty_progress(0, num_tiles, 0.0f);
    }
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (!circular_y && y + tile_size_y >= small_height) {
            int original_y = y;
            y              = small_height - tile_size_y;
            dy             = original_y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (!circular_x && x + tile_size_x >= small_width) {
                int original_x = x;
                x              = small_width - tile_size_x;
                dx             = original_x - x;
                if (decode) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = decode ? x : scale * x;
            int y_in  = decode ? y : scale * y;
            int x_out = decode ? x * scale : x;
            int y_out = decode ? y * scale : y;

            int overlap_x_out = decode ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = decode ? tile_overlap_y * scale : tile_overlap_y;

            int64_t t1       = ggml_time_ms();
            auto input_tile  = sd_tensor_split_2d(input, input_tile_size_x, input_tile_size_y, x_in, y_in);
            auto output_tile = on_processing(input_tile);
            if (output_tile.empty()) {
                return {};
            }
            GGML_ASSERT(output_tile.shape()[0] == output_tile_size_x && output_tile.shape()[1] == output_tile_size_y);
            if (output.empty()) {
                std::vector<int64_t> output_shape = output_tile.shape();
                output_shape[0]                   = output_width;
                output_shape[1]                   = output_height;
                output                            = sd::Tensor<float>::zeros(std::move(output_shape));
            }
            sd_tensor_merge_2d(output_tile, &output, x_out, y_out, overlap_x_out, overlap_y_out, circular_x, circular_y, dx, dy);

            if (!silent) {
                int64_t t2 = ggml_time_ms();
                last_time  = (t2 - t1) / 1000.0f;
                pretty_progress(tile_count, num_tiles, last_time);
            }
            tile_count++;
        }
        last_x = false;
    }
    if (!silent && tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    if (output.empty()) {
        return {};
    }
    return output;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm_32(ggml_context* ctx,
                                                      ggml_tensor* a) {
    const float eps = 1e-6f;  // default eps parameter
    return ggml_group_norm(ctx, a, 32, eps);
}

__STATIC_INLINE__ bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
    return x->nb[0] == ggml_type_size(x->type) &&
           x->nb[2] == x->nb[1] * x->ne[1] &&
           x->nb[3] == x->nb[2] * x->ne[2];
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                                              ggml_tensor* x,
                                              float factor,
                                              bool inplace = false) {
    if (!ggml_ext_is_padded_1d(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu(ggml_context* ctx,
                                             ggml_tensor* x,
                                             bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_inplace(ctx, x);
    } else {
        x = ggml_gelu(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_gelu_quick(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   bool inplace = false) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_quick_inplace(ctx, x);
    } else {
        x = ggml_gelu_quick(ctx, x);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* w,
                                               ggml_tensor* b,
                                               bool force_prec_f32 = false,
                                               float scale         = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad_ext(ggml_context* ctx,
                                                ggml_tensor* x,
                                                int lp0,
                                                int rp0,
                                                int lp1,
                                                int rp1,
                                                int lp2,
                                                int rp2,
                                                int lp3,
                                                int rp3,
                                                bool circular_x = false,
                                                bool circular_y = false) {
    if (circular_x && circular_y) {
        return ggml_pad_ext_circular(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }

    if (circular_x && (lp0 != 0 || rp0 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, lp0, rp0, 0, 0, 0, 0, 0, 0);
        lp0 = rp0 = 0;
    }
    if (circular_y && (lp1 != 0 || rp1 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, 0, 0, lp1, rp1, 0, 0, 0, 0);
        lp1 = rp1 = 0;
    }

    if (lp0 != 0 || rp0 != 0 || lp1 != 0 || rp1 != 0 || lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0) {
        x = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_pad(ggml_context* ctx,
                                            ggml_tensor* x,
                                            int p0,
                                            int p1,
                                            int p2          = 0,
                                            int p3          = 0,
                                            bool circular_x = false,
                                            bool circular_y = false) {
    return ggml_ext_pad_ext(ctx, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_2d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int s0          = 1,
                                                int s1          = 1,
                                                int p0          = 0,
                                                int p1          = 0,
                                                int d0          = 1,
                                                int d1          = 1,
                                                bool direct     = false,
                                                bool circular_x = false,
                                                bool circular_y = false,
                                                float scale     = 1.f) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(ctx, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N*OC, OD, OH, OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* w,
                                                ggml_tensor* b,
                                                int64_t IC,
                                                int s0              = 1,
                                                int s1              = 1,
                                                int s2              = 1,
                                                int p0              = 0,
                                                int p1              = 0,
                                                int p2              = 0,
                                                int d0              = 1,
                                                int d1              = 1,
                                                int d2              = 1,
                                                bool force_prec_f32 = false) {
    if (force_prec_f32) {
        ggml_tensor* im2col = ggml_im2col_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, w->type);

        int64_t OC = w->ne[3] / IC;
        int64_t N  = x->ne[3] / IC;
        x          = ggml_mul_mat(ctx,
                                  ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                                  ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2] * IC, OC));
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);

        int64_t OD = im2col->ne[3] / N;
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1] * im2col->ne[2], OD, N, OC);
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1], im2col->ne[2], OD, OC * N);
    } else {
        x = ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);
    }

    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
__STATIC_INLINE__ ggml_tensor* ggml_ext_conv_3d_nx1x1(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* w,
                                                      ggml_tensor* b,
                                                      int s2 = 1,
                                                      int p2 = 1,
                                                      int d2 = 1) {
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;  // [N, OC, T, OH * OW]
}

// qkv: [N, L, 3*C]
// return: ([N, L, C], [N, L, C], [N, L, C])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_qkv(ggml_context* ctx,
                                                      ggml_tensor* qkv) {
    qkv = ggml_reshape_4d(ctx, qkv, qkv->ne[0] / 3, 3, qkv->ne[1], qkv->ne[2]);  // [N, L, 3, C]
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));                    // [3, N, L, C]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 0);  // [N, L, C]
    auto k         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 1);  // [N, L, C]
    auto v         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 2);  // [N, L, C]
    return {q, k, v};
}

// qkv: [N, 3*C, H, W]
// return: ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<ggml_tensor*> split_image_qkv(ggml_context* ctx,
                                                            ggml_tensor* qkv) {
    int64_t W   = qkv->ne[0];
    int64_t H   = qkv->ne[1];
    int64_t C   = qkv->ne[2] / 3;
    int64_t N   = qkv->ne[3];
    int64_t nb1 = qkv->nb[1];
    int64_t nb2 = qkv->nb[2];
    qkv         = ggml_reshape_4d(ctx, qkv, W * H, C, 3, N);                     // [N, 3, C, H*W]
    qkv         = ggml_cont(ctx, ggml_ext_torch_permute(ctx, qkv, 0, 1, 3, 2));  // [3, N, C, H*W]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 0);  // [N, C, H, W]
    auto k         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 1);  // [N, C, H, W]
    auto v         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 2);  // [N, C, H, W]
    return {q, k, v};
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_full(ggml_context* ctx,
                                             float value,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(ctx, one, value);             // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros(ggml_context* ctx,
                                              int64_t ne0,
                                              int64_t ne1,
                                              int64_t ne2,
                                              int64_t ne3) {
    return ggml_ext_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_zeros_like(ggml_context* ctx,
                                                   ggml_tensor* x) {
    return ggml_ext_zeros(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones(ggml_context* ctx,
                                             int64_t ne0,
                                             int64_t ne1,
                                             int64_t ne2,
                                             int64_t ne3) {
    return ggml_ext_full(ctx, 1.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_ones_like(ggml_context* ctx,
                                                  ggml_tensor* x) {
    return ggml_ext_ones(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_cast_f32(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* a) {
    if (sd_backend_is(backend, "Vulkan")) {
        auto zero_index = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:zero_int");
        auto out        = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out             = ggml_get_rows(ctx, out, zero_index);
        out             = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    } else {
        auto out         = ggml_reshape_2d(ctx, a, 1, ggml_nelements(a));
        ggml_tensor* one = ggml_ext_ones(ctx, 1, 1, 1, 1);  // [1,]
        if (ggml_is_transposed(out)) {
            out = ggml_mul_mat(ctx, one, out);
        } else {
            out = ggml_mul_mat(ctx, out, one);
        }
        out = ggml_reshape(ctx, out, a);
        return out;
    }
}

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
__STATIC_INLINE__ ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
                                                      ggml_backend_t backend,
                                                      ggml_tensor* q,
                                                      ggml_tensor* k,
                                                      ggml_tensor* v,
                                                      int64_t n_head,
                                                      ggml_tensor* mask = nullptr,
                                                      bool skip_reshape = false,
                                                      bool flash_attn   = false,
                                                      float kv_scale    = 1.0f) {  // avoid overflow
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    if (!skip_reshape) {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        C         = q->ne[0];
        N         = q->ne[2];
        d_head    = C / n_head;
        n_kv_head = k->ne[0] / d_head;

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);       // [N, L_q, n_head, d_head]
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);      // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);    // [N, L_k, n_kv_head, d_head]
        k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]

        v = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);  // [N, L_k, n_kv_head, d_head]
    } else {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        d_head    = v->ne[0];
        N         = v->ne[3];
        n_kv_head = k->ne[2] / N;
        C         = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    ggml_tensor* kqv = nullptr;

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) -> ggml_tensor* {
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        if (kv_scale != 1.0f) {
            v_in = ggml_ext_scale(ctx, v_in, kv_scale);
        }
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);

        if (mask_in != nullptr) {
            mask_in = ggml_transpose(ctx, mask_in);
        }

        if (mask_in != nullptr) {
            mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
        }

        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    if (flash_attn) {
        // LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (mask != nullptr) {
            // TODO: figure out if we can bend t5 to work too
            can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
        }

        if (can_use_flash_attn) {
            kqv = build_kqv(q, k, v, mask);
            if (!ggml_backend_supports_op(backend, kqv)) {
                kqv = nullptr;
            } else {
                kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
            }
        }
    }

    if (kqv == nullptr) {
        // if (flash_attn) {
        //     LOG_DEBUG("fallback to default attention, L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        // }
        v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_group_norm(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   int num_groups = 32) {
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ void ggml_ext_backend_tensor_get_and_sync(ggml_backend_t backend, const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if ((sd_backend_is(backend, "ROCm") || sd_backend_is(backend, "CUDA") || sd_backend_is(backend, "SYCL")) &&
        !sd_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
        return;
    }

    ggml_backend_tensor_get(tensor, data, offset, size);
}

__STATIC_INLINE__ float ggml_ext_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_BF16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else if (tensor->type == GGML_TYPE_BF16) {
        ggml_bf16_t bf16_value;
        ggml_backend_tensor_get(tensor, &bf16_value, 0, sizeof(bf16_value));
        value = ggml_bf16_to_fp32(bf16_value);
    } else if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
    } else {  // GGML_TYPE_I32
        int int32_value;
        ggml_backend_tensor_get(tensor, &int32_value, 0, sizeof(int32_value));
        value = (float)int32_value;
    }
    return value;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor(ggml_context* ctx,
                                                     const std::vector<float>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ ggml_tensor* vector_to_ggml_tensor_i32(ggml_context* ctx,
                                                         const std::vector<int>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ std::vector<float> arange(float start, float end, float step = 1.f) {
    std::vector<float> result;

    for (float value = start; value < end; value += step) {
        result.push_back(value);
    }

    return result;
}

// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
__STATIC_INLINE__ std::vector<float> timestep_embedding(std::vector<float> timesteps,
                                                        int dim,
                                                        int max_period       = 10000,
                                                        bool flip_sin_to_cos = true,
                                                        float scale          = 1.f) {
    // timesteps: [N,]
    // embedding: [N, dim]
    size_t N = timesteps.size();
    std::vector<float> embedding(N * dim, 0.f);
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = timesteps[i] * freqs[j] * scale;
            if (flip_sin_to_cos) {
                embedding[i * dim + j]        = std::cos(arg);
                embedding[i * dim + j + half] = std::sin(arg);
            } else {
                embedding[i * dim + j]        = std::sin(arg);
                embedding[i * dim + j + half] = std::cos(arg);
            }
        }
    }
    return embedding;
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              ggml_tensor* embedding,
                                              int dim,
                                              int max_period = 10000) {
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
}

__STATIC_INLINE__ void set_timestep_embedding(std::vector<float> timesteps,
                                              sd::Tensor<float>* embedding,
                                              int dim,
                                              int max_period = 10000) {
    GGML_ASSERT(embedding != nullptr);
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    if (embedding->numel() != static_cast<int64_t>(embedding_vec.size())) {
        embedding->resize({dim, static_cast<int64_t>(timesteps.size())});
    }
    std::copy(embedding_vec.begin(), embedding_vec.end(), embedding->values().begin());
}

__STATIC_INLINE__ ggml_tensor* new_timestep_embedding(ggml_context* ctx,
                                                      std::vector<float> timesteps,
                                                      int dim,
                                                      int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [N, dim]
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    ggml_tensor* embedding           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps.size());
    if (embedding->data != nullptr) {
        memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
    } else {
        ggml_backend_tensor_set(embedding, embedding_vec.data(), 0, ggml_nbytes(embedding));
    }
    return embedding;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_timestep_embedding(
    ggml_context* ctx,
    ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f) {
    timesteps = ggml_ext_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

__STATIC_INLINE__ ggml_tensor* ggml_ext_vec_concat(ggml_context* ctx,
                                                   std::vector<ggml_tensor*>& tensors,
                                                   int dim) {
    while (tensors.size() > 1) {
        std::vector<ggml_tensor*> next_level;
        for (size_t i = 0; i < tensors.size(); i += 2) {
            if (i + 1 < tensors.size()) {
                next_level.push_back(ggml_concat(ctx, tensors[i], tensors[i + 1], dim));
            } else {
                next_level.push_back(tensors[i]);
            }
        }
        tensors = std::move(next_level);
    }
    return tensors[0];
}

/* SDXL with LoRA requires more space */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680

struct WeightAdapter {
    struct ForwardParams {
        enum class op_type_t {
            OP_LINEAR,
            OP_CONV2D,
        } op_type;
        struct {
            bool force_prec_f32 = false;
            float scale         = 1.f;
        } linear;
        struct conv2d_params_t {
            int s0          = 1;
            int s1          = 1;
            int p0          = 0;
            int p1          = 0;
            int d0          = 1;
            int d1          = 1;
            bool direct     = false;
            bool circular_x = false;
            bool circular_y = false;
            float scale     = 1.f;
        } conv2d;
    };
    virtual ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name) = 0;
    virtual ggml_tensor* forward_with_lora(ggml_context* ctx,
                                           ggml_backend_t backend,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* b,
                                           const std::string& prefix,
                                           ForwardParams forward_params)                                                              = 0;
    virtual size_t get_extra_graph_size()                                                                                             = 0;
};

struct GGMLRunnerContext {
    ggml_backend_t backend                                           = nullptr;
    ggml_context* ggml_ctx                                           = nullptr;
    bool flash_attn_enabled                                          = false;
    bool conv2d_direct_enabled                                       = false;
    bool circular_x_enabled                                          = false;
    bool circular_y_enabled                                          = false;
    std::shared_ptr<WeightAdapter> weight_adapter                    = nullptr;
    std::vector<std::pair<ggml_tensor*, std::string>>* debug_tensors = nullptr;
    std::function<ggml_tensor*(const std::string&)> get_cache_tensor;
    std::function<void(const std::string&, ggml_tensor*)> cache_tensor;

    void capture_tensor(const std::string& name, ggml_tensor* tensor) {
        if (debug_tensors == nullptr || tensor == nullptr) {
            return;
        }
        ggml_tensor* snapshot = ggml_cont(ggml_ctx, tensor);
        ggml_tensor* dst      = ggml_dup_tensor(ggml_ctx, snapshot);
        snapshot              = ggml_cpy(ggml_ctx, snapshot, dst);
        ggml_set_output(snapshot);
        debug_tensors->push_back({snapshot, name});
    }

    ggml_tensor* load_cache_tensor(const std::string& name) const {
        if (!get_cache_tensor) {
            return nullptr;
        }
        return get_cache_tensor(name);
    }

    void persist_cache_tensor(const std::string& name, ggml_tensor* tensor) const {
        if (!cache_tensor || tensor == nullptr) {
            return;
        }
        cache_tensor(name, tensor);
    }
};

struct GGMLRunner {
protected:
    typedef std::function<ggml_cgraph*()> get_graph_cb_t;
    using GraphCutSegment = sd::ggml_graph_cut::Segment;
    using GraphCutPlan    = sd::ggml_graph_cut::Plan;

    ggml_backend_t params_backend  = nullptr;
    ggml_backend_t runtime_backend = nullptr;

    ggml_context* params_ctx                    = nullptr;
    ggml_backend_buffer_t params_buffer         = nullptr;
    ggml_context* offload_ctx                   = nullptr;
    ggml_backend_buffer_t runtime_params_buffer = nullptr;
    bool params_on_runtime_backend              = false;

    ggml_context* cache_ctx            = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;

    ggml_context* compute_ctx    = nullptr;
    ggml_gallocr* compute_allocr = nullptr;

    ggml_context* partial_offload_ctx                   = nullptr;
    ggml_backend_buffer_t partial_runtime_params_buffer = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> partial_offload_pairs;

    // Params kept on the runtime backend across streaming segments.
    ggml_context* resident_offload_ctx = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> resident_offload_pairs;
    ggml_backend_buffer_t resident_runtime_params_buffer = nullptr;
    std::unordered_set<ggml_tensor*> resident_param_set;
    uint64_t resident_state_token = 0;

    size_t max_graph_vram_bytes           = 0;
    bool stream_layers_enabled            = false;
    size_t observed_max_effective_budget_ = 0;

    sd::layer_registry::LayerRegistry layer_registry_;

    std::shared_ptr<WeightAdapter> weight_adapter = nullptr;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = nullptr;

    std::vector<int> zero_int_vec = {0};
    ggml_tensor* zero_int_tensor  = nullptr;

    std::map<ggml_tensor*, const void*> backend_tensor_data_map;
    std::map<std::string, ggml_tensor*> cache_tensor_map;  // name -> tensor
    std::vector<std::pair<ggml_tensor*, std::string>> debug_tensors;
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    bool flash_attn_enabled    = false;
    bool conv2d_direct_enabled = false;
    bool circular_x_enabled    = false;
    bool circular_y_enabled    = false;

    sd::ggml_graph_cut::PlanCache graph_cut_plan_cache_;
    std::unordered_set<const ggml_tensor*> params_tensor_set_;

    template <typename T>
    static sd::Tensor<T> take_or_empty(std::optional<sd::Tensor<T>> tensor) {
        if (!tensor.has_value()) {
            return {};
        }
        return std::move(*tensor);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(std::optional<sd::Tensor<T>> tensor,
                                                         size_t expected_dim) {
        return restore_trailing_singleton_dims(take_or_empty(std::move(tensor)), expected_dim);
    }

    template <typename T>
    static sd::Tensor<T> restore_trailing_singleton_dims(sd::Tensor<T> tensor,
                                                         size_t expected_dim) {
        if (tensor.empty()) {
            return tensor;
        }
        while (static_cast<size_t>(tensor.dim()) < expected_dim) {
            tensor.unsqueeze_(tensor.dim());
        }
        return tensor;
    }

    void alloc_params_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != nullptr);
        params_tensor_set_.clear();
        if (params_backend != runtime_backend) {
            offload_ctx = ggml_init(params);
            GGML_ASSERT(offload_ctx != nullptr);
        }
    }

    void free_params_ctx() {
        if (params_ctx != nullptr) {
            ggml_free(params_ctx);
            params_ctx = nullptr;
        }
        params_tensor_set_.clear();
        if (offload_ctx != nullptr) {
            ggml_free(offload_ctx);
            offload_ctx = nullptr;
        }
        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    void alloc_cache_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        cache_ctx = ggml_init(params);
        GGML_ASSERT(cache_ctx != nullptr);
    }

    void free_cache_ctx() {
        if (cache_ctx != nullptr) {
            ggml_free(cache_ctx);
            cache_ctx = nullptr;
        }
    }

    void alloc_compute_ctx() {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != nullptr);
    }

    void free_compute_ctx() {
        debug_tensors.clear();
        if (compute_ctx != nullptr) {
            ggml_free(compute_ctx);
            compute_ctx = nullptr;
        }
        backend_tensor_data_map.clear();
    }

    void rebuild_params_tensor_set() {
        params_tensor_set_.clear();
        if (params_ctx == nullptr) {
            return;
        }
        for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
            params_tensor_set_.insert(t);
        }
    }

    void prepare_build_in_tensor_before() {
        one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
        set_backend_tensor_data(one_tensor, one_vec.data());

        zero_int_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(zero_int_tensor, "ggml_runner_build_in_tensor:zero_int");
        set_backend_tensor_data(zero_int_tensor, zero_int_vec.data());
    }

    void prepare_build_in_tensor_after(ggml_cgraph* gf) {
        ggml_build_forward_expand(gf, one_tensor);
        ggml_build_forward_expand(gf, zero_int_tensor);
    }

    ggml_cgraph* new_graph_custom(size_t graph_size) {
        if (weight_adapter) {
            graph_size += weight_adapter->get_extra_graph_size();
        }
        return ggml_new_graph_custom(compute_ctx, graph_size, false);
    }

    ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph) {
        prepare_build_in_tensor_before();
        ggml_cgraph* gf = get_graph();
        if (ggml_graph_n_nodes(gf) > 0) {
            auto result = ggml_graph_node(gf, -1);
            ggml_set_name(result, final_result_name.c_str());
        }
        for (const auto& entry : debug_tensors) {
            if (entry.first != nullptr) {
                ggml_build_forward_expand(gf, entry.first);
            }
        }
        for (const auto& entry : cache_tensor_map) {
            if (entry.second != nullptr) {
                ggml_build_forward_expand(gf, entry.second);
            }
        }
        prepare_build_in_tensor_after(gf);
        return gf;
    }

    bool prepare_compute_graph(get_graph_cb_t get_graph,
                               ggml_cgraph** gf_out) {
        GGML_ASSERT(gf_out != nullptr);

        reset_compute_ctx();
        ggml_cgraph* gf = get_compute_graph(get_graph);
        if (gf == nullptr) {
            free_compute_ctx();
            return false;
        }

        *gf_out = gf;
        return true;
    }

    bool alloc_compute_buffer(ggml_cgraph* gf) {
        if (compute_allocr != nullptr) {
            return true;
        }
        compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_backend));

        if (!ggml_gallocr_reserve(compute_allocr, gf)) {
            // failed to allocate the compute buffer
            LOG_ERROR("%s: failed to allocate the compute buffer\n", get_desc().c_str());
            free_compute_buffer();
            return false;
        }

        // compute the required memory
        size_t compute_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 0);
        LOG_DEBUG("%s compute buffer size: %.2f MB(%s)",
                  get_desc().c_str(),
                  compute_buffer_size / 1024.0 / 1024.0,
                  sd_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM");
        return true;
    }

    void free_cache_buffer() {
        if (cache_buffer != nullptr) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = nullptr;
        }
    }

    bool copy_cache_tensors_to_cache_buffer(const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        ggml_context* old_cache_ctx            = cache_ctx;
        ggml_backend_buffer_t old_cache_buffer = cache_buffer;
        cache_ctx                              = nullptr;
        cache_buffer                           = nullptr;
        std::map<std::string, ggml_tensor*> merged_cache_sources;
        if (old_cache_ctx != nullptr) {
            for (ggml_tensor* tensor = ggml_get_first_tensor(old_cache_ctx); tensor != nullptr; tensor = ggml_get_next_tensor(old_cache_ctx, tensor)) {
                if (cache_keep_names != nullptr && cache_keep_names->find(tensor->name) == cache_keep_names->end()) {
                    continue;
                }
                merged_cache_sources[tensor->name] = tensor;
            }
        }
        for (const auto& kv : cache_tensor_map) {
            if (cache_keep_names != nullptr && cache_keep_names->find(kv.first) == cache_keep_names->end()) {
                continue;
            }
            merged_cache_sources[kv.first] = kv.second;
        }
        cache_tensor_map.clear();
        if (merged_cache_sources.empty()) {
            if (old_cache_buffer != nullptr) {
                ggml_backend_buffer_free(old_cache_buffer);
            }
            if (old_cache_ctx != nullptr) {
                ggml_free(old_cache_ctx);
            }
            return true;
        }

        alloc_cache_ctx();
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> source_to_cache_tensors;
        source_to_cache_tensors.reserve(merged_cache_sources.size());
        for (const auto& kv : merged_cache_sources) {
            ggml_tensor* source_tensor = sd::ggml_graph_cut::cache_source_tensor(kv.second);
            auto cache_tensor          = ggml_dup_tensor(cache_ctx, source_tensor);
            ggml_set_name(cache_tensor, kv.first.c_str());
            source_to_cache_tensors.push_back({source_tensor, cache_tensor});
        }
        size_t num_tensors = ggml_tensor_num(cache_ctx);
        cache_buffer       = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
        GGML_ASSERT(cache_buffer != nullptr);
        for (const auto& kv : source_to_cache_tensors) {
            ggml_tensor* src              = kv.first;
            ggml_tensor* dst              = kv.second;
            ggml_backend_buffer_t src_buf = sd::ggml_graph_cut::tensor_buffer(src);
            ggml_backend_buffer_t dst_buf = sd::ggml_graph_cut::tensor_buffer(dst);
            if (src_buf == nullptr || dst_buf == nullptr) {
                LOG_ERROR("%s cache copy tensor buffer missing: name=%s op=%s src0=%p src0_name=%s src0_buffer=%p src_buffer=%p src_view_src=%p src_view_src_buffer=%p dst_buffer=%p",
                          get_desc().c_str(),
                          src && src->name[0] != '\0' ? src->name : "<unnamed>",
                          src ? ggml_op_name(src->op) : "<null>",
                          src ? src->src[0] : nullptr,
                          (src && src->src[0] && src->src[0]->name[0] != '\0') ? src->src[0]->name : "<unnamed>",
                          (src && src->src[0]) ? sd::ggml_graph_cut::tensor_buffer(src->src[0]) : nullptr,
                          src ? src->buffer : nullptr,
                          src ? src->view_src : nullptr,
                          (src && src->view_src) ? src->view_src->buffer : nullptr,
                          dst ? dst->buffer : nullptr);
                return false;
            }
            const bool use_staging_copy = src->view_src != nullptr || !ggml_is_contiguous(src) || src->buffer == nullptr;
            if (use_staging_copy) {
                std::vector<uint8_t> host_data(ggml_nbytes(src));
                ggml_backend_tensor_get(src, host_data.data(), 0, host_data.size());
                ggml_backend_tensor_set(dst, host_data.data(), 0, host_data.size());
            } else {
                ggml_backend_tensor_copy(src, dst);
            }
        }
        ggml_backend_synchronize(runtime_backend);
        size_t cache_buffer_size = ggml_backend_buffer_get_size(cache_buffer);
        LOG_DEBUG("%s cache backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  cache_buffer_size / (1024.f * 1024.f),
                  sd_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM",
                  num_tensors);
        if (old_cache_buffer != nullptr) {
            ggml_backend_buffer_free(old_cache_buffer);
        }
        if (old_cache_ctx != nullptr) {
            ggml_free(old_cache_ctx);
        }
        return true;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> read_graph_tensor(ggml_tensor* tensor, const char* label) {
        if (tensor == nullptr) {
            LOG_ERROR("%s %s tensor is null", get_desc().c_str(), label);
            return std::nullopt;
        }
        if (tensor->type != sd::GGMLTypeTraits<T>::type) {
            LOG_ERROR("%s %s tensor type mismatch: got %s",
                      get_desc().c_str(),
                      label,
                      ggml_type_name(tensor->type));
            return std::nullopt;
        }
        ggml_backend_buffer_t buf = sd::ggml_graph_cut::tensor_buffer(tensor);
        if (buf == nullptr) {
            LOG_ERROR("%s %s tensor buffer missing: name=%s op=%s buffer=%p view_src=%p view_src_buffer=%p data=%p",
                      get_desc().c_str(),
                      label,
                      tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                      ggml_op_name(tensor->op),
                      tensor->buffer,
                      tensor->view_src,
                      tensor->view_src ? tensor->view_src->buffer : nullptr,
                      tensor->data);
            return std::nullopt;
        }

        sd::Tensor<T> result(sd::shape_from_ggml(tensor));
        if (tensor->view_src != nullptr || !ggml_is_contiguous(tensor) || tensor->buffer == nullptr) {
            ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
        } else {
            ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
        }
        return result;
    }

    void copy_data_to_backend_tensor(ggml_cgraph* gf, bool clear_after_copy = true) {
        GGML_ASSERT(gf != nullptr);
        std::unordered_set<const ggml_tensor*> graph_tensor_set;
        const int n_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_nodes = ggml_graph_n_nodes(gf);
        graph_tensor_set.reserve(static_cast<size_t>(n_leafs + n_nodes));
        for (int i = 0; i < n_leafs; ++i) {
            graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_nodes; ++i) {
            graph_tensor_set.insert(ggml_graph_node(gf, i));
        }

        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;
            if (tensor == nullptr || data == nullptr) {
                continue;
            }
            const char* name = ggml_get_name(tensor);
            if (graph_tensor_set.find(tensor) == graph_tensor_set.end()) {
                continue;
            }
            if (tensor->buffer == nullptr) {
                LOG_WARN("%s skip backend tensor copy: tensor buffer not set, name='%s', ne=[%lld,%lld,%lld,%lld], type=%s",
                         get_desc().c_str(),
                         name != nullptr ? name : "",
                         (long long)tensor->ne[0],
                         (long long)tensor->ne[1],
                         (long long)tensor->ne[2],
                         (long long)tensor->ne[3],
                         ggml_type_name(tensor->type));
                continue;
            }

            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (buf == nullptr) {
                LOG_WARN("%s graph exec skip tensor copy: name=%s op=%s reason=buffer_not_set data=%p view_src=%p view_src_buffer=%p",
                         get_desc().c_str(),
                         tensor && tensor->name[0] != '\0' ? tensor->name : "<unnamed>",
                         tensor ? ggml_op_name(tensor->op) : "<null>",
                         data,
                         tensor ? tensor->view_src : nullptr,
                         (tensor && tensor->view_src) ? tensor->view_src->buffer : nullptr);
                continue;
            }

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        if (clear_after_copy) {
            backend_tensor_data_map.clear();
        }
    }

    bool offload_all_params() {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (params_on_runtime_backend) {
            return true;
        }
        GGML_ASSERT(runtime_params_buffer == nullptr);
        int64_t t0         = ggml_time_ms();
        size_t num_tensors = ggml_tensor_num(offload_ctx);
        if (num_tensors == 0) {
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                GGML_ASSERT(t->view_src == nullptr);
                ggml_dup_tensor(offload_ctx, t);
            }
        }
        num_tensors = ggml_tensor_num(offload_ctx);
        GGML_ASSERT(num_tensors == ggml_tensor_num(params_ctx));

        runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);

        if (runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc runtime params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        ggml_backend_buffer_set_usage(runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            ggml_backend_tensor_copy(t, offload_t);
            std::swap(t->buffer, offload_t->buffer);
            std::swap(t->data, offload_t->data);
            std::swap(t->extra, offload_t->extra);

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        int64_t t1 = ggml_time_ms();

        size_t params_buffer_size = ggml_backend_buffer_get_size(runtime_params_buffer);
        LOG_INFO("%s offload params (%6.2f MB, %i tensors) to runtime backend (%s), taking %.2fs",
                 get_desc().c_str(),
                 params_buffer_size / (1024.f * 1024.f),
                 num_tensors,
                 ggml_backend_name(runtime_backend),
                 (t1 - t0) * 1.0f / 1000);

        params_on_runtime_backend = true;

        return true;
    }

    bool offload_partial_params(const std::vector<ggml_tensor*>& tensors) {
        restore_partial_params();
        if (params_backend == runtime_backend) {
            return true;
        }
        if (tensors.empty()) {
            return true;
        }
        GGML_ASSERT(!params_on_runtime_backend);
        GGML_ASSERT(partial_runtime_params_buffer == nullptr);

        std::vector<ggml_tensor*> unique_tensors;
        std::unordered_set<ggml_tensor*> seen_tensors;
        unique_tensors.reserve(tensors.size());
        seen_tensors.reserve(tensors.size());
        for (ggml_tensor* tensor : tensors) {
            if (tensor == nullptr) {
                continue;
            }
            if (resident_param_set.find(tensor) != resident_param_set.end()) {
                continue;
            }
            if (seen_tensors.insert(tensor).second) {
                unique_tensors.push_back(tensor);
            }
        }
        if (unique_tensors.empty()) {
            return true;
        }

        ggml_init_params params;
        params.mem_size   = std::max<size_t>(1, unique_tensors.size()) * ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        partial_offload_ctx = ggml_init(params);
        GGML_ASSERT(partial_offload_ctx != nullptr);

        partial_offload_pairs.clear();
        partial_offload_pairs.reserve(unique_tensors.size());

        for (ggml_tensor* tensor : unique_tensors) {
            GGML_ASSERT(tensor->view_src == nullptr);
            ggml_tensor* offload_tensor = ggml_dup_tensor(partial_offload_ctx, tensor);
            ggml_set_name(offload_tensor, tensor->name);
            partial_offload_pairs.push_back({tensor, offload_tensor});
        }

        partial_runtime_params_buffer = ggml_backend_alloc_ctx_tensors(partial_offload_ctx, runtime_backend);
        if (partial_runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc partial runtime params backend buffer failed, num_tensors = %zu",
                      get_desc().c_str(),
                      partial_offload_pairs.size());
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
            partial_offload_pairs.clear();
            return false;
        }
        ggml_backend_buffer_set_usage(partial_runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            ggml_backend_tensor_copy(tensor, offload_tensor);
            std::swap(tensor->buffer, offload_tensor->buffer);
            std::swap(tensor->data, offload_tensor->data);
            std::swap(tensor->extra, offload_tensor->extra);
        }

        size_t params_buffer_size = ggml_backend_buffer_get_size(partial_runtime_params_buffer);
        LOG_DEBUG("%s offload partial params (%6.2f MB, %zu tensors) to runtime backend (%s)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  partial_offload_pairs.size(),
                  ggml_backend_name(runtime_backend));

        return true;
    }

    void restore_all_params() {
        restore_partial_params();
        if (!params_on_runtime_backend) {
            return;
        }
        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != nullptr && offload_t != nullptr) {
            t->buffer         = offload_t->buffer;
            t->data           = offload_t->data;
            t->extra          = offload_t->extra;
            offload_t->buffer = nullptr;
            offload_t->data   = nullptr;
            offload_t->extra  = nullptr;

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        if (runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(runtime_params_buffer);
            runtime_params_buffer = nullptr;
        }
        params_on_runtime_backend = false;
    }

    void restore_partial_params() {
        if (partial_offload_pairs.empty()) {
            if (partial_runtime_params_buffer != nullptr) {
                ggml_backend_buffer_free(partial_runtime_params_buffer);
                partial_runtime_params_buffer = nullptr;
            }
            if (partial_offload_ctx != nullptr) {
                ggml_free(partial_offload_ctx);
                partial_offload_ctx = nullptr;
            }
            return;
        }

        for (auto& pair : partial_offload_pairs) {
            ggml_tensor* tensor         = pair.first;
            ggml_tensor* offload_tensor = pair.second;

            tensor->buffer         = offload_tensor->buffer;
            tensor->data           = offload_tensor->data;
            tensor->extra          = offload_tensor->extra;
            offload_tensor->buffer = nullptr;
            offload_tensor->data   = nullptr;
            offload_tensor->extra  = nullptr;
        }

        if (partial_runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(partial_runtime_params_buffer);
            partial_runtime_params_buffer = nullptr;
        }
        partial_offload_pairs.clear();

        if (partial_offload_ctx != nullptr) {
            ggml_free(partial_offload_ctx);
            partial_offload_ctx = nullptr;
        }
    }

    bool offload_resident_params(const std::vector<ggml_tensor*>& tensors) {
        if (params_backend == runtime_backend) {
            return true;
        }
        if (tensors.empty()) {
            return true;
        }
        GGML_ASSERT(resident_runtime_params_buffer == nullptr);
        GGML_ASSERT(resident_offload_ctx == nullptr);
        GGML_ASSERT(resident_offload_pairs.empty());
        GGML_ASSERT(resident_param_set.empty());

        std::vector<ggml_tensor*> unique_tensors;
        std::unordered_set<ggml_tensor*> seen;
        unique_tensors.reserve(tensors.size());
        seen.reserve(tensors.size());
        for (ggml_tensor* t : tensors) {
            if (t == nullptr)
                continue;
            if (seen.insert(t).second)
                unique_tensors.push_back(t);
        }
        if (unique_tensors.empty())
            return true;

        ggml_init_params init = {};
        init.mem_size         = std::max<size_t>(1, unique_tensors.size()) * ggml_tensor_overhead();
        init.mem_buffer       = nullptr;
        init.no_alloc         = true;
        resident_offload_ctx  = ggml_init(init);
        GGML_ASSERT(resident_offload_ctx != nullptr);

        resident_offload_pairs.reserve(unique_tensors.size());
        for (ggml_tensor* t : unique_tensors) {
            GGML_ASSERT(t->view_src == nullptr);
            ggml_tensor* twin = ggml_dup_tensor(resident_offload_ctx, t);
            ggml_set_name(twin, t->name);
            resident_offload_pairs.push_back({t, twin});
        }

        resident_runtime_params_buffer = ggml_backend_alloc_ctx_tensors(resident_offload_ctx, runtime_backend);
        if (resident_runtime_params_buffer == nullptr) {
            LOG_ERROR("%s alloc resident runtime params backend buffer failed, num_tensors = %zu",
                      get_desc().c_str(), resident_offload_pairs.size());
            ggml_free(resident_offload_ctx);
            resident_offload_ctx = nullptr;
            resident_offload_pairs.clear();
            return false;
        }
        ggml_backend_buffer_set_usage(resident_runtime_params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (auto& pair : resident_offload_pairs) {
            ggml_tensor* t    = pair.first;
            ggml_tensor* twin = pair.second;
            ggml_backend_tensor_copy(t, twin);
            std::swap(t->buffer, twin->buffer);
            std::swap(t->data, twin->data);
            std::swap(t->extra, twin->extra);
            resident_param_set.insert(t);
        }
        ggml_backend_synchronize(runtime_backend);

        size_t sz = ggml_backend_buffer_get_size(resident_runtime_params_buffer);
        LOG_INFO("%s offload resident params (%6.2f MB, %zu tensors) to runtime backend (%s)",
                 get_desc().c_str(),
                 sz / (1024.f * 1024.f),
                 resident_offload_pairs.size(),
                 ggml_backend_name(runtime_backend));
        return true;
    }

    void restore_resident_params() {
        if (resident_offload_pairs.empty()) {
            if (resident_runtime_params_buffer != nullptr) {
                ggml_backend_buffer_free(resident_runtime_params_buffer);
                resident_runtime_params_buffer = nullptr;
            }
            if (resident_offload_ctx != nullptr) {
                ggml_free(resident_offload_ctx);
                resident_offload_ctx = nullptr;
            }
            resident_param_set.clear();
            resident_state_token = 0;
            return;
        }
        for (auto& pair : resident_offload_pairs) {
            ggml_tensor* t    = pair.first;
            ggml_tensor* twin = pair.second;
            t->buffer         = twin->buffer;
            t->data           = twin->data;
            t->extra          = twin->extra;
            twin->buffer      = nullptr;
            twin->data        = nullptr;
            twin->extra       = nullptr;
        }
        if (resident_runtime_params_buffer != nullptr) {
            ggml_backend_buffer_free(resident_runtime_params_buffer);
            resident_runtime_params_buffer = nullptr;
        }
        resident_offload_pairs.clear();
        if (resident_offload_ctx != nullptr) {
            ggml_free(resident_offload_ctx);
            resident_offload_ctx = nullptr;
        }
        resident_param_set.clear();
        resident_state_token = 0;
    }

    bool should_use_graph_cut_segmented_compute(const GraphCutPlan& plan) {
        return plan.has_cuts &&
               plan.valid &&
               max_graph_vram_bytes > 0 &&
               plan.segments.size() > 1 &&
               params_backend != runtime_backend &&
               !sd_backend_is_cpu(runtime_backend);
    }

    bool can_attempt_graph_cut_segmented_compute() const {
        return max_graph_vram_bytes > 0 &&
               params_backend != runtime_backend &&
               !sd_backend_is_cpu(runtime_backend);
    }

    bool resolve_graph_cut_plan(ggml_cgraph* gf,
                                GraphCutPlan* plan_out,
                                size_t* effective_budget_out = nullptr) {
        GGML_ASSERT(plan_out != nullptr);
        GGML_ASSERT(gf != nullptr);

        // Keep the plan and resident params under the same live-VRAM cap.
        // Add back our own resident buffer so we don't see chunk-K's
        // allocation as "taken" VRAM and shrink the budget on every step.
        size_t effective_budget = max_graph_vram_bytes;
        if (stream_layers_enabled && max_graph_vram_bytes > 0 && runtime_backend != nullptr) {
            ggml_backend_dev_t dev = ggml_backend_get_device(runtime_backend);
            if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                size_t free_vram = 0, total_vram = 0;
                ggml_backend_dev_memory(dev, &free_vram, &total_vram);
                if (resident_runtime_params_buffer != nullptr) {
                    free_vram += ggml_backend_buffer_get_size(resident_runtime_params_buffer);
                }
                constexpr size_t safety_margin = 512ull * 1024 * 1024;
                size_t free_clamp              = (free_vram > safety_margin) ? (free_vram - safety_margin) : 0;
                if (free_clamp < effective_budget) {
                    LOG_DEBUG("%s clamping streaming budget: actual free VRAM %.2f MB < user cap %.2f MB",
                              get_desc().c_str(),
                              free_clamp / (1024.0 * 1024.0),
                              effective_budget / (1024.0 * 1024.0));
                    effective_budget = free_clamp;
                }
            }
        }

        bool budget_increased = false;
        if (stream_layers_enabled) {
            if (effective_budget > observed_max_effective_budget_) {
                observed_max_effective_budget_ = effective_budget;
                budget_increased               = true;
            } else {
                effective_budget = observed_max_effective_budget_;
            }
        }

        if (effective_budget_out != nullptr) {
            *effective_budget_out = effective_budget;
        }

        // When streaming and the model dwarfs the budget, cap the planner at
        // a quarter so it builds smaller merged segments and chunk-K can fit
        // alongside. Without streaming the cap only adds dispatch overhead.
        size_t planner_budget = effective_budget;
        if (stream_layers_enabled) {
            size_t total_params_bytes = 0;
            for (const ggml_tensor* t : params_tensor_set_) {
                if (t != nullptr) {
                    total_params_bytes += ggml_nbytes(t);
                }
            }
            if (total_params_bytes * 4 > effective_budget * 3) {
                planner_budget = effective_budget / 4;
            }
        }

        *plan_out = sd::ggml_graph_cut::resolve_plan(runtime_backend,
                                                     gf,
                                                     &graph_cut_plan_cache_,
                                                     planner_budget,
                                                     params_tensor_set_,
                                                     get_desc().c_str());
        if (stream_layers_enabled) {
            if (budget_increased) {
                LOG_INFO("%s streaming budget = %.2f MB",
                         get_desc().c_str(),
                         effective_budget / (1024.0 * 1024.0));
            } else {
                LOG_DEBUG("%s streaming budget = %.2f MB",
                          get_desc().c_str(),
                          effective_budget / (1024.0 * 1024.0));
            }
        }
        return true;
    }

    struct PersistentExternalBinding {
        ggml_backend_buffer_t buffer = nullptr;
        void* data                   = nullptr;
        void* extra                  = nullptr;
    };

    void snapshot_persistent_externals(const sd::ggml_graph_cut::Plan& plan,
                                       ggml_cgraph* gf,
                                       std::unordered_map<ggml_tensor*, PersistentExternalBinding>& out) {
        GGML_ASSERT(gf != nullptr);
        out.clear();
        for (const auto& segment : plan.segments) {
            for (const auto& input : segment.input_refs) {
                if (input.type != GraphCutSegment::INPUT_EXTERNAL) {
                    continue;
                }
                ggml_tensor* tensor = sd::ggml_graph_cut::input_tensor(gf, input);
                if (tensor == nullptr || tensor->buffer == nullptr) {
                    continue;
                }
                PersistentExternalBinding binding;
                binding.buffer = tensor->buffer;
                binding.data   = tensor->data;
                binding.extra  = tensor->extra;
                out[tensor]    = binding;
            }
        }
    }

    void reset_segment_runtime_tensors(const GraphCutSegment& segment,
                                       ggml_cgraph* gf,
                                       const std::unordered_map<ggml_tensor*, PersistentExternalBinding>* persistent_externals = nullptr) {
        GGML_ASSERT(gf != nullptr);

        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT:
                    input_tensor->buffer = nullptr;
                    input_tensor->data   = nullptr;
                    input_tensor->extra  = nullptr;
                    break;
                case GraphCutSegment::INPUT_EXTERNAL: {
                    if (persistent_externals != nullptr) {
                        auto it = persistent_externals->find(input_tensor);
                        if (it != persistent_externals->end()) {
                            input_tensor->buffer = it->second.buffer;
                            input_tensor->data   = it->second.data;
                            input_tensor->extra  = it->second.extra;
                            break;
                        }
                    }
                    input_tensor->buffer = nullptr;
                    input_tensor->data   = nullptr;
                    input_tensor->extra  = nullptr;
                    break;
                }
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }

        for (int node_idx : segment.internal_node_indices) {
            ggml_tensor* node = ggml_graph_node(gf, node_idx);
            if (node == nullptr) {
                continue;
            }
            node->buffer = nullptr;
            node->data   = nullptr;
            node->extra  = nullptr;
        }
    }

    bool bind_segment_cached_inputs(ggml_cgraph* gf, const GraphCutSegment& segment) {
        GGML_ASSERT(gf != nullptr);
        for (const auto& input : segment.input_refs) {
            ggml_tensor* input_tensor = sd::ggml_graph_cut::input_tensor(gf, input);
            if (input_tensor == nullptr) {
                continue;
            }
            switch (input.type) {
                case GraphCutSegment::INPUT_PREVIOUS_CUT: {
                    ggml_tensor* cache_tensor = get_cache_tensor_by_name(input.display_name);
                    if (cache_tensor == nullptr) {
                        LOG_ERROR("%s missing graph cut cache tensor: %s",
                                  get_desc().c_str(),
                                  input.display_name.c_str());
                        return false;
                    }
                    if (input_tensor->view_src != nullptr) {
                        input_tensor->view_src = cache_tensor;
                        input_tensor->buffer   = nullptr;
                        input_tensor->data     = cache_tensor->data == nullptr
                                                     ? nullptr
                                                     : static_cast<void*>(static_cast<char*>(cache_tensor->data) + input_tensor->view_offs);
                        input_tensor->extra    = cache_tensor->extra;
                    } else {
                        input_tensor->buffer = cache_tensor->buffer;
                        input_tensor->data   = cache_tensor->data;
                        input_tensor->extra  = cache_tensor->extra;
                    }
                    for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                        input_tensor->src[src_idx] = nullptr;
                    }
                    input_tensor->op = GGML_OP_NONE;
                    break;
                }
                case GraphCutSegment::INPUT_EXTERNAL:
                case GraphCutSegment::INPUT_PARAM:
                    break;
            }
        }
        return true;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> execute_graph(ggml_cgraph* gf,
                                               int n_threads,
                                               bool free_compute_buffer_immediately,
                                               const std::vector<ggml_tensor*>& runtime_param_tensors,
                                               bool preserve_backend_tensor_data_map,
                                               bool no_return                                          = false,
                                               const std::unordered_set<std::string>* cache_keep_names = nullptr) {
        int64_t t_execute_begin              = ggml_time_ms();
        const bool use_partial_param_offload = !runtime_param_tensors.empty();
        int64_t t_offload_begin              = ggml_time_ms();
        if (use_partial_param_offload) {
            if (!offload_partial_params(runtime_param_tensors)) {
                LOG_ERROR("%s offload partial params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        } else {
            if (!offload_all_params()) {
                LOG_ERROR("%s offload params to runtime backend failed", get_desc().c_str());
                return std::nullopt;
            }
        }
        int64_t t_offload_end = ggml_time_ms();

        int64_t t_alloc_begin = ggml_time_ms();
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        if (!ggml_gallocr_alloc_graph(compute_allocr, gf)) {
            LOG_ERROR("%s alloc compute graph failed", get_desc().c_str());
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        int64_t t_alloc_end = ggml_time_ms();

        int64_t t_copy_begin = ggml_time_ms();
        copy_data_to_backend_tensor(gf, !preserve_backend_tensor_data_map);
        int64_t t_copy_end = ggml_time_ms();
        if (sd_backend_is_cpu(runtime_backend)) {
            sd_backend_cpu_set_n_threads(runtime_backend, n_threads);
        }

        int64_t t_compute_begin = ggml_time_ms();
        ggml_status status      = ggml_backend_graph_compute(runtime_backend, gf);
        int64_t t_compute_end   = ggml_time_ms();
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERROR("%s compute failed: %s", get_desc().c_str(), ggml_status_to_string(status));
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }

        std::unordered_set<const ggml_tensor*> debug_graph_tensor_set;
        const int n_debug_leafs = sd::ggml_graph_cut::leaf_count(gf);
        const int n_debug_nodes = ggml_graph_n_nodes(gf);
        debug_graph_tensor_set.reserve(static_cast<size_t>(n_debug_leafs + n_debug_nodes));
        for (int i = 0; i < n_debug_leafs; ++i) {
            debug_graph_tensor_set.insert(sd::ggml_graph_cut::leaf_tensor(gf, i));
        }
        for (int i = 0; i < n_debug_nodes; ++i) {
            debug_graph_tensor_set.insert(ggml_graph_node(gf, i));
        }

        for (const auto& entry : debug_tensors) {
            auto tensor = entry.first;
            if (tensor == nullptr) {
                continue;
            }
            if (debug_graph_tensor_set.find(tensor) == debug_graph_tensor_set.end()) {
                continue;
            }
            ggml_backend_buffer_t tensor_buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
            if (tensor_buf == nullptr) {
                LOG_WARN("%s skip debug tensor '%s': tensor buffer not set",
                         get_desc().c_str(),
                         entry.second.c_str());
                continue;
            }
            if (tensor->type != GGML_TYPE_F32) {
                LOG_WARN("%s skip debug tensor '%s': only GGML_TYPE_F32 is supported, got %s",
                         get_desc().c_str(),
                         entry.second.c_str(),
                         ggml_type_name(tensor->type));
                continue;
            }
            auto debug_tensor = sd::make_sd_tensor_from_ggml<float>(tensor);
            print_sd_tensor(debug_tensor, false, entry.second.c_str());
        }

        int64_t t_cache_begin = ggml_time_ms();
        if (!copy_cache_tensors_to_cache_buffer(cache_keep_names)) {
            if (free_compute_buffer_immediately) {
                free_compute_buffer();
            } else if (use_partial_param_offload) {
                restore_partial_params();
            }
            return std::nullopt;
        }
        int64_t t_cache_end = ggml_time_ms();
        auto result         = ggml_get_tensor(compute_ctx, final_result_name.c_str());
        std::optional<sd::Tensor<T>> output;
        if (!no_return) {
            output = read_graph_tensor<T>(result, "output");
            if (!output.has_value()) {
                if (free_compute_buffer_immediately) {
                    free_compute_buffer();
                } else if (use_partial_param_offload) {
                    restore_partial_params();
                }
                return std::nullopt;
            }
        } else {
            output = sd::Tensor<T>();
        }

        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        } else if (use_partial_param_offload) {
            restore_partial_params();
        }
        if (use_partial_param_offload) {
            LOG_DEBUG("%s execute_graph timing: offload=%lld ms alloc=%lld ms copy_in=%lld ms compute=%lld ms cache=%lld ms total=%lld ms",
                      get_desc().c_str(),
                      t_offload_end - t_offload_begin,
                      t_alloc_end - t_alloc_begin,
                      t_copy_end - t_copy_begin,
                      t_compute_end - t_compute_begin,
                      t_cache_end - t_cache_begin,
                      ggml_time_ms() - t_execute_begin);
        }
        return output;
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute_with_graph_cuts(ggml_cgraph* gf,
                                                         const GraphCutPlan& plan,
                                                         int n_threads,
                                                         bool free_compute_buffer_immediately,
                                                         bool no_return = false) {
        GGML_ASSERT(gf != nullptr);

        free_compute_buffer();
        free_cache_ctx_and_buffer();

        std::unordered_map<ggml_tensor*, PersistentExternalBinding> persistent_externals;
        snapshot_persistent_externals(plan, gf, persistent_externals);

        std::optional<sd::Tensor<T>> output = sd::Tensor<T>();
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); ++seg_idx) {
            int64_t t_segment_begin = ggml_time_ms();
            const auto& segment     = plan.segments[seg_idx];
            auto future_cut_names   = sd::ggml_graph_cut::collect_future_input_names(gf, plan, seg_idx);
            LOG_DEBUG("%s graph cut executing segment %zu/%zu: %s",
                      get_desc().c_str(),
                      seg_idx + 1,
                      plan.segments.size(),
                      segment.group_name.c_str());

            reset_segment_runtime_tensors(segment, gf, &persistent_externals);
            if (!bind_segment_cached_inputs(gf, segment)) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }

            const bool is_last_segment = seg_idx + 1 == plan.segments.size();
            if (!is_last_segment) {
                for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
                    ggml_tensor* output_tensor = sd::ggml_graph_cut::output_tensor(gf, segment, output_idx);
                    if (output_tensor != nullptr &&
                        sd::ggml_graph_cut::is_graph_cut_tensor(output_tensor) &&
                        future_cut_names.find(output_tensor->name) != future_cut_names.end()) {
                        cache(output_tensor->name, output_tensor);
                    }
                }
            }

            ggml_context* segment_graph_ctx = nullptr;
            ggml_cgraph* segment_graph      = sd::ggml_graph_cut::build_segment_graph(gf, segment, &segment_graph_ctx);
            auto segment_output             = execute_graph<T>(segment_graph,
                                                   n_threads,
                                                   true,
                                                   sd::ggml_graph_cut::runtime_param_tensors(gf, segment, get_desc().c_str()),
                                                   true,
                                                   !is_last_segment || no_return,
                                                   &future_cut_names);
            ggml_free(segment_graph_ctx);
            if (!segment_output.has_value()) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }
            output = std::move(segment_output);
        }

        backend_tensor_data_map.clear();
        free_cache_ctx_and_buffer();
        free_compute_ctx();
        return output;
    }

public:
    void release_streaming_residency() {
        restore_resident_params();
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute_streaming_segments(ggml_cgraph* gf,
                                                            const GraphCutPlan& plan,
                                                            size_t residency_budget_bytes,
                                                            int n_threads,
                                                            bool free_compute_buffer_immediately,
                                                            bool no_return = false) {
        GGML_ASSERT(gf != nullptr);

        // Runtime LoRA composes `weight + diff` in the compute graph via
        // ggml_add; the resident weight tensor's data is never mutated, so
        // chunk-K residency stays valid across sampling steps.
        // Reserve room for the worst merged segment so chunk-K can't grow
        // large enough to starve later partial-param allocations.
        size_t worst_merged_segment_footprint = 0;
        for (const auto& seg : plan.segments) {
            const size_t fp = seg.input_param_bytes +
                              seg.compute_buffer_size +
                              seg.output_bytes +
                              seg.input_previous_cut_bytes +
                              seg.input_external_bytes;
            if (fp > worst_merged_segment_footprint) {
                worst_merged_segment_footprint = fp;
            }
        }
        const size_t residency_budget_for_annotate =
            residency_budget_bytes > worst_merged_segment_footprint
                ? residency_budget_bytes - worst_merged_segment_footprint
                : 0;

        sd::ggml_graph_cut::Plan& base_plan = graph_cut_plan_cache_.graph_cut_plan;
        if (base_plan.available) {
            sd::ggml_graph_cut::annotate_residency(base_plan, residency_budget_for_annotate);

            std::vector<ggml_tensor*> resident_params;
            uint64_t token = 0;
            for (const auto& segment : base_plan.segments) {
                if (segment.residency != sd::ggml_graph_cut::SegmentResidency::RESIDENT) {
                    continue;
                }
                auto seg_params = sd::ggml_graph_cut::param_tensors(gf, segment);
                for (ggml_tensor* t : seg_params) {
                    if (t == nullptr)
                        continue;
                    resident_params.push_back(t);
                    token ^= reinterpret_cast<uintptr_t>(t) * 0x9E3779B97F4A7C15ull;
                }
            }
            if (token != resident_state_token) {
                restore_resident_params();
                if (!resident_params.empty()) {
                    if (offload_resident_params(resident_params)) {
                        resident_state_token = token;
                    } else {
                        LOG_ERROR("%s chunk-K: resident offload failed; continuing with per-segment streaming",
                                  get_desc().c_str());
                        restore_resident_params();
                    }
                }
            }
        }

        free_compute_buffer();
        free_cache_ctx_and_buffer();

        layer_registry_.move_layer_to_gpu("_global");

        std::unordered_map<ggml_tensor*, PersistentExternalBinding> persistent_externals;
        snapshot_persistent_externals(plan, gf, persistent_externals);

        std::optional<sd::Tensor<T>> output = sd::Tensor<T>();
        for (size_t seg_idx = 0; seg_idx < plan.segments.size(); ++seg_idx) {
            int64_t t_segment_begin = ggml_time_ms();
            const auto& segment     = plan.segments[seg_idx];
            const bool is_last      = seg_idx + 1 == plan.segments.size();
            auto future_cut_names   = sd::ggml_graph_cut::collect_future_input_names(gf, plan, seg_idx);

            LOG_DEBUG("%s streaming-cut executing segment %zu/%zu: %s (residency=%s)",
                      get_desc().c_str(),
                      seg_idx + 1,
                      plan.segments.size(),
                      segment.group_name.c_str(),
                      segment.residency == sd::ggml_graph_cut::SegmentResidency::RESIDENT ? "RESIDENT" : "STREAMED");

            if (!layer_registry_.move_layer_to_gpu(segment.group_name)) {
                LOG_DEBUG("%s streaming: no registry entry for group '%s' (using upstream offload path)",
                          get_desc().c_str(),
                          segment.group_name.c_str());
            }

            reset_segment_runtime_tensors(segment, gf, &persistent_externals);
            if (!bind_segment_cached_inputs(gf, segment)) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }

            if (!is_last) {
                for (size_t output_idx = 0; output_idx < segment.output_node_indices.size(); ++output_idx) {
                    ggml_tensor* out_tensor = sd::ggml_graph_cut::output_tensor(gf, segment, output_idx);
                    if (out_tensor != nullptr &&
                        sd::ggml_graph_cut::is_graph_cut_tensor(out_tensor) &&
                        future_cut_names.find(out_tensor->name) != future_cut_names.end()) {
                        cache(out_tensor->name, out_tensor);
                    }
                }
            }

            ggml_context* segment_graph_ctx = nullptr;
            ggml_cgraph* segment_graph      = sd::ggml_graph_cut::build_segment_graph(gf, segment, &segment_graph_ctx);
            auto segment_output             = execute_graph<T>(segment_graph,
                                                   n_threads,
                                                   /*free_compute_buffer_immediately=*/true,
                                                   sd::ggml_graph_cut::runtime_param_tensors(gf, segment, get_desc().c_str()),
                                                   /*preserve_backend_tensor_data_map=*/true,
                                                   /*no_return=*/!is_last || no_return,
                                                   &future_cut_names);
            ggml_free(segment_graph_ctx);
            if (!segment_output.has_value()) {
                free_cache_ctx_and_buffer();
                free_compute_buffer();
                free_compute_ctx();
                return std::nullopt;
            }
            output = std::move(segment_output);

            if (segment.residency == sd::ggml_graph_cut::SegmentResidency::STREAMED) {
                layer_registry_.move_layer_to_cpu(segment.group_name);
            }
            (void)t_segment_begin;
        }

        backend_tensor_data_map.clear();
        free_cache_ctx_and_buffer();
        free_compute_ctx();
        return output;
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend, ggml_backend_t params_backend)
        : params_backend(params_backend),
          runtime_backend(backend) {
        GGML_ASSERT(runtime_backend != nullptr);
        GGML_ASSERT(params_backend != nullptr);
        alloc_params_ctx();
        layer_registry_.set_backends(runtime_backend, params_backend);
    }

    virtual ~GGMLRunner() {
        restore_resident_params();
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
        free_cache_ctx_and_buffer();
    }

    virtual GGMLRunnerContext get_context() {
        GGMLRunnerContext runner_ctx;
        runner_ctx.ggml_ctx              = compute_ctx;
        runner_ctx.backend               = runtime_backend;
        runner_ctx.flash_attn_enabled    = flash_attn_enabled;
        runner_ctx.conv2d_direct_enabled = conv2d_direct_enabled;
        runner_ctx.circular_x_enabled    = circular_x_enabled;
        runner_ctx.circular_y_enabled    = circular_y_enabled;
        runner_ctx.weight_adapter        = weight_adapter;
        runner_ctx.debug_tensors         = &debug_tensors;
        runner_ctx.get_cache_tensor      = [this](const std::string& name) {
            return this->get_cache_tensor_by_name(name);
        };
        runner_ctx.cache_tensor = [this](const std::string& name, ggml_tensor* tensor) {
            this->cache(name, tensor);
        };
        return runner_ctx;
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        if (num_tensors > 0) {
            // ggml_backend_alloc_ctx_tensors fails when all tensors are already allocated
            // (typical for memory-mapped weights). See ggml-alloc.c n_buffers==0 branch.
            bool all_have_data = true;
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != nullptr; t = ggml_get_next_tensor(params_ctx, t)) {
                if (t->data == nullptr) {
                    all_have_data = false;
                    break;
                }
            }
            if (all_have_data) {
                LOG_DEBUG("%s all params already mmap-allocated (no separate buffer needed)", get_desc().c_str());
                params_buffer = nullptr;
                rebuild_params_tensor_set();
                return true;
            }
        } else {
            LOG_DEBUG("%s skipping params allocation (no tensors)", get_desc().c_str());
            return true;
        }
        // Pinned host buffer when CPU-offloaded for DMA-direct H2D.
        ggml_backend_buffer_type_t params_buft = nullptr;
        if (params_backend != runtime_backend) {
            ggml_backend_dev_t runtime_dev = ggml_backend_get_device(runtime_backend);
            if (runtime_dev != nullptr) {
                params_buft = ggml_backend_dev_host_buffer_type(runtime_dev);
            }
        }
        if (params_buft == nullptr) {
            params_buft = ggml_backend_get_default_buffer_type(params_backend);
        }
        params_buffer = ggml_backend_alloc_ctx_tensors_from_buft(params_ctx, params_buft);
        if (params_buffer == nullptr) {
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        rebuild_params_tensor_set();
        ggml_backend_buffer_set_usage(params_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  sd_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                  num_tensors);
        return true;
    }

    void free_params_buffer() {
        // Restore swapped resident params before freeing their backing buffer.
        restore_resident_params();
        if (params_buffer != nullptr) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = nullptr;
        }
        observed_max_effective_budget_ = 0;
    }

    size_t get_params_buffer_size() {
        if (params_buffer != nullptr) {
            return ggml_backend_buffer_get_size(params_buffer);
        }
        return 0;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
    }

    void free_compute_buffer() {
        if (compute_allocr != nullptr) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = nullptr;
        }
        restore_partial_params();
        restore_all_params();
    }

    // do copy after alloc graph
    void set_backend_tensor_data(ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    template <typename T>
    ggml_tensor* make_input(const sd::Tensor<T>& tensor) {
        ggml_tensor* input = sd::make_ggml_tensor(compute_ctx, tensor, false);
        set_backend_tensor_data(input, tensor.data());
        return input;
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>& tensor) {
        if (tensor.empty()) {
            return nullptr;
        }
        return make_input(tensor);
    }

    template <typename T>
    ggml_tensor* make_optional_input(const sd::Tensor<T>* tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }
        return make_input(*tensor);
    }

    ggml_tensor* to_backend(ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != nullptr);
        if (tensor == nullptr) {
            return nullptr;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!sd_backend_is_cpu(runtime_backend) && (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void cache(const std::string name, ggml_tensor* tensor) {
        if (tensor != nullptr && tensor->view_src != nullptr) {
            tensor = ggml_cont(compute_ctx, tensor);
        }
        cache_tensor_map[name] = tensor;
    }

    ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        if (cache_ctx == nullptr) {
            return nullptr;
        }
        return ggml_get_tensor(cache_ctx, name.c_str());
    }

    template <typename T>
    std::optional<sd::Tensor<T>> compute(get_graph_cb_t get_graph,
                                         int n_threads,
                                         bool free_compute_buffer_immediately,
                                         bool no_return = false) {
        ggml_cgraph* gf = nullptr;
        if (!prepare_compute_graph(get_graph, &gf)) {
            return std::nullopt;
        }
        GGML_ASSERT(gf != nullptr);

        if (can_attempt_graph_cut_segmented_compute()) {
            GraphCutPlan plan;
            size_t effective_graph_vram_bytes = 0;
            if (!resolve_graph_cut_plan(gf, &plan, &effective_graph_vram_bytes)) {
                free_compute_ctx();
                return std::nullopt;
            }
            if (should_use_graph_cut_segmented_compute(plan)) {
                if (stream_layers_enabled) {
                    return compute_streaming_segments<T>(gf,
                                                         plan,
                                                         effective_graph_vram_bytes,
                                                         n_threads,
                                                         free_compute_buffer_immediately,
                                                         no_return);
                }
                return compute_with_graph_cuts<T>(gf,
                                                  plan,
                                                  n_threads,
                                                  free_compute_buffer_immediately,
                                                  no_return);
            }
        }
        if (!alloc_compute_buffer(gf)) {
            LOG_ERROR("%s alloc compute buffer failed", get_desc().c_str());
            return std::nullopt;
        }
        return execute_graph<T>(gf,
                                n_threads,
                                free_compute_buffer_immediately,
                                {},
                                false,
                                no_return);
    }

    void set_flash_attention_enabled(bool enabled) {
        flash_attn_enabled = enabled;
    }

    void set_conv2d_direct_enabled(bool enabled) {
        conv2d_direct_enabled = enabled;
    }

    void set_circular_axes(bool circular_x, bool circular_y) {
        circular_x_enabled = circular_x;
        circular_y_enabled = circular_y;
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {
        weight_adapter = adapter;
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) {
        max_graph_vram_bytes = max_vram_bytes;
    }

    void set_stream_layers_enabled(bool enabled) {
        stream_layers_enabled = enabled;
    }

    sd::layer_registry::LayerRegistry& get_layer_registry() { return layer_registry_; }

    ggml_backend_t get_runtime_backend() {
        return runtime_backend;
    }

    ggml_backend_t get_params_backend() {
        return params_backend;
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2TensorStorage& tensor_storage_map, ggml_type default_type) {
        ggml_type wtype = default_type;
        auto iter       = tensor_storage_map.find(name);
        if (iter != tensor_storage_map.end()) {
            const TensorStorage& tensor_storage = iter->second;
            if (tensor_storage.expected_type != GGML_TYPE_COUNT) {
                wtype = tensor_storage.expected_type;
            } else {
                wtype = tensor_storage.type;
            }
        }
        return wtype;
    }

    void init_blocks(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_storage_map, prefix + pair.first);
        }
    }

    virtual void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") {}

public:
    void init(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_params(ctx, tensor_storage_map, prefix);
        init_blocks(ctx, tensor_storage_map, prefix);
    }

    size_t get_params_num() {
        size_t num_tensors = params.size();
        for (auto& pair : blocks) {
            auto& block = pair.second;

            num_tensors += block->get_params_num();
        }
        return num_tensors;
    };

    size_t get_params_mem_size() {
        size_t mem_size = 0;
        for (auto& pair : blocks) {
            auto& block = pair.second;

            mem_size += block->get_params_mem_size();
        }

        for (auto& pair : params) {
            mem_size += ggml_nbytes(pair.second);
        }

        return mem_size;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            ggml_tensor* param           = pair.second;
            tensors[prefix + pair.first] = pair.second;
            ggml_set_name(param, (prefix + pair.first).c_str());
        }
    }

    virtual std::string get_desc() {
        return "GGMLBlock";
    }

    void get_all_blocks(std::vector<GGMLBlock*>& result) {
        result.push_back(this);
        for (auto& block_iter : blocks) {
            if (block_iter.second) {
                block_iter.second->get_all_blocks(result);
            }
        }
    }
};

class UnaryBlock : public GGMLBlock {
public:
    virtual ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) = 0;
};

class Identity : public UnaryBlock {
public:
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        return x;
    }
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;
    bool force_prec_f32;
    bool allow_weight_scale;
    bool has_weight_scale = false;
    float scale;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix         = prefix;
        has_weight_scale     = false;
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
        if (allow_weight_scale && tensor_storage_map.find(prefix + "weight_scale") != tensor_storage_map.end()) {
            params["weight_scale"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
            has_weight_scale       = true;
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias               = true,
           bool force_f32          = false,
           bool force_prec_f32     = false,
           float scale             = 1.f,
           bool allow_weight_scale = false)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32),
          allow_weight_scale(allow_weight_scale),
          scale(scale) {}

    void set_scale(float scale_) {
        scale = scale_;
    }

    void set_force_prec_f32(bool force_prec_f32_) {
        force_prec_f32 = force_prec_f32_;
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        ggml_tensor* linear_bias = has_weight_scale ? nullptr : b;
        ggml_tensor* out         = nullptr;
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type               = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
            forward_params.linear.force_prec_f32 = force_prec_f32;
            forward_params.linear.scale          = scale;
            out                                  = ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, linear_bias, prefix, forward_params);
        } else {
            out = ggml_ext_linear(ctx->ggml_ctx, x, w, linear_bias, force_prec_f32, scale);
        }
        if (has_weight_scale) {
            out = ggml_mul(ctx->ggml_ctx, out, params["weight_scale"]);
            if (b != nullptr) {
                out = ggml_add_inplace(ctx->ggml_ctx, out, b);
            }
        }
        return out;
    }
};

__STATIC_INLINE__ bool support_get_rows(ggml_type wtype) {
    std::set<ggml_type> allow_types = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0};
    if (allow_types.find(wtype) != allow_types.end()) {
        return true;
    }
    return false;
}

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
        if (!support_get_rows(wtype)) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
    }

public:
    Embedding(int64_t num_embeddings, int64_t embedding_dim)
        : embedding_dim(embedding_dim),
          num_embeddings(num_embeddings) {
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        input_ids      = ggml_reshape_3d(ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto embedding = ggml_get_rows(ctx->ggml_ctx, weight, input_ids);
        embedding      = ggml_reshape_3d(ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

        // [N, n_token, embedding_dim]
        return embedding;
    }
};

class Conv2d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int, int> kernel_size,
           std::pair<int, int> stride   = {1, 1},
           std::pair<int, int> padding  = {0, 0},
           std::pair<int, int> dilation = {1, 1},
           bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }
        if (ctx->weight_adapter) {
            WeightAdapter::ForwardParams forward_params;
            forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
            forward_params.conv2d.s0         = stride.second;
            forward_params.conv2d.s1         = stride.first;
            forward_params.conv2d.p0         = padding.second;
            forward_params.conv2d.p1         = padding.first;
            forward_params.conv2d.d0         = dilation.second;
            forward_params.conv2d.d1         = dilation.first;
            forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
            forward_params.conv2d.circular_x = ctx->circular_x_enabled;
            forward_params.conv2d.circular_y = ctx->circular_y_enabled;
            forward_params.conv2d.scale      = scale;
            return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
        }
        return ggml_ext_conv_2d(ctx->ggml_ctx,
                                x,
                                w,
                                b,
                                stride.second,
                                stride.first,
                                padding.second,
                                padding.first,
                                dilation.second,
                                dilation.first,
                                ctx->conv2d_direct_enabled,
                                ctx->circular_x_enabled,
                                ctx->circular_y_enabled,
                                scale);
    }
};

class Conv2d_grouped : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int groups;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
    std::pair<int, int> dilation;
    bool bias;
    float scale = 1.f;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels / groups, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv2d_grouped(int64_t in_channels,
                   int64_t out_channels,
                   int groups,
                   std::pair<int, int> kernel_size,
                   std::pair<int, int> stride   = {1, 1},
                   std::pair<int, int> padding  = {0, 0},
                   std::pair<int, int> dilation = {1, 1},
                   bool bias                    = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          groups(groups),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    void set_scale(float scale_value) {
        scale = scale_value;
    }

    std::string get_desc() {
        return "Conv2d_grouped";
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (bias) {
            b = params["bias"];
        }

        if (groups == 1) {
            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                return ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x, w, b, prefix, forward_params);
            }
            return ggml_ext_conv_2d(ctx->ggml_ctx, x, w, b,
                                    stride.second, stride.first,
                                    padding.second, padding.first,
                                    dilation.second, dilation.first,
                                    ctx->conv2d_direct_enabled,
                                    ctx->circular_x_enabled,
                                    ctx->circular_y_enabled,
                                    scale);
        }

        if (groups == in_channels && groups == out_channels) {
            ggml_tensor* res;
            if (ctx->conv2d_direct_enabled) {
                res = ggml_conv_2d_dw_direct(ctx->ggml_ctx, x, w,
                                             stride.second, stride.first,
                                             padding.second, padding.first,
                                             dilation.second, dilation.first);
            } else {
                res = ggml_conv_2d_dw(ctx->ggml_ctx, x, w,
                                      stride.second, stride.first,
                                      padding.second, padding.first,
                                      dilation.second, dilation.first);
            }
            if (b) {
                res = ggml_add(ctx->ggml_ctx, res, b);
            }
            return res;
        }

        int64_t ic_g = in_channels / groups;
        int64_t oc_g = out_channels / groups;

        std::vector<ggml_tensor*> out_slices(groups);

        for (int i = 0; i < groups; ++i) {
            size_t x_offset  = i * ic_g * x->nb[2];
            ggml_tensor* x_i = ggml_view_4d(ctx->ggml_ctx, x,
                                            x->ne[0], x->ne[1], ic_g, x->ne[3],
                                            x->nb[1], x->nb[2], x->nb[3],
                                            x_offset);

            size_t w_offset  = i * oc_g * w->nb[3];
            ggml_tensor* w_i = ggml_view_4d(ctx->ggml_ctx, w,
                                            w->ne[0], w->ne[1], w->ne[2], oc_g,
                                            w->nb[1], w->nb[2], w->nb[3],
                                            w_offset);

            ggml_tensor* b_i = nullptr;
            if (b) {
                size_t b_offset = i * oc_g * b->nb[0];
                b_i             = ggml_view_1d(ctx->ggml_ctx, b, oc_g, b_offset);
            }

            if (ctx->weight_adapter) {
                WeightAdapter::ForwardParams forward_params;
                forward_params.op_type           = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
                forward_params.conv2d.s0         = stride.second;
                forward_params.conv2d.s1         = stride.first;
                forward_params.conv2d.p0         = padding.second;
                forward_params.conv2d.p1         = padding.first;
                forward_params.conv2d.d0         = dilation.second;
                forward_params.conv2d.d1         = dilation.first;
                forward_params.conv2d.direct     = ctx->conv2d_direct_enabled;
                forward_params.conv2d.circular_x = ctx->circular_x_enabled;
                forward_params.conv2d.circular_y = ctx->circular_y_enabled;
                forward_params.conv2d.scale      = scale;
                out_slices[i]                    = ctx->weight_adapter->forward_with_lora(ctx->ggml_ctx, ctx->backend, x_i, w_i, b_i, prefix, forward_params);
            } else {
                out_slices[i] = ggml_ext_conv_2d(ctx->ggml_ctx, x_i, w_i, b_i,
                                                 stride.second, stride.first,
                                                 padding.second, padding.first,
                                                 dilation.second, dilation.first,
                                                 ctx->conv2d_direct_enabled,
                                                 ctx->circular_x_enabled,
                                                 ctx->circular_y_enabled,
                                                 scale);
            }
        }

        ggml_tensor* out = ggml_ext_vec_concat(ctx->ggml_ctx, out_slices, 2);

        return out;
    }
};

class Conv3d : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    std::tuple<int, int, int> kernel_size;
    std::tuple<int, int, int> stride;
    std::tuple<int, int, int> padding;
    std::tuple<int, int, int> dilation;
    bool bias;
    bool force_prec_f32;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx,
                                                  wtype,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
        if (bias) {
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

public:
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           std::tuple<int, int, int> kernel_size,
           std::tuple<int, int, int> stride   = {1, 1, 1},
           std::tuple<int, int, int> padding  = {0, 0, 0},
           std::tuple<int, int, int> dilation = {1, 1, 1},
           bool bias                          = true,
           bool force_prec_f32                = false)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias),
          force_prec_f32(force_prec_f32) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        ggml_tensor* b = nullptr;
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            if (w->type != GGML_TYPE_F16) {
                w = ggml_cast(ctx->ggml_ctx, w, GGML_TYPE_F16);
            }
        }
        if (bias) {
            b = params["bias"];
            if (ctx->weight_adapter) {
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                std::get<2>(padding), std::get<1>(padding), std::get<0>(padding),
                                std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation),
                                force_prec_f32);
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;
                params["bias"]       = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            }
        }
    }

public:
    LayerNorm(int64_t normalized_shape,
              float eps               = 1e-05f,
              bool elementwise_affine = true,
              bool bias               = true)
        : normalized_shape(normalized_shape),
          eps(eps),
          elementwise_affine(elementwise_affine),
          bias(bias) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;

        if (elementwise_affine) {
            w = params["weight"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
            }
            if (bias) {
                b = params["bias"];
                if (ctx->weight_adapter) {
                    b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
                }
            }
        }
        return ggml_ext_layer_norm(ctx->ggml_ctx, x, w, b, eps);
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int num_groups;
    int64_t num_channels;
    float eps;
    bool affine;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
        this->prefix = prefix;
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = nullptr;
        ggml_tensor* b = nullptr;
        if (affine) {
            w = params["weight"];
            b = params["bias"];
            if (ctx->weight_adapter) {
                w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
                b = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, b, prefix + "bias");
            }
        }
        return ggml_ext_group_norm(ctx->ggml_ctx, x, w, b, num_groups);
    }
};

class GroupNorm32 : public GroupNorm {
public:
    GroupNorm32(int64_t num_channels)
        : GroupNorm(32, num_channels, 1e-06f) {}
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;
    std::string prefix;

    void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, std::string prefix = "") override {
        this->prefix         = prefix;
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        ggml_tensor* w = params["weight"];
        if (ctx->weight_adapter) {
            w = ctx->weight_adapter->patch_weight(ctx->ggml_ctx, ctx->backend, w, prefix + "weight");
        }
        x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
        x = ggml_mul_inplace(ctx->ggml_ctx, x, w);
        return x;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    bool proj_in;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string in_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t n_head,
                       bool qkv_proj_bias        = true,
                       bool out_proj_bias        = true,
                       bool proj_in              = false,
                       std::string q_proj_name   = "q_proj",
                       std::string k_proj_name   = "k_proj",
                       std::string v_proj_name   = "v_proj",
                       std::string in_proj_name  = "in_proj",
                       std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim),
          n_head(n_head),
          proj_in(proj_in),
          q_proj_name(q_proj_name),
          k_proj_name(k_proj_name),
          v_proj_name(v_proj_name),
          in_proj_name(in_proj_name),
          out_proj_name(out_proj_name) {
        if (proj_in) {
            blocks[in_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim * 3, qkv_proj_bias));
        } else {
            blocks[q_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[k_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
            blocks[v_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        }
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    ggml_tensor* forward(GGMLRunnerContext* ctx,
                         ggml_tensor* x,
                         ggml_tensor* mask = nullptr) {
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        ggml_tensor* q;
        ggml_tensor* k;
        ggml_tensor* v;
        if (proj_in) {
            auto in_proj = std::dynamic_pointer_cast<Linear>(blocks[in_proj_name]);
            auto qkv     = in_proj->forward(ctx, x);
            auto qkv_vec = split_qkv(ctx->ggml_ctx, qkv);
            q            = qkv_vec[0];
            k            = qkv_vec[1];
            v            = qkv_vec[2];
        } else {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);

            q = q_proj->forward(ctx, x);
            k = k_proj->forward(ctx, x);
            v = v_proj->forward(ctx, x);
        }

        x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, n_head, mask, false);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

__STATIC_INLINE__ ggml_tensor* ggml_ext_lokr_forward(
    ggml_context* ctx,
    ggml_backend_t backend,
    ggml_tensor* h,    // Input: [q, batch] or [W, H, q, batch]
    ggml_tensor* w1,   // Outer C (Full rank)
    ggml_tensor* w1a,  // Outer A (Low rank part 1)
    ggml_tensor* w1b,  // Outer B (Low rank part 2)
    ggml_tensor* w2,   // Inner BA (Full rank)
    ggml_tensor* w2a,  // Inner A (Low rank part 1)
    ggml_tensor* w2b,  // Inner B (Low rank part 2)
    bool is_conv,
    WeightAdapter::ForwardParams::conv2d_params_t conv_params,
    float scale) {
    GGML_ASSERT((w1 != nullptr || (w1a != nullptr && w1b != nullptr)));
    GGML_ASSERT((w2 != nullptr || (w2a != nullptr && w2b != nullptr)));

    int uq = (w1 != nullptr) ? (int)w1->ne[0] : (int)w1a->ne[0];
    int up = (w1 != nullptr) ? (int)w1->ne[1] : (int)w1b->ne[1];

    int q_actual = is_conv ? (int)h->ne[2] : (int)h->ne[0];
    int vq       = q_actual / uq;

    int vp = (w2 != nullptr) ? (is_conv ? (int)w2->ne[3] : (int)w2->ne[1])
                             : (int)w2a->ne[1];
    GGML_ASSERT(q_actual == (uq * vq) && "Input dimension mismatch for LoKR split");

    ggml_tensor* hb;

    if (!is_conv) {
        int batch          = (int)h->ne[1];
        int merge_batch_uq = batch;
        int merge_batch_vp = batch;

        if (sd_backend_is(backend, "Vulkan")) {
            if (batch > 1) {
                // no access to backend here, worst case is slightly worse perfs for other backends when built alongside Vulkan backend
                int max_batch    = 65535;
                int max_batch_uq = max_batch / uq;
                merge_batch_uq   = 1;
                for (int i = max_batch_uq; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_uq = i;
                        break;
                    }
                }

                int max_batch_vp = max_batch / vp;
                merge_batch_vp   = 1;
                for (int i = max_batch_vp; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_vp = i;
                        break;
                    }
                }
            }
        }

        ggml_tensor* h_split = ggml_reshape_3d(ctx, h, vq, uq * merge_batch_uq, batch / merge_batch_uq);
        if (w2 != nullptr) {
            hb = ggml_mul_mat(ctx, w2, h_split);
        } else {
            hb = ggml_mul_mat(ctx, w2b, ggml_mul_mat(ctx, w2a, h_split));
        }

        if (batch > 1) {
            hb = ggml_reshape_3d(ctx, hb, vp, uq, batch);
        }
        ggml_tensor* hb_t = ggml_cont(ctx, ggml_transpose(ctx, hb));
        hb_t              = ggml_reshape_3d(ctx, hb_t, uq, vp * merge_batch_vp, batch / merge_batch_vp);

        ggml_tensor* hc_t;
        if (w1 != nullptr) {
            hc_t = ggml_mul_mat(ctx, w1, hb_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_t));
        }

        if (batch > 1) {
            hc_t = ggml_reshape_3d(ctx, hc_t, up, vp, batch);
        }

        ggml_tensor* hc  = ggml_transpose(ctx, hc_t);
        ggml_tensor* out = ggml_reshape_2d(ctx, ggml_cont(ctx, hc), up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    } else {
        int batch = (int)h->ne[3];
        // 1. Reshape input: [W, H, vq*uq, batch] -> [W, H, vq, uq * batch]
        ggml_tensor* h_split = ggml_reshape_4d(ctx, h, h->ne[0], h->ne[1], vq, uq * batch);

        if (w2 != nullptr) {
            hb = ggml_ext_conv_2d(ctx, h_split, w2, nullptr,
                                  conv_params.s0,
                                  conv_params.s1,
                                  conv_params.p0,
                                  conv_params.p1,
                                  conv_params.d0,
                                  conv_params.d1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        } else {
            // swap a and b order for conv lora
            ggml_tensor* a = w2b;
            ggml_tensor* b = w2a;

            // unpack conv2d weights if needed
            if (ggml_n_dims(a) < 4) {
                int k = (int)sqrt(a->ne[0] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[0]);
                a = ggml_reshape_4d(ctx, a, k, k, a->ne[0] / (k * k), a->ne[1]);
            } else if (a->ne[2] != h_split->ne[2]) {
                int k = (int)sqrt(a->ne[2] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[2]);
                a = ggml_reshape_4d(ctx, a, a->ne[0] * k, a->ne[1] * k, a->ne[2] / (k * k), a->ne[3]);
            }
            ggml_tensor* ha = ggml_ext_conv_2d(ctx, h_split, a, nullptr,
                                               conv_params.s0,
                                               conv_params.s1,
                                               conv_params.p0,
                                               conv_params.p1,
                                               conv_params.d0,
                                               conv_params.d1,
                                               conv_params.direct,
                                               conv_params.circular_x,
                                               conv_params.circular_y,
                                               conv_params.scale);

            // not supporting lora_mid here
            hb = ggml_ext_conv_2d(ctx,
                                  ha,
                                  b,
                                  nullptr,
                                  1,
                                  1,
                                  0,
                                  0,
                                  1,
                                  1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        }

        // Current hb shape: [W_out, H_out, vp, uq * batch]
        int w_out = (int)hb->ne[0];
        int h_out = (int)hb->ne[1];

        // ggml_tensor* hb_cat = ggml_reshape_4d(ctx, hb, w_out , h_out , vp * uq, batch);
        // [W_out, H_out, vp * uq,  batch]
        // Now left to compute (W1 kr Id) * hb_cat == (W1 kr W2) cv h

        // merge the uq groups of size vp*w_out*h_out
        ggml_tensor* hb_merged = ggml_reshape_2d(ctx, hb, w_out * h_out * vp, uq * batch);
        ggml_tensor* hc_t;
        ggml_tensor* hb_merged_t = ggml_cont(ctx, ggml_transpose(ctx, hb_merged));
        if (w1 != nullptr) {
            // Would be great to be able to transpose w1 instead to avoid transposing both hb and hc
            hc_t = ggml_mul_mat(ctx, w1, hb_merged_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_merged_t));
        }
        ggml_tensor* hc = ggml_transpose(ctx, hc_t);
        // ungroup
        ggml_tensor* out = ggml_reshape_4d(ctx, ggml_cont(ctx, hc), w_out, h_out, up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    }
}

#endif  // __SD_CORE_GGML_EXTEND_HPP__

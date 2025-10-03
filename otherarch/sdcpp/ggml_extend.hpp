#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "model.h"

#ifdef SD_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef SD_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef SD_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef SD_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "rng.hpp"
#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

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

// n-mode trensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
__STATIC_INLINE__ struct ggml_tensor* ggml_mul_n_mode(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int mode = 0) {
    // reshape A
    // swap 0th and nth axis
    a       = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0));
    int ne1 = a->ne[1];
    int ne2 = a->ne[2];
    int ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshape_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    struct ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));

    // reshape output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0);
    return result;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_merge_lora(ggml_context* ctx, struct ggml_tensor* lora_down, struct ggml_tensor* lora_up, struct ggml_tensor* lora_mid = NULL) {
    struct ggml_tensor* updown;
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
    if (lora_mid == NULL) {
        updown = ggml_mul_mat(ctx, lora_up, lora_down);
        updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
    } else {
        // undoing tucker decomposition for conv layers.
        // lora_mid  has shape (3,    3,   Rank, Rank)
        // lora_down has shape (Rank, In,  1,    1)
        // lora_up   has shape (Rank, Out, 1,    1)
        // conv layer shape is (3,    3,   Out,  In)
        updown = ggml_mul_n_mode(ctx, ggml_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
        updown = ggml_cont(ctx, updown);
    }
    return updown;
}

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
__STATIC_INLINE__ struct ggml_tensor* ggml_kronecker(ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
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

__STATIC_INLINE__ void ggml_tensor_set_f32_randn(struct ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
__STATIC_INLINE__ void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(float));
    return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_tensor_get_i32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    if (tensor->buffer != NULL) {
        float value;
        ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(int));
        return value;
    }
    GGML_ASSERT(tensor->nb[0] == sizeof(int));
    return *(int*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0) {
    GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
    return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ float sd_image_get_f32(sd_image_t image, int iw, int ih, int ic, bool scale = true) {
    float value = *(image.data + ih * image.width * image.channel + iw * image.channel + ic);
    if (scale) {
        value /= 255.f;
    }
    return value;
}

__STATIC_INLINE__ float sd_image_get_f32(sd_image_f32_t image, int iw, int ih, int ic) {
    float value = *(image.data + ih * image.width * image.channel + iw * image.channel + ic);
    return value;
}

__STATIC_INLINE__ void print_ggml_tensor(struct ggml_tensor* tensor, bool shape_only = false, const char* mark = "") {
    printf("%s (%s): shape(%zu, %zu, %zu, %zu)\n", mark, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fflush(stdout);
    if (shape_only) {
        return;
    }
    int range = 3;
    for (int i = 0; i < tensor->ne[3]; i++) {
        if (i >= range && i + range < tensor->ne[3]) {
            continue;
        }
        for (int j = 0; j < tensor->ne[2]; j++) {
            if (j >= range && j + range < tensor->ne[2]) {
                continue;
            }
            for (int k = 0; k < tensor->ne[1]; k++) {
                if (k >= range && k + range < tensor->ne[1]) {
                    continue;
                }
                for (int l = 0; l < tensor->ne[0]; l++) {
                    if (l >= range && l + range < tensor->ne[0]) {
                        continue;
                    }
                    if (tensor->type == GGML_TYPE_F32) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_tensor_get_f32(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_F16) {
                        printf("  [%d, %d, %d, %d] = %f\n", i, j, k, l, ggml_fp16_to_fp32(ggml_tensor_get_f16(tensor, l, k, j, i)));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_i32(tensor, l, k, j, i));
                    }
                    fflush(stdout);
                }
            }
        }
    }
}

__STATIC_INLINE__ void ggml_tensor_iter(
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

__STATIC_INLINE__ void ggml_tensor_iter(
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

__STATIC_INLINE__ void ggml_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap = 0.1f) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    ggml_tensor_iter(a, [&](ggml_tensor* a, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float a_value = ggml_tensor_get_f32(a, i0, i1, i2, i3);
        float b_value = ggml_tensor_get_f32(b, i0, i1, i2, i3);
        if (abs(a_value - b_value) > gap) {
            LOG_WARN("[%ld, %ld, %ld, %ld] %f %f", i3, i2, i1, i0, a_value, b_value);
        }
    });
}

__STATIC_INLINE__ ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
    #ifdef _WIN32
        std::filesystem::path fpath = std::filesystem::u8path(file_path);
    #else
        std::filesystem::path fpath = std::filesystem::path(file_path);
    #endif
    std::ifstream file(fpath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return NULL;
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
        return NULL;
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

__STATIC_INLINE__ void copy_ggml_tensor(struct ggml_tensor* dst, struct ggml_tensor* src) {
    if (dst->type == src->type) {
        dst->nb[0] = src->nb[0];
        dst->nb[1] = src->nb[1];
        dst->nb[2] = src->nb[2];
        dst->nb[3] = src->nb[3];

        memcpy(((char*)dst->data), ((char*)src->data), ggml_nbytes(dst));
        return;
    }
    struct ggml_init_params params;
    params.mem_size          = 10 * 1024 * 1024;  // for padding
    params.mem_buffer        = NULL;
    params.no_alloc          = false;
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }
    ggml_tensor* final = ggml_cpy(ctx, src, dst);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    ggml_free(ctx);
}

__STATIC_INLINE__ float sigmoid(float x) {
    return 1 / (1.0f + expf(-x));
}

// SPECIAL OPERATIONS WITH TENSORS

__STATIC_INLINE__ uint8_t* sd_tensor_to_image(struct ggml_tensor* input) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_tensor_get_f32(input, ix, iy, k);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ uint8_t* sd_tensor_to_image(struct ggml_tensor* input, int idx, bool video = false) {
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
                    value = ggml_tensor_get_f32(input, iw, ih, idx, ic);
                } else {
                    value = ggml_tensor_get_f32(input, iw, ih, ic, idx);
                }
                *(image_data + ih * width * channels + iw * channels + ic) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void sd_image_to_tensor(sd_image_t image,
                                          ggml_tensor* tensor,
                                          bool scale = true) {
    GGML_ASSERT(image.width == tensor->ne[0]);
    GGML_ASSERT(image.height == tensor->ne[1]);
    GGML_ASSERT(image.channel == tensor->ne[2]);
    GGML_ASSERT(1 == tensor->ne[3]);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    ggml_tensor_iter(tensor, [&](ggml_tensor* tensor, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float value = sd_image_get_f32(image, i0, i1, i2, scale);
        ggml_tensor_set_f32(tensor, value, i0, i1, i2, i3);
    });
}

__STATIC_INLINE__ void sd_apply_mask(struct ggml_tensor* image_data,
                                     struct ggml_tensor* mask,
                                     struct ggml_tensor* output) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(output->type == GGML_TYPE_F32);
    for (int ix = 0; ix < width; ix++) {
        for (int iy = 0; iy < height; iy++) {
            float m = ggml_tensor_get_f32(mask, ix, iy);
            m       = round(m);  // inpaint models need binary masks
            ggml_tensor_set_f32(mask, m, ix, iy);
            for (int k = 0; k < channels; k++) {
                float value = (1 - m) * (ggml_tensor_get_f32(image_data, ix, iy, k) - .5) + .5;
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ void sd_image_f32_to_tensor(const float* image_data,
                                              struct ggml_tensor* output,
                                              bool scale = true) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                int value = *(image_data + iy * width * channels + ix * channels + k);
                if (scale) {
                    value /= 255.f;
                }
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ void ggml_split_tensor_2d(struct ggml_tensor* input,
                                            struct ggml_tensor* output,
                                            int x,
                                            int y) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = ggml_tensor_get_f32(input, ix + x, iy + y, k);
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

// unclamped -> expects x in the range [0-1]
__STATIC_INLINE__ float ggml_smootherstep_f32(const float x) {
    GGML_ASSERT(x >= 0.f && x <= 1.f);
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

__STATIC_INLINE__ void ggml_merge_tensor_2d(struct ggml_tensor* input,
                                            struct ggml_tensor* output,
                                            int x,
                                            int y,
                                            int overlap_x,
                                            int overlap_y,
                                            int x_skip = 0,
                                            int y_skip = 0) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];

    int64_t img_width  = output->ne[0];
    int64_t img_height = output->ne[1];

    GGML_ASSERT(input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
    for (int iy = y_skip; iy < height; iy++) {
        for (int ix = x_skip; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float new_value = ggml_tensor_get_f32(input, ix, iy, k);
                if (overlap_x > 0 || overlap_y > 0) {  // blend colors in overlapped area
                    float old_value = ggml_tensor_get_f32(output, x + ix, y + iy, k);

                    const float x_f_0 = (overlap_x > 0 && x > 0) ? (ix - x_skip) / float(overlap_x) : 1;
                    const float x_f_1 = (overlap_x > 0 && x < (img_width - width)) ? (width - ix) / float(overlap_x) : 1;
                    const float y_f_0 = (overlap_y > 0 && y > 0) ? (iy - y_skip) / float(overlap_y) : 1;
                    const float y_f_1 = (overlap_y > 0 && y < (img_height - height)) ? (height - iy) / float(overlap_y) : 1;

                    const float x_f = std::min(std::min(x_f_0, x_f_1), 1.f);
                    const float y_f = std::min(std::min(y_f_0, y_f_1), 1.f);

                    ggml_tensor_set_f32(
                        output,
                        old_value + new_value * ggml_smootherstep_f32(y_f) * ggml_smootherstep_f32(x_f),
                        x + ix, y + iy, k);
                } else {
                    ggml_tensor_set_f32(output, new_value, x + ix, y + iy, k);
                }
            }
        }
    }
}

__STATIC_INLINE__ float ggml_tensor_mean(struct ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

// a = a+b
__STATIC_INLINE__ void ggml_tensor_add(struct ggml_tensor* a, struct ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

__STATIC_INLINE__ void ggml_tensor_scale(struct ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

__STATIC_INLINE__ void ggml_tensor_clamp(struct ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

__STATIC_INLINE__ struct ggml_tensor* ggml_tensor_concat(struct ggml_context* ctx,
                                                         struct ggml_tensor* a,
                                                         struct ggml_tensor* b,
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
    struct ggml_tensor* result = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);
    int64_t o[4]               = {0, 0, 0, 0};
    o[dim]                     = a->ne[dim];

    float v;
    for (int i3 = 0; i3 < result->ne[3]; i3++) {
        for (int i2 = 0; i2 < result->ne[2]; i2++) {
            for (int i1 = 0; i1 < result->ne[1]; i1++) {
                for (int i0 = 0; i0 < result->ne[0]; i0++) {
                    if (i0 < a->ne[0] && i1 < a->ne[1] && i2 < a->ne[2] && i3 < a->ne[3]) {
                        v = ggml_tensor_get_f32(a, i0, i1, i2, i3);
                    } else {
                        v = ggml_tensor_get_f32(b, i0 - o[0], i1 - o[1], i2 - o[2], i3 - o[3]);
                    }

                    ggml_tensor_set_f32(result, v, i0, i1, i2, i3);
                }
            }
        }
    }
    return result;
}

// convert values from [0, 1] to [-1, 1]
__STATIC_INLINE__ void process_vae_input_tensor(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void process_vae_output_tensor(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_cont(struct ggml_context* ctx,
                                                   struct ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

// torch like permute
__STATIC_INLINE__ struct ggml_tensor* ggml_torch_permute(struct ggml_context* ctx,
                                                         struct ggml_tensor* x,
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

__STATIC_INLINE__ struct ggml_tensor* ggml_slice(struct ggml_context* ctx,
                                                 struct ggml_tensor* x,
                                                 int64_t dim,
                                                 int64_t start,
                                                 int64_t end) {
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

    int perm[4] = {0, 1, 2, 3};
    for (int i = dim; i < 3; ++i)
        perm[i] = perm[i + 1];
    perm[3] = dim;

    int inv_perm[4];
    for (int i = 0; i < 4; ++i)
        inv_perm[perm[i]] = i;

    if (dim != 3) {
        x = ggml_torch_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
        x = ggml_cont(ctx, x);
    }

    x = ggml_view_4d(
        ctx, x,
        x->ne[0], x->ne[1], x->ne[2], end - start,
        x->nb[1], x->nb[2], x->nb[3], x->nb[3] * start);

    if (dim != 3) {
        x = ggml_torch_permute(ctx, x, inv_perm[0], inv_perm[1], inv_perm[2], inv_perm[3]);
        x = ggml_cont(ctx, x);
    }

    return x;
}

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
__STATIC_INLINE__ std::vector<struct ggml_tensor*> ggml_chunk(struct ggml_context* ctx,
                                                              struct ggml_tensor* x,
                                                              int num,
                                                              int64_t dim) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    int perm[4] = {0, 1, 2, 3};
    for (int i = dim; i < 3; ++i)
        perm[i] = perm[i + 1];
    perm[3] = dim;

    int inv_perm[4];
    for (int i = 0; i < 4; ++i)
        inv_perm[perm[i]] = i;

    if (dim != 3) {
        x = ggml_torch_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
        x = ggml_cont(ctx, x);
    }

    std::vector<struct ggml_tensor*> chunks;
    int64_t chunk_size = x->ne[3] / num;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            x->ne[0], x->ne[1], x->ne[2], chunk_size,
            x->nb[1], x->nb[2], x->nb[3], x->nb[3] * i * chunk_size);

        if (dim != 3) {
            chunk = ggml_torch_permute(ctx, chunk, inv_perm[0], inv_perm[1], inv_perm[2], inv_perm[3]);
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void sd_tiling_calc_tiles(int& num_tiles_dim,
                                            float& tile_overlap_factor_dim,
                                            int small_dim,
                                            int tile_size,
                                            const float tile_overlap_factor) {
    int tile_overlap     = (tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

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
__STATIC_INLINE__ void sd_tiling_non_square(ggml_tensor* input,
                                            ggml_tensor* output,
                                            const int scale,
                                            const int p_tile_size_x,
                                            const int p_tile_size_y,
                                            const float tile_overlap_factor,
                                            on_tile_process on_processing) {
    output = ggml_set_f32(output, 0);

    int input_width   = (int)input->ne[0];
    int input_height  = (int)input->ne[1];
    int output_width  = (int)output->ne[0];
    int output_height = (int)output->ne[1];

    GGML_ASSERT(((input_width / output_width) == (input_height / output_height)) &&
                ((output_width / input_width) == (output_height / input_height)));
    GGML_ASSERT(((input_width / output_width) == scale) ||
                ((output_width / input_width) == scale));

    int small_width  = output_width;
    int small_height = output_height;

    bool decode = output_width > input_width;
    if (decode) {
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor);

    LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
    LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);

    GGML_ASSERT(input_width % 2 == 0 && input_height % 2 == 0 && output_width % 2 == 0 && output_height % 2 == 0);  // should be multiple of 2

    int tile_overlap_x     = (int32_t)(p_tile_size_x * tile_overlap_factor_x);
    int non_tile_overlap_x = p_tile_size_x - tile_overlap_x;

    int tile_overlap_y     = (int32_t)(p_tile_size_y * tile_overlap_factor_y);
    int non_tile_overlap_y = p_tile_size_y - tile_overlap_y;

    int tile_size_x = p_tile_size_x < small_width ? p_tile_size_x : small_width;
    int tile_size_y = p_tile_size_y < small_height ? p_tile_size_y : small_height;

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

    struct ggml_init_params params = {};
    params.mem_size += input_tile_size_x * input_tile_size_y * input->ne[2] * sizeof(float);     // input chunk
    params.mem_size += output_tile_size_x * output_tile_size_y * output->ne[2] * sizeof(float);  // output chunk
    params.mem_size += 3 * ggml_tensor_overhead();
    params.mem_buffer = NULL;
    params.no_alloc   = false;

    LOG_DEBUG("tile work buffer size: %.2f MB", params.mem_size / 1024.f / 1024.f);

    // draft context
    struct ggml_context* tiles_ctx = ggml_init(params);
    if (!tiles_ctx) {
        LOG_ERROR("ggml_init() failed");
        return;
    }

    // tiling
    ggml_tensor* input_tile  = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, input_tile_size_x, input_tile_size_y, input->ne[2], 1);
    ggml_tensor* output_tile = ggml_new_tensor_4d(tiles_ctx, GGML_TYPE_F32, output_tile_size_x, output_tile_size_y, output->ne[2], 1);
    int num_tiles            = num_tiles_x * num_tiles_y;
    LOG_INFO("processing %i tiles", num_tiles);
    pretty_progress(0, num_tiles, 0.0f);
    int tile_count = 1;
    bool last_y = false, last_x = false;
    float last_time = 0.0f;
    for (int y = 0; y < small_height && !last_y; y += non_tile_overlap_y) {
        int dy = 0;
        if (y + tile_size_y >= small_height) {
            int _y = y;
            y      = small_height - tile_size_y;
            dy     = _y - y;
            if (decode) {
                dy *= scale;
            }
            last_y = true;
        }
        for (int x = 0; x < small_width && !last_x; x += non_tile_overlap_x) {
            int dx = 0;
            if (x + tile_size_x >= small_width) {
                int _x = x;
                x      = small_width - tile_size_x;
                dx     = _x - x;
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

            int64_t t1 = ggml_time_ms();
            ggml_split_tensor_2d(input, input_tile, x_in, y_in);
            on_processing(input_tile, output_tile, false);
            ggml_merge_tensor_2d(output_tile, output, x_out, y_out, overlap_x_out, overlap_y_out, dx, dy);

            int64_t t2 = ggml_time_ms();
            last_time  = (t2 - t1) / 1000.0f;
            pretty_progress(tile_count, num_tiles, last_time);
            tile_count++;
        }
        last_x = false;
    }
    if (tile_count < num_tiles) {
        pretty_progress(num_tiles, num_tiles, last_time);
    }
    ggml_free(tiles_ctx);
}

__STATIC_INLINE__ void sd_tiling(ggml_tensor* input,
                                 ggml_tensor* output,
                                 const int scale,
                                 const int tile_size,
                                 const float tile_overlap_factor,
                                 on_tile_process on_processing) {
    sd_tiling_non_square(input, output, scale, tile_size, tile_size, tile_overlap_factor, on_processing);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
                                                         struct ggml_tensor* a) {
    const float eps = 1e-6f;  // default eps parameter
    return ggml_group_norm(ctx, a, 32, eps);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
                                                     struct ggml_tensor* x,
                                                     struct ggml_tensor* w,
                                                     struct ggml_tensor* b,
                                                     bool force_prec_f32 = false) {
    x = ggml_mul_mat(ctx, w, x);
    if (force_prec_f32) {
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);
    }
    if (b != NULL) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx,
                                                      struct ggml_tensor* x,
                                                      struct ggml_tensor* w,
                                                      struct ggml_tensor* b,
                                                      int s0 = 1,
                                                      int s1 = 1,
                                                      int p0 = 0,
                                                      int p1 = 0,
                                                      int d0 = 1,
                                                      int d1 = 1) {
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC*IC, KD, KH, KW]
// x: [N*IC, ID, IH, IW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_2d_direct(struct ggml_context* ctx,
                                                             struct ggml_tensor* x,
                                                             struct ggml_tensor* w,
                                                             struct ggml_tensor* b,
                                                             int s0 = 1,
                                                             int s1 = 1,
                                                             int p0 = 0,
                                                             int p1 = 0,
                                                             int d0 = 1,
                                                             int d1 = 1) {
    x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N*OC, OD, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_3d(struct ggml_context* ctx,
                                                      struct ggml_tensor* x,
                                                      struct ggml_tensor* w,
                                                      struct ggml_tensor* b,
                                                      int64_t IC,
                                                      int s0 = 1,
                                                      int s1 = 1,
                                                      int s2 = 1,
                                                      int p0 = 0,
                                                      int p1 = 0,
                                                      int p2 = 0,
                                                      int d0 = 1,
                                                      int d1 = 1,
                                                      int d2 = 1) {
    int64_t OC = w->ne[3] / IC;
    int64_t N  = x->ne[3] / IC;
    x          = ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);

    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_3d_nx1x1(struct ggml_context* ctx,
                                                            struct ggml_tensor* x,
                                                            struct ggml_tensor* w,
                                                            struct ggml_tensor* b,
                                                            int s2 = 1,
                                                            int p2 = 1,
                                                            int d2 = 1) {
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;  // [N, OC, T, OH * OW]
}

// qkv: [N, L, 3*C]
// return: ([N, L, C], [N, L, C], [N, L, C])
__STATIC_INLINE__ std::vector<struct ggml_tensor*> split_qkv(struct ggml_context* ctx,
                                                             struct ggml_tensor* qkv) {
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
__STATIC_INLINE__ std::vector<struct ggml_tensor*> split_image_qkv(struct ggml_context* ctx,
                                                                   struct ggml_tensor* qkv) {
    int64_t W   = qkv->ne[0];
    int64_t H   = qkv->ne[1];
    int64_t C   = qkv->ne[2] / 3;
    int64_t N   = qkv->ne[3];
    int64_t nb1 = qkv->nb[1];
    int64_t nb2 = qkv->nb[2];
    qkv         = ggml_reshape_4d(ctx, qkv, W * H, C, 3, N);                 // [N, 3, C, H*W]
    qkv         = ggml_cont(ctx, ggml_torch_permute(ctx, qkv, 0, 1, 3, 2));  // [3, N, C, H*W]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 0);  // [N, C, H, W]
    auto k         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 1);  // [N, C, H, W]
    auto v         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 2);  // [N, C, H, W]
    return {q, k, v};
}

__STATIC_INLINE__ struct ggml_tensor* ggml_full(struct ggml_context* ctx,
                                                float value,
                                                int64_t ne0,
                                                int64_t ne1,
                                                int64_t ne2,
                                                int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_scale(ctx, one, value);                 // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_zeros(struct ggml_context* ctx,
                                                 int64_t ne0,
                                                 int64_t ne1,
                                                 int64_t ne2,
                                                 int64_t ne3) {
    return ggml_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_ones(struct ggml_context* ctx,
                                                int64_t ne0,
                                                int64_t ne1,
                                                int64_t ne2,
                                                int64_t ne3) {
    return ggml_full(ctx, 1.f, ne0, ne1, ne2, ne3);
}

// q: [N * n_head, n_token, d_head]
// k: [N * n_head, n_k, d_head]
// v: [N * n_head, d_head, n_k]
// return: [N * n_head, n_token, d_head]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_attention(struct ggml_context* ctx,
                                                        struct ggml_tensor* q,
                                                        struct ggml_tensor* k,
                                                        struct ggml_tensor* v,
                                                        bool mask = false) {
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUDA) && !defined(SD_USE_METAL) && !defined(SD_USE_VULKAN) && !defined(SD_USE_SYCL)
    struct ggml_tensor* kqv = ggml_flash_attn(ctx, q, k, v, false);  // [N * n_head, n_token, d_head]
#else
    float d_head           = (float)q->ne[0];
    struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, n_token, n_k]
    kq                     = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq                      = ggml_soft_max_inplace(ctx, kq);
    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, n_token, d_head]
#endif
    return kqv;
}

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_attention_ext(struct ggml_context* ctx,
                                                            ggml_backend_t backend,
                                                            struct ggml_tensor* q,
                                                            struct ggml_tensor* k,
                                                            struct ggml_tensor* v,
                                                            int64_t n_head,
                                                            struct ggml_tensor* mask = NULL,
                                                            bool diag_mask_inf       = false,
                                                            bool skip_reshape        = false,
                                                            bool flash_attn          = false,  // avoid overflow
                                                            float kv_scale           = 1.0f) {
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

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);      // [N, L_q, n_head, d_head]
        q = ggml_nn_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);     // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);   // [N, L_k, n_kv_head, d_head]
        k = ggml_nn_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);  // [N * n_kv_head, L_k, d_head]

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

    int kv_pad       = 0;
    ggml_tensor* kqv = nullptr;

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) -> ggml_tensor* {
        if (kv_pad != 0) {
            k_in = ggml_pad(ctx, k_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_scale(ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        v_in = ggml_nn_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        if (kv_pad != 0) {
            v_in = ggml_pad(ctx, v_in, 0, kv_pad, 0, 0);
        }
        if (kv_scale != 1.0f) {
            v_in = ggml_scale(ctx, v_in, kv_scale);
        }
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);

        if (mask_in != nullptr) {
            mask_in = ggml_transpose(ctx, mask_in);
        } else {
            if (kv_pad > 0) {
                mask_in         = ggml_zeros(ctx, L_k, L_q, 1, 1);
                auto pad_tensor = ggml_full(ctx, -INFINITY, kv_pad, L_q, 1, 1);
                mask_in         = ggml_concat(ctx, mask_in, pad_tensor, 0);
            }
        }

        if (mask_in != nullptr) {
            int mask_pad = 0;
            if (mask_in->ne[1] % GGML_KQ_MASK_PAD != 0) {
                mask_pad = GGML_PAD(L_q, GGML_KQ_MASK_PAD) - mask_in->ne[1];
            }
            if (mask_pad > 0) {
                mask_in = ggml_pad(ctx, mask_in, 0, mask_pad, 0, 0);
            }
            mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
        }

        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    if (flash_attn) {
        // LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (can_use_flash_attn && L_k % 256 != 0) {
            kv_pad = GGML_PAD(L_k, 256) - L_k;
        }

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
        v = ggml_nn_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);  // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        kq      = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        if (diag_mask_inf) {
            kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_nn_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
                                                         struct ggml_tensor* x,
                                                         struct ggml_tensor* w,
                                                         struct ggml_tensor* b,
                                                         float eps = EPS) {
    x = ggml_norm(ctx, x, eps);
    if (w != NULL) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != NULL) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_group_norm(struct ggml_context* ctx,
                                                         struct ggml_tensor* x,
                                                         struct ggml_tensor* w,
                                                         struct ggml_tensor* b,
                                                         int num_groups = 32) {
    if (ggml_n_dims(x) >= 3 && w != NULL && b != NULL) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != NULL && b != NULL) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

__STATIC_INLINE__ void ggml_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
#if defined(SD_USE_CUDA) || defined(SD_USE_SYCL)
    if (!ggml_backend_is_cpu(backend)) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
#else
    ggml_backend_tensor_get(tensor, data, offset, size);
#endif
}

__STATIC_INLINE__ float ggml_backend_tensor_get_f32(ggml_tensor* tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_I32);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
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

__STATIC_INLINE__ struct ggml_tensor* vector_to_ggml_tensor(struct ggml_context* ctx,
                                                            const std::vector<float>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

__STATIC_INLINE__ struct ggml_tensor* vector_to_ggml_tensor_i32(struct ggml_context* ctx,
                                                                const std::vector<int>& vec) {
    struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
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
                                              struct ggml_tensor* embedding,
                                              int dim,
                                              int max_period = 10000) {
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
}

__STATIC_INLINE__ struct ggml_tensor* new_timestep_embedding(struct ggml_context* ctx,
                                                             std::vector<float> timesteps,
                                                             int dim,
                                                             int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [N, dim]
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    struct ggml_tensor* embedding    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps.size());
    if (embedding->data != NULL) {
        memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
    } else {
        ggml_backend_tensor_set(embedding, embedding_vec.data(), 0, ggml_nbytes(embedding));
    }
    return embedding;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_timestep_embedding(
    struct ggml_context* ctx,
    struct ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f) {
    timesteps = ggml_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

/* SDXL with LoRA requires more space */
#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 327680

typedef std::map<std::string, enum ggml_type> String2GGMLType;

struct GGMLRunner {
protected:
    typedef std::function<struct ggml_cgraph*()> get_graph_cb_t;

    ggml_backend_t params_backend  = NULL;
    ggml_backend_t runtime_backend = NULL;

    struct ggml_context* params_ctx             = NULL;
    ggml_backend_buffer_t params_buffer         = NULL;
    struct ggml_context* offload_ctx            = NULL;
    ggml_backend_buffer_t runtime_params_buffer = NULL;
    bool params_on_runtime_backend              = false;

    struct ggml_context* cache_ctx     = NULL;
    ggml_backend_buffer_t cache_buffer = NULL;

    struct ggml_context* compute_ctx    = NULL;
    struct ggml_gallocr* compute_allocr = NULL;

    std::vector<float> one_vec = {1.f};
    ggml_tensor* one_tensor    = NULL;

    std::map<struct ggml_tensor*, const void*> backend_tensor_data_map;
    std::map<std::string, struct ggml_tensor*> cache_tensor_map;  // name -> tensor
    const std::string final_result_name = "ggml_runner_final_result_tensor";

    void alloc_params_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != NULL);
        if (params_backend != runtime_backend) {
            offload_ctx = ggml_init(params);
            GGML_ASSERT(offload_ctx != NULL);
        }
    }

    void free_params_ctx() {
        if (params_ctx != NULL) {
            ggml_free(params_ctx);
            params_ctx = NULL;
        }
        if (offload_ctx != NULL) {
            ggml_free(offload_ctx);
            offload_ctx = NULL;
        }
    }

    void alloc_cache_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        cache_ctx = ggml_init(params);
        GGML_ASSERT(cache_ctx != NULL);
    }

    void free_cache_ctx() {
        if (cache_ctx != NULL) {
            ggml_free(cache_ctx);
            cache_ctx = NULL;
        }
    }

    void alloc_compute_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        compute_ctx = ggml_init(params);
        GGML_ASSERT(compute_ctx != NULL);
    }

    void free_compute_ctx() {
        if (compute_ctx != NULL) {
            ggml_free(compute_ctx);
            compute_ctx = NULL;
        }
    }

    void prepare_build_in_tensor_before() {
        one_tensor = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(one_tensor, "ggml_runner_build_in_tensor:one");
        set_backend_tensor_data(one_tensor, one_vec.data());
    }

    void prepare_build_in_tensor_after(struct ggml_cgraph* gf) {
        ggml_build_forward_expand(gf, one_tensor);
    }

    struct ggml_cgraph* get_compute_graph(get_graph_cb_t get_graph) {
        prepare_build_in_tensor_before();
        struct ggml_cgraph* gf = get_graph();
        auto result            = ggml_graph_node(gf, -1);
        ggml_set_name(result, final_result_name.c_str());
        prepare_build_in_tensor_after(gf);
        return gf;
    }

    bool alloc_compute_buffer(get_graph_cb_t get_graph) {
        if (compute_allocr != NULL) {
            return true;
        }
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_compute_graph(get_graph);
        backend_tensor_data_map.clear();
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
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM");
        return true;
    }

    void free_cache_buffer() {
        if (cache_buffer != NULL) {
            ggml_backend_buffer_free(cache_buffer);
            cache_buffer = NULL;
        }
    }

    void copy_cache_tensors_to_cache_buffer() {
        if (cache_tensor_map.size() == 0) {
            return;
        }
        free_cache_ctx_and_buffer();
        alloc_cache_ctx();
        GGML_ASSERT(cache_buffer == NULL);
        std::map<ggml_tensor*, ggml_tensor*> runtime_tensor_to_cache_tensor;
        for (auto kv : cache_tensor_map) {
            auto cache_tensor = ggml_dup_tensor(cache_ctx, kv.second);
            ggml_set_name(cache_tensor, kv.first.c_str());
            runtime_tensor_to_cache_tensor[kv.second] = cache_tensor;
        }
        size_t num_tensors = ggml_tensor_num(cache_ctx);
        cache_buffer       = ggml_backend_alloc_ctx_tensors(cache_ctx, runtime_backend);
        GGML_ASSERT(cache_buffer != NULL);
        for (auto kv : runtime_tensor_to_cache_tensor) {
            ggml_backend_tensor_copy(kv.first, kv.second);
        }
        ggml_backend_synchronize(runtime_backend);
        cache_tensor_map.clear();
        size_t cache_buffer_size = ggml_backend_buffer_get_size(cache_buffer);
        LOG_DEBUG("%s cache backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  cache_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(runtime_backend) ? "RAM" : "VRAM",
                  num_tensors);
    }

    void copy_data_to_backend_tensor() {
        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        backend_tensor_data_map.clear();
    }

    bool offload_params_to_runtime_backend() {
        if (params_backend == runtime_backend) {
            return true;
        }
        if (params_on_runtime_backend) {
            return true;
        }
        GGML_ASSERT(runtime_params_buffer == NULL);
        int64_t t0         = ggml_time_ms();
        size_t num_tensors = ggml_tensor_num(offload_ctx);
        if (num_tensors == 0) {
            for (ggml_tensor* t = ggml_get_first_tensor(params_ctx); t != NULL; t = ggml_get_next_tensor(params_ctx, t)) {
                GGML_ASSERT(t->view_src == NULL);
                ggml_dup_tensor(offload_ctx, t);
            }
        }
        num_tensors = ggml_tensor_num(offload_ctx);
        GGML_ASSERT(num_tensors == ggml_tensor_num(params_ctx));

        runtime_params_buffer = ggml_backend_alloc_ctx_tensors(offload_ctx, runtime_backend);

        if (runtime_params_buffer == NULL) {
            LOG_ERROR("%s alloc runtime params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }

        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != NULL && offload_t != NULL) {
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

    void offload_params_to_params_backend() {
        if (!params_on_runtime_backend) {
            return;
        }
        ggml_tensor* t         = ggml_get_first_tensor(params_ctx);
        ggml_tensor* offload_t = ggml_get_first_tensor(offload_ctx);

        while (t != NULL && offload_t != NULL) {
            t->buffer         = offload_t->buffer;
            t->data           = offload_t->data;
            t->extra          = offload_t->extra;
            offload_t->buffer = NULL;
            offload_t->data   = NULL;
            offload_t->extra  = NULL;

            t         = ggml_get_next_tensor(params_ctx, t);
            offload_t = ggml_get_next_tensor(offload_ctx, offload_t);
        }

        if (runtime_params_buffer != NULL) {
            ggml_backend_buffer_free(runtime_params_buffer);
            runtime_params_buffer = NULL;
        }
        params_on_runtime_backend = false;
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend, bool offload_params_to_cpu = false)
        : runtime_backend(backend) {
        alloc_params_ctx();
        if (!ggml_backend_is_cpu(runtime_backend) && offload_params_to_cpu) {
            params_backend = ggml_backend_cpu_init();
        } else {
            params_backend = runtime_backend;
        }
    }

    virtual ~GGMLRunner() {
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
        if (params_backend != runtime_backend) {
            ggml_backend_free(params_backend);
        }
        free_cache_ctx_and_buffer();
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        params_buffer      = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
        if (params_buffer == NULL) {
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.f * 1024.f),
                  ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                  num_tensors);
        return true;
    }

    void free_params_buffer() {
        if (params_buffer != NULL) {
            ggml_backend_buffer_free(params_buffer);
            params_buffer = NULL;
        }
    }

    size_t get_params_buffer_size() {
        if (params_buffer != NULL) {
            return ggml_backend_buffer_get_size(params_buffer);
        }
        return 0;
    }

    void free_cache_ctx_and_buffer() {
        free_cache_buffer();
        free_cache_ctx();
    }

    void free_compute_buffer() {
        if (compute_allocr != NULL) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = NULL;
        }
        offload_params_to_params_backend();
    }

    // do copy after alloc graph
    void set_backend_tensor_data(struct ggml_tensor* tensor, const void* data) {
        backend_tensor_data_map[tensor] = data;
    }

    struct ggml_tensor* to_backend(struct ggml_tensor* tensor) {
        GGML_ASSERT(compute_ctx != NULL);
        if (tensor == NULL) {
            return NULL;
        }
        // it's performing a compute, check if backend isn't cpu
        if (!ggml_backend_is_cpu(runtime_backend) && (tensor->buffer == NULL || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void cache(const std::string name, struct ggml_tensor* tensor) {
        cache_tensor_map[name] = tensor;
    }

    struct ggml_tensor* get_cache_tensor_by_name(const std::string& name) {
        if (cache_ctx == NULL) {
            return NULL;
        }
        return ggml_get_tensor(cache_ctx, name.c_str());
    }

    void compute(get_graph_cb_t get_graph,
                 int n_threads,
                 bool free_compute_buffer_immediately = true,
                 struct ggml_tensor** output          = NULL,
                 struct ggml_context* output_ctx      = NULL) {
        if (!offload_params_to_runtime_backend()) {
            LOG_ERROR("%s offload params to runtime backend failed", get_desc().c_str());
            return;
        }
        alloc_compute_buffer(get_graph);
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_compute_graph(get_graph);
        GGML_ASSERT(ggml_gallocr_alloc_graph(compute_allocr, gf));
        copy_data_to_backend_tensor();
        if (ggml_backend_is_cpu(runtime_backend)) {
            ggml_backend_cpu_set_n_threads(runtime_backend, n_threads);
        }

        ggml_backend_graph_compute(runtime_backend, gf);
#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif
        copy_cache_tensors_to_cache_buffer();
        if (output != NULL) {
            auto result = ggml_get_tensor(compute_ctx, final_result_name.c_str());
            if (*output == NULL && output_ctx != NULL) {
                *output = ggml_dup_tensor(output_ctx, result);
            }
            if (*output != NULL) {
                ggml_backend_tensor_get_and_sync(runtime_backend, result, (*output)->data, 0, ggml_nbytes(*output));
            }
        }

        if (free_compute_buffer_immediately) {
            free_compute_buffer();
        }
    }
};

class GGMLBlock {
protected:
    typedef std::unordered_map<std::string, struct ggml_tensor*> ParameterMap;
    typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
    GGMLBlockMap blocks;
    ParameterMap params;

    ggml_type get_type(const std::string& name, const String2GGMLType& tensor_types, ggml_type default_type) {
        auto iter = tensor_types.find(name);
        if (iter != tensor_types.end()) {
            return iter->second;
        }
        return default_type;
    }

    void init_blocks(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_types, prefix + pair.first);
        }
    }

    virtual void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {}

public:
    void init(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        init_blocks(ctx, tensor_types, prefix);
        init_params(ctx, tensor_types, prefix);
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

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix = "") {
        if (prefix.size() > 0) {
            prefix = prefix + ".";
        }
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->get_param_tensors(tensors, prefix + pair.first);
        }

        for (auto& pair : params) {
            struct ggml_tensor* param    = pair.second;
            tensors[prefix + pair.first] = pair.second;
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
    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) = 0;
};

class Identity : public UnaryBlock {
public:
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
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

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_types, GGML_TYPE_F32);
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias      = true,
           bool force_f32 = false,
           bool force_prec_f32 = false)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32),
          force_prec_f32(force_prec_f32) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_linear(ctx, x, w, b, force_prec_f32);
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
    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = get_type(prefix + "weight", tensor_types, GGML_TYPE_F32);
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

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto weight = params["weight"];

        // There are issues with ggml batch inference, so we are expanding it here first.
        // TODO: fix ggml batch inference
        int64_t n = input_ids->ne[1];
        input_ids = ggml_reshape_1d(ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

        input_ids      = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto embedding = ggml_get_rows(ctx, weight, input_ids);
        embedding      = ggml_reshape_3d(ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

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
    bool direct = false;

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types, const std::string prefix = "") {
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

    void enable_direct() {
        direct = true;
    }

    std::string get_desc() {
        return "Conv2d";
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        if (direct) {
            return ggml_nn_conv_2d_direct(ctx, x, w, b, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
        } else {
            return ggml_nn_conv_2d(ctx, x, w, b, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
        }
    }
};

class Conv3dnx1x1 : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;
    int64_t kernel_size;
    int64_t stride;
    int64_t padding;
    int64_t dilation;
    bool bias;

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, 1, kernel_size, in_channels, out_channels);  // 5d => 4d
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_channels);
        }
    }

public:
    Conv3dnx1x1(int64_t in_channels,
                int64_t out_channels,
                int64_t kernel_size,
                int64_t stride   = 1,
                int64_t padding  = 0,
                int64_t dilation = 1,
                bool bias        = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    // x: [N, IC, ID, IH*IW]
    // result: [N, OC, OD, OH*OW]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_conv_3d_nx1x1(ctx, x, w, b, stride, padding, dilation);
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

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types, const std::string prefix = "") {
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
           bool bias                          = true)
        : in_channels(in_channels),
          out_channels(out_channels),
          kernel_size(kernel_size),
          stride(stride),
          padding(padding),
          dilation(dilation),
          bias(bias) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_conv_3d(ctx, x, w, b, in_channels,
                               std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                               std::get<2>(padding), std::get<1>(padding), std::get<0>(padding),
                               std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation));
    }
};

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = NULL;
        struct ggml_tensor* b = NULL;

        if (elementwise_affine) {
            w = params["weight"];
            if (bias) {
                b = params["bias"];
            }
        }
        return ggml_nn_layer_norm(ctx, x, w, b, eps);
    }
};

class GroupNorm : public GGMLBlock {
protected:
    int64_t num_groups;
    int64_t num_channels;
    float eps;
    bool affine;

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;
            params["weight"]          = ggml_new_tensor_1d(ctx, wtype, num_channels);
            params["bias"]            = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
        }
    }

public:
    GroupNorm(int64_t num_groups,
              int64_t num_channels,
              float eps   = 1e-05f,
              bool affine = true)
        : num_groups(num_groups),
          num_channels(num_channels),
          eps(eps),
          affine(affine) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = NULL;
        struct ggml_tensor* b = NULL;
        if (affine) {
            w = params["weight"];
            b = params["bias"];
        }
        return ggml_nn_group_norm(ctx, x, w, b, num_groups);
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

    void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        x                     = ggml_rms_norm(ctx, x, eps);
        x                     = ggml_mul_inplace(ctx, x, w);
        return x;
    }
};

class MultiheadAttention : public GGMLBlock {
protected:
    int64_t embed_dim;
    int64_t n_head;
    std::string q_proj_name;
    std::string k_proj_name;
    std::string v_proj_name;
    std::string out_proj_name;

public:
    MultiheadAttention(int64_t embed_dim,
                       int64_t n_head,
                       bool qkv_proj_bias        = true,
                       bool out_proj_bias        = true,
                       std::string q_proj_name   = "q_proj",
                       std::string k_proj_name   = "k_proj",
                       std::string v_proj_name   = "v_proj",
                       std::string out_proj_name = "out_proj")
        : embed_dim(embed_dim),
          n_head(n_head),
          q_proj_name(q_proj_name),
          k_proj_name(k_proj_name),
          v_proj_name(v_proj_name),
          out_proj_name(out_proj_name) {
        blocks[q_proj_name]   = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[k_proj_name]   = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[v_proj_name]   = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, qkv_proj_bias));
        blocks[out_proj_name] = std::shared_ptr<GGMLBlock>(new Linear(embed_dim, embed_dim, out_proj_bias));
    }

    // x: [N, n_token, embed_dim]
    struct ggml_tensor* forward(struct ggml_context* ctx,
                                ggml_backend_t backend,
                                struct ggml_tensor* x,
                                bool mask = false) {
        auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
        auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
        auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        struct ggml_tensor* q = q_proj->forward(ctx, x);
        struct ggml_tensor* k = k_proj->forward(ctx, x);
        struct ggml_tensor* v = v_proj->forward(ctx, x);

        x = ggml_nn_attention_ext(ctx, backend, q, k, v, n_head, NULL, mask);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

#endif  // __GGML_EXTEND__HPP__

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

#ifdef SD_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "rng.hpp"
#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif

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
                    ggml_upscale_ext(ctx,
                                     a,
                                     a->ne[0] * b->ne[0],
                                     a->ne[1] * b->ne[1],
                                     a->ne[2] * b->ne[2],
                                     a->ne[3] * b->ne[3],
                                     ggml_scale_mode::GGML_SCALE_MODE_NEAREST),
                    b);
}

__STATIC_INLINE__ void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
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

// static struct ggml_tensor* get_tensor_from_graph(struct ggml_cgraph* gf, const char* name) {
//     struct ggml_tensor* res = NULL;
//     for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
//         struct ggml_tensor* node = ggml_graph_node(gf, i);
//         // printf("%d, %s \n", i, ggml_get_name(node));
//         if (strcmp(ggml_get_name(node), name) == 0) {
//             res = node;
//             break;
//         }
//     }
//     return res;
// }

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
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_f16(tensor, l, k, j, i));
                    } else if (tensor->type == GGML_TYPE_I32) {
                        printf("  [%d, %d, %d, %d] = %i\n", i, j, k, l, ggml_tensor_get_i32(tensor, l, k, j, i));
                    }
                    fflush(stdout);
                }
            }
        }
    }
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

__STATIC_INLINE__ uint8_t* sd_tensor_to_mul_image(struct ggml_tensor* input, int idx) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(channels == 3 && input->type == GGML_TYPE_F32);
    uint8_t* image_data = (uint8_t*)malloc(width * height * channels);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value                                               = ggml_tensor_get_f32(input, ix, iy, k, idx);
                *(image_data + iy * width * channels + ix * channels + k) = (uint8_t)(value * 255.0f);
            }
        }
    }
    return image_data;
}

__STATIC_INLINE__ void sd_image_to_tensor(const uint8_t* image_data,
                                          struct ggml_tensor* output,
                                          bool scale = true) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                float value = *(image_data + iy * width * channels + ix * channels + k);
                if (scale) {
                    value /= 255.f;
                }
                ggml_tensor_set_f32(output, value, ix, iy, k);
            }
        }
    }
}

__STATIC_INLINE__ void sd_mask_to_tensor(const uint8_t* image_data,
                                         struct ggml_tensor* output,
                                         bool scale = true) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 1 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            float value = *(image_data + iy * width * channels + ix);
            if (scale) {
                value /= 255.f;
            }
            ggml_tensor_set_f32(output, value, ix, iy);
        }
    }
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

__STATIC_INLINE__ void sd_mul_images_to_tensor(const uint8_t* image_data,
                                               struct ggml_tensor* output,
                                               int idx,
                                               float* mean = NULL,
                                               float* std  = NULL) {
    int64_t width    = output->ne[0];
    int64_t height   = output->ne[1];
    int64_t channels = output->ne[2];
    GGML_ASSERT(channels == 3 && output->type == GGML_TYPE_F32);
    for (int iy = 0; iy < height; iy++) {
        for (int ix = 0; ix < width; ix++) {
            for (int k = 0; k < channels; k++) {
                int value       = *(image_data + iy * width * channels + ix * channels + k);
                float pixel_val = value / 255.0f;
                if (mean != NULL && std != NULL)
                    pixel_val = (pixel_val - mean[k]) / std[k];
                ggml_tensor_set_f32(output, pixel_val, ix, iy, k, idx);
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
__STATIC_INLINE__ void ggml_tensor_scale_input(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

// convert values from [-1, 1] to [0, 1]
__STATIC_INLINE__ void ggml_tensor_scale_output(struct ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

typedef std::function<void(ggml_tensor*, ggml_tensor*, bool)> on_tile_process;

__STATIC_INLINE__ void
sd_tiling_calc_tiles(int &num_tiles_dim, float& tile_overlap_factor_dim, int small_dim, int tile_size, const float tile_overlap_factor) {

    int tile_overlap     = (tile_size * tile_overlap_factor);
    int non_tile_overlap = tile_size - tile_overlap;

    num_tiles_dim = (small_dim - tile_overlap) / non_tile_overlap;
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
__STATIC_INLINE__ void sd_tiling_non_square(ggml_tensor* input, ggml_tensor* output, const int scale,
                                            const int p_tile_size_x, const int p_tile_size_y,
                                            const float tile_overlap_factor, on_tile_process on_processing) {

    output = ggml_set_f32(output, 0);

    int input_width   = (int)input->ne[0];
    int input_height  = (int)input->ne[1];
    int output_width  = (int)output->ne[0];
    int output_height = (int)output->ne[1];

    GGML_ASSERT(input_width / output_width == input_height / output_height && output_width / input_width == output_height / input_height);
    GGML_ASSERT(input_width / output_width == scale || output_width / input_width == scale);

    int small_width  = output_width;
    int small_height = output_height;

    bool big_out = output_width > input_width;
    if (big_out) {
        // Ex: decode
        small_width  = input_width;
        small_height = input_height;
    }

    int num_tiles_x;
    float tile_overlap_factor_x;
    sd_tiling_calc_tiles(num_tiles_x, tile_overlap_factor_x, small_width, p_tile_size_x, tile_overlap_factor);

    int num_tiles_y;
    float tile_overlap_factor_y;
    sd_tiling_calc_tiles(num_tiles_y, tile_overlap_factor_y, small_height, p_tile_size_y, tile_overlap_factor);

    // LOG_DEBUG("num tiles : %d, %d ", num_tiles_x, num_tiles_y);
    // LOG_DEBUG("optimal overlap : %f, %f (targeting %f)", tile_overlap_factor_x, tile_overlap_factor_y, tile_overlap_factor);

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

    if (big_out) {
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
            if (big_out) {
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
                if (big_out) {
                    dx *= scale;
                }
                last_x = true;
            }

            int x_in  = big_out ? x : scale * x;
            int y_in  = big_out ? y : scale * y;
            int x_out = big_out ? x * scale : x;
            int y_out = big_out ? y * scale : y;

            int overlap_x_out = big_out ? tile_overlap_x * scale : tile_overlap_x;
            int overlap_y_out = big_out ? tile_overlap_y * scale : tile_overlap_y;

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

__STATIC_INLINE__ void sd_tiling(ggml_tensor* input, ggml_tensor* output, const int scale,
    const int tile_size, const float tile_overlap_factor, on_tile_process on_processing) {
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
                                                     struct ggml_tensor* b) {
    x = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        x = ggml_add(ctx, x, b);
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
        x = ggml_add(ctx, x, b);
    }
    return x;
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_3d_nx1x1_bak(struct ggml_context* ctx,
                                                                struct ggml_tensor* x,
                                                                struct ggml_tensor* w,
                                                                struct ggml_tensor* b,
                                                                int s2 = 1,
                                                                int p2 = 1,
                                                                int d2 = 1) {
    GGML_ASSERT(w->ne[0] == 1);
    // timesteps = x.shape[0]
    // x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
    // x = conv3d(x)
    // return rearrange(x, "b c t h w -> (b t) c h w")
    int64_t T = x->ne[3];
    int64_t B = x->ne[3] / T;
    int64_t C = x->ne[2];
    int64_t H = x->ne[1];
    int64_t W = x->ne[0];

    x = ggml_reshape_4d(ctx, x, W * H, C, T, B);           // (b t) c h w -> b t c (h w)
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);      // [B, OC, T, OH * OW]
    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
    x = ggml_reshape_4d(ctx, x, W, H, C, T * B);           // b t c (h w) -> (b t) c h w
    return x;                                              // [B*T, OC, OH, OW]
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

// q: [N, L_q, C] or [N*n_head, L_q, d_head]
// k: [N, L_k, C] or [N*n_head, L_k, d_head]
// v: [N, L_k, C] or [N, L_k, n_head, d_head]
// return: [N, L_q, C]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_attention_ext(struct ggml_context* ctx,
                                                            struct ggml_tensor* q,
                                                            struct ggml_tensor* k,
                                                            struct ggml_tensor* v,
                                                            int64_t n_head,
                                                            struct ggml_tensor* mask = NULL,
                                                            bool diag_mask_inf       = false,
                                                            bool skip_reshape        = false,
                                                            bool flash_attn          = false) {
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    if (!skip_reshape) {
        L_q    = q->ne[1];
        L_k    = k->ne[1];
        C      = q->ne[0];
        N      = q->ne[2];
        d_head = C / n_head;
        q      = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);   // [N, L_q, n_head, d_head]
        q      = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q      = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);  // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_head, L_k, N);   // [N, L_k, n_head, d_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_head * N);  // [N * n_head, L_k, d_head]

        v = ggml_reshape_4d(ctx, v, d_head, n_head, L_k, N);  // [N, L_k, n_head, d_head]
    } else {
        L_q    = q->ne[1];
        L_k    = k->ne[1];
        d_head = v->ne[0];
        N      = v->ne[3];
        C      = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    // if (flash_attn) {
    //     LOG_DEBUG("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
    // }
    //  is there anything oddly shaped?? ping Green-Sky if you can trip this assert
    GGML_ASSERT(((L_k % 256 == 0) && L_q == L_k) || !(L_k % 256 == 0));

    bool can_use_flash_attn = true;
    can_use_flash_attn      = can_use_flash_attn && L_k % 256 == 0;
    can_use_flash_attn      = can_use_flash_attn && d_head % 64 == 0;  // double check

    // cuda max d_head seems to be 256, cpu does seem to work with 512
    can_use_flash_attn = can_use_flash_attn && d_head <= 256;  // double check

    if (mask != nullptr) {
        // TODO(Green-Sky): figure out if we can bend t5 to work too
        can_use_flash_attn = can_use_flash_attn && mask->ne[2] == 1;
        can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
    }

    // TODO(Green-Sky): more pad or disable for funny tensor shapes

    ggml_tensor* kqv = nullptr;
    // GGML_ASSERT((flash_attn && can_use_flash_attn) || !flash_attn);
    if (can_use_flash_attn && flash_attn) {
        // LOG_DEBUG("using flash attention");
        k = ggml_cast(ctx, k, GGML_TYPE_F16);

        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [N, n_head, L_k, d_head]
        v = ggml_reshape_3d(ctx, v, d_head, L_k, n_head * N);  // [N * n_head, L_k, d_head]
        v = ggml_cast(ctx, v, GGML_TYPE_F16);

        if (mask != nullptr) {
            mask = ggml_transpose(ctx, mask);

            if (mask->ne[1] < GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD)) {
                LOG_DEBUG("mask dims %ld, %ld, %ld, %ld\n", mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3]);
                LOG_DEBUG("needs padding, padding from %ld to %ld\n", mask->ne[1], GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD));
                mask = ggml_pad(ctx, mask, 0, GGML_PAD(q->ne[1], GGML_KQ_MASK_PAD) - mask->ne[1], 0, 0);
            }

            mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
        }

        kqv = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);

        // kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_k, kqv->nb[1], kqv->nb[2], 0);
        kqv = ggml_view_3d(ctx, kqv, d_head, n_head, L_q, kqv->nb[1], kqv->nb[2], 0);
    } else {
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_head * N);  // [N * n_head, d_head, L_k]

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

    kqv = ggml_cont(ctx, kqv);
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
        x = ggml_mul(ctx, x, w);
        if (b != NULL) {
            x = ggml_add(ctx, x, b);
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
        x = ggml_mul(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add(ctx, x, b);
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
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
    float value;
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
    } else {  // GGML_TYPE_F16
        ggml_fp16_t f16_value;
        ggml_backend_tensor_get(tensor, &f16_value, 0, sizeof(f16_value));
        value = ggml_fp16_to_fp32(f16_value);
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
                                                        int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [N, dim]
    size_t N        = timesteps.size();
    int acutual_dim = dim;
    if (dim % 2 != 0) {
        acutual_dim = dim + 1;
    }
    std::vector<float> embedding(N * acutual_dim, 0.f);
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg                             = timesteps[i] * freqs[j];
            embedding[i * acutual_dim + j]        = std::cos(arg);
            embedding[i * acutual_dim + j + half] = std::sin(arg);
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
    int acutual_dim                  = dim;
    if (dim % 2 != 0) {
        acutual_dim = dim + 1;
    }
    struct ggml_tensor* embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, acutual_dim, timesteps.size());
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
#define MAX_GRAPH_SIZE 32768

struct GGMLRunner {
protected:
    typedef std::function<struct ggml_cgraph*()> get_graph_cb_t;

    struct ggml_context* params_ctx     = NULL;
    ggml_backend_buffer_t params_buffer = NULL;

    struct ggml_context* compute_ctx    = NULL;
    struct ggml_gallocr* compute_allocr = NULL;

    std::map<struct ggml_tensor*, const void*> backend_tensor_data_map;

    ggml_backend_t backend = NULL;

    void alloc_params_ctx() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
        params.mem_buffer = NULL;
        params.no_alloc   = true;

        params_ctx = ggml_init(params);
        GGML_ASSERT(params_ctx != NULL);
    }

    void free_params_ctx() {
        if (params_ctx != NULL) {
            ggml_free(params_ctx);
            params_ctx = NULL;
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

    bool alloc_compute_buffer(get_graph_cb_t get_graph) {
        if (compute_allocr != NULL) {
            return true;
        }
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_graph();
        backend_tensor_data_map.clear();
        compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

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
                  ggml_backend_is_cpu(backend) ? "RAM" : "VRAM");
        return true;
    }

    void cpy_data_to_backend_tensor() {
        for (auto& kv : backend_tensor_data_map) {
            auto tensor = kv.first;
            auto data   = kv.second;

            ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        }

        backend_tensor_data_map.clear();
    }

public:
    virtual std::string get_desc() = 0;

    GGMLRunner(ggml_backend_t backend)
        : backend(backend) {
        alloc_params_ctx();
    }

    virtual ~GGMLRunner() {
        free_params_buffer();
        free_compute_buffer();
        free_params_ctx();
        free_compute_ctx();
    }

    void reset_compute_ctx() {
        free_compute_ctx();
        alloc_compute_ctx();
    }

    bool alloc_params_buffer() {
        size_t num_tensors = ggml_tensor_num(params_ctx);
        params_buffer      = ggml_backend_alloc_ctx_tensors(params_ctx, backend);
        if (params_buffer == NULL) {
            LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
                      get_desc().c_str(),
                      num_tensors);
            return false;
        }
        size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
        LOG_DEBUG("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)",
                  get_desc().c_str(),
                  params_buffer_size / (1024.0 * 1024.0),
                  ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                  num_tensors);
        // printf("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)\n",
        //           get_desc().c_str(),
        //           params_buffer_size / (1024.0 * 1024.0),
        //           ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
        //           num_tensors);
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

    void free_compute_buffer() {
        if (compute_allocr != NULL) {
            ggml_gallocr_free(compute_allocr);
            compute_allocr = NULL;
        }
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
        if (!ggml_backend_is_cpu(backend) && (tensor->buffer == NULL || ggml_backend_buffer_is_host(tensor->buffer))) {
            // pass input tensors to gpu memory
            auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

            set_backend_tensor_data(backend_tensor, tensor->data);
            return backend_tensor;
        } else {
            return tensor;
        }
    }

    void compute(get_graph_cb_t get_graph,
                 int n_threads,
                 bool free_compute_buffer_immediately = true,
                 struct ggml_tensor** output          = NULL,
                 struct ggml_context* output_ctx      = NULL) {
        alloc_compute_buffer(get_graph);
        reset_compute_ctx();
        struct ggml_cgraph* gf = get_graph();
        GGML_ASSERT(ggml_gallocr_alloc_graph(compute_allocr, gf));
        cpy_data_to_backend_tensor();
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

// #ifdef SD_USE_METAL
//         if (ggml_backend_is_metal(backend)) {
//             ggml_backend_metal_set_n_cb(backend, n_threads);
//         }
// #endif
        ggml_backend_graph_compute(backend, gf);

#ifdef GGML_PERF
        ggml_graph_print(gf);
#endif
        if (output != NULL) {
            auto result = ggml_graph_node(gf, -1);
            if (*output == NULL && output_ctx != NULL) {
                *output = ggml_dup_tensor(output_ctx, result);
            }
            if (*output != NULL) {
                ggml_backend_tensor_get_and_sync(backend, result, (*output)->data, 0, ggml_nbytes(*output));
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

    void init_blocks(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        for (auto& pair : blocks) {
            auto& block = pair.second;
            block->init(ctx, tensor_types, prefix + pair.first);
        }
    }

    virtual void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {}

public:
    void init(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
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
};

class UnaryBlock : public GGMLBlock {
public:
    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) = 0;
};

class Linear : public UnaryBlock {
protected:
    int64_t in_features;
    int64_t out_features;
    bool bias;
    bool force_f32;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = (tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
        if (in_features % ggml_blck_size(wtype) != 0 || force_f32) {
            wtype = GGML_TYPE_F32;
        }
        params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
            params["bias"]       = ggml_new_tensor_1d(ctx, wtype, out_features);
        }
    }

public:
    Linear(int64_t in_features,
           int64_t out_features,
           bool bias      = true,
           bool force_f32 = false)
        : in_features(in_features),
          out_features(out_features),
          bias(bias),
          force_f32(force_f32) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_linear(ctx, x, w, b);
    }
};

class Embedding : public UnaryBlock {
protected:
    int64_t embedding_dim;
    int64_t num_embeddings;
    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = (tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
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

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F16;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;  // (tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
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

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        struct ggml_tensor* b = NULL;
        if (bias) {
            b = params["bias"];
        }
        return ggml_nn_conv_2d(ctx, x, w, b, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
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

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F16;                                                              //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
        params["weight"]     = ggml_new_tensor_4d(ctx, wtype, 1, kernel_size, in_channels, out_channels);  // 5d => 4d
        if (bias) {
            enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
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

class LayerNorm : public UnaryBlock {
protected:
    int64_t normalized_shape;
    float eps;
    bool elementwise_affine;
    bool bias;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        if (elementwise_affine) {
            enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
            params["weight"]     = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
            if (bias) {
                enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
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

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        if (affine) {
            enum ggml_type wtype      = GGML_TYPE_F32;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
            enum ggml_type bias_wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
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
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool mask = false) {
        auto q_proj   = std::dynamic_pointer_cast<Linear>(blocks[q_proj_name]);
        auto k_proj   = std::dynamic_pointer_cast<Linear>(blocks[k_proj_name]);
        auto v_proj   = std::dynamic_pointer_cast<Linear>(blocks[v_proj_name]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks[out_proj_name]);

        struct ggml_tensor* q = q_proj->forward(ctx, x);
        struct ggml_tensor* k = k_proj->forward(ctx, x);
        struct ggml_tensor* v = v_proj->forward(ctx, x);

        x = ggml_nn_attention_ext(ctx, q, k, v, n_head, NULL, mask);  // [N, n_token, embed_dim]

        x = out_proj->forward(ctx, x);  // [N, n_token, embed_dim]
        return x;
    }
};

#endif  // __GGML_EXTEND__HPP__
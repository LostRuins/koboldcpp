#include "core/ggml_tensor_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "core/ggml_extend_backend.h"
#include "core/rng.hpp"

void ggml_ext_im_set_randn_f32(ggml_tensor* tensor, std::shared_ptr<RNG> rng) {
    uint32_t n                        = (uint32_t)ggml_nelements(tensor);
    std::vector<float> random_numbers = rng->randn(n);
    for (uint32_t i = 0; i < n; i++) {
        ggml_ext_im_set_f32_1d(tensor, i, random_numbers[i]);
    }
}

void print_ggml_tensor(ggml_tensor* tensor, bool shape_only, const char* mark) {
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

void ggml_ext_tensor_iter(
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

void ggml_ext_tensor_iter(
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

void ggml_ext_tensor_diff(
    ggml_tensor* a,
    ggml_tensor* b,
    float gap) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    ggml_ext_tensor_iter(a, [&](ggml_tensor* a, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        float a_value = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
        float b_value = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
        if (abs(a_value - b_value) > gap) {
            LOG_WARN("[%ld, %ld, %ld, %ld] %f %f", i3, i2, i1, i0, a_value, b_value);
        }
    });
}

ggml_tensor* load_tensor_from_file(ggml_context* ctx, const std::string& file_path) {
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

    LOG_VERBOSE("load_tensor_from_file %d %d %d", n_dims, length, ttype);

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

uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, uint8_t* image_data) {
    int64_t width    = input->ne[0];
    int64_t height   = input->ne[1];
    int64_t channels = input->ne[2];
    GGML_ASSERT(input->type == GGML_TYPE_F32);
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

uint8_t* ggml_tensor_to_sd_image(ggml_tensor* input, int idx, bool video) {
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

void sd_image_to_ggml_tensor(sd_image_t image,
                             ggml_tensor* tensor,
                             bool scale) {
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

void ggml_ext_tensor_apply_mask(ggml_tensor* image_data,
                                ggml_tensor* mask,
                                ggml_tensor* output,
                                float masked_value) {
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

float ggml_ext_tensor_mean(ggml_tensor* src) {
    float mean        = 0.0f;
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        mean += data[i] / nelements * 1.0f;
    }
    return mean;
}

void ggml_ext_tensor_add_inplace(ggml_tensor* a, ggml_tensor* b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));
    int64_t nelements = ggml_nelements(a);
    float* vec_a      = (float*)a->data;
    float* vec_b      = (float*)b->data;
    for (int i = 0; i < nelements; i++) {
        vec_a[i] = vec_a[i] + vec_b[i];
    }
}

void ggml_ext_tensor_scale_inplace(ggml_tensor* src, float scale) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        data[i] = data[i] * scale;
    }
}

void ggml_ext_tensor_clamp_inplace(ggml_tensor* src, float min, float max) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val < min ? min : (val > max ? max : val);
    }
}

ggml_tensor* ggml_ext_tensor_concat(ggml_context* ctx,
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

void scale_to_minus1_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = val * 2.0f - 1.0f;
    }
}

void scale_to_0_1(ggml_tensor* src) {
    int64_t nelements = ggml_nelements(src);
    float* data       = (float*)src->data;
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i]   = (val + 1.0f) * 0.5f;
    }
}

ggml_tensor* vector_to_ggml_tensor(ggml_context* ctx,
                                   const std::vector<float>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

ggml_tensor* vector_to_ggml_tensor_i32(ggml_context* ctx,
                                       const std::vector<int>& vec) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vec.size());
    memcpy(t->data, (const void*)vec.data(), ggml_nbytes(t));
    return t;
}

std::vector<float> arange(float start, float end, float step) {
    std::vector<float> result;

    for (float value = start; value < end; value += step) {
        result.push_back(value);
    }

    return result;
}

std::vector<float> timestep_embedding(std::vector<float> timesteps,
                                      int dim,
                                      int max_period,
                                      bool flip_sin_to_cos,
                                      float scale) {
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

void set_timestep_embedding(std::vector<float> timesteps,
                            ggml_tensor* embedding,
                            int dim,
                            int max_period) {
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    memcpy(((char*)embedding->data), ((char*)embedding_vec.data()), ggml_nbytes(embedding));
}

void set_timestep_embedding(std::vector<float> timesteps,
                            sd::Tensor<float>* embedding,
                            int dim,
                            int max_period) {
    GGML_ASSERT(embedding != nullptr);
    std::vector<float> embedding_vec = timestep_embedding(timesteps, dim, max_period);
    if (embedding->numel() != static_cast<int64_t>(embedding_vec.size())) {
        embedding->resize({dim, static_cast<int64_t>(timesteps.size())});
    }
    std::copy(embedding_vec.begin(), embedding_vec.end(), embedding->values().begin());
}

ggml_tensor* new_timestep_embedding(ggml_context* ctx,
                                    std::vector<float> timesteps,
                                    int dim,
                                    int max_period) {
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

size_t ggml_tensor_num(ggml_context* ctx) {
    size_t num = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        num++;
    }
    return num;
}

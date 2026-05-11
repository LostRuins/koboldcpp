#ifndef __PREPROCESSING_HPP__
#define __PREPROCESSING_HPP__

#include <cmath>
#include <limits>

#include "ggml_extend.hpp"

#define M_PI_ 3.14159265358979323846f

static inline int64_t preprocessing_offset_4d(const sd::Tensor<float>& tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    const auto& shape = tensor.shape();
    int64_t n0        = shape.size() > 0 ? shape[0] : 1;
    int64_t n1        = shape.size() > 1 ? shape[1] : 1;
    int64_t n2        = shape.size() > 2 ? shape[2] : 1;
    return ((i3 * n2 + i2) * n1 + i1) * n0 + i0;
}

static inline float preprocessing_get_4d(const sd::Tensor<float>& tensor, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    return tensor.values()[static_cast<size_t>(preprocessing_offset_4d(tensor, i0, i1, i2, i3))];
}

static inline void preprocessing_set_4d(sd::Tensor<float>& tensor, float value, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    tensor.values()[static_cast<size_t>(preprocessing_offset_4d(tensor, i0, i1, i2, i3))] = value;
}

static inline uint8_t preprocessing_float_to_u8(float value) {
    if (value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

static inline void preprocessing_tensor_frame_to_sd_image(const sd::Tensor<float>& tensor, int frame_index, uint8_t* image_data) {
    const auto& shape = tensor.shape();
    GGML_ASSERT(shape.size() == 4 || shape.size() == 5);
    GGML_ASSERT(image_data != nullptr);

    const int width     = static_cast<int>(shape[0]);
    const int height    = static_cast<int>(shape[1]);
    const int channel   = static_cast<int>(shape[shape.size() == 5 ? 3 : 2]);
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const float* src    = tensor.data();

    if (shape.size() == 4) {
        GGML_ASSERT(frame_index >= 0 && frame_index < shape[3]);
        const size_t frame_stride = pixels * static_cast<size_t>(channel);
        const float* frame_ptr    = src + static_cast<size_t>(frame_index) * frame_stride;
        if (channel == 3) {
            const float* c0 = frame_ptr;
            const float* c1 = frame_ptr + pixels;
            const float* c2 = frame_ptr + pixels * 2;
            for (size_t i = 0; i < pixels; ++i) {
                image_data[i * 3 + 0] = preprocessing_float_to_u8(c0[i]);
                image_data[i * 3 + 1] = preprocessing_float_to_u8(c1[i]);
                image_data[i * 3 + 2] = preprocessing_float_to_u8(c2[i]);
            }
            return;
        }

        for (size_t i = 0; i < pixels; ++i) {
            for (int c = 0; c < channel; ++c) {
                image_data[i * static_cast<size_t>(channel) + static_cast<size_t>(c)] =
                    preprocessing_float_to_u8(frame_ptr[i + pixels * static_cast<size_t>(c)]);
            }
        }
        return;
    }

    GGML_ASSERT(frame_index >= 0 && frame_index < shape[2]);
    const size_t channel_stride = pixels * static_cast<size_t>(shape[2]);
    const float* frame_ptr      = src + static_cast<size_t>(frame_index) * pixels;
    if (channel == 3) {
        const float* c0 = frame_ptr;
        const float* c1 = frame_ptr + channel_stride;
        const float* c2 = frame_ptr + channel_stride * 2;
        for (size_t i = 0; i < pixels; ++i) {
            image_data[i * 3 + 0] = preprocessing_float_to_u8(c0[i]);
            image_data[i * 3 + 1] = preprocessing_float_to_u8(c1[i]);
            image_data[i * 3 + 2] = preprocessing_float_to_u8(c2[i]);
        }
        return;
    }

    for (size_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < channel; ++c) {
            image_data[i * static_cast<size_t>(channel) + static_cast<size_t>(c)] =
                preprocessing_float_to_u8(frame_ptr[i + channel_stride * static_cast<size_t>(c)]);
        }
    }
}

static inline sd::Tensor<float> sd_image_to_preprocessing_tensor(sd_image_t image) {
    sd::Tensor<float> tensor({static_cast<int64_t>(image.width), static_cast<int64_t>(image.height), static_cast<int64_t>(image.channel), 1});
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            for (uint32_t c = 0; c < image.channel; ++c) {
                preprocessing_set_4d(tensor, sd_image_get_f32(image, x, y, c), x, y, c, 0);
            }
        }
    }
    return tensor;
}

static inline void preprocessing_tensor_to_sd_image(const sd::Tensor<float>& tensor, uint8_t* image_data) {
    GGML_ASSERT(tensor.dim() == 4);
    GGML_ASSERT(tensor.shape()[3] == 1);
    preprocessing_tensor_frame_to_sd_image(tensor, 0, image_data);
}

static inline sd::Tensor<float> gaussian_kernel_tensor(int kernel_size) {
    sd::Tensor<float> kernel({kernel_size, kernel_size, 1, 1});
    int ks_mid   = kernel_size / 2;
    float sigma  = 1.4f;
    float normal = 1.f / (2.0f * M_PI_ * std::pow(sigma, 2.0f));
    for (int y = 0; y < kernel_size; ++y) {
        float gx = static_cast<float>(-ks_mid + y);
        for (int x = 0; x < kernel_size; ++x) {
            float gy = static_cast<float>(-ks_mid + x);
            float k  = std::exp(-((gx * gx + gy * gy) / (2.0f * std::pow(sigma, 2.0f)))) * normal;
            preprocessing_set_4d(kernel, k, x, y, 0, 0);
        }
    }
    return kernel;
}

static inline sd::Tensor<float> convolve_tensor(const sd::Tensor<float>& input, const sd::Tensor<float>& kernel, int padding) {
    GGML_ASSERT(input.dim() == 4);
    GGML_ASSERT(kernel.dim() == 4);
    GGML_ASSERT(input.shape()[3] == 1);
    GGML_ASSERT(kernel.shape()[2] == 1);
    GGML_ASSERT(kernel.shape()[3] == 1);

    sd::Tensor<float> output(input.shape());
    int64_t width    = input.shape()[0];
    int64_t height   = input.shape()[1];
    int64_t channels = input.shape()[2];
    int64_t kernel_w = kernel.shape()[0];
    int64_t kernel_h = kernel.shape()[1];

    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int64_t ky = 0; ky < kernel_h; ++ky) {
                    int64_t iy = y + ky - padding;
                    if (iy < 0 || iy >= height) {
                        continue;
                    }
                    for (int64_t kx = 0; kx < kernel_w; ++kx) {
                        int64_t ix = x + kx - padding;
                        if (ix < 0 || ix >= width) {
                            continue;
                        }
                        sum += preprocessing_get_4d(input, ix, iy, c, 0) * preprocessing_get_4d(kernel, kx, ky, 0, 0);
                    }
                }
                preprocessing_set_4d(output, sum, x, y, c, 0);
            }
        }
    }
    return output;
}

static inline sd::Tensor<float> grayscale_tensor(const sd::Tensor<float>& rgb_img) {
    GGML_ASSERT(rgb_img.dim() == 4);
    GGML_ASSERT(rgb_img.shape()[2] >= 3);
    sd::Tensor<float> grayscale({rgb_img.shape()[0], rgb_img.shape()[1], 1, rgb_img.shape()[3]});
    for (int64_t iy = 0; iy < rgb_img.shape()[1]; ++iy) {
        for (int64_t ix = 0; ix < rgb_img.shape()[0]; ++ix) {
            float r    = preprocessing_get_4d(rgb_img, ix, iy, 0, 0);
            float g    = preprocessing_get_4d(rgb_img, ix, iy, 1, 0);
            float b    = preprocessing_get_4d(rgb_img, ix, iy, 2, 0);
            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
            preprocessing_set_4d(grayscale, gray, ix, iy, 0, 0);
        }
    }
    return grayscale;
}

static inline sd::Tensor<float> tensor_hypot(const sd::Tensor<float>& x, const sd::Tensor<float>& y) {
    sd::tensor_check_same_shape(x, y);
    sd::Tensor<float> out(x.shape());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out[i] = std::sqrt(x[i] * x[i] + y[i] * y[i]);
    }
    return out;
}

static inline sd::Tensor<float> tensor_arctan2(const sd::Tensor<float>& x, const sd::Tensor<float>& y) {
    sd::tensor_check_same_shape(x, y);
    sd::Tensor<float> out(x.shape());
    for (int64_t i = 0; i < out.numel(); ++i) {
        out[i] = std::atan2(y[i], x[i]);
    }
    return out;
}

static inline void normalize_tensor(sd::Tensor<float>* g) {
    GGML_ASSERT(g != nullptr);
    if (g->empty()) {
        return;
    }
    float max_value = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < g->numel(); ++i) {
        max_value = std::max(max_value, (*g)[i]);
    }
    if (max_value == 0.0f || !std::isfinite(max_value)) {
        return;
    }
    *g *= (1.0f / max_value);
}

static inline sd::Tensor<float> non_max_supression(const sd::Tensor<float>& G, const sd::Tensor<float>& D) {
    GGML_ASSERT(G.shape() == D.shape());
    sd::Tensor<float> result = sd::Tensor<float>::zeros(G.shape());
    for (int64_t iy = 1; iy < result.shape()[1] - 1; ++iy) {
        for (int64_t ix = 1; ix < result.shape()[0] - 1; ++ix) {
            float angle = preprocessing_get_4d(D, ix, iy, 0, 0) * 180.0f / M_PI_;
            angle       = angle < 0.0f ? angle + 180.0f : angle;
            float q     = 1.0f;
            float r     = 1.0f;

            if ((0 >= angle && angle < 22.5f) || (157.5f >= angle && angle <= 180.0f)) {
                q = preprocessing_get_4d(G, ix, iy + 1, 0, 0);
                r = preprocessing_get_4d(G, ix, iy - 1, 0, 0);
            } else if (22.5f >= angle && angle < 67.5f) {
                q = preprocessing_get_4d(G, ix + 1, iy - 1, 0, 0);
                r = preprocessing_get_4d(G, ix - 1, iy + 1, 0, 0);
            } else if (67.5f >= angle && angle < 112.5f) {
                q = preprocessing_get_4d(G, ix + 1, iy, 0, 0);
                r = preprocessing_get_4d(G, ix - 1, iy, 0, 0);
            } else if (112.5f >= angle && angle < 157.5f) {
                q = preprocessing_get_4d(G, ix - 1, iy - 1, 0, 0);
                r = preprocessing_get_4d(G, ix + 1, iy + 1, 0, 0);
            }

            float cur = preprocessing_get_4d(G, ix, iy, 0, 0);
            preprocessing_set_4d(result, (cur >= q && cur >= r) ? cur : 0.0f, ix, iy, 0, 0);
        }
    }
    return result;
}

static inline void threshold_hystersis(sd::Tensor<float>* img, float high_threshold, float low_threshold, float weak, float strong) {
    GGML_ASSERT(img != nullptr);
    if (img->empty()) {
        return;
    }
    float max_value = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < img->numel(); ++i) {
        max_value = std::max(max_value, (*img)[i]);
    }

    float ht = max_value * high_threshold;
    float lt = ht * low_threshold;
    for (int64_t i = 0; i < img->numel(); ++i) {
        float img_v = (*img)[i];
        if (img_v >= ht) {
            (*img)[i] = strong;
        } else if (img_v <= ht && img_v >= lt) {
            (*img)[i] = weak;
        }
    }

    for (int64_t iy = 0; iy < img->shape()[1]; ++iy) {
        for (int64_t ix = 0; ix < img->shape()[0]; ++ix) {
            if (!(ix >= 3 && ix <= img->shape()[0] - 3 && iy >= 3 && iy <= img->shape()[1] - 3)) {
                preprocessing_set_4d(*img, 0.0f, ix, iy, 0, 0);
            }
        }
    }

    for (int64_t iy = 1; iy < img->shape()[1] - 1; ++iy) {
        for (int64_t ix = 1; ix < img->shape()[0] - 1; ++ix) {
            float imd_v = preprocessing_get_4d(*img, ix, iy, 0, 0);
            if (imd_v == weak) {
                bool has_strong_neighbor =
                    preprocessing_get_4d(*img, ix + 1, iy - 1, 0, 0) == strong ||
                    preprocessing_get_4d(*img, ix + 1, iy, 0, 0) == strong ||
                    preprocessing_get_4d(*img, ix, iy - 1, 0, 0) == strong ||
                    preprocessing_get_4d(*img, ix, iy + 1, 0, 0) == strong ||
                    preprocessing_get_4d(*img, ix - 1, iy - 1, 0, 0) == strong ||
                    preprocessing_get_4d(*img, ix - 1, iy, 0, 0) == strong;
                preprocessing_set_4d(*img, has_strong_neighbor ? strong : 0.0f, ix, iy, 0, 0);
            }
        }
    }
}

bool preprocess_canny(sd_image_t img, float high_threshold, float low_threshold, float weak, float strong, bool inverse) {
    float kX[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    float kY[9] = {
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1};

    sd::Tensor<float> gkernel = gaussian_kernel_tensor(5);
    sd::Tensor<float> sf_kx({3, 3, 1, 1}, std::vector<float>(kX, kX + 9));
    sd::Tensor<float> sf_ky({3, 3, 1, 1}, std::vector<float>(kY, kY + 9));

    sd::Tensor<float> image      = sd_image_to_preprocessing_tensor(img);
    sd::Tensor<float> image_gray = grayscale_tensor(image);
    image_gray                   = convolve_tensor(image_gray, gkernel, 2);
    sd::Tensor<float> iX         = convolve_tensor(image_gray, sf_kx, 1);
    sd::Tensor<float> iY         = convolve_tensor(image_gray, sf_ky, 1);
    sd::Tensor<float> G          = tensor_hypot(iX, iY);
    normalize_tensor(&G);
    sd::Tensor<float> theta = tensor_arctan2(iX, iY);
    image_gray              = non_max_supression(G, theta);
    threshold_hystersis(&image_gray, high_threshold, low_threshold, weak, strong);

    for (uint32_t iy = 0; iy < img.height; ++iy) {
        for (uint32_t ix = 0; ix < img.width; ++ix) {
            float gray = preprocessing_get_4d(image_gray, ix, iy, 0, 0);
            gray       = inverse ? 1.0f - gray : gray;
            for (uint32_t c = 0; c < img.channel; ++c) {
                preprocessing_set_4d(image, gray, ix, iy, c, 0);
            }
        }
    }

    preprocessing_tensor_to_sd_image(image, img.data);
    return true;
}

#endif  // __PREPROCESSING_HPP__

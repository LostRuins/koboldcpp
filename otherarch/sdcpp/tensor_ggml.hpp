#ifndef __SD_TENSOR_GGML_HPP__
#define __SD_TENSOR_GGML_HPP__

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ggml.h"
#include "tensor.hpp"

namespace sd {

    template <typename T>
    struct GGMLTypeTraits;

    template <>
    struct GGMLTypeTraits<float> {
        static constexpr ggml_type type = GGML_TYPE_F32;
    };

    template <>
    struct GGMLTypeTraits<ggml_fp16_t> {
        static constexpr ggml_type type = GGML_TYPE_F16;
    };

    template <>
    struct GGMLTypeTraits<int32_t> {
        static constexpr ggml_type type = GGML_TYPE_I32;
    };

    template <>
    struct GGMLTypeTraits<int64_t> {
        static constexpr ggml_type type = GGML_TYPE_I64;
    };

    inline std::vector<int64_t> shape_from_ggml(const ggml_tensor* tensor) {
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(ggml_n_dims(tensor)));
        for (int i = 0; i < ggml_n_dims(tensor); ++i) {
            shape.push_back(tensor->ne[i]);
        }
        return shape;
    }

    template <typename T>
    inline Tensor<T> make_sd_tensor_from_ggml(const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            return {};
        }
        if (tensor->type != GGMLTypeTraits<T>::type) {
            GGML_ABORT("ggml tensor type does not match sd::Tensor type");
        }
        Tensor<T> result(shape_from_ggml(tensor));
        if (tensor->buffer != nullptr) {
            ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
        } else {
            std::memcpy(result.data(), tensor->data, ggml_nbytes(tensor));
        }
        return result;
    }

    template <typename T>
    inline ggml_tensor* make_ggml_tensor(ggml_context* ctx, const Tensor<T>& tensor, bool copy_data = true) {
        GGML_ASSERT(tensor.dim() > 0 && tensor.dim() <= 5);

        int n_dims = std::min(static_cast<int>(tensor.dim()), GGML_MAX_DIMS);

        std::array<int64_t, GGML_MAX_DIMS> ne = {1, 1, 1, 1};
        for (int64_t i = 0; i < n_dims; ++i) {
            ne[static_cast<size_t>(i)] = tensor.shape()[static_cast<size_t>(i)];
        }

        if (tensor.dim() == 5) {
            ne[3] *= tensor.shape()[4];
        }

        ggml_tensor* result = ggml_new_tensor(ctx, GGMLTypeTraits<T>::type, n_dims, ne.data());
        if (copy_data && tensor.numel() > 0) {
            std::memcpy(result->data, tensor.data(), static_cast<size_t>(ggml_nbytes(result)));
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> load_tensor_from_file_as_tensor(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open tensor file: " + file_path);
        }

        int32_t n_dims = 0;
        int32_t length = 0;
        int32_t ttype  = 0;
        file.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        file.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));
        if (!file.good()) {
            throw std::runtime_error("incomplete tensor file header: " + file_path);
        }
        if (static_cast<ggml_type>(ttype) != GGMLTypeTraits<T>::type) {
            throw std::invalid_argument("tensor file type does not match requested sd::Tensor type");
        }

        std::vector<int64_t> shape(4, 1);
        for (int i = 0; i < n_dims; ++i) {
            int32_t dim = 1;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            shape[static_cast<size_t>(i)] = dim;
        }
        std::string name(static_cast<size_t>(length), '\0');
        file.read(name.data(), length);

        shape.resize(static_cast<size_t>(n_dims));
        Tensor<T> tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(tensor.numel() * sizeof(T)));
        if (!file.good()) {
            throw std::runtime_error("incomplete tensor file data: " + file_path);
        }
        return tensor;
    }

}  // namespace sd

#endif

#ifndef __SD_TENSOR_HPP__
#define __SD_TENSOR_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rng.hpp"

namespace sd {

    template <typename T>
    class Tensor;

    inline std::vector<int64_t> tensor_unravel_index(int64_t flat, const std::vector<int64_t>& shape);

    [[noreturn]] inline void tensor_throw_invalid_argument(const std::string& message) {
        std::fprintf(stderr, "sd::Tensor error: %s\n", message.c_str());
        std::fflush(stderr);
        throw std::invalid_argument(message);
    }

    inline std::string tensor_shape_to_string(const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }

    inline int64_t tensor_numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        int64_t numel = 1;
        for (int64_t dim : shape) {
            if (dim < 0) {
                tensor_throw_invalid_argument("Tensor shape must be non-negative, got shape=" +
                                              tensor_shape_to_string(shape));
            }
            numel *= dim;
        }
        return numel;
    }

    template <typename T>
    class Tensor {
    public:
        Tensor() = default;

        explicit Tensor(std::vector<int64_t> shape)
            : data_(static_cast<size_t>(tensor_numel(shape))), shape_(std::move(shape)) {
        }

        Tensor(std::vector<int64_t> shape, std::vector<T> data)
            : data_(std::move(data)), shape_(std::move(shape)) {
            if (static_cast<int64_t>(data_.size()) != tensor_numel(shape_)) {
                tensor_throw_invalid_argument("Tensor data size does not match shape: data.size()=" +
                                              std::to_string(data_.size()) + ", shape=" +
                                              tensor_shape_to_string(shape_) + ", numel=" +
                                              std::to_string(tensor_numel(shape_)));
            }
        }

        const std::vector<int64_t>& shape() const {
            return shape_;
        }

        int64_t dim() const {
            return static_cast<int64_t>(shape_.size());
        }

        int64_t numel() const {
            return static_cast<int64_t>(data_.size());
        }

        bool empty() const {
            return data_.empty();
        }

        T* data() {
            return data_.data();
        }

        const T* data() const {
            return data_.data();
        }

        std::vector<T>& values() {
            return data_;
        }

        const std::vector<T>& values() const {
            return data_;
        }

        void resize(std::vector<int64_t> shape) {
            shape_ = std::move(shape);
            data_.resize(static_cast<size_t>(tensor_numel(shape_)));
        }

        Tensor& reshape_(std::vector<int64_t> shape) {
            if (tensor_numel(shape) != numel()) {
                tensor_throw_invalid_argument("Tensor reshape changes element count: from shape=" +
                                              tensor_shape_to_string(shape_) + " (numel=" +
                                              std::to_string(numel()) + ") to shape=" +
                                              tensor_shape_to_string(shape) + " (numel=" +
                                              std::to_string(tensor_numel(shape)) + ")");
            }
            shape_ = std::move(shape);
            return *this;
        }

        Tensor reshape(std::vector<int64_t> shape) const {
            Tensor result = *this;
            result.reshape_(std::move(shape));
            return result;
        }

        Tensor& squeeze_() {
            std::vector<int64_t> new_shape;
            new_shape.reserve(shape_.size());
            for (int64_t dim : shape_) {
                if (dim != 1) {
                    new_shape.push_back(dim);
                }
            }
            shape_ = std::move(new_shape);
            return *this;
        }

        Tensor& squeeze_(size_t dim) {
            if (dim >= shape_.size()) {
                tensor_throw_invalid_argument("Tensor squeeze dimension out of range: dim=" +
                                              std::to_string(dim) + ", shape=" +
                                              tensor_shape_to_string(shape_));
            }
            if (shape_[dim] != 1) {
                tensor_throw_invalid_argument("Tensor squeeze requires dimension size 1: dim=" +
                                              std::to_string(dim) + ", shape=" +
                                              tensor_shape_to_string(shape_));
            }
            shape_.erase(shape_.begin() + static_cast<std::ptrdiff_t>(dim));
            return *this;
        }

        Tensor squeeze() const {
            Tensor result = *this;
            result.squeeze_();
            return result;
        }

        Tensor squeeze(size_t dim) const {
            Tensor result = *this;
            result.squeeze_(dim);
            return result;
        }

        Tensor& unsqueeze_(size_t dim) {
            if (dim > shape_.size()) {
                tensor_throw_invalid_argument("Tensor unsqueeze dimension out of range: dim=" +
                                              std::to_string(dim) + ", shape=" +
                                              tensor_shape_to_string(shape_));
            }
            shape_.insert(shape_.begin() + static_cast<std::ptrdiff_t>(dim), 1);
            return *this;
        }

        Tensor unsqueeze(size_t dim) const {
            Tensor result = *this;
            result.unsqueeze_(dim);
            return result;
        }

        Tensor permute(const std::vector<size_t>& dims) const {
            if (dims.size() != static_cast<size_t>(dim())) {
                tensor_throw_invalid_argument("Tensor permute requires one dimension index per axis: tensor_shape=" +
                                              tensor_shape_to_string(shape_) + ", dims_size=" +
                                              std::to_string(dims.size()));
            }

            std::vector<bool> seen(dims.size(), false);
            std::vector<int64_t> out_shape(dims.size(), 1);
            for (size_t i = 0; i < dims.size(); ++i) {
                size_t dim_index = dims[i];
                if (dim_index >= dims.size() || seen[dim_index]) {
                    tensor_throw_invalid_argument("Tensor permute dimensions must be a valid permutation: tensor_shape=" +
                                                  tensor_shape_to_string(shape_));
                }
                seen[dim_index] = true;
                out_shape[i]    = shape_[dim_index];
            }

            Tensor result(out_shape);
            if (result.numel() == 0) {
                return result;
            }

            for (int64_t flat = 0; flat < result.numel(); ++flat) {
                std::vector<int64_t> out_coord = tensor_unravel_index(flat, out_shape);
                std::vector<int64_t> src_coord(static_cast<size_t>(dim()), 0);
                for (size_t i = 0; i < dims.size(); ++i) {
                    src_coord[dims[i]] = out_coord[i];
                }
                result[flat] = index(src_coord);
            }

            return result;
        }

        Tensor& permute_(const std::vector<size_t>& dims) {
            *this = permute(dims);
            return *this;
        }

        void fill_(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        Tensor& masked_fill_(const Tensor<uint8_t>& mask, const T& value);

        T mean() const;

        static Tensor zeros(std::vector<int64_t> shape) {
            return Tensor(std::move(shape));
        }

        static Tensor zeros_like(const Tensor& other) {
            return zeros(other.shape());
        }

        static Tensor ones(std::vector<int64_t> shape) {
            return full(std::move(shape), static_cast<T>(1));
        }

        static Tensor ones_like(const Tensor& other) {
            return ones(other.shape());
        }

        static Tensor full(std::vector<int64_t> shape, const T& value) {
            Tensor tensor(std::move(shape));
            tensor.fill_(value);
            return tensor;
        }

        static Tensor randn(std::vector<int64_t> shape, const std::shared_ptr<RNG>& rng) {
            static_assert(std::is_same_v<T, float>, "Tensor::randn currently requires Tensor<float>");
            if (!rng) {
                tensor_throw_invalid_argument("Tensor randn requires a valid RNG");
            }
            const uint32_t size = static_cast<uint32_t>(tensor_numel(shape));
            return Tensor(std::move(shape), rng->randn(size));
        }

        static Tensor randn_like(const Tensor& other, const std::shared_ptr<RNG>& rng) {
            return randn(other.shape(), rng);
        }

        static Tensor from_vector(std::vector<T> data) {
            const int64_t size = static_cast<int64_t>(data.size());
            return Tensor({size}, std::move(data));
        }

        T& index(const std::vector<int64_t>& coord) {
            return data_.at(offset_of(coord));
        }

        const T& index(const std::vector<int64_t>& coord) const {
            return data_.at(offset_of(coord));
        }

        template <typename... Indices, typename = std::enable_if_t<(std::is_convertible_v<Indices, int64_t> && ...)>>
        T& index(Indices... indices) {
            return index(std::vector<int64_t>{static_cast<int64_t>(indices)...});
        }

        template <typename... Indices, typename = std::enable_if_t<(std::is_convertible_v<Indices, int64_t> && ...)>>
        const T& index(Indices... indices) const {
            return index(std::vector<int64_t>{static_cast<int64_t>(indices)...});
        }

        T& operator[](int64_t index) {
            return data_.at(static_cast<size_t>(index));
        }

        const T& operator[](int64_t index) const {
            return data_.at(static_cast<size_t>(index));
        }

    private:
        size_t offset_of(const std::vector<int64_t>& coord) const {
            if (coord.size() != shape_.size()) {
                tensor_throw_invalid_argument("Tensor index rank mismatch: coord_rank=" +
                                              std::to_string(coord.size()) + ", shape=" +
                                              tensor_shape_to_string(shape_));
            }
            size_t offset = 0;
            size_t stride = 1;
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (coord[i] < 0 || coord[i] >= shape_[i]) {
                    tensor_throw_invalid_argument("Tensor index out of range: shape=" +
                                                  tensor_shape_to_string(shape_));
                }
                offset += static_cast<size_t>(coord[i]) * stride;
                stride *= static_cast<size_t>(shape_[i]);
            }
            return offset;
        }

        std::vector<T> data_;
        std::vector<int64_t> shape_;
    };

    template <typename T>
    inline T Tensor<T>::mean() const {
        if (empty()) {
            return T{};
        }
        T sum = T{};
        for (const T& value : data_) {
            sum += value;
        }
        return sum / static_cast<T>(numel());
    }

    template <>
    inline float Tensor<float>::mean() const {
        if (empty()) {
            return 0.0f;
        }
        double sum = 0.0;
        for (float value : data_) {
            sum += static_cast<double>(value);
        }
        return static_cast<float>(sum / static_cast<double>(numel()));
    }

    template <typename T>
    inline void tensor_check_same_shape(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (lhs.shape() != rhs.shape()) {
            tensor_throw_invalid_argument("Tensor shapes must match: lhs_shape=" +
                                          tensor_shape_to_string(lhs.shape()) + ", rhs_shape=" +
                                          tensor_shape_to_string(rhs.shape()));
        }
    }

    inline std::vector<int64_t> tensor_broadcast_shape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
        size_t ndim = std::max(lhs.size(), rhs.size());
        std::vector<int64_t> shape(ndim, 1);
        for (size_t i = 0; i < ndim; ++i) {
            int64_t lhs_dim = lhs.size() > i ? lhs[i] : 1;
            int64_t rhs_dim = rhs.size() > i ? rhs[i] : 1;
            if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
                tensor_throw_invalid_argument("Tensor shapes are not broadcastable: lhs_shape=" +
                                              tensor_shape_to_string(lhs) + ", rhs_shape=" +
                                              tensor_shape_to_string(rhs));
            }
            shape[i] = std::max(lhs_dim, rhs_dim);
        }
        return shape;
    }

    inline std::vector<int64_t> tensor_unravel_index(int64_t flat, const std::vector<int64_t>& shape) {
        std::vector<int64_t> coord(shape.size(), 0);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] <= 0) {
                tensor_throw_invalid_argument("Tensor unravel_index requires positive shape: shape=" +
                                              tensor_shape_to_string(shape));
            }
            coord[i] = flat % shape[i];
            flat /= shape[i];
        }
        return coord;
    }

    inline std::vector<int64_t> tensor_compute_strides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        int64_t stride = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    template <typename F>
    inline void tensor_for_each_broadcast_offset(const std::vector<int64_t>& out_shape,
                                                 const std::vector<int64_t>& lhs_shape_raw,
                                                 const std::vector<int64_t>& lhs_strides_raw,
                                                 const std::vector<int64_t>& rhs_shape_raw,
                                                 const std::vector<int64_t>& rhs_strides_raw,
                                                 F&& fn) {
        const size_t ndim                = out_shape.size();
        std::vector<int64_t> out_strides = tensor_compute_strides(out_shape);
        std::vector<int64_t> lhs_shape(ndim, 1);
        std::vector<int64_t> lhs_strides(ndim, 0);
        std::vector<int64_t> rhs_shape(ndim, 1);
        std::vector<int64_t> rhs_strides(ndim, 0);

        for (size_t i = 0; i < lhs_shape_raw.size(); ++i) {
            lhs_shape[i]   = lhs_shape_raw[i];
            lhs_strides[i] = lhs_strides_raw[i];
        }
        for (size_t i = 0; i < rhs_shape_raw.size(); ++i) {
            rhs_shape[i]   = rhs_shape_raw[i];
            rhs_strides[i] = rhs_strides_raw[i];
        }

        const int64_t numel = tensor_numel(out_shape);
        for (int64_t flat = 0; flat < numel; ++flat) {
            int64_t remaining  = flat;
            int64_t lhs_offset = 0;
            int64_t rhs_offset = 0;
            for (size_t i = ndim; i-- > 0;) {
                int64_t coord = remaining / out_strides[i];
                remaining %= out_strides[i];
                if (lhs_shape[i] != 1) {
                    lhs_offset += coord * lhs_strides[i];
                }
                if (rhs_shape[i] != 1) {
                    rhs_offset += coord * rhs_strides[i];
                }
            }
            fn(flat, lhs_offset, rhs_offset);
        }
    }

    template <typename T>
    inline Tensor<T>& Tensor<T>::masked_fill_(const Tensor<uint8_t>& mask, const T& value) {
        if (empty()) {
            return *this;
        }
        tensor_broadcast_shape(shape_, mask.shape());
        const std::vector<int64_t> data_strides = tensor_compute_strides(shape_);
        const std::vector<int64_t> mask_strides = tensor_compute_strides(mask.shape());
        const uint8_t* mask_data                = mask.data();
        tensor_for_each_broadcast_offset(shape_,
                                         shape_,
                                         data_strides,
                                         mask.shape(),
                                         mask_strides,
                                         [&](int64_t, int64_t data_offset, int64_t mask_offset) {
                                             if (mask_data[mask_offset] != 0) {
                                                 data_[static_cast<size_t>(data_offset)] = value;
                                             }
                                         });
        return *this;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<uint8_t> operator<(const Tensor<T>& lhs, Scalar rhs) {
        Tensor<uint8_t> result(lhs.shape());
        const T value = static_cast<T>(rhs);
        for (int64_t i = 0; i < lhs.numel(); ++i) {
            result[i] = lhs[i] < value ? 1 : 0;
        }
        return result;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<uint8_t> operator<(Scalar lhs, const Tensor<T>& rhs) {
        Tensor<uint8_t> result(rhs.shape());
        const T value = static_cast<T>(lhs);
        for (int64_t i = 0; i < rhs.numel(); ++i) {
            result[i] = value < rhs[i] ? 1 : 0;
        }
        return result;
    }

    template <typename T>
    inline Tensor<uint8_t> operator<(const Tensor<T>& lhs, const Tensor<T>& rhs) {
        const std::vector<int64_t> out_shape = tensor_broadcast_shape(lhs.shape(), rhs.shape());
        Tensor<uint8_t> result(out_shape);
        const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
        const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
        const T* lhs_data                      = lhs.data();
        const T* rhs_data                      = rhs.data();
        tensor_for_each_broadcast_offset(out_shape,
                                         lhs.shape(),
                                         lhs_strides,
                                         rhs.shape(),
                                         rhs_strides,
                                         [&](int64_t flat, int64_t lhs_offset, int64_t rhs_offset) {
                                             result[flat] = lhs_data[lhs_offset] < rhs_data[rhs_offset] ? 1 : 0;
                                         });
        return result;
    }

    template <typename T>
    inline Tensor<T>& operator+=(Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (lhs.shape() == rhs.shape()) {
            for (int64_t i = 0; i < lhs.numel(); ++i) {
                lhs[i] += rhs[i];
            }
            return lhs;
        }
        tensor_broadcast_shape(lhs.shape(), rhs.shape());
        const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
        const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
        const T* rhs_data                      = rhs.data();
        tensor_for_each_broadcast_offset(lhs.shape(),
                                         lhs.shape(),
                                         lhs_strides,
                                         rhs.shape(),
                                         rhs_strides,
                                         [&](int64_t, int64_t lhs_offset, int64_t rhs_offset) {
                                             lhs[static_cast<int64_t>(lhs_offset)] += rhs_data[rhs_offset];
                                         });
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T>& operator+=(Tensor<T>& lhs, Scalar rhs) {
        const T value = static_cast<T>(rhs);
        for (int64_t i = 0; i < lhs.numel(); ++i) {
            lhs[i] += value;
        }
        return lhs;
    }

    template <typename T>
    inline Tensor<T>& operator-=(Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (lhs.shape() == rhs.shape()) {
            for (int64_t i = 0; i < lhs.numel(); ++i) {
                lhs[i] -= rhs[i];
            }
            return lhs;
        }
        tensor_broadcast_shape(lhs.shape(), rhs.shape());
        const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
        const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
        const T* rhs_data                      = rhs.data();
        tensor_for_each_broadcast_offset(lhs.shape(),
                                         lhs.shape(),
                                         lhs_strides,
                                         rhs.shape(),
                                         rhs_strides,
                                         [&](int64_t, int64_t lhs_offset, int64_t rhs_offset) {
                                             lhs[static_cast<int64_t>(lhs_offset)] -= rhs_data[rhs_offset];
                                         });
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T>& operator-=(Tensor<T>& lhs, Scalar rhs) {
        const T value = static_cast<T>(rhs);
        for (int64_t i = 0; i < lhs.numel(); ++i) {
            lhs[i] -= value;
        }
        return lhs;
    }

    template <typename T>
    inline Tensor<T>& operator*=(Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (lhs.shape() == rhs.shape()) {
            for (int64_t i = 0; i < lhs.numel(); ++i) {
                lhs[i] *= rhs[i];
            }
            return lhs;
        }
        tensor_broadcast_shape(lhs.shape(), rhs.shape());
        const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
        const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
        const T* rhs_data                      = rhs.data();
        tensor_for_each_broadcast_offset(lhs.shape(),
                                         lhs.shape(),
                                         lhs_strides,
                                         rhs.shape(),
                                         rhs_strides,
                                         [&](int64_t, int64_t lhs_offset, int64_t rhs_offset) {
                                             lhs[static_cast<int64_t>(lhs_offset)] *= rhs_data[rhs_offset];
                                         });
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T>& operator*=(Tensor<T>& lhs, Scalar rhs) {
        const T value = static_cast<T>(rhs);
        for (int64_t i = 0; i < lhs.numel(); ++i) {
            lhs[i] *= value;
        }
        return lhs;
    }

    template <typename T>
    inline Tensor<T>& operator/=(Tensor<T>& lhs, const Tensor<T>& rhs) {
        if (lhs.shape() == rhs.shape()) {
            for (int64_t i = 0; i < lhs.numel(); ++i) {
                lhs[i] /= rhs[i];
            }
            return lhs;
        }
        tensor_broadcast_shape(lhs.shape(), rhs.shape());
        const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
        const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
        const T* rhs_data                      = rhs.data();
        tensor_for_each_broadcast_offset(lhs.shape(),
                                         lhs.shape(),
                                         lhs_strides,
                                         rhs.shape(),
                                         rhs_strides,
                                         [&](int64_t, int64_t lhs_offset, int64_t rhs_offset) {
                                             lhs[static_cast<int64_t>(lhs_offset)] /= rhs_data[rhs_offset];
                                         });
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T>& operator/=(Tensor<T>& lhs, Scalar rhs) {
        const T value = static_cast<T>(rhs);
        for (int64_t i = 0; i < lhs.numel(); ++i) {
            lhs[i] /= value;
        }
        return lhs;
    }

    template <typename T>
    inline Tensor<T> operator+(Tensor<T> lhs, const Tensor<T>& rhs) {
        if (lhs.shape() != rhs.shape()) {
            const std::vector<int64_t> out_shape = tensor_broadcast_shape(lhs.shape(), rhs.shape());
            Tensor<T> result(out_shape);
            const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
            const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
            const T* lhs_data                      = lhs.data();
            const T* rhs_data                      = rhs.data();
            tensor_for_each_broadcast_offset(out_shape,
                                             lhs.shape(),
                                             lhs_strides,
                                             rhs.shape(),
                                             rhs_strides,
                                             [&](int64_t flat, int64_t lhs_offset, int64_t rhs_offset) {
                                                 result[flat] = lhs_data[lhs_offset] + rhs_data[rhs_offset];
                                             });
            return result;
        }
        lhs += rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator+(Tensor<T> lhs, Scalar rhs) {
        lhs += rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator+(Scalar lhs, Tensor<T> rhs) {
        rhs += lhs;
        return rhs;
    }

    template <typename T>
    inline Tensor<T> operator-(Tensor<T> lhs, const Tensor<T>& rhs) {
        if (lhs.shape() != rhs.shape()) {
            const std::vector<int64_t> out_shape = tensor_broadcast_shape(lhs.shape(), rhs.shape());
            Tensor<T> result(out_shape);
            const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
            const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
            const T* lhs_data                      = lhs.data();
            const T* rhs_data                      = rhs.data();
            tensor_for_each_broadcast_offset(out_shape,
                                             lhs.shape(),
                                             lhs_strides,
                                             rhs.shape(),
                                             rhs_strides,
                                             [&](int64_t flat, int64_t lhs_offset, int64_t rhs_offset) {
                                                 result[flat] = lhs_data[lhs_offset] - rhs_data[rhs_offset];
                                             });
            return result;
        }
        lhs -= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator-(Tensor<T> lhs, Scalar rhs) {
        lhs -= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator-(Scalar lhs, const Tensor<T>& rhs) {
        Tensor<T> result = rhs;
        const T value    = static_cast<T>(lhs);
        for (int64_t i = 0; i < result.numel(); ++i) {
            result[i] = value - result[i];
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> operator*(Tensor<T> lhs, const Tensor<T>& rhs) {
        if (lhs.shape() != rhs.shape()) {
            const std::vector<int64_t> out_shape = tensor_broadcast_shape(lhs.shape(), rhs.shape());
            Tensor<T> result(out_shape);
            const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
            const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
            const T* lhs_data                      = lhs.data();
            const T* rhs_data                      = rhs.data();
            tensor_for_each_broadcast_offset(out_shape,
                                             lhs.shape(),
                                             lhs_strides,
                                             rhs.shape(),
                                             rhs_strides,
                                             [&](int64_t flat, int64_t lhs_offset, int64_t rhs_offset) {
                                                 result[flat] = lhs_data[lhs_offset] * rhs_data[rhs_offset];
                                             });
            return result;
        }
        lhs *= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator*(Tensor<T> lhs, Scalar rhs) {
        lhs *= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator*(Scalar lhs, Tensor<T> rhs) {
        rhs *= lhs;
        return rhs;
    }

    template <typename T>
    inline Tensor<T> operator/(Tensor<T> lhs, const Tensor<T>& rhs) {
        if (lhs.shape() != rhs.shape()) {
            const std::vector<int64_t> out_shape = tensor_broadcast_shape(lhs.shape(), rhs.shape());
            Tensor<T> result(out_shape);
            const std::vector<int64_t> lhs_strides = tensor_compute_strides(lhs.shape());
            const std::vector<int64_t> rhs_strides = tensor_compute_strides(rhs.shape());
            const T* lhs_data                      = lhs.data();
            const T* rhs_data                      = rhs.data();
            tensor_for_each_broadcast_offset(out_shape,
                                             lhs.shape(),
                                             lhs_strides,
                                             rhs.shape(),
                                             rhs_strides,
                                             [&](int64_t flat, int64_t lhs_offset, int64_t rhs_offset) {
                                                 result[flat] = lhs_data[lhs_offset] / rhs_data[rhs_offset];
                                             });
            return result;
        }
        lhs /= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator/(Tensor<T> lhs, Scalar rhs) {
        lhs /= rhs;
        return lhs;
    }

    template <typename T, typename Scalar, typename = std::enable_if_t<std::is_arithmetic<Scalar>::value>>
    inline Tensor<T> operator/(Scalar lhs, const Tensor<T>& rhs) {
        Tensor<T> result = rhs;
        const T value    = static_cast<T>(lhs);
        for (int64_t i = 0; i < result.numel(); ++i) {
            result[i] = value / result[i];
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> operator-(const Tensor<T>& tensor) {
        Tensor<T> result = tensor;
        for (int64_t i = 0; i < result.numel(); ++i) {
            result[i] = -result[i];
        }
        return result;
    }

    template <typename T>
    inline Tensor<T> zeros(std::vector<int64_t> shape) {
        return Tensor<T>::zeros(std::move(shape));
    }

    template <typename T>
    inline Tensor<T> full(std::vector<int64_t> shape, const T& value) {
        return Tensor<T>::full(std::move(shape), value);
    }

    template <typename T>
    inline Tensor<T> randn(std::vector<int64_t> shape, const std::shared_ptr<RNG>& rng) {
        return Tensor<T>::randn(std::move(shape), rng);
    }

    template <typename T>
    inline Tensor<T> randn_like(const Tensor<T>& tensor, const std::shared_ptr<RNG>& rng) {
        return Tensor<T>::randn(tensor.shape(), rng);
    }

    template <typename T>
    inline std::vector<T> tensor_to_vector(const Tensor<T>& tensor) {
        return tensor.values();
    }

    namespace ops {
        enum class InterpolateMode {
            Nearest,
            NearestMax,
            NearestMin,
            NearestAvg,
        };

        inline int64_t normalize_slice_bound(int64_t index, int64_t dim_size) {
            if (index < 0) {
                index += dim_size;
            }
            return index;
        }

        template <typename T>
        inline std::pair<int64_t, int64_t> resolve_slice_bounds(const Tensor<T>& input,
                                                                size_t dim,
                                                                int64_t start,
                                                                int64_t end) {
            if (dim >= static_cast<size_t>(input.dim())) {
                tensor_throw_invalid_argument("Tensor slice dimension out of range: dim=" +
                                              std::to_string(dim) + ", rank=" +
                                              std::to_string(input.dim()) + ", input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }

            int64_t dim_size = input.shape()[dim];
            start            = normalize_slice_bound(start, dim_size);
            end              = normalize_slice_bound(end, dim_size);

            if (start < 0 || start > dim_size || end < 0 || end > dim_size || start > end) {
                tensor_throw_invalid_argument("Tensor slice bounds out of range: dim=" +
                                              std::to_string(dim) + ", start=" +
                                              std::to_string(start) + ", end=" +
                                              std::to_string(end) + ", input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }

            return {start, end};
        }

        template <typename T>
        inline Tensor<T> exp(const Tensor<T>& input) {
            Tensor<T> output(input.shape());
            for (int64_t i = 0; i < input.numel(); ++i) {
                output[i] = static_cast<T>(std::exp(static_cast<double>(input[i])));
            }
            return output;
        }

        template <typename T>
        inline Tensor<T> clamp(const Tensor<T>& input, const T& min_value, const T& max_value) {
            if (min_value > max_value) {
                tensor_throw_invalid_argument("Tensor clamp requires min_value <= max_value");
            }
            Tensor<T> output(input.shape());
            for (int64_t i = 0; i < input.numel(); ++i) {
                output[i] = std::clamp(input[i], min_value, max_value);
            }
            return output;
        }

        template <typename T>
        inline Tensor<T> round(const Tensor<T>& input) {
            Tensor<T> output(input.shape());
            for (int64_t i = 0; i < input.numel(); ++i) {
                output[i] = static_cast<T>(std::round(static_cast<double>(input[i])));
            }
            return output;
        }

        template <typename T>
        inline Tensor<T> slice(const Tensor<T>& input,
                               size_t dim,
                               int64_t start,
                               int64_t end) {
            auto [resolved_start, resolved_end] = resolve_slice_bounds(input, dim, start, end);
            std::vector<int64_t> out_shape      = input.shape();
            out_shape[dim]                      = resolved_end - resolved_start;

            Tensor<T> output(out_shape);
            if (output.numel() == 0) {
                return output;
            }

            int64_t inner = 1;
            for (size_t i = 0; i < dim; ++i) {
                inner *= input.shape()[i];
            }

            int64_t outer = 1;
            for (size_t i = dim + 1; i < static_cast<size_t>(input.dim()); ++i) {
                outer *= input.shape()[i];
            }

            int64_t src_chunk  = (resolved_end - resolved_start) * inner;
            int64_t src_stride = input.shape()[dim] * inner;
            for (int64_t i = 0; i < outer; ++i) {
                const int64_t src_offset = i * src_stride + resolved_start * inner;
                const int64_t dst_offset = i * src_chunk;
                std::copy_n(input.data() + src_offset, src_chunk, output.data() + dst_offset);
            }

            return output;
        }

        template <typename T>
        inline Tensor<T> narrow(const Tensor<T>& input,
                                size_t dim,
                                int64_t start,
                                int64_t length) {
            if (length < 0) {
                tensor_throw_invalid_argument("Tensor narrow requires non-negative length: length=" +
                                              std::to_string(length) + ", input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }
            return slice(input, dim, start, start + length);
        }

        template <typename T>
        inline void slice_assign(Tensor<T>* dst,
                                 size_t dim,
                                 int64_t start,
                                 int64_t end,
                                 const Tensor<T>& src) {
            if (dst == nullptr) {
                tensor_throw_invalid_argument("Tensor slice_assign requires non-null dst");
            }

            auto [resolved_start, resolved_end] = resolve_slice_bounds(*dst, dim, start, end);
            if (src.dim() != dst->dim()) {
                tensor_throw_invalid_argument("Tensor slice_assign requires matching rank: dst_shape=" +
                                              tensor_shape_to_string(dst->shape()) + ", src_shape=" +
                                              tensor_shape_to_string(src.shape()));
            }

            std::vector<int64_t> expected_shape = dst->shape();
            expected_shape[dim]                 = resolved_end - resolved_start;
            if (src.shape() != expected_shape) {
                tensor_throw_invalid_argument("Tensor slice_assign requires matching source shape: dst_shape=" +
                                              tensor_shape_to_string(dst->shape()) + ", src_shape=" +
                                              tensor_shape_to_string(src.shape()) + ", expected_src_shape=" +
                                              tensor_shape_to_string(expected_shape));
            }

            if (src.numel() == 0) {
                return;
            }

            int64_t inner = 1;
            for (size_t i = 0; i < dim; ++i) {
                inner *= dst->shape()[i];
            }

            int64_t outer = 1;
            for (size_t i = dim + 1; i < static_cast<size_t>(dst->dim()); ++i) {
                outer *= dst->shape()[i];
            }

            int64_t dst_chunk  = (resolved_end - resolved_start) * inner;
            int64_t dst_stride = dst->shape()[dim] * inner;
            for (int64_t i = 0; i < outer; ++i) {
                const int64_t dst_offset = i * dst_stride + resolved_start * inner;
                const int64_t src_offset = i * dst_chunk;
                std::copy_n(src.data() + src_offset, dst_chunk, dst->data() + dst_offset);
            }
        }

        template <typename T>
        inline void fill_slice(Tensor<T>* dst,
                               size_t dim,
                               int64_t start,
                               int64_t end,
                               const T& value) {
            if (dst == nullptr) {
                tensor_throw_invalid_argument("Tensor fill_slice requires non-null dst");
            }

            auto [resolved_start, resolved_end] = resolve_slice_bounds(*dst, dim, start, end);
            int64_t inner                       = 1;
            for (size_t i = 0; i < dim; ++i) {
                inner *= dst->shape()[i];
            }

            int64_t outer = 1;
            for (size_t i = dim + 1; i < static_cast<size_t>(dst->dim()); ++i) {
                outer *= dst->shape()[i];
            }

            int64_t chunk  = (resolved_end - resolved_start) * inner;
            int64_t stride = dst->shape()[dim] * inner;
            for (int64_t i = 0; i < outer; ++i) {
                const int64_t offset = i * stride + resolved_start * inner;
                std::fill_n(dst->data() + offset, chunk, value);
            }
        }

        template <typename T>
        inline Tensor<T> interpolate(const Tensor<T>& input,
                                     std::vector<int64_t> output_shape,
                                     InterpolateMode mode = InterpolateMode::Nearest,
                                     bool align_corners   = false) {
            const bool is_nearest_like_mode = (mode == InterpolateMode::Nearest ||
                                               mode == InterpolateMode::NearestMax ||
                                               mode == InterpolateMode::NearestMin ||
                                               mode == InterpolateMode::NearestAvg);
            if (!is_nearest_like_mode) {
                tensor_throw_invalid_argument("Only nearest-like interpolate modes are implemented, got mode=" +
                                              std::to_string(static_cast<int>(mode)));
            }
            if (align_corners) {
                tensor_throw_invalid_argument("align_corners is not supported for nearest-like interpolate: input_shape=" +
                                              tensor_shape_to_string(input.shape()) + ", output_shape=" +
                                              tensor_shape_to_string(output_shape));
            }
            if (input.shape() == output_shape) {
                return input;
            }
            if (input.dim() != static_cast<int64_t>(output_shape.size())) {
                tensor_throw_invalid_argument("Tensor interpolate requires matching rank: input_dim=" +
                                              std::to_string(input.dim()) + ", output_dim=" +
                                              std::to_string(output_shape.size()) + ", input_shape=" +
                                              tensor_shape_to_string(input.shape()) + ", output_shape=" +
                                              tensor_shape_to_string(output_shape));
            }
            for (size_t i = 0; i < output_shape.size(); ++i) {
                if (output_shape[i] <= 0) {
                    tensor_throw_invalid_argument("Tensor interpolate output shape must be positive: input_shape=" +
                                                  tensor_shape_to_string(input.shape()) + ", output_shape=" +
                                                  tensor_shape_to_string(output_shape));
                }
                if (input.shape()[i] <= 0) {
                    tensor_throw_invalid_argument("Tensor interpolate input shape must be positive: input_shape=" +
                                                  tensor_shape_to_string(input.shape()) + ", output_shape=" +
                                                  tensor_shape_to_string(output_shape));
                }
            }

            bool has_downsampling = false;
            for (int64_t i = 0; i < input.dim(); ++i) {
                if (input.shape()[i] > output_shape[i]) {
                    has_downsampling = true;
                    break;
                }
            }

            Tensor<T> output(std::move(output_shape));
            if (mode == InterpolateMode::Nearest || !has_downsampling) {
                for (int64_t flat = 0; flat < output.numel(); ++flat) {
                    std::vector<int64_t> output_coord = tensor_unravel_index(flat, output.shape());
                    std::vector<int64_t> input_coord(static_cast<size_t>(input.dim()), 0);
                    for (size_t i = 0; i < static_cast<size_t>(input.dim()); ++i) {
                        input_coord[i] = output_coord[i] * input.shape()[i] / output.shape()[i];
                    }
                    output[flat] = input.index(input_coord);
                }

                return output;
            }

            auto init_reduction = [&]() -> T {
                switch (mode) {
                    case InterpolateMode::NearestMax:
                        return std::numeric_limits<T>::lowest();
                    case InterpolateMode::NearestMin:
                        return std::numeric_limits<T>::max();
                    case InterpolateMode::NearestAvg:
                        return T(0);
                    case InterpolateMode::Nearest:
                        return T(0);
                }

                tensor_throw_invalid_argument("Unsupported interpolate mode: mode=" +
                                              std::to_string(static_cast<int>(mode)));
            };

            auto reduce_value = [&](T& acc, const T& sample) {
                switch (mode) {
                    case InterpolateMode::NearestMax:
                        acc = std::max(acc, sample);
                        break;
                    case InterpolateMode::NearestMin:
                        acc = std::min(acc, sample);
                        break;
                    case InterpolateMode::NearestAvg:
                        acc += sample;
                        break;
                    case InterpolateMode::Nearest:
                        break;
                }
            };

            // Reduction modes only differ from nearest mode when downsampling.
            for (int64_t flat_out = 0; flat_out < output.numel(); ++flat_out) {
                std::vector<int64_t> output_coord = tensor_unravel_index(flat_out, output.shape());

                std::vector<int64_t> input_start(output.dim(), 0);
                std::vector<int64_t> input_end(output.dim(), 0);

                for (size_t i = 0; i < static_cast<size_t>(output.dim()); ++i) {
                    const int64_t input_dim  = input.shape()[i];
                    const int64_t output_dim = output.shape()[i];

                    input_start[i] = std::max(int64_t(0), static_cast<int64_t>(output_coord[i] * input_dim / output_dim));
                    input_end[i]   = std::min(input_dim, ((output_coord[i] + 1) * input_dim + output_dim - 1) / output_dim);
                }

                T value                               = init_reduction();
                bool done_window                      = false;
                std::vector<int64_t> current_in_coord = input_start;

                while (!done_window) {
                    reduce_value(value, input.index(current_in_coord));

                    for (int d = static_cast<int>(output.dim()) - 1; d >= 0; --d) {
                        if (++current_in_coord[d] < input_end[d]) {
                            break;
                        }
                        current_in_coord[d] = input_start[d];
                        if (d == 0) {
                            done_window = true;
                        }
                    }
                }

                if (mode == InterpolateMode::NearestAvg) {
                    int64_t window_size = 1;
                    for (size_t i = 0; i < static_cast<size_t>(output.dim()); ++i) {
                        window_size *= (input_end[i] - input_start[i]);
                    }
                    value /= static_cast<T>(window_size);
                }

                output[flat_out] = value;
            }

            return output;
        }

        template <typename T>
        inline Tensor<T> interpolate(const Tensor<T>& input,
                                     const std::optional<std::vector<int64_t>>& size,
                                     const std::optional<std::vector<double>>& scale_factor,
                                     InterpolateMode mode = InterpolateMode::Nearest,
                                     bool align_corners   = false) {
            const bool is_nearest_like_mode = (mode == InterpolateMode::Nearest ||
                                               mode == InterpolateMode::NearestMax ||
                                               mode == InterpolateMode::NearestMin ||
                                               mode == InterpolateMode::NearestAvg);
            if (!is_nearest_like_mode) {
                tensor_throw_invalid_argument("Only nearest-like interpolate modes are implemented, got mode=" +
                                              std::to_string(static_cast<int>(mode)));
            }
            if (align_corners) {
                tensor_throw_invalid_argument("align_corners is not supported for nearest-like interpolate: input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }
            if (size.has_value() == scale_factor.has_value()) {
                tensor_throw_invalid_argument("Tensor interpolate requires exactly one of size or scale_factor: input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }

            std::vector<int64_t> output_shape = input.shape();
            if (size.has_value()) {
                if (size->empty() || size->size() > output_shape.size()) {
                    tensor_throw_invalid_argument("Tensor interpolate size must target low dimensions: input_shape=" +
                                                  tensor_shape_to_string(input.shape()) + ", size_rank=" +
                                                  std::to_string(size->size()));
                }
                for (size_t i = 0; i < size->size(); ++i) {
                    if ((*size)[i] <= 0) {
                        tensor_throw_invalid_argument("Tensor interpolate size must be positive: input_shape=" +
                                                      tensor_shape_to_string(input.shape()) + ", size=" +
                                                      tensor_shape_to_string(*size));
                    }
                    output_shape[i] = (*size)[i];
                }
            } else {
                if (scale_factor->empty() || scale_factor->size() > output_shape.size()) {
                    tensor_throw_invalid_argument("Tensor interpolate scale_factor must target low dimensions: input_shape=" +
                                                  tensor_shape_to_string(input.shape()) + ", scale_factor_rank=" +
                                                  std::to_string(scale_factor->size()));
                }
                for (size_t i = 0; i < scale_factor->size(); ++i) {
                    if ((*scale_factor)[i] <= 0.0) {
                        tensor_throw_invalid_argument("Tensor interpolate scale_factor must be positive: input_shape=" +
                                                      tensor_shape_to_string(input.shape()));
                    }
                    output_shape[i] = static_cast<int64_t>(
                        std::floor(static_cast<double>(output_shape[i]) * (*scale_factor)[i]));
                    if (output_shape[i] <= 0) {
                        tensor_throw_invalid_argument("Tensor interpolate output shape must be positive: input_shape=" +
                                                      tensor_shape_to_string(input.shape()) + ", output_shape=" +
                                                      tensor_shape_to_string(output_shape));
                    }
                }
            }

            return interpolate(input, std::move(output_shape), mode, align_corners);
        }

        template <typename T>
        inline Tensor<T> interpolate(const Tensor<T>& input,
                                     const std::optional<std::vector<int64_t>>& size,
                                     double scale_factor,
                                     InterpolateMode mode = InterpolateMode::Nearest,
                                     bool align_corners   = false) {
            return interpolate(input,
                               size,
                               std::vector<double>(size.has_value() ? size->size() : input.dim(), scale_factor),
                               mode,
                               align_corners);
        }

        template <typename T>
        inline Tensor<T> max_pool_2d(const Tensor<T>& input,
                                     std::vector<int64_t> kernel_size,
                                     std::vector<int64_t> stride,
                                     std::vector<int64_t> padding) {
            if (input.dim() < 2) {
                tensor_throw_invalid_argument("Tensor max_pool_2d requires input_dim >= 2: input_dim=" +
                                              std::to_string(input.dim()) + ", input_shape=" +
                                              tensor_shape_to_string(input.shape()));
            }
            if (kernel_size.size() != 2 || stride.size() != 2 || padding.size() != 2) {
                tensor_throw_invalid_argument("Tensor max_pool_2d requires kernel_size, stride, and padding to have length 2");
            }
            for (size_t i = 0; i < 2; ++i) {
                if (kernel_size[i] <= 0) {
                    tensor_throw_invalid_argument("Tensor max_pool_2d kernel_size must be positive: kernel_size=" +
                                                  tensor_shape_to_string(kernel_size));
                }
                if (stride[i] <= 0) {
                    tensor_throw_invalid_argument("Tensor max_pool_2d stride must be positive: stride=" +
                                                  tensor_shape_to_string(stride));
                }
                if (padding[i] < 0) {
                    tensor_throw_invalid_argument("Tensor max_pool_2d padding must be non-negative: padding=" +
                                                  tensor_shape_to_string(padding));
                }
            }

            const int64_t in_height = input.shape()[0];
            const int64_t in_width  = input.shape()[1];

            const int64_t out_height = (in_height + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
            const int64_t out_width  = (in_width + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;

            if (out_height <= 0 || out_width <= 0) {
                tensor_throw_invalid_argument("max_pool_2d results in invalid output dimensions: " +
                                              std::to_string(out_height) + "x" + std::to_string(out_width));
            }

            std::vector<int64_t> output_shape = input.shape();
            output_shape[0]                   = out_height;
            output_shape[1]                   = out_width;

            Tensor<T> output(std::move(output_shape));

            for (int64_t flat_out = 0; flat_out < output.numel(); ++flat_out) {
                std::vector<int64_t> output_coord = tensor_unravel_index(flat_out, output.shape());
                std::vector<int64_t> input_coord  = output_coord;

                const int64_t oh = output_coord[0];
                const int64_t ow = output_coord[1];

                T max_val            = std::numeric_limits<T>::lowest();
                bool has_valid_input = false;

                for (int64_t kh = 0; kh < kernel_size[0]; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size[1]; ++kw) {
                        const int64_t ih = oh * stride[0] + kh - padding[0];
                        const int64_t iw = ow * stride[1] + kw - padding[1];

                        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                            input_coord[0]  = ih;
                            input_coord[1]  = iw;
                            max_val         = std::max(max_val, input.index(input_coord));
                            has_valid_input = true;
                        }
                    }
                }

                output[flat_out] = has_valid_input ? max_val : T(0);
            }
            return output;
        }

        template <typename T>
        inline Tensor<T> concat(const Tensor<T>& lhs, const Tensor<T>& rhs, size_t dim) {
            if (lhs.dim() != rhs.dim()) {
                tensor_throw_invalid_argument("Tensor concat requires same rank: lhs_dim=" +
                                              std::to_string(lhs.dim()) + ", rhs_dim=" +
                                              std::to_string(rhs.dim()) + ", lhs_shape=" +
                                              tensor_shape_to_string(lhs.shape()) + ", rhs_shape=" +
                                              tensor_shape_to_string(rhs.shape()));
            }
            if (dim >= static_cast<size_t>(lhs.dim())) {
                tensor_throw_invalid_argument("Tensor concat dimension out of range: dim=" +
                                              std::to_string(dim) + ", rank=" +
                                              std::to_string(lhs.dim()) + ", lhs_shape=" +
                                              tensor_shape_to_string(lhs.shape()));
            }
            std::vector<int64_t> out_shape = lhs.shape();
            for (size_t i = 0; i < static_cast<size_t>(lhs.dim()); ++i) {
                if (i == dim) {
                    continue;
                }
                if (lhs.shape()[i] != rhs.shape()[i]) {
                    tensor_throw_invalid_argument("Tensor concat requires matching non-concat dimensions: dim=" +
                                                  std::to_string(dim) + ", lhs_shape=" +
                                                  tensor_shape_to_string(lhs.shape()) + ", rhs_shape=" +
                                                  tensor_shape_to_string(rhs.shape()));
                }
            }
            out_shape[dim] += rhs.shape()[dim];

            Tensor<T> out(out_shape);
            int64_t inner = 1;
            for (size_t i = 0; i < dim; ++i) {
                inner *= lhs.shape()[i];
            }

            int64_t outer = 1;
            for (size_t i = dim + 1; i < static_cast<size_t>(lhs.dim()); ++i) {
                outer *= lhs.shape()[i];
            }

            int64_t lhs_chunk = lhs.shape()[dim] * inner;
            int64_t rhs_chunk = rhs.shape()[dim] * inner;
            int64_t out_chunk = lhs_chunk + rhs_chunk;

            for (int64_t i = 0; i < outer; ++i) {
                int64_t lhs_offset = i * lhs_chunk;
                int64_t rhs_offset = i * rhs_chunk;
                int64_t out_offset = i * out_chunk;

                std::copy_n(lhs.data() + lhs_offset, lhs_chunk, out.data() + out_offset);
                std::copy_n(rhs.data() + rhs_offset, rhs_chunk, out.data() + out_offset + lhs_chunk);
            }
            return out;
        }

        template <typename T>
        inline std::vector<Tensor<T>> chunk(const Tensor<T>& tensor, int64_t chunks, size_t dim) {
            if (chunks <= 0) {
                tensor_throw_invalid_argument("Tensor chunk requires chunks > 0: chunks=" +
                                              std::to_string(chunks) + ", tensor_shape=" +
                                              tensor_shape_to_string(tensor.shape()));
            }
            if (dim >= static_cast<size_t>(tensor.dim())) {
                tensor_throw_invalid_argument("Tensor chunk dimension out of range: dim=" +
                                              std::to_string(dim) + ", rank=" +
                                              std::to_string(tensor.dim()) + ", tensor_shape=" +
                                              tensor_shape_to_string(tensor.shape()));
            }

            const int64_t dim_size = tensor.shape()[dim];
            if (dim_size == 0) {
                return {};
            }
            if (dim_size % chunks != 0) {
                tensor_throw_invalid_argument("Tensor chunk requires the dimension size to be divisible by chunks: dim=" +
                                              std::to_string(dim) + ", dim_size=" +
                                              std::to_string(dim_size) + ", chunks=" +
                                              std::to_string(chunks) + ", tensor_shape=" +
                                              tensor_shape_to_string(tensor.shape()));
            }

            const int64_t chunk_size = dim_size / chunks;
            int64_t inner            = 1;
            for (size_t i = 0; i < dim; ++i) {
                inner *= tensor.shape()[i];
            }

            int64_t outer = 1;
            for (size_t i = dim + 1; i < static_cast<size_t>(tensor.dim()); ++i) {
                outer *= tensor.shape()[i];
            }

            std::vector<Tensor<T>> parts;
            parts.reserve(static_cast<size_t>(chunks));

            for (int64_t start = 0; start < dim_size; start += chunk_size) {
                std::vector<int64_t> part_shape = tensor.shape();
                part_shape[dim]                 = chunk_size;
                Tensor<T> part(part_shape);

                const int64_t src_chunk = chunk_size * inner;
                const int64_t dst_chunk = src_chunk;
                for (int64_t i = 0; i < outer; ++i) {
                    const int64_t src_offset = (i * dim_size + start) * inner;
                    const int64_t dst_offset = i * dst_chunk;
                    std::copy_n(tensor.data() + src_offset, src_chunk, part.data() + dst_offset);
                }

                parts.push_back(std::move(part));
            }

            return parts;
        }

    }  // namespace ops

}  // namespace sd

#endif

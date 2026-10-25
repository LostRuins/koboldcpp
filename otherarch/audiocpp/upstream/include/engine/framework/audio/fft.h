#pragma once

#include <cstddef>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::audio {

using TensorShape = std::vector<size_t>;
using TensorStrideBytes = std::vector<std::ptrdiff_t>;

class RealFFTPlan {
public:
    explicit RealFFTPlan(size_t fft_size);
    ~RealFFTPlan();

    RealFFTPlan(const RealFFTPlan &);
    RealFFTPlan & operator=(const RealFFTPlan &);
    RealFFTPlan(RealFFTPlan &&) noexcept;
    RealFFTPlan & operator=(RealFFTPlan &&) noexcept;

    size_t fft_size() const;

    void forward(
        const TensorShape & input_shape,
        const TensorStrideBytes & input_strides,
        const TensorStrideBytes & output_strides,
        size_t axis,
        const float * data_in,
        std::complex<float> * data_out,
        float scale = 1.0f,
        size_t threads = 0) const;

    void inverse(
        const TensorShape & output_shape,
        const TensorStrideBytes & input_strides,
        const TensorStrideBytes & output_strides,
        size_t axis,
        const std::complex<float> * data_in,
        float * data_out,
        float scale = 1.0f,
        size_t threads = 0) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

std::shared_ptr<RealFFTPlan> get_real_fft_plan(size_t fft_size);

void real_fft_forward(
    const TensorShape & input_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const float * data_in,
    std::complex<float> * data_out,
    float scale = 1.0f,
    size_t threads = 0);

void real_fft_inverse(
    const TensorShape & output_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const std::complex<float> * data_in,
    float * data_out,
    float scale = 1.0f,
    size_t threads = 0);

}  // namespace engine::audio

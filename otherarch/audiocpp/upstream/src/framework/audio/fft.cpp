#include "engine/framework/audio/fft.h"

#define ENGINE_SPEECHFFT_CACHE_SIZE 16
#include "detail/speech_fft_internal.h"

#include <stdexcept>

namespace engine::audio {

namespace {

namespace internal_fft = audio_cpp_fft_internal;
using InternalRealPlan = internal_fft::detail::internal_fft_r<float>;

}  // namespace

struct RealFFTPlan::Impl {
    explicit Impl(size_t fft_size_)
        : fft_size(fft_size_),
          internal(internal_fft::detail::get_plan<InternalRealPlan>(fft_size_)) {}

    size_t fft_size = 0;
    std::shared_ptr<InternalRealPlan> internal;
};

RealFFTPlan::RealFFTPlan(size_t fft_size)
    : impl_(std::make_shared<Impl>(fft_size)) {
    if (fft_size == 0) {
        throw std::runtime_error("RealFFTPlan fft_size must be > 0");
    }
}

RealFFTPlan::~RealFFTPlan() = default;
RealFFTPlan::RealFFTPlan(const RealFFTPlan &) = default;
RealFFTPlan & RealFFTPlan::operator=(const RealFFTPlan &) = default;
RealFFTPlan::RealFFTPlan(RealFFTPlan &&) noexcept = default;
RealFFTPlan & RealFFTPlan::operator=(RealFFTPlan &&) noexcept = default;

size_t RealFFTPlan::fft_size() const {
    return impl_->fft_size;
}

void RealFFTPlan::forward(
    const TensorShape & input_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const float * data_in,
    std::complex<float> * data_out,
    float scale,
    size_t threads) const {
    if (axis >= input_shape.size()) {
        throw std::runtime_error("RealFFTPlan forward axis out of range");
    }
    if (input_shape[axis] != impl_->fft_size) {
        throw std::runtime_error("RealFFTPlan forward shape mismatch");
    }
    real_fft_forward(input_shape, input_strides, output_strides, axis, data_in, data_out, scale, threads);
}

void RealFFTPlan::inverse(
    const TensorShape & output_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const std::complex<float> * data_in,
    float * data_out,
    float scale,
    size_t threads) const {
    if (axis >= output_shape.size()) {
        throw std::runtime_error("RealFFTPlan inverse axis out of range");
    }
    if (output_shape[axis] != impl_->fft_size) {
        throw std::runtime_error("RealFFTPlan inverse shape mismatch");
    }
    real_fft_inverse(output_shape, input_strides, output_strides, axis, data_in, data_out, scale, threads);
}

std::shared_ptr<RealFFTPlan> get_real_fft_plan(size_t fft_size) {
    return std::make_shared<RealFFTPlan>(fft_size);
}

void real_fft_forward(
    const TensorShape & input_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const float * data_in,
    std::complex<float> * data_out,
    float scale,
    size_t threads) {
    internal_fft::r2c(
        input_shape,
        input_strides,
        output_strides,
        axis,
        internal_fft::FORWARD,
        data_in,
        data_out,
        scale,
        threads);
}

void real_fft_inverse(
    const TensorShape & output_shape,
    const TensorStrideBytes & input_strides,
    const TensorStrideBytes & output_strides,
    size_t axis,
    const std::complex<float> * data_in,
    float * data_out,
    float scale,
    size_t threads) {
    internal_fft::c2r(
        output_shape,
        input_strides,
        output_strides,
        axis,
        internal_fft::BACKWARD,
        data_in,
        data_out,
        scale,
        threads);
}

}  // namespace engine::audio

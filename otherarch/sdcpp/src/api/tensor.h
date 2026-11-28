#ifndef __SD_API_TENSOR_H__
#define __SD_API_TENSOR_H__

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>
#include "core/tensor.hpp"

#include "core/media.h"
#include "stable-diffusion.h"

namespace sd::api {
    size_t tensor_byte_size(sd_tensor_type_t type, const std::vector<int64_t>& shape);
}

struct sd_tensor_t {
    using Storage = std::variant<sd::Tensor<uint8_t>, sd::Tensor<int32_t>, sd::Tensor<int64_t>, sd::Tensor<float>, sd::Tensor<double>>;
    mutable std::atomic<size_t> references{1};
    sd_tensor_type_t type;
    std::vector<int64_t> shape;
    size_t bytes;

private:
    Storage storage_;

public:
    sd_tensor_t(sd_tensor_type_t type, std::vector<int64_t> shape);
    sd_tensor_t(sd_tensor_type_t type, std::vector<int64_t> shape, const void* data, size_t size);

    template <typename T>
    explicit sd_tensor_t(sd::Tensor<T> tensor)
        : type(std::is_same_v<T, uint8_t> ? SD_TENSOR_U8 : std::is_same_v<T, int32_t> ? SD_TENSOR_I32
                                                       : std::is_same_v<T, int64_t>   ? SD_TENSOR_I64
                                                       : std::is_same_v<T, float>     ? SD_TENSOR_F32
                                                                                      : SD_TENSOR_F64),
          shape(tensor.shape()),
          bytes(sd::api::tensor_byte_size(type, shape)),
          storage_(std::move(tensor)) {
        static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> || std::is_same_v<T, float> || std::is_same_v<T, double>, "unsupported tensor type");
        if (std::get<sd::Tensor<T>>(storage_).numel() != static_cast<int64_t>(bytes / sizeof(T)))
            throw std::invalid_argument("tensor shape and storage size do not match");
    }

    const void* data() const {
        return bytes ? std::visit([](const auto& tensor) -> const void* { return tensor.data(); }, storage_) : nullptr;
    }
    void* mutable_data() {
        return bytes ? std::visit([](auto& tensor) -> void* { return tensor.data(); }, storage_) : nullptr;
    }
};

namespace sd::api {
    class TensorRef {
        sd_tensor_t* tensor_ = nullptr;

    public:
        TensorRef() = default;
        explicit TensorRef(const sd_tensor_t* tensor)
            : tensor_(sd_tensor_retain(tensor)) {}
        static TensorRef adopt(sd_tensor_t* tensor) {
            TensorRef result;
            result.tensor_ = tensor;
            return result;
        }
        TensorRef(const TensorRef& other)
            : TensorRef(other.tensor_) {}
        TensorRef(TensorRef&& other) noexcept
            : tensor_(other.tensor_) { other.tensor_ = nullptr; }
        TensorRef& operator=(TensorRef other) noexcept {
            std::swap(tensor_, other.tensor_);
            return *this;
        }
        ~TensorRef() { sd_tensor_free(tensor_); }
        sd_tensor_t* get() const { return tensor_; }
        const sd_tensor_t* operator->() const { return tensor_; }
        explicit operator bool() const { return tensor_ != nullptr; }
    };

    size_t tensor_byte_size(sd_tensor_type_t type, const std::vector<int64_t>& shape);
    TensorRef own_tensor(sd_tensor_type_t type, std::vector<int64_t> shape, void* data);
    TensorRef allocate_tensor(sd_tensor_type_t type, std::vector<int64_t> shape);
    sd::Image image_view(const sd_tensor_t& tensor);
    sd::Audio audio_view(const sd_audio_t& audio);
    std::vector<sd::Image> video_frames(const sd_tensor_t& tensor);
    void preview_images(const std::function<void(int, int, const sd_tensor_t* const*, bool, void*)>& callback, int step, int count, const sd::Image* images, bool noisy, void* data);
}

#endif  // __SD_API_TENSOR_H__

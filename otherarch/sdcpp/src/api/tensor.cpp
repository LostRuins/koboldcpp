#include "tensor.h"
#include "api.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

static size_t element_size(sd_tensor_type_t type) {
    switch (type) {
        case SD_TENSOR_U8:
            return sizeof(uint8_t);
        case SD_TENSOR_I32:
            return sizeof(int32_t);
        case SD_TENSOR_I64:
            return sizeof(int64_t);
        case SD_TENSOR_F32:
            return sizeof(float);
        case SD_TENSOR_F64:
            return sizeof(double);
        default:
            throw std::invalid_argument("invalid tensor type");
    }
}

size_t sd::api::tensor_byte_size(sd_tensor_type_t type, const std::vector<int64_t>& shape) {
    if (shape.size() > 8)
        throw std::invalid_argument("tensor rank must not exceed 8");
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0 || (count && uint64_t(dim) > SIZE_MAX / count))
            throw std::invalid_argument("negative or overflowing tensor dimensions");
        count *= static_cast<size_t>(dim);
    }
    const size_t size = element_size(type);
    if (count > uint64_t(INT64_MAX) || count > SIZE_MAX / size)
        throw std::invalid_argument("tensor byte size overflow");
    return count * size;
}

static sd_tensor_t::Storage allocate_storage(sd_tensor_type_t type, std::vector<int64_t> shape, size_t bytes) {
    if (bytes && shape.empty())
        shape.push_back(1);
    if (!bytes)
        shape.clear();
    switch (type) {
        case SD_TENSOR_U8:
            return sd::Tensor<uint8_t>(std::move(shape));
        case SD_TENSOR_I32:
            return sd::Tensor<int32_t>(std::move(shape));
        case SD_TENSOR_I64:
            return sd::Tensor<int64_t>(std::move(shape));
        case SD_TENSOR_F32:
            return sd::Tensor<float>(std::move(shape));
        case SD_TENSOR_F64:
            return sd::Tensor<double>(std::move(shape));
        default:
            throw std::invalid_argument("invalid tensor type");
    }
}

sd_tensor_t::sd_tensor_t(sd_tensor_type_t type, std::vector<int64_t> shape)
    : type(type), shape(std::move(shape)), bytes(sd::api::tensor_byte_size(type, this->shape)), storage_(allocate_storage(type, this->shape, bytes)) {}

static std::vector<int64_t> validated_shape(sd_tensor_type_t type, std::vector<int64_t> shape, const void* data, size_t size) {
    if (size != sd::api::tensor_byte_size(type, shape) || (size && !data))
        throw std::invalid_argument("tensor data size does not match type and shape");
    return shape;
}

sd_tensor_t::sd_tensor_t(sd_tensor_type_t type, std::vector<int64_t> shape, const void* data, size_t size)
    : sd_tensor_t(type, validated_shape(type, std::move(shape), data, size)) {
    if (bytes)
        memcpy(mutable_data(), data, bytes);
}

sd::api::TensorRef sd::api::allocate_tensor(sd_tensor_type_t type, std::vector<int64_t> shape) {
    return TensorRef::adopt(new sd_tensor_t(type, std::move(shape)));
}

sd::api::TensorRef sd::api::own_tensor(sd_tensor_type_t type, std::vector<int64_t> shape, void* data) {
    std::unique_ptr<void, decltype(&std::free)> owner(data, std::free);
    const size_t bytes = tensor_byte_size(type, shape);
    return TensorRef::adopt(new sd_tensor_t(type, std::move(shape), data, bytes));
}

sd::Image sd::api::image_view(const sd_tensor_t& tensor) {
    const auto& s = tensor.shape;
    if (tensor.type != SD_TENSOR_U8 || s.size() != 3 || s[0] < 1 || s[0] > 4 ||
        s[1] <= 0 || s[2] <= 0 || s[1] > INT32_MAX || s[2] > INT32_MAX || !tensor.data())
        throw std::invalid_argument("image requires U8 [channels,width,height], channels in [1,4], and positive dimensions");
    return {uint32_t(s[1]), uint32_t(s[2]), uint32_t(s[0]), const_cast<uint8_t*>(static_cast<const uint8_t*>(tensor.data()))};
}

sd::Audio sd::api::audio_view(const sd_audio_t& audio) {
    if (!audio.data || audio.sample_rate == 0)
        throw std::invalid_argument("audio requires samples and a positive sample rate");
    const auto& t = *audio.data;
    if (t.type != SD_TENSOR_F32 || t.shape.size() != 2 || t.shape[0] <= 0 || t.shape[0] > UINT32_MAX || t.shape[1] <= 0 || !t.data())
        throw std::invalid_argument("audio requires F32 [channels,samples] with positive dimensions");
    return {audio.sample_rate, uint32_t(t.shape[0]), uint64_t(t.shape[1]), const_cast<float*>(static_cast<const float*>(t.data()))};
}

std::vector<sd::Image> sd::api::video_frames(const sd_tensor_t& tensor) {
    const auto& s = tensor.shape;
    if (tensor.type != SD_TENSOR_U8 || s.size() != 4 || s[0] < 1 || s[0] > 4 || s[1] <= 0 || s[2] <= 0 || s[3] <= 0 ||
        s[1] > INT32_MAX || s[2] > INT32_MAX || s[3] > INT32_MAX || !tensor.data())
        throw std::invalid_argument("video requires U8 [channels,width,height,frames] with positive dimensions");
    const size_t stride = size_t(s[0]) * size_t(s[1]) * size_t(s[2]);
    auto* data          = const_cast<uint8_t*>(static_cast<const uint8_t*>(tensor.data()));
    std::vector<sd::Image> result;
    result.reserve(size_t(s[3]));
    for (int64_t i = 0; i < s[3]; ++i)
        result.push_back({uint32_t(s[1]), uint32_t(s[2]), uint32_t(s[0]), data + size_t(i) * stride});
    return result;
}

void sd::api::preview_images(const std::function<void(int, int, const sd_tensor_t* const*, bool, void*)>& callback, int step, int count, const sd::Image* images, bool noisy, void* data) {
    std::vector<TensorRef> tensors;
    std::vector<const sd_tensor_t*> views;
    tensors.reserve(count);
    views.reserve(count);
    for (int i = 0; i < count; ++i) {
        const auto& image     = images[i];
        const int64_t shape[] = {image.channel, image.width, image.height};
        auto* tensor          = sd_tensor_create(SD_TENSOR_U8, 3, shape, image.data, size_t(image.channel) * image.width * image.height);
        if (!tensor)
            throw std::bad_alloc();
        tensors.push_back(TensorRef::adopt(tensor));
        views.push_back(tensors.back().get());
    }
    callback(step, count, views.data(), noisy, data);
}

sd_tensor_t* sd_tensor_create(sd_tensor_type_t type, size_t rank, const int64_t* shape, const void* data, size_t data_size) {
    sd_tensor_t* result = nullptr;
    sd::api::call([&] {
        if (rank > 8 || (rank && !shape))
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "invalid tensor shape");
        std::vector<int64_t> dimensions;
        if (rank)
            dimensions.assign(shape, shape + rank);
        const size_t bytes = sd::api::tensor_byte_size(type, dimensions);
        if (data_size != bytes || (bytes && !data))
            return sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "tensor data size does not match type and shape");
        result = new sd_tensor_t(type, std::move(dimensions), data, data_size);
        return SD_STATUS_OK;
    });
    return result;
}

sd_tensor_t* sd_tensor_retain(const sd_tensor_t* tensor) {
    if (tensor)
        tensor->references.fetch_add(1, std::memory_order_relaxed);
    return const_cast<sd_tensor_t*>(tensor);
}

void sd_tensor_free(sd_tensor_t* tensor) {
    if (tensor && tensor->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete tensor;
}

static bool valid_tensor(const sd_tensor_t* tensor) {
    sd::api::clear_error();
    if (!tensor)
        sd::api::error(SD_STATUS_INVALID_ARGUMENT, nullptr, "tensor is required");
    return tensor != nullptr;
}

sd_tensor_type_t sd_tensor_type(const sd_tensor_t* tensor) {
    return valid_tensor(tensor) ? tensor->type : 0;
}
size_t sd_tensor_rank(const sd_tensor_t* tensor) {
    return valid_tensor(tensor) ? tensor->shape.size() : 0;
}
size_t sd_tensor_byte_size(const sd_tensor_t* tensor) {
    return valid_tensor(tensor) ? tensor->bytes : 0;
}
const void* sd_tensor_data(const sd_tensor_t* tensor) {
    return valid_tensor(tensor) ? tensor->data() : nullptr;
}
int64_t sd_tensor_dim(const sd_tensor_t* tensor, size_t axis) {
    if (!valid_tensor(tensor))
        return 0;
    if (axis >= tensor->shape.size()) {
        sd::api::error(SD_STATUS_OUT_OF_RANGE, nullptr, "invalid tensor axis");
        return 0;
    }
    return tensor->shape[axis];
}

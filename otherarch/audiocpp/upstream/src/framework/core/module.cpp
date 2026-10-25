#include "engine/framework/core/module.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace engine::core {

std::array<int64_t, kMaxTensorRank> to_ggml_dims(const TensorShape & shape) {
    std::array<int64_t, kMaxTensorRank> ggml_dims = {1, 1, 1, 1};
    for (size_t i = 0; i < shape.rank; ++i) {
        ggml_dims[i] = shape.dims[shape.rank - 1 - i];
    }
    return ggml_dims;
}

namespace {

void ensure_positive_dims(const TensorShape & shape) {
    if (shape.rank == 0 || shape.rank > kMaxTensorRank) {
        throw std::runtime_error("Tensor rank must be between 1 and 4");
    }

    for (size_t i = 0; i < shape.rank; ++i) {
        if (shape.dims[i] <= 0) {
            throw std::runtime_error("Tensor dimensions must be positive");
        }
    }
}

std::string name_or_default(const char * name) {
    return name == nullptr ? std::string("tensor") : std::string(name);
}

}  // namespace

TensorShape TensorShape::from_dims(std::initializer_list<int64_t> dims_init) {
    TensorShape shape;
    shape.rank = dims_init.size();
    if (shape.rank == 0 || shape.rank > kMaxTensorRank) {
        throw std::runtime_error("Tensor rank must be between 1 and 4");
    }

    size_t index = 0;
    for (const int64_t dim : dims_init) {
        shape.dims[index++] = dim;
    }

    ensure_positive_dims(shape);
    return shape;
}

int64_t TensorShape::at(size_t index) const {
    if (index >= rank) {
        throw std::runtime_error("TensorShape index out of range");
    }
    return dims[index];
}

int64_t TensorShape::last_dim() const {
    return at(rank - 1);
}

int64_t TensorShape::prefix_elements() const {
    if (rank <= 1) {
        return 1;
    }

    int64_t total = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        total *= dims[i];
    }
    return total;
}

int64_t TensorShape::num_elements() const {
    int64_t total = 1;
    for (size_t i = 0; i < rank; ++i) {
        total *= dims[i];
    }
    return total;
}

TensorShape TensorShape::with_last_dim(int64_t value) const {
    TensorShape result = *this;
    result.dims[rank - 1] = value;
    ensure_positive_dims(result);
    return result;
}

std::string TensorShape::to_string() const {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < rank; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << dims[i];
    }
    out << "]";
    return out.str();
}

bool TensorValue::valid() const noexcept {
    return tensor != nullptr && shape.rank > 0;
}

TensorValue make_tensor(ModuleBuildContext & ctx, ggml_type type, const TensorShape & shape) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }

    ensure_positive_dims(shape);
    const auto ggml_dims = to_ggml_dims(shape);

    ggml_tensor * tensor = nullptr;
    switch (shape.rank) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx.ggml, type, ggml_dims[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx.ggml, type, ggml_dims[0], ggml_dims[1]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(ctx.ggml, type, ggml_dims[0], ggml_dims[1], ggml_dims[2]);
            break;
        case 4:
            tensor = ggml_new_tensor_4d(ctx.ggml, type, ggml_dims[0], ggml_dims[1], ggml_dims[2], ggml_dims[3]);
            break;
        default:
            throw std::runtime_error("Unsupported tensor rank");
    }

    return wrap_tensor(tensor, shape, type);
}

int logical_axis_to_ggml_axis(size_t rank, int logical_axis) {
    if (logical_axis < 0 || logical_axis >= static_cast<int>(rank)) {
        throw std::runtime_error("Logical axis out of range");
    }
    return static_cast<int>(rank) - 1 - logical_axis;
}

TensorValue wrap_tensor(ggml_tensor * tensor, const TensorShape & shape, ggml_type type) {
    if (tensor == nullptr) {
        throw std::runtime_error("Cannot wrap null ggml tensor");
    }

    ensure_positive_dims(shape);
    const auto ggml_dims = to_ggml_dims(shape);
    for (size_t i = 0; i < shape.rank; ++i) {
        if (tensor->ne[i] != ggml_dims[i]) {
            throw std::runtime_error("Wrapped ggml tensor shape does not match logical shape " + shape.to_string());
        }
    }

    return TensorValue{tensor, shape, type};
}

TensorValue reshape_tensor(ModuleBuildContext & ctx, const TensorValue & value, const TensorShape & new_shape) {
    if (!value.valid()) {
        throw std::runtime_error("Cannot reshape an invalid tensor");
    }
    if (value.shape.num_elements() != new_shape.num_elements()) {
        throw std::runtime_error("Reshape element count mismatch");
    }
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }

    const auto ggml_dims = to_ggml_dims(new_shape);
    if (ctx.elide_identity_reshapes && ggml_is_contiguous(value.tensor) &&
        std::equal(ggml_dims.begin(), ggml_dims.end(), value.tensor->ne)) {
        return wrap_tensor(value.tensor, new_shape, value.type);
    }
    ggml_tensor * reshaped = nullptr;
    switch (new_shape.rank) {
        case 1:
            reshaped = ggml_reshape_1d(ctx.ggml, value.tensor, ggml_dims[0]);
            break;
        case 2:
            reshaped = ggml_reshape_2d(ctx.ggml, value.tensor, ggml_dims[0], ggml_dims[1]);
            break;
        case 3:
            reshaped = ggml_reshape_3d(ctx.ggml, value.tensor, ggml_dims[0], ggml_dims[1], ggml_dims[2]);
            break;
        case 4:
            reshaped = ggml_reshape_4d(ctx.ggml, value.tensor, ggml_dims[0], ggml_dims[1], ggml_dims[2], ggml_dims[3]);
            break;
        default:
            throw std::runtime_error("Unsupported reshape rank");
    }

    return wrap_tensor(reshaped, new_shape, value.type);
}

bool has_backend_addressable_layout(const ggml_tensor * tensor) noexcept {
    return tensor != nullptr &&
        ggml_is_contiguous(tensor) &&
        tensor->view_offs == 0;
}

TensorValue ensure_backend_addressable_layout(ModuleBuildContext & ctx, const TensorValue & value) {
    if (!value.valid()) {
        throw std::runtime_error("Cannot materialize an invalid tensor");
    }
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (has_backend_addressable_layout(value.tensor)) {
        return value;
    }
    return wrap_tensor(ggml_cont(ctx.ggml, value.tensor), value.shape, value.type);
}

void validate_shape(const TensorValue & value, const TensorShape & expected, const char * name) {
    if (!value.valid()) {
        throw std::runtime_error(name_or_default(name) + " is invalid");
    }
    if (value.shape.rank != expected.rank) {
        throw std::runtime_error(name_or_default(name) + " rank mismatch: expected " + expected.to_string() +
                                 ", got " + value.shape.to_string());
    }
    for (size_t i = 0; i < expected.rank; ++i) {
        if (value.shape.dims[i] != expected.dims[i]) {
            throw std::runtime_error(name_or_default(name) + " shape mismatch: expected " + expected.to_string() +
                                     ", got " + value.shape.to_string());
        }
    }
}

void validate_last_dim(const TensorValue & value, int64_t expected, const char * name) {
    if (!value.valid()) {
        throw std::runtime_error(name_or_default(name) + " is invalid");
    }
    if (value.shape.last_dim() != expected) {
        throw std::runtime_error(name_or_default(name) + " last dimension mismatch: expected " +
                                 std::to_string(expected) + ", got " + std::to_string(value.shape.last_dim()));
    }
}

void validate_rank_between(const TensorValue & value, size_t min_rank, size_t max_rank, const char * name) {
    if (!value.valid()) {
        throw std::runtime_error(name_or_default(name) + " is invalid");
    }
    if (value.shape.rank < min_rank || value.shape.rank > max_rank) {
        throw std::runtime_error(name_or_default(name) + " rank mismatch: expected between " +
                                 std::to_string(min_rank) + " and " + std::to_string(max_rank) +
                                 ", got " + std::to_string(value.shape.rank));
    }
}

}  // namespace engine::core

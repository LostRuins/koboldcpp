#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace engine::neon {

enum class Layout { ChannelFirst, FrameMajor };
enum class DataType { Float32, Int32 };

// Dimensions remain logical [channels, frames] or [channels, height, width].
// FrameMajor stores a sequence as [frames, channels].
struct Shape {
    std::array<int, 4> dimensions{};
    size_t rank = 0;
    Layout layout = Layout::ChannelFirst;

    Shape(std::initializer_list<int> dimensions, Layout layout = Layout::ChannelFirst);
    size_t elements() const;
    bool operator==(const Shape &other) const noexcept;
};

struct TensorView {
    const float *data;
    size_t size;
    Shape shape;
    const int32_t *indices = nullptr;
    DataType type = DataType::Float32;

    TensorView(const std::vector<float> &values, Shape shape);
    TensorView(const float *data, size_t size, Shape shape);
    TensorView(const int32_t *indices, size_t size, Shape shape);
};

struct Tensor {
    Shape shape;
    std::vector<float> values;
};

} // namespace engine::neon

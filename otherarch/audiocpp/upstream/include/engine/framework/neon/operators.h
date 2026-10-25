#pragma once

#include "engine/framework/neon/tensor.h"

#include <memory>
#include <vector>

namespace engine::neon {
namespace detail {
class Kernel;
}

// Prepared operators own immutable weights. They can be shared by independent plans.
class Operator {
public:
    virtual ~Operator();

protected:
    explicit Operator(std::shared_ptr<const detail::Kernel> kernel);

private:
    std::shared_ptr<const detail::Kernel> kernel_;
    friend class ExecutionPlan;
};

struct Conv1dConfig {
    int input_channels = 0;
    int output_channels = 0;
    int kernel_size = 0;
    int stride = 1;
    int dilation = 1;
    int padding_left = 0;
    int padding_right = 0;
};

// Row-major [output_features, input_features] weights. Sequence inputs may be
// channel-first or frame-major; unlike convolution this performs no padding.
class Linear : public Operator {
public:
    Linear(int input_features, int output_features, std::vector<float> weight, std::vector<float> bias = {});
};

// Symmetric signed 8-bit blocks of 32 values, in row-major order. Scales are
// supplied as floats; this representation has no container-format dependency.
struct BlockInt8Weights {
    std::vector<int8_t> values;
    std::vector<float> scales;
};

class QuantizedLinear : public Operator {
public:
    // Activations use symmetric 32-value blocks with a binary16-rounded scale
    // and nearest-integer quantization. Supplied weight scales remain unchanged.
    QuantizedLinear(int input_features, int output_features, BlockInt8Weights weight, std::vector<float> bias = {});
};

class Embedding : public Operator {
public:
    Embedding(int features, int rows, std::vector<float> weight);
    Embedding(int features, int rows, BlockInt8Weights weight);
};

class Conv1d : public Operator {
public:
    Conv1d(Conv1dConfig config, const std::vector<float> &weight, const std::vector<float> &bias = {});
};

struct Conv2dConfig {
    int input_channels = 0;
    int output_channels = 0;
    int kernel_height = 0;
    int kernel_width = 0;
    int stride_height = 1;
    int stride_width = 1;
    int dilation_height = 1;
    int dilation_width = 1;
    int padding_height = 0;
    int padding_width = 0;
};

class Conv2d : public Operator {
public:
    Conv2d(Conv2dConfig config, const std::vector<float> &weight, const std::vector<float> &bias = {});
};

class DepthwiseConv1d : public Operator {
public:
    DepthwiseConv1d(int channels, int kernel_size, const std::vector<float> &weight, const std::vector<float> &bias);
    DepthwiseConv1d(Conv1dConfig config, const std::vector<float> &weight, const std::vector<float> &bias);
};

class LayerNorm : public Operator {
public:
    LayerNorm(const std::vector<float> &weight, const std::vector<float> &bias, float epsilon);
};

enum class ActivationKind { SiLU, GELU, LeakyReLU, Sigmoid, Tanh };

class Activation : public Operator {
public:
    explicit Activation(ActivationKind kind, float negative_slope = 0.0F);
};

class Scale : public Operator {
public:
    explicit Scale(float scale);
};

// Add [channels, 1] or [channels, frames] to a channel-first sequence.
class BroadcastAdd : public Operator {
public:
    BroadcastAdd();
};

class ReflectPad1d : public Operator {
public:
    ReflectPad1d(int left, int right);
};

// Inputs: [channels, frames] and [2*channels, 1] affine parameters.
// output = normalize_time(input) * (1 + affine[:channels]) + affine[channels:].
class AdaptiveInstanceNorm : public Operator {
public:
    explicit AdaptiveInstanceNorm(float epsilon);
};

// Channel-wise x + sin(alpha*x)^2 / alpha. Alpha must be nonzero.
class Snake : public Operator {
public:
    explicit Snake(std::vector<float> alpha);
};

class ClampMin : public Operator {
public:
    explicit ClampMin(float minimum);
};

class MaxPool2d : public Operator {
public:
    MaxPool2d(int kernel_height, int kernel_width, int stride_height, int stride_width);
};

// output = input[0] + scale[channel] * input[1]; scale may be scalar.
class ScaleAdd : public Operator {
public:
    explicit ScaleAdd(const std::vector<float> &scale);
};

class ConcatChannels : public Operator {
public:
    ConcatChannels();
};

// Logical-axis operations on contiguous channel-first tensors.
class Concat : public Operator {
public:
    explicit Concat(int axis);
};

class Slice : public Operator {
public:
    Slice(int axis, int start, int length);
};

class Reshape : public Operator {
public:
    explicit Reshape(Shape shape);
};

// Matrix transpose: [channels, frames] becomes [frames, channels].
class Transpose2d : public Operator {
public:
    Transpose2d();
};

class Clamp : public Operator {
public:
    Clamp(float minimum, float maximum);
};

// Round ties away from zero, clamp to integer bounds, then dequantize.
class QuantizeDequantize : public Operator {
public:
    QuantizeDequantize(float scale, int zero_point, int minimum, int maximum);
};

// Spatial dimensions follow the input; values do not depend on its contents.
class ConstantChannels : public Operator {
public:
    ConstantChannels(int channels, float value);
};

// Adjacent-pair rotary embedding: feature tensor plus a scalar Int32 start position.
class RotaryEmbedding : public Operator {
public:
    RotaryEmbedding(int head_dimension, int rotary_dimension, float theta);
};

// Three inputs perform full attention: Q, K, V. Six inputs attend a cached prefix
// plus the current K/V: Q, past K, past V, current K, current V, Int32 position.
// Q is [heads*dimension, query_frames]. K/V may be frame-major sequences or
// contiguous [key_value_heads, frames, dimension] blocks produced by SplitHeads.
class Attention : public Operator {
public:
    Attention(int heads, int key_value_heads, int head_dimension);
};

// First half is the value, second half the SiLU gate.
class SwiGLU : public Operator {
public:
    SwiGLU();
};

// Pack a frame-major [heads*dimension, frames] sequence into contiguous
// [heads, frames, dimension] blocks for repeated attention reads.
class SplitHeads : public Operator {
public:
    explicit SplitHeads(int heads);
};

} // namespace engine::neon

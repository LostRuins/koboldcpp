#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine::assets {

template <typename BatchNorm1dWeights>
BatchNorm1dWeights load_batch_norm1d_metadata(
    const TensorSource & source,
    const std::string & prefix) {
    BatchNorm1dWeights bn;
    const auto weight = source.require_metadata(prefix + ".weight");
    if (weight.shape.size() != 1) {
        throw std::runtime_error("batch norm weight must be 1D at " + prefix);
    }
    bn.channels = weight.shape[0];
    bn.weight = source.require_f32(prefix + ".weight", {bn.channels});
    bn.bias = source.require_f32(prefix + ".bias", {bn.channels});
    bn.running_mean = source.require_f32(prefix + ".running_mean", {bn.channels});
    bn.running_var = source.require_f32(prefix + ".running_var", {bn.channels});
    return bn;
}

template <typename Conv1dWeights>
Conv1dWeights load_conv1d_metadata(
    const TensorSource & source,
    const std::string & prefix,
    int64_t stride,
    int64_t dilation,
    int64_t padding,
    int64_t groups,
    bool use_bias) {
    Conv1dWeights conv;
    const auto weight = source.require_metadata(prefix + ".weight");
    if (weight.shape.size() != 3) {
        throw std::runtime_error("conv weight must be 3D at " + prefix);
    }
    conv.weight_name = prefix + ".weight";
    conv.weight_source_shape = weight.shape;
    conv.out_channels = weight.shape[0];
    conv.in_channels = weight.shape[1] * groups;
    conv.kernel = weight.shape[2];
    conv.stride = stride;
    conv.dilation = dilation;
    conv.padding = padding;
    conv.groups = groups;
    conv.use_bias = use_bias;
    if (use_bias) {
        (void) source.require_metadata(prefix + ".bias");
        conv.bias_name = prefix + ".bias";
    }
    return conv;
}

template <typename Conv1dWeights>
Conv1dWeights load_linear_as_conv1d_metadata(
    const TensorSource & source,
    const std::string & prefix,
    bool use_bias) {
    Conv1dWeights conv;
    const auto weight = source.require_metadata(prefix + ".weight");
    if (weight.shape.size() != 2) {
        throw std::runtime_error("linear weight must be 2D at " + prefix);
    }
    conv.weight_name = prefix + ".weight";
    conv.weight_source_shape = weight.shape;
    conv.out_channels = weight.shape[0];
    conv.in_channels = weight.shape[1];
    conv.kernel = 1;
    conv.stride = 1;
    conv.dilation = 1;
    conv.padding = 0;
    conv.groups = 1;
    conv.use_bias = use_bias;
    if (use_bias) {
        (void) source.require_metadata(prefix + ".bias");
        conv.bias_name = prefix + ".bias";
    }
    return conv;
}

}  // namespace engine::assets

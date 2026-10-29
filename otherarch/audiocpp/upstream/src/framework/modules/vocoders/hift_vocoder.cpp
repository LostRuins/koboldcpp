#include "engine/framework/modules/vocoders/hift_vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/sampling/torch_random.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

using Conv1dWeights = HiftVocoderWeights::Conv1dWeights;
using ConvTranspose1dWeights = HiftVocoderWeights::ConvTranspose1dWeights;
using LinearWeights = HiftVocoderWeights::LinearWeights;
using ResBlockWeights = HiftVocoderWeights::ResBlockWeights;
using SnakeWeights = HiftVocoderWeights::SnakeWeights;

constexpr int64_t kHiftStftBins = 9;
constexpr int64_t kHiftStftChannels = kHiftStftBins * 2;

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("HiFT tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("HiFT tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

std::string tensor_name(const HiftVocoderConfig & config, const std::string & name) {
    return config.tensor_prefix + name;
}

std::vector<float> fold_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t leading,
    int64_t inner_size) {
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t row = 0; row < leading; ++row) {
        double norm = 0.0;
        for (int64_t col = 0; col < inner_size; ++col) {
            const float value = v[static_cast<size_t>(row * inner_size + col)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(row)] / std::sqrt(static_cast<float>(norm) + 1.0e-12F);
        for (int64_t col = 0; col < inner_size; ++col) {
            weight[static_cast<size_t>(row * inner_size + col)] =
                v[static_cast<size_t>(row * inner_size + col)] * scale;
        }
    }
    return weight;
}

void validate_config(const HiftVocoderConfig & config) {
    if (config.in_channels <= 0 || config.base_channels <= 0 || config.nb_harmonics < 0 ||
        config.sampling_rate <= 0 || config.istft_n_fft <= 0 || config.istft_hop <= 0 ||
        config.f0_num_class != 1 || config.f0_in_channels <= 0 || config.f0_cond_channels <= 0) {
        throw std::runtime_error("HiFT config contains invalid dimensions");
    }
    if (config.istft_n_fft != 16) {
        throw std::runtime_error("HiFT framework module currently expects n_fft=16");
    }
    if (config.upsample_rates.empty() ||
        config.upsample_rates.size() != config.upsample_kernel_sizes.size() ||
        config.source_resblock_kernel_sizes.size() != config.upsample_rates.size() ||
        config.source_resblock_dilation_sizes.size() != config.source_resblock_kernel_sizes.size() ||
        config.resblock_dilation_sizes.size() != config.resblock_kernel_sizes.size()) {
        throw std::runtime_error("HiFT config arrays are inconsistent");
    }
    for (const auto rate : config.upsample_rates) {
        if (rate <= 0) {
            throw std::runtime_error("HiFT upsample rate must be positive");
        }
    }
}

int64_t causal_conv1d_padding(int64_t kernel, int64_t dilation) {
    return ((kernel * dilation - dilation) / 2) * 2 + (kernel + 1) % 2;
}

Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiftVocoderConfig & config,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    assets::TensorStorageType storage_type,
    bool use_weight_norm) {
    Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    if (use_weight_norm) {
        const auto g = source.require_f32(
            tensor_name(config, prefix + ".parametrizations.weight.original0"),
            {out_channels, 1, 1});
        const auto v = source.require_f32(
            tensor_name(config, prefix + ".parametrizations.weight.original1"),
            {out_channels, in_channels, kernel});
        conv.weight = fold_weight_norm(g, v, out_channels, in_channels * kernel);
    } else {
        conv.weight = source.require_f32(tensor_name(config, prefix + ".weight"), {out_channels, in_channels, kernel});
    }
    conv.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        storage_type,
        conv.weight);
    if (use_bias) {
        conv.bias = source.require_f32(tensor_name(config, prefix + ".bias"), {out_channels});
        conv.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_channels}), assets::TensorStorageType::F32, conv.bias);
    }
    return conv;
}

ConvTranspose1dWeights load_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiftVocoderConfig & config,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    bool use_bias,
    assets::TensorStorageType storage_type,
    bool use_weight_norm) {
    ConvTranspose1dWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.use_bias = use_bias;
    if (use_weight_norm) {
        const auto g = source.require_f32(
            tensor_name(config, prefix + ".parametrizations.weight.original0"),
            {in_channels, 1, 1});
        const auto v = source.require_f32(
            tensor_name(config, prefix + ".parametrizations.weight.original1"),
            {in_channels, out_channels, kernel});
        conv.weight = fold_weight_norm(g, v, in_channels, out_channels * kernel);
    } else {
        conv.weight = source.require_f32(tensor_name(config, prefix + ".weight"), {in_channels, out_channels, kernel});
    }
    conv.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({in_channels, out_channels, kernel}),
        storage_type,
        conv.weight);
    if (use_bias) {
        conv.bias = source.require_f32(tensor_name(config, prefix + ".bias"), {out_channels});
        conv.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_channels}), assets::TensorStorageType::F32, conv.bias);
    }
    return conv;
}

LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiftVocoderConfig & config,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    assets::TensorStorageType storage_type) {
    LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight = source.require_f32(tensor_name(config, prefix + ".weight"), {out_features, in_features});
    linear.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({out_features, in_features}),
        storage_type,
        linear.weight);
    if (use_bias) {
        linear.bias = source.require_f32(tensor_name(config, prefix + ".bias"), {out_features});
        linear.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_features}), assets::TensorStorageType::F32, linear.bias);
    }
    return linear;
}

SnakeWeights load_snake(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiftVocoderConfig & config,
    const std::string & name,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    SnakeWeights snake;
    snake.alpha = source.require_f32(tensor_name(config, name), {channels});
    std::vector<float> inv_alpha(snake.alpha.size(), 0.0F);
    for (size_t index = 0; index < snake.alpha.size(); ++index) {
        inv_alpha[index] = 1.0F / (snake.alpha[index] + 1.0e-9F);
    }
    snake.alpha_tensor = store.make_from_f32(core::TensorShape::from_dims({channels, 1}), assets::TensorStorageType::F32, snake.alpha);
    snake.inv_alpha_tensor =
        store.make_from_f32(core::TensorShape::from_dims({channels, 1}), assets::TensorStorageType::F32, std::move(inv_alpha));
    return snake;
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiftVocoderConfig & config,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations,
    assets::TensorStorageType storage_type,
    bool causal = false) {
    ResBlockWeights block;
    block.convs1.reserve(dilations.size());
    block.convs2.reserve(dilations.size());
    block.activations1.reserve(dilations.size());
    block.activations2.reserve(dilations.size());
    for (size_t index = 0; index < dilations.size(); ++index) {
        const int64_t dilation = dilations[index];
        auto conv1 = load_conv1d(
            store,
            source,
            config,
            prefix + ".convs1." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            causal ? 0 : (kernel * dilation - dilation) / 2,
            dilation,
            true,
            storage_type,
            config.weight_layout == HiftVocoderWeightLayout::TorchParametrizedWeightNorm);
        if (causal) {
            conv1.pad_left = causal_conv1d_padding(kernel, dilation);
        }
        block.convs1.push_back(std::move(conv1));
        auto conv2 = load_conv1d(
            store,
            source,
            config,
            prefix + ".convs2." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            causal ? 0 : (kernel - 1) / 2,
            1,
            true,
            storage_type,
            config.weight_layout == HiftVocoderWeightLayout::TorchParametrizedWeightNorm);
        if (causal) {
            conv2.pad_left = causal_conv1d_padding(kernel, 1);
        }
        block.convs2.push_back(std::move(conv2));
        block.activations1.push_back(load_snake(
            store,
            source,
            config,
            prefix + ".activations1." + std::to_string(index) + ".alpha",
            channels,
            storage_type));
        block.activations2.push_back(load_snake(
            store,
            source,
            config,
            prefix + ".activations2." + std::to_string(index) + ".alpha",
            channels,
            storage_type));
    }
    return block;
}

ggml_tensor * named(ggml_tensor * tensor, const char * name) {
    return ggml_set_name(tensor, name);
}

ggml_tensor * contiguous_if_needed(ggml_context * ctx, ggml_tensor * tensor) {
    return core::has_backend_addressable_layout(tensor) ? tensor : ggml_cont(ctx, tensor);
}

ggml_tensor * weight_2d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    const std::string & name) {
    return named(ggml_reshape_2d(ctx, value.tensor, ne0, ne1), name.c_str());
}

ggml_tensor * weight_3d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    const std::string & name) {
    return named(ggml_reshape_3d(ctx, value.tensor, ne0, ne1, ne2), name.c_str());
}

ggml_tensor * reflect_pad_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int64_t left,
    int64_t right) {
    if (left < 0 || right < 0 || left >= x->ne[0] || right >= x->ne[0]) {
        throw std::runtime_error("HiFT reflect pad size mismatch");
    }
    return ggml_pad_reflect_1d(ctx, x, static_cast<int>(left), static_cast<int>(right));
}

ggml_tensor * conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const Conv1dWeights & conv,
    const std::string & name,
    bool allow_pointwise_fastpath) {
    if (allow_pointwise_fastpath &&
        conv.kernel == 1 &&
        conv.stride == 1 &&
        conv.padding == 0 &&
        conv.pad_left == 0 &&
        conv.pad_right == 0 &&
        conv.dilation == 1) {
        ggml_tensor * weight = weight_2d(ctx, conv.weight_tensor, conv.in_channels, conv.out_channels, name + ".weight");
        ggml_tensor * x_cf = contiguous_if_needed(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        ggml_tensor * y_cf = ggml_mul_mat(ctx, weight, x_cf);
        ggml_tensor * y = contiguous_if_needed(ctx, ggml_permute(ctx, y_cf, 1, 0, 2, 3));
        y = ggml_reshape_2d(ctx, y, x->ne[0], conv.out_channels);
        if (conv.use_bias) {
            y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
        }
        return named(y, name.c_str());
    }

    ggml_tensor * input = contiguous_if_needed(ctx, x);
    int64_t padding = conv.padding;
    if (conv.pad_left != 0 || conv.pad_right != 0) {
        input = ggml_pad_ext(
            ctx,
            input,
            static_cast<int>(conv.pad_left),
            static_cast<int>(conv.pad_right),
            0,
            0,
            0,
            0,
            0,
            0);
        padding = 0;
    }

    ggml_tensor * weight =
        weight_3d(ctx, conv.weight_tensor, conv.kernel, conv.in_channels, conv.out_channels, name + ".weight");
    ggml_tensor * x3 = ggml_reshape_3d(ctx, input, input->ne[0], input->ne[1], 1);
    ggml_tensor * y3 =
        ggml_conv_1d(ctx, weight, x3, static_cast<int>(conv.stride), static_cast<int>(padding), static_cast<int>(conv.dilation));
    ggml_tensor * y = ggml_reshape_2d(ctx, y3, y3->ne[0], y3->ne[1]);
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * conv_transpose1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const ConvTranspose1dWeights & conv,
    const std::string & name) {
    ggml_tensor * weight =
        weight_3d(ctx, conv.weight_tensor, conv.kernel, conv.out_channels, conv.in_channels, name + ".weight");
    ggml_tensor * y_full = ggml_conv_transpose_1d(ctx, weight, contiguous_if_needed(ctx, x), static_cast<int>(conv.stride), 0, 1);
    const int64_t cropped_len = (x->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel;
    ggml_tensor * y =
        ggml_view_2d(ctx, y_full, cropped_len, conv.out_channels, y_full->nb[1], static_cast<size_t>(conv.padding) * sizeof(float));
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * causal_conv1d_nearest_upsample(
    ggml_context * ctx,
    ggml_tensor * x,
    const Conv1dWeights & conv,
    int64_t scale,
    const std::string & name) {
    if (scale <= 0) {
        throw std::runtime_error("HiFT nearest upsample scale must be positive");
    }
    auto * contiguous = contiguous_if_needed(ctx, x);
    auto * upsampled = ggml_interpolate(
        ctx,
        contiguous,
        contiguous->ne[0] * scale,
        contiguous->ne[1],
        contiguous->ne[2],
        contiguous->ne[3],
        GGML_SCALE_MODE_NEAREST);
    upsampled = named(upsampled, (name + ".nearest").c_str());
    auto * padded = ggml_pad_ext(ctx, upsampled, static_cast<int>(conv.kernel - 1), 0, 0, 0, 0, 0, 0, 0);
    return conv1d(ctx, padded, conv, name, false);
}

ggml_tensor * snake(
    ggml_context * ctx,
    ggml_tensor * x,
    const SnakeWeights & act,
    const std::string & name) {
    ggml_tensor * alpha = weight_2d(ctx, act.alpha_tensor, 1, x->ne[1], name + ".alpha");
    ggml_tensor * inv = weight_2d(ctx, act.inv_alpha_tensor, 1, x->ne[1], name + ".inv_alpha");
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    s = ggml_mul(ctx, ggml_sqr(ctx, s), inv);
    return named(ggml_add(ctx, x, s), name.c_str());
}

ggml_tensor * resblock(
    ggml_context * ctx,
    ggml_tensor * x,
    const ResBlockWeights & block,
    const std::string & name,
    bool allow_pointwise_fastpath) {
    for (size_t index = 0; index < block.convs1.size(); ++index) {
        ggml_tensor * xt = snake(ctx, x, block.activations1[index], name + ".act1." + std::to_string(index));
        xt = conv1d(ctx, xt, block.convs1[index], name + ".convs1." + std::to_string(index), allow_pointwise_fastpath);
        xt = snake(ctx, xt, block.activations2[index], name + ".act2." + std::to_string(index));
        xt = conv1d(ctx, xt, block.convs2[index], name + ".convs2." + std::to_string(index), allow_pointwise_fastpath);
        x = named(ggml_add(ctx, x, xt), (name + ".residual." + std::to_string(index)).c_str());
    }
    return x;
}

ggml_tensor * build_f0_graph(
    ggml_context * ctx,
    const HiftVocoderWeights & weights,
    ggml_tensor * input) {
    ggml_tensor * x = input;
    for (size_t index = 0; index < weights.f0_predictor.condnet.size(); ++index) {
        x = conv1d(ctx, x, weights.f0_predictor.condnet[index], "f0.condnet." + std::to_string(index), false);
        x = named(ggml_elu(ctx, x), ("f0.elu." + std::to_string(index)).c_str());
    }
    auto linear_as_conv = Conv1dWeights{};
    linear_as_conv.weight = weights.f0_predictor.classifier.weight;
    linear_as_conv.bias = weights.f0_predictor.classifier.bias;
    linear_as_conv.weight_tensor = weights.f0_predictor.classifier.weight_tensor;
    linear_as_conv.bias_tensor = weights.f0_predictor.classifier.bias_tensor;
    linear_as_conv.out_channels = weights.f0_predictor.classifier.out_features;
    linear_as_conv.in_channels = weights.f0_predictor.classifier.in_features;
    linear_as_conv.kernel = 1;
    linear_as_conv.stride = 1;
    linear_as_conv.padding = 0;
    linear_as_conv.dilation = 1;
    linear_as_conv.use_bias = weights.f0_predictor.classifier.use_bias;
    x = conv1d(ctx, x, linear_as_conv, "f0.classifier", false);
    return named(ggml_abs(ctx, x), "f0.output");
}

ggml_tensor * build_backend_graph(
    ggml_context * ctx,
    const HiftVocoderWeights & weights,
    ggml_tensor * speech_in,
    ggml_tensor * source_in,
    bool allow_pointwise_fastpath) {
    ggml_tensor * x = conv1d(ctx, speech_in, weights.conv_pre, "conv_pre", allow_pointwise_fastpath);
    const int64_t num_kernels = static_cast<int64_t>(weights.config.resblock_kernel_sizes.size());
    const size_t upsample_count = weights.config.upsample_rates.size();
    for (size_t up_index = 0; up_index < upsample_count; ++up_index) {
        x = named(ggml_leaky_relu(ctx, x, weights.config.lrelu_slope, false), ("ups." + std::to_string(up_index) + ".act").c_str());
        switch (weights.config.upsample_mode) {
            case HiftVocoderUpsampleMode::ConvTranspose1d:
                x = conv_transpose1d(ctx, x, weights.ups[up_index], "ups." + std::to_string(up_index));
                break;
            case HiftVocoderUpsampleMode::CausalConv1dNearest:
                x = causal_conv1d_nearest_upsample(
                    ctx,
                    x,
                    weights.up_convs[up_index],
                    weights.config.upsample_rates[up_index],
                    "ups." + std::to_string(up_index));
                break;
            default:
                throw std::runtime_error("HiFT unsupported upsample mode");
        }
        if (up_index + 1 == upsample_count) {
            x = named(reflect_pad_1d(ctx, x, 1, 0), "reflection_pad");
        }

        ggml_tensor * si =
            conv1d(ctx, source_in, weights.source_downs[up_index], "source_downs." + std::to_string(up_index), allow_pointwise_fastpath);
        si = resblock(ctx, si, weights.source_resblocks[up_index], "source_resblocks." + std::to_string(up_index), allow_pointwise_fastpath);
        x = named(ggml_add(ctx, x, si), ("fusion." + std::to_string(up_index)).c_str());

        ggml_tensor * sum = nullptr;
        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = static_cast<int64_t>(up_index) * num_kernels + kernel_index;
            ggml_tensor * block_out = resblock(
                ctx,
                x,
                weights.resblocks[static_cast<size_t>(block_index)],
                "resblocks." + std::to_string(block_index),
                allow_pointwise_fastpath);
            sum = sum == nullptr ? block_out : ggml_add(ctx, sum, block_out);
        }
        x = named(ggml_scale(ctx, sum, 1.0F / static_cast<float>(num_kernels)), ("upsample." + std::to_string(up_index)).c_str());
    }
    x = named(ggml_leaky_relu(ctx, x, 0.01F, false), "post.act");
    return conv1d(ctx, x, weights.conv_post, "conv_post", allow_pointwise_fastpath);
}

int64_t upsample_scale(const HiftVocoderConfig & config) {
    return std::accumulate(config.upsample_rates.begin(), config.upsample_rates.end(), config.istft_hop, std::multiplies<int64_t>{});
}

std::vector<float> nearest_upsample_f0(const std::vector<float> & f0, int64_t frames, int64_t scale) {
    if (static_cast<int64_t>(f0.size()) != frames) {
        throw std::runtime_error("HiFT F0 shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(frames * scale), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        std::fill(
            out.begin() + static_cast<std::ptrdiff_t>(t * scale),
            out.begin() + static_cast<std::ptrdiff_t>((t + 1) * scale),
            f0[static_cast<size_t>(t)]);
    }
    return out;
}

std::vector<float> make_sine_source(
    const HiftVocoderWeights & weights,
    const std::vector<float> & f0_upsampled,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwoPi = 2.0F * kPi;
    const int64_t frames = static_cast<int64_t>(f0_upsampled.size());
    const int64_t harmonics = weights.config.nb_harmonics + 1;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t gaussian_count = static_cast<size_t>(harmonics * frames);
    if (source_random_values != nullptr &&
        source_random_values->size() != phase_count + gaussian_count) {
        throw std::runtime_error("HiFT source random value count mismatch");
    }
    std::vector<float> phase_uniform = source_random_values == nullptr
        ? sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values)
        : std::vector<float>(source_random_values->begin(), source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count));
    std::vector<float> phase(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t h = 0; h < harmonics; ++h) {
        phase[static_cast<size_t>(h)] = phase_uniform[static_cast<size_t>(h)] * kTwoPi - kPi;
    }
    phase[0] = 0.0F;
    const uint64_t gaussian_offset = prior_noise_values + static_cast<uint64_t>(phase_uniform.size());
    std::vector<float> gaussian = source_random_values == nullptr
        ? sampling::generate_torch_cuda_randn(
              gaussian_count,
              seed,
              sampling::TorchRandnPrecision::Float32,
              gaussian_offset)
        : std::vector<float>(source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count), source_random_values->end());

    std::vector<float> source(static_cast<size_t>(frames), 0.0F);
    std::vector<float> frac(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        const float base_f0 = f0_upsampled[static_cast<size_t>(t)];
        const float uv = base_f0 > weights.config.nsf_voiced_threshold ? 1.0F : 0.0F;
        float sum = weights.source_linear.bias[0];
        for (int64_t h = 0; h < harmonics; ++h) {
            frac[static_cast<size_t>(h)] +=
                base_f0 * static_cast<float>(h + 1) / static_cast<float>(weights.config.sampling_rate);
            frac[static_cast<size_t>(h)] -= std::floor(frac[static_cast<size_t>(h)]);
            const float theta = kTwoPi * frac[static_cast<size_t>(h)];
            float wave =
                weights.config.nsf_alpha * std::sin(theta + phase[static_cast<size_t>(h)]);
            const float noise_amp =
                uv * weights.config.nsf_sigma + (1.0F - uv) * weights.config.nsf_alpha / 3.0F;
            wave = wave * uv + noise_amp * gaussian[static_cast<size_t>(h * frames + t)];
            sum += wave * weights.source_linear.weight[static_cast<size_t>(h)];
        }
        source[static_cast<size_t>(t)] = std::tanh(sum);
    }
    return source;
}

std::vector<float> make_causal_sinegen2_source(
    const HiftVocoderWeights & weights,
    const std::vector<float> & f0,
    int64_t frames,
    int64_t scale,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwoPi = 2.0F * kPi;
    if (static_cast<int64_t>(f0.size()) != frames) {
        throw std::runtime_error("HiFT CausalSineGen2 F0 shape mismatch");
    }
    const int64_t samples = frames * scale;
    const int64_t harmonics = weights.config.nb_harmonics + 1;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t noise_count = static_cast<size_t>(harmonics * samples);
    if (source_random_values != nullptr &&
        source_random_values->size() != phase_count + noise_count) {
        throw std::runtime_error("HiFT CausalSineGen2 source random value count mismatch");
    }

    std::vector<float> phase_uniform = source_random_values == nullptr
        ? sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values)
        : std::vector<float>(source_random_values->begin(), source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count));
    (void) phase_uniform;
    const uint64_t noise_offset = prior_noise_values + static_cast<uint64_t>(phase_uniform.size());
    std::vector<float> noise_uniform = source_random_values == nullptr
        ? sampling::generate_torch_cuda_uniform(noise_count, seed, noise_offset)
        : std::vector<float>(source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count), source_random_values->end());

    auto high_rad = [&](int64_t sample, int64_t harmonic) {
        const int64_t frame = std::clamp<int64_t>(sample / scale, 0, frames - 1);
        float rad = std::fmod(
            f0[static_cast<size_t>(frame)] * static_cast<float>(harmonic + 1) /
                static_cast<float>(weights.config.sampling_rate),
            1.0F);
        if (sample == 0) {
            rad += phase_uniform[static_cast<size_t>(harmonic)];
        }
        return rad;
    };

    std::vector<float> source(static_cast<size_t>(samples), 0.0F);
    std::vector<float> cumulative(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float base_f0 = f0[static_cast<size_t>(frame)];
        const float uv = base_f0 > weights.config.nsf_voiced_threshold ? 1.0F : 0.0F;
        const float noise_amp =
            uv * weights.config.nsf_sigma + (1.0F - uv) * weights.config.nsf_alpha / 3.0F;
        std::vector<float> frame_wave(static_cast<size_t>(harmonics), 0.0F);
        const float source_pos = (static_cast<float>(frame) + 0.5F) * static_cast<float>(scale) - 0.5F;
        const int64_t left = static_cast<int64_t>(std::floor(source_pos));
        const int64_t right = std::min<int64_t>(left + 1, samples - 1);
        const float lerp = source_pos - static_cast<float>(left);
        for (int64_t h = 0; h < harmonics; ++h) {
            const float rad = high_rad(std::max<int64_t>(left, 0), h) * (1.0F - lerp) + high_rad(right, h) * lerp;
            cumulative[static_cast<size_t>(h)] += rad;
            const float phase = cumulative[static_cast<size_t>(h)] * kTwoPi * static_cast<float>(scale);
            frame_wave[static_cast<size_t>(h)] = weights.config.nsf_alpha * std::sin(phase) * uv;
        }
        for (int64_t sample_in_frame = 0; sample_in_frame < scale; ++sample_in_frame) {
            const int64_t sample = frame * scale + sample_in_frame;
            float sum = weights.source_linear.bias[0];
            for (int64_t h = 0; h < harmonics; ++h) {
                const float wave = frame_wave[static_cast<size_t>(h)] +
                    noise_amp * noise_uniform[static_cast<size_t>(sample * harmonics + h)];
                sum += wave * weights.source_linear.weight[static_cast<size_t>(h)];
            }
            source[static_cast<size_t>(sample)] = std::tanh(sum);
        }
    }
    return source;
}

std::vector<float> make_sinegen2_source(
    const HiftVocoderWeights & weights,
    const std::vector<float> & f0,
    int64_t frames,
    int64_t scale,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values,
    const sampling::TorchCudaSamplingPolicy * random_policy) {
    constexpr float kTwoPi = 6.28318530717958647692F;
    const int64_t harmonics = weights.config.nb_harmonics + 1;
    const int64_t samples = frames * scale;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t noise_count = static_cast<size_t>(samples * harmonics);
    if (f0.size() != static_cast<size_t>(frames) || (source_random_values &&
        source_random_values->size() != phase_count + noise_count)) {
        throw std::runtime_error("HiFT SineGen2 source shape mismatch");
    }
    auto initial_phase = source_random_values
        ? std::vector<float>(source_random_values->begin(), source_random_values->begin() + phase_count)
        : sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values);
    initial_phase[0] = 0.0f;
    std::vector<float> noise(noise_count);
    if (source_random_values) {
        std::copy(source_random_values->begin() + phase_count, source_random_values->end(), noise.begin());
    } else if (random_policy != nullptr) {
        const auto offset = prior_noise_values + sampling::torch_cuda_tensor_iterator_offset_blocks(phase_count, *random_policy);
        sampling::fill_torch_cuda_tensor_iterator_randn(noise.data(), noise_count, seed, offset, *random_policy);
    } else {
        constexpr int64_t block_size = 16384;
        const int64_t blocks = (static_cast<int64_t>(noise_count) + block_size - 1) / block_size;
#ifdef _OPENMP
        #pragma omp parallel for if(blocks > 1)
#endif
        for (int64_t block = 0; block < blocks; ++block) {
            const auto offset = static_cast<size_t>(block * block_size);
            sampling::fill_torch_cuda_randn(noise.data() + offset,
                std::min<size_t>(block_size, noise_count - offset), seed,
                sampling::TorchRandnPrecision::Float32, prior_noise_values + phase_count + offset);
        }
    }
    const auto radians = [&](int64_t sample, int64_t harmonic) {
        float value = std::fmod(f0[static_cast<size_t>(sample / scale)] * static_cast<float>(harmonic + 1) /
            static_cast<float>(weights.config.sampling_rate), 1.0f);
        if (sample == 0) value += initial_phase[static_cast<size_t>(harmonic)];
        return value;
    };
    std::vector<float> phase(static_cast<size_t>(frames * harmonics));
    std::vector<float> cumulative(static_cast<size_t>(harmonics), 0.0f);
    // SineGen2 downsamples radians, integrates, then linearly upsamples the phase.
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float position = (static_cast<float>(frame) + 0.5f) * static_cast<float>(scale) - 0.5f;
        const auto left = static_cast<int64_t>(std::floor(position));
        const auto right = std::min(left + 1, samples - 1);
        const float fraction = position - static_cast<float>(left);
        for (int64_t h = 0; h < harmonics; ++h) {
            cumulative[h] += radians(std::max<int64_t>(0, left), h) * (1.0f - fraction) + radians(right, h) * fraction;
            phase[static_cast<size_t>(frame * harmonics + h)] = cumulative[h] * kTwoPi * static_cast<float>(scale);
        }
    }
    std::vector<float> source(static_cast<size_t>(samples));
#ifdef _OPENMP
    #pragma omp parallel for if(samples >= 4096)
#endif
    for (int64_t sample = 0; sample < samples; ++sample) {
        const float position = (static_cast<float>(sample) + 0.5f) / static_cast<float>(scale) - 0.5f;
        const auto floor = static_cast<int64_t>(std::floor(position));
        const auto left = std::clamp<int64_t>(floor, 0, frames - 1);
        const auto right = std::clamp<int64_t>(floor + 1, 0, frames - 1);
        const float fraction = position - static_cast<float>(floor);
        const float uv = f0[static_cast<size_t>(sample / scale)] > weights.config.nsf_voiced_threshold ? 1.0f : 0.0f;
        const float amplitude = uv * weights.config.nsf_sigma + (1.0f - uv) * weights.config.nsf_alpha / 3.0f;
        float sum = weights.source_linear.bias[0];
        for (int64_t h = 0; h < harmonics; ++h) {
            const float angle = phase[static_cast<size_t>(left * harmonics + h)] * (1.0f - fraction) +
                phase[static_cast<size_t>(right * harmonics + h)] * fraction;
            // Python randn_like preserves the transposed [harmonic, time] storage.
            const auto noise_index = random_policy != nullptr && !source_random_values
                ? h * samples + sample : sample * harmonics + h;
            const float value = weights.config.nsf_alpha * std::sin(angle) * uv +
                amplitude * noise[static_cast<size_t>(noise_index)];
            sum += value * weights.source_linear.weight[static_cast<size_t>(h)];
        }
        source[static_cast<size_t>(sample)] = std::tanh(sum);
    }
    return source;
}

std::vector<float> source_stft_bct(
    const HiftVocoderWeights & weights,
    const std::vector<float> & source,
    int64_t & stft_frames,
    std::vector<float> & window) {
    const audio::STFTConfig window_config{
        weights.config.istft_n_fft,
        weights.config.istft_hop,
        weights.config.istft_n_fft,
        true,
        audio::STFTPadMode::Reflect,
        audio::STFTFamily::Kokoro,
    };
    window = audio::get_cached_stft_window(window_config);
    const auto complex = audio::STFT().compute_complex(
        source,
        window,
        1,
        static_cast<int64_t>(source.size()),
        {weights.config.istft_n_fft, weights.config.istft_hop, weights.config.istft_n_fft});
    stft_frames = complex.shape[2];
    std::vector<float> stft(static_cast<size_t>(kHiftStftChannels * stft_frames), 0.0F);
    for (int64_t f = 0; f < kHiftStftBins; ++f) {
        for (int64_t t = 0; t < stft_frames; ++t) {
            const size_t complex_index = static_cast<size_t>((f * stft_frames + t) * 2);
            stft[static_cast<size_t>(f * stft_frames + t)] = complex.values[complex_index];
            stft[static_cast<size_t>((kHiftStftBins + f) * stft_frames + t)] = complex.values[complex_index + 1];
        }
    }
    return stft;
}

std::vector<float> waveform_from_post(
    const HiftVocoderWeights & weights,
    const std::vector<float> & post,
    int64_t frames,
    const std::vector<float> & window) {
    if (static_cast<int64_t>(post.size()) != kHiftStftChannels * frames) {
        throw std::runtime_error("HiFT post shape mismatch");
    }
    std::vector<float> complex_spec(static_cast<size_t>(kHiftStftBins * frames * 2), 0.0F);
    for (int64_t f = 0; f < kHiftStftBins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            float magnitude = std::exp(post[static_cast<size_t>(f * frames + t)]);
            magnitude = std::min(magnitude, 1.0e2F);
            const float phase = std::sin(post[static_cast<size_t>((kHiftStftBins + f) * frames + t)]);
            const size_t complex_index = static_cast<size_t>((f * frames + t) * 2);
            complex_spec[complex_index] = magnitude * std::cos(phase);
            complex_spec[complex_index + 1] = magnitude * std::sin(phase);
        }
    }
    const int64_t samples = weights.config.istft_hop * (frames - 1);
    auto wav = audio::ISTFT().compute(
        complex_spec,
        window,
        1,
        kHiftStftBins,
        frames,
        samples,
        {weights.config.istft_n_fft, weights.config.istft_hop, weights.config.istft_n_fft});
    for (float & sample : wav.values) {
        sample = std::clamp(sample, -weights.config.audio_limit, weights.config.audio_limit);
    }
    return wav.values;
}

class F0Runner {
public:
    F0Runner(const HiftVocoderWeights & weights, int64_t frames)
        : weights_(&weights), capacity_frames_(frames) {
        ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFT F0 graph context");
        }
        input_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, frames, weights.config.in_channels), "f0.input");
        output_ = build_f0_graph(ctx_, weights, input_);
        graph_ = ggml_new_graph_custom(ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            ggml_free(ctx_);
            ctx_ = nullptr;
            throw std::runtime_error("failed to allocate HiFT F0 graph");
        }
    }

    ~F0Runner() {
        if (weights_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
        }
    }

    bool supports(const HiftVocoderWeights & weights, int64_t frames) const {
        return weights_ == &weights && capacity_frames_ == frames;
    }

    std::vector<float> run(const std::vector<float> & mel, int64_t frames) {
        if (frames != capacity_frames_) {
            throw std::runtime_error("HiFT F0 graph frame mismatch");
        }
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HiFT F0 predictor");
        }
        auto raw = core::read_tensor_f32(output_);
        std::vector<float> f0(static_cast<size_t>(frames), 0.0F);
        for (int64_t t = 0; t < frames; ++t) {
            f0[static_cast<size_t>(t)] = raw[static_cast<size_t>(t)];
        }
        return f0;
    }

private:
    const HiftVocoderWeights * weights_ = nullptr;
    int64_t capacity_frames_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class BackendRunner {
public:
    BackendRunner(const HiftVocoderWeights & weights, int64_t frames, int64_t stft_frames)
        : weights_(&weights), capacity_frames_(frames), capacity_stft_frames_(stft_frames) {
        constexpr uint64_t kBaseBytes = 512ull * 1024ull * 1024ull;
        constexpr uint64_t kPerFrameBytes = 4ull * 1024ull * 1024ull;
        constexpr uint64_t kMaxSizeT = static_cast<uint64_t>(std::numeric_limits<size_t>::max());

        const uint64_t clamped_frames = static_cast<uint64_t>(std::max<int64_t>(frames, 1));
        if (kBaseBytes > kMaxSizeT || clamped_frames > (kMaxSizeT - kBaseBytes) / kPerFrameBytes) {
            const uint64_t requested_mib = (kBaseBytes + clamped_frames * kPerFrameBytes) / (1024ull * 1024ull);
            throw std::runtime_error(
                "HiFT vocoder requested context size (" + std::to_string(requested_mib) +
                " MiB for " + std::to_string(clamped_frames) +
                " frames) exceeds size_t addressable memory limit");
        }
        const size_t context_size = static_cast<size_t>(kBaseBytes + clamped_frames * kPerFrameBytes);
        ggml_init_params params{
            context_size,
            nullptr,
            true,
        };
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFT backend graph context");
        }
        speech_in_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, frames, weights.config.in_channels), "speech_in");
        source_in_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, stft_frames, kHiftStftChannels), "source_in");
        post_ = build_backend_graph(
            ctx_,
            weights,
            speech_in_,
            source_in_,
            weights.execution_context->uses_host_graph_plan());
        graph_ = ggml_new_graph_custom(ctx_, 65536, false);
        ggml_build_forward_expand(graph_, post_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            ggml_free(ctx_);
            ctx_ = nullptr;
            throw std::runtime_error("failed to allocate HiFT backend graph");
        }
        if (weights.config.source_mode == HiftVocoderSourceMode::SineGen2 &&
            weights.execution_context->backend_type() == core::BackendType::Cuda) {
            cuda_istft_ = std::make_unique<audio::CudaLogMagnitudePhaseISTFT>(
                audio::CudaLogMagnitudePhaseISTFTConfig{stft_frames, weights.config.istft_n_fft,
                    weights.config.istft_hop, kHiftStftChannels, weights.execution_context->config().device});
        }
    }

    ~BackendRunner() {
        if (weights_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
        }
    }

    bool supports(const HiftVocoderWeights & weights, int64_t frames, int64_t stft_frames) const {
        return weights_ == &weights && capacity_frames_ == frames && capacity_stft_frames_ == stft_frames;
    }

    std::vector<float> run(
        const std::vector<float> & mel,
        int64_t frames,
        const std::vector<float> & source_stft,
        int64_t stft_frames,
        const std::vector<float> & window) {
        if (frames != capacity_frames_ || stft_frames != capacity_stft_frames_) {
            throw std::runtime_error("HiFT backend graph frame mismatch");
        }
        ggml_backend_tensor_set(speech_in_, mel.data(), 0, mel.size() * sizeof(float));
        ggml_backend_tensor_set(source_in_, source_stft.data(), 0, source_stft.size() * sizeof(float));
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HiFT backend");
        }
        const auto post = core::read_tensor_f32(post_);
        if (!cuda_istft_) return waveform_from_post(*weights_, post, stft_frames, window);
        std::vector<float> spectrum(post.size());
        for (int64_t frame = 0; frame < stft_frames; ++frame) {
            for (int64_t frequency = 0; frequency < kHiftStftBins; ++frequency) {
                spectrum[static_cast<size_t>(frame * kHiftStftChannels + frequency)] =
                    post[static_cast<size_t>(frequency * stft_frames + frame)];
                spectrum[static_cast<size_t>(frame * kHiftStftChannels + kHiftStftBins + frequency)] =
                    std::sin(post[static_cast<size_t>((kHiftStftBins + frequency) * stft_frames + frame)]);
            }
        }
        auto reconstructed = cuda_istft_->compute(spectrum, window).audio;
        // The CUDA helper uses Vocos padding; HiFT uses centered ISTFT padding.
        const auto offset = weights_->config.istft_n_fft / 2 -
            (weights_->config.istft_n_fft - weights_->config.istft_hop) / 2;
        const auto samples = weights_->config.istft_hop * (stft_frames - 1);
        std::vector<float> audio(reconstructed.begin() + offset, reconstructed.begin() + offset + samples);
        for (auto & sample : audio) sample = std::clamp(sample, -weights_->config.audio_limit, weights_->config.audio_limit);
        return audio;
    }

private:
    const HiftVocoderWeights * weights_ = nullptr;
    int64_t capacity_frames_ = 0;
    int64_t capacity_stft_frames_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * speech_in_ = nullptr;
    ggml_tensor * source_in_ = nullptr;
    ggml_tensor * post_ = nullptr;
    std::unique_ptr<audio::CudaLogMagnitudePhaseISTFT> cuda_istft_;
};

class HiftRunner {
public:
    HiftRunner(const HiftVocoderWeights & weights, core::BackendConfig backend)
        : weights_(&weights), backend_(backend) {
        if (weights.config.source_mode == HiftVocoderSourceMode::SineGen2 &&
            weights.execution_context->backend_type() == core::BackendType::Cuda) {
            random_policy_ = sampling::resolve_torch_cuda_sampling_policy(
                core::BackendType::Cuda, backend.device, "framework.hift", "HiFT SineGen2");
        }
    }

    std::vector<float> predict_f0(const std::vector<float> & mel, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        return predict_f0_locked(mel, frames);
    }

    HiftVocoderOutput run(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed,
        uint64_t prior_noise_values,
        const std::vector<float> * source_random_values) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto f0 = predict_f0_locked(mel, frames);
        const int64_t scale = upsample_scale(weights_->config);
        std::vector<float> source;
        if (weights_->config.source_mode == HiftVocoderSourceMode::CausalSineGen2) {
            source = make_causal_sinegen2_source(*weights_, f0, frames, scale, seed, prior_noise_values, source_random_values);
        } else if (weights_->config.source_mode == HiftVocoderSourceMode::SineGen2) {
            source = make_sinegen2_source(*weights_, f0, frames, scale, seed, prior_noise_values,
                source_random_values, random_policy_ ? &*random_policy_ : nullptr);
        } else {
            const auto f0_upsampled = nearest_upsample_f0(f0, frames, scale);
            source = make_sine_source(*weights_, f0_upsampled, seed, prior_noise_values, source_random_values);
        }
        std::vector<float> window;
        int64_t stft_frames = 0;
        const auto source_stft = source_stft_bct(*weights_, source, stft_frames, window);
        if (backend_runner_ != nullptr && !backend_runner_->supports(*weights_, frames, stft_frames)) {
            backend_runner_.reset();
        }
        if (backend_runner_ == nullptr) {
            backend_runner_ = std::make_unique<BackendRunner>(*weights_, frames, stft_frames);
        }
        HiftVocoderOutput out;
        out.waveform = backend_runner_->run(mel, frames, source_stft, stft_frames, window);
        out.batch = 1;
        out.samples = static_cast<int64_t>(out.waveform.size());
        out.sample_rate = weights_->config.sampling_rate;
        return out;
    }

    void release_runtime_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        f0_runner_.reset();
        backend_runner_.reset();
    }

private:
    std::vector<float> predict_f0_locked(const std::vector<float> & mel, int64_t frames) {
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != weights_->config.in_channels * frames) {
            throw std::runtime_error("HiFT mel shape mismatch");
        }
        if (f0_runner_ != nullptr && !f0_runner_->supports(*weights_, frames)) {
            f0_runner_.reset();
        }
        if (f0_runner_ == nullptr) {
            f0_runner_ = std::make_unique<F0Runner>(*weights_, frames);
        }
        return f0_runner_->run(mel, frames);
    }

    const HiftVocoderWeights * weights_ = nullptr;
    core::BackendConfig backend_;
    std::optional<sampling::TorchCudaSamplingPolicy> random_policy_;
    std::mutex mutex_;
    std::unique_ptr<F0Runner> f0_runner_;
    std::unique_ptr<BackendRunner> backend_runner_;
};

}  // namespace

struct HiftVocoderComponent::State {
    std::unique_ptr<HiftRunner> runner;
};

HiftVocoderComponent HiftVocoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    HiftVocoderConfig config) {
    const auto source = assets::open_tensor_source(checkpoint_path);
    return load_from_tensor_source(std::move(source), std::move(backend), std::move(config));
}

HiftVocoderComponent HiftVocoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    HiftVocoderConfig config) {
    if (source == nullptr) {
        throw std::runtime_error("HiFT tensor source is missing");
    }
    validate_config(config);
    auto weights = std::make_shared<HiftVocoderWeights>();
    weights->config = std::move(config);
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.hift.weights",
        768ull * 1024ull * 1024ull);

    for (const auto & tensor : source->tensors()) {
        if (!weights->config.tensor_prefix.empty() &&
            tensor.name.rfind(weights->config.tensor_prefix, 0) != 0) {
            continue;
        }
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    const bool use_weight_norm = weights->config.weight_layout == HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    const auto conv_storage_type = use_weight_norm
        ? weights->config.weight_storage_type
        : assets::TensorStorageType::F32;
    const auto linear_storage_type = use_weight_norm
        ? weights->config.weight_storage_type
        : assets::TensorStorageType::F32;

    const std::vector<int64_t> f0_kernels = weights->config.f0_condnet_kernel_sizes.empty()
        ? std::vector<int64_t>(5, 3)
        : weights->config.f0_condnet_kernel_sizes;
    if (f0_kernels.size() != 5) {
        throw std::runtime_error("HiFT F0 condnet requires 5 kernel sizes");
    }
    for (int index = 0; index < 5; ++index) {
        const int64_t kernel = f0_kernels[static_cast<size_t>(index)];
        if (kernel <= 0) {
            throw std::runtime_error("HiFT F0 condnet kernel size must be positive");
        }
        auto conv = load_conv1d(
            *weights->store,
            *source,
            weights->config,
            "f0_predictor.condnet." + std::to_string(index * 2),
            weights->config.f0_cond_channels,
            index == 0 ? weights->config.f0_in_channels : weights->config.f0_cond_channels,
            kernel,
            1,
            weights->config.causal_convolutions ? 0 : kernel / 2,
            1,
            true,
            conv_storage_type,
            use_weight_norm);
        if (weights->config.causal_convolutions) {
            if (index == 0) {
                conv.pad_right = causal_conv1d_padding(kernel, 1);
            } else {
                conv.pad_left = causal_conv1d_padding(kernel, 1);
            }
        }
        weights->f0_predictor.condnet.push_back(std::move(conv));
    }
    weights->f0_predictor.classifier = load_linear(
        *weights->store,
        *source,
        weights->config,
        "f0_predictor.classifier",
        weights->config.f0_num_class,
        weights->config.f0_cond_channels,
        true,
        assets::TensorStorageType::F32);

    weights->conv_pre = load_conv1d(
        *weights->store,
        *source,
        weights->config,
        "conv_pre",
        weights->config.base_channels,
        weights->config.in_channels,
        weights->config.conv_pre_kernel_size,
        1,
        weights->config.causal_convolutions ? 0 : weights->config.conv_pre_kernel_size / 2,
        1,
        true,
        conv_storage_type,
        use_weight_norm);
    if (weights->config.causal_convolutions) {
        weights->conv_pre.pad_right = causal_conv1d_padding(weights->config.conv_pre_kernel_size, 1);
    }

    const int64_t upsample_count = static_cast<int64_t>(weights->config.upsample_rates.size());
    weights->ups.reserve(static_cast<size_t>(upsample_count));
    weights->up_convs.reserve(static_cast<size_t>(upsample_count));
    weights->source_downs.reserve(static_cast<size_t>(upsample_count));
    weights->source_resblocks.reserve(static_cast<size_t>(upsample_count));
    const int64_t num_kernels = static_cast<int64_t>(weights->config.resblock_kernel_sizes.size());
    weights->resblocks.reserve(static_cast<size_t>(upsample_count * num_kernels));

    std::vector<int64_t> downsample_rates = {1};
    if (upsample_count > 1) {
        for (int64_t index = upsample_count - 1; index > 0; --index) {
            downsample_rates.push_back(weights->config.upsample_rates[static_cast<size_t>(index)]);
        }
    }
    std::vector<int64_t> downsample_cum(downsample_rates.size(), 1);
    std::partial_sum(
        downsample_rates.begin(),
        downsample_rates.end(),
        downsample_cum.begin(),
        std::multiplies<int64_t>{});
    std::reverse(downsample_cum.begin(), downsample_cum.end());

    for (int64_t up_index = 0; up_index < upsample_count; ++up_index) {
        const int64_t in_channels = weights->config.base_channels / (int64_t{1} << up_index);
        const int64_t out_channels = weights->config.base_channels / (int64_t{1} << (up_index + 1));
        const int64_t kernel = weights->config.upsample_kernel_sizes[static_cast<size_t>(up_index)];
        const int64_t stride = weights->config.upsample_rates[static_cast<size_t>(up_index)];
        switch (weights->config.upsample_mode) {
            case HiftVocoderUpsampleMode::ConvTranspose1d:
                weights->ups.push_back(load_conv_transpose1d(
                    *weights->store,
                    *source,
                    weights->config,
                    "ups." + std::to_string(up_index),
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    (kernel - stride) / 2,
                    true,
                    assets::TensorStorageType::F32,
                    use_weight_norm));
                break;
            case HiftVocoderUpsampleMode::CausalConv1dNearest:
                weights->up_convs.push_back(load_conv1d(
                    *weights->store,
                    *source,
                    weights->config,
                    "ups." + std::to_string(up_index),
                    out_channels,
                    in_channels,
                    kernel,
                    1,
                    0,
                    1,
                    true,
                    conv_storage_type,
                    use_weight_norm));
                break;
            default:
                throw std::runtime_error("HiFT unsupported upsample mode");
        }

        const int64_t source_stride = downsample_cum[static_cast<size_t>(up_index)];
        const int64_t source_kernel = source_stride == 1 ? 1 : source_stride * 2;
        const int64_t source_padding = weights->config.causal_convolutions ? 0 : (source_stride == 1 ? 0 : source_stride / 2);
        auto source_down = load_conv1d(
            *weights->store,
            *source,
            weights->config,
            "source_downs." + std::to_string(up_index),
            out_channels,
            kHiftStftChannels,
            source_kernel,
            source_stride,
            source_padding,
            1,
            true,
            conv_storage_type,
            false);
        if (weights->config.causal_convolutions && source_stride > 1) {
            source_down.pad_left = source_stride - 1;
        }
        weights->source_downs.push_back(std::move(source_down));

        weights->source_resblocks.push_back(load_resblock(
            *weights->store,
            *source,
            weights->config,
            "source_resblocks." + std::to_string(up_index),
            out_channels,
            weights->config.source_resblock_kernel_sizes[static_cast<size_t>(up_index)],
            weights->config.source_resblock_dilation_sizes[static_cast<size_t>(up_index)],
            conv_storage_type,
            weights->config.causal_convolutions));

        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = up_index * num_kernels + kernel_index;
            weights->resblocks.push_back(load_resblock(
                *weights->store,
                *source,
                weights->config,
                "resblocks." + std::to_string(block_index),
                out_channels,
                weights->config.resblock_kernel_sizes[static_cast<size_t>(kernel_index)],
                weights->config.resblock_dilation_sizes[static_cast<size_t>(kernel_index)],
                conv_storage_type,
                weights->config.causal_convolutions));
        }
    }

    const int64_t post_channels = weights->config.base_channels / (int64_t{1} << upsample_count);
    weights->conv_post = load_conv1d(
        *weights->store,
        *source,
        weights->config,
        "conv_post",
        kHiftStftChannels,
        post_channels,
        7,
        1,
        weights->config.causal_convolutions ? 0 : 3,
        1,
        true,
        conv_storage_type,
        use_weight_norm);
    if (weights->config.causal_convolutions) {
        weights->conv_post.pad_left = causal_conv1d_padding(7, 1);
    }
    weights->source_linear = load_linear(
        *weights->store,
        *source,
        weights->config,
        "m_source.l_linear",
        1,
        weights->config.nb_harmonics + 1,
        true,
        linear_storage_type);

    weights->store->upload();
    source->release_storage();
    return HiftVocoderComponent(std::move(weights), backend);
}

HiftVocoderComponent::HiftVocoderComponent(
    std::shared_ptr<const HiftVocoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("HiFT component requires weights");
    }
    state_->runner = std::make_unique<HiftRunner>(*weights_, backend_);
}

const core::BackendConfig & HiftVocoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const HiftVocoderWeights> & HiftVocoderComponent::weights() const noexcept {
    return weights_;
}

int64_t HiftVocoderComponent::sample_rate() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.sampling_rate;
}

int64_t HiftVocoderComponent::num_mels() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.in_channels;
}

int64_t HiftVocoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t HiftVocoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

HiftVocoderOutput HiftVocoderComponent::synthesize(
    const std::vector<float> & mel,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    return state_->runner->run(mel, frames, seed, prior_noise_values, source_random_values);
}

std::vector<float> HiftVocoderComponent::predict_f0(const std::vector<float> & mel, int64_t frames) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    return state_->runner->predict_f0(mel, frames);
}

void HiftVocoderComponent::release_runtime_cache() const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    state_->runner->release_runtime_cache();
}

}  // namespace engine::modules

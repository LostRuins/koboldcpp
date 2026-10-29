#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

bool same_backend(const core::BackendConfig & lhs, const core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("CAMPPlus tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

std::vector<float> read_f32_tensor(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    return source.require_f32(name, expected_shape);
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue concat_along_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs,
    int logical_axis) {
    auto output_shape = lhs.shape;
    output_shape.dims[logical_axis] += rhs.shape.dims[logical_axis];
    return core::wrap_tensor(
        ggml_concat(
            ctx.ggml,
            lhs.tensor,
            rhs.tensor,
            core::logical_axis_to_ggml_axis(lhs.shape.rank, logical_axis)),
        output_shape,
        lhs.type);
}

core::TensorValue permute_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    std::array<int, core::kMaxTensorRank> axes) {
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;
    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        output_shape.dims[out_logical_axis] = input.shape.dims[in_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[static_cast<size_t>(out_ggml_axis)] =
            core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
    }
    return core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

CampplusEncoderWeights::BatchNorm1dWeights load_bn1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    bool affine = true) {
    CampplusEncoderWeights::BatchNorm1dWeights bn;
    bn.affine = affine;
    if (affine) {
        bn.weight = read_f32_tensor(source, prefix + ".weight", {channels});
        bn.bias = read_f32_tensor(source, prefix + ".bias", {channels});
    }
    bn.running_mean = read_f32_tensor(source, prefix + ".running_mean", {channels});
    bn.running_var = read_f32_tensor(source, prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0F);
    std::vector<float> shift(static_cast<size_t>(channels), 0.0F);
    for (int64_t c = 0; c < channels; ++c) {
        const size_t index = static_cast<size_t>(c);
        const float gamma = affine ? bn.weight[index] : 1.0F;
        const float beta = affine ? bn.bias[index] : 0.0F;
        const float inv = 1.0F / std::sqrt(bn.running_var[index] + 1.0e-5F);
        scale[index] = gamma * inv;
        shift[index] = beta - bn.running_mean[index] * gamma * inv;
    }
    bn.scale_tensor = store.make_f32(core::TensorShape::from_dims({1, channels, 1}), std::move(scale));
    bn.shift_tensor = store.make_f32(core::TensorShape::from_dims({1, channels, 1}), std::move(shift));
    return bn;
}

CampplusEncoderWeights::BatchNorm2dWeights load_bn2d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    CampplusEncoderWeights::BatchNorm2dWeights bn;
    bn.weight = read_f32_tensor(source, prefix + ".weight", {channels});
    bn.bias = read_f32_tensor(source, prefix + ".bias", {channels});
    bn.running_mean = read_f32_tensor(source, prefix + ".running_mean", {channels});
    bn.running_var = read_f32_tensor(source, prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0F);
    std::vector<float> shift(static_cast<size_t>(channels), 0.0F);
    for (int64_t c = 0; c < channels; ++c) {
        const size_t index = static_cast<size_t>(c);
        const float inv = 1.0F / std::sqrt(bn.running_var[index] + 1.0e-5F);
        scale[index] = bn.weight[index] * inv;
        shift[index] = bn.bias[index] - bn.running_mean[index] * bn.weight[index] * inv;
    }
    bn.scale_tensor = store.make_f32(core::TensorShape::from_dims({1, channels, 1, 1}), std::move(scale));
    bn.shift_tensor = store.make_f32(core::TensorShape::from_dims({1, channels, 1, 1}), std::move(shift));
    return bn;
}

CampplusEncoderWeights::Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    assets::TensorStorageType storage_type) {
    CampplusEncoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.weight = read_f32_tensor(source, prefix + ".weight", {out_channels, in_channels, kernel});
    const bool pointwise = kernel == 1 && stride == 1 && padding == 0 && dilation == 1;
    conv.weight_tensor = pointwise
        ? store.make_from_f32(core::TensorShape::from_dims({out_channels, in_channels}), storage_type, conv.weight)
        : store.make_from_f32(
              core::TensorShape::from_dims({out_channels, in_channels, kernel}),
              assets::TensorStorageType::F32,
              conv.weight);
    if (use_bias) {
        conv.bias = read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_from_f32(
            core::TensorShape::from_dims({out_channels}),
            assets::TensorStorageType::F32,
            conv.bias);
    }
    return conv;
}

CampplusEncoderWeights::Conv2dWeights load_conv2d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool use_bias = false) {
    CampplusEncoderWeights::Conv2dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel_h = kernel_h;
    conv.kernel_w = kernel_w;
    conv.stride_h = stride_h;
    conv.stride_w = stride_w;
    conv.padding_h = padding_h;
    conv.padding_w = padding_w;
    conv.use_bias = use_bias;
    conv.weight = read_f32_tensor(source, prefix + ".weight", {out_channels, in_channels, kernel_h, kernel_w});
    conv.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel_h, kernel_w}),
        assets::TensorStorageType::F32,
        conv.weight);
    if (use_bias) {
        conv.bias = read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_f32(core::TensorShape::from_dims({out_channels}), conv.bias);
    }
    return conv;
}

CampplusEncoderWeights::Conv1dWeights fold_bn_after_conv1d(
    core::BackendWeightStore & store,
    const CampplusEncoderWeights::Conv1dWeights & conv,
    const CampplusEncoderWeights::BatchNorm1dWeights & bn,
    assets::TensorStorageType storage_type) {
    CampplusEncoderWeights::Conv1dWeights folded = conv;
    folded.use_bias = true;
    folded.bias.assign(static_cast<size_t>(conv.out_channels), 0.0F);
    const int64_t kernel_elems = conv.in_channels * conv.kernel;
    for (int64_t oc = 0; oc < conv.out_channels; ++oc) {
        const float gamma = bn.affine ? bn.weight[static_cast<size_t>(oc)] : 1.0F;
        const float beta = bn.affine ? bn.bias[static_cast<size_t>(oc)] : 0.0F;
        const float inv = 1.0F / std::sqrt(bn.running_var[static_cast<size_t>(oc)] + 1.0e-5F);
        const float scale = gamma * inv;
        const float shift = beta - bn.running_mean[static_cast<size_t>(oc)] * scale;
        const size_t oc_offset = static_cast<size_t>(oc * kernel_elems);
        for (int64_t k = 0; k < kernel_elems; ++k) {
            folded.weight[oc_offset + static_cast<size_t>(k)] *= scale;
        }
        const float original_bias = conv.use_bias ? conv.bias[static_cast<size_t>(oc)] : 0.0F;
        folded.bias[static_cast<size_t>(oc)] = original_bias * scale + shift;
    }
    const bool pointwise = folded.kernel == 1 && folded.stride == 1 && folded.padding == 0 && folded.dilation == 1;
    folded.weight_tensor = pointwise
        ? store.make_from_f32(core::TensorShape::from_dims({folded.out_channels, folded.in_channels}), storage_type, folded.weight)
        : store.make_from_f32(
              core::TensorShape::from_dims({folded.out_channels, folded.in_channels, folded.kernel}),
              assets::TensorStorageType::F32,
              folded.weight);
    folded.bias_tensor = store.make_f32(core::TensorShape::from_dims({folded.out_channels}), folded.bias);
    return folded;
}

CampplusEncoderWeights::Conv2dWeights fold_bn_after_conv2d(
    core::BackendWeightStore & store,
    const CampplusEncoderWeights::Conv2dWeights & conv,
    const CampplusEncoderWeights::BatchNorm2dWeights & bn,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    CampplusEncoderWeights::Conv2dWeights folded = conv;
    folded.use_bias = true;
    folded.bias.assign(static_cast<size_t>(conv.out_channels), 0.0F);
    const int64_t kernel_elems = conv.in_channels * conv.kernel_h * conv.kernel_w;
    for (int64_t oc = 0; oc < conv.out_channels; ++oc) {
        const float gamma = bn.weight[static_cast<size_t>(oc)];
        const float beta = bn.bias[static_cast<size_t>(oc)];
        const float inv = 1.0F / std::sqrt(bn.running_var[static_cast<size_t>(oc)] + 1.0e-5F);
        const float scale = gamma * inv;
        const float shift = beta - bn.running_mean[static_cast<size_t>(oc)] * scale;
        const size_t oc_offset = static_cast<size_t>(oc * kernel_elems);
        for (int64_t k = 0; k < kernel_elems; ++k) {
            folded.weight[oc_offset + static_cast<size_t>(k)] *= scale;
        }
        folded.bias[static_cast<size_t>(oc)] = shift;
    }
    folded.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({folded.out_channels, folded.in_channels, folded.kernel_h, folded.kernel_w}),
        assets::TensorStorageType::F32,
        folded.weight);
    folded.bias_tensor = store.make_f32(core::TensorShape::from_dims({folded.out_channels}), folded.bias);
    return folded;
}

void validate_config(const CampplusEncoderConfig & config) {
    if (config.feat_dim != 80) {
        throw std::runtime_error("CAMPPlus framework module currently supports 80-dim fbank input");
    }
    if (config.embedding_size <= 0) {
        throw std::runtime_error("CAMPPlus framework module requires positive embedding size");
    }
    if (config.tensor_prefix.empty()) {
        throw std::runtime_error("CAMPPlus framework module requires non-empty tensor prefix");
    }
    if (config.stats_variance_floor < 0.0F) {
        throw std::runtime_error("CAMPPlus framework module requires non-negative stats variance floor");
    }
}

class CampplusBackendRunner {
public:
    CampplusBackendRunner(
        int64_t frames,
        const CampplusEncoderWeights & weights)
        : execution_context_(*weights.execution_context) {
        ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for CAMPPlus graph");
        }
        core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "framework.campplus";
        input_tensor_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 80, frames}));

        const auto conv2d = [&](const core::TensorValue & input,
                                int64_t height,
                                int64_t width,
                                const CampplusEncoderWeights::Conv2dWeights & conv) {
            const int64_t out_h = (height + 2 * conv.padding_h - conv.kernel_h) / conv.stride_h + 1;
            const int64_t out_w = (width + 2 * conv.padding_w - conv.kernel_w) / conv.stride_w + 1;
            auto output = core::wrap_tensor(
                ggml_conv_2d(
                    ctx.ggml,
                    conv.weight_tensor.tensor,
                    contiguous(ctx, input).tensor,
                    static_cast<int>(conv.stride_w),
                    static_cast<int>(conv.stride_h),
                    static_cast<int>(conv.padding_w),
                    static_cast<int>(conv.padding_h),
                    1,
                    1),
                core::TensorShape::from_dims({1, conv.out_channels, out_h, out_w}),
                GGML_TYPE_F32);
            if (conv.use_bias) {
                const auto bias = core::reshape_tensor(
                    ctx,
                    conv.bias_tensor,
                    core::TensorShape::from_dims({1, conv.out_channels, 1, 1}));
                output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, bias.tensor), output.shape, GGML_TYPE_F32);
            }
            return output;
        };

        const auto conv1d = [&](const core::TensorValue & input,
                                int64_t in_channels,
                                int64_t time_steps,
                                const CampplusEncoderWeights::Conv1dWeights & conv) {
            if (conv.kernel == 1 && conv.stride == 1 && conv.padding == 0 && conv.dilation == 1) {
                auto weight = core::reshape_tensor(
                    ctx,
                    conv.weight_tensor,
                    core::TensorShape::from_dims({conv.out_channels, conv.in_channels}));
                auto btc = permute_tensor(ctx, input, {0, 2, 1});
                auto flat = core::reshape_tensor(
                    ctx,
                    contiguous(ctx, btc),
                    core::TensorShape::from_dims({time_steps, in_channels}));
                auto projected = core::wrap_tensor(
                    ggml_mul_mat(ctx.ggml, weight.tensor, flat.tensor),
                    core::TensorShape::from_dims({time_steps, conv.out_channels}),
                    GGML_TYPE_F32);
                if (conv.use_bias) {
                    auto bias = core::reshape_tensor(
                        ctx,
                        conv.bias_tensor,
                        core::TensorShape::from_dims({1, conv.out_channels}));
                    projected = core::wrap_tensor(
                        ggml_add(ctx.ggml, projected.tensor, bias.tensor),
                        projected.shape,
                        GGML_TYPE_F32);
                }
                auto projected_btc = core::reshape_tensor(
                    ctx,
                    projected,
                    core::TensorShape::from_dims({1, time_steps, conv.out_channels}));
                return contiguous(ctx, permute_tensor(ctx, projected_btc, {0, 2, 1}));
            }
            const int64_t out_frames =
                (time_steps + 2 * conv.padding - conv.dilation * (conv.kernel - 1) - 1) / conv.stride + 1;
            auto output = core::wrap_tensor(
                ggml_conv_1d(
                    ctx.ggml,
                    conv.weight_tensor.tensor,
                    contiguous(ctx, input).tensor,
                    static_cast<int>(conv.stride),
                    static_cast<int>(conv.padding),
                    static_cast<int>(conv.dilation)),
                core::TensorShape::from_dims({1, conv.out_channels, out_frames}),
                GGML_TYPE_F32);
            if (conv.use_bias) {
                auto bias = core::reshape_tensor(
                    ctx,
                    conv.bias_tensor,
                    core::TensorShape::from_dims({1, conv.out_channels, 1}));
                output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, bias.tensor), output.shape, GGML_TYPE_F32);
            }
            return output;
        };

        const auto batch_norm_1d = [&](const core::TensorValue & input,
                                       const CampplusEncoderWeights::BatchNorm1dWeights & bn) {
            auto scaled = core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, bn.scale_tensor.tensor), input.shape, GGML_TYPE_F32);
            return core::wrap_tensor(ggml_add(ctx.ggml, scaled.tensor, bn.shift_tensor.tensor), input.shape, GGML_TYPE_F32);
        };

        const auto relu = [&](const core::TensorValue & input) {
            return core::wrap_tensor(ggml_relu(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
        };

        const auto as_bct = [&](const core::TensorValue & input4d, int64_t channels, int64_t time_steps) {
            return core::reshape_tensor(ctx, contiguous(ctx, input4d), core::TensorShape::from_dims({1, channels, time_steps}));
        };

        const auto as_bchw = [&](const core::TensorValue & input3d, int64_t channels, int64_t time_steps) {
            return core::reshape_tensor(ctx, input3d, core::TensorShape::from_dims({1, channels, 1, time_steps}));
        };

        const auto avg_pool_repeat = [&](const core::TensorValue & input,
                                         int64_t channels,
                                         int64_t time_steps,
                                         int64_t seg_len) {
            if (weights.config.normalize_partial_segment_by_full_length) {
                std::optional<core::TensorValue> expanded;
                for (int64_t begin = 0;
                     begin < time_steps;
                     begin += seg_len) {
                    const int64_t length =
                        std::min<int64_t>(seg_len, time_steps - begin);
                    const auto segment =
                        SliceModule({2, begin, length}).build(ctx, input);
                    const auto segment_contiguous =
                        contiguous(ctx, segment);
                    auto mean = core::wrap_tensor(
                        ggml_mean(ctx.ggml, segment_contiguous.tensor),
                        core::TensorShape::from_dims({1, channels, 1}),
                        GGML_TYPE_F32);
                    if (length != seg_len) {
                        mean = core::wrap_tensor(
                            ggml_scale(
                                ctx.ggml,
                                mean.tensor,
                                static_cast<float>(length) /
                                    static_cast<float>(seg_len)),
                            mean.shape,
                            GGML_TYPE_F32);
                    }
                    const auto repeated = core::wrap_tensor(
                        ggml_repeat(
                            ctx.ggml,
                            mean.tensor,
                            segment_contiguous.tensor),
                        segment.shape,
                        GGML_TYPE_F32);
                    expanded = expanded.has_value()
                        ? concat_along_axis(
                            ctx, *expanded, repeated, 2)
                        : repeated;
                }
                return *expanded;
            }

            // Established shared behavior used by Chatterbox, IndexTTS2, and
            // Seed-VC. Keep this graph unchanged unless a caller explicitly
            // selects the GLM-TTS partial-segment normalization above.
            const int64_t seg_frames = (time_steps + seg_len - 1) / seg_len;
            const int64_t padded_time = seg_frames * seg_len;
            auto input4d = as_bchw(input, channels, time_steps);
            auto padded = core::wrap_tensor(
                ggml_pad(ctx.ggml, input4d.tensor, static_cast<int>(padded_time - time_steps), 0, 0, 0),
                core::TensorShape::from_dims({1, channels, 1, padded_time}),
                GGML_TYPE_F32);
            auto pooled = core::wrap_tensor(
                ggml_pool_2d(
                    ctx.ggml,
                    padded.tensor,
                    GGML_OP_POOL_AVG,
                    static_cast<int>(seg_len),
                    1,
                    static_cast<int>(seg_len),
                    1,
                    0,
                    0),
                core::TensorShape::from_dims({1, channels, 1, seg_frames}),
                GGML_TYPE_F32);
            std::vector<float> correction_values(static_cast<size_t>(seg_frames), 1.0F);
            for (int64_t seg = 0; seg < seg_frames; ++seg) {
                const int64_t repeat_len = std::min<int64_t>(seg_len, time_steps - seg * seg_len);
                correction_values[static_cast<size_t>(seg)] =
                    static_cast<float>(seg_len) / static_cast<float>(repeat_len);
            }
            auto correction = writer_.make_f32_tensor(
                ctx,
                core::TensorShape::from_dims({1, 1, 1, seg_frames}),
                correction_values);
            auto ratio = core::wrap_tensor(ggml_mul(ctx.ggml, pooled.tensor, correction.tensor), pooled.shape, GGML_TYPE_F32);
            auto expanded_padded = core::wrap_tensor(
                ggml_interpolate(
                    ctx.ggml,
                    contiguous(ctx, ratio).tensor,
                    padded_time,
                    1,
                    channels,
                    1,
                    GGML_SCALE_MODE_NEAREST),
                core::TensorShape::from_dims({1, channels, 1, padded_time}),
                GGML_TYPE_F32);
            if (padded_time == time_steps) {
                return as_bct(expanded_padded, channels, time_steps);
            }
            auto * trimmed_view = ggml_view_4d(
                ctx.ggml,
                expanded_padded.tensor,
                time_steps,
                1,
                channels,
                1,
                expanded_padded.tensor->nb[1],
                expanded_padded.tensor->nb[2],
                expanded_padded.tensor->nb[3],
                0);
            auto trimmed = core::wrap_tensor(
                trimmed_view,
                core::TensorShape::from_dims({1, channels, 1, time_steps}),
                GGML_TYPE_F32);
            return as_bct(trimmed, channels, time_steps);
        };

        const auto global_avg = [&](const core::TensorValue & input, int64_t channels, int64_t time_steps) {
            (void) time_steps;
            return core::wrap_tensor(ggml_mean(ctx.ggml, input.tensor), core::TensorShape::from_dims({1, channels, 1}), GGML_TYPE_F32);
        };

        const auto stats_pool = [&](const core::TensorValue & input, int64_t channels, int64_t time_steps) {
            auto mean = global_avg(input, channels, time_steps);
            auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, mean.tensor), input.shape, GGML_TYPE_F32);
            auto squared = core::wrap_tensor(ggml_mul(ctx.ggml, centered.tensor, centered.tensor), input.shape, GGML_TYPE_F32);
            auto var = global_avg(squared, channels, time_steps);
            if (time_steps > 1) {
                auto correction = writer_.make_f32_tensor(
                    ctx,
                    core::TensorShape::from_dims({1, 1, 1}),
                    std::vector<float>{static_cast<float>(time_steps) / static_cast<float>(time_steps - 1)});
                var = core::wrap_tensor(ggml_mul(ctx.ggml, var.tensor, correction.tensor), var.shape, GGML_TYPE_F32);
            }
            if (weights.config.stats_variance_floor > 0.0F) {
                var = core::wrap_tensor(
                    ggml_clamp(ctx.ggml, var.tensor, weights.config.stats_variance_floor, INFINITY),
                    var.shape,
                    GGML_TYPE_F32);
            }
            auto stddev = core::wrap_tensor(ggml_sqrt(ctx.ggml, var.tensor), var.shape, GGML_TYPE_F32);
            return concat_along_axis(ctx, mean, stddev, 1);
        };

        auto x4 = input_tensor_;
        x4 = relu(conv2d(x4, 80, frames, weights.head_conv1_folded));
        auto run_res_block = [&](const core::TensorValue & input,
                                 int64_t height,
                                 int64_t width,
                                 const CampplusEncoderWeights::BasicResBlockWeights & block) {
            const int64_t out_h = (height + 2 * block.conv1_folded.padding_h - block.conv1_folded.kernel_h) /
                    block.conv1_folded.stride_h + 1;
            auto main = relu(conv2d(input, height, width, block.conv1_folded));
            main = conv2d(main, out_h, width, block.conv2_folded);
            auto shortcut = input;
            if (block.use_shortcut) {
                shortcut = conv2d(input, height, width, block.shortcut_conv_folded);
            }
            return relu(core::wrap_tensor(ggml_add(ctx.ggml, main.tensor, shortcut.tensor), main.shape, GGML_TYPE_F32));
        };

        int64_t height = 80;
        x4 = run_res_block(x4, height, frames, weights.head_layer1[0]);
        height = (height + 1) / 2;
        x4 = run_res_block(x4, height, frames, weights.head_layer1[1]);
        x4 = run_res_block(x4, height, frames, weights.head_layer2[0]);
        height = (height + 1) / 2;
        x4 = run_res_block(x4, height, frames, weights.head_layer2[1]);
        x4 = relu(conv2d(x4, height, frames, weights.head_conv2_folded));
        height = (height + 1) / 2;
        int64_t channels = 32 * height;
        int64_t time_steps = frames;
        auto x = as_bct(x4, channels, time_steps);
        x = relu(conv1d(x, channels, time_steps, weights.tdnn_linear_folded));
        channels = 128;
        time_steps = (time_steps + 1) / 2;
        const int block_layers[3] = {12, 24, 16};
        const bool fused = weights.config.weight_layout == CampplusEncoderWeightLayout::Fused;
        for (int block_index = 0; block_index < 3; ++block_index) {
            for (int layer_index = 0; layer_index < block_layers[block_index]; ++layer_index) {
                const auto & layer = weights.blocks[static_cast<size_t>(block_index)].layers[static_cast<size_t>(layer_index)];
                auto h = relu(batch_norm_1d(x, layer.nonlinear1_bn));
                h = conv1d(h, channels, time_steps, layer.linear1);
                h = fused ? relu(h) : relu(batch_norm_1d(h, layer.nonlinear2_bn));
                auto local = conv1d(h, 128, time_steps, layer.cam_layer.linear_local);
                auto avg = avg_pool_repeat(h, 128, time_steps, 100);
                auto global = global_avg(h, 128, time_steps);
                auto context = core::wrap_tensor(ggml_add(ctx.ggml, avg.tensor, global.tensor), avg.shape, GGML_TYPE_F32);
                auto gate = relu(conv1d(context, 128, time_steps, layer.cam_layer.linear1));
                auto sigmoid = core::wrap_tensor(
                    ggml_sigmoid(ctx.ggml, conv1d(gate, 64, time_steps, layer.cam_layer.linear2).tensor),
                    core::TensorShape::from_dims({1, 32, time_steps}),
                    GGML_TYPE_F32);
                auto out = core::wrap_tensor(ggml_mul(ctx.ggml, local.tensor, sigmoid.tensor), local.shape, GGML_TYPE_F32);
                x = concat_along_axis(ctx, x, out, 1);
                channels += 32;
            }
            x = relu(batch_norm_1d(x, weights.transits[static_cast<size_t>(block_index)].nonlinear_bn));
            x = conv1d(x, channels, time_steps, weights.transits[static_cast<size_t>(block_index)].linear);
            channels /= 2;
        }

        x = fused ? relu(x) : relu(batch_norm_1d(x, weights.out_nonlinear_bn));
        auto pooled = stats_pool(x, channels, time_steps);
        output_tensor_ = conv1d(pooled, channels * 2, 1, weights.dense_linear_folded);

        graph_ = ggml_new_graph_custom(ggml_, 65536, false);
        ggml_build_forward_expand(graph_, output_tensor_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            ggml_free(ggml_);
            ggml_ = nullptr;
            throw std::runtime_error("failed to allocate backend tensors for CAMPPlus graph");
        }
        writer_.flush();
    }

    ~CampplusBackendRunner() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_context_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    std::vector<float> run(const std::vector<float> & input) {
        std::lock_guard<std::mutex> lock(run_mutex_);
        core::write_tensor_f32(input_tensor_, input);
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for CAMPPlus graph");
        }
        return core::read_tensor_f32(output_tensor_.tensor);
    }

private:
    core::ExecutionContext & execution_context_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_tensor_;
    core::TensorValue output_tensor_;
    core::DeferredTensorWriter writer_;
    std::mutex run_mutex_;
};

void count_source_tensors(const assets::TensorSource & source, CampplusEncoderWeights & weights) {
    for (const auto & tensor : source.tensors()) {
        weights.parameter_count += tensor_elements(tensor.shape);
        ++weights.loaded_tensor_count;
    }
}

std::shared_ptr<const CampplusEncoderWeights> load_typed_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    CampplusEncoderConfig config) {
    validate_config(config);
    if (source == nullptr) {
        throw std::runtime_error("CampPlus encoder requires tensor source");
    }
    auto weights = std::make_shared<CampplusEncoderWeights>();
    weights->config = config;
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.campplus.weights",
        512ull * 1024ull * 1024ull);
    count_source_tensors(*source, *weights);

    const std::string base = weights->config.tensor_prefix;
    auto tensor_name = [&](const std::string & suffix) {
        return base + "." + suffix;
    };
    const bool fused = weights->config.weight_layout == CampplusEncoderWeightLayout::Fused;
    auto has_bias = [&](const std::string & prefix) {
        return source->has_tensor(prefix + ".bias");
    };

    auto load_res_block = [&](const std::string & prefix, bool use_shortcut, int64_t stride_h) {
        CampplusEncoderWeights::BasicResBlockWeights block;
        if (fused) {
            block.conv1_folded = load_conv2d(*weights->store, *source, prefix + ".conv1", 32, 32, 3, 3, stride_h, 1, 1, 1, true);
            block.conv2_folded = load_conv2d(*weights->store, *source, prefix + ".conv2", 32, 32, 3, 3, 1, 1, 1, 1, true);
        } else {
            auto conv1 = load_conv2d(*weights->store, *source, prefix + ".conv1", 32, 32, 3, 3, stride_h, 1, 1, 1);
            auto bn1 = load_bn2d(*weights->store, *source, prefix + ".bn1", 32);
            block.conv1_folded = fold_bn_after_conv2d(*weights->store, conv1, bn1, weights->config.weight_storage_type);
            auto conv2 = load_conv2d(*weights->store, *source, prefix + ".conv2", 32, 32, 3, 3, 1, 1, 1, 1);
            auto bn2 = load_bn2d(*weights->store, *source, prefix + ".bn2", 32);
            block.conv2_folded = fold_bn_after_conv2d(*weights->store, conv2, bn2, weights->config.weight_storage_type);
        }
        block.use_shortcut = use_shortcut;
        if (use_shortcut) {
            if (fused) {
                block.shortcut_conv_folded =
                    load_conv2d(*weights->store, *source, prefix + ".shortcut.0", 32, 32, 1, 1, stride_h, 1, 0, 0, true);
            } else {
                auto shortcut_conv = load_conv2d(*weights->store, *source, prefix + ".shortcut.0", 32, 32, 1, 1, stride_h, 1, 0, 0);
                auto shortcut_bn = load_bn2d(*weights->store, *source, prefix + ".shortcut.1", 32);
                block.shortcut_conv_folded =
                    fold_bn_after_conv2d(*weights->store, shortcut_conv, shortcut_bn, weights->config.weight_storage_type);
            }
        }
        return block;
    };

    if (fused) {
        weights->head_conv1_folded = load_conv2d(*weights->store, *source, tensor_name("head.conv1"), 32, 1, 3, 3, 1, 1, 1, 1, true);
        weights->head_conv2_folded = load_conv2d(*weights->store, *source, tensor_name("head.conv2"), 32, 32, 3, 3, 2, 1, 1, 1, true);
    } else {
        auto head_conv1 = load_conv2d(*weights->store, *source, tensor_name("head.conv1"), 32, 1, 3, 3, 1, 1, 1, 1);
        auto head_bn1 = load_bn2d(*weights->store, *source, tensor_name("head.bn1"), 32);
        weights->head_conv1_folded = fold_bn_after_conv2d(*weights->store, head_conv1, head_bn1, weights->config.weight_storage_type);
        auto head_conv2 = load_conv2d(*weights->store, *source, tensor_name("head.conv2"), 32, 32, 3, 3, 2, 1, 1, 1);
        auto head_bn2 = load_bn2d(*weights->store, *source, tensor_name("head.bn2"), 32);
        weights->head_conv2_folded = fold_bn_after_conv2d(*weights->store, head_conv2, head_bn2, weights->config.weight_storage_type);
    }
    weights->head_layer1 = {
        load_res_block(tensor_name("head.layer1.0"), true, 2),
        load_res_block(tensor_name("head.layer1.1"), false, 1),
    };
    weights->head_layer2 = {
        load_res_block(tensor_name("head.layer2.0"), true, 2),
        load_res_block(tensor_name("head.layer2.1"), false, 1),
    };

    auto tdnn_linear = load_conv1d(
        *weights->store,
        *source,
        tensor_name("xvector.tdnn.linear"),
        128,
        320,
        5,
        2,
        2,
        1,
        fused && has_bias(tensor_name("xvector.tdnn.linear")),
        weights->config.weight_storage_type);
    if (fused) {
        weights->tdnn_linear_folded = std::move(tdnn_linear);
    } else {
        auto tdnn_bn = load_bn1d(*weights->store, *source, tensor_name("xvector.tdnn.nonlinear.batchnorm"), 128);
        weights->tdnn_linear_folded = fold_bn_after_conv1d(*weights->store, tdnn_linear, tdnn_bn, weights->config.weight_storage_type);
    }

    const int block_layers[3] = {12, 24, 16};
    const int block_input_channels[3] = {128, 256, 512};
    weights->blocks.resize(3);
    weights->transits.resize(3);
    for (int block_index = 0; block_index < 3; ++block_index) {
        auto & block = weights->blocks[static_cast<size_t>(block_index)];
        for (int layer = 1; layer <= block_layers[block_index]; ++layer) {
            const int in_channels = block_input_channels[block_index] + (layer - 1) * 32;
            const std::string prefix =
                tensor_name("xvector.block" + std::to_string(block_index + 1) + ".tdnnd" + std::to_string(layer));
            CampplusEncoderWeights::CAMDenseTDNNLayerWeights layer_weights;
            layer_weights.nonlinear1_bn = load_bn1d(*weights->store, *source, prefix + ".nonlinear1.batchnorm", in_channels);
            layer_weights.linear1 = load_conv1d(
                *weights->store,
                *source,
                prefix + ".linear1",
                128,
                in_channels,
                1,
                1,
                0,
                1,
                fused && has_bias(prefix + ".linear1"),
                weights->config.weight_storage_type);
            if (!fused) {
                layer_weights.nonlinear2_bn = load_bn1d(*weights->store, *source, prefix + ".nonlinear2.batchnorm", 128);
            }
            layer_weights.cam_layer.linear_local =
                load_conv1d(
                    *weights->store,
                    *source,
                    prefix + ".cam_layer.linear_local",
                    32,
                    128,
                    3,
                    1,
                    block_index == 0 ? 1 : 2,
                    block_index == 0 ? 1 : 2,
                    false,
                    weights->config.weight_storage_type);
            layer_weights.cam_layer.linear1 =
                load_conv1d(
                    *weights->store,
                    *source,
                    prefix + ".cam_layer.linear1",
                    64,
                    128,
                    1,
                    1,
                    0,
                    1,
                    true,
                    weights->config.weight_storage_type);
            layer_weights.cam_layer.linear2 =
                load_conv1d(
                    *weights->store,
                    *source,
                    prefix + ".cam_layer.linear2",
                    32,
                    64,
                    1,
                    1,
                    0,
                    1,
                    true,
                    weights->config.weight_storage_type);
            block.layers.push_back(std::move(layer_weights));
        }
        const int transit_in = block_input_channels[block_index] + block_layers[block_index] * 32;
        const int transit_out = transit_in / 2;
        const std::string transit_prefix = tensor_name("xvector.transit" + std::to_string(block_index + 1));
        weights->transits[static_cast<size_t>(block_index)].nonlinear_bn =
            load_bn1d(*weights->store, *source, transit_prefix + ".nonlinear.batchnorm", transit_in);
        weights->transits[static_cast<size_t>(block_index)].linear =
            load_conv1d(
                *weights->store,
                *source,
                transit_prefix + ".linear",
                transit_out,
                transit_in,
                1,
                1,
                0,
                1,
                fused && has_bias(transit_prefix + ".linear"),
                weights->config.weight_storage_type);
    }

    if (!fused) {
        weights->out_nonlinear_bn = load_bn1d(*weights->store, *source, tensor_name("xvector.out_nonlinear.batchnorm"), 512);
    }
    auto dense_linear = load_conv1d(
        *weights->store,
        *source,
        tensor_name("xvector.dense.linear"),
        weights->config.embedding_size,
        1024,
        1,
        1,
        0,
        1,
        false,
        weights->config.weight_storage_type);
    auto dense_bn = load_bn1d(
        *weights->store,
        *source,
        tensor_name("xvector.dense.nonlinear.batchnorm"),
        weights->config.embedding_size,
        false);
    weights->dense_linear_folded = fold_bn_after_conv1d(*weights->store, dense_linear, dense_bn, weights->config.weight_storage_type);
    weights->store->upload();
    source->release_storage();
    return weights;
}

}  // namespace

struct CampplusEncoderComponent::State {
    std::mutex mutex;
    std::shared_ptr<CampplusBackendRunner> runner;
    const CampplusEncoderWeights * weights = nullptr;
    int64_t frames = 0;
    core::BackendConfig backend;
};

CampplusEncoderComponent CampplusEncoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    CampplusEncoderConfig config) {
    return load_from_tensor_source(assets::open_tensor_source(checkpoint_path), std::move(backend), std::move(config));
}

CampplusEncoderComponent CampplusEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    CampplusEncoderConfig config) {
    return CampplusEncoderComponent(load_typed_weights(std::move(source), backend, config), backend);
}

CampplusEncoderComponent::CampplusEncoderComponent(
    std::shared_ptr<const CampplusEncoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)), backend_(backend), state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("CAMPPlus component requires weights");
    }
}

CampplusEncoderComponent::~CampplusEncoderComponent() = default;
CampplusEncoderComponent::CampplusEncoderComponent(CampplusEncoderComponent &&) noexcept = default;
CampplusEncoderComponent & CampplusEncoderComponent::operator=(CampplusEncoderComponent &&) noexcept = default;

const core::BackendConfig & CampplusEncoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const CampplusEncoderWeights> & CampplusEncoderComponent::weights() const noexcept {
    return weights_;
}

int64_t CampplusEncoderComponent::embedding_size() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.embedding_size;
}

int64_t CampplusEncoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t CampplusEncoderComponent::skipped_buffer_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->skipped_buffer_count;
}

int64_t CampplusEncoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

CampplusEncoderOutputs CampplusEncoderComponent::embed_from_features(
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims) const {
    if (weights_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("CAMPPlus component is not initialized");
    }
    if (dims != weights_->config.feat_dim) {
        throw std::runtime_error("CAMPPlus feature dimension mismatch");
    }
    if (frames <= 0 || static_cast<int64_t>(features.size()) != frames * dims) {
        throw std::runtime_error("CAMPPlus feature shape mismatch");
    }
    if (!same_backend(weights_->execution_context->config(), backend_)) {
        throw std::runtime_error("CAMPPlus backend does not match uploaded weight backend");
    }
    std::vector<float> graph_input(static_cast<size_t>(dims * frames), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t d = 0; d < dims; ++d) {
            graph_input[static_cast<size_t>(d * frames + t)] = features[static_cast<size_t>(t * dims + d)];
        }
    }

    std::shared_ptr<CampplusBackendRunner> runner;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->runner ||
            state_->weights != weights_.get() ||
            state_->frames != frames ||
            !same_backend(state_->backend, backend_)) {
            state_->runner = std::make_shared<CampplusBackendRunner>(frames, *weights_);
            state_->weights = weights_.get();
            state_->frames = frames;
            state_->backend = backend_;
        }
        runner = state_->runner;
    }
    auto dense = runner->run(graph_input);
    CampplusEncoderOutputs outputs;
    outputs.embedding_size = weights_->config.embedding_size;
    outputs.embedding.assign(dense.begin(), dense.begin() + outputs.embedding_size);
    return outputs;
}

void CampplusEncoderComponent::release_runtime_graph() {
    if (state_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->runner.reset();
    state_->weights = nullptr;
    state_->frames = 0;
    state_->backend = {};
}

}  // namespace engine::modules

#include "engine/models/vibevoice/tokenizer_audio.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::vibevoice {
namespace {

namespace binding = modules::binding;

constexpr int kTokenizerSampleRate = 24000;
constexpr int64_t kTokenizerKernelSize = 7;
constexpr int64_t kTokenizerLastKernelSize = 7;
constexpr int64_t kTokenizerFfnExpansion = 4;
constexpr float kTokenizerConvTransposeTrimRightRatio = 1.0F;

std::vector<float> normalize_vibevoice_prompt_waveform(
    std::vector<float> waveform,
    const VibeVoiceProcessorConfig & processor) {
    if (!processor.db_normalize || waveform.empty()) {
        return waveform;
    }
    double energy = 0.0;
    for (const float sample : waveform) {
        energy += static_cast<double>(sample) * static_cast<double>(sample);
    }
    const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(waveform.size())));
    const float target = std::pow(10.0F, processor.audio_processor.target_db_fs / 20.0F);
    const float scalar = target / (rms + processor.audio_processor.eps);
    float max_abs = 0.0F;
    for (float & sample : waveform) {
        sample *= scalar;
        max_abs = std::max(max_abs, std::abs(sample));
    }
    if (max_abs > 1.0F) {
        const float clip_scalar = max_abs + processor.audio_processor.eps;
        for (float & sample : waveform) {
            sample /= clip_scalar;
        }
    }
    return waveform;
}

std::vector<float> convert_vibevoice_audio_to_mono_resampled(
    const runtime::AudioBuffer & audio,
    int target_sample_rate_hz,
    const char * warning_context) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate == target_sample_rate_hz || mono.empty()) {
        return mono;
    }
    engine::audio::SoxrResampleOptions options;
    options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
    options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
    options.output_padding = 256;
    options.reject_empty_output = true;
    options.warning_context = warning_context;
    options.fallback_description = "linear resampling";
    return engine::audio::resample_mono_soxr_or_linear(mono, audio.sample_rate, target_sample_rate_hz, options);
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<int64_t> parse_depths(const std::string & value, const char * label) {
    if (value.empty()) {
        throw std::runtime_error(std::string("VibeVoice tokenizer missing ") + label);
    }
    std::vector<int64_t> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, '-')) {
        if (item.empty()) {
            throw std::runtime_error(std::string("VibeVoice tokenizer invalid ") + label);
        }
        size_t pos = 0;
        const auto depth = std::stoll(item, &pos);
        if (pos != item.size() || depth <= 0) {
            throw std::runtime_error(std::string("VibeVoice tokenizer invalid ") + label);
        }
        out.push_back(depth);
    }
    if (out.empty()) {
        throw std::runtime_error(std::string("VibeVoice tokenizer empty ") + label);
    }
    return out;
}

std::vector<int64_t> decoder_depths(const VibeVoiceTokenizerConfig & config) {
    if (!config.decoder_depths.empty()) {
        return parse_depths(config.decoder_depths, "decoder_depths");
    }
    auto depths = parse_depths(config.encoder_depths, "encoder_depths");
    std::reverse(depths.begin(), depths.end());
    return depths;
}

void validate_tokenizer_config(const VibeVoiceTokenizerConfig & config, bool require_decoder) {
    if (!config.causal) {
        throw std::runtime_error("VibeVoice tokenizer loader expects causal tokenizers");
    }
    if (config.channels <= 0 || config.vae_dim <= 0 || config.encoder_n_filters <= 0) {
        throw std::runtime_error("VibeVoice tokenizer config dimensions must be positive");
    }
    if (require_decoder && config.decoder_n_filters <= 0) {
        throw std::runtime_error("VibeVoice acoustic tokenizer decoder_n_filters must be positive");
    }
    if (config.mixer_layer != "depthwise_conv") {
        throw std::runtime_error("VibeVoice tokenizer loader expects depthwise_conv mixer");
    }
    if (config.pad_mode != "constant") {
        throw std::runtime_error("VibeVoice tokenizer loader expects constant tokenizer padding");
    }
    if (config.conv_norm != "none") {
        throw std::runtime_error("VibeVoice tokenizer loader expects conv_norm none");
    }
    if (config.layernorm != "RMSNorm") {
        throw std::runtime_error("VibeVoice tokenizer loader expects RMSNorm blocks");
    }
    if (!config.layernorm_elementwise_affine) {
        throw std::runtime_error("VibeVoice tokenizer loader expects affine RMSNorm");
    }
    if (!config.conv_bias) {
        throw std::runtime_error("VibeVoice tokenizer loader expects convolution bias");
    }
    if (!(config.layer_scale_init_value > 0.0F)) {
        throw std::runtime_error("VibeVoice tokenizer loader expects layer scale tensors");
    }
    if (config.encoder_ratios.empty()) {
        throw std::runtime_error("VibeVoice tokenizer encoder ratios must not be empty");
    }
}

modules::Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    assets::TensorStorageType weight_storage_type) {
    modules::Conv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel_size});
    weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    return weights;
}

modules::DepthwiseConv1dWeights load_depthwise_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel_size,
    assets::TensorStorageType weight_storage_type) {
    modules::DepthwiseConv1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", weight_storage_type, {channels, 1, kernel_size});
    weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {channels});
    return weights;
}

modules::ConvTranspose1dWeights load_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    assets::TensorStorageType weight_storage_type) {
    modules::ConvTranspose1dWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", weight_storage_type, {in_channels, out_channels, kernel_size});
    weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    return weights;
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_features,
    int64_t out_features,
    assets::TensorStorageType weight_storage_type) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    weights.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_features});
    return weights;
}

VibeVoiceTokenizerBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType weight_storage_type) {
    if (channels <= 0) {
        throw std::runtime_error("VibeVoice tokenizer block channels must be positive");
    }
    VibeVoiceTokenizerBlockWeights weights;
    weights.channels = channels;
    weights.norm = source.require_f32_tensor(prefix + ".norm.weight", {channels});
    weights.mixer = load_depthwise_conv1d(
        store,
        source,
        prefix + ".mixer.conv.conv.conv",
        channels,
        kTokenizerKernelSize,
        weight_storage_type);
    weights.gamma = source.require_f32_tensor(prefix + ".gamma", {channels});
    weights.ffn_norm = source.require_f32_tensor(prefix + ".ffn_norm.weight", {channels});
    weights.ffn_linear1 = load_linear(
        store,
        source,
        prefix + ".ffn.linear1",
        channels,
        kTokenizerFfnExpansion * channels,
        weight_storage_type);
    weights.ffn_linear2 = load_linear(
        store,
        source,
        prefix + ".ffn.linear2",
        kTokenizerFfnExpansion * channels,
        channels,
        weight_storage_type);
    weights.ffn_gamma = source.require_f32_tensor(prefix + ".ffn_gamma", {channels});
    return weights;
}

VibeVoiceTokenizerEncoderWeights load_encoder(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const VibeVoiceTokenizerConfig & config,
    assets::TensorStorageType weight_storage_type) {
    validate_tokenizer_config(config, false);
    auto depths = parse_depths(config.encoder_depths, "encoder_depths");
    auto ratios = config.encoder_ratios;
    std::reverse(ratios.begin(), ratios.end());
    if (depths.size() != ratios.size() + 1) {
        throw std::runtime_error("VibeVoice tokenizer encoder depths/ratios mismatch");
    }

    VibeVoiceTokenizerEncoderWeights weights;
    weights.downsample_layers.reserve(depths.size());
    weights.downsample_layers.push_back(load_conv1d(
        store,
        source,
        prefix + ".downsample_layers.0.0.conv.conv",
        config.channels,
        config.encoder_n_filters,
        kTokenizerKernelSize,
        weight_storage_type));
    for (size_t i = 0; i < ratios.size(); ++i) {
        const int64_t in_ch = config.encoder_n_filters * (int64_t{1} << static_cast<int64_t>(i));
        const int64_t out_ch = config.encoder_n_filters * (int64_t{1} << static_cast<int64_t>(i + 1));
        weights.downsample_layers.push_back(load_conv1d(
            store,
            source,
            prefix + ".downsample_layers." + std::to_string(i + 1) + ".0.conv.conv",
            in_ch,
            out_ch,
            ratios[i] * 2,
            weight_storage_type));
    }

    weights.stages.reserve(depths.size());
    for (size_t stage = 0; stage < depths.size(); ++stage) {
        const int64_t channels = config.encoder_n_filters * (int64_t{1} << static_cast<int64_t>(stage));
        std::vector<VibeVoiceTokenizerBlockWeights> blocks;
        blocks.reserve(static_cast<size_t>(depths[stage]));
        for (int64_t block = 0; block < depths[stage]; ++block) {
            blocks.push_back(load_block(
                store,
                source,
                prefix + ".stages." + std::to_string(stage) + "." + std::to_string(block),
                channels,
                weight_storage_type));
        }
        weights.stages.push_back(std::move(blocks));
    }
    const int64_t final_channels = config.encoder_n_filters * (int64_t{1} << static_cast<int64_t>(depths.size() - 1));
    if (config.disable_last_norm) {
        weights.norm = std::nullopt;
    } else {
        weights.norm = source.require_f32_tensor(prefix + ".norm.weight", {final_channels});
    }
    weights.head = load_conv1d(
        store,
        source,
        prefix + ".head.conv.conv",
        final_channels,
        config.vae_dim,
        kTokenizerLastKernelSize,
        weight_storage_type);
    return weights;
}

VibeVoiceTokenizerDecoderWeights load_decoder(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const VibeVoiceTokenizerConfig & config,
    assets::TensorStorageType weight_storage_type) {
    validate_tokenizer_config(config, true);
    auto depths = decoder_depths(config);
    const auto & ratios = config.decoder_ratios;
    if (ratios.empty() || depths.size() != ratios.size() + 1) {
        throw std::runtime_error("VibeVoice tokenizer decoder depths/ratios mismatch");
    }

    VibeVoiceTokenizerDecoderWeights weights;
    const int64_t top_channels = config.decoder_n_filters * (int64_t{1} << static_cast<int64_t>(depths.size() - 1));
    weights.stem = load_conv1d(
        store,
        source,
        prefix + ".upsample_layers.0.0.conv.conv",
        config.vae_dim,
        top_channels,
        kTokenizerKernelSize,
        weight_storage_type);
    weights.upsample_layers.reserve(ratios.size());
    for (size_t i = 0; i < ratios.size(); ++i) {
        const int64_t in_ch = config.decoder_n_filters * (int64_t{1} << static_cast<int64_t>(depths.size() - 1 - i));
        const int64_t out_ch = config.decoder_n_filters * (int64_t{1} << static_cast<int64_t>(depths.size() - 2 - i));
        weights.upsample_layers.push_back(load_conv_transpose1d(
            store,
            source,
            prefix + ".upsample_layers." + std::to_string(i + 1) + ".0.convtr.convtr",
            in_ch,
            out_ch,
            ratios[i] * 2,
            weight_storage_type));
    }

    weights.stages.reserve(depths.size());
    for (size_t stage = 0; stage < depths.size(); ++stage) {
        const int64_t channels = config.decoder_n_filters * (int64_t{1} << static_cast<int64_t>(depths.size() - 1 - stage));
        std::vector<VibeVoiceTokenizerBlockWeights> blocks;
        blocks.reserve(static_cast<size_t>(depths[stage]));
        for (int64_t block = 0; block < depths[stage]; ++block) {
            blocks.push_back(load_block(
                store,
                source,
                prefix + ".stages." + std::to_string(stage) + "." + std::to_string(block),
                channels,
                weight_storage_type));
        }
        weights.stages.push_back(std::move(blocks));
    }
    if (config.disable_last_norm) {
        weights.norm = std::nullopt;
    } else {
        weights.norm = source.require_f32_tensor(prefix + ".norm.weight", {config.decoder_n_filters});
    }
    weights.head = load_conv1d(
        store,
        source,
        prefix + ".head.conv.conv",
        config.decoder_n_filters,
        config.channels,
        kTokenizerLastKernelSize,
        weight_storage_type);
    return weights;
}

int64_t extra_padding_for_conv1d(int64_t length, int64_t kernel_size, int64_t stride, int64_t padding_total) {
    if (length <= 0 || kernel_size <= 0 || stride <= 0 || padding_total < 0) {
        throw std::runtime_error("VibeVoice tokenizer convolution padding inputs are invalid");
    }
    const double n_frames =
        (static_cast<double>(length - kernel_size + padding_total) / static_cast<double>(stride)) + 1.0;
    const int64_t ceil_frames = static_cast<int64_t>(std::ceil(n_frames));
    if (ceil_frames <= 0) {
        throw std::runtime_error("VibeVoice tokenizer convolution would produce no frames");
    }
    const int64_t ideal_length = (ceil_frames - 1) * stride + (kernel_size - padding_total);
    if (ideal_length < length) {
        throw std::runtime_error("VibeVoice tokenizer computed negative extra convolution padding");
    }
    return ideal_length - length;
}

core::TensorValue zero_like_frames(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("VibeVoice tokenizer zero frame template requires positive frame count");
    }
    auto one = modules::SliceModule({2, 0, 1}).build(ctx, input);
    if (one.type != GGML_TYPE_F32) {
        one = core::wrap_tensor(ggml_cast(ctx.ggml, one.tensor, GGML_TYPE_F32), one.shape, GGML_TYPE_F32);
    }
    auto repeated = modules::RepeatModule({
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames})}).build(ctx, one);
    return core::wrap_tensor(ggml_scale(ctx.ggml, repeated.tensor, 0.0F), repeated.shape, GGML_TYPE_F32);
}

core::TensorValue constant_pad_frames(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right) {
    if (left < 0 || right < 0) {
        throw std::runtime_error("VibeVoice tokenizer constant padding must be non-negative");
    }
    auto out = input;
    if (left > 0) {
        out = modules::ConcatModule({2}).build(ctx, zero_like_frames(ctx, input, left), out);
    }
    if (right > 0) {
        out = modules::ConcatModule({2}).build(ctx, out, zero_like_frames(ctx, input, right));
    }
    return out;
}

core::TensorValue sconv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t stride,
    int64_t dilation = 1) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice tokenizer Conv1d weight must be rank 3");
    }
    const int64_t out_channels = weights.weight.shape.dims[0];
    const int64_t in_channels = weights.weight.shape.dims[1];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    const int64_t padding_total = (kernel_size - 1) * dilation - (stride - 1);
    auto padded = constant_pad_frames(
        ctx,
        input,
        padding_total,
        extra_padding_for_conv1d(input.shape.dims[2], kernel_size, stride, padding_total));
    const modules::Conv1dModule conv({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        static_cast<int>(dilation),
        weights.bias.has_value()});
    if (ctx.backend_type == core::BackendType::Vulkan && padded.shape.dims[0] > 1) {
        core::TensorValue output;
        for (int64_t batch = 0; batch < padded.shape.dims[0]; ++batch) {
            auto input_slice = modules::SliceModule({0, batch, 1}).build(ctx, padded);
            input_slice = core::ensure_backend_addressable_layout(ctx, input_slice);
            auto output_slice = conv.build(ctx, input_slice, weights);
            output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, output_slice) : output_slice;
        }
        return output;
    }
    return conv.build(ctx, padded, weights);
}

core::TensorValue sconv_depthwise1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::DepthwiseConv1dWeights & weights,
    int64_t stride,
    int64_t dilation = 1) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice tokenizer depthwise Conv1d weight must be rank 3");
    }
    const int64_t channels = weights.weight.shape.dims[0];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    const int64_t padding_total = (kernel_size - 1) * dilation - (stride - 1);
    auto padded = constant_pad_frames(
        ctx,
        input,
        padding_total,
        extra_padding_for_conv1d(input.shape.dims[2], kernel_size, stride, padding_total));
    modules::DepthwiseConv1dModule conv({
        channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        static_cast<int>(dilation),
        weights.bias.has_value()});
    if (padded.shape.dims[0] == 1) {
        return conv.build(ctx, padded, weights);
    }
    core::TensorValue output;
    for (int64_t batch = 0; batch < padded.shape.dims[0]; ++batch) {
        auto input_slice = modules::SliceModule({0, batch, 1}).build(ctx, padded);
        auto output_slice = conv.build(ctx, input_slice, weights);
        output = output.valid() ? modules::ConcatModule({0}).build(ctx, output, output_slice) : output_slice;
    }
    return output;
}

core::TensorValue sconv_transpose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t stride) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice tokenizer ConvTranspose1d weight must be rank 3");
    }
    const int64_t in_channels = weights.weight.shape.dims[0];
    const int64_t out_channels = weights.weight.shape.dims[1];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    auto full = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        1,
        weights.bias.has_value()}).build(ctx, input, weights);
    const int64_t padding_total = kernel_size - stride;
    if (padding_total < 0) {
        throw std::runtime_error("VibeVoice tokenizer ConvTranspose1d padding_total is negative");
    }
    const int64_t padding_right = static_cast<int64_t>(std::ceil(
        static_cast<double>(padding_total) * static_cast<double>(kTokenizerConvTransposeTrimRightRatio)));
    const int64_t padding_left = padding_total - padding_right;
    const int64_t frames = full.shape.dims[2] - padding_left - padding_right;
    if (frames <= 0) {
        throw std::runtime_error("VibeVoice tokenizer ConvTranspose1d unpad removed all frames");
    }
    return modules::SliceModule({2, padding_left, frames}).build(ctx, full);
}

struct VibeVoiceStreamingBuildState {
    std::vector<ggml_tensor *> cache_inputs;
    std::vector<ggml_tensor *> cache_outputs;
    std::vector<size_t> cache_bytes;
};

core::TensorValue make_streaming_cache_input(
    core::ModuleBuildContext & ctx,
    VibeVoiceStreamingBuildState & state,
    int64_t batch,
    int64_t channels,
    int64_t frames) {
    if (batch <= 0 || channels <= 0 || frames <= 0) {
        throw std::runtime_error("VibeVoice streaming cache input requires positive dimensions");
    }
    auto * tensor = ggml_new_tensor_3d(ctx.ggml, GGML_TYPE_F32, frames, channels, batch);
    ggml_set_input(tensor);
    state.cache_inputs.push_back(tensor);
    state.cache_bytes.push_back(static_cast<size_t>(batch * channels * frames) * sizeof(float));
    return core::wrap_tensor(tensor, core::TensorShape::from_dims({batch, channels, frames}), GGML_TYPE_F32);
}

void add_streaming_cache_output(
    core::ModuleBuildContext & ctx,
    VibeVoiceStreamingBuildState & state,
    const core::TensorValue & cache) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, cache);
    ggml_set_output(contiguous.tensor);
    state.cache_outputs.push_back(contiguous.tensor);
}

core::TensorValue last_frames(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0 || input.shape.dims[2] < frames) {
        throw std::runtime_error("VibeVoice streaming cache slice is outside tensor bounds");
    }
    return modules::SliceModule({2, input.shape.dims[2] - frames, frames}).build(ctx, input);
}

core::TensorValue sconv1d_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t stride,
    VibeVoiceStreamingBuildState & state,
    int64_t dilation = 1) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice streaming tokenizer Conv1d weight must be rank 3");
    }
    const int64_t out_channels = weights.weight.shape.dims[0];
    const int64_t in_channels = weights.weight.shape.dims[1];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    const int64_t context_frames = (kernel_size - 1) * dilation - (stride - 1);
    auto cache = make_streaming_cache_input(ctx, state, input.shape.dims[0], in_channels, context_frames);
    auto full_input = modules::ConcatModule({2}).build(ctx, cache, input);
    auto output = modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        static_cast<int>(dilation),
        weights.bias.has_value()}).build(ctx, full_input, weights);
    add_streaming_cache_output(ctx, state, last_frames(ctx, full_input, context_frames));
    return output;
}

core::TensorValue sconv_depthwise1d_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::DepthwiseConv1dWeights & weights,
    int64_t stride,
    VibeVoiceStreamingBuildState & state,
    int64_t dilation = 1) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice streaming tokenizer depthwise Conv1d weight must be rank 3");
    }
    const int64_t channels = weights.weight.shape.dims[0];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    const int64_t context_frames = (kernel_size - 1) * dilation - (stride - 1);
    auto cache = make_streaming_cache_input(ctx, state, input.shape.dims[0], channels, context_frames);
    auto full_input = modules::ConcatModule({2}).build(ctx, cache, input);
    auto output = modules::DepthwiseConv1dModule({
        channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        static_cast<int>(dilation),
        weights.bias.has_value()}).build(ctx, full_input, weights);
    add_streaming_cache_output(ctx, state, last_frames(ctx, full_input, context_frames));
    return output;
}

core::TensorValue sconv_transpose1d_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t stride,
    VibeVoiceStreamingBuildState & state) {
    if (weights.weight.shape.rank != 3) {
        throw std::runtime_error("VibeVoice streaming tokenizer ConvTranspose1d weight must be rank 3");
    }
    const int64_t in_channels = weights.weight.shape.dims[0];
    const int64_t out_channels = weights.weight.shape.dims[1];
    const int64_t kernel_size = weights.weight.shape.dims[2];
    const int64_t context_frames = kernel_size - 1;
    auto cache = make_streaming_cache_input(ctx, state, input.shape.dims[0], in_channels, context_frames);
    auto full_input = modules::ConcatModule({2}).build(ctx, cache, input);
    auto full = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        1,
        weights.bias.has_value()}).build(ctx, full_input, weights);
    const int64_t padding_total = kernel_size - stride;
    if (padding_total < 0) {
        throw std::runtime_error("VibeVoice streaming tokenizer ConvTranspose1d padding_total is negative");
    }
    const int64_t padding_right = static_cast<int64_t>(std::ceil(
        static_cast<double>(padding_total) * static_cast<double>(kTokenizerConvTransposeTrimRightRatio)));
    const int64_t padding_left = padding_total - padding_right;
    const int64_t unpadded_frames = full.shape.dims[2] - padding_left - padding_right;
    if (unpadded_frames <= 0) {
        throw std::runtime_error("VibeVoice streaming tokenizer ConvTranspose1d unpad removed all frames");
    }
    auto unpadded = modules::SliceModule({2, padding_left, unpadded_frames}).build(ctx, full);
    const int64_t output_frames = input.shape.dims[2] * stride;
    auto output = last_frames(ctx, unpadded, output_frames);
    add_streaming_cache_output(ctx, state, last_frames(ctx, full_input, context_frames));
    return output;
}

core::TensorValue channel_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const assets::TensorDataF32 & weight,
    core::ConstantTensorCache & constants,
    float eps) {
    auto btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    btc = modules::RMSNormModule({input.shape.dims[1], eps, true, false})
              .build(ctx, btc, binding::norm_data(constants, weight));
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, btc);
}

core::TensorValue scale_channels(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const assets::TensorDataF32 & values,
    core::ConstantTensorCache & constants) {
    if (values.shape.rank != 1 || values.shape.dims[0] != input.shape.dims[1]) {
        throw std::runtime_error("VibeVoice tokenizer channel scale shape mismatch");
    }
    auto scale = binding::tensor_data(constants, values);
    scale = core::reshape_tensor(ctx, scale, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    scale = modules::RepeatModule({input.shape}).build(ctx, scale);
    return modules::MulModule{}.build(ctx, input, scale);
}

core::TensorValue linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights) {
    if (weights.weight.shape.rank != 2) {
        throw std::runtime_error("VibeVoice tokenizer Linear weight must be rank 2");
    }
    return modules::LinearModule({
        weights.weight.shape.dims[1],
        weights.weight.shape.dims[0],
        weights.bias.has_value(),
        GGML_PREC_DEFAULT}).build(ctx, input, weights);
}

core::TensorValue tokenizer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerBlockWeights & weights,
    core::ConstantTensorCache & constants,
    float eps) {
    auto residual = input;
    auto hidden = channel_rms_norm(ctx, input, weights.norm, constants, eps);
    hidden = sconv_depthwise1d(ctx, hidden, weights.mixer, 1);
    hidden = scale_channels(ctx, hidden, weights.gamma, constants);
    hidden = modules::AddModule{}.build(ctx, residual, hidden);

    residual = hidden;
    hidden = channel_rms_norm(ctx, hidden, weights.ffn_norm, constants, eps);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = linear(ctx, hidden, weights.ffn_linear1);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = linear(ctx, hidden, weights.ffn_linear2);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = scale_channels(ctx, hidden, weights.ffn_gamma, constants);
    return modules::AddModule{}.build(ctx, residual, hidden);
}

core::TensorValue tokenizer_block_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerBlockWeights & weights,
    core::ConstantTensorCache & constants,
    VibeVoiceStreamingBuildState & state,
    float eps) {
    auto residual = input;
    auto hidden = channel_rms_norm(ctx, input, weights.norm, constants, eps);
    hidden = sconv_depthwise1d_streaming(ctx, hidden, weights.mixer, 1, state);
    hidden = scale_channels(ctx, hidden, weights.gamma, constants);
    hidden = modules::AddModule{}.build(ctx, residual, hidden);

    residual = hidden;
    hidden = channel_rms_norm(ctx, hidden, weights.ffn_norm, constants, eps);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = linear(ctx, hidden, weights.ffn_linear1);
    hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
    hidden = linear(ctx, hidden, weights.ffn_linear2);
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = scale_channels(ctx, hidden, weights.ffn_gamma, constants);
    return modules::AddModule{}.build(ctx, residual, hidden);
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerEncoderWeights & weights,
    const VibeVoiceTokenizerConfig & config,
    core::ConstantTensorCache & constants) {
    if (weights.downsample_layers.size() != weights.stages.size()) {
        throw std::runtime_error("VibeVoice tokenizer encoder layer/stage count mismatch");
    }
    auto hidden = input;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        const int64_t stride = stage == 0 ? 1 : config.encoder_ratios[config.encoder_ratios.size() - stage];
        hidden = sconv1d(ctx, hidden, weights.downsample_layers[stage], stride);
        for (const auto & block : weights.stages[stage]) {
            hidden = tokenizer_block(ctx, hidden, block, constants, config.layernorm_eps);
        }
    }
    if (weights.norm.has_value()) {
        hidden = channel_rms_norm(ctx, hidden, *weights.norm, constants, config.layernorm_eps);
    }
    hidden = sconv1d(ctx, hidden, weights.head, 1);
    return hidden;
}

core::TensorValue build_encoder_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerEncoderWeights & weights,
    const VibeVoiceTokenizerConfig & config,
    core::ConstantTensorCache & constants,
    VibeVoiceStreamingBuildState & state) {
    if (weights.downsample_layers.size() != weights.stages.size()) {
        throw std::runtime_error("VibeVoice streaming tokenizer encoder layer/stage count mismatch");
    }
    auto hidden = input;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        const int64_t stride = stage == 0 ? 1 : config.encoder_ratios[config.encoder_ratios.size() - stage];
        hidden = sconv1d_streaming(ctx, hidden, weights.downsample_layers[stage], stride, state);
        for (const auto & block : weights.stages[stage]) {
            hidden = tokenizer_block_streaming(ctx, hidden, block, constants, state, config.layernorm_eps);
        }
    }
    if (weights.norm.has_value()) {
        hidden = channel_rms_norm(ctx, hidden, *weights.norm, constants, config.layernorm_eps);
    }
    return sconv1d_streaming(ctx, hidden, weights.head, 1, state);
}

core::TensorValue build_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerDecoderWeights & weights,
    const VibeVoiceTokenizerConfig & config,
    core::ConstantTensorCache & constants) {
    if (weights.stages.size() != weights.upsample_layers.size() + 1) {
        throw std::runtime_error("VibeVoice tokenizer decoder layer/stage count mismatch");
    }
    auto hidden = input;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        if (stage == 0) {
            hidden = sconv1d(ctx, hidden, weights.stem, 1);
        } else {
            hidden = sconv_transpose1d(ctx, hidden, weights.upsample_layers[stage - 1], config.decoder_ratios[stage - 1]);
        }
        for (const auto & block : weights.stages[stage]) {
            hidden = tokenizer_block(ctx, hidden, block, constants, config.layernorm_eps);
        }
    }
    if (weights.norm.has_value()) {
        hidden = channel_rms_norm(ctx, hidden, *weights.norm, constants, config.layernorm_eps);
    }
    return sconv1d(ctx, hidden, weights.head, 1);
}

core::TensorValue build_decoder_streaming(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VibeVoiceTokenizerDecoderWeights & weights,
    const VibeVoiceTokenizerConfig & config,
    core::ConstantTensorCache & constants,
    VibeVoiceStreamingBuildState & state) {
    if (weights.stages.size() != weights.upsample_layers.size() + 1) {
        throw std::runtime_error("VibeVoice streaming tokenizer decoder layer/stage count mismatch");
    }
    auto hidden = input;
    for (size_t stage = 0; stage < weights.stages.size(); ++stage) {
        if (stage == 0) {
            hidden = sconv1d_streaming(ctx, hidden, weights.stem, 1, state);
        } else {
            hidden = sconv_transpose1d_streaming(ctx, hidden, weights.upsample_layers[stage - 1], config.decoder_ratios[stage - 1], state);
        }
        for (const auto & block : weights.stages[stage]) {
            hidden = tokenizer_block_streaming(ctx, hidden, block, constants, state, config.layernorm_eps);
        }
    }
    if (weights.norm.has_value()) {
        hidden = channel_rms_norm(ctx, hidden, *weights.norm, constants, config.layernorm_eps);
    }
    return sconv1d_streaming(ctx, hidden, weights.head, 1, state);
}

}  // namespace

class VibeVoiceTokenizerEncoderGraph {
public:
    VibeVoiceTokenizerEncoderGraph(
        std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights,
        const VibeVoiceTokenizerEncoderWeights & encoder,
        const VibeVoiceTokenizerConfig & config,
        const char * graph_name,
        int64_t batch_size,
        int64_t sample_capacity,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          encoder_(&encoder),
          config_(&config),
          batch_size_(batch_size),
          sample_capacity_(sample_capacity),
          backend_(backend),
          backend_type_(backend_type),
          compute_threads_(std::max(1, threads)),
          graph_name_(graph_name) {
        if (weights_ == nullptr || encoder_ == nullptr || config_ == nullptr) {
            throw std::runtime_error("VibeVoice tokenizer encoder graph requires weights and config");
        }
        if (batch_size_ <= 0) {
            throw std::runtime_error("VibeVoice tokenizer encoder graph requires positive batch size");
        }
        if (sample_capacity_ <= 0) {
            throw std::runtime_error("VibeVoice tokenizer encoder graph requires positive sample capacity");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("VibeVoice tokenizer encoder backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice tokenizer encoder ggml context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), graph_name_.c_str(), backend_type_};
        input_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, sample_capacity_, 1, batch_size_);
        ggml_set_input(input_);
        auto input_value = core::wrap_tensor(
            input_,
            core::TensorShape::from_dims({batch_size_, 1, sample_capacity_}),
            GGML_TYPE_F32);

        constants.begin_graph();
        auto output = build_encoder(
            build_ctx,
            input_value,
            *encoder_,
            *config_,
            constants);
        output = core::ensure_backend_addressable_layout(build_ctx, output);
        output_ = output.tensor;
        latent_dim_ = output.shape.dims[1];
        latent_frames_ = output.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate VibeVoice tokenizer encoder graph");
        }
    }

    ~VibeVoiceTokenizerEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        const VibeVoiceTokenizerEncoderWeights & encoder,
        int64_t batch_size,
        int64_t sample_capacity,
        ggml_backend_t backend,
        int threads) const {
        return encoder_ == &encoder && batch_size_ == batch_size && sample_capacity_ == sample_capacity &&
            backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    std::vector<VibeVoiceTokenizerLatents> run(const std::vector<float> & waveforms) {
        if (static_cast<int64_t>(waveforms.size()) != batch_size_ * sample_capacity_) {
            throw std::runtime_error("VibeVoice tokenizer encoder waveform size does not match graph capacity");
        }
        ggml_backend_tensor_set(input_, waveforms.data(), 0, waveforms.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice tokenizer encoder graph compute failed");
        }

        std::vector<float> channel_major(static_cast<size_t>(batch_size_ * latent_dim_ * latent_frames_), 0.0F);
        ggml_backend_tensor_get(output_, channel_major.data(), 0, channel_major.size() * sizeof(float));
        std::vector<VibeVoiceTokenizerLatents> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & latents = out[static_cast<size_t>(batch)];
            latents.frames = latent_frames_;
            latents.dim = latent_dim_;
            latents.values.assign(static_cast<size_t>(latent_frames_ * latent_dim_), 0.0F);
            for (int64_t frame = 0; frame < latent_frames_; ++frame) {
                for (int64_t dim = 0; dim < latent_dim_; ++dim) {
                    latents.values[static_cast<size_t>(frame * latent_dim_ + dim)] =
                        channel_major[static_cast<size_t>(
                            (batch * latent_dim_ + dim) * latent_frames_ + frame)];
                }
            }
        }
        return out;
    }

private:
    std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights_;
    const VibeVoiceTokenizerEncoderWeights * encoder_ = nullptr;
    const VibeVoiceTokenizerConfig * config_ = nullptr;
    int64_t batch_size_ = 0;
    int64_t sample_capacity_ = 0;
    int64_t latent_dim_ = 0;
    int64_t latent_frames_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    std::string graph_name_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class VibeVoiceTokenizerDecoderGraph {
public:
    VibeVoiceTokenizerDecoderGraph(
        std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights,
        const VibeVoiceTokenizerDecoderWeights & decoder,
        const VibeVoiceTokenizerConfig & config,
        int64_t latent_frames,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          decoder_(&decoder),
          config_(&config),
          latent_frames_(latent_frames),
          backend_(backend),
          backend_type_(backend_type),
          compute_threads_(std::max(1, threads)) {
        if (weights_ == nullptr || decoder_ == nullptr || config_ == nullptr) {
            throw std::runtime_error("VibeVoice tokenizer decoder graph requires weights and config");
        }
        if (latent_frames_ <= 0) {
            throw std::runtime_error("VibeVoice tokenizer decoder graph requires positive latent frames");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("VibeVoice tokenizer decoder backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice tokenizer decoder ggml context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), "vibevoice.tokenizer.decoder", backend_type_};
        input_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, latent_frames_, config_->vae_dim, 1);
        ggml_set_input(input_);
        auto input_value = core::wrap_tensor(
            input_,
            core::TensorShape::from_dims({1, config_->vae_dim, latent_frames_}),
            GGML_TYPE_F32);

        constants.begin_graph();
        auto output = build_decoder(build_ctx, input_value, *decoder_, *config_, constants);
        output = core::ensure_backend_addressable_layout(build_ctx, output);
        output_ = output.tensor;
        output_samples_ = output.shape.dims[2];
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate VibeVoice tokenizer decoder graph");
        }
    }

    ~VibeVoiceTokenizerDecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(
        const VibeVoiceTokenizerDecoderWeights & decoder,
        int64_t latent_frames,
        ggml_backend_t backend,
        int threads) const {
        return decoder_ == &decoder && latent_frames_ == latent_frames && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const VibeVoiceTokenizerLatents & latents) {
        if (latents.frames != latent_frames_ || latents.dim != config_->vae_dim) {
            throw std::runtime_error("VibeVoice tokenizer decoder latent shape does not match graph");
        }
        if (static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
            throw std::runtime_error("VibeVoice tokenizer decoder latent payload size mismatch");
        }
        std::vector<float> channel_major(static_cast<size_t>(latents.frames * latents.dim), 0.0F);
        for (int64_t frame = 0; frame < latents.frames; ++frame) {
            for (int64_t dim = 0; dim < latents.dim; ++dim) {
                channel_major[static_cast<size_t>(dim * latents.frames + frame)] =
                    latents.values[static_cast<size_t>(frame * latents.dim + dim)];
            }
        }
        ggml_backend_tensor_set(input_, channel_major.data(), 0, channel_major.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice tokenizer decoder graph compute failed");
        }
        std::vector<float> samples(static_cast<size_t>(output_samples_), 0.0F);
        ggml_backend_tensor_get(output_, samples.data(), 0, samples.size() * sizeof(float));
        return samples;
    }

private:
    std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights_;
    const VibeVoiceTokenizerDecoderWeights * decoder_ = nullptr;
    const VibeVoiceTokenizerConfig * config_ = nullptr;
    int64_t latent_frames_ = 0;
    int64_t output_samples_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class VibeVoiceTokenizerStreamingGraph {
public:
    enum class Kind {
        SemanticEncoder,
        AcousticDecoder,
    };

    VibeVoiceTokenizerStreamingGraph(
        std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights,
        const VibeVoiceTokenizerEncoderWeights * encoder,
        const VibeVoiceTokenizerDecoderWeights * decoder,
        const VibeVoiceTokenizerConfig & config,
        Kind kind,
        int64_t batch_size,
        int64_t input_frames,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        core::ConstantTensorCache & constants,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          encoder_(encoder),
          decoder_(decoder),
          config_(&config),
          kind_(kind),
          batch_size_(batch_size),
          input_frames_(input_frames),
          backend_(backend),
          backend_type_(backend_type),
          compute_threads_(std::max(1, threads)) {
        if (weights_ == nullptr || config_ == nullptr) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph requires weights and config");
        }
        if (kind_ == Kind::SemanticEncoder && encoder_ == nullptr) {
            throw std::runtime_error("VibeVoice semantic streaming graph requires encoder weights");
        }
        if (kind_ == Kind::AcousticDecoder && decoder_ == nullptr) {
            throw std::runtime_error("VibeVoice acoustic streaming graph requires decoder weights");
        }
        if (batch_size_ <= 0) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph requires positive batch size");
        }
        if (input_frames_ <= 0) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph requires positive input frames");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("VibeVoice streaming tokenizer backend is not initialized");
        }

        ggml_init_params params{
            /*.mem_size   =*/ graph_arena_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VibeVoice streaming tokenizer graph context");
        }

        const int64_t input_channels = kind_ == Kind::SemanticEncoder ? 1 : config_->vae_dim;
        core::ModuleBuildContext build_ctx{ctx_.get(), graph_name(), backend_type_};
        input_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, input_frames_, input_channels, batch_size_);
        ggml_set_input(input_);
        auto input_value = core::wrap_tensor(
            input_,
            core::TensorShape::from_dims({batch_size_, input_channels, input_frames_}),
            GGML_TYPE_F32);

        constants.begin_graph();
        VibeVoiceStreamingBuildState streaming_state;
        auto output = kind_ == Kind::SemanticEncoder
            ? build_encoder_streaming(build_ctx, input_value, *encoder_, *config_, constants, streaming_state)
            : build_decoder_streaming(build_ctx, input_value, *decoder_, *config_, constants, streaming_state);
        output = core::ensure_backend_addressable_layout(build_ctx, output);
        output_ = output.tensor;
        output_channels_ = output.shape.dims[1];
        output_frames_ = output.shape.dims[2];
        ggml_set_output(output_);
        cache_inputs_ = std::move(streaming_state.cache_inputs);
        cache_outputs_ = std::move(streaming_state.cache_outputs);
        cache_bytes_ = std::move(streaming_state.cache_bytes);
        cache_sample_bytes_.reserve(cache_bytes_.size());
        for (const size_t bytes : cache_bytes_) {
            if (bytes % static_cast<size_t>(batch_size_) != 0) {
                throw std::runtime_error("VibeVoice streaming tokenizer cache batch size mismatch");
            }
            cache_sample_bytes_.push_back(bytes / static_cast<size_t>(batch_size_));
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        for (auto * cache_output : cache_outputs_) {
            ggml_build_forward_expand(graph_, cache_output);
        }
        constants.finish_graph();
        constants.ensure_uploaded();

        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate VibeVoice streaming tokenizer graph");
        }
    }

    ~VibeVoiceTokenizerStreamingGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(Kind kind, int64_t batch_size, int64_t input_frames, ggml_backend_t backend, int threads) const {
        return kind_ == kind && batch_size_ == batch_size && input_frames_ == input_frames && backend_ == backend &&
            compute_threads_ == std::max(1, threads);
    }

    VibeVoiceTokenizerLatents run_semantic_encoder(
        const std::vector<float> & waveform,
        VibeVoiceTokenizerStreamingState & state) {
        if (kind_ != Kind::SemanticEncoder) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph is not a semantic encoder");
        }
        run_graph(waveform, state);
        std::vector<float> channel_major(static_cast<size_t>(output_channels_ * output_frames_), 0.0F);
        ggml_backend_tensor_get(output_, channel_major.data(), 0, channel_major.size() * sizeof(float));
        VibeVoiceTokenizerLatents out;
        out.frames = output_frames_;
        out.dim = output_channels_;
        out.values.assign(static_cast<size_t>(output_frames_ * output_channels_), 0.0F);
        for (int64_t frame = 0; frame < output_frames_; ++frame) {
            for (int64_t dim = 0; dim < output_channels_; ++dim) {
                out.values[static_cast<size_t>(frame * output_channels_ + dim)] =
                    channel_major[static_cast<size_t>(dim * output_frames_ + frame)];
            }
        }
        return out;
    }

    std::vector<VibeVoiceTokenizerLatents> run_semantic_encoder_batch(
        const std::vector<std::vector<float>> & waveforms,
        const std::vector<VibeVoiceTokenizerStreamingState *> & states) {
        if (kind_ != Kind::SemanticEncoder) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph is not a semantic encoder");
        }
        if (static_cast<int64_t>(waveforms.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice streaming semantic encoder waveform batch size mismatch");
        }
        const size_t per_sample = static_cast<size_t>(input_frames_);
        std::vector<float> input(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & waveform = waveforms[static_cast<size_t>(batch)];
            if (waveform.size() != per_sample) {
                throw std::runtime_error("VibeVoice streaming semantic encoder waveform size mismatch");
            }
            std::copy(
                waveform.begin(),
                waveform.end(),
                input.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(per_sample)));
        }
        run_graph_batch(input, states);
        std::vector<float> channel_major(static_cast<size_t>(batch_size_ * output_channels_ * output_frames_), 0.0F);
        ggml_backend_tensor_get(output_, channel_major.data(), 0, channel_major.size() * sizeof(float));
        std::vector<VibeVoiceTokenizerLatents> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto & item = out[static_cast<size_t>(batch)];
            item.frames = output_frames_;
            item.dim = output_channels_;
            item.values.assign(static_cast<size_t>(output_frames_ * output_channels_), 0.0F);
            for (int64_t frame = 0; frame < output_frames_; ++frame) {
                for (int64_t dim = 0; dim < output_channels_; ++dim) {
                    item.values[static_cast<size_t>(frame * output_channels_ + dim)] =
                        channel_major[static_cast<size_t>(
                            (batch * output_channels_ + dim) * output_frames_ + frame)];
                }
            }
        }
        return out;
    }

    runtime::AudioBuffer run_acoustic_decoder(
        const VibeVoiceTokenizerLatents & latents,
        VibeVoiceTokenizerStreamingState & state) {
        if (kind_ != Kind::AcousticDecoder) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph is not an acoustic decoder");
        }
        if (latents.frames != input_frames_ || latents.dim != config_->vae_dim) {
            throw std::runtime_error("VibeVoice streaming acoustic decoder latent shape mismatch");
        }
        if (static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
            throw std::runtime_error("VibeVoice streaming acoustic decoder latent payload size mismatch");
        }
        std::vector<float> channel_major(static_cast<size_t>(latents.frames * latents.dim), 0.0F);
        for (int64_t frame = 0; frame < latents.frames; ++frame) {
            for (int64_t dim = 0; dim < latents.dim; ++dim) {
                channel_major[static_cast<size_t>(dim * latents.frames + frame)] =
                    latents.values[static_cast<size_t>(frame * latents.dim + dim)];
            }
        }
        run_graph(channel_major, state);
        std::vector<float> samples(static_cast<size_t>(output_frames_), 0.0F);
        ggml_backend_tensor_get(output_, samples.data(), 0, samples.size() * sizeof(float));
        return runtime::AudioBuffer{kTokenizerSampleRate, 1, std::move(samples)};
    }

    std::vector<runtime::AudioBuffer> run_acoustic_decoder_batch(
        const std::vector<VibeVoiceTokenizerLatents> & latents,
        const std::vector<VibeVoiceTokenizerStreamingState *> & states) {
        if (kind_ != Kind::AcousticDecoder) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph is not an acoustic decoder");
        }
        if (static_cast<int64_t>(latents.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice streaming acoustic decoder latent batch size mismatch");
        }
        const size_t per_sample = static_cast<size_t>(input_frames_ * config_->vae_dim);
        std::vector<float> input(static_cast<size_t>(batch_size_) * per_sample, 0.0F);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const auto & item = latents[static_cast<size_t>(batch)];
            if (item.frames != input_frames_ || item.dim != config_->vae_dim) {
                throw std::runtime_error("VibeVoice streaming acoustic decoder latent shape mismatch");
            }
            if (item.values.size() != per_sample) {
                throw std::runtime_error("VibeVoice streaming acoustic decoder latent payload size mismatch");
            }
            const size_t base = static_cast<size_t>(batch) * per_sample;
            for (int64_t frame = 0; frame < input_frames_; ++frame) {
                for (int64_t dim = 0; dim < item.dim; ++dim) {
                    input[base + static_cast<size_t>(dim * input_frames_ + frame)] =
                        item.values[static_cast<size_t>(frame * item.dim + dim)];
                }
            }
        }
        run_graph_batch(input, states);
        if (output_channels_ != 1) {
            throw std::runtime_error("VibeVoice streaming acoustic decoder expected mono output");
        }
        std::vector<float> samples(static_cast<size_t>(batch_size_ * output_frames_), 0.0F);
        ggml_backend_tensor_get(output_, samples.data(), 0, samples.size() * sizeof(float));
        std::vector<runtime::AudioBuffer> out(static_cast<size_t>(batch_size_));
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            auto begin = samples.begin() + static_cast<std::ptrdiff_t>(batch * output_frames_);
            auto end = begin + static_cast<std::ptrdiff_t>(output_frames_);
            out[static_cast<size_t>(batch)] = runtime::AudioBuffer{
                kTokenizerSampleRate,
                1,
                std::vector<float>(begin, end)};
        }
        return out;
    }

private:
    const char * graph_name() const noexcept {
        return kind_ == Kind::SemanticEncoder
            ? "vibevoice.tokenizer.semantic_streaming_encoder"
            : "vibevoice.tokenizer.acoustic_streaming_decoder";
    }

    void ensure_state(VibeVoiceTokenizerStreamingState & state) const {
        if (state.caches.empty()) {
            state.caches.reserve(cache_bytes_.size());
            for (size_t bytes : cache_sample_bytes_) {
                state.caches.emplace_back(bytes / sizeof(float), 0.0F);
            }
        }
        if (state.caches.size() != cache_inputs_.size()) {
            throw std::runtime_error("VibeVoice streaming tokenizer cache count mismatch");
        }
        for (size_t i = 0; i < cache_inputs_.size(); ++i) {
            if (state.caches[i].size() * sizeof(float) != cache_sample_bytes_[i]) {
                throw std::runtime_error("VibeVoice streaming tokenizer cache shape mismatch");
            }
        }
    }

    void ensure_states(const std::vector<VibeVoiceTokenizerStreamingState *> & states) const {
        if (static_cast<int64_t>(states.size()) != batch_size_) {
            throw std::runtime_error("VibeVoice streaming tokenizer state batch size mismatch");
        }
        for (auto * state : states) {
            if (state == nullptr) {
                throw std::runtime_error("VibeVoice streaming tokenizer received null state");
            }
            ensure_state(*state);
        }
    }

    void run_graph(const std::vector<float> & input, VibeVoiceTokenizerStreamingState & state) {
        const size_t expected_input = static_cast<size_t>(
            input_frames_ * (kind_ == Kind::SemanticEncoder ? 1 : config_->vae_dim));
        if (input.size() != expected_input) {
            throw std::runtime_error("VibeVoice streaming tokenizer input payload size mismatch");
        }
        ensure_state(state);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        for (size_t i = 0; i < cache_inputs_.size(); ++i) {
            ggml_backend_tensor_set(cache_inputs_[i], state.caches[i].data(), 0, cache_sample_bytes_[i]);
        }
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph compute failed");
        }
        for (size_t i = 0; i < cache_outputs_.size(); ++i) {
            ggml_backend_tensor_get(cache_outputs_[i], state.caches[i].data(), 0, cache_sample_bytes_[i]);
        }
    }

    void run_graph_batch(
        const std::vector<float> & input,
        const std::vector<VibeVoiceTokenizerStreamingState *> & states) {
        const size_t expected_input = static_cast<size_t>(
            batch_size_ * input_frames_ * (kind_ == Kind::SemanticEncoder ? 1 : config_->vae_dim));
        if (input.size() != expected_input) {
            throw std::runtime_error("VibeVoice streaming tokenizer batch input payload size mismatch");
        }
        ensure_states(states);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        for (size_t i = 0; i < cache_inputs_.size(); ++i) {
            const size_t sample_bytes = cache_sample_bytes_[i];
            const size_t sample_count = sample_bytes / sizeof(float);
            std::vector<float> cache(static_cast<size_t>(batch_size_) * sample_count, 0.0F);
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                const auto & source = states[static_cast<size_t>(batch)]->caches[i];
                std::copy(
                    source.begin(),
                    source.end(),
                    cache.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(sample_count)));
            }
            ggml_backend_tensor_set(cache_inputs_[i], cache.data(), 0, cache.size() * sizeof(float));
        }
        core::set_backend_threads(backend_, compute_threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VibeVoice streaming tokenizer graph compute failed");
        }
        for (size_t i = 0; i < cache_outputs_.size(); ++i) {
            const size_t sample_bytes = cache_sample_bytes_[i];
            const size_t sample_count = sample_bytes / sizeof(float);
            std::vector<float> cache(static_cast<size_t>(batch_size_) * sample_count, 0.0F);
            ggml_backend_tensor_get(cache_outputs_[i], cache.data(), 0, cache.size() * sizeof(float));
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                auto & target = states[static_cast<size_t>(batch)]->caches[i];
                std::copy(
                    cache.begin() + static_cast<std::ptrdiff_t>(batch * static_cast<int64_t>(sample_count)),
                    cache.begin() + static_cast<std::ptrdiff_t>((batch + 1) * static_cast<int64_t>(sample_count)),
                    target.begin());
            }
        }
    }

    std::shared_ptr<const VibeVoiceTokenizerWeightsBundle> weights_;
    const VibeVoiceTokenizerEncoderWeights * encoder_ = nullptr;
    const VibeVoiceTokenizerDecoderWeights * decoder_ = nullptr;
    const VibeVoiceTokenizerConfig * config_ = nullptr;
    Kind kind_ = Kind::SemanticEncoder;
    int64_t batch_size_ = 0;
    int64_t input_frames_ = 0;
    int64_t output_channels_ = 0;
    int64_t output_frames_ = 0;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_tensor *> cache_inputs_;
    std::vector<ggml_tensor *> cache_outputs_;
    std::vector<size_t> cache_bytes_;
    std::vector<size_t> cache_sample_bytes_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

VibeVoiceTokenizerWeightsBundle load_vibevoice_tokenizer_weights(
    const VibeVoiceAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.model_weights == nullptr) {
        throw std::runtime_error("VibeVoice tokenizer weights require model weights");
    }
    VibeVoiceTokenizerWeightsBundle weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vibevoice.tokenizer.weights",
        weight_context_bytes);
    weights.acoustic.encoder = load_encoder(
        *weights.store,
        *assets.model_weights,
        "model.acoustic_tokenizer.encoder",
        assets.config.acoustic_tokenizer,
        weight_storage_type);
    weights.acoustic.decoder = load_decoder(
        *weights.store,
        *assets.model_weights,
        "model.acoustic_tokenizer.decoder",
        assets.config.acoustic_tokenizer,
        weight_storage_type);
    weights.semantic.encoder = load_encoder(
        *weights.store,
        *assets.model_weights,
        "model.semantic_tokenizer.encoder",
        assets.config.semantic_tokenizer,
        weight_storage_type);
    weights.store->upload();
    return weights;
}

VibeVoiceAcousticLatentSample sample_vibevoice_acoustic_latents_gaussian(
    const std::vector<VibeVoiceTokenizerLatents> & means,
    float fixed_std,
    uint64_t seed,
    uint64_t start_rng_index,
    sampling::TorchRandnPrecision precision,
    const std::vector<float> * prompt_noise_values) {
    if (means.empty()) {
        throw std::runtime_error("VibeVoice acoustic latent sampler requires non-empty means");
    }
    if (fixed_std <= 0.0F) {
        throw std::runtime_error("VibeVoice acoustic latent sampler requires positive fixed std");
    }
    const int64_t frames = means.front().frames;
    const int64_t dim = means.front().dim;
    if (frames <= 0 || dim <= 0) {
        throw std::runtime_error("VibeVoice acoustic latent sampler received invalid latent shape");
    }
    const size_t per_sample = static_cast<size_t>(frames * dim);
    for (const auto & mean : means) {
        if (mean.frames != frames || mean.dim != dim || mean.values.size() != per_sample) {
            throw std::runtime_error("VibeVoice acoustic latent sampler requires uniform batched latent shapes");
        }
    }

    const size_t batch_size = means.size();
    std::vector<float> std_noise;
    std::vector<float> latent_noise;
    uint64_t rng_index = start_rng_index;
    if (prompt_noise_values != nullptr) {
        const size_t latent_count = batch_size * per_sample;
        const size_t required_count = batch_size + latent_count;
        if (prompt_noise_values->size() < required_count) {
            throw std::runtime_error(
                "VibeVoice prompt noise file is too short: expected at least " +
                std::to_string(required_count) + " floats, got " +
                std::to_string(prompt_noise_values->size()));
        }
        std_noise.assign(
            prompt_noise_values->begin(),
            prompt_noise_values->begin() + static_cast<std::ptrdiff_t>(batch_size));
        latent_noise.assign(
            prompt_noise_values->begin() + static_cast<std::ptrdiff_t>(batch_size),
            prompt_noise_values->begin() + static_cast<std::ptrdiff_t>(required_count));
        rng_index += static_cast<uint64_t>(required_count);
    } else {
        std_noise = engine::sampling::generate_torch_cuda_randn(
            batch_size,
            seed,
            precision,
            start_rng_index);
        rng_index = start_rng_index + static_cast<uint64_t>(batch_size);
        latent_noise = engine::sampling::generate_torch_cuda_randn(
            batch_size * per_sample,
            seed,
            precision,
            rng_index);
        rng_index += static_cast<uint64_t>(batch_size * per_sample);
    }
    VibeVoiceAcousticLatentSample out;
    out.latents.resize(batch_size);
    out.std_values.resize(batch_size);
    out.next_rng_index = rng_index;
    const float std_scale = fixed_std / 0.8F;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        out.std_values[batch] = std_noise[batch] * std_scale;
        auto & sampled = out.latents[batch];
        sampled.frames = frames;
        sampled.dim = dim;
        sampled.values.resize(per_sample);
        const auto & mean = means[batch].values;
        const size_t noise_base = batch * per_sample;
        for (size_t i = 0; i < per_sample; ++i) {
            sampled.values[i] = mean[i] + out.std_values[batch] * latent_noise[noise_base + i];
        }
    }
    return out;
}

VibeVoiceTokenizerLatents scale_vibevoice_acoustic_latents_for_connector(
    const VibeVoiceTokenizerLatents & latents,
    float speech_scaling_factor,
    float speech_bias_factor) {
    if (latents.frames <= 0 || latents.dim <= 0 ||
        static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
        throw std::runtime_error("VibeVoice acoustic connector scaling received invalid latent shape");
    }
    if (speech_scaling_factor == 0.0F) {
        throw std::runtime_error("VibeVoice acoustic connector scaling factor must be non-zero");
    }
    auto out = latents;
    for (float & value : out.values) {
        value = (value + speech_bias_factor) * speech_scaling_factor;
    }
    return out;
}

VibeVoiceTokenizerLatents unscale_vibevoice_acoustic_latents_for_decoder(
    const VibeVoiceTokenizerLatents & latents,
    float speech_scaling_factor,
    float speech_bias_factor) {
    if (latents.frames <= 0 || latents.dim <= 0 ||
        static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
        throw std::runtime_error("VibeVoice acoustic decoder unscaling received invalid latent shape");
    }
    if (speech_scaling_factor == 0.0F) {
        throw std::runtime_error("VibeVoice acoustic decoder unscaling factor must be non-zero");
    }
    auto out = latents;
    for (float & value : out.values) {
        value = value / speech_scaling_factor - speech_bias_factor;
    }
    return out;
}

VibeVoiceTokenizerWeightsRuntime::VibeVoiceTokenizerWeightsRuntime(
    std::shared_ptr<const VibeVoiceAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      backend_type_(backend_type),
      threads_(threads) {
    if (assets_ == nullptr) {
        throw std::runtime_error("VibeVoice tokenizer weights runtime requires assets");
    }
    if (threads_ <= 0) {
        throw std::runtime_error("VibeVoice tokenizer weights runtime requires positive thread count");
    }
    const auto backend_started = std::chrono::steady_clock::now();
    backend_ = core::init_backend({backend_type, device, threads_});
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.tokenizer_backend_init_ms",
        engine::debug::elapsed_ms(backend_started));
    const auto weights_started = std::chrono::steady_clock::now();
    weights_ = std::make_shared<VibeVoiceTokenizerWeightsBundle>(
        load_vibevoice_tokenizer_weights(
            *assets_,
            backend_,
            backend_type,
            weight_context_bytes,
            weight_storage_type));
    engine::debug::timing_log_scalar(
        "vibevoice.runtime.tokenizer_weights_load_ms",
        engine::debug::elapsed_ms(weights_started));
    acoustic_encoder_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.tokenizer.acoustic_encoder.constants",
        constant_context_bytes);
    semantic_encoder_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.tokenizer.semantic_encoder.constants",
        constant_context_bytes);
    acoustic_decoder_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.tokenizer.acoustic_decoder.constants",
        constant_context_bytes);
    semantic_streaming_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.tokenizer.semantic_streaming.constants",
        constant_context_bytes);
    acoustic_streaming_constants_ = std::make_unique<core::ConstantTensorCache>(
        backend_,
        threads_,
        "vibevoice.tokenizer.acoustic_streaming.constants",
        constant_context_bytes);
}

VibeVoiceTokenizerWeightsRuntime::~VibeVoiceTokenizerWeightsRuntime() {
    acoustic_streaming_graph_.reset();
    semantic_streaming_graph_.reset();
    acoustic_decoder_graph_.reset();
    semantic_encoder_graph_.reset();
    acoustic_encoder_graph_.reset();
    acoustic_streaming_constants_.reset();
    semantic_streaming_constants_.reset();
    acoustic_decoder_constants_.reset();
    semantic_encoder_constants_.reset();
    acoustic_encoder_constants_.reset();
    weights_.reset();
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
    }
}

void VibeVoiceTokenizerStreamingState::reset() {
    caches.clear();
}

void VibeVoiceTokenizerStreamingState::set_to_zero() {
    for (auto & cache : caches) {
        std::fill(cache.begin(), cache.end(), 0.0F);
    }
}

const VibeVoiceAssets & VibeVoiceTokenizerWeightsRuntime::assets() const noexcept {
    return *assets_;
}

const VibeVoiceTokenizerWeightsBundle & VibeVoiceTokenizerWeightsRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t VibeVoiceTokenizerWeightsRuntime::backend() const noexcept {
    return backend_;
}

core::ConstantTensorCache & VibeVoiceTokenizerWeightsRuntime::constants() const noexcept {
    return *acoustic_encoder_constants_;
}

int VibeVoiceTokenizerWeightsRuntime::threads() const noexcept {
    return threads_;
}

VibeVoiceTokenizerLatents VibeVoiceTokenizerWeightsRuntime::encode_acoustic(const runtime::AudioBuffer & audio) const {
    auto encoded = encode_acoustic_batch({audio});
    return std::move(encoded.front());
}

std::vector<VibeVoiceTokenizerLatents> VibeVoiceTokenizerWeightsRuntime::encode_acoustic_batch(
    const std::vector<runtime::AudioBuffer> & audio) const {
    if (audio.empty()) {
        throw std::runtime_error("VibeVoice acoustic tokenizer requires at least one audio prompt");
    }
    std::vector<std::vector<float>> waveforms;
    waveforms.reserve(audio.size());
    int64_t sample_count = 0;
    for (const auto & item : audio) {
        if (item.sample_rate <= 0 || item.channels <= 0 || item.samples.empty()) {
            throw std::runtime_error("VibeVoice acoustic tokenizer requires non-empty audio");
        }
        auto waveform = convert_vibevoice_audio_to_mono_resampled(
            item,
            kTokenizerSampleRate,
            "VibeVoice acoustic prompt");
        if (waveform.empty()) {
            throw std::runtime_error("VibeVoice acoustic tokenizer resampled audio is empty");
        }
        waveform = normalize_vibevoice_prompt_waveform(std::move(waveform), assets_->processor);
        sample_count = std::max(sample_count, static_cast<int64_t>(waveform.size()));
        waveforms.push_back(std::move(waveform));
    }
    const int64_t batch_size = static_cast<int64_t>(waveforms.size());
    std::vector<float> padded(static_cast<size_t>(batch_size * sample_count), 0.0F);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const auto & waveform = waveforms[static_cast<size_t>(batch)];
        std::copy(
            waveform.begin(),
            waveform.end(),
            padded.begin() + static_cast<std::ptrdiff_t>(batch * sample_count));
    }
    if (acoustic_encoder_graph_ == nullptr ||
        !acoustic_encoder_graph_->matches(weights_->acoustic.encoder, batch_size, sample_count, backend_, threads_)) {
        acoustic_encoder_graph_.reset();
        acoustic_encoder_graph_ = std::make_unique<VibeVoiceTokenizerEncoderGraph>(
            weights_,
            weights_->acoustic.encoder,
            assets_->config.acoustic_tokenizer,
            "vibevoice.tokenizer.acoustic_encoder",
            batch_size,
            sample_count,
            backend_,
            backend_type_,
            threads_,
            *acoustic_encoder_constants_,
            1024ull * 1024ull * 1024ull);
    }
    auto encoded = acoustic_encoder_graph_->run(padded);
    return encoded;
}

VibeVoiceTokenizerLatents VibeVoiceTokenizerWeightsRuntime::encode_semantic(const runtime::AudioBuffer & audio) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("VibeVoice semantic tokenizer requires non-empty audio");
    }
    const auto waveform = convert_vibevoice_audio_to_mono_resampled(
        audio,
        kTokenizerSampleRate,
        "VibeVoice semantic tokenizer");
    if (waveform.empty()) {
        throw std::runtime_error("VibeVoice semantic tokenizer resampled audio is empty");
    }
    const int64_t sample_count = static_cast<int64_t>(waveform.size());
    if (semantic_encoder_graph_ == nullptr ||
        !semantic_encoder_graph_->matches(weights_->semantic.encoder, 1, sample_count, backend_, threads_)) {
        semantic_encoder_graph_.reset();
        semantic_encoder_graph_ = std::make_unique<VibeVoiceTokenizerEncoderGraph>(
            weights_,
            weights_->semantic.encoder,
            assets_->config.semantic_tokenizer,
            "vibevoice.tokenizer.semantic_encoder",
            1,
            sample_count,
            backend_,
            backend_type_,
            threads_,
            *semantic_encoder_constants_,
            1024ull * 1024ull * 1024ull);
    }
    auto encoded = semantic_encoder_graph_->run(waveform);
    return std::move(encoded.front());
}

void VibeVoiceTokenizerWeightsRuntime::release_prompt_graphs() const {
    acoustic_encoder_graph_.reset();
    semantic_encoder_graph_.reset();
}

runtime::AudioBuffer VibeVoiceTokenizerWeightsRuntime::decode_acoustic(const VibeVoiceTokenizerLatents & latents) const {
    if (latents.frames <= 0 || latents.dim != assets_->config.acoustic_tokenizer.vae_dim) {
        throw std::runtime_error("VibeVoice acoustic decoder received invalid latent shape");
    }
    if (static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
        throw std::runtime_error("VibeVoice acoustic decoder latent payload size mismatch");
    }
    if (acoustic_decoder_graph_ == nullptr ||
        !acoustic_decoder_graph_->matches(weights_->acoustic.decoder, latents.frames, backend_, threads_)) {
        acoustic_decoder_graph_.reset();
        acoustic_decoder_graph_ = std::make_unique<VibeVoiceTokenizerDecoderGraph>(
            weights_,
            weights_->acoustic.decoder,
            assets_->config.acoustic_tokenizer,
            latents.frames,
            backend_,
            backend_type_,
            threads_,
            *acoustic_decoder_constants_,
            1024ull * 1024ull * 1024ull);
    }
    auto samples = acoustic_decoder_graph_->run(latents);
    return runtime::AudioBuffer{kTokenizerSampleRate, 1, std::move(samples)};
}

VibeVoiceTokenizerLatents VibeVoiceTokenizerWeightsRuntime::encode_semantic_streaming(
    const runtime::AudioBuffer & audio,
    VibeVoiceTokenizerStreamingState & state) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("VibeVoice streaming semantic tokenizer requires non-empty audio");
    }
    const auto waveform = convert_vibevoice_audio_to_mono_resampled(
        audio,
        kTokenizerSampleRate,
        "VibeVoice streaming semantic tokenizer");
    if (waveform.empty()) {
        throw std::runtime_error("VibeVoice streaming semantic tokenizer resampled audio is empty");
    }
    const int64_t sample_count = static_cast<int64_t>(waveform.size());
    if (semantic_streaming_graph_ == nullptr ||
        !semantic_streaming_graph_->matches(
            VibeVoiceTokenizerStreamingGraph::Kind::SemanticEncoder,
            1,
            sample_count,
            backend_,
            threads_)) {
        semantic_streaming_graph_.reset();
        state.reset();
        semantic_streaming_graph_ = std::make_unique<VibeVoiceTokenizerStreamingGraph>(
            weights_,
            &weights_->semantic.encoder,
            nullptr,
            assets_->config.semantic_tokenizer,
            VibeVoiceTokenizerStreamingGraph::Kind::SemanticEncoder,
            1,
            sample_count,
            backend_,
            backend_type_,
            threads_,
            *semantic_streaming_constants_,
            1024ull * 1024ull * 1024ull);
    }
    return semantic_streaming_graph_->run_semantic_encoder(waveform, state);
}

std::vector<VibeVoiceTokenizerLatents> VibeVoiceTokenizerWeightsRuntime::encode_semantic_streaming_batch(
    const std::vector<runtime::AudioBuffer> & audio,
    std::vector<VibeVoiceTokenizerStreamingState *> states) const {
    if (audio.empty()) {
        throw std::runtime_error("VibeVoice batch streaming semantic tokenizer requires non-empty audio");
    }
    if (audio.size() != states.size()) {
        throw std::runtime_error("VibeVoice batch streaming semantic tokenizer state count mismatch");
    }
    std::vector<std::vector<float>> waveforms;
    waveforms.reserve(audio.size());
    int64_t sample_count = 0;
    for (const auto & item : audio) {
        if (item.sample_rate <= 0 || item.channels <= 0 || item.samples.empty()) {
            throw std::runtime_error("VibeVoice batch streaming semantic tokenizer requires non-empty audio");
        }
        auto waveform = convert_vibevoice_audio_to_mono_resampled(
            item,
            kTokenizerSampleRate,
            "VibeVoice batch streaming semantic tokenizer");
        if (waveform.empty()) {
            throw std::runtime_error("VibeVoice batch streaming semantic tokenizer resampled audio is empty");
        }
        if (sample_count == 0) {
            sample_count = static_cast<int64_t>(waveform.size());
        } else if (static_cast<int64_t>(waveform.size()) != sample_count) {
            throw std::runtime_error("VibeVoice batch streaming semantic tokenizer requires uniform chunk size");
        }
        waveforms.push_back(std::move(waveform));
    }
    const int64_t batch_size = static_cast<int64_t>(audio.size());
    if (semantic_streaming_graph_ == nullptr ||
        !semantic_streaming_graph_->matches(
            VibeVoiceTokenizerStreamingGraph::Kind::SemanticEncoder,
            batch_size,
            sample_count,
            backend_,
            threads_)) {
        semantic_streaming_graph_.reset();
        semantic_streaming_graph_ = std::make_unique<VibeVoiceTokenizerStreamingGraph>(
            weights_,
            &weights_->semantic.encoder,
            nullptr,
            assets_->config.semantic_tokenizer,
            VibeVoiceTokenizerStreamingGraph::Kind::SemanticEncoder,
            batch_size,
            sample_count,
            backend_,
            backend_type_,
            threads_,
            *semantic_streaming_constants_,
            1024ull * 1024ull * 1024ull);
    }
    return semantic_streaming_graph_->run_semantic_encoder_batch(waveforms, states);
}

runtime::AudioBuffer VibeVoiceTokenizerWeightsRuntime::decode_acoustic_streaming(
    const VibeVoiceTokenizerLatents & latents,
    VibeVoiceTokenizerStreamingState & state) const {
    if (latents.frames <= 0 || latents.dim != assets_->config.acoustic_tokenizer.vae_dim) {
        throw std::runtime_error("VibeVoice streaming acoustic decoder received invalid latent shape");
    }
    if (static_cast<int64_t>(latents.values.size()) != latents.frames * latents.dim) {
        throw std::runtime_error("VibeVoice streaming acoustic decoder latent payload size mismatch");
    }
    if (acoustic_streaming_graph_ == nullptr ||
        !acoustic_streaming_graph_->matches(
            VibeVoiceTokenizerStreamingGraph::Kind::AcousticDecoder,
            1,
            latents.frames,
            backend_,
            threads_)) {
        acoustic_streaming_graph_.reset();
        state.reset();
        acoustic_streaming_graph_ = std::make_unique<VibeVoiceTokenizerStreamingGraph>(
            weights_,
            nullptr,
            &weights_->acoustic.decoder,
            assets_->config.acoustic_tokenizer,
            VibeVoiceTokenizerStreamingGraph::Kind::AcousticDecoder,
            1,
            latents.frames,
            backend_,
            backend_type_,
            threads_,
            *acoustic_streaming_constants_,
            1024ull * 1024ull * 1024ull);
    }
    return acoustic_streaming_graph_->run_acoustic_decoder(latents, state);
}

std::vector<runtime::AudioBuffer> VibeVoiceTokenizerWeightsRuntime::decode_acoustic_streaming_batch(
    const std::vector<VibeVoiceTokenizerLatents> & latents,
    std::vector<VibeVoiceTokenizerStreamingState *> states) const {
    if (latents.empty()) {
        throw std::runtime_error("VibeVoice batch streaming acoustic decoder requires non-empty latents");
    }
    if (latents.size() != states.size()) {
        throw std::runtime_error("VibeVoice batch streaming acoustic decoder state count mismatch");
    }
    const int64_t frames = latents.front().frames;
    if (frames <= 0 || latents.front().dim != assets_->config.acoustic_tokenizer.vae_dim) {
        throw std::runtime_error("VibeVoice batch streaming acoustic decoder received invalid latent shape");
    }
    const size_t per_sample = static_cast<size_t>(frames * latents.front().dim);
    for (const auto & item : latents) {
        if (item.frames != frames || item.dim != assets_->config.acoustic_tokenizer.vae_dim) {
            throw std::runtime_error("VibeVoice batch streaming acoustic decoder requires uniform latent shape");
        }
        if (item.values.size() != per_sample) {
            throw std::runtime_error("VibeVoice batch streaming acoustic decoder latent payload size mismatch");
        }
    }
    const int64_t batch_size = static_cast<int64_t>(latents.size());
    if (acoustic_streaming_graph_ == nullptr ||
        !acoustic_streaming_graph_->matches(
            VibeVoiceTokenizerStreamingGraph::Kind::AcousticDecoder,
            batch_size,
            frames,
            backend_,
            threads_)) {
        acoustic_streaming_graph_.reset();
        acoustic_streaming_graph_ = std::make_unique<VibeVoiceTokenizerStreamingGraph>(
            weights_,
            nullptr,
            &weights_->acoustic.decoder,
            assets_->config.acoustic_tokenizer,
            VibeVoiceTokenizerStreamingGraph::Kind::AcousticDecoder,
            batch_size,
            frames,
            backend_,
            backend_type_,
            threads_,
            *acoustic_streaming_constants_,
            1024ull * 1024ull * 1024ull);
    }
    return acoustic_streaming_graph_->run_acoustic_decoder_batch(latents, states);
}

}  // namespace engine::models::vibevoice

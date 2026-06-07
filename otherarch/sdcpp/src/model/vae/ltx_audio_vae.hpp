#ifndef __SD_MODEL_VAE_LTX_AUDIO_VAE_HPP__
#define __SD_MODEL_VAE_LTX_AUDIO_VAE_HPP__

#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"

namespace LTXV {

    struct LTXAudioVAEConfig {
        int sample_rate                                            = 16000;
        int mel_hop_length                                         = 160;
        int n_fft                                                  = 1024;
        int mel_bins                                               = 64;
        int latent_channels                                        = 8;
        int latent_frequency_bins                                  = 16;
        int audio_channels                                         = 2;
        int decoder_channels                                       = 128;
        std::vector<int> decoder_channel_multipliers               = {1, 2, 4};
        int decoder_num_res_blocks                                 = 2;
        int base_upsample_initial_channel                          = 1536;
        std::vector<int> base_upsample_rates                       = {5, 2, 2, 2, 2, 2};
        std::vector<int> base_upsample_kernel_sizes                = {11, 4, 4, 4, 4, 4};
        std::vector<int> base_resblock_kernel_sizes                = {3, 7, 11};
        std::vector<std::vector<int>> base_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
        bool has_bwe                                               = false;
        int bwe_input_sample_rate                                  = 16000;
        int bwe_output_sample_rate                                 = 48000;
        int bwe_hop_length                                         = 80;
        int bwe_n_fft                                              = 512;
        int bwe_num_mels                                           = 64;
        int bwe_upsample_initial_channel                           = 512;
        std::vector<int> bwe_upsample_rates                        = {6, 5, 2, 2, 2};
        std::vector<int> bwe_upsample_kernel_sizes                 = {12, 11, 4, 4, 4};
        std::vector<int> bwe_resblock_kernel_sizes                 = {3, 7, 11};
        std::vector<std::vector<int>> bwe_resblock_dilation_sizes  = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};

        int latent_downsample_factor() const {
            return 4;
        }

        int base_output_sample_rate() const {
            int upsample_factor = 1;
            for (int rate : base_upsample_rates) {
                upsample_factor *= rate;
            }
            return sample_rate * upsample_factor / mel_hop_length;
        }

        int output_sample_rate() const {
            if (has_bwe) {
                return bwe_output_sample_rate;
            }
            return base_output_sample_rate();
        }

        static LTXAudioVAEConfig detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix = "") {
            LTXAudioVAEConfig config;

            auto require = [&](const std::string& name) -> const TensorStorage* {
                std::string tensor_name = prefix.empty() ? name : prefix + "." + name;
                auto iter               = tensor_storage_map.find(tensor_name);
                if (iter == tensor_storage_map.end()) {
                    return nullptr;
                }
                return &iter->second;
            };

            const TensorStorage* decoder_conv_in   = require("audio_vae.decoder.conv_in.conv.weight");
            const TensorStorage* decoder_conv_out  = require("audio_vae.decoder.conv_out.conv.weight");
            const TensorStorage* latent_std        = require("audio_vae.per_channel_statistics.std-of-means");
            const TensorStorage* vocoder_conv_pre  = require("vocoder.vocoder.conv_pre.weight");
            const TensorStorage* vocoder_conv_post = require("vocoder.vocoder.conv_post.weight");
            if (decoder_conv_in == nullptr || decoder_conv_out == nullptr || latent_std == nullptr ||
                vocoder_conv_pre == nullptr || vocoder_conv_post == nullptr) {
                return config;
            }

            config.sample_rate                  = 16000;
            config.mel_hop_length               = 160;
            config.n_fft                        = 1024;
            config.base_upsample_rates          = {5, 2, 2, 2, 2, 2};
            config.base_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};

            config.latent_channels               = static_cast<int>(decoder_conv_in->ne[2]);
            config.audio_channels                = static_cast<int>(decoder_conv_out->ne[3]);
            config.latent_frequency_bins         = static_cast<int>(latent_std->ne[0] / std::max<int64_t>(1, decoder_conv_in->ne[2]));
            config.mel_bins                      = config.latent_frequency_bins * config.latent_downsample_factor();
            config.base_upsample_initial_channel = static_cast<int>(vocoder_conv_pre->ne[2]);

            if (latent_std->ne[0] % std::max<int64_t>(1, decoder_conv_in->ne[2]) != 0) {
                return config;
            }

            std::vector<std::pair<int, int>> level_channels;
            for (const auto& pair : tensor_storage_map) {
                const std::string& name  = pair.first;
                const std::string prefix = "audio_vae.decoder.up.";
                const std::string suffix = ".block.0.conv1.conv.weight";
                if (!starts_with(name, prefix) || !ends_with(name, suffix)) {
                    continue;
                }
                std::string level_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
                int level             = std::stoi(level_str);
                level_channels.push_back({level, static_cast<int>(pair.second.ne[3])});
            }
            std::sort(level_channels.begin(), level_channels.end());
            if (level_channels.empty()) {
                return config;
            }
            config.decoder_channels = level_channels.front().second;
            config.decoder_channel_multipliers.clear();
            for (const auto& level_channel : level_channels) {
                config.decoder_channel_multipliers.push_back(level_channel.second / std::max(1, config.decoder_channels));
            }

            int block_count = 0;
            while (tensor_storage_map.find("audio_vae.decoder.up.0.block." + std::to_string(block_count) + ".conv1.conv.weight") != tensor_storage_map.end()) {
                ++block_count;
            }
            if (block_count <= 0) {
                return config;
            }
            config.decoder_num_res_blocks = block_count - 1;

            config.base_upsample_kernel_sizes.clear();
            for (int i = 0;; ++i) {
                auto iter = tensor_storage_map.find("vocoder.vocoder.ups." + std::to_string(i) + ".weight");
                if (iter == tensor_storage_map.end()) {
                    break;
                }
                config.base_upsample_kernel_sizes.push_back(static_cast<int>(iter->second.ne[0]));
            }
            if (config.base_upsample_kernel_sizes.size() != config.base_upsample_rates.size()) {
                return config;
            }

            config.base_resblock_kernel_sizes.clear();
            for (int i = 0;; ++i) {
                auto iter = tensor_storage_map.find("vocoder.vocoder.resblocks." + std::to_string(i) + ".convs1.0.weight");
                if (iter == tensor_storage_map.end()) {
                    break;
                }
                config.base_resblock_kernel_sizes.push_back(static_cast<int>(iter->second.ne[0]));
            }
            if (config.base_resblock_kernel_sizes.size() < 3) {
                return config;
            }
            config.base_resblock_kernel_sizes.resize(3);

            config.has_bwe = tensor_storage_map.find("vocoder.bwe_generator.conv_pre.weight") != tensor_storage_map.end();
            if (config.has_bwe) {
                config.bwe_input_sample_rate        = 16000;
                config.bwe_output_sample_rate       = 48000;
                config.bwe_hop_length               = 80;
                config.bwe_n_fft                    = 512;
                config.bwe_num_mels                 = 64;
                config.bwe_upsample_initial_channel = 512;
                config.bwe_upsample_rates           = {6, 5, 2, 2, 2};
                config.bwe_upsample_kernel_sizes    = {12, 11, 4, 4, 4};
                config.bwe_resblock_kernel_sizes    = {3, 7, 11};
                config.bwe_resblock_dilation_sizes  = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
            }

            if (config.audio_channels != 2 || config.latent_channels != 8 || config.mel_bins != 64) {
                return config;
            }
            LOG_DEBUG("ltx_audio_vae: sample_rate = %d, mel_bins = %d, latent_channels = %d, latent_frequency_bins = %d, has_bwe = %s",
                      config.sample_rate,
                      config.mel_bins,
                      config.latent_channels,
                      config.latent_frequency_bins,
                      config.has_bwe ? "true" : "false");
            return config;
        }
    };

    static ggml_tensor* compute_log_mel_spectrogram(GGMLRunnerContext* runner_ctx,
                                                    ggml_tensor* waveform,
                                                    ggml_tensor* forward_basis,
                                                    ggml_tensor* mel_basis,
                                                    int hop_length) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(waveform != nullptr);
        GGML_ASSERT(forward_basis != nullptr);
        GGML_ASSERT(mel_basis != nullptr);
        GGML_ASSERT(waveform->type == GGML_TYPE_F32);
        GGML_ASSERT(forward_basis->type == GGML_TYPE_F32);
        GGML_ASSERT(mel_basis->type == GGML_TYPE_F32);
        GGML_ASSERT(forward_basis->ne[1] == 1);

        const int64_t time          = waveform->ne[0];
        const int64_t channels      = waveform->ne[1];
        const int64_t batch         = waveform->ne[2];
        const int64_t filter_len    = forward_basis->ne[0];
        const int64_t stft_channels = forward_basis->ne[2];
        const int64_t n_freqs       = stft_channels / 2;
        const int64_t n_mels        = mel_basis->ne[1];
        const int64_t left_pad      = std::max<int64_t>(0, filter_len - hop_length);
        const int64_t padded_time   = time + left_pad;
        const int64_t frame_count   = padded_time < filter_len ? 0 : 1 + (padded_time - filter_len) / hop_length;

        GGML_ASSERT(stft_channels % 2 == 0);
        GGML_ASSERT(mel_basis->ne[0] == n_freqs);
        GGML_ASSERT(waveform->ne[3] == 1);
        GGML_ASSERT(frame_count > 0);

        auto x = ggml_reshape_3d(ctx, waveform, time, 1, channels * batch);
        if (left_pad > 0) {
            x = ggml_pad_ext(ctx, x, static_cast<int>(left_pad), 0, 0, 0, 0, 0, 0, 0);
        }

        auto frames = ggml_conv_1d(ctx, forward_basis, x, hop_length, 0, 1);
        GGML_ASSERT(frames->ne[0] == frame_count);
        GGML_ASSERT(frames->ne[1] == stft_channels);
        GGML_ASSERT(frames->ne[2] == channels * batch);

        auto real      = ggml_ext_slice(ctx, frames, 1, 0, n_freqs);
        auto imag      = ggml_ext_slice(ctx, frames, 1, n_freqs, stft_channels);
        auto magnitude = ggml_sqrt(ctx,
                                   ggml_add(ctx,
                                            ggml_sqr(ctx, real),
                                            ggml_sqr(ctx, imag)));

        magnitude = ggml_cont(ctx, ggml_permute(ctx, magnitude, 1, 0, 2, 3));
        auto mel  = ggml_mul_mat(ctx, mel_basis, magnitude);
        mel       = ggml_log(ctx, ggml_clamp(ctx, mel, 1e-5f, std::numeric_limits<float>::max()));

        return ggml_reshape_4d(ctx, mel, n_mels, frame_count, channels, batch);
    }

    static std::vector<float> build_hann_resample_filter(int ratio) {
        constexpr double kPi           = 3.14159265358979323846;
        const double rolloff           = 0.99;
        const int lowpass_filter_width = 6;
        const int width                = static_cast<int>(std::ceil(static_cast<double>(lowpass_filter_width) / rolloff));
        const int kernel_size          = 2 * width * ratio + 1;
        const double half_lowpass_pi   = kPi / lowpass_filter_width / 2.0;
        std::vector<float> filter(static_cast<size_t>(kernel_size), 0.0f);
        for (int i = 0; i < kernel_size; ++i) {
            double t         = (static_cast<double>(i) / ratio - width) * rolloff;
            double t_clamped = std::clamp(t, -static_cast<double>(lowpass_filter_width), static_cast<double>(lowpass_filter_width));
            double window    = std::cos(t_clamped * half_lowpass_pi);
            window *= window;
            double sinc                    = t == 0.0 ? 1.0 : std::sin(kPi * t) / (kPi * t);
            filter[static_cast<size_t>(i)] = static_cast<float>(sinc * window * rolloff / ratio);
        }
        return filter;
    }

    static ggml_type audio_conv_weight_type(ggml_type type) {
        return type == GGML_TYPE_BF16 ? GGML_TYPE_F16 : type;
    }

    static ggml_tensor* repeat_with_vulkan_f32_workaround(ggml_backend_t backend,
                                                          ggml_context* ctx,
                                                          ggml_tensor* x,
                                                          int64_t ne0,
                                                          int64_t ne1,
                                                          int64_t ne2,
                                                          int64_t ne3) {
        if (x->type != GGML_TYPE_F32 &&
            (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
            sd_backend_is(backend, "vulkan")) {
            auto x_f32    = ggml_cast(ctx, x, GGML_TYPE_F32);
            auto repeated = ggml_repeat_4d(ctx,
                                           x_f32,
                                           ne0,
                                           ne1,
                                           ne2,
                                           ne3);
            return ggml_cast(ctx, repeated, x->type);
        }
        return ggml_repeat_4d(ctx, x, ne0, ne1, ne2, ne3);
    }

    static ggml_tensor* repeat_1d_value(GGMLRunnerContext* runner_ctx, ggml_tensor* x, int64_t count) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(x->ne[0] == 1);
        return repeat_with_vulkan_f32_workaround(runner_ctx->backend, ctx, x, count, x->ne[1], x->ne[2], x->ne[3]);
    }

    static ggml_tensor* replicate_pad_1d(GGMLRunnerContext* runner_ctx, ggml_tensor* x, int64_t left, int64_t right) {
        auto ctx = runner_ctx->ggml_ctx;
        if (left > 0) {
            auto first = ggml_ext_slice(ctx, x, 0, 0, 1);
            x          = ggml_concat(ctx, repeat_1d_value(runner_ctx, first, left), x, 0);
        }
        if (right > 0) {
            auto last = ggml_ext_slice(ctx, x, 0, x->ne[0] - 1, x->ne[0]);
            x         = ggml_concat(ctx, x, repeat_1d_value(runner_ctx, last, right), 0);
        }
        return x;
    }

    static ggml_tensor* tile_depthwise_filter_1d(GGMLRunnerContext* runner_ctx, ggml_tensor* filter, int64_t channels) {
        auto ctx          = runner_ctx->ggml_ctx;
        ggml_tensor* base = filter;
        if (ggml_n_dims(base) == 3) {
            base = ggml_reshape_4d(ctx, base, base->ne[0], 1, 1, 1);
        } else if (ggml_n_dims(base) == 1) {
            base = ggml_reshape_4d(ctx, base, base->ne[0], 1, 1, 1);
        }
        return repeat_with_vulkan_f32_workaround(runner_ctx->backend, ctx, base, base->ne[0], 1, channels, 1);
    }

    static ggml_tensor* depthwise_conv1d(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* filter,
                                         int stride,
                                         int padding) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(x->ne[3] == 1);
        auto tiled = tile_depthwise_filter_1d(runner_ctx, filter, x->ne[1]);
        auto out   = ggml_conv_1d_dw(ctx, tiled, x, stride, padding, 1);
        return ggml_reshape_4d(ctx, out, out->ne[0], out->ne[1], 1, 1);
    }

    static ggml_tensor* reverse_1d_filter(ggml_context* ctx, ggml_tensor* filter) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(filter != nullptr);
        GGML_ASSERT(filter->ne[1] == 1);
        GGML_ASSERT(filter->ne[2] == 1);
        GGML_ASSERT(filter->ne[3] == 1);

        ggml_tensor* reversed = nullptr;
        for (int64_t k = filter->ne[0] - 1; k >= 0; --k) {
            auto slice = ggml_ext_slice(ctx, filter, 0, k, k + 1);
            reversed   = reversed == nullptr ? slice : ggml_concat(ctx, reversed, slice, 0);
        }
        return reversed;
    }

    static ggml_tensor* depthwise_conv_transpose1d(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* filter,
                                                   int stride) {
        GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);
        GGML_ASSERT(filter->ne[1] == 1);
        GGML_ASSERT(filter->ne[2] == 1 && filter->ne[3] == 1);

        const int64_t time        = x->ne[0];
        const int64_t channels    = x->ne[1];
        const int64_t kernel_size = filter->ne[0];
        const int64_t out_time    = (time - 1) * stride + kernel_size;

        auto x_flat = ggml_reshape_3d(ctx, x, 1, time, channels);
        if (stride > 1) {
            auto zero_unit = ggml_ext_scale(ctx, x_flat, 0.0f);
            auto zero_tail = zero_unit;
            for (int i = 1; i < stride - 1; ++i) {
                zero_tail = ggml_concat(ctx, zero_tail, zero_unit, 0);
            }
            x_flat = ggml_concat(ctx, x_flat, zero_tail, 0);
        }
        x_flat = ggml_reshape_3d(ctx, x_flat, time * stride, 1, channels);

        auto reversed_filter = reverse_1d_filter(ctx, filter);
        auto out             = ggml_conv_1d(ctx, reversed_filter, x_flat, 1, static_cast<int>(kernel_size - 1), 1);
        if (out->ne[0] > out_time) {
            out = ggml_ext_slice(ctx, out, 0, 0, out_time);
        }
        GGML_ASSERT(out->ne[0] == out_time);
        GGML_ASSERT(out->ne[1] == 1);
        GGML_ASSERT(out->ne[2] == channels);

        out = ggml_ext_scale(ctx, out, static_cast<float>(stride));
        return ggml_reshape_4d(ctx, out, out_time, channels, 1, 1);
    }

    static ggml_tensor* upsample_waveform_hann(GGMLRunnerContext* runner_ctx,
                                               ggml_tensor* waveform,
                                               ggml_tensor* filter,
                                               int ratio) {
        auto ctx = runner_ctx->ggml_ctx;
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(waveform != nullptr);
        GGML_ASSERT(filter != nullptr);
        GGML_ASSERT(waveform->ne[3] == 1);
        if (ratio <= 1) {
            return waveform;
        }

        const int lowpass_filter_width = 6;
        const double rolloff           = 0.99;
        const int width                = static_cast<int>(std::ceil(static_cast<double>(lowpass_filter_width) / rolloff));
        const int kernel_size          = 2 * width * ratio + 1;
        const int pad                  = width;
        const int pad_left             = 2 * width * ratio;
        const int pad_right            = kernel_size - ratio;
        const int64_t time             = waveform->ne[0];
        const int64_t channels         = waveform->ne[1];
        const int64_t batch            = waveform->ne[2];

        GGML_ASSERT(filter->ne[0] == kernel_size);

        auto x = ggml_reshape_3d(ctx, waveform, time, channels * batch, 1);
        x      = replicate_pad_1d(runner_ctx, x, pad, pad);
        x      = depthwise_conv_transpose1d(ctx, x, filter, ratio);
        x      = ggml_ext_slice(ctx, x, 0, pad_left, x->ne[0] - pad_right);
        return ggml_reshape_3d(ctx, x, x->ne[0], channels, batch);
    }

    static ggml_tensor* crop_waveform_samples(ggml_context* ctx,
                                              ggml_tensor* waveform,
                                              int64_t target_samples) {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(waveform != nullptr);
        if (waveform->ne[0] == target_samples) {
            return waveform;
        }
        GGML_ASSERT(waveform->ne[0] > target_samples);
        return ggml_ext_slice(ctx, waveform, 0, 0, target_samples);
    }

    struct PixelNorm2D : public UnaryBlock {
        float eps = 1e-6f;

        explicit PixelNorm2D(float eps = 1e-6f)
            : eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto h = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));
            h      = ggml_rms_norm(ctx->ggml_ctx, h, eps);
            h      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 0, 3));
            return h;
        }
    };

    struct HeightCausalConv2D : public UnaryBlock {
        std::pair<int, int> kernel_size;

        HeightCausalConv2D(int64_t in_channels,
                           int64_t out_channels,
                           std::pair<int, int> kernel_size,
                           std::pair<int, int> stride = {1, 1},
                           bool bias                  = true)
            : kernel_size(kernel_size) {
            blocks["conv"] = std::make_shared<Conv2d>(in_channels, out_channels, kernel_size, stride, std::pair<int, int>{0, 0}, std::pair<int, int>{1, 1}, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);
            int pad_h = kernel_size.first - 1;
            int pad_w = kernel_size.second - 1;
            x         = ggml_ext_pad_ext(ctx->ggml_ctx,
                                         x,
                                         pad_w / 2,
                                         pad_w - pad_w / 2,
                                         pad_h,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0);
            x         = conv->forward(ctx, x);
            return x;
        }
    };

    struct AudioUpsample2D : public GGMLBlock {
        AudioUpsample2D(int64_t channels) {
            blocks["conv"] = std::make_shared<HeightCausalConv2D>(channels, channels, std::pair<int, int>{3, 3});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["conv"]);
            x         = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
            x         = conv->forward(ctx, x);
            return ggml_ext_slice(ctx->ggml_ctx, x, 1, 1, x->ne[1]);
        }
    };

    struct AudioResnetBlock2D : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        AudioResnetBlock2D(int64_t in_channels, int64_t out_channels)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<PixelNorm2D>();
            blocks["conv1"] = std::make_shared<HeightCausalConv2D>(in_channels, out_channels, std::pair<int, int>{3, 3});
            blocks["norm2"] = std::make_shared<PixelNorm2D>();
            blocks["conv2"] = std::make_shared<HeightCausalConv2D>(out_channels, out_channels, std::pair<int, int>{3, 3});
            if (in_channels != out_channels) {
                blocks["nin_shortcut"] = std::make_shared<HeightCausalConv2D>(in_channels, out_channels, std::pair<int, int>{1, 1});
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm1 = std::dynamic_pointer_cast<PixelNorm2D>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<PixelNorm2D>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["conv2"]);

            auto h = norm1->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = conv1->forward(ctx, h);
            h      = norm2->forward(ctx, h);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = conv2->forward(ctx, h);

            if (in_channels != out_channels) {
                auto shortcut = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["nin_shortcut"]);
                x             = shortcut->forward(ctx, x);
            }
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    struct Conv1D : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;
        int dilation;
        bool bias;
        std::string prefix;

        Conv1D(int64_t in_channels,
               int64_t out_channels,
               int kernel_size,
               int stride   = 1,
               int padding  = 0,
               int dilation = 1,
               bool bias    = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            this->prefix     = prefix;
            ggml_type wtype  = audio_conv_weight_type(get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F16));
            params["weight"] = ggml_new_tensor_4d(ctx, wtype, kernel_size, in_channels, out_channels, 1);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            x = ggml_conv_1d(ctx->ggml_ctx, params["weight"], x, stride, padding, dilation);
            if (bias) {
                auto b = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, params["bias"]->ne[0], 1, 1);
                x      = ggml_add_inplace(ctx->ggml_ctx, x, b);
            }
            return x;
        }
    };

    struct ConvTranspose1D : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;
        int dilation;
        bool bias;

        ConvTranspose1D(int64_t in_channels,
                        int64_t out_channels,
                        int kernel_size,
                        int stride,
                        int padding,
                        int dilation = 1,
                        bool bias    = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kernel_size, out_channels, in_channels, 1);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            GGML_ASSERT(dilation == 1);
            x = ggml_conv_transpose_1d(ctx->ggml_ctx, params["weight"], x, stride, 0, dilation);
            if (padding > 0) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 0, padding, x->ne[0] - padding);
            }
            if (bias) {
                auto b = ggml_reshape_4d(ctx->ggml_ctx, params["bias"], 1, params["bias"]->ne[0], 1, 1);
                x      = ggml_add_inplace(ctx->ggml_ctx, x, b);
            }
            return x;
        }
    };

    struct SnakeBeta1D : public UnaryBlock {
        int64_t channels;
        float eps = 1e-9f;

        explicit SnakeBeta1D(int64_t channels)
            : channels(channels) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            SD_UNUSED(tensor_storage_map);
            SD_UNUSED(prefix);
            params["alpha"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            params["beta"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto alpha       = ggml_exp(ctx->ggml_ctx, params["alpha"]);
            auto beta        = ggml_exp(ctx->ggml_ctx, params["beta"]);
            alpha            = ggml_reshape_4d(ctx->ggml_ctx, alpha, 1, alpha->ne[0], 1, 1);
            beta             = ggml_reshape_4d(ctx->ggml_ctx, beta, 1, beta->ne[0], 1, 1);
            auto oscillation = ggml_sin(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, alpha));
            oscillation      = ggml_mul(ctx->ggml_ctx, oscillation, oscillation);
            auto eps_tensor  = ggml_ext_scale(ctx->ggml_ctx, ggml_ext_ones(ctx->ggml_ctx, 1, 1, 1, 1), eps);
            oscillation      = ggml_div(ctx->ggml_ctx, oscillation, ggml_add(ctx->ggml_ctx, beta, eps_tensor));
            return ggml_add(ctx->ggml_ctx, x, oscillation);
        }
    };

    struct Activation1D : public GGMLBlock {
        int64_t channels;
        int up_ratio         = 2;
        int down_ratio       = 2;
        int up_kernel_size   = 12;
        int down_kernel_size = 12;

        explicit Activation1D(int64_t channels)
            : channels(channels) {
            blocks["act"] = std::make_shared<SnakeBeta1D>(channels);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            ggml_type down_type                 = audio_conv_weight_type(get_type(prefix + "downsample.lowpass.filter", tensor_storage_map, GGML_TYPE_F16));
            params["downsample.lowpass.filter"] = ggml_new_tensor_3d(ctx, down_type, down_kernel_size, 1, 1);
            params["upsample.filter"]           = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, up_kernel_size, 1, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto act         = std::dynamic_pointer_cast<SnakeBeta1D>(blocks["act"]);
            auto up_filter   = params["upsample.filter"];
            auto down_filter = params["downsample.lowpass.filter"];

            int up_pad       = up_kernel_size / up_ratio - 1;
            int up_pad_left  = up_pad * up_ratio + (up_kernel_size - up_ratio) / 2;
            int up_pad_right = up_pad * up_ratio + (up_kernel_size - up_ratio + 1) / 2;

            x = replicate_pad_1d(ctx, x, up_pad, up_pad);
            x = depthwise_conv_transpose1d(ctx->ggml_ctx, x, up_filter, up_ratio);
            x = ggml_ext_slice(ctx->ggml_ctx, x, 0, up_pad_left, x->ne[0] - up_pad_right);

            x = act->forward(ctx, x);

            int down_pad_left  = down_kernel_size / 2 - (down_kernel_size % 2 == 0 ? 1 : 0);
            int down_pad_right = down_kernel_size / 2;
            x                  = replicate_pad_1d(ctx, x, down_pad_left, down_pad_right);
            x                  = depthwise_conv1d(ctx, x, down_filter, down_ratio, 0);
            return x;
        }
    };

    struct AMPBlock1 : public GGMLBlock {
        int64_t channels;
        std::vector<int> dilation;

        AMPBlock1(int64_t channels, int kernel_size, const std::vector<int>& dilation)
            : channels(channels), dilation(dilation) {
            for (int i = 0; i < 3; ++i) {
                blocks["acts1." + std::to_string(i)]  = std::make_shared<Activation1D>(channels);
                blocks["acts2." + std::to_string(i)]  = std::make_shared<Activation1D>(channels);
                blocks["convs1." + std::to_string(i)] = std::make_shared<Conv1D>(channels,
                                                                                 channels,
                                                                                 kernel_size,
                                                                                 1,
                                                                                 (kernel_size * dilation[i] - dilation[i]) / 2,
                                                                                 dilation[i]);
                blocks["convs2." + std::to_string(i)] = std::make_shared<Conv1D>(channels,
                                                                                 channels,
                                                                                 kernel_size,
                                                                                 1,
                                                                                 kernel_size / 2,
                                                                                 1);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            for (int i = 0; i < 3; ++i) {
                auto act1  = std::dynamic_pointer_cast<Activation1D>(blocks["acts1." + std::to_string(i)]);
                auto act2  = std::dynamic_pointer_cast<Activation1D>(blocks["acts2." + std::to_string(i)]);
                auto conv1 = std::dynamic_pointer_cast<Conv1D>(blocks["convs1." + std::to_string(i)]);
                auto conv2 = std::dynamic_pointer_cast<Conv1D>(blocks["convs2." + std::to_string(i)]);

                auto h = act1->forward(ctx, x);
                h      = conv1->forward(ctx, h);
                h      = act2->forward(ctx, h);
                h      = conv2->forward(ctx, h);
                x      = ggml_add(ctx->ggml_ctx, x, h);
            }
            return x;
        }
    };

    struct Vocoder : public GGMLBlock {
        LTXAudioVAEConfig config;
        bool use_bwe_config;
        bool apply_final_activation;

        explicit Vocoder(const LTXAudioVAEConfig& config,
                         bool use_bwe_config         = false,
                         bool apply_final_activation = true)
            : config(config),
              use_bwe_config(use_bwe_config),
              apply_final_activation(apply_final_activation) {
            const int mel_bins                                           = use_bwe_config ? config.bwe_num_mels : config.mel_bins;
            const int initial_channels                                   = use_bwe_config ? config.bwe_upsample_initial_channel : config.base_upsample_initial_channel;
            const std::vector<int>& upsample_rates                       = use_bwe_config ? config.bwe_upsample_rates : config.base_upsample_rates;
            const std::vector<int>& upsample_kernel_sizes                = use_bwe_config ? config.bwe_upsample_kernel_sizes : config.base_upsample_kernel_sizes;
            const std::vector<int>& resblock_kernel_sizes                = use_bwe_config ? config.bwe_resblock_kernel_sizes : config.base_resblock_kernel_sizes;
            const std::vector<std::vector<int>>& resblock_dilation_sizes = use_bwe_config ? config.bwe_resblock_dilation_sizes : config.base_resblock_dilation_sizes;

            int in_channels    = mel_bins * config.audio_channels;
            blocks["conv_pre"] = std::make_shared<Conv1D>(in_channels,
                                                          initial_channels,
                                                          7,
                                                          1,
                                                          3);

            int current_channels = initial_channels;
            int resblock_index   = 0;
            for (size_t i = 0; i < upsample_rates.size(); ++i) {
                int next_channels                  = initial_channels / (1 << static_cast<int>(i + 1));
                blocks["ups." + std::to_string(i)] = std::make_shared<ConvTranspose1D>(current_channels,
                                                                                       next_channels,
                                                                                       upsample_kernel_sizes[i],
                                                                                       upsample_rates[i],
                                                                                       (upsample_kernel_sizes[i] - upsample_rates[i]) / 2);
                for (size_t j = 0; j < resblock_kernel_sizes.size(); ++j) {
                    blocks["resblocks." + std::to_string(resblock_index)] = std::make_shared<AMPBlock1>(next_channels,
                                                                                                        resblock_kernel_sizes[j],
                                                                                                        resblock_dilation_sizes[j]);
                    ++resblock_index;
                }
                current_channels = next_channels;
            }

            blocks["act_post"]  = std::make_shared<Activation1D>(current_channels);
            blocks["conv_post"] = std::make_shared<Conv1D>(current_channels, config.audio_channels, 7, 1, 3, 1, false);
        }

        ggml_tensor* prepare_input(GGMLRunnerContext* ctx, ggml_tensor* mel) {
            mel       = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, mel, 1, 0, 2, 3));
            auto mels = ggml_ext_chunk(ctx->ggml_ctx, mel, 2, 2);
            mel       = ggml_concat(ctx->ggml_ctx, mels[0], mels[1], 1);
            // mel = ggml_reshape_4d(ctx->ggml_ctx,
            //                       mel,
            //                       mel->ne[0],
            //                       mel->ne[1] * mel->ne[2],
            //                       mel->ne[3],
            //                       1); // [b, c*t, f]
            return mel;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* mel) {
            const std::vector<int>& upsample_rates        = use_bwe_config ? config.bwe_upsample_rates : config.base_upsample_rates;
            const std::vector<int>& resblock_kernel_sizes = use_bwe_config ? config.bwe_resblock_kernel_sizes : config.base_resblock_kernel_sizes;
            mel                                           = prepare_input(ctx, mel);
            auto conv_pre                                 = std::dynamic_pointer_cast<Conv1D>(blocks["conv_pre"]);
            auto act_post                                 = std::dynamic_pointer_cast<Activation1D>(blocks["act_post"]);
            auto conv_post                                = std::dynamic_pointer_cast<Conv1D>(blocks["conv_post"]);

            auto x             = conv_pre->forward(ctx, mel);
            int resblock_index = 0;
            for (size_t i = 0; i < upsample_rates.size(); ++i) {
                // x       = ggml_leaky_relu(ctx->ggml_ctx, x, 0.1f, false);
                auto up = std::dynamic_pointer_cast<ConvTranspose1D>(blocks["ups." + std::to_string(i)]);
                x       = up->forward(ctx, x);

                ggml_tensor* sum = nullptr;
                for (size_t j = 0; j < resblock_kernel_sizes.size(); ++j) {
                    auto resblock  = std::dynamic_pointer_cast<AMPBlock1>(blocks["resblocks." + std::to_string(resblock_index++)]);
                    auto block_out = resblock->forward(ctx, x);
                    sum            = sum == nullptr ? block_out : ggml_add(ctx->ggml_ctx, sum, block_out);
                }
                x = ggml_ext_scale(ctx->ggml_ctx, sum, 1.0f / static_cast<float>(resblock_kernel_sizes.size()));
            }

            x = act_post->forward(ctx, x);
            x = conv_post->forward(ctx, x);
            if (apply_final_activation) {
                x = ggml_clamp(ctx->ggml_ctx, x, -1.0f, 1.0f);
            }
            return x;
        }
    };

    struct AudioDecoder : public GGMLBlock {
        LTXAudioVAEConfig config;

        explicit AudioDecoder(const LTXAudioVAEConfig& config)
            : config(config) {
            int block_in          = config.decoder_channels * config.decoder_channel_multipliers.back();
            blocks["conv_in"]     = std::make_shared<HeightCausalConv2D>(config.latent_channels, block_in, std::pair<int, int>{3, 3});
            blocks["mid.block_1"] = std::make_shared<AudioResnetBlock2D>(block_in, block_in);
            blocks["mid.block_2"] = std::make_shared<AudioResnetBlock2D>(block_in, block_in);

            for (int level = static_cast<int>(config.decoder_channel_multipliers.size()) - 1; level >= 0; --level) {
                int block_out = config.decoder_channels * config.decoder_channel_multipliers[level];
                for (int block_idx = 0; block_idx < config.decoder_num_res_blocks + 1; ++block_idx) {
                    blocks["up." + std::to_string(level) + ".block." + std::to_string(block_idx)] =
                        std::make_shared<AudioResnetBlock2D>(block_in, block_out);
                    block_in = block_out;
                }
                if (level != 0) {
                    blocks["up." + std::to_string(level) + ".upsample"] = std::make_shared<AudioUpsample2D>(block_in);
                }
            }

            blocks["norm_out"] = std::make_shared<PixelNorm2D>();
            blocks["conv_out"] = std::make_shared<HeightCausalConv2D>(block_in, config.audio_channels, std::pair<int, int>{3, 3});
        }

        ggml_tensor* denormalize_latent(GGMLRunnerContext* ctx,
                                        ggml_tensor* latent,
                                        ggml_tensor* mean,
                                        ggml_tensor* stddev) {
            latent = ggml_permute(ctx->ggml_ctx, latent, 0, 2, 1, 3);
            latent = ggml_cont(ctx->ggml_ctx, latent);
            latent = ggml_reshape_4d(ctx->ggml_ctx, latent, config.latent_frequency_bins * config.latent_channels, latent->ne[2], 1, latent->ne[3]);

            mean   = ggml_reshape_4d(ctx->ggml_ctx, mean, mean->ne[0], 1, 1, 1);
            stddev = ggml_reshape_4d(ctx->ggml_ctx, stddev, stddev->ne[0], 1, 1, 1);
            latent = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, latent, stddev), mean);

            latent = ggml_reshape_4d(ctx->ggml_ctx,
                                     latent,
                                     config.latent_frequency_bins,
                                     config.latent_channels,
                                     latent->ne[1],
                                     latent->ne[3]);
            latent = ggml_permute(ctx->ggml_ctx, latent, 0, 2, 1, 3);
            return ggml_cont(ctx->ggml_ctx, latent);
        }

        ggml_tensor* adjust_output_shape(GGMLRunnerContext* ctx,
                                         ggml_tensor* decoded,
                                         int target_time,
                                         int target_freq) {
            int64_t time = std::min<int64_t>(decoded->ne[1], target_time);
            int64_t freq = std::min<int64_t>(decoded->ne[0], target_freq);
            decoded      = ggml_ext_slice(ctx->ggml_ctx, decoded, 0, 0, freq);
            decoded      = ggml_ext_slice(ctx->ggml_ctx, decoded, 1, 0, time);
            return decoded;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* latent,
                             ggml_tensor* mean,
                             ggml_tensor* stddev,
                             int target_time,
                             int target_freq) {
            auto conv_in     = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["conv_in"]);
            auto mid_block_1 = std::dynamic_pointer_cast<AudioResnetBlock2D>(blocks["mid.block_1"]);
            auto mid_block_2 = std::dynamic_pointer_cast<AudioResnetBlock2D>(blocks["mid.block_2"]);
            auto norm_out    = std::dynamic_pointer_cast<PixelNorm2D>(blocks["norm_out"]);
            auto conv_out    = std::dynamic_pointer_cast<HeightCausalConv2D>(blocks["conv_out"]);

            auto x = denormalize_latent(ctx, latent, mean, stddev);
            x      = conv_in->forward(ctx, x);
            x      = mid_block_1->forward(ctx, x);
            x      = mid_block_2->forward(ctx, x);

            for (int level = static_cast<int>(config.decoder_channel_multipliers.size()) - 1; level >= 0; --level) {
                for (int block_idx = 0; block_idx < config.decoder_num_res_blocks + 1; ++block_idx) {
                    auto block = std::dynamic_pointer_cast<AudioResnetBlock2D>(blocks["up." + std::to_string(level) + ".block." + std::to_string(block_idx)]);
                    x          = block->forward(ctx, x);
                }
                if (level != 0) {
                    auto upsample = std::dynamic_pointer_cast<AudioUpsample2D>(blocks["up." + std::to_string(level) + ".upsample"]);
                    x             = upsample->forward(ctx, x);
                }
            }

            x = norm_out->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = conv_out->forward(ctx, x);
            return adjust_output_shape(ctx, x, target_time, target_freq);
        }
    };

    struct LTXAudioVAE : public GGMLBlock {
        LTXAudioVAEConfig config;

        explicit LTXAudioVAE(const LTXAudioVAEConfig& config)
            : config(config) {
            blocks["audio_vae.decoder"] = std::make_shared<AudioDecoder>(config);
            blocks["vocoder.vocoder"]   = std::make_shared<Vocoder>(config);
            if (config.has_bwe) {
                blocks["vocoder.bwe_generator"] = std::make_shared<Vocoder>(config, true, false);
            }
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            GGMLBlock::init_params(ctx, tensor_storage_map, prefix);
            params["audio_vae.per_channel_statistics.mean-of-means"] =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.latent_channels * config.latent_frequency_bins);
            params["audio_vae.per_channel_statistics.std-of-means"] =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.latent_channels * config.latent_frequency_bins);
            if (config.has_bwe) {
                params["vocoder.mel_stft.mel_basis"] =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.bwe_n_fft / 2 + 1, config.bwe_num_mels);
                params["vocoder.mel_stft.stft_fn.forward_basis"] =
                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.bwe_n_fft, 1, (config.bwe_n_fft / 2 + 1) * 2);
                params["vocoder.mel_stft.stft_fn.inverse_basis"] =
                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.bwe_n_fft, 1, (config.bwe_n_fft / 2 + 1) * 2);
            }
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* latent,
                            ggml_tensor* bwe_skip_filter) {
            int target_time = static_cast<int>(latent->ne[1]) * config.latent_downsample_factor() -
                              (config.latent_downsample_factor() - 1);
            int target_freq = config.mel_bins;

            auto decoder  = std::dynamic_pointer_cast<AudioDecoder>(blocks["audio_vae.decoder"]);
            auto mean     = params["audio_vae.per_channel_statistics.mean-of-means"];
            auto stddev   = params["audio_vae.per_channel_statistics.std-of-means"];
            auto mel      = decoder->forward(ctx, latent, mean, stddev, target_time, target_freq);
            auto vocoder  = std::dynamic_pointer_cast<Vocoder>(blocks["vocoder.vocoder"]);
            auto waveform = vocoder->forward(ctx, mel);

            if (config.has_bwe) {
                GGML_ASSERT(bwe_skip_filter != nullptr);
                const int bwe_ratio    = config.bwe_output_sample_rate / config.bwe_input_sample_rate;
                const int64_t low_time = waveform->ne[0];
                const int64_t out_time = low_time * bwe_ratio;
                int64_t remainder      = low_time % config.bwe_hop_length;
                auto bwe_waveform      = waveform;
                if (remainder != 0) {
                    bwe_waveform = ggml_pad_ext(ctx->ggml_ctx,
                                                bwe_waveform,
                                                0,
                                                static_cast<int>(config.bwe_hop_length - remainder),
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0);
                }

                auto mel_basis  = params["vocoder.mel_stft.mel_basis"];
                auto stft_basis = params["vocoder.mel_stft.stft_fn.forward_basis"];
                GGML_ASSERT(mel_basis != nullptr && stft_basis != nullptr);
                auto bwe_mel       = compute_log_mel_spectrogram(ctx, bwe_waveform, stft_basis, mel_basis, config.bwe_hop_length);
                auto bwe_generator = std::dynamic_pointer_cast<Vocoder>(blocks["vocoder.bwe_generator"]);
                auto residual      = bwe_generator->forward(ctx, bwe_mel);

                auto skip = upsample_waveform_hann(ctx,
                                                   bwe_waveform,
                                                   bwe_skip_filter,
                                                   bwe_ratio);
                waveform  = ggml_clamp(ctx->ggml_ctx,
                                       ggml_add(ctx->ggml_ctx, residual, skip),
                                       -1.0f,
                                       1.0f);
                waveform  = crop_waveform_samples(ctx->ggml_ctx, waveform, out_time);
            }

            return waveform;
        }
    };

    struct LTXAudioVAERunner : public GGMLRunner {
        LTXAudioVAEConfig config;
        LTXAudioVAE model;
        sd::Tensor<float> bwe_skip_filter_tensor;

        LTXAudioVAERunner(ggml_backend_t backend,
                          ggml_backend_t params_backend,
                          const String2TensorStorage& tensor_storage_map,
                          const std::string& prefix = "")
            : GGMLRunner(backend, params_backend),
              config(LTXAudioVAEConfig::detect_from_weights(tensor_storage_map)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
            if (config.has_bwe) {
                const int bwe_ratio    = config.bwe_output_sample_rate / config.bwe_input_sample_rate;
                bwe_skip_filter_tensor = sd::Tensor<float>::from_vector(build_hann_resample_filter(bwe_ratio));
            }
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        size_t get_params_buffer_size() {
            return model.get_params_mem_size();
        }

        std::string get_desc() {
            return "ltx_audio_vae";
        }

        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& latent_tensor) {
            int64_t t0     = ggml_time_ms();
            auto get_graph = [&]() -> ggml_cgraph* {
                auto latent                  = make_input(latent_tensor);
                ggml_tensor* bwe_skip_filter = config.has_bwe ? make_input(bwe_skip_filter_tensor) : nullptr;
                ggml_cgraph* gf              = new_graph_custom(655360);
                auto runner_ctx              = GGMLRunner::get_context();
                auto waveform                = model.decode(&runner_ctx, latent, bwe_skip_filter);
                ggml_build_forward_expand(gf, waveform);
                return gf;
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false), 4);
            int64_t t1  = ggml_time_ms();
            LOG_INFO("ltx audio vae decode completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            return result;
        }

        void test(const std::string& input_path) {
            auto z = sd::load_tensor_from_file_as_tensor<float>(input_path);
            GGML_ASSERT(!z.empty());
            print_sd_tensor(z, false, "ltx_audio_vae_z");

            int64_t t0 = ggml_time_ms();
            auto out   = decode(8, z);
            int64_t t1 = ggml_time_ms();

            GGML_ASSERT(!out.empty());
            print_sd_tensor(out, false, "ltx_audio_vae_out");
            LOG_DEBUG("ltx audio vae test done in %lldms", t1 - t0);
        }

        static void load_from_file_and_test(const std::string& model_path,
                                            const std::string& input_path,
                                            const std::string& prefix = "") {
            ggml_backend_t backend = sd_backend_cpu_init();
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            LOG_INFO("loading ltx audio vae from '%s'", model_path.c_str());

            ModelLoader model_loader;
            if (!model_loader.init_from_file(model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            auto ltx_audio_vae       = std::make_shared<LTXAudioVAERunner>(backend,
                                                                     backend,
                                                                     tensor_storage_map,
                                                                     prefix);

            if (!ltx_audio_vae->alloc_params_buffer()) {
                LOG_ERROR("ltx audio vae buffer allocation failed");
                return;
            }

            std::map<std::string, ggml_tensor*> tensors;
            ltx_audio_vae->get_param_tensors(tensors, "");

            if (!model_loader.load_tensors(tensors)) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("ltx audio vae model loaded");
            ltx_audio_vae->test(input_path);
        }
    };

}  // namespace LTXV

#endif  // __SD_MODEL_VAE_LTX_AUDIO_VAE_HPP__

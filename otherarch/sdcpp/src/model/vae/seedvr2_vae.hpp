#ifndef __SD_MODEL_VAE_SEEDVR2_VAE_HPP__
#define __SD_MODEL_VAE_SEEDVR2_VAE_HPP__

#include "model/common/ggml_block.hpp"

namespace SeedVR2 {

    struct SeedVR2VAEConfig {
        std::vector<int> channels;
        int latent_channels = 16;
        int layers          = 2;

        static SeedVR2VAEConfig detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            SeedVR2VAEConfig config;
            auto at = [&](const std::string& key) -> const TensorStorage& {
                auto found = weights.find(prefix + "." + key);
                if (found == weights.end())
                    throw std::runtime_error("Missing SeedVR2 VAE tensor: " + prefix + "." + key);
                return found->second;
            };
            for (int i = 0; i < 4; ++i)
                config.channels.push_back(int(at("encoder.down_blocks." + std::to_string(i) + ".resnets.0.norm2.weight").ne[0]));
            if (config.channels != std::vector<int>({128, 256, 512, 512}) || at("encoder.conv_out.bias").ne[0] != 32 ||
                at("decoder.conv_in.weight").ne[3] != 16 * 512 || at("decoder.conv_out.bias").ne[0] != 3 ||
                !weights.count(prefix + ".decoder.up_blocks.0.upsamplers.0.upscale_conv.weight"))
                throw std::runtime_error("SeedVR2 requires its matching 16-channel causal VAE");
            LOG_VERBOSE("seedvr2 vae: latent_channels = %d, layers = %d, spatial factor = 8, temporal factor = 4", config.latent_channels, config.layers);
            return config;
        }
    };

    class InflatedCausalConv3d : public Conv3d {
        int temporal_padding_;

    public:
        InflatedCausalConv3d(int in, int out, std::tuple<int, int, int> kernel = {3, 3, 3}, std::tuple<int, int, int> stride = {1, 1, 1}, std::tuple<int, int, int> padding = {1, 1, 1})
            : Conv3d(in, out, kernel, stride, {0, std::get<1>(padding), std::get<2>(padding)}), temporal_padding_(2 * std::get<0>(padding)) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto c       = ctx->ggml_ctx;
            const int kt = std::get<0>(kernel_size), kh = std::get<1>(kernel_size), kw = std::get<2>(kernel_size);
            if (ctx->conv3d_direct_enabled && !ctx->weight_adapter && x->ne[2] == 1 && kt > 1 &&
                temporal_padding_ == kt - 1 && std::get<0>(dilation) == 1) {
                // All causal taps read the same frame. Sum in F32 before rounding the
                // equivalent spatial filter to F16, instead of copying and convolving the halo.
                auto w       = ggml_cast(c, params["weight"], GGML_TYPE_F32);
                w            = ggml_reshape_3d(c, w, kw * kh, kt, in_channels * out_channels);
                auto spatial = ggml_ext_slice(ctx, w, 1, 0, 1, false);
                for (int i = 1; i < kt; ++i)
                    spatial = ggml_add(c, spatial, ggml_ext_slice(ctx, w, 1, i, i + 1, false));
                w = ggml_cast(c, ggml_reshape_4d(c, spatial, kw, kh, 1, in_channels * out_channels), GGML_TYPE_F16);
                if (ctx->conv2d_direct_enabled) {
                    w = ggml_reshape_4d(c, w, kw, kh, in_channels, out_channels);
                    x = ggml_reshape_4d(c, ggml_ext_cont(ctx, x), x->ne[0], x->ne[1], in_channels, 1);
                    x = ggml_ext_conv_2d(ctx, x, w, bias ? params["bias"] : nullptr,
                                         std::get<2>(stride), std::get<1>(stride),
                                         std::get<2>(padding), std::get<1>(padding),
                                         std::get<2>(dilation), std::get<1>(dilation), true);
                    return ggml_reshape_4d(c, x, x->ne[0], x->ne[1], 1, out_channels);
                }
                return ggml_ext_conv_3d(ctx, x, w, bias ? params["bias"] : nullptr, in_channels,
                                        std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                        std::get<2>(padding), std::get<1>(padding), 0,
                                        std::get<2>(dilation), std::get<1>(dilation), 1, force_prec_f32, true);
            }
            if (temporal_padding_) {
                const int64_t w = x->ne[0], h = x->ne[1], t = x->ne[2], ch = x->ne[3];
                // Merge spatial axes so concat does not launch separately for every channel.
                x          = ggml_reshape_3d(c, ggml_ext_cont(ctx, x), w * h, t, ch);
                auto head  = ggml_ext_slice(ctx, x, 1, 0, 1);
                auto shape = ggml_new_tensor_3d(c, x->type, w * h, temporal_padding_, ch);
                x          = ggml_concat(c, ggml_repeat(c, head, shape), x, 1);
                x          = ggml_reshape_4d(c, x, w, h, t + temporal_padding_, ch);
            }
            if (ctx->conv3d_direct_enabled)
                return Conv3d::forward(ctx, x);
            const int st = std::get<0>(stride), sh = std::get<1>(stride), sw = std::get<2>(stride);
            const int ph = std::get<1>(padding), pw = std::get<2>(padding);
            const int64_t oh              = (x->ne[1] + 2 * ph - kh) / sh + 1;
            const int64_t ow              = (x->ne[0] + 2 * pw - kw) / sw + 1;
            const int64_t ot              = (x->ne[2] - kt) / st + 1;
            const int64_t row_bytes       = ow * in_channels * kt * kh * kw * 2;
            const int64_t max_patch_bytes = 256LL * 1024 * 1024;
            if (row_bytes * oh * ot <= max_patch_bytes)
                return Conv3d::forward(ctx, x);
            // Bound im2col storage without spatial tiling: every strip keeps the exact
            // convolution halo, and normalization still sees the complete frame.
            const int64_t rows = std::max<int64_t>(1, max_patch_bytes / row_bytes);
            struct RestorePadding {
                std::tuple<int, int, int>& target;
                std::tuple<int, int, int> value;
                ~RestorePadding() { target = value; }
            } restore{padding, padding};
            padding     = {0, 0, pw};
            auto output = ggml_ext_zeros(ctx, ow, oh, ot, out_channels);
            for (int64_t t = 0; t < ot; ++t) {
                auto frame = ggml_ext_slice(ctx, x, 2, t * st, t * st + kt, false);
                for (int64_t y = 0; y < oh; y += rows) {
                    const int64_t end = std::min(oh, y + rows);
                    const int64_t top = y * sh - ph, bottom = (end - 1) * sh - ph + kh;
                    auto strip = ggml_ext_slice(ctx, frame, 1, std::max<int64_t>(0, top), std::min(x->ne[1], bottom), false);
                    strip      = ggml_ext_pad_ext(ctx, strip, 0, 0, int(std::max<int64_t>(0, -top)),
                                                  int(std::max<int64_t>(0, bottom - x->ne[1])), 0, 0, 0, 0);
                    strip      = ggml_ext_cont(ctx, strip);
                    strip      = Conv3d::forward(ctx, strip);
                    output     = ggml_set_inplace(c, output, strip, output->nb[1], output->nb[2], output->nb[3], y * output->nb[1] + t * output->nb[2]);
                }
            }
            return output;
        }
    };

    static ggml_tensor* causal_norm(GGMLRunnerContext* ctx, ggml_tensor* x, const std::shared_ptr<GGMLBlock>& norm) {
        auto c = ctx->ggml_ctx;
        // [B, C, T, H, W] -> [B*T, C, H, W]: GroupNorm never mixes frames.
        x = ggml_cont(c, ggml_permute(c, x, 0, 1, 3, 2));
        x = std::dynamic_pointer_cast<GroupNorm>(norm)->forward(ctx, x);
        return ggml_cont(c, ggml_permute(c, x, 0, 1, 3, 2));
    }

    class ResnetBlock3D : public UnaryBlock {
    public:
        ResnetBlock3D(int in, int out) {
            blocks["norm1"] = std::make_shared<GroupNorm32>(in);
            blocks["norm2"] = std::make_shared<GroupNorm32>(out);
            blocks["conv1"] = std::make_shared<InflatedCausalConv3d>(in, out);
            blocks["conv2"] = std::make_shared<InflatedCausalConv3d>(out, out);
            if (in != out)
                blocks["conv_shortcut"] = std::make_shared<InflatedCausalConv3d>(in, out, std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{0, 0, 0});
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto h = ggml_silu(ctx->ggml_ctx, causal_norm(ctx, x, blocks["norm1"]));
            h      = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv1"])->forward(ctx, h);
            h      = ggml_silu(ctx->ggml_ctx, causal_norm(ctx, h, blocks["norm2"]));
            h      = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv2"])->forward(ctx, h);
            if (blocks.count("conv_shortcut"))
                x = std::dynamic_pointer_cast<UnaryBlock>(blocks["conv_shortcut"])->forward(ctx, x);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    class SpatialAttention : public UnaryBlock {
    public:
        explicit SpatialAttention(int channels) {
            blocks["group_norm"] = std::make_shared<GroupNorm32>(channels);
            for (const auto& name : {"to_q", "to_k", "to_v", "to_out.0"})
                blocks[name] = std::make_shared<Linear>(channels, channels);
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto c              = ctx->ggml_ctx;
            const int64_t width = x->ne[0], height = x->ne[1], frames = x->ne[2], channels = x->ne[3];
            auto h      = causal_norm(ctx, x, blocks["group_norm"]);
            h           = ggml_reshape_3d(c, h, width * height, frames, channels);
            h           = ggml_cont(c, ggml_permute(c, h, 1, 2, 0, 3));
            auto linear = [&](const char* name, ggml_tensor* value) { return std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, value); };
            h           = ggml_ext_attention_ext(ctx, linear("to_q", h), linear("to_k", h), linear("to_v", h), 1, nullptr, false, ctx->flash_attn_enabled);
            h           = linear("to_out.0", h);
            h           = ggml_cont(c, ggml_permute(c, h, 2, 0, 1, 3));
            return ggml_add(c, x, ggml_reshape_4d(c, h, width, height, frames, channels));
        }
    };

    class Upsample3D : public UnaryBlock {
        int temporal_;

    public:
        Upsample3D(int channels, bool temporal)
            : temporal_(temporal ? 2 : 1) {
            blocks["upscale_conv"] = std::make_shared<InflatedCausalConv3d>(channels, channels * 4 * temporal_, std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{1, 1, 1}, std::tuple<int, int, int>{0, 0, 0});
            blocks["conv"]         = std::make_shared<InflatedCausalConv3d>(channels, channels);
        }
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto c          = ctx->ggml_ctx;
            const int64_t w = x->ne[0], h = x->ne[1], t = x->ne[2], ch = x->ne[3];
            x = std::dynamic_pointer_cast<UnaryBlock>(blocks["upscale_conv"])->forward(ctx, x);
            // [B, (y x z C), T, H, W] -> [B, C, T*z, H*y, W*x].
            x = ggml_reshape_4d(c, x, w * h * t, ch, temporal_, 4);
            x = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
            x = ggml_reshape_4d(c, x, temporal_, w * h, t, ch * 4);
            x = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));
            x = ggml_reshape_4d(c, x, w, h * t * temporal_ * ch, 2, 2);
            x = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
            x = ggml_reshape_4d(c, x, w * 2, h, t * temporal_ * ch, 2);
            x = ggml_cont(c, ggml_permute(c, x, 0, 2, 3, 1));
            x = ggml_reshape_4d(c, x, w * 2, h * 2, t * temporal_, ch);
            if (temporal_ == 2) {
                auto first = ggml_ext_slice(ctx, x, 2, 0, 1);
                x          = t == 1 ? first : ggml_concat(c, first, ggml_ext_slice(ctx, x, 2, 2, t * 2), 2);
            }
            return std::dynamic_pointer_cast<UnaryBlock>(blocks["conv"])->forward(ctx, x);
        }
    };

    class VideoAutoencoderKL : public GGMLBlock {
        SeedVR2VAEConfig config_;

    public:
        explicit VideoAutoencoderKL(const SeedVR2VAEConfig& config)
            : config_(config) {
            for (bool decode : {false, true}) {
                std::string root = decode ? "decoder." : "encoder.";
                auto channels    = config.channels;
                if (decode)
                    std::reverse(channels.begin(), channels.end());
                int previous             = channels.front();
                blocks[root + "conv_in"] = std::make_shared<InflatedCausalConv3d>(decode ? 16 : 3, previous);
                for (int i = 0; i < 4; ++i) {
                    auto stage = root + (decode ? "up_blocks." : "down_blocks.") + std::to_string(i) + ".";
                    for (int j = 0; j < config.layers + int(decode); ++j) {
                        blocks[stage + "resnets." + std::to_string(j)] = std::make_shared<ResnetBlock3D>(previous, channels[i]);
                        previous                                       = channels[i];
                    }
                    if (i < 3) {
                        if (decode)
                            blocks[stage + "upsamplers.0"] = std::make_shared<Upsample3D>(previous, i < 2);
                        else
                            blocks[stage + "downsamplers.0.conv"] = std::make_shared<InflatedCausalConv3d>(previous, previous,
                                                                                                           std::tuple<int, int, int>{i ? 3 : 1, 3, 3}, std::tuple<int, int, int>{i ? 2 : 1, 2, 2}, std::tuple<int, int, int>{i ? 1 : 0, 0, 0});
                    }
                }
                blocks[root + "mid_block.resnets.0"]    = std::make_shared<ResnetBlock3D>(512, 512);
                blocks[root + "mid_block.attentions.0"] = std::make_shared<SpatialAttention>(512);
                blocks[root + "mid_block.resnets.1"]    = std::make_shared<ResnetBlock3D>(512, 512);
                blocks[root + "conv_norm_out"]          = std::make_shared<GroupNorm32>(previous);
                blocks[root + "conv_out"]               = std::make_shared<InflatedCausalConv3d>(previous, decode ? 3 : 32);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool decode) {
            auto c           = ctx->ggml_ctx;
            std::string root = decode ? "decoder." : "encoder.";
            auto apply       = [&](const std::string& name) {
                x = std::dynamic_pointer_cast<UnaryBlock>(blocks[root + name])->forward(ctx, x);
                sd::ggml_graph_cut::mark_graph_cut(x, "seedvr2.vae." + root + name, "hidden");
            };
            auto mid = [&]() {
                apply("mid_block.resnets.0");
                apply("mid_block.attentions.0");
                apply("mid_block.resnets.1");
            };
            apply("conv_in");
            if (decode)
                mid();
            for (int i = 0; i < 4; ++i) {
                auto stage = std::string(decode ? "up_blocks." : "down_blocks.") + std::to_string(i) + ".";
                for (int j = 0; j < config_.layers + int(decode); ++j)
                    apply(stage + "resnets." + std::to_string(j));
                if (i < 3) {
                    if (!decode)
                        x = ggml_pad(c, x, 1, 1, 0, 0);
                    apply(stage + (decode ? "upsamplers.0" : "downsamplers.0.conv"));
                }
            }
            if (!decode)
                mid();
            x = ggml_silu(c, causal_norm(ctx, x, blocks[root + "conv_norm_out"]));
            apply("conv_out");
            return decode ? x : ggml_ext_slice(ctx, x, 3, 0, 16);
        }
    };

    struct SeedVR2VAERunner : GGMLRunner {
        SeedVR2VAEConfig config;
        VideoAutoencoderKL model;

        SeedVR2VAERunner(ggml_backend_t backend, const String2TensorStorage& weights, std::shared_ptr<DeviceResidencyManager> manager)
            : GGMLRunner(backend, manager), config(SeedVR2VAEConfig::detect_from_weights(weights, "vae")), model(config) {
            model.init(params_ctx, weights, "vae");
        }
        std::string get_desc() override { return "SeedVR2 VAE"; }
        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) { model.get_param_tensors(tensors, "vae"); }
        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& input, bool decode) {
            auto graph = [&]() {
                auto gf  = new_graph_custom(MAX_GRAPH_SIZE * 4);
                auto ctx = get_context(gf);
                auto out = model.forward(&ctx, make_input(input), decode);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto output = GGMLRunner::compute(graph, n_threads);
            return restore_trailing_singleton_dims(output, 4);
        }
    };
}

#endif  // __SD_MODEL_VAE_SEEDVR2_VAE_HPP__

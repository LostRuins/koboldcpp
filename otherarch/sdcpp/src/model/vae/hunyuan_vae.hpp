#ifndef __SD_MODEL_VAE_HUNYUAN_VAE_HPP__
#define __SD_MODEL_VAE_HUNYUAN_VAE_HPP__

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "model/vae/wan_vae.hpp"
#include "model_manager.h"

namespace Hunyuan {
    constexpr int HUNYUAN_VIDEO_VAE_GRAPH_SIZE                  = 65536;
    constexpr int HUNYUAN_VIDEO_VAE_GRAPH_SIZE_PER_LATENT_FRAME = 8192;
    constexpr int HUNYUAN_VIDEO_VAE_TEMPORAL_CHUNK_SIZE         = 1;

    struct TemporalConvCarry {
        const std::vector<ggml_tensor*>* input = nullptr;
        std::vector<ggml_tensor*>* output      = nullptr;
        size_t input_index                     = 0;

        bool is_continuation() const {
            return input != nullptr;
        }

        ggml_tensor* take() {
            GGML_ASSERT(input != nullptr && input_index < input->size());
            return (*input)[input_index++];
        }

        void push(ggml_tensor* tensor) {
            if (output != nullptr) {
                output->push_back(tensor);
            }
        }

        void finish() const {
            GGML_ASSERT(input == nullptr || input_index == input->size());
        }
    };

    static ggml_tensor* repeat_interleave_channels(GGMLRunnerContext* ctx,
                                                   ggml_tensor* x,
                                                   int64_t repeats,
                                                   int64_t width,
                                                   int64_t height,
                                                   int64_t frames) {
        GGML_ASSERT(repeats > 0);
        GGML_ASSERT(width * height * frames == x->ne[0] * x->ne[1] * x->ne[2]);
        int64_t channels = x->ne[3];
        if (repeats == 1) {
            return ggml_reshape_4d(ctx->ggml_ctx, x, width, height, frames, channels);
        }
        x           = ggml_reshape_3d(ctx->ggml_ctx, x, width * height * frames, 1, channels);
        auto target = ggml_new_tensor_3d(ctx->ggml_ctx, x->type, width * height * frames, repeats, channels);
        x           = ggml_repeat(ctx->ggml_ctx, x, target);
        return ggml_reshape_4d(ctx->ggml_ctx, x, width, height, frames, channels * repeats);
    }

    class CausalConv3d : public GGMLBlock {
    protected:
        std::tuple<int, int, int> kernel_size;

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> padding  = {0, 0, 0},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true)
            : kernel_size(kernel_size) {
            blocks["conv"] = std::make_shared<Conv3d>(in_channels, out_channels, kernel_size, stride, padding, dilation, bias);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             TemporalConvCarry* carry = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, OD, OH, OW]
            // assert N == 1

            auto conv = std::dynamic_pointer_cast<Conv3d>(blocks["conv"]);

            int pad_w = std::get<2>(kernel_size) / 2;
            int pad_h = std::get<1>(kernel_size) / 2;
            int pad_t = std::get<0>(kernel_size) - 1;
            std::vector<ggml_tensor*> temporal_frames;
            temporal_frames.reserve(x->ne[2] + pad_t);
            if (pad_t > 0) {
                if (carry != nullptr && carry->is_continuation()) {
                    auto previous = carry->take();
                    GGML_ASSERT(previous->ne[2] <= pad_t);
                    for (int64_t frame = 0; frame < previous->ne[2]; frame++) {
                        temporal_frames.push_back(ggml_ext_slice(ctx->ggml_ctx, previous, 2, frame, frame + 1));
                    }
                    for (int64_t frame = previous->ne[2]; frame < pad_t; frame++) {
                        temporal_frames.push_back(ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1));
                    }
                } else {
                    auto first = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                    for (int frame = 0; frame < pad_t; frame++) {
                        temporal_frames.push_back(first);
                    }
                }
            }
            for (int64_t frame = 0; frame < x->ne[2]; frame++) {
                temporal_frames.push_back(ggml_ext_slice(ctx->ggml_ctx, x, 2, frame, frame + 1));
            }

            if (pad_t > 0 && carry != nullptr && carry->output != nullptr) {
                ggml_tensor* next = nullptr;
                for (int frame = pad_t; frame > 0; frame--) {
                    auto item = temporal_frames[temporal_frames.size() - frame];
                    next      = next == nullptr ? item : ggml_concat(ctx->ggml_ctx, next, item, 2);
                }
                carry->push(ggml_cont(ctx->ggml_ctx, next));
            }

            ggml_tensor* padded = nullptr;
            for (auto frame : temporal_frames) {
                padded = padded == nullptr ? frame : ggml_concat(ctx->ggml_ctx, padded, frame, 2);
            }
            auto replicate_pad = [&](ggml_tensor* input, int dim, int left, int right) {
                if (left > 0) {
                    auto first = ggml_ext_slice(ctx->ggml_ctx, input, dim, 0, 1);
                    for (int i = 0; i < left; i++) {
                        input = ggml_concat(ctx->ggml_ctx, first, input, dim);
                    }
                }
                if (right > 0) {
                    auto last = ggml_ext_slice(ctx->ggml_ctx, input, dim, input->ne[dim] - 1, input->ne[dim]);
                    for (int i = 0; i < right; i++) {
                        input = ggml_concat(ctx->ggml_ctx, input, last, dim);
                    }
                }
                return input;
            };
            padded = replicate_pad(padded, 0, pad_w, pad_w);
            padded = replicate_pad(padded, 1, pad_h, pad_h);
            return conv->forward(ctx, padded);
        }
    };

    class AttnBlock : public UnaryBlock {
    protected:
        int64_t in_channels;

    public:
        AttnBlock(int64_t in_channels)
            : in_channels(in_channels) {
            blocks["norm"]     = std::make_shared<WAN::RMS_norm>(in_channels);
            blocks["q"]        = std::make_shared<Conv3d>(in_channels, in_channels, std::tuple{1, 1, 1});
            blocks["k"]        = std::make_shared<Conv3d>(in_channels, in_channels, std::tuple{1, 1, 1});
            blocks["v"]        = std::make_shared<Conv3d>(in_channels, in_channels, std::tuple{1, 1, 1});
            blocks["proj_out"] = std::make_shared<Conv3d>(in_channels, in_channels, std::tuple{1, 1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x) override {
            // x: [b*c, t, h, w]
            auto norm     = std::dynamic_pointer_cast<WAN::RMS_norm>(blocks["norm"]);
            auto q_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["q"]);
            auto k_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["k"]);
            auto v_proj   = std::dynamic_pointer_cast<UnaryBlock>(blocks["v"]);
            auto proj_out = std::dynamic_pointer_cast<UnaryBlock>(blocks["proj_out"]);

            const int64_t b = x->ne[3] / in_channels;

            auto identity = x;

            x = norm->forward(ctx, x);

            const int64_t c = x->ne[3] / b;
            const int64_t t = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto q = q_proj->forward(ctx, x);  // [b*c, t, h, w]
            auto k = k_proj->forward(ctx, x);  // [b*c, t, h, w]
            auto v = v_proj->forward(ctx, x);  // [b*c, t, h, w]

            q = ggml_reshape_3d(ctx->ggml_ctx, q, w * h * t, c, b);                                  // [b, c, t*h*w]
            q = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 1, 0, 2, 3));  // [b, t*h*w, c]

            k = ggml_reshape_3d(ctx->ggml_ctx, k, w * h * t, c, b);                                  // [b, c, t*h*w]
            k = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 1, 0, 2, 3));  // [b, t*h*w, c]

            v = ggml_reshape_3d(ctx->ggml_ctx, v, w * h * t, c, b);                                  // [b, c, t*h*w]
            v = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));  // [b, t*h*w, c]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);  // [b, t*h*w, c]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [b, c, t*h*w]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, t, c * b);                         // [b*c, t, h, w]

            x = proj_out->forward(ctx, x);

            x = ggml_add(ctx->ggml_ctx, x, identity);
            return x;
        }
    };

    class ResnetBlock : public UnaryBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;

    public:
        ResnetBlock(int64_t in_channels,
                    int64_t out_channels)
            : in_channels(in_channels),
              out_channels(out_channels) {
            blocks["norm1"] = std::make_shared<WAN::RMS_norm>(in_channels);
            blocks["conv1"] = std::make_shared<CausalConv3d>(in_channels, out_channels, std::tuple{3, 3, 3});

            blocks["norm2"] = std::make_shared<WAN::RMS_norm>(out_channels);
            blocks["conv2"] = std::make_shared<CausalConv3d>(out_channels, out_channels, std::tuple{3, 3, 3});

            if (out_channels != in_channels) {
                blocks["nin_shortcut"] = std::make_shared<CausalConv3d>(in_channels, out_channels, std::tuple{1, 1, 1});
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            return forward(ctx, x, nullptr);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             TemporalConvCarry* carry) {
            // x: [B*IC, IT, OH, OW]
            // return: [B*OC, OT, OH, OW]
            auto norm1 = std::dynamic_pointer_cast<WAN::RMS_norm>(blocks["norm1"]);
            auto conv1 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto norm2 = std::dynamic_pointer_cast<WAN::RMS_norm>(blocks["norm2"]);
            auto conv2 = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto h = x;
            h      = norm1->forward(ctx, h);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
            h      = conv1->forward(ctx, h, carry);

            h = norm2->forward(ctx, h);
            h = ggml_silu_inplace(ctx->ggml_ctx, h);  // swish
            // dropout, skip for inference
            h = conv2->forward(ctx, h, carry);

            // skip connection
            if (out_channels != in_channels) {
                auto nin_shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["nin_shortcut"]);

                x = nin_shortcut->forward(ctx, x);  // [B*OC, OT, OH, OW]
            }

            h = ggml_add(ctx->ggml_ctx, h, x);
            return h;  // [B*OC, OT, OH, OW]
        }
    };

    class Upsample : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s;
        int64_t factor;
        int64_t repeats;

    public:
        Upsample(int64_t in_channels, int64_t out_channels, bool add_temporal_upsample)
            : in_channels(in_channels), out_channels(out_channels) {
            if (add_temporal_upsample) {
                factor_t = 2;
            } else {
                factor_t = 1;
            }
            factor_s = 2;
            factor   = factor_t * factor_s * factor_s;
            GGML_ASSERT(out_channels * factor % in_channels == 0);
            repeats        = out_channels * factor / in_channels;
            blocks["conv"] = std::make_shared<CausalConv3d>(in_channels, out_channels * factor, std::tuple{3, 3, 3});
        }

        static ggml_tensor* _pixel_shuffle_3d(GGMLRunnerContext* ctx,
                                              ggml_tensor* x,
                                              int64_t factor_t,
                                              int64_t factor_s,
                                              int64_t B = 1) {
            // x: [B*factor*C, T, H, W]
            // return: [B*C, T*factor_t, H*factor_s, W*factor_s]
            GGML_ASSERT(B == 1);
            int64_t factor = factor_t * factor_s * factor_s;
            int64_t C      = x->ne[3] / factor;
            int64_t T      = x->ne[2];
            int64_t H      = x->ne[1];
            int64_t W      = x->ne[0];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, C, factor);                          // [factor, C, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // [C, factor, T*H, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor_s, factor_s * factor_t * C);  // [C*factor_t*factor_s, factor_s, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C*factor_t*factor_s, T*H, W, factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, H * T, factor_s, factor_t * C);  // [C*factor_t, factor_s, T*H, W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C*factor_t, T*H, factor_s, W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, T, factor_t, C);  // [C, factor_t, T, H*factor_s*W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C, T, factor_t, H*factor_s*W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s * H, factor_t * T, C);  // [C, T*factor_t, H*factor_s, W*factor_s]

            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             TemporalConvCarry* carry = nullptr) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, 1 + (T - 1)*factor_t, H*factor_s, W*factor_s]
            const int64_t B = x->ne[3] / in_channels;
            GGML_ASSERT(B == 1);

            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);

            const bool continuation = carry != nullptr && carry->is_continuation();
            auto h                  = conv->forward(ctx, x, carry);  // [B*factor*OC, T, H, W]

            ggml_tensor* shortcut = nullptr;
            if (factor_t == 2 && !continuation) {
                auto h_first = ggml_ext_slice(ctx->ggml_ctx, h, 2, 0, 1);                   // [B*factor*OC, 1, H, W]
                h_first      = _pixel_shuffle_3d(ctx, h_first, 1, factor_s, B);             // [B*2*OC, 1, H*factor_s, W*factor_s]
                h_first      = ggml_ext_slice(ctx->ggml_ctx, h_first, 3, 0, out_channels);  // [B*OC, 1, H*factor_s, W*factor_s]

                auto x_first = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x_first      = repeat_interleave_channels(ctx, x_first, repeats / 2, x->ne[0], x->ne[1], 1);
                x_first      = _pixel_shuffle_3d(ctx, x_first, 1, factor_s, B);

                if (x->ne[2] == 1) {
                    return ggml_add(ctx->ggml_ctx, h_first, x_first);
                }

                auto h_next = ggml_ext_slice(ctx->ggml_ctx, h, 2, 1, h->ne[2]);       // [B*factor*OC, T - 1, H, W]
                h_next      = _pixel_shuffle_3d(ctx, h_next, factor_t, factor_s, B);  // [B*OC, (T - 1)*factor_t, H*factor_s, W*factor_s]

                h = ggml_concat(ctx->ggml_ctx, h_first, h_next, 2);  // [B*OC, 1 + (T - 1)*factor_t, H*factor_s, W*factor_s]

                auto x_next = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, x->ne[2]);
                x_next      = repeat_interleave_channels(ctx, x_next, repeats, x->ne[0], x->ne[1], x->ne[2] - 1);
                x_next      = _pixel_shuffle_3d(ctx, x_next, factor_t, factor_s, B);

                shortcut = ggml_concat(ctx->ggml_ctx, x_first, x_next, 2);  // [B*OC, 1 + (T - 1)*factor_t, H*factor_s, W*factor_s]
            } else {
                h        = _pixel_shuffle_3d(ctx, h, factor_t, factor_s, B);
                shortcut = repeat_interleave_channels(ctx, x, repeats, x->ne[0], x->ne[1], x->ne[2]);
                shortcut = _pixel_shuffle_3d(ctx, shortcut, factor_t, factor_s, B);  // [B*OC, T*factor_t, H*factor_s, W*factor_s]
            }

            return ggml_add(ctx->ggml_ctx, h, shortcut);
        }
    };

    static ggml_tensor* pixel_unshuffle_3d(GGMLRunnerContext* ctx,
                                           ggml_tensor* x,
                                           int64_t factor_t,
                                           int64_t factor_s) {
        GGML_ASSERT(x->ne[0] % factor_s == 0);
        GGML_ASSERT(x->ne[1] % factor_s == 0);
        GGML_ASSERT(x->ne[2] % factor_t == 0);
        int64_t W      = x->ne[0] / factor_s;
        int64_t H      = x->ne[1] / factor_s;
        int64_t T      = x->ne[2] / factor_t;
        int64_t C      = x->ne[3];
        int64_t factor = factor_t * factor_s * factor_s;

        x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, factor_t, T, C);
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s, H * T, factor_t * C);
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));
        x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s, W, H * T, factor_s * factor_t * C);
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));
        x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor, C);
        x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));
        return ggml_reshape_4d(ctx->ggml_ctx, x, W, H, T, C * factor);
    }

    static ggml_tensor* mean_channel_groups(GGMLRunnerContext* ctx,
                                            ggml_tensor* x,
                                            int64_t group_size) {
        GGML_ASSERT(group_size > 0);
        GGML_ASSERT(x->ne[3] % group_size == 0);
        if (group_size == 1) {
            return x;
        }
        int64_t W       = x->ne[0];
        int64_t H       = x->ne[1];
        int64_t T       = x->ne[2];
        int64_t spatial = W * H * T;
        int64_t groups  = x->ne[3] / group_size;
        x               = ggml_reshape_3d(ctx->ggml_ctx, x, spatial, group_size, groups);
        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x               = ggml_sum_rows(ctx->ggml_ctx, x);
        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        x               = ggml_reshape_4d(ctx->ggml_ctx, x, W, H, T, groups);
        return ggml_scale(ctx->ggml_ctx, x, 1.f / static_cast<float>(group_size));
    }

    class Downsample : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s = 2;
        int64_t factor;
        int64_t group_size;

    public:
        Downsample(int64_t in_channels, int64_t out_channels, bool add_temporal_downsample)
            : in_channels(in_channels),
              out_channels(out_channels),
              factor_t(add_temporal_downsample ? 2 : 1),
              factor(factor_t * factor_s * factor_s),
              group_size(factor * in_channels / out_channels) {
            GGML_ASSERT(out_channels % factor == 0);
            GGML_ASSERT(factor * in_channels % out_channels == 0);
            blocks["conv"] = std::make_shared<CausalConv3d>(in_channels,
                                                            out_channels / factor,
                                                            std::tuple{3, 3, 3});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv"]);
            auto h    = conv->forward(ctx, x);

            ggml_tensor* h_first = nullptr;
            ggml_tensor* x_first = nullptr;
            if (factor_t == 2) {
                h_first = ggml_ext_slice(ctx->ggml_ctx, h, 2, 0, 1);
                h_first = pixel_unshuffle_3d(ctx, h_first, 1, factor_s);
                h_first = ggml_concat(ctx->ggml_ctx, h_first, h_first, 3);

                x_first = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);
                x_first = pixel_unshuffle_3d(ctx, x_first, 1, factor_s);
                x_first = mean_channel_groups(ctx, x_first, group_size / 2);

                if (x->ne[2] == 1) {
                    return ggml_add(ctx->ggml_ctx, h_first, x_first);
                }
                h = ggml_ext_slice(ctx->ggml_ctx, h, 2, 1, h->ne[2]);
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1, x->ne[2]);
            }

            GGML_ASSERT(h->ne[2] % factor_t == 0);
            h = pixel_unshuffle_3d(ctx, h, factor_t, factor_s);
            x = pixel_unshuffle_3d(ctx, x, factor_t, factor_s);
            x = mean_channel_groups(ctx, x, group_size);

            if (factor_t == 2) {
                h = ggml_concat(ctx->ggml_ctx, h_first, h, 2);
                x = ggml_concat(ctx->ggml_ctx, x_first, x, 2);
            }
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    class MidBlock : public UnaryBlock {
    protected:
        int64_t in_channels;
        int num_layers;
        bool add_attention;

    public:
        MidBlock(int64_t in_channels,
                 int num_layers     = 1,
                 bool add_attention = true)
            : in_channels(in_channels),
              num_layers(num_layers),
              add_attention(add_attention) {
            blocks["block_1"] = std::make_shared<ResnetBlock>(in_channels, in_channels);
            for (int i = 0; i < num_layers; i++) {
                if (add_attention) {
                    blocks["attn_" + std::to_string(i + 1)] = std::make_shared<AttnBlock>(in_channels);
                }
                blocks["block_" + std::to_string(i + 2)] = std::make_shared<ResnetBlock>(in_channels, in_channels);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [B*C, T, H, W]
            // return: [B*C, T, H, W]
            auto block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["block_1"]);

            x = block_1->forward(ctx, x);
            for (int i = 0; i < num_layers; i++) {
                if (add_attention) {
                    auto block = std::dynamic_pointer_cast<AttnBlock>(blocks["attn_" + std::to_string(i + 1)]);
                    x          = block->forward(ctx, x);
                }
                auto block = std::dynamic_pointer_cast<ResnetBlock>(blocks["block_" + std::to_string(i + 2)]);
                x          = block->forward(ctx, x);
            }
            return x;
        }
    };

    class UpBlock : public UnaryBlock {
    protected:
        int num_layers;
        int64_t upsample_out_channels;

    public:
        UpBlock(int64_t in_channels,
                int64_t out_channels,
                int num_layers                = 1,
                int64_t upsample_out_channels = 0,
                bool add_temporal_upsample    = true)
            : num_layers(num_layers),
              upsample_out_channels(upsample_out_channels) {
            for (int i = 0; i < num_layers; i++) {
                int64_t IC                           = i == 0 ? in_channels : out_channels;
                blocks["block." + std::to_string(i)] = std::make_shared<ResnetBlock>(IC, out_channels);
            }
            if (upsample_out_channels > 0) {
                blocks["upsample"] = std::make_shared<Upsample>(out_channels, upsample_out_channels, add_temporal_upsample);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            return forward(ctx, x, nullptr);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             TemporalConvCarry* carry) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T, H, W] or [B*OC, T, H*2, W*2] or [B*OC, T*2, H*2, W*2]
            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<ResnetBlock>(blocks["block." + std::to_string(i)]);
                x          = block->forward(ctx, x, carry);
            }
            if (upsample_out_channels > 0) {
                auto upsample = std::dynamic_pointer_cast<Upsample>(blocks["upsample"]);
                x             = upsample->forward(ctx, x, carry);
            }
            return x;
        }
    };

    class DownBlock : public UnaryBlock {
    protected:
        int num_layers;
        int64_t downsample_out_channels;

    public:
        DownBlock(int64_t in_channels,
                  int64_t out_channels,
                  int num_layers,
                  int64_t downsample_out_channels = 0,
                  bool add_temporal_downsample    = false)
            : num_layers(num_layers),
              downsample_out_channels(downsample_out_channels) {
            for (int i = 0; i < num_layers; i++) {
                int64_t IC                           = i == 0 ? in_channels : out_channels;
                blocks["block." + std::to_string(i)] = std::make_shared<ResnetBlock>(IC, out_channels);
            }
            if (downsample_out_channels > 0) {
                blocks["downsample"] = std::make_shared<Downsample>(out_channels,
                                                                    downsample_out_channels,
                                                                    add_temporal_downsample);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<ResnetBlock>(blocks["block." + std::to_string(i)]);
                x          = block->forward(ctx, x);
            }
            if (downsample_out_channels > 0) {
                auto downsample = std::dynamic_pointer_cast<Downsample>(blocks["downsample"]);
                x               = downsample->forward(ctx, x);
            }
            return x;
        }
    };

    class Encoder : public GGMLBlock {
    protected:
        int64_t z_channels;
        std::vector<int64_t> block_out_channels;

    public:
        Encoder(int64_t in_channels                     = 3,
                int64_t z_channels                      = 32,
                std::vector<int64_t> block_out_channels = {128, 256, 512, 1024, 1024},
                int layers_per_block                    = 2,
                int spatial_compression_ratio           = 16,
                int temporal_compression_ratio          = 4,
                bool downsample_match_channel           = true)
            : z_channels(z_channels),
              block_out_channels(std::move(block_out_channels)) {
            blocks["conv_in"] = std::make_shared<CausalConv3d>(in_channels,
                                                               this->block_out_channels[0],
                                                               std::tuple{3, 3, 3});

            int spatial_depth  = static_cast<int>(std::log2(static_cast<double>(spatial_compression_ratio)));
            int temporal_start = static_cast<int>(std::log2(static_cast<double>(spatial_compression_ratio / temporal_compression_ratio)));
            int64_t channels   = this->block_out_channels[0];
            for (int i = 0; i < static_cast<int>(this->block_out_channels.size()); i++) {
                int64_t out_channels = this->block_out_channels[i];
                if (i < spatial_depth) {
                    int64_t next_channels               = downsample_match_channel ? this->block_out_channels[i + 1] : out_channels;
                    blocks["down." + std::to_string(i)] = std::make_shared<DownBlock>(channels,
                                                                                      out_channels,
                                                                                      layers_per_block,
                                                                                      next_channels,
                                                                                      i >= temporal_start);
                    channels                            = next_channels;
                } else {
                    blocks["down." + std::to_string(i)] = std::make_shared<DownBlock>(channels,
                                                                                      out_channels,
                                                                                      layers_per_block);
                    channels                            = out_channels;
                }
            }

            blocks["mid"]      = std::make_shared<MidBlock>(channels);
            blocks["norm_out"] = std::make_shared<WAN::RMS_norm>(channels);
            blocks["conv_out"] = std::make_shared<CausalConv3d>(channels,
                                                                z_channels * 2,
                                                                std::tuple{3, 3, 3});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv_in  = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto mid      = std::dynamic_pointer_cast<MidBlock>(blocks["mid"]);
            auto norm_out = std::dynamic_pointer_cast<WAN::RMS_norm>(blocks["norm_out"]);
            auto conv_out = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            x = conv_in->forward(ctx, x);
            for (int i = 0; i < static_cast<int>(block_out_channels.size()); i++) {
                auto down = std::dynamic_pointer_cast<DownBlock>(blocks["down." + std::to_string(i)]);
                x         = down->forward(ctx, x);
            }
            x = mid->forward(ctx, x);

            auto shortcut = mean_channel_groups(ctx, x, x->ne[3] / (z_channels * 2));
            x             = norm_out->forward(ctx, x);
            x             = ggml_silu_inplace(ctx->ggml_ctx, x);
            x             = conv_out->forward(ctx, x);
            x             = ggml_add(ctx->ggml_ctx, x, shortcut);
            return ggml_ext_slice(ctx->ggml_ctx, x, 3, 0, z_channels);
        }
    };

    class Decoder : public GGMLBlock {
    protected:
        int64_t repeats;
        std::vector<int64_t> block_out_channels;

    public:
        Decoder(int64_t in_channels                     = 32,
                int64_t out_channels                    = 3,
                std::vector<int64_t> block_out_channels = {1024, 1024, 512, 256, 128},
                int layers_per_block                    = 2,
                int spatial_compression_ratio           = 16,
                int temporal_compression_ratio          = 4,
                bool upsample_match_channel             = true)
            : block_out_channels(std::move(block_out_channels)) {
            repeats           = this->block_out_channels[0] / in_channels;
            blocks["conv_in"] = std::make_shared<CausalConv3d>(in_channels, this->block_out_channels[0], std::tuple{3, 3, 3});
            blocks["mid"]     = std::make_shared<MidBlock>(this->block_out_channels[0]);

            int64_t IC = this->block_out_channels[0];
            for (int i = 0; i < this->block_out_channels.size(); i++) {
                int64_t OC                 = this->block_out_channels[i];
                bool add_spatial_upsample  = i < std::log2(static_cast<double>(spatial_compression_ratio));
                bool add_temporal_upsample = i < std::log2(static_cast<double>(temporal_compression_ratio));

                if (add_spatial_upsample || add_temporal_upsample) {
                    int64_t upsample_out_channels     = upsample_match_channel ? this->block_out_channels[i + 1] : OC;
                    blocks["up." + std::to_string(i)] = std::make_shared<UpBlock>(IC, OC, layers_per_block + 1, upsample_out_channels, add_temporal_upsample);
                    IC                                = upsample_out_channels;
                } else {
                    blocks["up." + std::to_string(i)] = std::make_shared<UpBlock>(IC, OC, layers_per_block + 1, 0, false);
                }
            }

            blocks["norm_out"] = std::make_shared<WAN::RMS_norm>(this->block_out_channels.back());
            blocks["conv_out"] = std::make_shared<CausalConv3d>(this->block_out_channels.back(), out_channels, std::tuple{3, 3, 3});
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* z) {
            auto conv_in   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_in"]);
            auto mid_block = std::dynamic_pointer_cast<MidBlock>(blocks["mid"]);
            auto norm_out  = std::dynamic_pointer_cast<WAN::RMS_norm>(blocks["norm_out"]);
            auto conv_out  = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv_out"]);

            auto h = conv_in->forward(ctx, z);

            auto shortcut = repeat_interleave_channels(ctx, z, repeats, z->ne[0], z->ne[1], z->ne[2]);
            h             = ggml_add(ctx->ggml_ctx, h, shortcut);

            h = mid_block->forward(ctx, h);

            ggml_tensor* output = nullptr;
            std::vector<ggml_tensor*> carry_input;
            const int64_t frames = h->ne[2];
            for (int64_t start = 0; start < frames; start += HUNYUAN_VIDEO_VAE_TEMPORAL_CHUNK_SIZE) {
                const int64_t end = std::min(start + HUNYUAN_VIDEO_VAE_TEMPORAL_CHUNK_SIZE, frames);
                auto chunk        = ggml_ext_slice(ctx->ggml_ctx, h, 2, start, end);

                std::vector<ggml_tensor*> carry_output;
                TemporalConvCarry carry{
                    start == 0 ? nullptr : &carry_input,
                    end == frames ? nullptr : &carry_output,
                };

                for (int i = 0; i < block_out_channels.size(); i++) {
                    auto up_block = std::dynamic_pointer_cast<UpBlock>(blocks["up." + std::to_string(i)]);
                    chunk         = up_block->forward(ctx, chunk, &carry);
                }

                chunk = norm_out->forward(ctx, chunk);
                chunk = ggml_silu_inplace(ctx->ggml_ctx, chunk);  // nonlinearity/swish
                chunk = conv_out->forward(ctx, chunk, &carry);
                carry.finish();

                output      = output == nullptr ? chunk : ggml_concat(ctx->ggml_ctx, output, chunk, 2);
                carry_input = std::move(carry_output);
            }
            return output;
        }
    };

    class HunyuanVideoVAERunner : public VAE {
    protected:
        bool decode_only;
        Encoder encoder;
        Decoder decoder;

    public:
        HunyuanVideoVAERunner(ggml_backend_t backend,
                              const String2TensorStorage& tensor_storage_map,
                              const std::string& prefix,
                              bool decode_only,
                              SDVersion version,
                              std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : VAE(version, backend, prefix, weight_manager),
              decode_only(decode_only ||
                          tensor_storage_map.find(prefix + ".encoder.conv_in.conv.weight") == tensor_storage_map.end()) {
            if (!this->decode_only) {
                encoder.init(params_ctx, tensor_storage_map, prefix + ".encoder");
            }
            decoder.init(params_ctx, tensor_storage_map, prefix + ".decoder");
        }

        std::string get_desc() override {
            return "hunyuan_video_vae";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            if (!decode_only) {
                encoder.get_param_tensors(tensors, weight_prefix + ".encoder");
            }
            decoder.get_param_tensors(tensors, weight_prefix + ".decoder");
        }

        int get_encoder_output_channels(int input_channels) override {
            SD_UNUSED(input_channels);
            return 32;
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output,
                                                std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            return latents / 1.03682f;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            return latents * 1.03682f;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& input_tensor, bool decode_graph) {
            size_t graph_size = HUNYUAN_VIDEO_VAE_GRAPH_SIZE;
            if (decode_graph) {
                graph_size = std::max(graph_size,
                                      HUNYUAN_VIDEO_VAE_GRAPH_SIZE_PER_LATENT_FRAME *
                                          static_cast<size_t>(input_tensor.shape()[2]));
            }
            ggml_cgraph* gf     = new_graph_custom(graph_size);
            ggml_tensor* input  = make_input(input_tensor);
            auto runner_ctx     = get_context();
            ggml_tensor* output = decode_graph ? decoder.forward(&runner_ctx, input)
                                               : encoder.forward(&runner_ctx, input);
            ggml_build_forward_expand(gf, output);
            return gf;
        }

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& input,
                                   bool decode_graph) override {
            if (!decode_graph && decode_only) {
                LOG_ERROR("Hunyuan Video VAE encoder weights are not available");
                return {};
            }

            sd::Tensor<float> expanded;
            if (input.dim() == 4) {
                expanded = input.unsqueeze(2);
            }
            const auto& graph_input = expanded.empty() ? input : expanded;
            auto get_graph          = [&]() -> ggml_cgraph* {
                return build_graph(graph_input, decode_graph);
            };
            auto output = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph,
                                                                                     n_threads,
                                                                                     true,
                                                                                     true,
                                                                                     true),
                                                          graph_input.dim());
            if (!output.empty() && input.dim() == 4) {
                output.squeeze_(2);
            }
            return output;
        }
    };

}  // namespace Hunyuan

#endif  // __SD_MODEL_VAE_HUNYUAN_VAE_HPP__

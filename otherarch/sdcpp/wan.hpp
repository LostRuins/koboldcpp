#ifndef __WAN_HPP__
#define __WAN_HPP__

#include <map>
#include <memory>
#include <utility>

#include "common.hpp"
#include "flux.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include "vae.hpp"

namespace WAN {

    constexpr int CACHE_T        = 2;
    constexpr int WAN_GRAPH_SIZE = 10240;

    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> padding;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["weight"] = ggml_new_tensor_4d(ctx,
                                                  GGML_TYPE_F16,
                                                  std::get<2>(kernel_size),
                                                  std::get<1>(kernel_size),
                                                  std::get<0>(kernel_size),
                                                  in_channels * out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

    public:
        CausalConv3d(int64_t in_channels,
                     int64_t out_channels,
                     std::tuple<int, int, int> kernel_size,
                     std::tuple<int, int, int> stride   = {1, 1, 1},
                     std::tuple<int, int, int> padding  = {0, 0, 0},
                     std::tuple<int, int, int> dilation = {1, 1, 1},
                     bool bias                          = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(std::move(kernel_size)),
              stride(std::move(stride)),
              padding(std::move(padding)),
              dilation(std::move(dilation)),
              bias(bias) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x, struct ggml_tensor* cache_x = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            struct ggml_tensor* w = params["weight"];
            struct ggml_tensor* b = nullptr;
            if (bias) {
                b = params["bias"];
            }

            int lp0 = std::get<2>(padding);
            int rp0 = std::get<2>(padding);
            int lp1 = std::get<1>(padding);
            int rp1 = std::get<1>(padding);
            int lp2 = 2 * std::get<0>(padding);
            int rp2 = 0;

            if (cache_x != nullptr && lp2 > 0) {
                x = ggml_concat(ctx->ggml_ctx, cache_x, x, 2);
                lp2 -= (int)cache_x->ne[2];
            }

            x = ggml_ext_pad_ext(ctx->ggml_ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return ggml_ext_conv_3d(ctx->ggml_ctx, x, w, b, in_channels,
                                    std::get<2>(stride), std::get<1>(stride), std::get<0>(stride),
                                    0, 0, 0,
                                    std::get<2>(dilation), std::get<1>(dilation), std::get<0>(dilation));
        }
    };

    class RMS_norm : public UnaryBlock {
    protected:
        int64_t dim;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            auto iter       = tensor_storage_map.find(prefix + "gamma");
            if (iter != tensor_storage_map.end()) {
                params["gamma"] = ggml_new_tensor(ctx, wtype, iter->second.n_dims, &iter->second.ne[0]);
            } else {
                params["gamma"] = ggml_new_tensor_1d(ctx, wtype, dim);
            }
        }

    public:
        RMS_norm(int64_t dim)
            : dim(dim) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            // x: [N*IC, ID, IH, IW], IC == dim
            // assert N == 1

            struct ggml_tensor* w = params["gamma"];
            w                     = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            auto h                = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));  // [ID, IH, IW, N*IC]
            h                     = ggml_rms_norm(ctx->ggml_ctx, h, 1e-12f);
            h                     = ggml_mul(ctx->ggml_ctx, h, w);
            h                     = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));

            return h;
        }
    };

    class Resample : public GGMLBlock {
    protected:
        int64_t dim;
        std::string mode;

    public:
        Resample(int64_t dim, const std::string& mode, bool wan2_2 = false)
            : dim(dim), mode(mode) {
            if (mode == "upsample2d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
            } else if (mode == "upsample3d") {
                if (wan2_2) {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {1, 1}, {1, 1}));
                } else {
                    blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim / 2, {3, 3}, {1, 1}, {1, 1}));
                }
                blocks["time_conv"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim * 2, {3, 1, 1}, {1, 1, 1}, {1, 0, 0}));
            } else if (mode == "downsample2d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
            } else if (mode == "downsample3d") {
                blocks["resample.1"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {3, 3}, {2, 2}));
                blocks["time_conv"]  = std::shared_ptr<GGMLBlock>(new CausalConv3d(dim, dim, {3, 1, 1}, {2, 1, 1}, {0, 0, 0}));
            } else if (mode == "none") {
                // nn.Identity()
            } else {
                GGML_ASSERT(false && "invalid mode");
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            int64_t c = x->ne[3] / b;
            int64_t t = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            if (mode == "upsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    feat_idx += 1;
                    if (chunk_idx == 0) {
                        // feat_cache[idx] == nullptr, pass
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {  // chunk_idx >= 2
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }
                        if (chunk_idx == 1 && cache_x->ne[2] < 2) {  // Rep
                            cache_x = ggml_pad_ext(ctx->ggml_ctx, cache_x, 0, 0, 0, 0, (int)cache_x->ne[2], 0, 0, 0);
                            // aka cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device),cache_x],dim=2)
                        }
                        if (chunk_idx == 1) {
                            x = time_conv->forward(ctx, x);
                        } else {
                            x = time_conv->forward(ctx, x, feat_cache[idx]);
                        }
                        feat_cache[idx] = cache_x;
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w * h, t, c, 2);                                   // (2, c, t, h*w)
                        x               = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 3, 1, 2));  // (c, t, 2, h*w)
                        x               = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, 2 * t, c);                                   // (c, t*2, h, w)
                    }
                }
            }

            t = x->ne[2];
            if (mode != "none") {
                auto resample_1 = std::dynamic_pointer_cast<Conv2d>(blocks["resample.1"]);

                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)
                if (mode == "upsample2d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "upsample3d") {
                    x = ggml_upscale(ctx->ggml_ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                } else if (mode == "downsample2d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                } else if (mode == "downsample3d") {
                    x = ggml_ext_pad(ctx->ggml_ctx, x, 1, 1, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
                }
                x = resample_1->forward(ctx, x);
                x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)
            }

            if (mode == "downsample3d") {
                if (feat_cache.size() > 0) {
                    int idx = feat_idx;
                    if (feat_cache[idx] == nullptr) {
                        feat_cache[idx] = x;
                        feat_idx += 1;
                    } else {
                        auto time_conv = std::dynamic_pointer_cast<CausalConv3d>(blocks["time_conv"]);

                        auto cache_x    = ggml_ext_slice(ctx->ggml_ctx, x, 2, -1, x->ne[2]);
                        x               = ggml_concat(ctx->ggml_ctx,
                                                      ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                      x,
                                                      2);
                        x               = time_conv->forward(ctx, x);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                }
            }

            return x;
        }
    };

    class AvgDown3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int factor_t;
        int factor_s;
        int factor;
        int64_t group_size;

    public:
        AvgDown3D(int64_t in_channels, int64_t out_channels, int factor_t, int factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(in_channels * factor % out_channels == 0);
            group_size = in_channels * factor / out_channels;
        }
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t B = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            int pad_t = (factor_t - T % factor_t) % factor_t;

            x = ggml_pad_ext(ctx->ggml_ctx, x, 0, 0, 0, 0, pad_t, 0, 0, 0);
            T = x->ne[2];

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W * H, factor_t, T / factor_t, C);                                                  // [C, T/factor_t, factor_t, H*W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C, factor_t, T/factor_t, H*W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, factor_s, (H / factor_s) * (T / factor_t), factor_t * C);                        // [C*factor_t, T/factor_t*H/factor_s, factor_s, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));                                       // [C*factor_t, factor_s, T/factor_t*H/factor_s, W]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s, W / factor_s, (H / factor_s) * (T / factor_t), factor_s * factor_t * C);  // [C*factor_t*factor_s, T/factor_t*H/factor_s, W/factor_s, factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));                                       // [C*factor_t*factor_s, factor_s, T/factor_t*H/factor_s, W/factor_s]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, (W / factor_s) * (H / factor_s) * (T / factor_t), group_size, out_channels);        // [out_channels, group_size, T/factor_t*H/factor_s*W/factor_s]

            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [out_channels, T/factor_t*H/factor_s*W/factor_s, group_size]
            x = ggml_mean(ctx->ggml_ctx, x);                                                     // [out_channels, T/factor_t*H/factor_s*W/factor_s, 1]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, W / factor_s, H / factor_s, T / factor_t, out_channels);
            return x;
        }
    };

    class DupUp3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        int64_t factor_t;
        int64_t factor_s;
        int64_t factor;
        int64_t repeats;

    public:
        DupUp3D(int64_t in_channels, int64_t out_channels, int64_t factor_t, int64_t factor_s = 1)
            : in_channels(in_channels), out_channels(out_channels), factor_t(factor_t), factor_s(factor_s) {
            factor = factor_t * factor_s * factor_s;
            GGML_ASSERT(out_channels * factor % in_channels == 0);
            repeats = out_channels * factor / in_channels;
        }
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    bool first_chunk = false,
                                    int64_t B        = 1) {
            // x: [B*IC, T, H, W]
            // return: [B*OC, T/factor_t, H/factor_s, W/factor_s]
            GGML_ASSERT(B == 1);
            int64_t C = x->ne[3];
            int64_t T = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];

            auto x_ = x;
            for (int64_t i = 1; i < repeats; i++) {
                x = ggml_concat(ctx->ggml_ctx, x, x_, 2);
            }

            C = out_channels;

            x = ggml_reshape_4d(ctx->ggml_ctx, x, W, H * T, factor_s, factor_s * factor_t * C);  // [C*factor_t*factor_s, factor_s, T*H, W]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 2, 0, 1, 3));  // [C*factor_t*factor_s, T*H, W, factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, H * T, factor_s, factor_t * C);  // [C*factor_t, factor_s, T*H, W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C*factor_t, T*H, factor_s, W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W * factor_s * H, T, factor_t, C);  // [C, factor_t, T, H*factor_s*W*factor_s]
            x = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 2, 1, 3));  // [C, T, factor_t, H*factor_s*W*factor_s]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, factor_s * W, factor_s * H, factor_t * T, C);  // [C, T*factor_t, H*factor_s, W*factor_s]

            if (first_chunk) {
                x = ggml_ext_slice(ctx->ggml_ctx, x, 2, factor_t - 1, x->ne[2]);
            }

            return x;
        }
    };

    class ResidualBlock : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t out_dim;

    public:
        ResidualBlock(int64_t in_dim, int64_t out_dim)
            : in_dim(in_dim), out_dim(out_dim) {
            blocks["residual.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(in_dim));
            // residual.1 is nn.SiLU()
            blocks["residual.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            blocks["residual.3"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // residual.4 is nn.SiLU()
            // residual.5 is nn.Dropout()
            blocks["residual.6"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            if (in_dim != out_dim) {
                blocks["shortcut"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(in_dim, out_dim, {1, 1, 1}));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* h = x;
            if (in_dim != out_dim) {
                auto shortcut = std::dynamic_pointer_cast<CausalConv3d>(blocks["shortcut"]);

                h = shortcut->forward(ctx, x);
            }

            for (int i = 0; i < 7; i++) {
                if (i == 0 || i == 3) {  // RMS_norm
                    auto layer = std::dynamic_pointer_cast<RMS_norm>(blocks["residual." + std::to_string(i)]);
                    x          = layer->forward(ctx, x);
                } else if (i == 2 || i == 6) {  // CausalConv3d
                    auto layer = std::dynamic_pointer_cast<CausalConv3d>(blocks["residual." + std::to_string(i)]);

                    if (feat_cache.size() > 0) {
                        int idx      = feat_idx;
                        auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                        if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                            // cache last frame of last two chunk
                            cache_x = ggml_concat(ctx->ggml_ctx,
                                                  ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                                  cache_x,
                                                  2);
                        }

                        x               = layer->forward(ctx, x, feat_cache[idx]);
                        feat_cache[idx] = cache_x;
                        feat_idx += 1;
                    }
                } else if (i == 1 || i == 4) {
                    x = ggml_silu(ctx->ggml_ctx, x);
                } else {  // i == 5
                    // nn.Dropout(), ignore
                }
            }

            x = ggml_add(ctx->ggml_ctx, x, h);
            return x;
        }
    };

    class Down_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool down_flag;

    public:
        Down_ResidualBlock(int64_t in_dim,
                           int64_t out_dim,
                           int mult,
                           bool temperal_downsample = false,
                           bool down_flag           = false)
            : mult(mult), down_flag(down_flag) {
            blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new AvgDown3D(in_dim, out_dim, temperal_downsample ? 2 : 1, down_flag ? 2 : 1));

            int i = 0;
            for (; i < mult; i++) {
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                     = out_dim;
            }
            if (down_flag) {
                std::string mode                           = temperal_downsample ? "downsample3d" : "downsample2d";
                blocks["downsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* x_copy = x;

            auto avg_shortcut = std::dynamic_pointer_cast<AvgDown3D>(blocks["avg_shortcut"]);

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (down_flag) {
                std::string block_name = "downsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
            }

            auto shortcut = avg_shortcut->forward(ctx, x_copy, b);

            x = ggml_add(ctx->ggml_ctx, x, shortcut);

            return x;
        }
    };

    class Up_ResidualBlock : public GGMLBlock {
    protected:
        int mult;
        bool up_flag;

    public:
        Up_ResidualBlock(int64_t in_dim,
                         int64_t out_dim,
                         int mult,
                         bool temperal_upsample = false,
                         bool up_flag           = false)
            : mult(mult), up_flag(up_flag) {
            if (up_flag) {
                blocks["avg_shortcut"] = std::shared_ptr<GGMLBlock>(new DupUp3D(in_dim, out_dim, temperal_upsample ? 2 : 1, up_flag ? 2 : 1));
            }

            int i = 0;
            for (; i < mult; i++) {
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                in_dim                                   = out_dim;
            }
            if (up_flag) {
                std::string mode                         = temperal_upsample ? "upsample3d" : "upsample2d";
                blocks["upsamples." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode, true));
                i++;
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            struct ggml_tensor* x_copy = x;

            int i = 0;
            for (; i < mult; i++) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<ResidualBlock>(blocks[block_name]);

                x = block->forward(ctx, x, b, feat_cache, feat_idx);
            }

            if (up_flag) {
                std::string block_name = "upsamples." + std::to_string(i);
                auto block             = std::dynamic_pointer_cast<Resample>(blocks[block_name]);
                x                      = block->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);

                auto avg_shortcut = std::dynamic_pointer_cast<DupUp3D>(blocks["avg_shortcut"]);
                auto shortcut     = avg_shortcut->forward(ctx, x_copy, chunk_idx == 0, b);

                x = ggml_add(ctx->ggml_ctx, x, shortcut);
            }

            return x;
        }
    };

    class AttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

    public:
        AttentionBlock(int64_t dim)
            : dim(dim) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new RMS_norm(dim));
            blocks["to_qkv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim * 3, {1, 1}));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv2d(dim, dim, {1, 1}));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto norm   = std::dynamic_pointer_cast<RMS_norm>(blocks["norm"]);
            auto to_qkv = std::dynamic_pointer_cast<Conv2d>(blocks["to_qkv"]);
            auto proj   = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

            auto identity = x;

            x = norm->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (t, c, h, w)

            const int64_t n = x->ne[3];
            const int64_t c = x->ne[2];
            const int64_t h = x->ne[1];
            const int64_t w = x->ne[0];

            auto qkv     = to_qkv->forward(ctx, x);
            auto qkv_vec = split_image_qkv(ctx->ggml_ctx, qkv);

            auto q = qkv_vec[0];
            q      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 2, 0, 1, 3));  // [t, h, w, c]
            q      = ggml_reshape_3d(ctx->ggml_ctx, q, c, h * w, n);                                      // [t, h * w, c]

            auto k = qkv_vec[1];
            k      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 2, 0, 1, 3));  // [t, h, w, c]
            k      = ggml_reshape_3d(ctx->ggml_ctx, k, c, h * w, n);                                      // [t, h * w, c]

            auto v = qkv_vec[2];
            v      = ggml_reshape_3d(ctx->ggml_ctx, v, h * w, c, n);  // [t, c, h * w]

            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));                // [t, h * w, c]
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, true, false);  // [t, h * w, c]

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [t, c, h * w]
            x = ggml_reshape_4d(ctx->ggml_ctx, x, w, h, c, n);                             // [t, c, h, w]

            x = proj->forward(ctx, x);

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 0, 1, 3, 2));  // (c, t, h, w)

            x = ggml_add(ctx->ggml_ctx, x, identity);
            return x;
        }
    };

    class Encoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_downsample;

    public:
        Encoder3d(int64_t dim                           = 128,
                  int64_t z_dim                         = 4,
                  std::vector<int> dim_mult             = {1, 2, 4, 4},
                  int num_res_blocks                    = 2,
                  std::vector<bool> temperal_downsample = {false, true, true},
                  bool wan2_2                           = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_downsample(temperal_downsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }

            if (wan2_2) {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(12, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            } else {
                blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(3, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }

            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_down_flag = i < temperal_downsample.size() ? temperal_downsample[i] : false;
                    auto block       = std::shared_ptr<GGMLBlock>(new Down_ResidualBlock(in_dim,
                                                                                         out_dim,
                                                                                         num_res_blocks,
                                                                                         t_down_flag,
                                                                                         i != dim_mult.size() - 1));

                    blocks["downsamples." + std::to_string(index++)] = block;
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto block                                       = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["downsamples." + std::to_string(index++)] = block;
                        in_dim                                           = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                                 = temperal_downsample[i] ? "downsample3d" : "downsample2d";
                        auto block                                       = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["downsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(out_dim));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(out_dim, out_dim));

            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, z_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // downsamples
            std::vector<int64_t> dims = {dim};
            for (int u : dim_mult) {
                dims.push_back(dim * u);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Down_ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["downsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class Decoder3d : public GGMLBlock {
    protected:
        bool wan2_2;
        int64_t dim;
        int64_t z_dim;
        std::vector<int> dim_mult;
        int num_res_blocks;
        std::vector<bool> temperal_upsample;

    public:
        Decoder3d(int64_t dim                         = 128,
                  int64_t z_dim                       = 4,
                  std::vector<int> dim_mult           = {1, 2, 4, 4},
                  int num_res_blocks                  = 2,
                  std::vector<bool> temperal_upsample = {true, true, false},
                  bool wan2_2                         = false)
            : dim(dim),
              z_dim(z_dim),
              dim_mult(dim_mult),
              num_res_blocks(num_res_blocks),
              temperal_upsample(temperal_upsample),
              wan2_2(wan2_2) {
            // attn_scales is always []
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }

            // init block
            blocks["conv1"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, dims[0], {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            // middle blocks
            blocks["middle.0"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));
            blocks["middle.1"] = std::shared_ptr<GGMLBlock>(new AttentionBlock(dims[0]));
            blocks["middle.2"] = std::shared_ptr<GGMLBlock>(new ResidualBlock(dims[0], dims[0]));

            // upsample blocks
            int index = 0;
            int64_t in_dim;
            int64_t out_dim;
            for (int i = 0; i < dims.size() - 1; i++) {
                in_dim  = dims[i];
                out_dim = dims[i + 1];
                if (wan2_2) {
                    bool t_up_flag = i < temperal_upsample.size() ? temperal_upsample[i] : false;
                    auto block     = std::shared_ptr<GGMLBlock>(new Up_ResidualBlock(in_dim,
                                                                                     out_dim,
                                                                                     num_res_blocks + 1,
                                                                                     t_up_flag,
                                                                                     i != dim_mult.size() - 1));

                    blocks["upsamples." + std::to_string(index++)] = block;
                } else {
                    if (i == 1 || i == 2 || i == 3) {
                        in_dim = in_dim / 2;
                    }
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto block                                     = std::shared_ptr<GGMLBlock>(new ResidualBlock(in_dim, out_dim));
                        blocks["upsamples." + std::to_string(index++)] = block;
                        in_dim                                         = out_dim;
                    }

                    if (i != dim_mult.size() - 1) {
                        std::string mode                               = temperal_upsample[i] ? "upsample3d" : "upsample2d";
                        auto block                                     = std::shared_ptr<GGMLBlock>(new Resample(out_dim, mode));
                        blocks["upsamples." + std::to_string(index++)] = block;
                    }
                }
            }

            // output blocks
            blocks["head.0"] = std::shared_ptr<GGMLBlock>(new RMS_norm(out_dim));
            // head.1 is nn.SiLU()
            if (wan2_2) {
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 12, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));

            } else {
                blocks["head.2"] = std::shared_ptr<GGMLBlock>(new CausalConv3d(out_dim, 3, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}));
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    int64_t b,
                                    std::vector<struct ggml_tensor*>& feat_cache,
                                    int& feat_idx,
                                    int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            auto conv1    = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);
            auto middle_0 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.0"]);
            auto middle_1 = std::dynamic_pointer_cast<AttentionBlock>(blocks["middle.1"]);
            auto middle_2 = std::dynamic_pointer_cast<ResidualBlock>(blocks["middle.2"]);
            auto head_0   = std::dynamic_pointer_cast<RMS_norm>(blocks["head.0"]);
            auto head_2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["head.2"]);

            // conv1
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = conv1->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = conv1->forward(ctx, x);
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);

            // upsamples
            std::vector<int64_t> dims = {dim_mult[dim_mult.size() - 1] * dim};
            for (int i = static_cast<int>(dim_mult.size()) - 1; i >= 0; i--) {
                dims.push_back(dim * dim_mult[i]);
            }
            int index = 0;
            for (int i = 0; i < dims.size() - 1; i++) {
                if (wan2_2) {
                    auto layer = std::dynamic_pointer_cast<Up_ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                    x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                } else {
                    for (int j = 0; j < num_res_blocks + 1; j++) {
                        auto layer = std::dynamic_pointer_cast<ResidualBlock>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx);
                    }

                    if (i != dim_mult.size() - 1) {
                        auto layer = std::dynamic_pointer_cast<Resample>(blocks["upsamples." + std::to_string(index++)]);

                        x = layer->forward(ctx, x, b, feat_cache, feat_idx, chunk_idx);
                    }
                }
            }

            // head
            x = head_0->forward(ctx, x);
            x = ggml_silu(ctx->ggml_ctx, x);
            if (feat_cache.size() > 0) {
                int idx      = feat_idx;
                auto cache_x = ggml_ext_slice(ctx->ggml_ctx, x, 2, -CACHE_T, x->ne[2]);
                if (cache_x->ne[2] < 2 && feat_cache[idx] != nullptr) {
                    // cache last frame of last two chunk
                    cache_x = ggml_concat(ctx->ggml_ctx,
                                          ggml_ext_slice(ctx->ggml_ctx, feat_cache[idx], 2, -1, feat_cache[idx]->ne[2]),
                                          cache_x,
                                          2);
                }

                x               = head_2->forward(ctx, x, feat_cache[idx]);
                feat_cache[idx] = cache_x;
                feat_idx += 1;
            } else {
                x = head_2->forward(ctx, x);
            }

            return x;
        }
    };

    class WanVAE : public GGMLBlock {
    public:
        bool wan2_2                           = false;
        bool decode_only                      = true;
        int64_t dim                           = 96;
        int64_t dec_dim                       = 96;
        int64_t z_dim                         = 16;
        std::vector<int> dim_mult             = {1, 2, 4, 4};
        int num_res_blocks                    = 2;
        std::vector<bool> temperal_upsample   = {true, true, false};
        std::vector<bool> temperal_downsample = {false, true, true};

        int _conv_num = 33;
        int _conv_idx = 0;
        std::vector<struct ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<struct ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<struct ggml_tensor*>(_conv_num, nullptr);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<struct ggml_tensor*>(_enc_conv_num, nullptr);
        }

    public:
        WanVAE(bool decode_only = true, bool wan2_2 = false)
            : decode_only(decode_only), wan2_2(wan2_2) {
            // attn_scales is always []
            if (wan2_2) {
                dim     = 160;
                dec_dim = 256;
                z_dim   = 48;

                _conv_num     = 34;
                _enc_conv_num = 26;
            }
            if (!decode_only) {
                blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, temperal_downsample, wan2_2));
                blocks["conv1"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim * 2, z_dim * 2, {1, 1, 1}));
            }
            blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, temperal_upsample, wan2_2));
            blocks["conv2"]   = std::shared_ptr<GGMLBlock>(new CausalConv3d(z_dim, z_dim, {1, 1, 1}));
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x,
                                     int64_t patch_size,
                                     int64_t b = 1) {
            // x: [b*c, f, h*q, w*r]
            // return: [b*c*r*q, f, h, w]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1] / q;
            int64_t w = x->ne[0] / r;

            x = ggml_reshape_4d(ctx, x, r * w, q, h, f * c * b);                 // [b*c*f, h, q, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, q, h, w*r]
            x = ggml_reshape_4d(ctx, x, r, w, h * q, f * c * b);                 // [b*c*f, q*h, w, r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [b*c*f, r, q*h, w]
            x = ggml_reshape_4d(ctx, x, w * h, q * r, f, c * b);                 // [b*c, f, r*q, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, r*q, f, h*w]
            x = ggml_reshape_4d(ctx, x, w, h, f, q * r * c * b);                 // [b*c*r*q, f, h, w]

            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t patch_size,
                                       int64_t b = 1) {
            // x: [b*c*r*q, f, h, w]
            // return: [b*c, f, h*q, w*r]
            if (patch_size == 1) {
                return x;
            }
            int64_t r = patch_size;
            int64_t q = patch_size;
            int64_t c = x->ne[3] / b / q / r;
            int64_t f = x->ne[2];
            int64_t h = x->ne[1];
            int64_t w = x->ne[0];

            x = ggml_reshape_4d(ctx, x, w * h, f, q * r, c * b);                 // [b*c, r*q, f, h*w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c, f, r*q, h*w]
            x = ggml_reshape_4d(ctx, x, w, h * q, r, f * c * b);                 // [b*c*f, r, q*h, w]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [b*c*f, q*h, w, r]
            x = ggml_reshape_4d(ctx, x, r * w, h, q, f * c * b);                 // [b*c*f, q, h, w*r]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));  // [b*c*f, h, q, w*r]
            x = ggml_reshape_4d(ctx, x, r * w, q * h, f, c * b);                 // [b*c, f, h*q, w*r]
            return x;
        }

        struct ggml_tensor* encode(GGMLRunnerContext* ctx,
                                   struct ggml_tensor* x,
                                   int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            struct ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _enc_conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, 0, 1);  // [b*c, 1, h, w]
                    out     = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, 1 + 4 * (i - 1), 1 + 4 * i);  // [b*c, 4, h, w]
                    auto out_ = encoder->forward(ctx, in, b, _enc_feat_map, _enc_conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            out     = conv1->forward(ctx, out);
            auto mu = ggml_ext_chunk(ctx->ggml_ctx, out, 2, 3)[0];
            clear_cache();
            return mu;
        }

        struct ggml_tensor* decode(GGMLRunnerContext* ctx,
                                   struct ggml_tensor* z,
                                   int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            struct ggml_tensor* out;
            for (int i = 0; i < iter_; i++) {
                _conv_idx = 0;
                if (i == 0) {
                    auto in = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    out     = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                } else {
                    auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
                    auto out_ = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
                    out       = ggml_concat(ctx->ggml_ctx, out, out_, 2);
                }
            }
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            clear_cache();
            return out;
        }

        struct ggml_tensor* decode_partial(GGMLRunnerContext* ctx,
                                           struct ggml_tensor* z,
                                           int i,
                                           int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto x    = conv2->forward(ctx, z);
            auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
            _conv_idx = 0;
            auto out  = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            return out;
        }
    };

    struct WanVAERunner : public VAE {
        bool decode_only = true;
        WanVAE ae;

        WanVAERunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only), ae(decode_only, version == VERSION_WAN2_2_TI2V), VAE(backend, offload_params_to_cpu) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "wan_vae";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
            struct ggml_cgraph* gf = new_graph_custom(10240 * z->ne[2]);

            z = to_backend(z);

            auto runner_ctx = get_context();

            struct ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        struct ggml_cgraph* build_graph_partial(struct ggml_tensor* z, bool decode_graph, int i) {
            struct ggml_cgraph* gf = new_graph_custom(20480);

            ae.clear_cache();

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                auto feat_cache        = get_cache_tensor_by_name("feat_idx:" + std::to_string(feat_idx));
                ae._feat_map[feat_idx] = feat_cache;
            }

            z = to_backend(z);

            auto runner_ctx = get_context();

            struct ggml_tensor* out = decode_graph ? ae.decode_partial(&runner_ctx, z, i) : ae.encode(&runner_ctx, z);

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                ggml_tensor* feat_cache = ae._feat_map[feat_idx];
                if (feat_cache != nullptr) {
                    cache("feat_idx:" + std::to_string(feat_idx), feat_cache);
                    ggml_build_forward_expand(gf, feat_cache);
                }
            }

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(const int n_threads,
                     struct ggml_tensor* z,
                     bool decode_graph,
                     struct ggml_tensor** output,
                     struct ggml_context* output_ctx = nullptr) override {
            if (true) {
                auto get_graph = [&]() -> struct ggml_cgraph* {
                    return build_graph(z, decode_graph);
                };
                return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
            } else {  // chunk 1 result is weird
                ae.clear_cache();
                int64_t t      = z->ne[2];
                int i          = 0;
                auto get_graph = [&]() -> struct ggml_cgraph* {
                    return build_graph_partial(z, decode_graph, i);
                };
                struct ggml_tensor* out = nullptr;
                bool res                = GGMLRunner::compute(get_graph, n_threads, true, &out, output_ctx);
                ae.clear_cache();
                if (t == 1) {
                    *output = out;
                    return res;
                }

                *output = ggml_new_tensor_4d(output_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], (t - 1) * 4 + 1, out->ne[3]);

                auto copy_to_output = [&]() {
                    for (int64_t i3 = 0; i3 < out->ne[3]; i3++) {
                        for (int64_t i2 = 0; i2 < out->ne[2]; i2++) {
                            for (int64_t i1 = 0; i1 < out->ne[1]; i1++) {
                                for (int64_t i0 = 0; i0 < out->ne[0]; i0++) {
                                    float value    = ggml_ext_tensor_get_f32(out, i0, i1, i2, i3);
                                    int64_t offset = (i == 0) ? 0 : (1 + (i - 1) * 4);
                                    ggml_ext_tensor_set_f32(*output, value, i0, i1, offset + i2, i3);
                                }
                            }
                        }
                    }
                };

                copy_to_output();

                out = ggml_new_tensor_4d(output_ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], 4, out->ne[3]);

                for (i = 1; i < t; i++) {
                    res = res || GGMLRunner::compute(get_graph, n_threads, true, &out);
                    ae.clear_cache();
                    copy_to_output();
                }
                free_cache_ctx_and_buffer();
                return res;
            }
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            if (true) {
                // cpu f32, pass
                // cpu f16, pass
                // cuda f16, pass
                // cuda f32, pass
                auto z = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 104, 60, 2, 16);
                ggml_set_f32(z, 0.5f);
                z = load_tensor_from_file(work_ctx, "wan_vae_z.bin");
                print_ggml_tensor(z);
                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, z, true, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("decode test done in %ldms", t1 - t0);
            }
        };

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend            = ggml_backend_cpu_init();
            ggml_type model_data_type         = GGML_TYPE_F16;
            std::shared_ptr<WanVAERunner> vae = std::make_shared<WanVAERunner>(backend, false, String2TensorStorage{}, "", false, VERSION_WAN2_2_TI2V);
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                vae->alloc_params_buffer();
                std::map<std::string, ggml_tensor*> tensors;
                vae->get_param_tensors(tensors, "first_stage_model");

                ModelLoader model_loader;
                if (!model_loader.init_from_file_and_convert_name(file_path, "vae.")) {
                    LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                    return;
                }

                bool success = model_loader.load_tensors(tensors);

                if (!success) {
                    LOG_ERROR("load tensors from model loader failed");
                    return;
                }

                LOG_INFO("vae model loaded");
            }
            vae->test();
        }
    };

    class WanSelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t head_dim;

    public:
        WanSelfAttention(int64_t dim,
                         int64_t num_heads,
                         bool qk_norm = true,
                         float eps    = 1e-6)
            : num_heads(num_heads) {
            head_dim    = dim / num_heads;
            blocks["q"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["k"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["o"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new Identity());
                blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                            struct ggml_tensor* x,
                                            struct ggml_tensor* pe,
                                            struct ggml_tensor* mask = nullptr) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, x);
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, x);  // [N, n_token, n_head*d_head]

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, n_token, N);  // [N, n_token, n_head, d_head]

            x = Rope::attention(ctx, q, k, v, pe, mask);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    class WanCrossAttention : public WanSelfAttention {
    public:
        WanCrossAttention(int64_t dim,
                          int64_t num_heads,
                          bool qk_norm = true,
                          float eps    = 1e-6)
            : WanSelfAttention(dim, num_heads, qk_norm, eps) {}
        virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                            struct ggml_tensor* x,
                                            struct ggml_tensor* context,
                                            int64_t context_img_len) = 0;
    };

    class WanT2VCrossAttention : public WanCrossAttention {
    public:
        WanT2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {}
        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* context,
                                    int64_t context_img_len) override {
            // x: [N, n_token, dim]
            // context: [N, n_context, dim]
            // context_img_len: unused
            // return [N, n_token, dim]
            int64_t N       = x->ne[2];
            int64_t n_token = x->ne[1];

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context);  // [N, n_context, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context);  // [N, n_context, dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, false, ctx->flash_attn_enabled);  // [N, n_token, dim]

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    class WanI2VCrossAttention : public WanCrossAttention {
    public:
        WanI2VCrossAttention(int64_t dim,
                             int64_t num_heads,
                             bool qk_norm = true,
                             float eps    = 1e-6)
            : WanCrossAttention(dim, num_heads, qk_norm, eps) {
            blocks["k_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v_img"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));

            if (qk_norm) {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            } else {
                blocks["norm_k_img"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* context,
                                    int64_t context_img_len) override {
            // x: [N, n_token, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

            auto k_img_proj = std::dynamic_pointer_cast<Linear>(blocks["k_img"]);
            auto v_img_proj = std::dynamic_pointer_cast<Linear>(blocks["v_img"]);

            auto norm_q     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k     = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);
            auto norm_k_img = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k_img"]);

            int64_t N               = x->ne[2];
            int64_t n_token         = x->ne[1];
            int64_t dim             = x->ne[0];
            int64_t context_txt_len = context->ne[1] - context_img_len;

            context          = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, context, 0, 2, 1, 3));  // [context_img_len + context_txt_len, N, dim]
            auto context_img = ggml_view_3d(ctx->ggml_ctx, context, dim, N, context_img_len, context->nb[1], context->nb[2], 0);
            auto context_txt = ggml_view_3d(ctx->ggml_ctx, context, dim, N, context_txt_len, context->nb[1], context->nb[2], context_img_len * context->nb[2]);
            context_img      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, context_img, 0, 2, 1, 3));  // [N, context_img_len, dim]
            context_txt      = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, context_txt, 0, 2, 1, 3));  // [N, context_txt_len, dim]

            auto q = q_proj->forward(ctx, x);
            q      = norm_q->forward(ctx, q);
            auto k = k_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]
            k      = norm_k->forward(ctx, k);
            auto v = v_proj->forward(ctx, context_txt);  // [N, context_txt_len, dim]

            auto k_img = k_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]
            k_img      = norm_k_img->forward(ctx, k_img);
            auto v_img = v_img_proj->forward(ctx, context_img);  // [N, context_img_len, dim]

            auto img_x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k_img, v_img, num_heads, nullptr, false, false, ctx->flash_attn_enabled);  // [N, n_token, dim]
            x          = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, nullptr, false, false, ctx->flash_attn_enabled);          // [N, n_token, dim]

            x = ggml_add(ctx->ggml_ctx, x, img_x);

            x = o_proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }
    };

    static struct ggml_tensor* modulate_add(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_add(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_add(ctx, x, e);
        }
        return x;
    }

    static struct ggml_tensor* modulate_mul(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* e) {
        // x: [N, n_token, dim]
        // e: [N, 1, dim] or [N, T, 1, dim]
        if (ggml_n_dims(e) == 3) {
            int64_t T = e->ne[2];
            x         = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / T, T, x->ne[2]);  // [N, T, n_token/T, dim]
            x         = ggml_mul(ctx, x, e);
            x         = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);  // [N, n_token, dim]
        } else {
            x = ggml_mul(ctx, x, e);
        }
        return x;
    }

    class WanAttentionBlock : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        WanAttentionBlock(bool t2v_cross_attn,
                          int64_t dim,
                          int64_t ffn_dim,
                          int64_t num_heads,
                          bool qk_norm         = true,
                          bool cross_attn_norm = false,
                          float eps            = 1e-6)
            : dim(dim) {
            blocks["norm1"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new WanSelfAttention(dim, num_heads, qk_norm, eps));
            if (cross_attn_norm) {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, true));
            } else {
                blocks["norm3"] = std::shared_ptr<GGMLBlock>(new Identity());
            }
            if (t2v_cross_attn) {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanT2VCrossAttention(dim, num_heads, qk_norm, eps));
            } else {
                blocks["cross_attn"] = std::shared_ptr<GGMLBlock>(new WanI2VCrossAttention(dim, num_heads, qk_norm, eps));
            }

            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));

            blocks["ffn.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            // ffn.1 is nn.GELU(approximate='tanh')
            blocks["ffn.2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
        }

        virtual struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                            struct ggml_tensor* x,
                                            struct ggml_tensor* e,
                                            struct ggml_tensor* pe,
                                            struct ggml_tensor* context,
                                            int64_t context_img_len = 257) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]

            auto modulation = params["modulation"];
            e               = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 6, dim] or [N, T, 6, dim]
            auto es         = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);  // ([N, 1, dim], ...) or [N, T, 1, dim]

            auto norm1      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto self_attn  = std::dynamic_pointer_cast<WanSelfAttention>(blocks["self_attn"]);
            auto norm3      = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm3"]);
            auto cross_attn = std::dynamic_pointer_cast<WanCrossAttention>(blocks["cross_attn"]);
            auto norm2      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);

            // self-attention
            auto y = norm1->forward(ctx, x);
            y      = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[1]));
            y      = modulate_add(ctx->ggml_ctx, y, es[0]);
            y      = self_attn->forward(ctx, y, pe);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[2]));

            // cross-attention
            x = ggml_add(ctx->ggml_ctx,
                         x,
                         cross_attn->forward(ctx, norm3->forward(ctx, x), context, context_img_len));

            // ffn
            y = norm2->forward(ctx, x);
            y = ggml_add(ctx->ggml_ctx, y, modulate_mul(ctx->ggml_ctx, y, es[4]));
            y = modulate_add(ctx->ggml_ctx, y, es[3]);

            y = ffn_0->forward(ctx, y);
            y = ggml_gelu_inplace(ctx->ggml_ctx, y);
            y = ffn_2->forward(ctx, y);

            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, y, es[5]));

            return x;
        }
    };

    class VaceWanAttentionBlock : public WanAttentionBlock {
    protected:
        int block_id;
        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 6, 1);
        }

    public:
        VaceWanAttentionBlock(bool t2v_cross_attn,
                              int64_t dim,
                              int64_t ffn_dim,
                              int64_t num_heads,
                              bool qk_norm         = true,
                              bool cross_attn_norm = false,
                              float eps            = 1e-6,
                              int block_id         = 0)
            : WanAttentionBlock(t2v_cross_attn, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps), block_id(block_id) {
            if (block_id == 0) {
                blocks["before_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            }
            blocks["after_proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      struct ggml_tensor* c,
                                                      struct ggml_tensor* x,
                                                      struct ggml_tensor* e,
                                                      struct ggml_tensor* pe,
                                                      struct ggml_tensor* context,
                                                      int64_t context_img_len = 257) {
            // x: [N, n_token, dim]
            // e: [N, 6, dim] or [N, T, 6, dim]
            // context: [N, context_img_len + context_txt_len, dim]
            // return [N, n_token, dim]
            if (block_id == 0) {
                auto before_proj = std::dynamic_pointer_cast<Linear>(blocks["before_proj"]);

                c = before_proj->forward(ctx, c);
                c = ggml_add(ctx->ggml_ctx, c, x);
            }

            auto after_proj = std::dynamic_pointer_cast<Linear>(blocks["after_proj"]);

            c           = WanAttentionBlock::forward(ctx, c, e, pe, context, context_img_len);
            auto c_skip = after_proj->forward(ctx, c);

            return {c_skip, c};
        }
    };

    class Head : public GGMLBlock {
    protected:
        int64_t dim;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            enum ggml_type wtype = get_type(prefix + "weight", tensor_storage_map, GGML_TYPE_F32);
            params["modulation"] = ggml_new_tensor_3d(ctx, wtype, dim, 2, 1);
        }

    public:
        Head(int64_t dim,
             int64_t out_dim,
             std::tuple<int, int, int> patch_size,
             float eps = 1e-6)
            : dim(dim) {
            out_dim = out_dim * std::get<0>(patch_size) * std::get<1>(patch_size) * std::get<2>(patch_size);

            blocks["norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* e) {
            // x: [N, n_token, dim]
            // e: [N, dim] or [N, T, dim]
            // return [N, n_token, out_dim]

            auto modulation = params["modulation"];
            e               = ggml_reshape_4d(ctx->ggml_ctx, e, e->ne[0], 1, e->ne[1], e->ne[2]);  // [N, 1, dim] or [N, T, 1, dim]
            e               = ggml_repeat_4d(ctx->ggml_ctx, e, e->ne[0], 2, e->ne[2], e->ne[3]);   // [N, 2, dim] or [N, T, 2, dim]

            e       = ggml_add(ctx->ggml_ctx, e, modulation);  // [N, 2, dim] or [N, T, 2, dim]
            auto es = ggml_ext_chunk(ctx->ggml_ctx, e, 2, 1);  // ([N, 1, dim], ...) or ([N, T, 1, dim], ...)

            auto norm = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto head = std::dynamic_pointer_cast<Linear>(blocks["head"]);

            x = norm->forward(ctx, x);
            x = ggml_add(ctx->ggml_ctx, x, modulate_mul(ctx->ggml_ctx, x, es[1]));
            x = modulate_add(ctx->ggml_ctx, x, es[0]);
            x = head->forward(ctx, x);
            return x;
        }
    };

    class MLPProj : public GGMLBlock {
    protected:
        int64_t in_dim;
        int64_t flf_pos_embed_token_number;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            if (flf_pos_embed_token_number > 0) {
                params["emb_pos"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, flf_pos_embed_token_number, 1);
            }
        }

    public:
        MLPProj(int64_t in_dim,
                int64_t out_dim,
                int64_t flf_pos_embed_token_number = 0)
            : in_dim(in_dim), flf_pos_embed_token_number(flf_pos_embed_token_number) {
            blocks["proj.0"] = std::shared_ptr<GGMLBlock>(new LayerNorm(in_dim));
            blocks["proj.1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, in_dim));
            // proj.2 is nn.GELU()
            blocks["proj.3"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, out_dim));
            blocks["proj.4"] = std::shared_ptr<GGMLBlock>(new LayerNorm(out_dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* image_embeds) {
            if (flf_pos_embed_token_number > 0) {
                auto emb_pos = params["emb_pos"];

                auto a = ggml_ext_slice(ctx->ggml_ctx, image_embeds, 1, 0, emb_pos->ne[1]);
                auto b = ggml_ext_slice(ctx->ggml_ctx, emb_pos, 1, 0, image_embeds->ne[1]);

                image_embeds = ggml_add(ctx->ggml_ctx, a, b);
            }

            auto proj_0 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.0"]);
            auto proj_1 = std::dynamic_pointer_cast<Linear>(blocks["proj.1"]);
            auto proj_3 = std::dynamic_pointer_cast<Linear>(blocks["proj.3"]);
            auto proj_4 = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.4"]);

            auto x = proj_0->forward(ctx, image_embeds);
            x      = proj_1->forward(ctx, x);
            x      = ggml_gelu_inplace(ctx->ggml_ctx, x);
            x      = proj_3->forward(ctx, x);
            x      = proj_4->forward(ctx, x);

            return x;  // clip_extra_context_tokens
        }
    };

    struct WanParams {
        std::string model_type                 = "t2v";
        std::tuple<int, int, int> patch_size   = {1, 2, 2};
        int64_t text_len                       = 512;
        int64_t in_dim                         = 16;
        int64_t dim                            = 2048;
        int64_t ffn_dim                        = 8192;
        int freq_dim                           = 256;
        int64_t text_dim                       = 4096;
        int64_t out_dim                        = 16;
        int64_t num_heads                      = 16;
        int num_layers                         = 32;
        int vace_layers                        = 0;
        int64_t vace_in_dim                    = 96;
        std::map<int, int> vace_layers_mapping = {};
        bool qk_norm                           = true;
        bool cross_attn_norm                   = true;
        float eps                              = 1e-6f;
        int64_t flf_pos_embed_token_number     = 0;
        int theta                              = 10000;
        // wan2.1 1.3B: 1536/12, wan2.1/2.2 14B: 5120/40, wan2.2 5B: 3074/24
        std::vector<int> axes_dim = {44, 42, 42};
        int64_t axes_dim_sum      = 128;
    };

    class Wan : public GGMLBlock {
    protected:
        WanParams params;

    public:
        Wan() {}
        Wan(WanParams params)
            : params(params) {
            // patch_embedding
            blocks["patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(params.in_dim, params.dim, params.patch_size, params.patch_size));

            // text_embedding
            blocks["text_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.text_dim, params.dim));
            // text_embedding.1 is nn.GELU()
            blocks["text_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));

            // time_embedding
            blocks["time_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(params.freq_dim, params.dim));
            // time_embedding.1 is nn.SiLU()
            blocks["time_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim));

            // time_projection.0 is nn.SiLU()
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(params.dim, params.dim * 6));

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block                            = std::shared_ptr<GGMLBlock>(new WanAttentionBlock(params.model_type == "t2v",
                                                                                                         params.dim,
                                                                                                         params.ffn_dim,
                                                                                                         params.num_heads,
                                                                                                         params.qk_norm,
                                                                                                         params.cross_attn_norm,
                                                                                                         params.eps));
                blocks["blocks." + std::to_string(i)] = block;
            }

            // head
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Head(params.dim, params.out_dim, params.patch_size, params.eps));

            // img_emb
            if (params.model_type == "i2v") {
                blocks["img_emb"] = std::shared_ptr<GGMLBlock>(new MLPProj(1280, params.dim, params.flf_pos_embed_token_number));
            }

            // vace
            if (params.vace_layers > 0) {
                for (int i = 0; i < params.vace_layers; i++) {
                    auto block                                 = std::shared_ptr<GGMLBlock>(new VaceWanAttentionBlock(params.model_type == "t2v",
                                                                                                                      params.dim,
                                                                                                                      params.ffn_dim,
                                                                                                                      params.num_heads,
                                                                                                                      params.qk_norm,
                                                                                                                      params.cross_attn_norm,
                                                                                                                      params.eps,
                                                                                                                      i));
                    blocks["vace_blocks." + std::to_string(i)] = block;
                }

                int step = params.num_layers / params.vace_layers;
                int n    = 0;
                for (int i = 0; i < params.num_layers; i += step) {
                    this->params.vace_layers_mapping[i] = n;
                    n++;
                }

                blocks["vace_patch_embedding"] = std::shared_ptr<GGMLBlock>(new Conv3d(params.vace_in_dim, params.dim, params.patch_size, params.patch_size));
            }
        }

        struct ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                              struct ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];

            int pad_t = (std::get<0>(params.patch_size) - T % std::get<0>(params.patch_size)) % std::get<0>(params.patch_size);
            int pad_h = (std::get<1>(params.patch_size) - H % std::get<1>(params.patch_size)) % std::get<1>(params.patch_size);
            int pad_w = (std::get<2>(params.patch_size) - W % std::get<2>(params.patch_size)) % std::get<2>(params.patch_size);
            ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, pad_t, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t t_len,
                                       int64_t h_len,
                                       int64_t w_len) {
            // x: [N, t_len*h_len*w_len, pt*ph*pw*C]
            // return: [N*C, t_len*pt, h_len*ph, w_len*pw]
            int64_t N  = x->ne[3];
            int64_t pt = std::get<0>(params.patch_size);
            int64_t ph = std::get<1>(params.patch_size);
            int64_t pw = std::get<2>(params.patch_size);
            int64_t C  = x->ne[0] / pt / ph / pw;

            GGML_ASSERT(C * pt * ph * pw == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, pw * ph * pt, w_len * h_len * t_len, N);  // [N, t_len*h_len*w_len, pt*ph*pw, C]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));      // [N, C, t_len*h_len*w_len, pt*ph*pw]
            x = ggml_reshape_4d(ctx, x, pw, ph * pt, w_len, h_len * t_len * C * N);  // [N*C*t_len*h_len, w_len, pt*ph, pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, pt*ph, w_len, pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph, pt, h_len * t_len * C * N);  // [N*C*t_len*h_len, pt, ph, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len*h_len, ph, pt, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, pt, ph * h_len, t_len * C * N);  // [N*C*t_len, h_len*ph, pt, w_len*pw]
            x = ggml_ext_cont(ctx, ggml_ext_torch_permute(ctx, x, 0, 2, 1, 3));      // [N*C*t_len, pt, h_len*ph, w_len*pw]
            x = ggml_reshape_4d(ctx, x, pw * w_len, ph * h_len, pt * t_len, C * N);  // [N*C, t_len*pt, h_len*ph, w_len*pw]
            return x;
        }

        struct ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* timestep,
                                         struct ggml_tensor* context,
                                         struct ggml_tensor* pe,
                                         struct ggml_tensor* clip_fea     = nullptr,
                                         struct ggml_tensor* vace_context = nullptr,
                                         float vace_strength              = 1.f,
                                         int64_t N                        = 1) {
            // x: [N*C, T, H, W], C => in_dim
            // vace_context: [N*vace_in_dim, T, H, W]
            // timestep: [N,] or [T]
            // context: [N, L, text_dim]
            // return: [N, t_len*h_len*w_len, out_dim*pt*ph*pw]

            GGML_ASSERT(N == 1);

            auto patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["patch_embedding"]);

            auto text_embedding_0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto text_embedding_2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);

            auto time_embedding_0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto time_embedding_2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto time_projection_1 = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);

            auto head = std::dynamic_pointer_cast<Head>(blocks["head"]);

            // patch_embedding
            x = patch_embedding->forward(ctx, x);                                                    // [N*dim, t_len, h_len, w_len]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0] * x->ne[1] * x->ne[2], x->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]

            // time_embedding
            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, params.freq_dim);
            e      = time_embedding_0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = time_embedding_2->forward(ctx, e);  // [N, dim] or [N, T, dim]

            // time_projection
            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = time_projection_1->forward(ctx, e0);
            e0      = ggml_reshape_4d(ctx->ggml_ctx, e0, e0->ne[0] / 6, 6, e0->ne[1], e0->ne[2]);  //  [N, 6, dim] or [N, T, 6, dim]

            context = text_embedding_0->forward(ctx, context);
            context = ggml_gelu(ctx->ggml_ctx, context);
            context = text_embedding_2->forward(ctx, context);  // [N, context_txt_len, dim]

            int64_t context_img_len = 0;
            if (clip_fea != nullptr) {
                if (params.model_type == "i2v") {
                    auto img_emb     = std::dynamic_pointer_cast<MLPProj>(blocks["img_emb"]);
                    auto context_img = img_emb->forward(ctx, clip_fea);                      // [N, context_img_len, dim]
                    context          = ggml_concat(ctx->ggml_ctx, context_img, context, 1);  // [N, context_img_len + context_txt_len, dim]
                }
                context_img_len = clip_fea->ne[1];  // 257
            }

            // vace_patch_embedding
            ggml_tensor* c = nullptr;
            if (params.vace_layers > 0) {
                auto vace_patch_embedding = std::dynamic_pointer_cast<Conv3d>(blocks["vace_patch_embedding"]);

                c = vace_patch_embedding->forward(ctx, vace_context);                                    // [N*dim, t_len, h_len, w_len]
                c = ggml_reshape_3d(ctx->ggml_ctx, c, c->ne[0] * c->ne[1] * c->ne[2], c->ne[3] / N, N);  // [N, dim, t_len*h_len*w_len]
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 1, 0, 2, 3));  // [N, t_len*h_len*w_len, dim]
            }

            auto x_orig = x;

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<WanAttentionBlock>(blocks["blocks." + std::to_string(i)]);

                x = block->forward(ctx, x, e0, pe, context, context_img_len);

                auto iter = params.vace_layers_mapping.find(i);
                if (iter != params.vace_layers_mapping.end()) {
                    int n = iter->second;

                    auto vace_block = std::dynamic_pointer_cast<VaceWanAttentionBlock>(blocks["vace_blocks." + std::to_string(n)]);

                    auto result = vace_block->forward(ctx, c, x_orig, e0, pe, context, context_img_len);
                    auto c_skip = result.first;
                    c           = result.second;
                    c_skip      = ggml_scale(ctx->ggml_ctx, c_skip, vace_strength);
                    x           = ggml_add(ctx->ggml_ctx, x, c_skip);
                }
            }

            x = head->forward(ctx, x, e);  // [N, t_len*h_len*w_len, pt*ph*pw*out_dim]

            return x;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* clip_fea        = nullptr,
                                    struct ggml_tensor* time_dim_concat = nullptr,
                                    struct ggml_tensor* vace_context    = nullptr,
                                    float vace_strength                 = 1.f,
                                    int64_t N                           = 1) {
            // Forward pass of DiT.
            // x: [N*C, T, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // time_dim_concat: [N*C, T2, H, W]
            // return: [N*C, T, H, W]

            GGML_ASSERT(N == 1);

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t T = x->ne[2];
            int64_t C = x->ne[3];

            x = pad_to_patch_size(ctx, x);

            int64_t t_len = ((T + (std::get<0>(params.patch_size) / 2)) / std::get<0>(params.patch_size));
            int64_t h_len = ((H + (std::get<1>(params.patch_size) / 2)) / std::get<1>(params.patch_size));
            int64_t w_len = ((W + (std::get<2>(params.patch_size) / 2)) / std::get<2>(params.patch_size));

            if (time_dim_concat != nullptr) {
                time_dim_concat = pad_to_patch_size(ctx, time_dim_concat);
                x               = ggml_concat(ctx->ggml_ctx, x, time_dim_concat, 2);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]
                t_len           = ((x->ne[2] + (std::get<0>(params.patch_size) / 2)) / std::get<0>(params.patch_size));
            }

            auto out = forward_orig(ctx, x, timestep, context, pe, clip_fea, vace_context, vace_strength, N);  // [N, t_len*h_len*w_len, pt*ph*pw*C]

            out = unpatchify(ctx->ggml_ctx, out, t_len, h_len, w_len);  // [N*C, (T+pad_t) + (T2+pad_t2), H + pad_h, W + pad_w]

            // slice

            out = ggml_ext_slice(ctx->ggml_ctx, out, 2, 0, T);  // [N*C, T, H + pad_h, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N*C, T, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N*C, T, H, W]

            return out;
        }
    };

    struct WanRunner : public GGMLRunner {
    public:
        std::string desc = "wan";
        WanParams wan_params;
        Wan wan;
        std::vector<float> pe_vec;
        SDVersion version;

        WanRunner(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map = {},
                  const std::string prefix                       = "",
                  SDVersion version                              = VERSION_WAN2)
            : GGMLRunner(backend, offload_params_to_cpu) {
            wan_params.num_layers = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                size_t pos = tensor_name.find("vace_blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > wan_params.vace_layers) {
                            wan_params.vace_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                pos = tensor_name.find("blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > wan_params.num_layers) {
                            wan_params.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
                if (tensor_name.find("img_emb") != std::string::npos) {
                    wan_params.model_type = "i2v";
                }
                if (tensor_name.find("img_emb.emb_pos") != std::string::npos) {
                    wan_params.flf_pos_embed_token_number = 514;
                }
            }

            if (wan_params.num_layers == 30) {
                if (version == VERSION_WAN2_2_TI2V) {
                    desc                 = "Wan2.2-TI2V-5B";
                    wan_params.dim       = 3072;
                    wan_params.eps       = 1e-06f;
                    wan_params.ffn_dim   = 14336;
                    wan_params.freq_dim  = 256;
                    wan_params.in_dim    = 48;
                    wan_params.num_heads = 24;
                    wan_params.out_dim   = 48;
                    wan_params.text_len  = 512;
                } else {
                    if (wan_params.vace_layers > 0) {
                        desc              = "Wan2.1-VACE-1.3B";
                        wan_params.in_dim = 16;
                    } else if (wan_params.model_type == "i2v") {
                        desc              = "Wan2.1-I2V-1.3B";
                        wan_params.in_dim = 36;
                    } else {
                        desc              = "Wan2.1-T2V-1.3B";
                        wan_params.in_dim = 16;
                    }
                    wan_params.dim       = 1536;
                    wan_params.eps       = 1e-06f;
                    wan_params.ffn_dim   = 8960;
                    wan_params.freq_dim  = 256;
                    wan_params.num_heads = 12;
                    wan_params.out_dim   = 16;
                    wan_params.text_len  = 512;
                }
            } else if (wan_params.num_layers == 40) {
                if (wan_params.model_type == "t2v") {
                    if (version == VERSION_WAN2_2_I2V) {
                        desc              = "Wan2.2-I2V-14B";
                        wan_params.in_dim = 36;
                    } else {
                        if (wan_params.vace_layers > 0) {
                            desc = "Wan2.x-VACE-14B";
                        } else {
                            desc = "Wan2.x-T2V-14B";
                        }
                        wan_params.in_dim = 16;
                    }
                } else {
                    wan_params.in_dim = 36;
                    if (wan_params.flf_pos_embed_token_number > 0) {
                        desc = "Wan2.1-FLF2V-14B";
                    } else {
                        desc = "Wan2.1-I2V-14B";
                    }
                }
                wan_params.dim       = 5120;
                wan_params.eps       = 1e-06f;
                wan_params.ffn_dim   = 13824;
                wan_params.freq_dim  = 256;
                wan_params.num_heads = 40;
                wan_params.out_dim   = 16;
                wan_params.text_len  = 512;
            } else {
                GGML_ABORT("invalid num_layers(%d) of wan", wan_params.num_layers);
            }

            LOG_INFO("%s", desc.c_str());

            wan = Wan(wan_params);
            wan.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            wan.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        struct ggml_tensor* clip_fea        = nullptr,
                                        struct ggml_tensor* c_concat        = nullptr,
                                        struct ggml_tensor* time_dim_concat = nullptr,
                                        struct ggml_tensor* vace_context    = nullptr,
                                        float vace_strength                 = 1.f) {
            struct ggml_cgraph* gf = new_graph_custom(WAN_GRAPH_SIZE);

            x               = to_backend(x);
            timesteps       = to_backend(timesteps);
            context         = to_backend(context);
            clip_fea        = to_backend(clip_fea);
            c_concat        = to_backend(c_concat);
            time_dim_concat = to_backend(time_dim_concat);
            vace_context    = to_backend(vace_context);

            pe_vec      = Rope::gen_wan_pe(static_cast<int>(x->ne[2]),
                                           static_cast<int>(x->ne[1]),
                                           static_cast<int>(x->ne[0]),
                                           std::get<0>(wan_params.patch_size),
                                           std::get<1>(wan_params.patch_size),
                                           std::get<2>(wan_params.patch_size),
                                           1,
                                           wan_params.theta,
                                           wan_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / wan_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, wan_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe);
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());

            if (c_concat != nullptr) {
                x = ggml_concat(compute_ctx, x, c_concat, 3);
            }

            auto runner_ctx = get_context();

            struct ggml_tensor* out = wan.forward(&runner_ctx,
                                                  x,
                                                  timesteps,
                                                  context,
                                                  pe,
                                                  clip_fea,
                                                  time_dim_concat,
                                                  vace_context,
                                                  vace_strength);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* clip_fea        = nullptr,
                     struct ggml_tensor* c_concat        = nullptr,
                     struct ggml_tensor* time_dim_concat = nullptr,
                     struct ggml_tensor* vace_context    = nullptr,
                     float vace_strength                 = 1.f,
                     struct ggml_tensor** output         = nullptr,
                     struct ggml_context* output_ctx     = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, clip_fea, c_concat, time_dim_concat, vace_context, vace_strength);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(200 * 1024 * 1024);  // 200 MB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // cpu f16: pass
                // cuda f16: pass
                // cpu q8_0: pass
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 104, 60, 1, 16);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "wan_dit_x.bin");
                print_ggml_tensor(x);

                std::vector<float> timesteps_vec(3, 1000.f);
                timesteps_vec[0] = 0.f;
                auto timesteps   = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 512, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "wan_dit_context.bin");
                print_ggml_tensor(context);
                // auto clip_fea = load_tensor_from_file(work_ctx, "wan_dit_clip_fea.bin");
                // print_ggml_tensor(clip_fea);

                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, x, timesteps, context, nullptr, nullptr, nullptr, nullptr, 1.f, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("wan test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_F16;
            LOG_INFO("loading from '%s'", file_path.c_str());

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<WanRunner> wan = std::make_shared<WanRunner>(backend,
                                                                         false,
                                                                         tensor_storage_map,
                                                                         "model.diffusion_model",
                                                                         VERSION_WAN2_2_TI2V);

            wan->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            wan->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("wan model loaded");

            wan->test();
        }
    };

}  // namespace WAN

#endif  // __WAN_HPP__

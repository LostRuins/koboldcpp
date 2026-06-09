#ifndef __SD_MODEL_VAE_WAN_VAE_HPP__
#define __SD_MODEL_VAE_WAN_VAE_HPP__

#include <map>
#include <memory>
#include <utility>

#include "model/common/block.hpp"
#include "model/vae/vae.hpp"
#include "model_loader.h"

namespace WAN {

    constexpr int CACHE_T = 2;

    class CausalConv3d : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t out_channels;
        std::tuple<int, int, int> kernel_size;
        std::tuple<int, int, int> stride;
        std::tuple<int, int, int> padding;
        std::tuple<int, int, int> dilation;
        bool bias;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
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

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* cache_x = nullptr) {
            // x: [N*IC, ID, IH, IW]
            // result: x: [N*OC, ID, IH, IW]
            ggml_tensor* w = params["weight"];
            ggml_tensor* b = nullptr;
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

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
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

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            // x: [N*IC, ID, IH, IW], IC == dim
            // assert N == 1

            ggml_tensor* w = params["gamma"];
            w              = ggml_reshape_1d(ctx->ggml_ctx, w, ggml_nelements(w));
            auto h         = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 3, 0, 1, 2));  // [ID, IH, IW, N*IC]
            h              = ggml_rms_norm(ctx->ggml_ctx, h, 1e-12f);
            h              = ggml_mul(ctx->ggml_ctx, h, w);
            h              = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, h, 1, 2, 3, 0));

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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
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
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
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
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* h = x;
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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
                             int& feat_idx,
                             int chunk_idx) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            ggml_tensor* x_copy = x;

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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
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

            v = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, v, 1, 0, 2, 3));                            // [t, h * w, c]
            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, 1, nullptr, false, ctx->flash_attn_enabled);  // [t, h * w, c]

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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
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
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.prelude", "x");

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
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.down." + std::to_string(i), "x");
            }

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encoder.mid", "x");

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

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             int64_t b,
                             std::vector<ggml_tensor*>& feat_cache,
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
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.prelude", "x");

            // middle
            x = middle_0->forward(ctx, x, b, feat_cache, feat_idx);
            x = middle_1->forward(ctx, x, b);
            x = middle_2->forward(ctx, x, b, feat_cache, feat_idx);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.mid", "x");

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
                // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decoder.up." + std::to_string(i), "x");
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
        std::vector<ggml_tensor*> _feat_map;
        int _enc_conv_num = 28;
        int _enc_conv_idx = 0;
        std::vector<ggml_tensor*> _enc_feat_map;

        void clear_cache() {
            _conv_idx     = 0;
            _feat_map     = std::vector<ggml_tensor*>(_conv_num, nullptr);
            _enc_conv_idx = 0;
            _enc_feat_map = std::vector<ggml_tensor*>(_enc_conv_num, nullptr);
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

        static ggml_tensor* patchify(ggml_context* ctx,
                                     ggml_tensor* x,
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

        static ggml_tensor* unpatchify(ggml_context* ctx,
                                       ggml_tensor* x,
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

        ggml_tensor* encode(GGMLRunnerContext* ctx,
                            ggml_tensor* x,
                            int64_t b = 1) {
            // x: [b*c, t, h, w]
            GGML_ASSERT(b == 1);
            GGML_ASSERT(decode_only == false);

            clear_cache();

            if (wan2_2) {
                x = patchify(ctx->ggml_ctx, x, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.encode.prelude", "x");

            auto encoder = std::dynamic_pointer_cast<Encoder3d>(blocks["encoder"]);
            auto conv1   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv1"]);

            int64_t t     = x->ne[2];
            int64_t iter_ = 1 + (t - 1) / 4;
            ggml_tensor* out;
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
            // sd::ggml_graph_cut::mark_graph_cut(mu, "wan_vae.encode.final", "mu");
            clear_cache();
            return mu;
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx,
                            ggml_tensor* z,
                            int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            clear_cache();

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            int64_t iter_ = z->ne[2];
            auto x        = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode.prelude", "x");
            ggml_tensor* out;
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
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode.final", "out");
            clear_cache();
            return out;
        }

        ggml_tensor* decode_partial(GGMLRunnerContext* ctx,
                                    ggml_tensor* z,
                                    int i,
                                    int64_t b = 1) {
            // z: [b*c, t, h, w]
            GGML_ASSERT(b == 1);

            auto decoder = std::dynamic_pointer_cast<Decoder3d>(blocks["decoder"]);
            auto conv2   = std::dynamic_pointer_cast<CausalConv3d>(blocks["conv2"]);

            auto x = conv2->forward(ctx, z);
            // sd::ggml_graph_cut::mark_graph_cut(x, "wan_vae.decode_partial.prelude", "x");
            auto in   = ggml_ext_slice(ctx->ggml_ctx, x, 2, i, i + 1);  // [b*c, 1, h, w]
            _conv_idx = 0;
            auto out  = decoder->forward(ctx, in, b, _feat_map, _conv_idx, i);
            if (wan2_2) {
                out = unpatchify(ctx->ggml_ctx, out, 2, b);
            }
            // sd::ggml_graph_cut::mark_graph_cut(out, "wan_vae.decode_partial.final", "out");
            return out;
        }
    };

    struct WanVAERunner : public VAE {
        float scale_factor = 1.0f;
        bool decode_only   = true;
        WanVAE ae;

        WanVAERunner(ggml_backend_t backend,
                     ggml_backend_t params_backend,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     bool decode_only                               = false,
                     SDVersion version                              = VERSION_WAN2)
            : decode_only(decode_only), ae(decode_only, version == VERSION_WAN2_2_TI2V), VAE(version, backend, params_backend) {
            ae.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "wan_vae";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) override {
            ae.get_param_tensors(tensors, prefix);
        }

        sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
            SD_UNUSED(rng);
            return vae_output;
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> get_latents_mean_std(const sd::Tensor<float>& latents) {
            int channel_dim = latents.dim() == 5 ? 3 : 2;
            std::vector<int64_t> stats_shape(static_cast<size_t>(latents.dim()), 1);
            if (latents.shape()[channel_dim] == 16) {  // Wan2.1 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 16;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                                                   0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                                                  3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            if (latents.shape()[channel_dim] == 48) {  // Wan2.2 VAE
                stats_shape[static_cast<size_t>(channel_dim)] = 48;

                auto mean_tensor = sd::Tensor<float>::from_vector({-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f,
                                                                   -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
                                                                   -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f,
                                                                   -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
                                                                   -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f,
                                                                   0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f});
                mean_tensor.reshape_(stats_shape);
                auto std_tensor = sd::Tensor<float>::from_vector({0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
                                                                  0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
                                                                  0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
                                                                  0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
                                                                  0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
                                                                  0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f});
                std_tensor.reshape_(stats_shape);
                return {std::move(mean_tensor), std::move(std_tensor)};
            }
            GGML_ABORT("unexpected latent channel dimension %lld for version %d",
                       (long long)latents.shape()[channel_dim],
                       version);
        }

        sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return (latents * std_tensor) / scale_factor + mean_tensor;
        }

        sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
            auto [mean_tensor, std_tensor] = get_latents_mean_std(latents);
            return ((latents - mean_tensor) * scale_factor) / std_tensor;
        }

        int get_encoder_output_channels(int input_channels) {
            return static_cast<int>(ae.z_dim);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& z_tensor, bool decode_graph) {
            ggml_cgraph* gf = new_graph_custom(10240 * z_tensor.shape()[2]);
            ggml_tensor* z  = make_input(z_tensor);

            auto runner_ctx = get_context();

            ggml_tensor* out = decode_graph ? ae.decode(&runner_ctx, z) : ae.encode(&runner_ctx, z);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        ggml_cgraph* build_graph_partial(const sd::Tensor<float>& z_tensor, bool decode_graph, int i) {
            ggml_cgraph* gf = new_graph_custom(20480);

            ae.clear_cache();

            for (size_t feat_idx = 0; feat_idx < ae._feat_map.size(); feat_idx++) {
                auto feat_cache        = get_cache_tensor_by_name("feat_idx:" + std::to_string(feat_idx));
                ae._feat_map[feat_idx] = feat_cache;
            }

            ggml_tensor* z = make_input(z_tensor);

            auto runner_ctx = get_context();

            ggml_tensor* out = decode_graph ? ae.decode_partial(&runner_ctx, z, i) : ae.encode(&runner_ctx, z);

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

        sd::Tensor<float> _compute(const int n_threads,
                                   const sd::Tensor<float>& z,
                                   bool decode_graph) override {
            if (true) {
                sd::Tensor<float> input;
                if (z.dim() == 4) {
                    input = z.unsqueeze(2);
                }
                auto get_graph = [&]() -> ggml_cgraph* {
                    if (input.empty()) {
                        return build_graph(z, decode_graph);
                    } else {
                        return build_graph(input, decode_graph);
                    }
                };
                auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, true),
                                                              input.empty() ? z.dim() : input.dim());
                if (!result.empty() && z.dim() == 4) {
                    result.squeeze_(2);
                }
                return result;
            } else {  // chunk 1 result is weird
                ae.clear_cache();
                int64_t t      = z.shape()[2];
                int i          = 0;
                auto get_graph = [&]() -> ggml_cgraph* {
                    return build_graph_partial(z, decode_graph, i);
                };
                auto out_opt = GGMLRunner::compute<float>(get_graph, n_threads, true);
                if (!out_opt.has_value()) {
                    return {};
                }
                sd::Tensor<float> out = std::move(*out_opt);
                ae.clear_cache();
                if (t == 1) {
                    return out;
                }

                sd::Tensor<float> output = std::move(out);

                for (i = 1; i < t; i++) {
                    auto chunk_opt = GGMLRunner::compute<float>(get_graph, n_threads, true);
                    if (!chunk_opt.has_value()) {
                        return {};
                    }
                    out = std::move(*chunk_opt);
                    ae.clear_cache();
                    output = sd::ops::concat(output, out, 2);
                }
                free_cache_ctx_and_buffer();
                return output;
            }
        }

        void test() {
            ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            ggml_context* ctx = ggml_init(params);
            GGML_ASSERT(ctx != nullptr);

            if (true) {
                // cpu f32, pass
                // cpu f16, pass
                // cuda f16, pass
                // cuda f32, pass
                auto z = sd::load_tensor_from_file_as_tensor<float>("wan_vae_z.bin");
                print_sd_tensor(z);
                sd::Tensor<float> out;

                int64_t t0   = ggml_time_ms();
                auto out_opt = _compute(8, z, true);
                int64_t t1   = ggml_time_ms();

                GGML_ASSERT(!out_opt.empty());
                out = std::move(out_opt);
                print_sd_tensor(out);
                LOG_DEBUG("decode test done in %ldms", t1 - t0);
            }
        };

        static void load_from_file_and_test(const std::string& file_path) {
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend            = sd_backend_cpu_init();
            ggml_type model_data_type         = GGML_TYPE_F16;
            std::shared_ptr<WanVAERunner> vae = std::make_shared<WanVAERunner>(backend, backend, String2TensorStorage{}, "", false, VERSION_WAN2_2_TI2V);
            {
                LOG_INFO("loading from '%s'", file_path.c_str());

                if (!vae->alloc_params_buffer()) {
                    LOG_ERROR("vae buffer allocation failed");
                    return;
                }
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

}  // namespace WAN

#endif  // __SD_MODEL_VAE_WAN_VAE_HPP__

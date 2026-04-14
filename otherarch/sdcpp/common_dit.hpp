#ifndef __COMMON_DIT_HPP__
#define __COMMON_DIT_HPP__

#include "ggml_extend.hpp"

namespace DiT {
    ggml_tensor* patchify(ggml_context* ctx,
                          ggml_tensor* x,
                          int pw,
                          int ph,
                          bool patch_last = true) {
        // x: [N, C, H, W]
        // return: [N, h*w, C*ph*pw] if patch_last else [N, h*w, ph*pw*C]
        int64_t N = x->ne[3];
        int64_t C = x->ne[2];
        int64_t H = x->ne[1];
        int64_t W = x->ne[0];
        int64_t h = H / ph;
        int64_t w = W / pw;

        GGML_ASSERT(h * ph == H && w * pw == W);

        x = ggml_reshape_4d(ctx, x, pw, w, ph, h * C * N);     // [N*C*h, ph, w, pw]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, w, ph, pw]
        x = ggml_reshape_4d(ctx, x, pw * ph, w * h, C, N);     // [N, C, h*w, ph*pw]
        if (patch_last) {
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, h*w, C, ph*pw]
            x = ggml_reshape_3d(ctx, x, pw * ph * C, w * h, N);    // [N, h*w, C*ph*pw]
        } else {
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, h*w, C, ph*pw]
            x = ggml_reshape_3d(ctx, x, C * pw * ph, w * h, N);              // [N, h*w, ph*pw*C]
        }
        return x;
    }

    ggml_tensor* unpatchify(ggml_context* ctx,
                            ggml_tensor* x,
                            int64_t h,
                            int64_t w,
                            int ph,
                            int pw,
                            bool patch_last = true) {
        // x: [N, h*w, C*ph*pw] if patch_last else [N, h*w, ph*pw*C]
        // return: [N, C, H, W]
        int64_t N = x->ne[2];
        int64_t C = x->ne[0] / ph / pw;
        int64_t H = h * ph;
        int64_t W = w * pw;

        GGML_ASSERT(C * ph * pw == x->ne[0]);

        if (patch_last) {
            x = ggml_reshape_4d(ctx, x, pw * ph, C, w * h, N);     // [N, h*w, C, ph*pw]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, C, h*w, ph*pw]
        } else {
            x = ggml_reshape_4d(ctx, x, C, pw * ph, w * h, N);     // [N, h*w, ph*pw, C]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [N, C, h*w, ph*pw]
        }

        x = ggml_reshape_4d(ctx, x, pw, ph, w, h * C * N);     // [N*C*h, w, ph, pw]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, ph, w, pw]
        x = ggml_reshape_4d(ctx, x, W, H, C, N);               // [N, C, h*ph, w*pw]

        return x;
    }

    ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                   ggml_tensor* x,
                                   int ph,
                                   int pw) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];

        int pad_h = (ph - H % ph) % ph;
        int pad_w = (pw - W % pw) % pw;
        x         = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
        return x;
    }

    ggml_tensor* pad_and_patchify(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  int ph,
                                  int pw,
                                  bool patch_last = true) {
        x = pad_to_patch_size(ctx, x, ph, pw);
        x = patchify(ctx->ggml_ctx, x, ph, pw, patch_last);
        return x;
    }

    ggml_tensor* unpatchify_and_crop(ggml_context* ctx,
                                     ggml_tensor* x,
                                     int64_t H,
                                     int64_t W,
                                     int ph,
                                     int pw,
                                     bool patch_last = true) {
        int pad_h = (ph - H % ph) % ph;
        int pad_w = (pw - W % pw) % pw;
        int64_t h = ((H + pad_h) / ph);
        int64_t w = ((W + pad_w) / pw);
        x         = unpatchify(ctx, x, h, w, ph, pw, patch_last);  // [N, C, H + pad_h, W + pad_w]
        x         = ggml_ext_slice(ctx, x, 1, 0, H);               // [N, C, H, W + pad_w]
        x         = ggml_ext_slice(ctx, x, 0, 0, W);               // [N, C, H, W]
        return x;
    }
}  // namespace DiT

#endif  // __COMMON_DIT_HPP__
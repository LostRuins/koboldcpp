#include "core/ggml_extend.h"

#include <cmath>
#include <utility>

#include "core/ggml_extend_backend.h"

ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int mode) {
    // reshape A
    // swap 0th and nth axis
    a           = ggml_cont(ctx, ggml_permute(ctx, a, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0));
    int64_t ne1 = a->ne[1];
    int64_t ne2 = a->ne[2];
    int64_t ne3 = a->ne[3];
    // make 2D
    a = ggml_cont(ctx, ggml_reshape_2d(ctx, a, a->ne[0], (ne3 * ne2 * ne1)));

    ggml_tensor* result = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, a, b)));

    // reshape output (same shape as a after permutation except first dim)
    result = ggml_reshape_4d(ctx, result, result->ne[0], ne1, ne2, ne3);
    // swap back 0th and nth axis
    result = ggml_permute(ctx, result, mode, mode != 1 ? 1 : 0, mode != 2 ? 2 : 0, mode != 3 ? 3 : 0);
    return result;
}

ggml_tensor* ggml_ext_kronecker(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul(ctx,
                    ggml_interpolate(ctx,
                                     a,
                                     a->ne[0] * b->ne[0],
                                     a->ne[1] * b->ne[1],
                                     a->ne[2] * b->ne[2],
                                     a->ne[3] * b->ne[3],
                                     GGML_SCALE_MODE_NEAREST),
                    b);
}

ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                           ggml_tensor* x) {
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

ggml_tensor* ggml_ext_torch_permute(ggml_context* ctx,
                                    ggml_tensor* x,
                                    int axis0,
                                    int axis1,
                                    int axis2,
                                    int axis3) {
    int torch_axes[4] = {axis0, axis1, axis2, axis3};

    int ggml_axes[4] = {0};
    for (int i = 0; i < 4; ++i) {
        int found = 0;
        for (int j = 0; j < 4; ++j) {
            if (torch_axes[j] == i) {
                ggml_axes[i] = j;
                found        = 1;
                break;
            }
        }
        GGML_ASSERT(found && "Invalid permute input: must be a permutation of 0-3");
    }

    return ggml_permute(ctx, x, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]);
}

ggml_tensor* ggml_ext_slice(ggml_context* ctx,
                            ggml_tensor* x,
                            int dim,
                            int64_t start,
                            int64_t end,
                            bool cont) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    if (x->ne[dim] == 1) {
        return x;
    }
    while (start < 0) {
        start = x->ne[dim] + start;
    }
    while (end < 0) {
        end = x->ne[dim] + end;
    }
    GGML_ASSERT(end > start);
    GGML_ASSERT(start >= 0 && start < x->ne[dim]);
    GGML_ASSERT(end > start && end <= x->ne[dim]);

    int64_t slice_size  = end - start;
    int64_t slice_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    slice_ne[dim]       = slice_size;

    x = ggml_view_4d(ctx, x,
                     slice_ne[0], slice_ne[1], slice_ne[2], slice_ne[3],
                     x->nb[1], x->nb[2], x->nb[3], start * x->nb[dim]);

    if (cont) {
        x = ggml_cont(ctx, x);
    }

    return x;
}

std::vector<ggml_tensor*> ggml_ext_chunk(ggml_context* ctx,
                                         ggml_tensor* x,
                                         int num,
                                         int64_t dim,
                                         bool cont) {
    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(x->ne[dim] % num == 0);

    std::vector<ggml_tensor*> chunks;
    int64_t chunk_size  = x->ne[dim] / num;
    int64_t stride      = chunk_size * x->nb[dim];
    int64_t chunk_ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    chunk_ne[dim]       = chunk_size;
    for (int i = 0; i < num; i++) {
        auto chunk = ggml_view_4d(
            ctx, x,
            chunk_ne[0], chunk_ne[1], chunk_ne[2], chunk_ne[3],
            x->nb[1], x->nb[2], x->nb[3], stride * i);
        if (cont) {
            chunk = ggml_cont(ctx, chunk);
        }
        chunks.push_back(chunk);
    }

    return chunks;
}

ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first) {
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    auto x_vec = ggml_ext_chunk(ctx, x, 2, 0, false);
    ggml_tensor* gate;
    if (gate_first) {
        gate = x_vec[0];
        x    = x_vec[1];
    } else {
        x    = x_vec[0];
        gate = x_vec[1];
    }
    gate = ggml_cont(ctx, gate);
    gate = ggml_silu_inplace(ctx, gate);

    x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, ne0/2]

    return x;
}

ggml_tensor* ggml_ext_group_norm_32(ggml_context* ctx,
                                    ggml_tensor* a) {
    const float eps = 1e-6f;  // default eps parameter
    return ggml_group_norm(ctx, a, 32, eps);
}

static bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
    return x->nb[0] == ggml_type_size(x->type) &&
           x->nb[2] == x->nb[1] * x->ne[1] &&
           x->nb[3] == x->nb[2] * x->ne[2];
}

ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                            ggml_tensor* x,
                            float factor,
                            bool inplace) {
    if (!ggml_ext_is_padded_1d(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_scale_inplace(ctx, x, factor);
    } else {
        x = ggml_scale(ctx, x, factor);
    }
    return x;
}

ggml_tensor* ggml_ext_gelu(ggml_context* ctx,
                           ggml_tensor* x,
                           bool inplace) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_inplace(ctx, x);
    } else {
        x = ggml_gelu(ctx, x);
    }
    return x;
}

ggml_tensor* ggml_ext_gelu_quick(ggml_context* ctx,
                                 ggml_tensor* x,
                                 bool inplace) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (inplace) {
        x = ggml_gelu_quick_inplace(ctx, x);
    } else {
        x = ggml_gelu_quick(ctx, x);
    }
    return x;
}

ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                             ggml_tensor* x,
                             ggml_tensor* w,
                             ggml_tensor* b,
                             bool force_prec_f32,
                             float scale) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_mul_mat_set_prec(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

#if KCPP_MAINLINE_INT8_CONVROT
ggml_tensor* ggml_ext_linear_i8_tensorwise(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* weight_scale,
                                           ggml_tensor* b,
                                           int convrot_group_size,
                                           float scale) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 || (x->type == GGML_TYPE_I8 && scale == 1.f));
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }

    ggml_tensor* fused_bias = scale == 1.f ? b : nullptr;
    if (x->ne[2] * x->ne[3] > 1024) {
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat_i8_tensorwise(ctx, w, x, weight_scale, fused_bias, convrot_group_size);
        x           = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat_i8_tensorwise(ctx, w, x, weight_scale, fused_bias, convrot_group_size);
    }

    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}
#endif //kcpp

ggml_tensor* ggml_ext_pad_ext(ggml_context* ctx,
                              ggml_backend_t backend,
                              ggml_tensor* x,
                              int lp0,
                              int rp0,
                              int lp1,
                              int rp1,
                              int lp2,
                              int rp2,
                              int lp3,
                              int rp3,
                              bool circular_x,
                              bool circular_y) {
    if (circular_x && circular_y) {
        return ggml_pad_ext_circular(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    }

    if (circular_x && (lp0 != 0 || rp0 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, lp0, rp0, 0, 0, 0, 0, 0, 0);
        lp0 = rp0 = 0;
    }
    if (circular_y && (lp1 != 0 || rp1 != 0)) {
        x   = ggml_pad_ext_circular(ctx, x, 0, 0, lp1, rp1, 0, 0, 0, 0);
        lp1 = rp1 = 0;
    }

    if (lp0 != 0 || rp0 != 0 || lp1 != 0 || rp1 != 0 || lp2 != 0 || rp2 != 0 || lp3 != 0 || rp3 != 0) {
        ggml_tensor* padded = ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
        if (backend == nullptr || ggml_backend_supports_op(backend, padded)) {
            x = padded;
        } else {
            // Some backends (e.g. Metal) only implement right-padding for
            // GGML_OP_PAD (see #850): pad right by lp+rp instead, then roll
            // the padding around to the left. shift < ne always holds because
            // ne grew by lp+rp.
            x = ggml_pad_ext(ctx, x, 0, lp0 + rp0, 0, lp1 + rp1, 0, lp2 + rp2, 0, lp3 + rp3);
            x = ggml_roll(ctx, x, lp0, lp1, lp2, lp3);
        }
    }
    return x;
}

ggml_tensor* ggml_ext_pad(ggml_context* ctx,
                          ggml_tensor* x,
                          int p0,
                          int p1,
                          int p2,
                          int p3,
                          bool circular_x,
                          bool circular_y) {
    return ggml_ext_pad_ext(ctx, nullptr, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

ggml_tensor* ggml_ext_conv_2d(ggml_context* ctx,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int s0,
                              int s1,
                              int p0,
                              int p1,
                              int d0,
                              int d1,
                              bool direct,
                              bool circular_x,
                              bool circular_y,
                              float scale) {
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(ctx, nullptr, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_conv_3d(ggml_context* ctx,
                              ggml_backend_t backend,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int64_t IC,
                              int s0,
                              int s1,
                              int s2,
                              int p0,
                              int p1,
                              int p2,
                              int d0,
                              int d1,
                              int d2,
                              bool force_prec_f32) {
    if (force_prec_f32) {
        ggml_tensor* im2col = ggml_im2col_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, w->type);

        int64_t OC = w->ne[3] / IC;
        int64_t N  = x->ne[3] / IC;
        x          = ggml_mul_mat(ctx,
                                  ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                                  ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2] * IC, OC));
        ggml_mul_mat_set_prec(x, GGML_PREC_F32);

        int64_t OD = im2col->ne[3] / N;
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1] * im2col->ne[2], OD, N, OC);
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1], im2col->ne[2], OD, OC * N);
    } else {
        // ggml_conv_3d decomposes into GGML_OP_IM2COL_3D, which some backends
        // (e.g. Metal, see #850) do not implement. Fall back to
        // GGML_OP_CONV_3D on those backends.
        bool im2col_3d_supported = true;
        if (backend != nullptr) {
            ggml_tensor* im2col = ggml_im2col_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, w->type);
            im2col_3d_supported = ggml_backend_supports_op(backend, im2col);
        }
        if (im2col_3d_supported) {
            x = ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);
        } else {
            int64_t OC = w->ne[3] / IC;
            int64_t N  = x->ne[3] / IC;
            x          = ggml_conv_3d_direct(ctx, w, x, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC);
        }
    }

    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_conv_3d_nx1x1(ggml_context* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* w,
                                    ggml_tensor* b,
                                    int s2,
                                    int p2,
                                    int d2) {
    x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }
    return x;  // [N, OC, T, OH * OW]
}

std::vector<ggml_tensor*> split_qkv(ggml_context* ctx,
                                    ggml_tensor* qkv) {
    qkv = ggml_reshape_4d(ctx, qkv, qkv->ne[0] / 3, 3, qkv->ne[1], qkv->ne[2]);  // [N, L, 3, C]
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));                    // [3, N, L, C]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 0);  // [N, L, C]
    auto k         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 1);  // [N, L, C]
    auto v         = ggml_view_3d(ctx, qkv, qkv->ne[0], qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], offset * 2);  // [N, L, C]
    return {q, k, v};
}

std::vector<ggml_tensor*> split_image_qkv(ggml_context* ctx,
                                          ggml_tensor* qkv) {
    int64_t W   = qkv->ne[0];
    int64_t H   = qkv->ne[1];
    int64_t C   = qkv->ne[2] / 3;
    int64_t N   = qkv->ne[3];
    int64_t nb1 = qkv->nb[1];
    int64_t nb2 = qkv->nb[2];
    qkv         = ggml_reshape_4d(ctx, qkv, W * H, C, 3, N);                     // [N, 3, C, H*W]
    qkv         = ggml_cont(ctx, ggml_ext_torch_permute(ctx, qkv, 0, 1, 3, 2));  // [3, N, C, H*W]

    int64_t offset = qkv->nb[2] * qkv->ne[2];
    auto q         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 0);  // [N, C, H, W]
    auto k         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 1);  // [N, C, H, W]
    auto v         = ggml_view_4d(ctx, qkv, W, H, C, N, nb1, nb2, qkv->nb[3], offset * 2);  // [N, C, H, W]
    return {q, k, v};
}

ggml_tensor* ggml_ext_full(ggml_context* ctx,
                           float value,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3) {
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(ctx, one, value);             // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

ggml_tensor* ggml_ext_zeros(ggml_context* ctx,
                            int64_t ne0,
                            int64_t ne1,
                            int64_t ne2,
                            int64_t ne3) {
    return ggml_ext_full(ctx, 0.f, ne0, ne1, ne2, ne3);
}

ggml_tensor* ggml_ext_zeros_like(ggml_context* ctx,
                                 ggml_tensor* x) {
    return ggml_ext_zeros(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

ggml_tensor* ggml_ext_ones(ggml_context* ctx,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3) {
    return ggml_ext_full(ctx, 1.f, ne0, ne1, ne2, ne3);
}

ggml_tensor* ggml_ext_ones_like(ggml_context* ctx,
                                ggml_tensor* x) {
    return ggml_ext_ones(ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

ggml_tensor* ggml_ext_cast_f32(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* a) {
    if (sd_backend_is(backend, "Vulkan")) {
        auto zero_index = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:zero_int");
        auto out        = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out             = ggml_get_rows(ctx, out, zero_index);
        out             = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    } else {
        auto out         = ggml_reshape_2d(ctx, a, 1, ggml_nelements(a));
        ggml_tensor* one = ggml_ext_ones(ctx, 1, 1, 1, 1);  // [1,]
        if (ggml_is_transposed(out)) {
            out = ggml_mul_mat(ctx, one, out);
        } else {
            out = ggml_mul_mat(ctx, out, one);
        }
        out = ggml_reshape(ctx, out, a);
        return out;
    }
}

ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
                                    ggml_backend_t backend,
                                    ggml_tensor* q,
                                    ggml_tensor* k,
                                    ggml_tensor* v,
                                    int64_t n_head,
                                    ggml_tensor* mask,
                                    bool skip_reshape,
                                    bool flash_attn,
                                    float kv_scale) {  // avoid overflow
    int64_t L_q;
    int64_t L_k;
    int64_t C;
    int64_t N;
    int64_t d_head;
    int64_t n_kv_head;
    if (!skip_reshape) {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        C         = q->ne[0];
        N         = q->ne[2];
        d_head    = C / n_head;
        n_kv_head = k->ne[0] / d_head;

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);       // [N, L_q, n_head, d_head]
        q = ggml_ext_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);      // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);    // [N, L_k, n_kv_head, d_head]
        k = ggml_ext_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);   // [N * n_kv_head, L_k, d_head]

        v = ggml_reshape_4d(ctx, v, d_head, n_kv_head, L_k, N);  // [N, L_k, n_kv_head, d_head]
    } else {
        L_q       = q->ne[1];
        L_k       = k->ne[1];
        d_head    = v->ne[0];
        N         = v->ne[3];
        n_kv_head = k->ne[2] / N;
        C         = d_head * n_head;
    }

    float scale = (1.0f / sqrt((float)d_head));

    ggml_tensor* kqv = nullptr;

    auto build_kqv = [&](ggml_tensor* q_in, ggml_tensor* k_in, ggml_tensor* v_in, ggml_tensor* mask_in) -> ggml_tensor* {
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        v_in = ggml_ext_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));
        v_in = ggml_reshape_3d(ctx, v_in, d_head, L_k, n_kv_head * N);
        if (kv_scale != 1.0f) {
            v_in = ggml_ext_scale(ctx, v_in, kv_scale);
        }
        v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16);

        if (mask_in != nullptr) {
            // ggml_flash_attn_ext expects the mask as a contiguous F16 tensor shaped
            // [n_kv, n_q, (heads), (batch)] (ne0 = key length, ne1 = query length) and,
            // unlike the manual-attention path, does not broadcast the query dimension.
            // Some callers (e.g. Chroma/T5) pass a per-key padding mask broadcast over
            // queries ([n_kv, 1, ...]); materialize the query dimension to L_q so the
            // kernel indexes it correctly. (A bare ggml_transpose here produced a
            // [1, n_kv, ...] mask that the kernel silently misreads, yielding NaN/blank
            // output for masked flash attention.)
            if (mask_in->ne[1] != L_q) {
                mask_in = ggml_repeat(ctx, mask_in,
                                      ggml_new_tensor_4d(ctx, mask_in->type, mask_in->ne[0], L_q, mask_in->ne[2], mask_in->ne[3]));
            }
            mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
        }

        auto out = ggml_flash_attn_ext(ctx, q_in, k_in, v_in, mask_in, scale / kv_scale, 0, 0);
        if (!ggml_backend_supports_op(backend, out)) {
            return nullptr;
        }
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(ctx, out, 1.0f / kv_scale);
        }
        return out;
    };

    if (flash_attn) {
        // LOG_VERBOSE("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (mask != nullptr) {
            // TODO: figure out if we can bend t5 to work too
            can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
        }

        if (can_use_flash_attn) {
            kqv = build_kqv(q, k, v, mask);
            if (kqv != nullptr) {
                kqv = ggml_view_4d(ctx,
                                   kqv,
                                   d_head,
                                   n_head,
                                   L_q,
                                   N,
                                   kqv->nb[1],
                                   kqv->nb[2],
                                   kqv->nb[1] * n_head,
                                   0);
            }
        }
    }

    if (kqv == nullptr) {
        // if (flash_attn) {
        //     LOG_VERBOSE("fallback to default attention, L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        // }
        v = ggml_ext_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);   // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

ggml_tensor* ggml_ext_group_norm(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 int num_groups) {
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    const float eps = 1e-6f;  // default eps parameter
    x               = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        // b = ggml_repeat(ctx, b, x);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_timestep_embedding(
    ggml_context* ctx,
    ggml_tensor* timesteps,
    int dim,
    int max_period,
    float time_factor) {
    timesteps = ggml_ext_scale(ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

ggml_tensor* ggml_ext_vec_concat(ggml_context* ctx,
                                 std::vector<ggml_tensor*>& tensors,
                                 int dim) {
    while (tensors.size() > 1) {
        std::vector<ggml_tensor*> next_level;
        for (size_t i = 0; i < tensors.size(); i += 2) {
            if (i + 1 < tensors.size()) {
                next_level.push_back(ggml_concat(ctx, tensors[i], tensors[i + 1], dim));
            } else {
                next_level.push_back(tensors[i]);
            }
        }
        tensors = std::move(next_level);
    }
    return tensors[0];
}

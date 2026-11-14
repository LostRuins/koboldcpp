#include "core/ggml_extend.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/ggml_extend_backend.h"
#include "core/ggml_runner.h"

ggml_tensor* ggml_ext_cont(GGMLRunnerContext* runner_ctx, ggml_tensor* x) {
    auto ctx = runner_ctx->ggml_ctx;
    if (ggml_is_contiguous(x)) {
        return x;
    }
    return ggml_cont(ctx, x);
}

ggml_tensor* ggml_ext_torch_permute(GGMLRunnerContext* runner_ctx,
                                    ggml_tensor* x,
                                    int axis0,
                                    int axis1,
                                    int axis2,
                                    int axis3) {
    auto ctx          = runner_ctx->ggml_ctx;
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

ggml_tensor* ggml_ext_slice(GGMLRunnerContext* runner_ctx,
                            ggml_tensor* x,
                            int dim,
                            int64_t start,
                            int64_t end,
                            bool cont) {
    auto ctx = runner_ctx->ggml_ctx;
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

std::vector<ggml_tensor*> ggml_ext_chunk(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* x,
                                         int num,
                                         int64_t dim,
                                         bool cont) {
    auto ctx = runner_ctx->ggml_ctx;
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

ggml_tensor* ggml_ext_silu_act(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool gate_first) {
    auto ctx = runner_ctx->ggml_ctx;
    // x: [ne3, ne2, ne1, ne0]
    // return: [ne3, ne2, ne1, ne0/2]

    GGML_ASSERT(x->ne[0] % 2 == 0);
    if (!ggml_is_contiguous_1(x)) {
        x = ggml_cont(ctx, x);
    }
    return gate_first ? ggml_swiglu(ctx, x) : ggml_swiglu_swapped(ctx, x);
}

ggml_tensor* ggml_ext_situ(GGMLRunnerContext* runner_ctx, ggml_tensor* hidden, ggml_tensor* gate) {
#ifndef SD_USE_UPSTREAM_GGML
    if (runner_ctx->backend != nullptr) {
        auto out = gate ? ggml_glu_split(runner_ctx->ggml_ctx, gate, hidden, GGML_GLU_OP_SITU)
                        : ggml_glu(runner_ctx->ggml_ctx, hidden, GGML_GLU_OP_SITU, true);
        if (ggml_backend_supports_op(runner_ctx->backend, out)) {
            return out;
        }
    }
#endif
    if (!gate) {
        const auto dim = hidden->ne[0] / 2;
        gate           = ggml_ext_slice(runner_ctx, hidden, 0, dim, 2 * dim);
        hidden         = ggml_ext_slice(runner_ctx, hidden, 0, 0, dim);
    }
    auto a = ggml_scale(runner_ctx->ggml_ctx, ggml_tanh(runner_ctx->ggml_ctx, ggml_scale(runner_ctx->ggml_ctx, gate, 0.25f)), 4.f);
    a      = ggml_mul(runner_ctx->ggml_ctx, a, ggml_sigmoid(runner_ctx->ggml_ctx, gate));
    auto b = ggml_scale(runner_ctx->ggml_ctx, ggml_tanh(runner_ctx->ggml_ctx, ggml_scale(runner_ctx->ggml_ctx, hidden, 1.f / 25.f)), 25.f);
    return ggml_mul(runner_ctx->ggml_ctx, a, b);
}

static bool ggml_ext_is_padded_1d(const ggml_tensor* x) {
    return x->nb[0] == ggml_type_size(x->type) &&
           x->nb[2] == x->nb[1] * x->ne[1] &&
           x->nb[3] == x->nb[2] * x->ne[2];
}

ggml_tensor* ggml_ext_modulate(GGMLRunnerContext* runner_ctx,
                               ggml_tensor* x,
                               ggml_tensor* shift,
                               ggml_tensor* scale) {
#ifndef SD_USE_UPSTREAM_GGML
    if (x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && (!shift || shift->type == GGML_TYPE_F32) &&
        (sd_backend_is_cpu(runner_ctx->backend) || sd_backend_is(runner_ctx->backend, "CUDA") ||
         sd_backend_is(runner_ctx->backend, "ROCm") || sd_backend_is(runner_ctx->backend, "MUSA") || sd_backend_is(runner_ctx->backend, "Vulkan"))) {
        ggml_tensor* inputs[] = {x, scale, shift};
        return ggml_ext_op_create(runner_ctx->ggml_ctx, sd_ggml_modulate_op(), inputs, shift ? 3 : 2, nullptr);
    }
#endif
    // Preserve separate multiply/add nodes for rounding and backend fusion matching.
    x = ggml_add(runner_ctx->ggml_ctx, x, ggml_mul(runner_ctx->ggml_ctx, x, scale));
    if (shift != nullptr) {
        x = ggml_add(runner_ctx->ggml_ctx, x, shift);
    }
    return x;
}

ggml_tensor* ggml_ext_scale(GGMLRunnerContext* runner_ctx, ggml_tensor* x, float factor, bool inplace) {
    auto ctx = runner_ctx->ggml_ctx;
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

ggml_tensor* ggml_ext_gelu(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool inplace) {
    auto ctx = runner_ctx->ggml_ctx;
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

ggml_tensor* ggml_ext_gelu_quick(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool inplace) {
    auto ctx = runner_ctx->ggml_ctx;
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

ggml_tensor* ggml_ext_linear(GGMLRunnerContext* runner_ctx,
                             ggml_tensor* x,
                             ggml_tensor* w,
                             ggml_tensor* b,
                             bool force_prec_f32,
                             float scale) {
    auto ctx = runner_ctx->ggml_ctx;
    w        = ggml_ext_prepare_linear_weight(runner_ctx, w, x, force_prec_f32);
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, scale);
    }
    if (x->ne[2] * x->ne[3] > 1024) {
        // workaround: avoid ggml cuda error
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_prec_set_acc(x, GGML_PREC_F32);
        }
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = ggml_mul_mat(ctx, w, x);
        if (force_prec_f32) {
            ggml_prec_set_acc(x, GGML_PREC_F32);
        }
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_linear_i8_tensorwise(GGMLRunnerContext* runner_ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* weight_scale,
                                           ggml_tensor* b,
                                           int convrot_group_size,
                                           float scale) {
    auto ctx = runner_ctx->ggml_ctx;
#ifndef SD_USE_UPSTREAM_GGML
    x = ggml_ext_prepare_i8_input(runner_ctx, x);
    if (x->type == GGML_TYPE_F32 && convrot_group_size > 0 && scale == 1.f) {
        x = ggml_ext_quantize_i8_convrot(runner_ctx, x, convrot_group_size);
    }
    GGML_ASSERT(x->type == GGML_TYPE_F32 || (x->type == GGML_TYPE_I8 && scale == 1.f));
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, scale);
    }

    ggml_tensor* fused_bias = scale == 1.f ? b : nullptr;
    auto mul_mat            = [&](ggml_tensor* input) {
        if (input->type == GGML_TYPE_F32 && convrot_group_size > 0) {
            input = ggml_ext_quantize_i8_convrot(runner_ctx, input, convrot_group_size);
        }
        return ggml_ext_mul_mat_i8_tensorwise(runner_ctx, w, input, weight_scale, fused_bias, convrot_group_size);
    };
    if (x->ne[2] * x->ne[3] > 1024) {
        int64_t ne2 = x->ne[2];
        int64_t ne3 = x->ne[3];
        x           = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
        x           = mul_mat(x);
        x           = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1] / ne2 / ne3, ne2, ne3);
    } else {
        x = mul_mat(x);
    }

    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, 1.f / scale);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(x);
    GGML_UNUSED(w);
    GGML_UNUSED(weight_scale);
    GGML_UNUSED(b);
    GGML_UNUSED(convrot_group_size);
    GGML_UNUSED(scale);
    throw std::runtime_error("INT8 tensorwise/convrot is not supported by this ggml build");
#endif
}

ggml_tensor* ggml_ext_pad_ext(GGMLRunnerContext* runner_ctx,
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
    auto ctx     = runner_ctx->ggml_ctx;
    auto backend = runner_ctx->backend;
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

ggml_tensor* ggml_ext_pad(GGMLRunnerContext* runner_ctx,
                          ggml_tensor* x,
                          int p0,
                          int p1,
                          int p2,
                          int p3,
                          bool circular_x,
                          bool circular_y) {
    return ggml_ext_pad_ext(runner_ctx, x, 0, p0, 0, p1, 0, p2, 0, p3, circular_x, circular_y);
}

static ggml_tensor* conv_1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, int s0, int p0, int d0, bool force_prec_f32) {
    if (!force_prec_f32) {
        return ggml_conv_1d(ctx, w, x, s0, p0, d0);
    }
    ggml_tensor* patches = ggml_im2col(ctx, w, x, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);
    ggml_tensor* result  = ggml_mul_mat(ctx,
                                        ggml_reshape_2d(ctx, patches, patches->ne[0], patches->ne[2] * patches->ne[1]),
                                        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    result               = ggml_reshape_3d(ctx, result, patches->ne[1], patches->ne[2], w->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 2, 1, 3));
}

ggml_tensor* ggml_ext_conv_1d(GGMLRunnerContext* runner_ctx,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int s0,
                              int p0,
                              int d0,
                              int64_t groups,
                              bool force_prec_f32) {
    auto ctx = runner_ctx->ggml_ctx;
    GGML_ASSERT(s0 > 0 && p0 >= 0 && d0 > 0 && groups > 0);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[3] == 1 && w->ne[3] == 1);
    GGML_ASSERT(x->ne[1] % groups == 0 && w->ne[2] % groups == 0);
    GGML_ASSERT(w->ne[1] == x->ne[1] / groups);
    GGML_ASSERT(b == nullptr || (b->type == GGML_TYPE_F32 && ggml_is_vector(b) && b->ne[0] == w->ne[2]));

    // im2col requires contiguous time rows; group views must retain the real channel and batch strides.
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    if (force_prec_f32 && w->type != GGML_TYPE_F32) {
        w = ggml_cast(ctx, w, GGML_TYPE_F32);
    }
    if (!ggml_is_contiguous(w)) {
        w = ggml_cont(ctx, w);
    }

    ggml_tensor* result = nullptr;
    if (groups == 1) {
        result = conv_1d(ctx, x, w, s0, p0, d0, force_prec_f32);
    } else {
        const int64_t ic_g = x->ne[1] / groups;
        const int64_t oc_g = w->ne[2] / groups;
        std::vector<ggml_tensor*> outputs;
        outputs.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
            ggml_tensor* x_i = ggml_view_3d(ctx, x, x->ne[0], ic_g, x->ne[2], x->nb[1], x->nb[2], group * ic_g * x->nb[1]);
            ggml_tensor* w_i = ggml_view_3d(ctx, w, w->ne[0], ic_g, oc_g, w->nb[1], w->nb[2], group * oc_g * w->nb[2]);
            outputs.push_back(conv_1d(ctx, x_i, w_i, s0, p0, d0, force_prec_f32));
        }
        result = ggml_ext_vec_concat(runner_ctx, outputs, 1);
    }
    if (b != nullptr) {
        if (!ggml_is_contiguous(b)) {
            b = ggml_cont(ctx, b);
        }
        b      = ggml_reshape_3d(ctx, b, 1, w->ne[2], 1);
        result = ggml_add_inplace(ctx, result, b);
    }
    return result;
}

ggml_tensor* ggml_ext_conv_2d(GGMLRunnerContext* runner_ctx,
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
    auto ctx = runner_ctx->ggml_ctx;
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, scale);
    }
    if (w->ne[2] != x->ne[2] && ggml_n_dims(w) == 2) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], w->ne[1]);
    }

    if ((p0 != 0 || p1 != 0) && (circular_x || circular_y)) {
        x  = ggml_ext_pad_ext(runner_ctx, x, p0, p0, p1, p1, 0, 0, 0, 0, circular_x, circular_y);
        p0 = 0;
        p1 = 0;
    }

    if (direct) {
        x = ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    } else {
        x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_conv_3d(GGMLRunnerContext* runner_ctx,
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
                              bool force_prec_f32,
                              bool direct,
                              float scale,
                              bool allow_conv2d) {
    auto ctx     = runner_ctx->ggml_ctx;
    auto backend = runner_ctx->backend;
    if (allow_conv2d && w->ne[2] == 1 && x->ne[2] == 1 && x->ne[3] == IC && p2 == 0) {
        auto x2 = ggml_reshape_4d(ctx, ggml_ext_cont(runner_ctx, x), x->ne[0], x->ne[1], IC, 1);
        auto w2 = ggml_reshape_4d(ctx, w, w->ne[0], w->ne[1], IC, w->ne[3] / IC);
        x2      = ggml_ext_conv_2d(runner_ctx, x2, w2, b, s0, s1, p0, p1, d0, d1, runner_ctx->conv2d_direct_enabled, false, false, scale);
        return ggml_reshape_4d(ctx, x2, x2->ne[0], x2->ne[1], 1, w->ne[3] / IC);
    }
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, scale);
    }
    if (direct) {
        int64_t OC = w->ne[3] / IC;
        int64_t N  = x->ne[3] / IC;
        x          = ggml_conv_3d_direct(ctx, w, x, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC);
    } else if (force_prec_f32) {
        ggml_tensor* im2col = ggml_im2col_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, w->type);

        int64_t OC = w->ne[3] / IC;
        int64_t N  = x->ne[3] / IC;
        x          = ggml_mul_mat(ctx,
                                  ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
                                  ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2] * IC, OC));
        ggml_prec_set_acc(x, GGML_PREC_F32);

        int64_t OD = im2col->ne[3] / N;
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1] * im2col->ne[2], OD, N, OC);
        x          = ggml_cont(ctx, ggml_permute(ctx, x, 0, 1, 3, 2));
        x          = ggml_reshape_4d(ctx, x, im2col->ne[1], im2col->ne[2], OD, OC * N);
    } else {
        auto convolution = ggml_conv_3d(ctx, w, x, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2);
        if (backend != nullptr) {
            // im2col can support large buffers that its reshape/GEMM consumers cannot.
            std::vector<ggml_tensor*> nodes{convolution};
            bool supported = true;
            for (size_t i = 0; i < nodes.size() && supported; ++i) {
                auto node = nodes[i];
                if (node == x || node == w) {
                    continue;
                }
                supported = ggml_backend_supports_op(backend, node);
                for (auto source : node->src) {
                    if (source != nullptr) {
                        nodes.push_back(source);
                    }
                }
            }
            if (!supported) {
                int64_t OC    = w->ne[3] / IC;
                int64_t N     = x->ne[3] / IC;
                auto fallback = ggml_conv_3d_direct(ctx, w, x, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC);
                if (ggml_backend_supports_op(backend, fallback)) {
                    convolution = fallback;
                }
            }
        }
        x = convolution;
    }

    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, 1.f / scale);
    }
    if (b != nullptr) {
        b = ggml_reshape_4d(ctx, b, 1, 1, 1, b->ne[0]);  // [OC, 1, 1, 1]
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_full(GGMLRunnerContext* runner_ctx,
                           float value,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3) {
    auto ctx = runner_ctx->ggml_ctx;
    auto one = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:one");
    auto t   = ggml_ext_scale(runner_ctx, one, value);      // [1,]
    t        = ggml_repeat_4d(ctx, t, ne0, ne1, ne2, ne3);  // [ne0, ne1, ne2, ne3]
    return t;
}

ggml_tensor* ggml_ext_zeros(GGMLRunnerContext* runner_ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return ggml_ext_full(runner_ctx, 0.f, ne0, ne1, ne2, ne3);
}

ggml_tensor* ggml_ext_zeros_like(GGMLRunnerContext* runner_ctx, ggml_tensor* x) {
    return ggml_ext_zeros(runner_ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

ggml_tensor* ggml_ext_ones(GGMLRunnerContext* runner_ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return ggml_ext_full(runner_ctx, 1.f, ne0, ne1, ne2, ne3);
}

ggml_tensor* ggml_ext_ones_like(GGMLRunnerContext* runner_ctx, ggml_tensor* x) {
    return ggml_ext_ones(runner_ctx, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

ggml_tensor* ggml_ext_cast_f32(GGMLRunnerContext* runner_ctx, ggml_tensor* a) {
    auto ctx     = runner_ctx->ggml_ctx;
    auto backend = runner_ctx->backend;
    if (sd_backend_is(backend, "Vulkan")) {
        auto zero_index = ggml_get_tensor(ctx, "ggml_runner_build_in_tensor:zero_int");
        auto out        = ggml_reshape_1d(ctx, a, ggml_nelements(a));
        out             = ggml_get_rows(ctx, out, zero_index);
        out             = ggml_reshape(ctx, out, a);
        // auto out = ggml_cast(ctx, a, GGML_TYPE_F32);
        return out;
    } else {
        auto out         = ggml_reshape_2d(ctx, a, 1, ggml_nelements(a));
        ggml_tensor* one = ggml_ext_ones(runner_ctx, 1, 1, 1, 1);  // [1,]
        if (ggml_is_transposed(out)) {
            out = ggml_mul_mat(ctx, one, out);
        } else {
            out = ggml_mul_mat(ctx, out, one);
        }
        out = ggml_reshape(ctx, out, a);
        return out;
    }
}

ggml_tensor* ggml_ext_attention_ext(GGMLRunnerContext* runner_ctx,
                                    ggml_tensor* q,
                                    ggml_tensor* k,
                                    ggml_tensor* v,
                                    int64_t n_head,
                                    ggml_tensor* mask,
                                    bool skip_reshape,
                                    bool flash_attn,
                                    float kv_scale,
                                    bool* used_flash_attn) {
    auto ctx     = runner_ctx->ggml_ctx;
    auto backend = runner_ctx->backend;
    if (runner_ctx->attn_scale > 0.f) {
        kv_scale = runner_ctx->attn_scale;
    }
    const bool sage_attn = runner_ctx->sage_attn_enabled;
    if (used_flash_attn != nullptr) {
        *used_flash_attn = false;
    }
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

        q = ggml_reshape_4d(ctx, q, d_head, n_head, L_q, N);              // [N, L_q, n_head, d_head]
        q = ggml_ext_cont(runner_ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [N, n_head, L_q, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, L_q, n_head * N);             // [N * n_head, L_q, d_head]

        k = ggml_reshape_4d(ctx, k, d_head, n_kv_head, L_k, N);           // [N, L_k, n_kv_head, d_head]
        k = ggml_ext_cont(runner_ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [N, n_kv_head, L_k, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, L_k, n_kv_head * N);          // [N * n_kv_head, L_k, d_head]

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
        const bool pad_head = d_head > 0 && d_head < 64 && q_in->ne[0] == d_head && k_in->ne[0] == d_head &&
                              q_in->type == GGML_TYPE_F32 && k_in->type == GGML_TYPE_F32 &&
                              v_in->type == GGML_TYPE_F32 && sd_backend_supports_cuda_mma(backend);
        if (pad_head) {
            // CUDA FA MMA starts at 64 channels; keep the original head's attention scale.
            q_in = ggml_pad(ctx, q_in, static_cast<int>(64 - d_head), 0, 0, 0);
            k_in = ggml_pad(ctx, k_in, static_cast<int>(64 - d_head), 0, 0, 0);
            v_in = ggml_pad(ctx, v_in, static_cast<int>(64 - d_head), 0, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(runner_ctx, k_in, kv_scale);
        }
        k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16);

        if (kv_scale != 1.0f) {
            v_in = ggml_ext_scale(runner_ctx, v_in, kv_scale);
        }
        v_in       = ggml_permute(ctx, v_in, 0, 2, 1, 3);
        auto v_f16 = ggml_cast(ctx, v_in, GGML_TYPE_F16);
        if (!ggml_backend_supports_op(backend, v_f16)) {
            v_f16 = ggml_cast(ctx, ggml_ext_cont(runner_ctx, v_in), GGML_TYPE_F16);
        }
        v_in = v_f16;
        v_in = ggml_reshape_3d(ctx, v_in, v_in->ne[0], L_k, n_kv_head * N);

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
        ggml_prec_set_acc(out, GGML_PREC_F32);
        if (kv_scale != 1.0f) {
            out = ggml_ext_scale(runner_ctx, out, 1.0f / kv_scale);
        }
        if (pad_head) {
            out = ggml_ext_slice(runner_ctx, out, 0, 0, d_head);
        }
        return out;
    };

#ifndef SD_USE_UPSTREAM_GGML
    if (sage_attn && mask == nullptr && d_head > 0 && d_head <= 128) {
        auto q_in                 = ggml_reshape_4d(ctx, ggml_ext_cont(runner_ctx, q->type == GGML_TYPE_F32 ? q : ggml_cast(ctx, q, GGML_TYPE_F32)), d_head, L_q, n_head, N);
        auto k_in                 = ggml_reshape_4d(ctx, ggml_ext_cont(runner_ctx, k->type == GGML_TYPE_F32 ? k : ggml_cast(ctx, k, GGML_TYPE_F32)), d_head, L_k, n_kv_head, N);
        auto v_in                 = ggml_ext_cont(runner_ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        const int64_t padded_head = d_head <= 64 ? 64 : 128;
        if ((padded_head != d_head || kv_scale != 1.0f) && v_in->type != GGML_TYPE_F32) {
            v_in = ggml_cast(ctx, v_in, GGML_TYPE_F32);
        }
        if (padded_head != d_head) {
            // Keep the original head's softmax scale when padding for the CUDA kernel.
            q_in = ggml_pad(ctx, q_in, static_cast<int>(padded_head - d_head), 0, 0, 0);
            k_in = ggml_pad(ctx, k_in, static_cast<int>(padded_head - d_head), 0, 0, 0);
            v_in = ggml_pad(ctx, v_in, static_cast<int>(padded_head - d_head), 0, 0, 0);
        }
        if (kv_scale != 1.0f) {
            k_in = ggml_ext_scale(runner_ctx, k_in, kv_scale);
            v_in = ggml_ext_scale(runner_ctx, v_in, kv_scale);
        }
        v_in     = ggml_cast(ctx, v_in, GGML_TYPE_F16);
        auto out = ggml_ext_sage_attn(runner_ctx, q_in, k_in, v_in, scale / kv_scale, GGML_SAGE_ATTN_AUTO);
        if (ggml_backend_supports_op(backend, out)) {
            kqv = kv_scale != 1.0f ? ggml_ext_scale(runner_ctx, out, 1.0f / kv_scale) : out;
            if (padded_head != d_head) {
                kqv = ggml_ext_slice(runner_ctx, kqv, 0, 0, d_head);
            }
        }
    }
#endif

    if (kqv == nullptr && flash_attn) {
        // LOG_VERBOSE("attention_ext L_q:%d L_k:%d n_head:%d C:%d d_head:%d N:%d", L_q, L_k, n_head, C, d_head, N);
        bool can_use_flash_attn = true;
        if (mask != nullptr) {
            // TODO: figure out if we can bend t5 to work too
            can_use_flash_attn = can_use_flash_attn && mask->ne[3] == 1;
        }

        if (can_use_flash_attn) {
            kqv = build_kqv(q, k, v, mask);
            if (kqv != nullptr) {
                if (used_flash_attn != nullptr) {
                    *used_flash_attn = true;
                }
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
        v = ggml_ext_cont(runner_ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [N, n_kv_head, d_head, L_k]
        v = ggml_reshape_3d(ctx, v, L_k, d_head, n_kv_head * N);          // [N * n_kv_head, d_head, L_k]

        auto kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, L_q, L_k]
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        kq = ggml_scale_inplace(ctx, kq, scale);
        if (mask) {
            kq = ggml_add_inplace(ctx, kq, mask);
        }
        kq = ggml_soft_max_inplace(ctx, kq);

        kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, L_q, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, L_q, n_head, N);  // [N, n_head, L_q, d_head]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                 // [N, L_q, n_head, d_head]
    }

    kqv = ggml_ext_cont(runner_ctx, kqv);
    kqv = ggml_reshape_3d(ctx, kqv, d_head * n_head, L_q, N);  // [N, L_q, C]

    return kqv;
}

ggml_tensor* ggml_ext_layer_norm(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 float eps) {
    auto ctx = runner_ctx->ggml_ctx;
    x        = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        if (b != nullptr) {
            x = ggml_add_inplace(ctx, x, b);
        }
    }
    return x;
}

ggml_tensor* ggml_ext_group_norm(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 int num_groups,
                                 float eps) {
    auto ctx = runner_ctx->ggml_ctx;
    if (ggml_n_dims(x) >= 3 && w != nullptr && b != nullptr) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups, eps);
    if (w != nullptr && b != nullptr) {
        x = ggml_mul_inplace(ctx, x, w);
        x = ggml_add_inplace(ctx, x, b);
    }
    return x;
}

ggml_tensor* ggml_ext_timestep_embedding(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* timesteps,
                                         int dim,
                                         int max_period,
                                         float time_factor) {
    auto ctx  = runner_ctx->ggml_ctx;
    timesteps = ggml_ext_scale(runner_ctx, timesteps, time_factor);
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}

ggml_tensor* ggml_ext_vec_concat(GGMLRunnerContext* runner_ctx, std::vector<ggml_tensor*>& tensors, int dim) {
    auto ctx = runner_ctx->ggml_ctx;
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

#ifndef SD_USE_UPSTREAM_GGML
ggml_tensor* ggml_ext_mul_mat_i8_tensorwise(GGMLRunnerContext* runner_ctx,
                                            ggml_tensor* w,
                                            ggml_tensor* x,
                                            ggml_tensor* ws,
                                            ggml_tensor* bias,
                                            int group,
                                            ggml_tensor* xs) {
    auto ctx = runner_ctx->ggml_ctx;
    if (!w || !x)
        return nullptr;
    const auto* logical = x;
    if (x->type == GGML_TYPE_I8 && !xs) {
        int packed_group;
        if (!sd_ggml_convrot_describe(x, &packed_group) || packed_group != group)
            return nullptr;
        logical = x->src[0];
    }
    SDI8Attributes attr    = {{logical->ne[1], logical->ne[2], logical->ne[3]}, group, logical != x ? SD_I8_PACKED : 0};
    ggml_tensor* inputs[5] = {w, x};
    int count              = 2;
    if (xs) {
        attr.flags |= SD_I8_INPUT_SCALE;
        inputs[count++] = xs;
    }
    if (ws) {
        attr.flags |= SD_I8_WEIGHT_SCALE;
        inputs[count++] = ws;
    }
    if (bias) {
        attr.flags |= SD_I8_BIAS;
        inputs[count++] = bias;
    }
    return ggml_ext_op_create(ctx, sd_ggml_i8_op(), inputs, count, &attr);
}

ggml_tensor* ggml_ext_quantize_i8_convrot(GGMLRunnerContext* runner_ctx, ggml_tensor* x, int group) {
    auto ctx       = runner_ctx->ggml_ctx;
    const auto key = std::make_pair(x, group);
    auto cached    = runner_ctx->int8_convrot_cache.find(key);
    if (cached != runner_ctx->int8_convrot_cache.end()) {
        return cached->second;
    }
    int32_t attr = group;
    auto result  = ggml_ext_op_create(ctx, sd_ggml_convrot_op(), &x, 1, &attr);
    if (result) {
        runner_ctx->int8_convrot_cache.emplace(key, result);
    }
    return result;
}

ggml_tensor* ggml_ext_mul_mat_w4_convrot(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* w,
                                         ggml_tensor* x,
                                         ggml_tensor* ws,
                                         ggml_tensor* sc,
                                         ggml_tensor* sr,
                                         ggml_tensor* bias,
                                         SDW4Kind kind,
                                         int group) {
    auto ctx = runner_ctx->ggml_ctx;
    if (!w || !x || group < 0 || (group && !sd_ggml_valid_convrot_group(group)))
        return nullptr;
    const int quant_group = group ? group : 1;
    if (x->type == GGML_TYPE_F32)
        x = ggml_ext_quantize_i8_convrot(runner_ctx, x, quant_group);
    int packed_group;
    if (!sd_ggml_convrot_describe(x, &packed_group) || packed_group != quant_group)
        return nullptr;
    const auto* logical    = x->src[0];
    SDW4Attributes attr    = {{logical->ne[1], logical->ne[2], logical->ne[3]}, kind, bias ? 1 : 0};
    ggml_tensor* inputs[6] = {w, x, ws};
    int count              = 3;
    if (kind == SDW4Kind::W4A8) {
        inputs[count++] = sc;
        inputs[count++] = sr;
    } else if (sc || sr) {
        return nullptr;
    }
    if (bias)
        inputs[count++] = bias;
    return ggml_ext_op_create(ctx, sd_ggml_w4_op(), inputs, count, &attr);
}

ggml_tensor* ggml_ext_sage_attn(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* q,
                                ggml_tensor* k,
                                ggml_tensor* v,
                                float scale,
                                ggml_sage_attn_mode mode) {
    auto ctx                           = runner_ctx->ggml_ctx;
    ggml_tensor* inputs[]              = {q, k, v};
    const SDSageAttentionParams params = {scale, mode};
    return ggml_ext_op_create(ctx, sd_ggml_sage_attn_op(), inputs, 3, &params);
}

ggml_tensor* ggml_ext_sol_attn(GGMLRunnerContext* runner_ctx,
                               ggml_tensor* q,
                               ggml_tensor* k,
                               ggml_tensor* v,
                               float scale,
                               float tau) {
    auto ctx                          = runner_ctx->ggml_ctx;
    ggml_tensor* inputs[]             = {q, k, v};
    const SDSolAttentionParams params = {scale, tau};
    return ggml_ext_op_create(ctx, sd_ggml_sol_attn_op(), inputs, 3, &params);
}

ggml_tensor* ggml_ext_window_attn(GGMLRunnerContext* runner_ctx,
                                  ggml_tensor* q,
                                  ggml_tensor* k,
                                  ggml_tensor* v,
                                  const SDWindowAttentionParams& params) {
    auto ctx              = runner_ctx->ggml_ctx;
    auto backend          = runner_ctx->backend;
    ggml_tensor* inputs[] = {q, k, v};
    ggml_ext_tensor_desc out;
    if (!backend || !sd_ggml_window_attn_infer(inputs, 3, &params, &out))
        return nullptr;
    sd_ggml_register_extensions(backend);
    auto result = ggml_ext_op_create(ctx, sd_ggml_window_attn_op(), inputs, 3, &params);
    return ggml_backend_supports_op(backend, result) ? result : nullptr;
}

#endif

ggml_tensor* ggml_ext_rope_apply(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* pe,
                                 bool rope_interleaved) {
    auto ctx              = runner_ctx->ggml_ctx;
    const int64_t rot_dim = pe->ne[2] * 2;
    GGML_ASSERT(rot_dim > 0 && rot_dim <= x->ne[0]);
#ifndef SD_USE_UPSTREAM_GGML
    auto backend = runner_ctx->backend;
    if (backend != nullptr && x->type == GGML_TYPE_F32 && pe->type == GGML_TYPE_F32 &&
        x->ne[0] % 2 == 0 && pe->ne[0] == 2 && pe->ne[1] == 2 &&
        pe->ne[2] <= x->ne[0] / 2 && pe->ne[3] == x->ne[2]) {
        sd_ggml_register_extensions(backend);
        ggml_tensor* inputs[] = {x, pe};
        int32_t interleaved   = rope_interleaved ? 1 : 0;
        auto out              = ggml_ext_op_create(ctx, sd_ggml_rope_op(), inputs, 2, &interleaved);
        if (out && ggml_backend_supports_op(backend, out)) {
            return out;
        }
    }
#endif
    if (rot_dim < x->ne[0]) {
        auto rotated = ggml_ext_rope_apply(runner_ctx, ggml_ext_slice(runner_ctx, x, 0, 0, rot_dim), pe, rope_interleaved);
        auto tail    = ggml_ext_slice(runner_ctx, x, 0, rot_dim, x->ne[0]);
        tail         = ggml_cont(ctx, ggml_permute(ctx, tail, 0, 2, 1, 3));
        tail         = ggml_reshape_3d(ctx, tail, tail->ne[0], tail->ne[1], tail->ne[2] * tail->ne[3]);
        return ggml_concat(ctx, rotated, tail, 0);
    }
    // x: [N, L, n_head, d_head]
    // pe: [L, d_head/2, 2, 2], [[cos, -sin], [sin, cos]]
    int64_t d_head = x->ne[0];
    int64_t n_head = x->ne[1];
    int64_t L      = x->ne[2];
    int64_t N      = x->ne[3];
    x              = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, n_head, L, d_head]
    if (rope_interleaved) {
        x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);  // [N * n_head, L, d_head/2, 2]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));       // [2, N * n_head, L, d_head/2]
    } else {
        auto x_0 = ggml_view_3d(ctx, x, d_head / 2, L, n_head * N, x->nb[1], x->nb[2], 0);
        auto x_1 = ggml_view_3d(ctx, x, d_head / 2, L, n_head * N, x->nb[1], x->nb[2], (d_head / 2) * x->nb[0]);

        auto pe_2 = ggml_cont(ctx, ggml_permute(ctx, pe, 2, 3, 0, 1));
        auto cos  = ggml_view_2d(ctx, pe_2, d_head / 2, L, pe_2->nb[1], 0);
        auto sin  = ggml_view_2d(ctx, pe_2, d_head / 2, L, pe_2->nb[1], pe_2->nb[3]);

        auto part_0 = ggml_sub(ctx, ggml_mul(ctx, x_0, cos), ggml_mul(ctx, x_1, sin));
        auto part_1 = ggml_add(ctx, ggml_mul(ctx, x_0, sin), ggml_mul(ctx, x_1, cos));
        return ggml_concat(ctx, part_0, part_1, 0);
    }

    int64_t offset = x->nb[2] * x->ne[2];
    auto x_0       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);  // [N * n_head, L, d_head/2]
    auto x_1       = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);  // [N * n_head, L, d_head/2]
    x_0            = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);                    // [N * n_head, L, d_head/2, 1]
    x_1            = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);                    // [N * n_head, L, d_head/2, 1]
    auto temp_x    = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
    x_0            = ggml_repeat(ctx, x_0, temp_x);  // [N * n_head, L, d_head/2, 2]
    x_1            = ggml_repeat(ctx, x_1, temp_x);  // [N * n_head, L, d_head/2, 2]

    pe        = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));  // [2, L, d_head/2, 2]
    offset    = pe->nb[2] * pe->ne[2];
    auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);  // [L, d_head/2, 2]
    auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);  // [L, d_head/2, 2]

    auto x_out = ggml_add_inplace(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));  // [N * n_head, L, d_head/2, 2]
    x_out      = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);                         // [N*n_head, L, d_head]
    return x_out;
}

ggml_tensor* ggml_ext_repeat_4d(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* x,
                                int64_t ne0,
                                int64_t ne1,
                                int64_t ne2,
                                int64_t ne3) {
    auto ctx = runner_ctx->ggml_ctx;
    if ((x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) && sd_backend_is(runner_ctx->backend, "Vulkan")) {
        auto repeated = ggml_repeat_4d(ctx, ggml_cast(ctx, x, GGML_TYPE_F32), ne0, ne1, ne2, ne3);
        return ggml_cast(ctx, repeated, x->type);
    }
    return ggml_repeat_4d(ctx, x, ne0, ne1, ne2, ne3);
}

ggml_tensor* ggml_ext_repeat(GGMLRunnerContext* runner_ctx, ggml_tensor* x, ggml_tensor* shape) {
    return ggml_ext_repeat_4d(runner_ctx, x, shape->ne[0], shape->ne[1], shape->ne[2], shape->ne[3]);
}

bool ggml_ext_force_prec_f32(GGMLRunnerContext* runner_ctx,
                             bool requested,
                             uint32_t flags) {
    return requested ||
           ((flags & GGML_PRECISION_F32_VULKAN) != 0 && sd_backend_is(runner_ctx->backend, "Vulkan")) ||
           ((flags & GGML_PRECISION_F32_ROCM) != 0 && sd_backend_is(runner_ctx->backend, "ROCm"));
}

static void ggml_ext_get_rows_i8(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const int group_size      = static_cast<int>(reinterpret_cast<intptr_t>(userdata));
    const auto* weight        = dst->src[0];
    const auto* input_ids     = static_cast<const int32_t*>(dst->src[1]->data);
    const auto* scales        = static_cast<const float*>(dst->src[2]->data);
    const bool scalar_scale   = ggml_nelements(dst->src[2]) == 1;
    const float normalization = group_size > 0 ? 1.f / std::sqrt(static_cast<float>(group_size)) : 1.f;
    for (int64_t row = ith; row < dst->ne[1]; row += nth) {
        const int32_t token = input_ids[row];
        GGML_ASSERT(token >= 0 && token < weight->ne[1]);
        const auto* src   = reinterpret_cast<const int8_t*>(static_cast<const char*>(weight->data) + token * weight->nb[1]);
        auto* out         = reinterpret_cast<float*>(static_cast<char*>(dst->data) + row * dst->nb[1]);
        const float scale = scales[scalar_scale ? 0 : token] * normalization;
        for (int64_t i = 0; i < weight->ne[0]; ++i) {
            out[i] = static_cast<float>(src[i]) * scale;
        }
        // The regular Hadamard rotation is symmetric and its own inverse.
        for (int stride = 1; stride < group_size; stride *= 4) {
            for (int64_t base = 0; base < weight->ne[0]; base += 4 * stride) {
                for (int j = 0; j < stride; ++j) {
                    float* values      = out + base + j;
                    const float a      = values[0];
                    const float b      = values[stride];
                    const float c      = values[2 * stride];
                    const float d      = values[3 * stride];
                    values[0]          = a + b + c - d;
                    values[stride]     = a + b - c + d;
                    values[2 * stride] = a - b + c + d;
                    values[3 * stride] = -a + b + c + d;
                }
            }
        }
    }
}

ggml_tensor* ggml_ext_embedding(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* weight,
                                ggml_tensor* input_ids,
                                ggml_tensor* weight_scale,
                                int group_size) {
    // input_ids: [N, n_token]

    // There are issues with ggml batch inference, so we are expanding it here first.
    // TODO: fix ggml batch inference
    int64_t n = input_ids->ne[1];
    input_ids = ggml_reshape_1d(runner_ctx->ggml_ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

    ggml_tensor* embedding;
    if (weight->type == GGML_TYPE_I8) {
        GGML_ASSERT(input_ids->type == GGML_TYPE_I32);
        ggml_tensor* args[] = {weight, input_ids, weight_scale};
        embedding           = ggml_custom_4d(runner_ctx->ggml_ctx, GGML_TYPE_F32, weight->ne[0], input_ids->ne[0], 1, 1,
                                             args, 3, ggml_ext_get_rows_i8, GGML_N_TASKS_MAX, reinterpret_cast<void*>(static_cast<intptr_t>(group_size)));
    } else {
        input_ids = ggml_reshape_3d(runner_ctx->ggml_ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        embedding = ggml_get_rows(runner_ctx->ggml_ctx, weight, input_ids);
    }
    embedding = ggml_reshape_3d(runner_ctx->ggml_ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

    // [N, n_token, weight->ne[0]]
    return embedding;
}

ggml_tensor* ggml_ext_prepare_linear_weight(GGMLRunnerContext* runner_ctx,
                                            ggml_tensor* w,
                                            ggml_tensor* x,
                                            bool force_prec_f32) {
#ifndef SD_USE_UPSTREAM_GGML
    if (w->type == GGML_TYPE_F8_E4M3 || w->type == GGML_TYPE_F8_E5M2) {
        bool supports_fp8_matmul = false;
        if (runner_ctx->backend != nullptr) {
            ggml_tensor* fp8_matmul = ggml_mul_mat(runner_ctx->ggml_ctx, w, x);
            if (force_prec_f32) {
                ggml_prec_set_acc(fp8_matmul, GGML_PREC_F32);
            }
            supports_fp8_matmul = ggml_backend_supports_op(runner_ctx->backend, fp8_matmul) && !sd_backend_is_cpu(runner_ctx->backend);
        }
        if (!supports_fp8_matmul) {
            w = ggml_cast(runner_ctx->ggml_ctx, w, GGML_TYPE_BF16);
        }
    }
#endif
    return w;
}

ggml_tensor* ggml_ext_prepare_i8_input(GGMLRunnerContext* runner_ctx, ggml_tensor* x) {
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_I8) {
        x = ggml_ext_cast_f32(runner_ctx, x);
    }
    return ggml_ext_cont(runner_ctx, x);
}

#ifndef SD_USE_UPSTREAM_GGML
ggml_tensor* ggml_ext_linear_w4_convrot(GGMLRunnerContext* runner_ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* w,
                                        ggml_tensor* ws,
                                        ggml_tensor* sc,
                                        ggml_tensor* sr,
                                        ggml_tensor* bias,
                                        SDW4Kind kind,
                                        int group,
                                        float scale) {
    auto ctx = runner_ctx->ggml_ctx;
    x        = ggml_ext_prepare_i8_input(runner_ctx, x);
    if (scale != 1.f) {
        x = ggml_ext_scale(runner_ctx, x, scale);
    }
    auto out = ggml_ext_mul_mat_w4_convrot(runner_ctx, w, x, ws, sc, sr, scale == 1.f ? bias : nullptr, kind, group);
    GGML_ASSERT(out);
    if (scale != 1.f) {
        out = ggml_ext_scale(runner_ctx, out, 1.f / scale);
        if (bias) {
            out = ggml_add_inplace(ctx, out, bias);
        }
    }
    return out;
}
#endif

ggml_tensor* ggml_ext_conv_2d_grouped(GGMLRunnerContext* runner_ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* w,
                                      ggml_tensor* b,
                                      int groups,
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
    auto ctx = runner_ctx->ggml_ctx;
    GGML_ASSERT(groups > 0 && x->ne[2] % groups == 0 && w->ne[3] % groups == 0);
    if (groups == 1) {
        return ggml_ext_conv_2d(runner_ctx, x, w, b,
                                s0, s1, p0, p1, d0, d1,
                                direct, circular_x, circular_y, scale);
    }
    if (groups == x->ne[2] && groups == w->ne[3]) {
        auto out = direct ? ggml_conv_2d_dw_direct(ctx, w, x, s0, s1, p0, p1, d0, d1)
                          : ggml_conv_2d_dw(ctx, w, x, s0, s1, p0, p1, d0, d1);
        if (b) {
            out = ggml_add_inplace(ctx, out, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
        }
        return out;
    }
    const int64_t ic_g = x->ne[2] / groups, oc_g = w->ne[3] / groups;
    std::vector<ggml_tensor*> outputs;
    for (int i = 0; i < groups; ++i) {
        auto xi = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], ic_g, x->ne[3], x->nb[1], x->nb[2], x->nb[3], i * ic_g * x->nb[2]);
        auto wi = ggml_view_4d(ctx, w, w->ne[0], w->ne[1], w->ne[2], oc_g, w->nb[1], w->nb[2], w->nb[3], i * oc_g * w->nb[3]);
        auto bi = b ? ggml_view_1d(ctx, b, oc_g, i * oc_g * b->nb[0]) : nullptr;
        outputs.push_back(ggml_ext_conv_2d(runner_ctx, xi, wi, bi,
                                           s0, s1, p0, p1, d0, d1,
                                           direct, circular_x, circular_y, scale));
    }
    return ggml_ext_vec_concat(runner_ctx, outputs, 2);
}

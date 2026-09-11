#include "model/adapter/lora_ops.h"

#include <cmath>

#include "core/ggml_extend.h"
#include "core/ggml_extend_backend.h"

ggml_tensor* ggml_ext_merge_lora(ggml_context* ctx,
                                 ggml_tensor* lora_down,
                                 ggml_tensor* lora_up,
                                 ggml_tensor* lora_mid) {
    ggml_tensor* updown;
    // flat lora tensors to multiply it
    int64_t lora_up_rows  = lora_up->ne[ggml_n_dims(lora_up) - 1];
    lora_up               = ggml_reshape_2d(ctx, lora_up, ggml_nelements(lora_up) / lora_up_rows, lora_up_rows);
    auto lora_down_n_dims = ggml_n_dims(lora_down);
    // assume n_dims should always be a multiple of 2 (otherwise rank 1 doesn't work)
    lora_down_n_dims       = (lora_down_n_dims + lora_down_n_dims % 2);
    int64_t lora_down_rows = lora_down->ne[lora_down_n_dims - 1];
    lora_down              = ggml_reshape_2d(ctx, lora_down, ggml_nelements(lora_down) / lora_down_rows, lora_down_rows);

    // ggml_mul_mat requires tensor b transposed
    lora_down = ggml_cont(ctx, ggml_transpose(ctx, lora_down));
    if (lora_mid == nullptr) {
        updown = ggml_mul_mat(ctx, lora_up, lora_down);
        updown = ggml_cont(ctx, ggml_transpose(ctx, updown));
    } else {
        // undoing tucker decomposition for conv layers.
        // lora_mid  has shape (3,    3,   Rank, Rank)
        // lora_down has shape (Rank, In,  1,    1)
        // lora_up   has shape (Rank, Out, 1,    1)
        // conv layer shape is (3,    3,   Out,  In)
        updown = ggml_ext_mul_n_mode(ctx, ggml_ext_mul_n_mode(ctx, lora_mid, lora_down, 3), lora_up, 2);
        updown = ggml_cont(ctx, updown);
    }
    return updown;
}

ggml_tensor* ggml_ext_lokr_forward(
    ggml_context* ctx,
    ggml_backend_t backend,
    ggml_tensor* h,    // Input: [q, batch] or [W, H, q, batch]
    ggml_tensor* w1,   // Outer C (Full rank)
    ggml_tensor* w1a,  // Outer A (Low rank part 1)
    ggml_tensor* w1b,  // Outer B (Low rank part 2)
    ggml_tensor* w2,   // Inner BA (Full rank)
    ggml_tensor* w2a,  // Inner A (Low rank part 1)
    ggml_tensor* w2b,  // Inner B (Low rank part 2)
    bool is_conv,
    WeightAdapter::ForwardParams::conv2d_params_t conv_params,
    float scale) {
    GGML_ASSERT((w1 != nullptr || (w1a != nullptr && w1b != nullptr)));
    GGML_ASSERT((w2 != nullptr || (w2a != nullptr && w2b != nullptr)));

    int uq = (w1 != nullptr) ? (int)w1->ne[0] : (int)w1a->ne[0];
    int up = (w1 != nullptr) ? (int)w1->ne[1] : (int)w1b->ne[1];

    int q_actual = is_conv ? (int)h->ne[2] : (int)h->ne[0];
    int vq       = q_actual / uq;

    int vp = (w2 != nullptr) ? (is_conv ? (int)w2->ne[3] : (int)w2->ne[1])
                             : (int)w2a->ne[1];
    GGML_ASSERT(q_actual == (uq * vq) && "Input dimension mismatch for LoKR split");

    ggml_tensor* hb;

    if (!is_conv) {
        int batch          = (int)h->ne[1];
        int merge_batch_uq = batch;
        int merge_batch_vp = batch;

        if (sd_backend_is(backend, "Vulkan")) {
            if (batch > 1) {
                // no access to backend here, worst case is slightly worse perfs for other backends when built alongside Vulkan backend
                int max_batch    = 65535;
                int max_batch_uq = max_batch / uq;
                merge_batch_uq   = 1;
                for (int i = max_batch_uq; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_uq = i;
                        break;
                    }
                }

                int max_batch_vp = max_batch / vp;
                merge_batch_vp   = 1;
                for (int i = max_batch_vp; i > 0; i--) {
                    if (batch % i == 0) {
                        merge_batch_vp = i;
                        break;
                    }
                }
            }
        }

        ggml_tensor* h_split = ggml_reshape_3d(ctx, h, vq, uq * merge_batch_uq, batch / merge_batch_uq);
        if (w2 != nullptr) {
            hb = ggml_mul_mat(ctx, w2, h_split);
        } else {
            hb = ggml_mul_mat(ctx, w2b, ggml_mul_mat(ctx, w2a, h_split));
        }

        if (batch > 1) {
            hb = ggml_reshape_3d(ctx, hb, vp, uq, batch);
        }
        ggml_tensor* hb_t = ggml_cont(ctx, ggml_transpose(ctx, hb));
        hb_t              = ggml_reshape_3d(ctx, hb_t, uq, vp * merge_batch_vp, batch / merge_batch_vp);

        ggml_tensor* hc_t;
        if (w1 != nullptr) {
            hc_t = ggml_mul_mat(ctx, w1, hb_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_t));
        }

        if (batch > 1) {
            hc_t = ggml_reshape_3d(ctx, hc_t, up, vp, batch);
        }

        ggml_tensor* hc  = ggml_transpose(ctx, hc_t);
        ggml_tensor* out = ggml_reshape_2d(ctx, ggml_cont(ctx, hc), up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    } else {
        int batch = (int)h->ne[3];
        // 1. Reshape input: [W, H, vq*uq, batch] -> [W, H, vq, uq * batch]
        ggml_tensor* h_split = ggml_reshape_4d(ctx, h, h->ne[0], h->ne[1], vq, uq * batch);

        if (w2 != nullptr) {
            hb = ggml_ext_conv_2d(ctx, h_split, w2, nullptr,
                                  conv_params.s0,
                                  conv_params.s1,
                                  conv_params.p0,
                                  conv_params.p1,
                                  conv_params.d0,
                                  conv_params.d1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        } else {
            // swap a and b order for conv lora
            ggml_tensor* a = w2b;
            ggml_tensor* b = w2a;

            // unpack conv2d weights if needed
            if (ggml_n_dims(a) < 4) {
                int k = (int)sqrt(a->ne[0] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[0]);
                a = ggml_reshape_4d(ctx, a, k, k, a->ne[0] / (k * k), a->ne[1]);
            } else if (a->ne[2] != h_split->ne[2]) {
                int k = (int)sqrt(a->ne[2] / h_split->ne[2]);
                GGML_ASSERT(k * k * h_split->ne[2] == a->ne[2]);
                a = ggml_reshape_4d(ctx, a, a->ne[0] * k, a->ne[1] * k, a->ne[2] / (k * k), a->ne[3]);
            }
            ggml_tensor* ha = ggml_ext_conv_2d(ctx, h_split, a, nullptr,
                                               conv_params.s0,
                                               conv_params.s1,
                                               conv_params.p0,
                                               conv_params.p1,
                                               conv_params.d0,
                                               conv_params.d1,
                                               conv_params.direct,
                                               conv_params.circular_x,
                                               conv_params.circular_y,
                                               conv_params.scale);

            // not supporting lora_mid here
            hb = ggml_ext_conv_2d(ctx,
                                  ha,
                                  b,
                                  nullptr,
                                  1,
                                  1,
                                  0,
                                  0,
                                  1,
                                  1,
                                  conv_params.direct,
                                  conv_params.circular_x,
                                  conv_params.circular_y,
                                  conv_params.scale);
        }

        // Current hb shape: [W_out, H_out, vp, uq * batch]
        int w_out = (int)hb->ne[0];
        int h_out = (int)hb->ne[1];

        // ggml_tensor* hb_cat = ggml_reshape_4d(ctx, hb, w_out , h_out , vp * uq, batch);
        // [W_out, H_out, vp * uq,  batch]
        // Now left to compute (W1 kr Id) * hb_cat == (W1 kr W2) cv h

        // merge the uq groups of size vp*w_out*h_out
        ggml_tensor* hb_merged = ggml_reshape_2d(ctx, hb, w_out * h_out * vp, uq * batch);
        ggml_tensor* hc_t;
        ggml_tensor* hb_merged_t = ggml_cont(ctx, ggml_transpose(ctx, hb_merged));
        if (w1 != nullptr) {
            // Would be great to be able to transpose w1 instead to avoid transposing both hb and hc
            hc_t = ggml_mul_mat(ctx, w1, hb_merged_t);
        } else {
            hc_t = ggml_mul_mat(ctx, w1b, ggml_mul_mat(ctx, w1a, hb_merged_t));
        }
        ggml_tensor* hc = ggml_transpose(ctx, hc_t);
        // ungroup
        ggml_tensor* out = ggml_reshape_4d(ctx, ggml_cont(ctx, hc), w_out, h_out, up * vp, batch);
        return ggml_ext_scale(ctx, out, scale);
    }
}

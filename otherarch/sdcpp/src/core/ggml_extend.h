#ifndef __SD_CORE_GGML_EXTEND_H__
#define __SD_CORE_GGML_EXTEND_H__

#include <cstdint>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#define EPS 1e-05f

static_assert(GGML_MAX_NAME >= 160, "GGML_MAX_NAME must be at least 160");

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
ggml_tensor* ggml_ext_mul_n_mode(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int mode = 0);

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
ggml_tensor* ggml_ext_kronecker(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_ext_cont(ggml_context* ctx,
                           ggml_tensor* x);

// torch like permute
ggml_tensor* ggml_ext_torch_permute(ggml_context* ctx,
                                    ggml_tensor* x,
                                    int axis0,
                                    int axis1,
                                    int axis2,
                                    int axis3);

ggml_tensor* ggml_ext_slice(ggml_context* ctx,
                            ggml_tensor* x,
                            int dim,
                            int64_t start,
                            int64_t end,
                            bool cont = true);

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
std::vector<ggml_tensor*> ggml_ext_chunk(ggml_context* ctx,
                                         ggml_tensor* x,
                                         int num,
                                         int64_t dim,
                                         bool cont = true);

ggml_tensor* ggml_ext_silu_act(ggml_context* ctx, ggml_tensor* x, bool gate_first = true);

ggml_tensor* ggml_ext_group_norm_32(ggml_context* ctx,
                                    ggml_tensor* a);

ggml_tensor* ggml_ext_scale(ggml_context* ctx,
                            ggml_tensor* x,
                            float factor,
                            bool inplace = false);

ggml_tensor* ggml_ext_gelu(ggml_context* ctx,
                           ggml_tensor* x,
                           bool inplace = false);

ggml_tensor* ggml_ext_gelu_quick(ggml_context* ctx,
                                 ggml_tensor* x,
                                 bool inplace = false);

ggml_tensor* ggml_ext_linear(ggml_context* ctx,
                             ggml_tensor* x,
                             ggml_tensor* w,
                             ggml_tensor* b,
                             bool force_prec_f32 = false,
                             float scale         = 1.f);

ggml_tensor* ggml_ext_linear_i8_tensorwise(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* weight_scale,
                                           ggml_tensor* b,
                                           int convrot_group_size,
                                           float scale = 1.f);

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
                              bool circular_x = false,
                              bool circular_y = false);

ggml_tensor* ggml_ext_pad(ggml_context* ctx,
                          ggml_tensor* x,
                          int p0,
                          int p1,
                          int p2          = 0,
                          int p3          = 0,
                          bool circular_x = false,
                          bool circular_y = false);

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
ggml_tensor* ggml_ext_conv_2d(ggml_context* ctx,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int s0          = 1,
                              int s1          = 1,
                              int p0          = 0,
                              int p1          = 0,
                              int d0          = 1,
                              int d1          = 1,
                              bool direct     = false,
                              bool circular_x = false,
                              bool circular_y = false,
                              float scale     = 1.f);

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N*OC, OD, OH, OW]
ggml_tensor* ggml_ext_conv_3d(ggml_context* ctx,
                              ggml_backend_t backend,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int64_t IC,
                              int s0              = 1,
                              int s1              = 1,
                              int s2              = 1,
                              int p0              = 0,
                              int p1              = 0,
                              int p2              = 0,
                              int d0              = 1,
                              int d1              = 1,
                              int d2              = 1,
                              bool force_prec_f32 = false);

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
ggml_tensor* ggml_ext_conv_3d_nx1x1(ggml_context* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* w,
                                    ggml_tensor* b,
                                    int s2 = 1,
                                    int p2 = 1,
                                    int d2 = 1);

// qkv: [N, L, 3*C]
// return: ([N, L, C], [N, L, C], [N, L, C])
std::vector<ggml_tensor*> split_qkv(ggml_context* ctx,
                                    ggml_tensor* qkv);

// qkv: [N, 3*C, H, W]
// return: ([N, C, H, W], [N, C, H, W], [N, C, H, W])
std::vector<ggml_tensor*> split_image_qkv(ggml_context* ctx,
                                          ggml_tensor* qkv);

// Constant and cast helpers require the built-in tensors initialized by GGMLRunner.
ggml_tensor* ggml_ext_full(ggml_context* ctx,
                           float value,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3);

ggml_tensor* ggml_ext_zeros(ggml_context* ctx,
                            int64_t ne0,
                            int64_t ne1,
                            int64_t ne2,
                            int64_t ne3);

ggml_tensor* ggml_ext_zeros_like(ggml_context* ctx,
                                 ggml_tensor* x);

ggml_tensor* ggml_ext_ones(ggml_context* ctx,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3);

ggml_tensor* ggml_ext_ones_like(ggml_context* ctx,
                                ggml_tensor* x);

ggml_tensor* ggml_ext_cast_f32(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* a);

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
ggml_tensor* ggml_ext_attention_ext(ggml_context* ctx,
                                    ggml_backend_t backend,
                                    ggml_tensor* q,
                                    ggml_tensor* k,
                                    ggml_tensor* v,
                                    int64_t n_head,
                                    ggml_tensor* mask = nullptr,
                                    bool skip_reshape = false,
                                    bool flash_attn   = false,
                                    float kv_scale    = 1.0f);

ggml_tensor* ggml_ext_layer_norm(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 float eps = EPS);

ggml_tensor* ggml_ext_group_norm(ggml_context* ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 int num_groups = 32);

ggml_tensor* ggml_ext_timestep_embedding(
    ggml_context* ctx,
    ggml_tensor* timesteps,
    int dim,
    int max_period    = 10000,
    float time_factor = 1.0f);

ggml_tensor* ggml_ext_vec_concat(ggml_context* ctx,
                                 std::vector<ggml_tensor*>& tensors,
                                 int dim);

#endif  // __SD_CORE_GGML_EXTEND_H__

#ifndef __SD_CORE_GGML_EXTEND_H__
#define __SD_CORE_GGML_EXTEND_H__

#include <cstdint>
#include <vector>

#include "core/ggml_extensions.h"
#include "ggml-backend.h"
#include "ggml.h"

struct GGMLRunnerContext;

constexpr uint32_t GGML_PRECISION_F32_VULKAN = 1u << 0;
constexpr uint32_t GGML_PRECISION_F32_ROCM   = 1u << 1;

#define EPS 1e-05f

static_assert(GGML_MAX_NAME >= 160, "GGML_MAX_NAME must be at least 160");

ggml_tensor* ggml_ext_cont(GGMLRunnerContext* runner_ctx, ggml_tensor* x);

// torch like permute
ggml_tensor* ggml_ext_torch_permute(GGMLRunnerContext* runner_ctx,
                                    ggml_tensor* x,
                                    int axis0,
                                    int axis1,
                                    int axis2,
                                    int axis3);

ggml_tensor* ggml_ext_slice(GGMLRunnerContext* runner_ctx,
                            ggml_tensor* x,
                            int dim,
                            int64_t start,
                            int64_t end,
                            bool cont = true);

// example: [N, 3*C, H, W] => ([N, C, H, W], [N, C, H, W], [N, C, H, W])
std::vector<ggml_tensor*> ggml_ext_chunk(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* x,
                                         int num,
                                         int64_t dim,
                                         bool cont = true);

ggml_tensor* ggml_ext_silu_act(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool gate_first = true);

// Without gate, hidden contains [hidden, gate] packed along ne[0].
ggml_tensor* ggml_ext_situ(GGMLRunnerContext* runner_ctx, ggml_tensor* hidden, ggml_tensor* gate = nullptr);

ggml_tensor* ggml_ext_scale(GGMLRunnerContext* runner_ctx, ggml_tensor* x, float factor, bool inplace = false);

// scale and optional shift must already be broadcastable to x in ggml layout.
// Computes (x + x * scale) + shift; F32 uses sd.modulate.v1 when supported by ctx->backend.
ggml_tensor* ggml_ext_modulate(GGMLRunnerContext* runner_ctx,
                               ggml_tensor* x,
                               ggml_tensor* shift,
                               ggml_tensor* scale);

ggml_tensor* ggml_ext_gelu(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool inplace = false);

ggml_tensor* ggml_ext_gelu_quick(GGMLRunnerContext* runner_ctx, ggml_tensor* x, bool inplace = false);

ggml_tensor* ggml_ext_linear(GGMLRunnerContext* runner_ctx,
                             ggml_tensor* x,
                             ggml_tensor* w,
                             ggml_tensor* b,
                             bool force_prec_f32 = false,
                             float scale         = 1.f);

ggml_tensor* ggml_ext_linear_i8_tensorwise(GGMLRunnerContext* runner_ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* w,
                                           ggml_tensor* weight_scale,
                                           ggml_tensor* b,
                                           int convrot_group_size,
                                           float scale = 1.f);

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
                              bool circular_x = false,
                              bool circular_y = false);

ggml_tensor* ggml_ext_pad(GGMLRunnerContext* runner_ctx,
                          ggml_tensor* x,
                          int p0,
                          int p1,
                          int p2          = 0,
                          int p3          = 0,
                          bool circular_x = false,
                          bool circular_y = false);

// ggml layout: x [L, IC, N], w [K, IC/groups, OC], b [OC], result [OL, OC, N].
// force_prec_f32 keeps both input patches and weights in F32.
ggml_tensor* ggml_ext_conv_1d(GGMLRunnerContext* runner_ctx,
                              ggml_tensor* x,
                              ggml_tensor* w,
                              ggml_tensor* b,
                              int s0              = 1,
                              int p0              = 0,
                              int d0              = 1,
                              int64_t groups      = 1,
                              bool force_prec_f32 = false);

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
ggml_tensor* ggml_ext_conv_2d(GGMLRunnerContext* runner_ctx,
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
ggml_tensor* ggml_ext_conv_3d(GGMLRunnerContext* runner_ctx,
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
                              bool force_prec_f32 = false,
                              bool direct         = false,
                              float scale         = 1.f,
                              bool allow_conv2d   = false);

// Constant and cast helpers require the built-in tensors initialized by GGMLRunner.
ggml_tensor* ggml_ext_full(GGMLRunnerContext* runner_ctx,
                           float value,
                           int64_t ne0,
                           int64_t ne1,
                           int64_t ne2,
                           int64_t ne3);

ggml_tensor* ggml_ext_zeros(GGMLRunnerContext* runner_ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

ggml_tensor* ggml_ext_zeros_like(GGMLRunnerContext* runner_ctx, ggml_tensor* x);

ggml_tensor* ggml_ext_ones(GGMLRunnerContext* runner_ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

ggml_tensor* ggml_ext_ones_like(GGMLRunnerContext* runner_ctx, ggml_tensor* x);

ggml_tensor* ggml_ext_cast_f32(GGMLRunnerContext* runner_ctx, ggml_tensor* a);

// q: [N, L_q, C(n_head*d_head)] or [N*n_head, L_q, d_head]
// k: [N, L_k, n_kv_head*d_head] or [N*n_kv_head, L_k, d_head]
// v: [N, L_k, n_kv_head*d_head] or [N, L_k, n_kv_head, d_head]
// mask: [N, L_q, L_k]
// return: [N, L_q, C]
ggml_tensor* ggml_ext_attention_ext(GGMLRunnerContext* runner_ctx,
                                    ggml_tensor* q,
                                    ggml_tensor* k,
                                    ggml_tensor* v,
                                    int64_t n_head,
                                    ggml_tensor* mask     = nullptr,
                                    bool skip_reshape     = false,
                                    bool flash_attn       = false,
                                    float kv_scale        = 1.0f,
                                    bool* used_flash_attn = nullptr);

ggml_tensor* ggml_ext_layer_norm(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 float eps = EPS);

ggml_tensor* ggml_ext_group_norm(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* w,
                                 ggml_tensor* b,
                                 int num_groups = 32,
                                 float eps      = 1e-6f);

ggml_tensor* ggml_ext_timestep_embedding(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* timesteps,
                                         int dim,
                                         int max_period    = 10000,
                                         float time_factor = 1.0f);

ggml_tensor* ggml_ext_vec_concat(GGMLRunnerContext* runner_ctx, std::vector<ggml_tensor*>& tensors, int dim);

#ifndef SD_USE_UPSTREAM_GGML
ggml_tensor* ggml_ext_mul_mat_i8_tensorwise(GGMLRunnerContext* runner_ctx,
                                            ggml_tensor* weight,
                                            ggml_tensor* input,
                                            ggml_tensor* weight_scale,
                                            ggml_tensor* bias,
                                            int group_size,
                                            ggml_tensor* input_scale = nullptr);

ggml_tensor* ggml_ext_quantize_i8_convrot(GGMLRunnerContext* runner_ctx, ggml_tensor* input, int group_size);

// Packed I8 [K/2, N], low nibble first. W4A8 uses a 16-entry F32 codebook
// and F8_E4M3 [K/16, N] relative scales. F32 inputs get an explicit quantize node.
ggml_tensor* ggml_ext_mul_mat_w4_convrot(GGMLRunnerContext* runner_ctx,
                                         ggml_tensor* weight,
                                         ggml_tensor* input,
                                         ggml_tensor* weight_scale,
                                         ggml_tensor* s_channel,
                                         ggml_tensor* s_rel,
                                         ggml_tensor* bias,
                                         SDW4Kind kind,
                                         int group_size);

// Q/K: contiguous F32 [D, tokens, heads, batch], V: contiguous F16 in the same layout.
// Output: F32 [D, query heads, query tokens, batch]. CUDA-only, unmasked inference.
ggml_tensor* ggml_ext_sage_attn(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* q,
                                ggml_tensor* k,
                                ggml_tensor* v,
                                float scale,
                                ggml_sage_attn_mode mode);

// Noncausal Sol-Attn: contiguous F32 Q/K/V [128, tokens, heads, batch], output [128, heads, tokens, batch].
ggml_tensor* ggml_ext_sol_attn(GGMLRunnerContext* runner_ctx,
                               ggml_tensor* q,
                               ggml_tensor* k,
                               ggml_tensor* v,
                               float scale,
                               float tau);

// Contiguous F32 Q [D*heads, W*H], K/V [D*kv_heads, W*H*planes].
// Returns Q's layout, or nullptr when the backend has no compatible kernel.
ggml_tensor* ggml_ext_window_attn(GGMLRunnerContext* runner_ctx,
                                  ggml_tensor* q,
                                  ggml_tensor* k,
                                  ggml_tensor* v,
                                  const SDWindowAttentionParams& params);
#endif

// x: F32 [head_dim, heads, tokens, batch], pe: F32 [2, 2, rot_dim/2, tokens].
// Returns [head_dim, tokens, heads*batch]; channels beyond rot_dim are unchanged.
ggml_tensor* ggml_ext_rope_apply(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* x,
                                 ggml_tensor* pe,
                                 bool interleaved = true);

ggml_tensor* ggml_ext_repeat_4d(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* x,
                                int64_t ne0,
                                int64_t ne1,
                                int64_t ne2,
                                int64_t ne3);

ggml_tensor* ggml_ext_repeat(GGMLRunnerContext* runner_ctx, ggml_tensor* x, ggml_tensor* shape);

ggml_tensor* ggml_ext_embedding(GGMLRunnerContext* runner_ctx,
                                ggml_tensor* weight,
                                ggml_tensor* input_ids,
                                ggml_tensor* weight_scale = nullptr,
                                int group_size            = 0);

ggml_tensor* ggml_ext_prepare_linear_weight(GGMLRunnerContext* runner_ctx,
                                            ggml_tensor* w,
                                            ggml_tensor* x,
                                            bool force_prec_f32);

ggml_tensor* ggml_ext_prepare_i8_input(GGMLRunnerContext* runner_ctx, ggml_tensor* x);

bool ggml_ext_force_prec_f32(GGMLRunnerContext* runner_ctx,
                             bool requested,
                             uint32_t flags);

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
                                        float scale = 1.f);
#endif

ggml_tensor* ggml_ext_conv_2d_grouped(GGMLRunnerContext* runner_ctx,
                                      ggml_tensor* x,
                                      ggml_tensor* w,
                                      ggml_tensor* b,
                                      int groups,
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

#endif  // __SD_CORE_GGML_EXTEND_H__

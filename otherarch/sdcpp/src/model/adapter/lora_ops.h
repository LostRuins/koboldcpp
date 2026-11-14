#ifndef __SD_MODEL_ADAPTER_LORA_OPS_H__
#define __SD_MODEL_ADAPTER_LORA_OPS_H__

#include "core/ggml_runner.h"

// n-mode tensor-matrix product
// example: 2-mode product
// A: [ne03, k, ne01, ne00]
// B: k rows, m columns => [k, m]
// result is [ne03, m, ne01, ne00]
ggml_tensor* ggml_ext_mul_n_mode(GGMLRunnerContext* runner_ctx, ggml_tensor* a, ggml_tensor* b, int mode = 0);

// Kronecker product
// [ne03,ne02,ne01,ne00] x [ne13,ne12,ne11,ne10] => [ne03*ne13,ne02*ne12,ne01*ne11,ne00*ne10]
ggml_tensor* ggml_ext_kronecker(GGMLRunnerContext* runner_ctx, ggml_tensor* a, ggml_tensor* b);

ggml_tensor* ggml_ext_merge_lora(GGMLRunnerContext* runner_ctx,
                                 ggml_tensor* lora_down,
                                 ggml_tensor* lora_up,
                                 ggml_tensor* lora_mid = nullptr);

ggml_tensor* ggml_ext_lokr_forward(GGMLRunnerContext* runner_ctx,
                                   ggml_tensor* h,    // Input: [q, batch] or [W, H, q, batch]
                                   ggml_tensor* w1,   // Outer C (Full rank)
                                   ggml_tensor* w1a,  // Outer A (Low rank part 1)
                                   ggml_tensor* w1b,  // Outer B (Low rank part 2)
                                   ggml_tensor* w2,   // Inner BA (Full rank)
                                   ggml_tensor* w2a,  // Inner A (Low rank part 1)
                                   ggml_tensor* w2b,  // Inner B (Low rank part 2)
                                   bool is_conv,
                                   WeightAdapter::ForwardParams::conv2d_params_t conv_params,
                                   float scale);

#endif  // __SD_MODEL_ADAPTER_LORA_OPS_H__

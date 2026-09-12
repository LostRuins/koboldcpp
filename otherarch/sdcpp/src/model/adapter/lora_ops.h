#ifndef __SD_MODEL_ADAPTER_LORA_OPS_H__
#define __SD_MODEL_ADAPTER_LORA_OPS_H__

#include "core/ggml_runner.h"

ggml_tensor* ggml_ext_merge_lora(ggml_context* ctx,
                                 ggml_tensor* lora_down,
                                 ggml_tensor* lora_up,
                                 ggml_tensor* lora_mid = nullptr);

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
    float scale);

#endif  // __SD_MODEL_ADAPTER_LORA_OPS_H__

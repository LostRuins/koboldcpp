#ifndef __SD_CORE_GGML_EXTENSIONS_H__
#define __SD_CORE_GGML_EXTENSIONS_H__

#include "ggml-backend.h"

void sd_ggml_register_extensions(ggml_backend_t backend);

#ifndef SD_USE_UPSTREAM_GGML
#include "ggml-ext.h"

ggml_ext_op_t sd_ggml_modulate_op();
bool sd_ggml_modulate_describe(const ggml_tensor* node);
size_t sd_ggml_modulate_workspace_size(const ggml_tensor* node);

ggml_ext_op_t sd_ggml_rope_op();
bool sd_ggml_rope_describe(const ggml_tensor* node);
// Apply precomputed rotary matrices, with shapes in ne[] order:
// a: F32 [head_dim, heads, tokens, batch], pe: F32 [2, 2, rot_dim/2, tokens].
// pe stores [[cos, -sin], [sin, cos]] for each pair and token, shared across heads/batches.
// Returns contiguous F32 [head_dim, tokens, heads*batch]. Input strides are preserved.
// Interleaved pairs are (2*i, 2*i+1); otherwise pairs are (i, i+rot_dim/2).
// Channels from rot_dim to head_dim are copied unchanged.
ggml_tensor* sd_ggml_rope_apply(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* x, ggml_tensor* pe, bool interleaved);

struct SDI8Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* input_scale;
    const ggml_tensor* weight_scale;
    const ggml_tensor* bias;
    int group_size;
    bool packed;
};

ggml_ext_op_t sd_ggml_i8_op();
ggml_ext_op_t sd_ggml_convrot_op();
bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* operation);
bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group_size);
ggml_tensor* ggml_mul_mat_i8_tensorwise(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input, ggml_tensor* weight_scale, ggml_tensor* bias, int group_size, ggml_tensor* input_scale = nullptr);
ggml_tensor* ggml_quantize_i8_convrot(ggml_context* ctx, ggml_tensor* input, int group_size);

enum ggml_sage_attn_mode {
    GGML_SAGE_ATTN_AUTO = 0,
    GGML_SAGE_ATTN_2,
    GGML_SAGE_ATTN_2_PLUS_PLUS,
};

struct SDSageAttentionParams {
    float scale;
    ggml_sage_attn_mode mode;
};

ggml_ext_op_t sd_ggml_sage_attn_op();
bool sd_ggml_sage_attn_describe(const ggml_tensor* node, SDSageAttentionParams* params);
// Q/K: contiguous F32 [D, tokens, heads, batch], V: contiguous F16 in the same layout.
// Output: F32 [D, query heads, query tokens, batch]. CUDA-only, unmasked inference.
ggml_tensor* ggml_sage_attn(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, float scale, ggml_sage_attn_mode mode);

struct SDSolAttentionParams {
    float scale;
    float tau;
};

ggml_ext_op_t sd_ggml_sol_attn_op();
bool sd_ggml_sol_attn_describe(const ggml_tensor* node, SDSolAttentionParams* params);
// Noncausal Sol-Attn: contiguous F32 Q/K/V [128, tokens, heads, batch], output [128, heads, tokens, batch].
ggml_tensor* ggml_sol_attn(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, float scale, float tau);
#endif

#endif

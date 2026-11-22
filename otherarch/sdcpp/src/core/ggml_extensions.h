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

struct SDI8Attributes {
    int64_t shape[3];
    int32_t group_size;
    int32_t flags;
};
enum SDI8Flags { SD_I8_PACKED       = 1,
                 SD_I8_INPUT_SCALE  = 2,
                 SD_I8_WEIGHT_SCALE = 4,
                 SD_I8_BIAS         = 8 };

struct SDI8Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* input_scale;
    const ggml_tensor* weight_scale;
    const ggml_tensor* bias;
    int group_size;
    bool packed;
};

bool sd_ggml_valid_convrot_group(int group);

ggml_ext_op_t sd_ggml_i8_op();

ggml_ext_op_t sd_ggml_convrot_op();

bool sd_ggml_i8_describe(const ggml_tensor* node, SDI8Operation* operation);

bool sd_ggml_convrot_describe(const ggml_tensor* node, int* group_size);

enum class SDW4Kind : int32_t { W4A4 = 1,
                                W4A8 = 2 };

struct SDW4Attributes {
    int64_t shape[3];
    SDW4Kind kind;
    int32_t has_bias;
};

struct SDW4Operation {
    const ggml_tensor* weight;
    const ggml_tensor* input;
    const ggml_tensor* weight_scale;
    const ggml_tensor* s_channel;
    const ggml_tensor* s_rel;
    const ggml_tensor* bias;
    SDW4Kind kind;
};

ggml_ext_op_t sd_ggml_w4_op();

bool sd_ggml_w4_describe(const ggml_tensor* node, SDW4Operation* operation);

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

struct SDSolAttentionParams {
    float scale;
    float tau;
};

struct SDWindowAttentionParams {
    int32_t width, height, heads, planes;
    int16_t window_height[4], window_width[4];
    int32_t preserve_boundary;
};

bool sd_ggml_window_attn_infer(ggml_tensor* const* inputs,
                               int count,
                               const void* attributes,
                               ggml_ext_tensor_desc* out);

ggml_ext_op_t sd_ggml_window_attn_op();

bool sd_ggml_window_attn_describe(const ggml_tensor* node, SDWindowAttentionParams* params);

ggml_ext_op_t sd_ggml_sol_attn_op();

bool sd_ggml_sol_attn_describe(const ggml_tensor* node, SDSolAttentionParams* params);

#endif

#endif

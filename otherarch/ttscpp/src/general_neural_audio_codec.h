#pragma once

#include "tts_model.h"

// This namespace implements a general abstraction of the core functionality used in common neural audio codecs like DAC and SNAC.
namespace general_neural_audio_codec {
    enum gguf_tensor {
        LAYER_ALPHA,
        LAYER_INPUT_KERNEL,
        LAYER_INPUT_BIAS,
        LAYER_NOISE_KERNEL,
        RESIDUAL_UNIT_INPUT_ALPHA,
        RESIDUAL_UNIT_OUTPUT_ALPHA,
        RESIDUAL_UNIT_INPUT_KERNEL,
        RESIDUAL_UNIT_OUTPUT_KERNEL,
        RESIDUAL_UNIT_INPUT_BIAS,
        RESIDUAL_UNIT_OUTPUT_BIAS,
        QUANTIZER_LAYER_OUT_KERNEL,
        QUANTIZER_LAYER_OUT_BIAS,
        QUANTIZER_LAYER_CODEBOOK
    };

    struct residual_vector_quantize_layer {
        struct ggml_tensor * out_proj_kernel;
        struct ggml_tensor * out_proj_bias;
        struct ggml_tensor * codebook;
    };

    struct residual_unit {
        residual_unit(uint32_t padding, uint32_t dilation, uint32_t groups = 1): padding(padding), dilation(dilation), groups(groups) {}
        struct ggml_tensor * in_alpha;
        struct ggml_tensor * in_conv_kernel;
        struct ggml_tensor * in_conv_bias;
        struct ggml_tensor * out_alpha;
        struct ggml_tensor * out_conv_kernel;
        struct ggml_tensor * out_conv_bias;

        uint32_t padding;
        uint32_t dilation;
        uint32_t groups;
    };

    struct layer {
        layer(uint32_t padding, uint32_t stride, uint32_t groups = 1): padding(padding), stride(stride) {
            for (int i = 0; i < 3; i++) {
                residual_blocks.push_back(residual_unit{(uint32_t) pow(3, (i + 1)), (uint32_t) pow(3, i), groups});
            }
        }
        struct ggml_tensor * in_alpha;
        struct ggml_tensor * in_conv_kernel;
        struct ggml_tensor * in_conv_bias;
        struct ggml_tensor * noise_conv_kernel = nullptr;

        uint32_t padding;
        uint32_t stride;

        std::vector<residual_unit> residual_blocks;
    };

    void assign_to_residual_unit(tts_model * model, residual_unit & unit, std::string name, struct ggml_tensor * tensor);
    void assign_to_layer(tts_model * model, layer & l, std::string name, struct ggml_tensor * tensor);
    void assign_to_quantize_layer(tts_model * model, residual_vector_quantize_layer & l, std::string name, struct ggml_tensor * tensor);

    struct ggml_tensor * build_residual_unit(ggml_context * ctx, struct ggml_tensor * cur, residual_unit & unit);
    struct ggml_tensor * build_layer(ggml_context * ctx, struct ggml_tensor * cur, layer & l, struct ggml_tensor * noise = nullptr);
    struct ggml_tensor * build_quantize_layer(ggml_context * ctx, struct ggml_tensor * cur, residual_vector_quantize_layer & l);
}

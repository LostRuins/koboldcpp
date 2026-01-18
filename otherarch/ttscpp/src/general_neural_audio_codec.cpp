#include "general_neural_audio_codec.h"
#include <algorithm>
#include <stdexcept>
#include <map>

namespace general_neural_audio_codec {
    // This contains a mapping between string names and gguf_tensor enum values for the purposes of assigning the weights from a gguf file
    // to the general_neural_audio_codec::layer.
    // Please note that some gguf_tensor values have multiple keys; this is to support backwards compatibility with original DAC settings.
    static const std::map<std::string, gguf_tensor> GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP = {
        {".final.alpha", LAYER_ALPHA},
        {".final.bias", LAYER_INPUT_BIAS},
        {".final.weight", LAYER_INPUT_KERNEL},
        {".alpha", LAYER_ALPHA},
        {".bias", LAYER_INPUT_BIAS},
        {".weight", LAYER_INPUT_KERNEL},
        {".noise_weight", LAYER_NOISE_KERNEL},
        {".res.initial.alpha", RESIDUAL_UNIT_INPUT_ALPHA},
        {".res.initial.bias", RESIDUAL_UNIT_INPUT_BIAS},
        {".res.initial.weight", RESIDUAL_UNIT_INPUT_KERNEL},
        {".res.final.alpha", RESIDUAL_UNIT_OUTPUT_ALPHA},
        {".res.final.bias", RESIDUAL_UNIT_OUTPUT_BIAS},
        {".res.final.weight", RESIDUAL_UNIT_OUTPUT_KERNEL},
        {".in_alpha", RESIDUAL_UNIT_INPUT_ALPHA},
        {".in_bias", RESIDUAL_UNIT_INPUT_BIAS},
        {".in_weight", RESIDUAL_UNIT_INPUT_KERNEL},
        {".out_alpha", RESIDUAL_UNIT_OUTPUT_ALPHA},
        {".out_bias", RESIDUAL_UNIT_OUTPUT_BIAS},
        {".out_weight", RESIDUAL_UNIT_OUTPUT_KERNEL},
        {".out_proj.bias", QUANTIZER_LAYER_OUT_BIAS},
        {".out_proj.weight", QUANTIZER_LAYER_OUT_KERNEL},
        {".codebook.weight", QUANTIZER_LAYER_CODEBOOK},
    };

    void assign_to_residual_unit(tts_model * model, residual_unit & unit, std::string name, struct ggml_tensor * tensor) {
        try {
            gguf_tensor tensor_type = GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP.at(name);
            switch (tensor_type) {
                case RESIDUAL_UNIT_INPUT_ALPHA:
                    unit.in_alpha = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(unit.in_alpha, tensor);
                    break;
                case RESIDUAL_UNIT_OUTPUT_ALPHA:
                    unit.out_alpha = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(unit.out_alpha, tensor);
                    break;
                case RESIDUAL_UNIT_INPUT_KERNEL:
                    unit.in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(unit.in_conv_kernel, tensor);
                    break;
                case RESIDUAL_UNIT_OUTPUT_KERNEL:
                    unit.out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(unit.out_conv_kernel, tensor);
                    break;
                case RESIDUAL_UNIT_INPUT_BIAS:
                    unit.in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                    model->set_tensor(unit.in_conv_bias, tensor);
                    break;
                case RESIDUAL_UNIT_OUTPUT_BIAS:
                    unit.out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                    model->set_tensor(unit.out_conv_bias, tensor);
                    break;
                default:
                    fprintf(stdout, "residual unit unassigned tensor %s\n", name.c_str());
                    break;
            }
        } catch (const std::out_of_range& e) {
            TTS_ABORT("Tensor, '%s', is not a valid tensor general_neural_audio_codec::residual_unit tensor.", name.c_str());
        }
    }

    void assign_to_layer(tts_model * model, layer & l, std::string name, struct ggml_tensor * tensor) {
        if (GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP.find(name) != GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP.end()) {
            switch(GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP.at(name)) {
                case LAYER_ALPHA:
                    l.in_alpha = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(l.in_alpha, tensor);
                    break;
                case LAYER_INPUT_KERNEL:
                    l.in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(l.in_conv_kernel, tensor);
                    break;
                case LAYER_INPUT_BIAS:
                    l.in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                    model->set_tensor(l.in_conv_bias, tensor);
                    break;
                case LAYER_NOISE_KERNEL:
                    l.noise_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(l.noise_conv_kernel, tensor);
                    break;
                default:
                    fprintf(stdout, "layer unassigned tensor %s\n", name.c_str());
                    break;
            }
        } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
            auto pair = parse_layer_count(name);
            int i = pair.first;
            std::string lt_name = pair.second;
            assign_to_residual_unit(model, l.residual_blocks[i], lt_name, tensor);
        } else {
            TTS_ABORT("Tensor, '%s', is not a valid tensor general_neural_audio_codec::layer tensor.", name.c_str());
        }
    }

    void assign_to_quantize_layer(tts_model * model, residual_vector_quantize_layer & l, std::string name, struct ggml_tensor * tensor) {
        try {
            switch(GENERAL_NEURAL_AUDIO_CODEC_TENSOR_LOOKUP.at(name)) {
                case QUANTIZER_LAYER_OUT_KERNEL:
                    l.out_proj_kernel = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(l.out_proj_kernel, tensor);
                    break;
                case QUANTIZER_LAYER_OUT_BIAS:
                    l.out_proj_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                    model->set_tensor(l.out_proj_bias, tensor);
                    break;
                case QUANTIZER_LAYER_CODEBOOK:
                    l.codebook = ggml_dup_tensor(model->ctx, tensor);
                    model->set_tensor(l.codebook, tensor);
                    break;
                default:
                    fprintf(stdout, "quantized layer unassigned tensor %s\n", name.c_str());
                    break;
            }
        } catch (const std::out_of_range& e) {
            // older GGUF files still have the unused in_proj convolutional layer, so ignore it if we find it.
            if (!has_prefix(name, ".in_proj")) {
                TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
            }
        }
    }

    struct ggml_tensor * build_residual_unit(ggml_context * ctx, struct ggml_tensor * cur, residual_unit & unit) {
        struct ggml_tensor * residual = cur;
        cur = snake_1d(ctx, unit.in_alpha, cur);
        if (unit.groups > 1) {
            // depthwise 1d convolution is equivalent to convolution in which grouping is equal to filter size.
            // If there is a divergence between filter size and grouping then the kernel's output filters will not be zero.
            TTS_ASSERT(unit.in_conv_kernel->ne[1] == 1);
            cur = ggml_conv_1d_dw_tts(ctx, unit.in_conv_kernel, cur, 1, unit.padding, unit.dilation);
        } else {
            cur = ggml_conv_1d_tts(ctx, unit.in_conv_kernel, cur, 1, unit.padding, unit.dilation);
        }
        cur = ggml_add(ctx, cur, unit.in_conv_bias);
        cur = snake_1d(ctx, unit.out_alpha, cur);
        cur = ggml_conv_1d_tts(ctx, unit.out_conv_kernel, cur, 1, 0, 1);
        cur = ggml_add(ctx, cur, unit.out_conv_bias);
        return ggml_add(ctx, cur, residual);
    }

    struct ggml_tensor * build_layer(ggml_context * ctx, struct ggml_tensor * cur, layer & l, struct ggml_tensor * noise) {
        cur = snake_1d(ctx, l.in_alpha, cur);
        cur = ggml_conv_transpose_1d_tts(ctx, l.in_conv_kernel, cur, l.stride, l.padding, 1, 0, 1);
        cur = ggml_add(ctx, cur, l.in_conv_bias);
        if (l.noise_conv_kernel && noise) {
            struct ggml_tensor * x = ggml_conv_1d_tts(ctx, l.noise_conv_kernel, cur, 1, 0, 1);
            x = ggml_mul(ctx, x, noise);
            cur = ggml_add(ctx, cur, x);
        }
        for (int i = 0; i < l.residual_blocks.size(); i++) {
           cur = build_residual_unit(ctx, cur, l.residual_blocks[i]);
        }
        return cur;
    }

    struct ggml_tensor * build_quantize_layer(ggml_context * ctx, struct ggml_tensor * cur, residual_vector_quantize_layer & l) {
        cur = ggml_get_rows(ctx, l.codebook, cur);
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        cur = ggml_conv_1d_tts(ctx, l.out_proj_kernel, cur, 1, 0, 1);
        cur = ggml_add(ctx, cur, l.out_proj_bias);
        return cur;
    }
}

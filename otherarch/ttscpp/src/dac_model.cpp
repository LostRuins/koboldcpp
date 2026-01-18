#include "dac_model.h"
#include <algorithm>
#include <stdexcept>

// For loading DAC model from gguf file.
static const std::map<std::string, dac_tensor> DAC_TENSOR_GGUF_LOOKUP = {
    {"initial.bias", DAC_ENCODER_IN_BIAS},
    {"initial.weight", DAC_ENCODER_IN_KERNEL},
    {"final.bias", DAC_ENCODER_OUT_BIAS},
    {"final.weight", DAC_ENCODER_OUT_KERNEL},
    {"final.alpha", DAC_ENCODER_SNAKE_ALPHA},
};

void dac_model::prep_constants(gguf_context * meta) {
    int output_heads_key = search_for_gguf_keys(meta, {"parler-tts.decoder.output_heads", "output_heads", "dia.decoder.output_heads"});
    if (output_heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, output_heads_key);
    }

    int sampling_factor_key = search_for_gguf_keys(meta, {"dac.up_sampling_factor", "up_sampling_factor"});
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }

    int max_gen_key = search_for_gguf_keys(meta, {"parler-tts.decoder.max_generation", "max_generation", "dia.decoder.max_generation"});
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
}

void dac_model::prep_layers(gguf_context * meta) {
    for (int i = 0; i < n_heads; i++) {
        quantizer_layers.push_back(general_neural_audio_codec::residual_vector_quantize_layer{});
    }

    for (int i = 0; i < n_layers; i++) {
        std::string stride_key = "dac_layer_stride_" + std::to_string(i);
        std::string padding_key = "dac_layer_padding_" + std::to_string(i);
        int layer_stride_key = search_for_gguf_keys(meta, {"dac." + stride_key, stride_key});
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the DAC audio decoder.", stride_key.c_str());
        }
        int layer_padding_key = search_for_gguf_keys(meta, {"dac." + padding_key, padding_key});
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the DAC audio decoder.", padding_key.c_str());
        }
        layers.push_back(
            general_neural_audio_codec::layer{
                gguf_get_val_u32(meta, layer_padding_key),
                gguf_get_val_u32(meta, layer_stride_key),
            }
        );
    }
}

void dac_model::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_audio_encoder(this, name, tensor);
}

void assign_to_audio_encoder(dac_model * model, std::string name, ggml_tensor * tensor) {
    if (DAC_TENSOR_GGUF_LOOKUP.find(name) != DAC_TENSOR_GGUF_LOOKUP.end()) {
        switch(DAC_TENSOR_GGUF_LOOKUP.at(name)) {
            case DAC_ENCODER_IN_BIAS:
                model->in_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->in_conv_bias, tensor);
                break;
            case DAC_ENCODER_IN_KERNEL:
                model->in_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->in_conv_kernel, tensor);
                break;
            case DAC_ENCODER_OUT_BIAS:
                model->out_conv_bias = ggml_dup_tensor(model->ctx, ggml_transpose(model->ctx, tensor));
                model->set_tensor(model->out_conv_bias, tensor);
                break;
            case DAC_ENCODER_OUT_KERNEL:
                model->out_conv_kernel = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->out_conv_kernel, tensor);
                break;
            case DAC_ENCODER_SNAKE_ALPHA:
                model->snake_alpha = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->snake_alpha, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (std::find_if(name.begin(), name.end(), ::isdigit) != name.end())  {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        if (name.find("quantizers") != std::string::npos) {
            general_neural_audio_codec::assign_to_quantize_layer((tts_model *) model, model->quantizer_layers[l], lt_name, tensor);
        } else {
            general_neural_audio_codec::assign_to_layer((tts_model *) model, model->layers[l - 1], lt_name, tensor);
        }
    }
}

static struct ggml_tensor * dac_build_audio_inputs(struct ggml_context * ctx, struct dac_context * dctx, const dac_ubatch & batch, std::vector<general_neural_audio_codec::residual_vector_quantize_layer> layers) {
    struct ggml_tensor * embd;

    dctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.sequence_length*dctx->model->n_heads);
    ggml_set_input(dctx->inp_tokens);

    if (dctx->backend) {
        ggml_backend_sched_set_tensor_backend(dctx->sched, dctx->inp_tokens, dctx->backend);
    }

    for(int i = 0; i < dctx->model->n_heads; i++) {
        auto quantize_layer = dctx->model->quantizer_layers[i];
        struct ggml_tensor * code = ggml_cont(ctx, ggml_view_2d(ctx, dctx->inp_tokens, 1, batch.sequence_length, dctx->model->n_heads*ggml_type_size(GGML_TYPE_I32), i*ggml_type_size(GGML_TYPE_I32)));
        code = ggml_reshape_1d(ctx, code, batch.sequence_length);
        code = general_neural_audio_codec::build_quantize_layer(ctx, code, quantize_layer);

        if (i == 0) {
            embd = code;
        } else {
            embd = ggml_add(ctx, embd, code);
        }
    }
    return embd;
}

struct dac_context * build_new_dac_context(struct dac_model * model, int n_threads, bool use_cpu) {
    dac_context * dctx = new dac_context(model, n_threads);
    dctx->backend_cpu = ggml_backend_cpu_init();
    dctx->set_threads();
    dctx->build_schedule();
    dctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return dctx;
}

void dac_runner::prepare_post_load() {
    dac_ubatch batch;
    batch.sequence_length = model->max_generation_size;
    ggml_cgraph * gf = build_dac_graph(batch);
    dctx->prep_schedule(gf);
}

struct ggml_cgraph * dac_runner::build_dac_graph(dac_ubatch & batch) {
    init_build();
    // splitting this out from the primary graph so that we can better manage streaming (i.e. sentence chunks are better performed this way)
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inputs;

    inputs = dac_build_audio_inputs(ctx, dctx, batch, model->quantizer_layers);
    ggml_set_name(inputs, "quanitzed_inputs");

    // everything besides the inputs is just a forward pass
    cur = ggml_conv_1d_tts(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    for (auto l : model->layers) {
        cur = general_neural_audio_codec::build_layer(ctx, cur, l);
    }
    cur = snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d_tts(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void dac_runner::run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs) {
    dac_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.sequence_length = sequence_length;
    ggml_backend_sched_reset(dctx->sched);

    const size_t prev_size = dctx->buf_output ? ggml_backend_buffer_get_size(dctx->buf_output) : 0;
    const size_t new_size = model->max_generation_size * model->up_sampling_factor * sizeof(float);

    if (!dctx->buf_output || prev_size < new_size) {
        if (dctx->buf_output) {
            ggml_backend_buffer_free(dctx->buf_output);
            dctx->buf_output = nullptr;
            dctx->logits = nullptr;
        }

        dctx->buf_output = ggml_backend_buft_alloc_buffer(dctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(dctx->buf_output);
    ggml_backend_buffer_clear(dctx->buf_output, 0);

    struct ggml_cgraph * gf = NULL;
    gf = build_dac_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(dctx->sched, gf);

    ggml_backend_tensor_set(dctx->inp_tokens, batch.input_tokens, 0, batch.sequence_length*model->n_heads*ggml_element_size(dctx->inp_tokens));

    ggml_backend_sched_graph_compute_async(dctx->sched, gf);

    dctx->get_ggml_node_data(result, outputs->data, batch.sequence_length*sizeof(float)*model->up_sampling_factor);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(dctx->sched);
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
    return;
}


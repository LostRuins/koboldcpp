#include "snac_model.h"

void snac_model::prep_constants(gguf_context * meta) {
    int heads_key = gguf_find_key(meta, "snac.audio_token_channels");
    if (heads_key != -1) {
        n_heads = gguf_get_val_u32(meta, heads_key);
    }

    int sampling_factor_key = gguf_find_key(meta, "snac.up_sampling_factor");
    if (sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, sampling_factor_key);
    }

    int max_gen_key = gguf_find_key(meta, "snac.max_generation_size");
    if (max_gen_key != -1) {
        max_generation_size = gguf_get_val_u32(meta, max_gen_key);
    }
}

void snac_model::prep_layers(gguf_context * meta) {
    for (int i = 0; i < n_heads; i++) {
        quantizer_layers.push_back(general_neural_audio_codec::residual_vector_quantize_layer{});
    }

    for (int i = 0; i < n_layers; i++) {
        std::string stride_key = "snac.snac_layer_stride_" + std::to_string(i);
        std::string padding_key = "snac.snac_layer_padding_" + std::to_string(i);
        std::string grouping_key = "snac.snac_layer_grouping_" + std::to_string(i);
        int layer_stride_key = gguf_find_key(meta, stride_key.c_str());
        if (layer_stride_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the SNAC audio decoder.", stride_key.c_str());
        }
        int layer_padding_key = gguf_find_key(meta, padding_key.c_str());
        if (layer_padding_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the SNAC audio decoder.", padding_key.c_str());
        }
        int layer_grouping_key = gguf_find_key(meta, grouping_key.c_str());
        if (layer_grouping_key == -1) {
            TTS_ABORT("key %s must be specified in gguf file inorder to initialize the SNAC audio decoder.", grouping_key.c_str());
        }
        layers.push_back(
            general_neural_audio_codec::layer{
                gguf_get_val_u32(meta, layer_padding_key),
                gguf_get_val_u32(meta, layer_stride_key),
                gguf_get_val_u32(meta, layer_grouping_key)
            }
        );
    }
}

void snac_model::assign_weight(std::string name, ggml_tensor * tensor) {
    if (name == "alpha_out") {
        snake_alpha = ggml_dup_tensor(ctx, tensor);
        set_tensor(snake_alpha, tensor);
    } else if (name == "in.weight") {
        in_conv_kernel = ggml_dup_tensor(ctx, tensor);
        set_tensor(in_conv_kernel, tensor);
    } else if (name == "in.bias") {
        in_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(in_conv_bias, tensor);
    } else if (name == "up.weight") {
        up_conv_kernel = ggml_dup_tensor(ctx, tensor);
        set_tensor(up_conv_kernel, tensor);
    } else if (name == "up.bias") {
        up_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(up_conv_bias, tensor);
    } else if (name == "final.weight") {
        out_conv_kernel = ggml_dup_tensor(ctx, tensor);
        set_tensor(out_conv_kernel, tensor);
    } else if (name == "final.bias") {
        out_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
        set_tensor(out_conv_bias, tensor);
    } else if (has_prefix(name, "layers")) {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        general_neural_audio_codec::assign_to_layer((tts_model *) this, layers[l], lt_name, tensor);
    } else if (has_prefix(name, "quantizers")) {
        auto pair = parse_layer_count(name);
        int l = pair.first;
        std::string lt_name = pair.second;
        general_neural_audio_codec::assign_to_quantize_layer((tts_model *) this, quantizer_layers[l], lt_name, tensor);
    }
}

static struct ggml_tensor * snac_build_audio_inputs(struct ggml_context * ctx, struct snac_context * sctx, size_t sequence_length, std::vector<general_neural_audio_codec::residual_vector_quantize_layer> layers) {
    struct ggml_tensor * embd;
    // these devisors represent the discreate repeats performed against each of the three input heads.
    sctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length / 4 + sequence_length / 2 + sequence_length);
    ggml_set_input(sctx->inp_tokens);
    size_t last_stride = 0;
    for(int i = 0; i < sctx->model->n_heads; i++) {
        auto quantize_layer = sctx->model->quantizer_layers[i];
        struct ggml_tensor * inp_head = ggml_cont(ctx, ggml_view_1d(ctx, sctx->inp_tokens, sequence_length / sctx->model->repeats[i], last_stride));
        last_stride += (sequence_length / sctx->model->repeats[i]) * ggml_element_size(sctx->inp_tokens);
        struct ggml_tensor * code = general_neural_audio_codec::build_quantize_layer(ctx, inp_head, quantize_layer);
        if (sctx->model->repeats[i] > 1) {
            // this manipulation is equivalent to repeat_interleave against the first dimension of the tensor
            code = ggml_repeat(ctx, ggml_cont_3d(ctx, code, 1, code->ne[0], code->ne[1]), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sctx->model->repeats[i], code->ne[0], sctx->model->embd));
            code = ggml_cont_2d(ctx, code, sequence_length, code->ne[2]);
        }
        if (i == 0) {
            embd = code;
        } else {
            embd = ggml_add(ctx, embd, code);
        }
    }
    return embd;
}

snac_context * build_new_snac_context(struct snac_model * model, int n_threads, bool use_cpu) {
    snac_context * sctx = new snac_context(model, n_threads);
    sctx->backend_cpu = ggml_backend_cpu_init();
    sctx->set_threads();
    sctx->build_schedule();
    sctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return sctx;
}

void snac_runner::prepare_post_load() {
    ggml_cgraph * gf = build_snac_graph(model->max_generation_size);
    sctx->prep_schedule(gf);
}

struct ggml_cgraph * snac_runner::build_snac_graph(size_t sequence_length) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inputs;

    sctx->noise = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model->noise_steps_sum * sequence_length);
    ggml_set_input(sctx->noise);

    inputs = snac_build_audio_inputs(ctx, sctx, sequence_length, model->quantizer_layers);
    cur = ggml_conv_1d_dw_tts(ctx, model->in_conv_kernel, inputs, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->in_conv_bias);
    cur = ggml_conv_1d_tts(ctx, model->up_conv_kernel, cur, 1, 0, 1);
    cur = ggml_add(ctx, cur, model->up_conv_bias);
    size_t noise_offset = 0;
    for (int l = 0; l < model->layers.size(); l++) {
        auto layer = model->layers[l];
        struct ggml_tensor * noise = ggml_cont(ctx, ggml_view_1d(ctx, sctx->noise, model->noise_steps[l] * sequence_length, noise_offset));
        noise_offset += model->noise_steps[l] * sequence_length * sizeof(float);
        cur = general_neural_audio_codec::build_layer(ctx, cur, layer, noise);
    }
    cur = snake_1d(ctx, model->snake_alpha, cur);
    cur = ggml_conv_1d_tts(ctx, model->out_conv_kernel, cur, 1, 3, 1);
    cur = ggml_add(ctx, cur, model->out_conv_bias);
    cur = ggml_tanh(ctx, cur);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void snac_runner::set_inputs(std::vector<std::vector<uint32_t>> & tokens) {
    ggml_backend_tensor_set(
        sctx->inp_tokens, tokens[0].data(), 0,
        tokens[0].size()*ggml_element_size(sctx->inp_tokens)
    );

    ggml_backend_tensor_set(
        sctx->inp_tokens, tokens[1].data(), tokens[0].size() * ggml_element_size(sctx->inp_tokens),
        tokens[1].size() * ggml_element_size(sctx->inp_tokens)
    );

    ggml_backend_tensor_set(
        sctx->inp_tokens, tokens[2].data(),
        tokens[1].size()*ggml_element_size(sctx->inp_tokens)+tokens[0].size()*ggml_element_size(sctx->inp_tokens),
        tokens[2].size()*ggml_element_size(sctx->inp_tokens)
    );
    size_t sequence_length = tokens[2].size();
    random_normal_gen(model->noise_steps_sum * sequence_length, (float*) sctx->noise->data);
}

void snac_runner::run(std::vector<std::vector<uint32_t>> & tokens, struct tts_response * outputs) {
    size_t sequence_length = tokens[2].size();
    ggml_backend_sched_reset(sctx->sched);

    sctx->prep_output_buffer(model->max_generation_size * model->up_sampling_factor * sizeof(float));

    outputs->data = sctx->logits;
    ggml_backend_buffer_clear(sctx->buf_output, 0);

    struct ggml_cgraph * gf = NULL;
    gf = build_snac_graph(sequence_length);

    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(sctx->sched, gf);

    set_inputs(tokens);

    ggml_backend_sched_graph_compute_async(sctx->sched, gf);

    sctx->get_ggml_node_data(result, outputs->data, sequence_length*sizeof(float)*model->up_sampling_factor);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(sctx->sched);
    outputs->n_outputs = sequence_length * model->up_sampling_factor;
    return;
}


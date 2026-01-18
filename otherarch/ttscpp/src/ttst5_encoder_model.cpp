#include "ttst5_encoder_model.h"

static const std::map<std::string, t5_tensor> T5_TENSOR_GGUF_LOOKUP = {
    {"t5encoder.token_embd", T5_EMBD},
    {"t5encoder.enc.final_layer_norm", T5_NORM},
    {"t5encoder.down_proj", T5_DOWN_PROJ},
    {"t5encoder.down_proj_bias", T5_DOWN_PROJ_BIAS},
    {".attn_norm", T5_LAYER_ATTN_NORM},
    {".attn_q", T5_LAYER_ATTN_Q},
    {".attn_k", T5_LAYER_ATTN_K},
    {".attn_v", T5_LAYER_ATTN_V},
    {".attn_o", T5_LAYER_ATTN_O},
    {".attn_rel_b", T5_RELATIVE_BIAS},
    {".ffn_norm", T5_LAYER_OUT_NORM},
    {".ffn_gate", T5_LAYER_WI_1},
    {".ffn_down", T5_LAYER_WO},
    {".ffn_up", T5_LAYER_WI_0},
};

void assign_to_t5_layer(t5_encoder * model, t5_layer & layer, std::string name, ggml_tensor * tensor) {
    try {
        switch(T5_TENSOR_GGUF_LOOKUP.at(name)) {
            case T5_LAYER_ATTN_NORM:
                layer.attn_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.attn_norm, tensor);
                break;
            case T5_LAYER_ATTN_Q:
                layer.q = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.q, tensor);
                break;
            case T5_LAYER_ATTN_K:
                layer.k = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.k, tensor);
                break;
            case T5_LAYER_ATTN_V:
                layer.v = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.v, tensor);
                break;
            case T5_LAYER_ATTN_O:
                layer.o = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.o, tensor);
                break;
            case T5_LAYER_OUT_NORM:
                layer.mlp_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.mlp_norm, tensor);
                break;
            case T5_LAYER_WI_1:
                layer.wi_1 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wi_1, tensor);
                break;
            case T5_LAYER_WI_0:
                layer.wi_0 = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wi_0, tensor);
                break;
            case T5_LAYER_WO:
                layer.wo = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(layer.wo, tensor);
                break;
            case T5_RELATIVE_BIAS:
                model->relative_attn_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->relative_attn_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } catch (const std::out_of_range& e) {
        TTS_ABORT("Error: %s\nTensor, '%s', is not a valid tensor.", e.what(), name.c_str());
    }
}

void assign_to_t5_encoder(t5_encoder * model, const std::string name, ggml_tensor * tensor) {
    if (tensor->data == NULL) {
        return;
    }
    std::string::size_type pos = name.find(".", 0);
    std::string top_level(name.substr(0, pos));
    if (T5_TENSOR_GGUF_LOOKUP.find(name) != T5_TENSOR_GGUF_LOOKUP.end()) {
        switch (T5_TENSOR_GGUF_LOOKUP.at(name)) {
            case T5_EMBD:
                model->embd = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->embd, tensor);
                break;
            case T5_NORM:
                model->out_norm = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->out_norm, tensor);
                break;
            case T5_DOWN_PROJ:
                model->down_proj = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->down_proj, tensor);
                break;
            case T5_DOWN_PROJ_BIAS:
                model->down_proj_bias = ggml_dup_tensor(model->ctx, tensor);
                model->set_tensor(model->down_proj_bias, tensor);
                break;
            default:
                fprintf(stdout, "unassigned tensor %s\n", name.c_str());
                break;
        }
    } else if (top_level == "t5encoder") {
        auto pair = parse_layer_count(name, 2);
        int l = pair.first;
        std::string lt_name = pair.second;

        assign_to_t5_layer(model, model->layers[l], lt_name, tensor);
    } else {
        return;
    }
}

void t5_encoder::prep_layers(gguf_context * meta) {
	for (uint32_t i = 0; i < n_layers; i++) {
		t5_layer l;
		layers.push_back(l);
	}
}

void t5_encoder::prep_constants(gguf_context * meta) {
	int n_layers_key = gguf_find_key(meta, "t5encoder.block_count");
    if (n_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, n_layers_key);
    }

	int hidden_size_key = gguf_find_key(meta, "t5encoder.embedding_length");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

	int attn_heads_key = gguf_find_key(meta, "t5encoder.attention.head_count");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
    }

    int context_size_key = gguf_find_key(meta, "t5encoder.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);
    }

    int bos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.bos_token_id");
    if (bos_token_id_key != -1) {
        bos_token_id = gguf_get_val_u32(meta, bos_token_id_key);
    }

    int eos_token_id_key = gguf_find_key(meta, "tokenizer.ggml.eos_token_id");
    if (eos_token_id_key != -1) {
        eos_token_id = gguf_get_val_u32(meta, eos_token_id_key);
    }

    int vocab_size_key = gguf_find_key(meta, "t5encoder.vocab_size");
    if (vocab_size_key == -1) {
        TTS_ABORT("key 't5encoder.vocab_size' must be specified in gguf file.");
    }
    vocab_size = gguf_get_val_u32(meta, vocab_size_key);

    int output_size_key = gguf_find_key(meta, "t5encoder.output_size");
    if (output_size_key != -1) {
        output_size = gguf_get_val_u32(meta, output_size_key);
    }
}

void t5_encoder::assign_weight(std::string name, ggml_tensor * tensor) {
    assign_to_t5_encoder(this, name, tensor);
}

struct t5_context * build_new_t5_context(struct t5_encoder * model, int n_threads, bool use_cpu) {
	t5_context * t5ctx = new t5_context(model, n_threads);
    t5ctx->backend_cpu = ggml_backend_cpu_init();
    t5ctx->set_threads();
    t5ctx->build_schedule();
    t5ctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_nodes() + ggml_graph_overhead_custom(model->max_nodes(), false));
    return t5ctx;
}

static struct ggml_tensor * build_t5_norm(struct ggml_context * ctx, struct ggml_tensor * cur, struct ggml_tensor * weight) {
	// this is static for all versions of t5 flan
    float eps = 0.000001;
    cur = ggml_rms_norm(ctx, cur, eps);
    cur = ggml_mul(ctx, cur, weight);
    return cur;
}

static struct ggml_tensor * build_t5_attn_mask(ggml_context * ctx, struct t5_context *t5ctx, const t5_ubatch & batch) {
    t5ctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) batch.n_tokens, (int64_t) batch.n_tokens);
    ggml_set_input(t5ctx->attn_mask);

    return t5ctx->attn_mask;
}

static struct ggml_tensor * build_t5_pos_bias(ggml_context * ctx, struct ggml_tensor * pos_bucket, struct ggml_tensor * relative_attn_bias) {
    struct ggml_tensor * pos_bucket_1d = ggml_view_1d(ctx, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);
    struct ggml_tensor * pos_bias = ggml_get_rows(ctx, relative_attn_bias, pos_bucket_1d);

    pos_bias = ggml_view_3d(ctx, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1], ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * pos_bucket->ne[0],  0);
    pos_bias = ggml_permute(ctx, pos_bias, 2, 1, 0, 3);
    pos_bias = ggml_cont(ctx, pos_bias);
    return pos_bias;
}

t5_ubatch t5_runner::build_worst_case_batch()  {
    struct t5_ubatch batch;
    batch.n_tokens = model->max_context_length;
    return batch;
}

void t5_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf = build_t5_graph(batch);
    t5ctx->prep_schedule(gf);
}

struct ggml_cgraph * t5_runner::build_t5_graph(t5_ubatch & batch) {
    init_build();
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    //t5ctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    //ggml_set_input(t5ctx->positions);

    t5ctx->inp_pos_bucket = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, batch.n_tokens, batch.n_tokens);
    ggml_set_input(t5ctx->inp_pos_bucket);

    t5ctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(t5ctx->inp_tokens);

    inpL = ggml_get_rows(ctx, model->embd, t5ctx->inp_tokens);

    struct ggml_tensor * KQ_mask_dec = build_t5_attn_mask(ctx, t5ctx, batch);
    struct ggml_tensor * pos_bias = build_t5_pos_bias(ctx, t5ctx->inp_pos_bucket, model->relative_attn_bias);

    for (int l = 0; l < model->n_layers; l++) {
        struct ggml_tensor * residual = inpL;

        cur = build_t5_norm(ctx, inpL, model->layers[l].attn_norm);

        struct ggml_tensor * attn_out;

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model->layers[l].q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model->layers[l].k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx, model->layers[l].v, cur);

			Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_add(ctx, kq, pos_bias);

            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, 1.0f, 0.0f);

            struct ggml_tensor * v = ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
            attn_out = ggml_mul_mat(ctx, model->layers[l].o, attn_out);
        }

        cur = ggml_add(ctx, attn_out, residual);
        struct ggml_tensor * residualmlp = cur;

        // mlp
        {
        	cur = build_t5_norm(ctx, cur, model->layers[l].mlp_norm);
        	struct ggml_tensor * gate_proj = ggml_mul_mat(ctx, model->layers[l].wi_1, cur);
        	cur = ggml_mul(ctx, ggml_gelu(ctx, ggml_mul_mat(ctx, model->layers[l].wi_0, cur)), gate_proj);
        	cur = ggml_mul_mat(ctx, model->layers[l].wo, cur);
        }

		cur = ggml_add(ctx, cur, residualmlp);
        inpL = cur;
    }

    cur = build_t5_norm(ctx, cur, model->out_norm);

    if (model->down_proj) {
        cur = ggml_mul_mat(ctx, model->down_proj, cur);
    }

    if (model->down_proj_bias) {
        cur = ggml_add(ctx, cur, model->down_proj_bias);
    }

    ggml_build_forward_expand(gf, cur);

    free_build();

    return gf;
}

void t5_runner::set_inputs(t5_ubatch & batch) {
    ggml_backend_tensor_set(t5ctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(t5ctx->inp_tokens));
    float * attn_mask = nullptr;
    uint32_t * positions = nullptr;
    uint32_t * pos_bucket = nullptr;
    attn_mask = (float *) t5ctx->attn_mask->data;
    positions = (uint32_t *) t5ctx->positions->data;
    pos_bucket = (uint32_t *) t5ctx->inp_pos_bucket->data;
    int n_buckets = (int) model->relative_attn_buckets / 2;
    int max_exact = (int) n_buckets / 2;
	float logarithmic_denominator = log(128.0 / max_exact);
    for (int i = 0; i < batch.n_tokens; i++) {
        for (int ii = 0; ii < batch.n_tokens; ii++) {
        	int ab_rpos = abs(i - ii);
        	int rpos = i - ii;
            attn_mask[i*batch.n_tokens + ii] = 0.0f; //ii > i ? -INFINITY : 0.0f;
            pos_bucket[i*batch.n_tokens + ii] = (uint32_t) (rpos > 0 ? n_buckets : 0) + (ab_rpos < max_exact ? ab_rpos : std::min((n_buckets - 1), (max_exact + (int)((log((ab_rpos / max_exact)) / logarithmic_denominator) * max_exact))));
        }
    }

}

void t5_runner::run(uint32_t * input_tokens, uint32_t sequence_length, struct tts_response * outputs) {
	t5_ubatch batch;
    batch.input_tokens = input_tokens;
    batch.n_tokens = sequence_length;
    ggml_backend_sched_reset(t5ctx->sched);

    const size_t prev_size = t5ctx->buf_output ? ggml_backend_buffer_get_size(t5ctx->buf_output) : 0;
    const size_t new_size = model->max_context_length * model->output_size * sizeof(float);

    if (!t5ctx->buf_output || prev_size < new_size) {
        if (t5ctx->buf_output) {
            ggml_backend_buffer_free(t5ctx->buf_output);
            t5ctx->buf_output = nullptr;
            t5ctx->logits = nullptr;
        }

        t5ctx->buf_output = ggml_backend_buft_alloc_buffer(t5ctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(t5ctx->buf_output);
    ggml_backend_buffer_clear(t5ctx->buf_output, 0);
    struct ggml_cgraph * gf = NULL;
    gf = build_t5_graph(batch);
    // the output is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    ggml_backend_sched_alloc_graph(t5ctx->sched, gf);
    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(t5ctx->sched, gf);

    t5ctx->get_ggml_node_data(result, outputs->data, batch.n_tokens*sizeof(float)*model->output_size);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(t5ctx->sched);
    outputs->n_outputs = sequence_length;
    outputs->hidden_size = model->output_size;
    return;
}

int t5_runner::generate(std::string prompt, tts_response *response) {
	std::vector<uint32_t> tokens;
	tokenizer->tokenize(prompt, tokens);
    tokens.push_back(model->eos_token_id);
	run(tokens.data(), (uint32_t) tokens.size(), response);
	return 0;
}

struct t5_runner * text_encoder_from_file(std::string file_path, int n_threads, unigram_tokenizer * tokenizer, bool cpu_only) {
    t5_encoder * model = new t5_encoder;
    ggml_context * weight_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(file_path.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, file_path.c_str());
    }
    if (!tokenizer) {
        tokenizer = unigram_tokenizer_from_gguf(meta_ctx);
    }
    if (!tokenizer->init) {
        tokenizer->initialize_tokenizer();
    }
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        model->assign_weight(cur->name, cur);
    }

    struct t5_context * t5ctx = build_new_t5_context(model, n_threads, cpu_only);
    struct t5_runner * runner = new t5_runner(model, t5ctx, tokenizer);
    runner->prepare_post_load();
    gguf_free(meta_ctx);
    ggml_free(weight_ctx);

    return runner;
}
